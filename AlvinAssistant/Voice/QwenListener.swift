import AssistantKit
import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXAudioSTT
import Observation
import os
import UIKit

/// On-device Qwen3-ASR, used to transcribe a finished voice turn again, more accurately than the
/// live recognizer. It runs on the GPU, which iOS allows only while the app is in the foreground.
@MainActor
@Observable
final class QwenListener {
    enum Status: Equatable {
        case off
        case loading(Double)
        case ready
        case failed(String)
    }

    static let shared = QwenListener()
    static let modelID = "mlx-community/Qwen3-ASR-0.6B-8bit"
    /// The longest a turn waits for the second pass before the live caption is sent instead.
    static let maxWait: Duration = .seconds(2)
    static let maxSamples = 30 * 16_000

    private(set) var status: Status = .off
    /// A voice session is open: load the model once it's downloaded, and keep it loaded.
    @ObservationIgnored var isWanted = false
    @ObservationIgnored private var model: Qwen3ASRModel?
    @ObservationIgnored private var loading: Task<Void, Never>?
    /// A cancelled load that may still be reading weights; the next load waits for it.
    @ObservationIgnored private var orphan: Task<Void, Never>?
    /// `loading` is only downloading, which uses no GPU and can carry on in the background.
    @ObservationIgnored private var downloading = false

    /// Runs passes one at a time, off the Swift concurrency pool: generate() blocks for seconds.
    nonisolated private static let passes = DispatchQueue(label: "QwenListener.passes", qos: .userInitiated)
    /// Whether GPU work may start. iOS refuses GPU work from a background app and MLX aborts when
    /// it does, so passes start only in the foreground, and leaving waits for `inFlight`.
    nonisolated private static let gpuAllowed = OSAllocatedUnfairLock(initialState: true)
    nonisolated private static let inFlight = DispatchGroup()
    /// Tokens per second of the last pass, prefill included; bounds how long a pass may run.
    nonisolated private static let tokenRate = OSAllocatedUnfairLock(initialState: 20.0)

    private init() {
        let center = NotificationCenter.default
        _ = center.addObserver(forName: UIApplication.willResignActiveNotification, object: nil, queue: .main) { _ in
            Self.gpuAllowed.withLock { $0 = false }
        }
        _ = center.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { QwenListener.shared.unload() }
            Self.waitForGPU()
        }
        _ = center.addObserver(forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { QwenListener.shared.unload() }
        }
        _ = center.addObserver(forName: UIApplication.didBecomeActiveNotification, object: nil, queue: .main) { _ in
            Self.gpuAllowed.withLock { $0 = true }
            MainActor.assumeIsolated {
                // load() does nothing if the model is loaded or on its way.
                let listener = QwenListener.shared
                if listener.isWanted {
                    if case .failed = listener.status {} else { listener.load() }
                }
            }
        }
    }

    /// Downloads the model the first time (about 1 GB, Wi-Fi only), then, if a voice session is
    /// open, loads and warms it up.
    func load() {
        guard model == nil, loading == nil else { return }
        #if targetEnvironment(simulator)
        status = .failed("Qwen3-ASR needs a real iPhone. The Simulator uses Apple's recognizer.")
        #else
        guard UIApplication.shared.applicationState == .active else { return }
        status = .loading(0)
        let orphan = self.orphan
        loading = Task {
            await orphan?.value
            // Let passes still holding the old model finish first.
            await withCheckedContinuation { continuation in Self.passes.async { continuation.resume() } }
            await prepare()
        }
        #endif
    }

    /// Frees the model. A download in progress carries on unless `stopDownload`; `isWanted`
    /// decides whether it gets loaded.
    func unload(stopDownload: Bool = false) {
        model = nil
        guard stopDownload || !downloading else { return }
        if let loading { orphan = loading }
        loading?.cancel()
        loading = nil
        if case .failed = status {} else { status = .off }
    }

    /// The turn's text from Qwen3-ASR, or nil to keep the live caption: the model isn't ready,
    /// the language isn't covered, the clip holds no voice or is too long, or the pass was too slow.
    func transcribe(_ samples: [Float], locale: String) async -> String? {
        // Unloaded mid-session, e.g. by a memory warning: bring it back for the next turns.
        if model == nil, isWanted, status == .off { load() }
        guard let language = FinalTranscript.qwenLanguage(forLocale: locale),
              let voiced = FinalTranscript.voicedRange(of: samples),
              (8_000...Self.maxSamples).contains(voiced.count),
              let model
        else { return nil }
        let clip = Array(samples[voiced])
        let box = ModelBox(model: model)
        let maxTokens = Self.tokenBudget(seconds: Double(clip.count) / 16_000)
        // Set once the turn stops waiting, so a pass still queued doesn't run for nothing.
        let abandoned = OSAllocatedUnfairLock(initialState: false)
        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                let first = FirstResult(continuation)
                Self.passes.async {
                    let text = Self.run(box, clip, language: language, maxTokens: maxTokens, unless: abandoned)
                    Task { @MainActor in first.resume(text) }
                }
                Task {
                    try? await Task.sleep(for: Self.maxWait)
                    abandoned.withLock { $0 = true }
                    first.resume(nil)
                }
            }
        } onCancel: {
            abandoned.withLock { $0 = true }
        }
    }

    /// One pass, on the pass queue. Nil when its turn gave up waiting, the app is leaving the
    /// foreground, or the text was cut off at `maxTokens` (the live caption is complete).
    nonisolated private static func run(_ box: ModelBox, _ samples: [Float], language: String, maxTokens: Int, unless abandoned: OSAllocatedUnfairLock<Bool>) -> String? {
        guard !abandoned.withLock({ $0 }), beginGPU() else { return nil }
        defer { inFlight.leave() }
        let output = box.model.generate(audio: MLXArray(samples), maxTokens: maxTokens, language: language)
        if output.generationTokens >= 8 { tokenRate.withLock { $0 = output.generationTps } }
        // ponytail: measurement line; delete after checking the timings on the iPhone.
        print("[ASR] Qwen3-ASR \(output.totalTime) s, \(output.generationTokens) tokens, peak \(output.peakMemoryUsage) GB")
        return output.generationTokens < maxTokens ? output.text : nil
    }

    /// Room for fast speech (digits are a token each), but no more than this iPhone can produce in
    /// `maxWait`: a slower pass would be dropped anyway, and must not outlast `waitForGPU`.
    nonisolated private static func tokenBudget(seconds: Double) -> Int {
        // ponytail: the time cap trusts the last pass's speed; a slower pass can still overrun, and turns needing more tokens fall back to the live caption.
        min(16 + Int(10 * seconds), Int(tokenRate.withLock { $0 } * 1.5))
    }

    nonisolated private static func beginGPU() -> Bool {
        gpuAllowed.withLock { allowed in
            if allowed { inFlight.enter() }
            return allowed
        }
    }

    /// Blocks until the running pass finishes, so no GPU work is left when iOS takes the GPU away.
    nonisolated private static func waitForGPU() {
        // ponytail: bounded at 3 s (iOS allows about 5); tokenBudget keeps passes near 2 s, but one that runs longer would be refused the GPU and abort the app.
        _ = inFlight.wait(timeout: .now() + 3)
    }

    private func prepare() async {
        do {
            let directory = try await download()
            // Turned on in Settings with no voice session: stop at the download.
            guard isWanted else {
                status = .off
                loading = nil
                return
            }
            try checkStillWanted()
            guard os_proc_available_memory() > 1_500_000_000 else {
                status = .failed("Not enough free memory for Qwen3-ASR right now.")
                loading = nil
                return
            }
            status = .loading(1)
            Memory.cacheLimit = 64 * 1024 * 1024
            // On the CPU, so it's safe to finish in the background; checkStillWanted drops it then.
            let loaded = try await Task.detached { () throws -> ModelBox in
                do {
                    return ModelBox(model: try await Device.withDefaultDevice(.cpu) {
                        try await Qwen3ASRModel.fromModelDirectory(directory)
                    })
                } catch {
                    // An interrupted download can still look complete; start it over next time.
                    try? FileManager.default.removeItem(at: directory)
                    throw error
                }
            }.value
            try checkStillWanted()
            model = loaded.model
            status = .ready
            loading = nil
            // Compile the GPU kernels now instead of on the first turn.
            Self.passes.async {
                _ = Self.run(loaded, [Float](repeating: 0, count: 16_000), language: "English", maxTokens: 1, unless: OSAllocatedUnfairLock(initialState: false))
            }
        } catch where error is CancellationError || Task.isCancelled {
            // unload() already reset the state (a cancelled download throws URLError.cancelled).
        } catch let error as URLError where error.networkUnavailableReason != nil {
            status = .failed("Connect to Wi-Fi to download Qwen3-ASR (about 1 GB).")
            loading = nil
        } catch {
            status = .failed("Qwen3-ASR couldn't load: \(error.localizedDescription)")
            loading = nil
        }
    }

    private func download() async throws -> URL {
        downloading = true
        defer { downloading = false }
        var folder = URL.applicationSupportDirectory.appending(path: "Models", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try folder.setResourceValues(values)
        let cache = HubCache(cacheDirectory: folder)
        // About 1 GB: never over cellular, a hotspot, or in Low Data Mode.
        let configuration = URLSessionConfiguration.default
        configuration.allowsExpensiveNetworkAccess = false
        configuration.allowsConstrainedNetworkAccess = false
        return try await ModelUtils.resolveOrDownloadModel(
            client: HubClient(session: URLSession(configuration: configuration), cache: cache),
            cache: cache,
            repoID: Repo.ID(rawValue: Self.modelID)!,
            requiredExtension: "safetensors",
            progressHandler: { progress in
                if case .loading = QwenListener.shared.status {
                    QwenListener.shared.status = .loading(min(progress.fractionCompleted, 0.99))
                }
            }
        )
    }

    /// Throws if unload() ran or the app left the foreground while loading.
    private func checkStillWanted() throws {
        try Task.checkCancellation()
        guard UIApplication.shared.applicationState == .active else {
            unload()
            throw CancellationError()
        }
    }
}

/// Lets the model cross into the queue that runs it; passes never overlap.
private struct ModelBox: @unchecked Sendable {
    let model: Qwen3ASRModel
}

/// Resumes a continuation with the first result it's given.
@MainActor
private final class FirstResult {
    private var continuation: CheckedContinuation<String?, Never>?

    init(_ continuation: CheckedContinuation<String?, Never>) {
        self.continuation = continuation
    }

    func resume(_ text: String?) {
        continuation?.resume(returning: text)
        continuation = nil
    }
}
