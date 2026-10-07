import AssistantKit
import Foundation
import HuggingFace
import MLX
import MLXLLM
import MLXLMCommon
import Observation
import os
import Tokenizers
import UIKit

/// Runs the on-device reply model: downloads and loads it, keeps one chat session whose cache is
/// continued from turn to turn, and adds a draft model for speculative decoding where the
/// architecture allows. These are the parts of Husky (Conway Research's engine) that MLX
/// already provides; Husky's hand-tuned GPU kernels are not reproduced.
@MainActor
@Observable
final class LocalModelHost {
    enum Status: Equatable {
        case off
        case loading(Double)
        case ready
        case failed(String)
    }

    static let shared = LocalModelHost()

    private(set) var status: Status = .off
    /// What speculative decoding is doing for the loaded model, for Settings and the benchmark.
    private(set) var speculation = "off"
    /// Seconds the last load took after the download, warm-up included.
    private(set) var lastLoadTime: TimeInterval?

    private struct Key: Equatable {
        var modelID: String
        var speculative: Bool
    }

    @ObservationIgnored private var key: Key?
    @ObservationIgnored private var container: ModelContainer?
    @ObservationIgnored private var draft: ModelContainer?
    @ObservationIgnored private var loading: Task<Void, Never>?
    @ObservationIgnored private var downloading = false

    /// The session whose cache holds `sessionTurns` after `sessionSystem`. Nil turns mean the
    /// cache can't be trusted (a reply was cut short or failed), so the next reply rebuilds it.
    @ObservationIgnored private var session: ChatSession?
    @ObservationIgnored private var sessionSystem: String?
    @ObservationIgnored private var sessionTurns: [ChatTurn]?

    /// Whether GPU work may start. iOS refuses GPU work from a background app and MLX aborts when
    /// it does, so replies start only in the foreground, and leaving waits for `inFlight`.
    nonisolated private static let gpuAllowed = OSAllocatedUnfairLock(initialState: true)
    nonisolated private static let inFlight = DispatchGroup()

    private static let parameters = GenerateParameters(maxTokens: 1024, temperature: 0.7, topP: 0.8, topK: 20)
    private static let numDraftTokens = 4
    /// Room left for the rest of the app after the weights are loaded.
    private static let memoryMargin = 600_000_000
    private static let downloadPatterns = ["*.safetensors", "*.json", "*.jinja"]

    private init() {
        let center = NotificationCenter.default
        _ = center.addObserver(forName: UIApplication.willResignActiveNotification, object: nil, queue: .main) { _ in
            Self.gpuAllowed.withLock { $0 = false }
        }
        _ = center.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { LocalModelHost.shared.unload() }
            Self.waitForGPU()
        }
        _ = center.addObserver(forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { LocalModelHost.shared.unload() }
        }
        _ = center.addObserver(forName: UIApplication.didBecomeActiveNotification, object: nil, queue: .main) { _ in
            Self.gpuAllowed.withLock { $0 = true }
        }
    }

    // MARK: - Loading

    /// Starts downloading and loading the model the settings ask for, if it isn't already.
    func prepare(_ settings: AssistantSettings) {
        guard let option = LocalModelCatalog.option(for: settings.localModelID) else { return }
        let wanted = Key(modelID: option.id, speculative: settings.localSpeculativeDecoding && option.supportsSpeculativeDecoding)
        if wanted == key, container != nil || loading != nil { return }
        unload(stopDownload: true)
        key = wanted
        #if targetEnvironment(simulator)
        status = .failed("On-device models need a real iPhone.")
        #else
        guard UIApplication.shared.applicationState == .active else { return }
        status = .loading(0)
        loading = Task { await load(option, speculative: wanted.speculative) }
        #endif
    }

    /// Returns once the current load, if any, has finished or failed.
    func settle() async {
        await loading?.value
    }

    /// Frees the model. A download in progress carries on unless `stopDownload`.
    func unload(stopDownload: Bool = false) {
        container = nil
        draft = nil
        resetSession()
        guard stopDownload || !downloading else { return }
        loading?.cancel()
        loading = nil
        key = nil
        if case .failed = status {} else { status = .off }
        Memory.clearCache()
    }

    /// Forgets the cached session, so the next reply processes the whole conversation again.
    func resetSession() {
        session = nil
        sessionSystem = nil
        sessionTurns = nil
    }

    private func load(_ option: LocalModelOption, speculative: Bool) async {
        do {
            let downloader = HubDownloader(client: Self.hubClient())
            downloading = true
            let directory = try await downloader.download(id: option.id, progress: { fraction in
                // Most of the bytes are the main model's; the draft model follows.
                LocalModelHost.shared.setProgress(fraction * (speculative ? 0.85 : 0.99))
            })
            var draftDirectory: URL?
            if speculative, let draftID = option.draftModelID {
                draftDirectory = try await downloader.download(id: draftID, progress: { fraction in
                    LocalModelHost.shared.setProgress(0.85 + fraction * 0.14)
                })
            }
            downloading = false
            try checkStillWanted()

            let needed = option.approximateBytes + (speculative ? option.draftApproximateBytes : 0) + Self.memoryMargin
            guard os_proc_available_memory() > needed else {
                return fail("Not enough free memory for \(option.displayName) right now. Close other apps, turn off Qwen3-ASR listening, or pick a smaller model.")
            }
            status = .loading(1)
            let started = ContinuousClock.now
            // On the CPU, so it's safe to finish in the background; checkStillWanted drops it then.
            let loaded = try await Task.detached {
                try await Device.withDefaultDevice(.cpu) {
                    try await Self.loadContainer(directory: directory, id: option.id)
                }
            }.value
            var loadedDraft: ModelContainer?
            if let draftDirectory, let draftID = option.draftModelID {
                let draft = try await Task.detached {
                    try await Device.withDefaultDevice(.cpu) {
                        try await Self.loadContainer(directory: draftDirectory, id: draftID)
                    }
                }.value
                // Speculative decoding rolls the cache back after a rejected draft, which only
                // plain attention caches can do.
                let trimmable = await loaded.perform { context in
                    canTrimPromptCache(context.model.newCache(parameters: nil))
                }
                loadedDraft = trimmable ? draft : nil
                speculation = trimmable ? "on, \(draftID.split(separator: "/").last ?? "") drafting \(Self.numDraftTokens) tokens" : "off: this model's cache can't be rolled back"
            } else {
                speculation = option.supportsSpeculativeDecoding ? "off" : "not available for this model"
            }
            try checkStillWanted()
            container = loaded
            draft = loadedDraft
            await warmUp(loaded)
            try checkStillWanted()
            lastLoadTime = Self.seconds(started.duration(to: .now))
            status = .ready
            loading = nil
        } catch where error is CancellationError || Task.isCancelled {
            downloading = false
            // unload() already reset the state (a cancelled download throws URLError.cancelled).
        } catch let error as URLError where error.networkUnavailableReason != nil {
            downloading = false
            fail("Connect to Wi-Fi to download the on-device model.")
        } catch {
            downloading = false
            fail("The on-device model couldn't load: \(error.localizedDescription)")
        }
    }

    private func fail(_ message: String) {
        container = nil
        draft = nil
        status = .failed(message)
        loading = nil
    }

    private func setProgress(_ fraction: Double) {
        if case .loading = status { status = .loading(min(fraction, 0.99)) }
    }

    /// Compiles the GPU kernels now instead of on the first reply.
    private func warmUp(_ container: ModelContainer) async {
        guard Self.beginGPU() else { return }
        defer { Self.inFlight.leave() }
        let session = ChatSession(container, generateParameters: GenerateParameters(maxTokens: 1), additionalContext: Self.chatContext)
        do {
            for try await _ in session.streamDetails(to: "Hi") {}
        } catch {}
        await session.synchronize()
    }

    /// Throws if unload() ran or the app left the foreground while loading.
    private func checkStillWanted() throws {
        try Task.checkCancellation()
        guard UIApplication.shared.applicationState == .active else {
            unload(stopDownload: true)
            throw CancellationError()
        }
    }

    nonisolated private static func loadContainer(directory: URL, id: String) async throws -> ModelContainer {
        // The files are already downloaded; this downloader only finds them in the cache.
        let configuration = ModelConfiguration(id: id, extraEOSTokens: ["<|im_end|>"])
        return try await LLMModelFactory.shared.loadContainer(
            from: HubDownloader(client: hubClient()),
            using: TransformersTokenizerLoader(),
            configuration: configuration
        )
    }

    // MARK: - Replies

    /// Streams a reply to the last user turn, continuing the cached session when the
    /// conversation allows it. Returns the visible text and measurements.
    @discardableResult
    func respond(system: String, turns: [ChatTurn], settings: AssistantSettings, emit: (ReplyEvent) -> Void) async throws -> (text: String, stats: LocalGenerationStats) {
        let requested = ContinuousClock.now
        let showLoading = status != .ready
        if showLoading { emit(.activity("Loading the on-device model")) }
        prepare(settings)
        await settle()
        if case .failed(let message) = status { throw AssistantError.missingConfiguration(message) }
        try Task.checkCancellation()
        guard let container, status == .ready else {
            throw AssistantError.missingConfiguration("The on-device model isn't loaded. Try again.")
        }
        if showLoading { emit(.activity(nil)) }

        guard let plan = LocalSessionPlan.make(cachedSystem: sessionSystem, cachedTurns: sessionTurns, system: system, turns: turns) else {
            throw AssistantError.invalidResponse("There's no question to answer.")
        }
        let session: ChatSession
        if plan.action == .append, let cached = self.session {
            session = cached
        } else {
            let history: [ChatTurn]
            if case .rebuild(let turns) = plan.action { history = turns } else { history = [] }
            session = ChatSession(
                container,
                instructions: system,
                history: history.map(Self.message),
                speculativeDecoding: draft.map { SpeculativeDecodingConfig(draftModel: $0, numDraftTokens: Self.numDraftTokens, memoryPolicy: .recommendedWorkingSet) },
                generateParameters: Self.parameters,
                additionalContext: Self.chatContext
            )
        }
        // Trusted again only once this reply completes.
        self.session = session
        sessionSystem = system
        sessionTurns = nil

        // A Qwen3-ASR pass that outlived its turn would otherwise share the GPU with this reply.
        await QwenListener.finishPasses()
        guard Self.beginGPU() else { throw Self.leftForeground }
        Memory.peakMemory = 0

        var stats = LocalGenerationStats(reusedSession: plan.action == .append)
        var filter = ThinkingFilter()
        var text = ""
        var stopReason: GenerateStopReason?
        var failure: Error?
        func show(_ visible: String) {
            guard !visible.isEmpty else { return }
            if stats.timeToFirstText == nil { stats.timeToFirstText = Self.seconds(requested.duration(to: .now)) }
            text += visible
            emit(.text(visible))
        }
        do {
            for try await item in session.streamDetails(to: LocalSessionPlan.content(of: plan.newTurn)) {
                // Leaving the foreground: stop before iOS takes the GPU away.
                if Task.isCancelled || !Self.gpuAllowed.withLock({ $0 }) { break }
                switch item {
                case .chunk(let chunk):
                    show(filter.feed(chunk))
                case .info(let info):
                    stats.promptTokens = info.promptTokenCount
                    stats.promptTime = info.promptTime
                    stats.generatedTokens = info.generationTokenCount
                    stats.generateTime = info.generateTime
                    stats.draftTokens = info.speculativeDecodingTelemetry?.draftTokenCount ?? info.proposedDraftTokens
                    stats.acceptedDraftTokens = info.speculativeDecodingTelemetry?.acceptedDraftTokenCount ?? info.acceptedDraftTokens
                    stopReason = info.stopReason
                case .toolCall:
                    break
                }
            }
        } catch {
            failure = error
        }
        // Wait until generation has really stopped using the GPU and the cache.
        await session.synchronize()
        Self.inFlight.leave()
        show(filter.finish())
        stats.peakMemoryBytes = Memory.peakMemory

        if let failure { throw failure }
        try Task.checkCancellation()
        switch stopReason {
        case .stop:
            if self.session === session {
                sessionTurns = turns + [ChatTurn(role: .assistant, text: text)]
            }
            emit(.finished(.completed))
        case .length:
            emit(.finished(.truncated))
        case .cancelled, nil:
            throw Self.leftForeground
        }
        return (text, stats)
    }

    // MARK: - Helpers

    private static let leftForeground = AssistantError.stream(type: "background", message: "the app left the screen.")

    /// Qwen-family chat templates: answer directly, without a reasoning block.
    private static let chatContext: [String: any Sendable] = ["enable_thinking": false]

    private static func message(_ turn: ChatTurn) -> Chat.Message {
        switch turn.role {
        case .user: return .user(LocalSessionPlan.content(of: turn))
        case .assistant: return .assistant(turn.text)
        }
    }

    nonisolated private static func hubClient() -> HubClient {
        var folder = URL.applicationSupportDirectory.appending(path: "Models", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? folder.setResourceValues(values)
        // Gigabytes: never over cellular, a hotspot, or in Low Data Mode.
        let configuration = URLSessionConfiguration.default
        configuration.allowsExpensiveNetworkAccess = false
        configuration.allowsConstrainedNetworkAccess = false
        return HubClient(session: URLSession(configuration: configuration), cache: HubCache(cacheDirectory: folder))
    }

    nonisolated private static func beginGPU() -> Bool {
        gpuAllowed.withLock { allowed in
            if allowed { inFlight.enter() }
            return allowed
        }
    }

    /// Blocks until the running reply stops, so no GPU work is left when iOS takes the GPU away.
    nonisolated private static func waitForGPU() {
        _ = inFlight.wait(timeout: .now() + 3)
    }

    nonisolated static func seconds(_ duration: Duration) -> TimeInterval {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}

/// Fetches model files through swift-huggingface, preferring files already on the iPhone so a
/// downloaded model loads offline.
private struct HubDownloader: MLXLMCommon.Downloader {
    let client: HubClient

    func download(id: String, revision: String?, matching patterns: [String], useLatest: Bool, progressHandler: @Sendable @escaping (Progress) -> Void) async throws -> URL {
        guard let repo = Repo.ID(rawValue: id) else {
            throw AssistantError.missingConfiguration("“\(id)” isn't a valid model ID.")
        }
        let revision = revision ?? "main"
        if !useLatest, let cached = try? await client.downloadSnapshot(of: repo, revision: revision, matching: patterns, localFilesOnly: true) {
            return cached
        }
        return try await client.downloadSnapshot(of: repo, revision: revision, matching: patterns, progressHandler: { progress in
            progressHandler(progress)
        })
    }

    func download(id: String, progress: @escaping @MainActor @Sendable (Double) -> Void) async throws -> URL {
        try await download(id: id, revision: nil, matching: ["*.safetensors", "*.json", "*.jinja"], useLatest: false) { value in
            let fraction = value.fractionCompleted
            Task { @MainActor in progress(fraction) }
        }
    }
}

/// Loads tokenizers with swift-transformers, as MLXHuggingFace's macro would, without the macro.
private struct TransformersTokenizerLoader: MLXLMCommon.TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        TokenizerBridge(upstream: try await AutoTokenizer.from(modelFolder: directory))
    }
}

private struct TokenizerBridge: MLXLMCommon.Tokenizer {
    let upstream: any Tokenizers.Tokenizer

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(messages: messages, tools: tools, additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}
