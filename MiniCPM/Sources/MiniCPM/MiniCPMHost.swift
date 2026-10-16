import AssistantKit
import Foundation
import Observation
import os
#if os(iOS)
import UIKit
#endif

/// The app's one owner of the MiniCPM-V models: the 1B that reads photos and the 35B behind
/// "Most accurate on device". It downloads, verifies, loads, runs and unloads them.
///
/// - `status` describes each model's files: `.ready` once they are on the device and verified.
///   A model loads on first use and stays `.ready` while it does, so the photo controls stay put.
/// - Every model call is a run. It loads the model if needed and runs inside the app's GPU lease
///   (`withGPU`). One model loads at a time; a load short of memory first unloads the other model
///   if it's idle. When the host stops a run (the app leaves the screen, an unload, a Woof load, or
///   still too little memory to load), its stream ends with `CancellationError`, so the caller can
///   run it again; any other error is a real failure. Ending a stream cancels only that run; the
///   model stays loaded.
/// - Lifecycle (plan decision 25): the 1B unloads after 3 minutes idle; a memory warning unloads
///   an idle model at once and a busy one after its runs; resigning active cancels the runs;
///   entering the background stops them for good, waits up to 3 s for the GPU work to retire,
///   unloads, and pauses downloads, which resume once the app is active again. The simulator never
///   loads them.
@MainActor @Observable public final class MiniCPMHost {
    public static let shared = MiniCPMHost()

    public enum Model: Hashable, Sendable { case photoReader, accurate }
    public enum Status: Equatable { case notDownloaded, downloading(Double), verifying, ready, loading, failed(String) }

    public private(set) var status: [Model: Status] = [.photoReader: .notDownloaded, .accurate: .notDownloaded]

    /// Runs `body` while no other engine uses the GPU and `needBytes` (margin included) are free.
    /// The app sets it at launch; the default runs `body` directly.
    public static var withGPU: @MainActor (_ needBytes: Int, _ body: @escaping () async throws -> Void) async throws -> Void = { _, body in
        try await body()
    }

    /// The fixed reader prompt (decision 14): a reading that doesn't depend on the question, so the
    /// note serves every turn. MiniCPM-V reads English and Chinese, so the text stays in its language.
    nonisolated static let readerInstruction = """
        Copy every legible text in this photo exactly, in its language and line order, under "Text:". \
        Then write at most three sentences on what the photo shows. Write "unclear" rather than guess.
        """
    /// Image slices per photo. D16 on the iPhone: both 1 and 4 read every key string of the four
    /// test cards (p50 1.4 s vs 2.6 s), but with the same cards smaller in the frame, 1 slice
    /// dropped to 87.5 % at half size and 62.5 % at a third, misreading prices without saying
    /// "unclear", while 4 stayed at 100 %. So 4, pending the owner's real photos.
    nonisolated static let photoSlices = 4
    /// The longest photo note, in tokens (decision 12).
    nonisolated static let photoMaxTokens = 384
    /// How long the 1B stays loaded after its last read.
    nonisolated static let photoIdleUnload: Duration = .seconds(180)
    /// What a run on a loaded model leaves free for the rest of the app. The owner kept these values
    /// (16 Oct 2026) because D18 passed with them on the iPhone (no jetsam over 20 turns); the
    /// D18-derived 1.75 GB / 4.5 GB / 3.2 GB were never measured.
    nonisolated static let runBytes = 600_000_000

    /// What `model` needs free to load, as `os_proc_available_memory` counts it (mapped weights
    /// aside): its peak footprint plus `runBytes` (decision 21: 4.3 GB for the 35B, 2.2 GB for the
    /// 1B). D18 measured load-and-reply peaks of 2.12–2.69 GB for the 35B (about 2.73 GB in use
    /// once its expert cache filled) and 1.26–1.38 GB for the 1B.
    nonisolated static func loadBytes(_ model: Model) -> Int {
        (model == .accurate ? 3_700_000_000 : 1_600_000_000) + runBytes
    }

    private let photoReaderDirectory: URL
    private let accurateDirectory: URL
    /// The download or verification of each model's files: one at a time per model.
    @ObservationIgnored private var jobs: [Model: (task: Task<Void, Never>, download: Bool)] = [:]
    /// Jobs the background stopped, by whether they download, to restart once the app is active.
    @ObservationIgnored private var resumeOnActive: [Model: Bool] = [:]
    @ObservationIgnored private var sessions: [Model: MiniCPMSession] = [:]
    /// A session's load while it runs; its session is in `sessions` already.
    @ObservationIgnored private var loads: [Model: Task<Void, Error>] = [:]
    @ObservationIgnored private var runs: [UUID: (model: Model, task: Task<Void, Never>)] = [:]
    @ObservationIgnored private var idleUnload: Task<Void, Never>?
    @ObservationIgnored private var unloadAfterRuns = false
    @ObservationIgnored private var inBackground = false
    @ObservationIgnored private var idleTimerBeforeDownloads: Bool?

    /// Tests pass their own folders; the app uses the artifacts' own. Files found there are
    /// verified now (hashed once, then compared by size and date), so `status` is known early.
    /// While "Most accurate on device" is hidden, the 35B's files are only shown to be deleted, so
    /// they aren't checked and its status stays `.notDownloaded`.
    init(photoReaderDirectory: URL = MiniCPMArtifacts.directory, accurateDirectory: URL = MiniCPM35BArtifacts.directory) {
        self.photoReaderDirectory = photoReaderDirectory
        self.accurateDirectory = accurateDirectory
        #if os(iOS)
        let center = NotificationCenter.default
        _ = center.addObserver(forName: UIApplication.willResignActiveNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.cancelRuns() }
        }
        _ = center.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.enteredBackground() }
        }
        _ = center.addObserver(forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.memoryWarning() }
        }
        _ = center.addObserver(forName: UIApplication.didBecomeActiveNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.becameActive() }
        }
        #endif
        let checked: [Model] = Self.accurateEligible() ? [.photoReader, .accurate] : [.photoReader]
        for model in checked where FileManager.default.fileExists(atPath: directory(model).path) {
            start(model, download: false)
        }
    }

    // MARK: Files

    /// Verifies the model's files and downloads what's missing, over Wi-Fi. Files already
    /// downloaded are kept, so this also resumes a stopped download.
    public func download(_ m: Model) {
        resumeOnActive[m] = nil
        guard status[m] != .ready else { return }
        start(m, download: true)
    }

    /// Stops a download or verification; finished files and resume data are kept.
    public func cancelDownload(_ m: Model) {
        resumeOnActive[m] = nil
        jobs[m]?.task.cancel()
    }

    /// Deletes the model's files and returns the bytes freed.
    public func delete(_ m: Model) async -> Int64 {
        resumeOnActive[m] = nil
        // First, so no run starts loading files that are going away.
        status[m] = .notDownloaded
        if let job = jobs[m]?.task {
            job.cancel()
            await job.value
        }
        await unload(m)
        let freed = MiniCPMArtifacts.delete(in: directory(m))
        status[m] = .notDownloaded
        return freed
    }

    public func bytesOnDisk(_ m: Model) -> Int64 {
        MiniCPMArtifacts.bytesOnDisk(in: directory(m))
    }

    /// Whether "Most accurate on device" is offered. The owner hid it on 16 Oct 2026 (decision 31):
    /// in voice mode the 35B's first words took p50 7.5 s / p90 8.5 s against D23's 5 s / 8 s line.
    /// Return `MiniCPM35BArtifacts.eligible(physicalMemory:)` (12 GB iPhones) to offer it again; it
    /// then starts off for everyone, as `SettingsStore` saved it off while hidden.
    public static func accurateEligible(physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) -> Bool {
        false
    }

    /// Verifies `m`'s files (`.verifying`), then with `download` fetches what's missing
    /// (`.downloading`), keeping the screen on meanwhile.
    private func start(_ m: Model, download: Bool) {
        guard jobs[m] == nil else { return }
        status[m] = .verifying
        let directory = directory(m)
        let progress = progressReporter(m)
        let task = Task {
            let outcome: Status
            do {
                do {
                    _ = try await Self.files(m, in: directory, download: false)
                } catch is NotDownloaded where download {
                    status[m] = .downloading(0)
                    _ = try await Self.files(m, in: directory, download: true, progress: progress)
                }
                outcome = .ready
            } catch is NotDownloaded {
                outcome = .notDownloaded
            } catch where error is CancellationError || Task.isCancelled {
                outcome = .notDownloaded
            } catch let error as URLError where error.networkUnavailableReason != nil {
                // Cellular, a hotspot or Low Data Mode: downloads use Wi-Fi only.
                outcome = .failed("Connect to Wi-Fi to download.")
            } catch {
                outcome = .failed(error.localizedDescription)
            }
            jobs[m] = nil
            status[m] = outcome
            keepScreenOn()
        }
        jobs[m] = (task, download)
        keepScreenOn()
    }

    /// Shows a download's progress in whole percent: the transfer reports far more often.
    private func progressReporter(_ m: Model) -> @Sendable (Double) -> Void {
        let shown = OSAllocatedUnfairLock(initialState: -1)
        return { [weak self] fraction in
            let percent = Int(fraction * 100)
            let changed = shown.withLock { last in
                defer { last = percent }
                return last != percent
            }
            guard changed else { return }
            Task { @MainActor in
                guard let self, case .downloading = self.status[m] else { return }
                self.status[m] = .downloading(Double(percent) / 100)
            }
        }
    }

    /// Keeps the screen on while a download runs (decision 23): a foreground download stops when
    /// the iPhone locks.
    // ponytail: saves and restores the flag as the benchmark views do, so voice mode starting or
    // stopping mid-download can leave it wrong until the next switch; a shared count of reasons
    // to keep the screen on is the upgrade.
    private func keepScreenOn() {
        #if os(iOS)
        let downloading = jobs.values.contains { $0.download }
        if downloading, idleTimerBeforeDownloads == nil {
            idleTimerBeforeDownloads = UIApplication.shared.isIdleTimerDisabled
            UIApplication.shared.isIdleTimerDisabled = true
        } else if !downloading, let previous = idleTimerBeforeDownloads {
            UIApplication.shared.isIdleTimerDisabled = previous
            idleTimerBeforeDownloads = nil
        }
        #endif
    }

    private func directory(_ m: Model) -> URL {
        m == .photoReader ? photoReaderDirectory : accurateDirectory
    }

    /// Where the 35B keeps its system prefixes; deleting the model deletes them.
    private var prefixDirectory: URL {
        accurateDirectory.appending(path: "prefix", directoryHint: .isDirectory)
    }

    /// What loading a model reads.
    private struct Files: Sendable {
        var model: URL
        var projector: URL?
        var experts: URL?
        var draft: URL?
    }

    /// A file isn't on the device, and this check doesn't download.
    private struct NotDownloaded: Error {}

    /// `m`'s verified files in `directory`. Without `download`, a missing file throws
    /// `NotDownloaded` instead of downloading.
    private nonisolated static func files(_ m: Model, in directory: URL, download: Bool,
                                          progress: @escaping @Sendable (Double) -> Void = { _ in }) async throws -> Files {
        let offline: MiniCPMArtifacts.Downloader = { _, _, _ in throw NotDownloaded() }
        let anySpace: MiniCPMArtifacts.FreeBytes = { _ in .max }
        switch m {
        case .photoReader:
            let artifacts = download ? MiniCPMArtifacts(directory: directory)
                : MiniCPMArtifacts(directory: directory, artifacts: MiniCPMArtifacts.artifacts, downloader: offline, freeBytes: anySpace)
            let found = try await artifacts.prepare(progress: progress)
            return Files(model: found.model, projector: found.projector)
        case .accurate:
            let artifacts = download ? MiniCPM35BArtifacts(directory: directory)
                : MiniCPM35BArtifacts(directory: directory, artifacts: MiniCPM35BArtifacts.artifacts, downloader: offline, freeBytes: anySpace)
            let found = try await artifacts.prepare(progress: progress)
            return Files(model: found.model, experts: found.experts, draft: found.draft)
        }
    }

    // MARK: Runs

    /// What the 1B reads in a JPEG, as streamed text pieces.
    public func readPhoto(_ jpeg: Data) -> AsyncThrowingStream<String, Error> {
        run(.photoReader) { session, out in
            for try await event in session.describePhoto(jpeg: jpeg, system: "", instruction: Self.readerInstruction,
                                                         maxTokens: Self.photoMaxTokens, maxSlices: Self.photoSlices) {
                if case .text(let piece) = event { out.yield(piece) }
            }
        }
    }

    /// Starts loading the 35B and its cached system prefix ahead of a reply. Does nothing while the
    /// 35B isn't ready or is already busy.
    public func prepareAccurate(system: String) {
        guard status[.accurate] == .ready, !runs.values.contains(where: { $0.model == .accurate }) else { return }
        let prefix = prefixDirectory
        let warming: AsyncThrowingStream<Never, Error> = run(.accurate) { session, _ in
            try await session.prewarm(system: system, prefixDirectory: prefix)
        }
        // Ending the stream would cancel the prewarm, so read it to its end.
        Task { for try await _ in warming {} }
    }

    /// The 35B's reply to `turns`, reusing the conversation it holds (see `MiniCPMSession.reply`).
    public func replyAccurate(system: String, turns: [ChatTurn], maxTokens: Int) -> AsyncThrowingStream<MiniCPMEvent, Error> {
        let prefix = prefixDirectory
        return run(.accurate) { session, out in
            for try await event in session.reply(system: system, turns: turns, maxTokens: maxTokens, prefixDirectory: prefix) {
                out.yield(event)
            }
        }
    }

    /// Whether `m` is in memory, so a run starts without loading it.
    public func isLoaded(_ m: Model) -> Bool {
        sessions[m] != nil && loads[m] == nil
    }

    /// Whether a run is in progress (or waiting for the GPU): a prewarm of another model would
    /// unload it.
    public var isBusy: Bool { !runs.isEmpty }

    /// Returns once the runs in progress have ended, such as the prewarm that loads the 35B when
    /// voice mode starts, so a model loaded next measures the memory they leave.
    public func settle() async {
        for run in Array(runs.values) { await run.task.value }
    }

    public func unloadAll() async {
        idleUnload?.cancel()
        // Before any wait: a run waiting in the lease (for Qwen3-ASR's passes) must see its
        // cancellation before it can evict, and so wait for, the Woof load that called this.
        runs.values.forEach { $0.task.cancel() }
        sessions.values.forEach { $0.cancel() }
        await unload(.photoReader)
        await unload(.accurate)
    }

    /// Runs `body` on `m`'s session, loading it first if needed, inside the GPU lease. The returned
    /// stream carries what `body` yields; ending it cancels the run.
    private func run<T>(_ m: Model, _ body: @escaping (MiniCPMSession, AsyncThrowingStream<T, Error>.Continuation) async throws -> Void)
        -> AsyncThrowingStream<T, Error> {
        let (stream, continuation) = AsyncThrowingStream<T, Error>.makeStream()
        let id = UUID()
        if m == .photoReader { idleUnload?.cancel() }
        let task = Task {
            do {
                let loaded = isLoaded(m)
                if !loaded { try checkLoadable(m) }
                try await Self.withGPU(loaded ? Self.runBytes : Self.loadBytes(m)) {
                    try Task.checkCancellation()
                    let session = try await self.session(m)
                    try Task.checkCancellation()
                    var failure: Error?
                    do { try await body(session, continuation) } catch { failure = error }
                    // The lease ends only once the native call has retired: the session's queue
                    // answers this after it.
                    _ = await session.info
                    if let failure { throw failure }
                }
                try Task.checkCancellation()
                continuation.finish()
            } catch {
                continuation.finish(throwing: Task.isCancelled ? CancellationError() : error)
            }
            ended(id, m)
        }
        runs[id] = (m, task)
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }

    /// Why `m` can't load now, thrown; a run in the background is stopped rather than failed.
    private func checkLoadable(_ m: Model) throws {
        #if targetEnvironment(simulator)
        throw MiniCPMError.runtime("On-device models need a real iPhone.")
        #else
        if inBackground { throw CancellationError() }
        guard status[m] == .ready else { throw MiniCPMError.runtime("\(m.name) isn't downloaded. Download it in Settings.") }
        #endif
    }

    /// `m`'s session, loading it if needed. Called inside the GPU lease.
    private func session(_ m: Model) async throws -> MiniCPMSession {
        if sessions[m] == nil {
            try checkLoadable(m)
            let session = MiniCPMSession()
            let directory = directory(m)
            // One load at a time, so each memory check sees the other model's weights resident.
            // Taken before this load is added, so two loads never wait for each other.
            let earlier = Array(loads.values)
            sessions[m] = session
            loads[m] = Task {
                do {
                    for load in earlier { _ = await load.result }
                    #if os(iOS)
                    // Measured after the lease's eviction (decision 21). The other model gives way
                    // when it's idle: the 35B prewarmed in voice mode, or the 1B in its idle minutes.
                    let other: Model = m == .accurate ? .photoReader : .accurate
                    if os_proc_available_memory() < Self.loadBytes(m), !runs.values.contains(where: { $0.model == other }) {
                        await unload(other)
                    }
                    let available = os_proc_available_memory()
                    guard available >= Self.loadBytes(m) else {
                        // Memory frees up as other work ends, so the caller may try again.
                        Logger(subsystem: "MiniCPM", category: "host").notice(
                            "\(m.name, privacy: .public) needs \(Self.loadBytes(m)) bytes to load; \(available) are left")
                        throw CancellationError()
                    }
                    #endif
                    let files = try await Self.files(m, in: directory, download: false)
                    let production = MiniCPMRuntimeConfiguration.production
                    try await session.load(modelURL: files.model, projectorURL: files.projector, expertsURL: files.experts,
                                           draftURL: production.useDraft ? files.draft : nil,
                                           configuration: m == .accurate ? production.configuration : .baseline)
                    if sessions[m] === session { loads[m] = nil }
                } catch {
                    if sessions[m] === session {
                        sessions[m] = nil
                        loads[m] = nil
                    }
                    guard !(error is NotDownloaded) else {
                        status[m] = .notDownloaded
                        throw MiniCPMError.runtime("\(m.name) isn't downloaded. Download it in Settings.")
                    }
                    throw error
                }
            }
        }
        guard let session = sessions[m] else { throw CancellationError() }
        try await loads[m]?.value
        // Unloaded while it loaded.
        guard sessions[m] === session else { throw CancellationError() }
        return session
    }

    private func ended(_ id: UUID, _ m: Model) {
        runs[id] = nil
        if unloadAfterRuns, runs.isEmpty {
            unloadAfterRuns = false
            Task { await unloadAll() }
        } else if m == .photoReader, !runs.values.contains(where: { $0.model == .photoReader }) {
            idleUnload = Task {
                try? await Task.sleep(for: Self.photoIdleUnload)
                guard !Task.isCancelled, !runs.values.contains(where: { $0.model == .photoReader }) else { return }
                await unload(.photoReader)
            }
        }
    }

    /// Stops `m`'s runs and load, waits for them, and frees its memory. The next run loads it again.
    private func unload(_ m: Model) async {
        let session = sessions.removeValue(forKey: m)
        let load = loads.removeValue(forKey: m)
        // Latched: its load and native calls stop at once.
        session?.cancel()
        let stopping = runs.values.filter { $0.model == m }.map(\.task)
        stopping.forEach { $0.cancel() }
        for task in stopping { await task.value }
        _ = await load?.result
        await session?.unload()
    }

    // MARK: Lifecycle

    /// The app is resigning active: stop the runs (each can run again).
    private func cancelRuns() {
        runs.values.forEach { $0.task.cancel() }
    }

    /// iOS refuses GPU work in the background: stop everything for good, give the native work up
    /// to 3 s to retire, then free the memory. Downloads stop with their resume data.
    private func enteredBackground() {
        inBackground = true
        for (m, job) in jobs {
            job.task.cancel()
            resumeOnActive[m] = job.download
        }
        cancelRuns()
        let stopping = Array(sessions.values)
        sessions.removeAll()
        loads.removeAll()
        stopping.forEach { $0.cancel() }
        let deadline = Date().addingTimeInterval(3)
        for session in stopping { _ = session.waitUntilIdle(timeout: deadline.timeIntervalSinceNow) }
        Task { for session in stopping { await session.unload() } }
    }

    private func becameActive() {
        inBackground = false
        for (m, download) in resumeOnActive {
            let stopped = jobs[m]?.task
            Task {
                await stopped?.value
                guard !inBackground, resumeOnActive.removeValue(forKey: m) != nil else { return }
                start(m, download: download)
            }
        }
    }

    /// Unloads an idle model now and a busy one after its runs (decision 25).
    private func memoryWarning() {
        for m in [Model.photoReader, .accurate] where !runs.values.contains(where: { $0.model == m }) {
            Task { await unload(m) }
        }
        unloadAfterRuns = !runs.isEmpty
    }
}

private extension MiniCPMHost.Model {
    var name: String { self == .photoReader ? "The photo reader" : "MiniCPM-V 35B" }
}
