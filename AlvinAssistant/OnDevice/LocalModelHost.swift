import AssistantKit
import Foundation
import HuggingFace
import LocalEngine
import MLX
import MLXLMCommon
import Observation
import os
import UIKit

/// Runs the on-device reply model: downloads and loads it, and answers with either the Alvin
/// engine (LocalEngine: token-exact session reuse, a system prefix kept on disk, speculative
/// decoding with exact rollback, native tool calls) or MLX's stock `ChatSession` over the same
/// loaded model.
///
/// Which one answers is decided per reply by `localEngineMode`: `.alvin` and `.stock` force one;
/// `.automatic` uses the engine only once its on-device self-test has passed for this model and
/// build (`EngineVerificationStore`). An engine error before the first token answers that reply
/// with the stock session instead.
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

    /// The base system prompt and tool definitions on-device replies are sent with, for
    /// `prewarm(settings:)`, which must render exactly what replies will. The system prompt is
    /// the base one (replies add `PromptBuilder.localSystemPrompt`'s lines), and `tools` are the
    /// definitions of the executor replies pass; `handoff_to_cloud` among them means a handoff is
    /// available. Launch setup replaces the default.
    static var systemProvider: @MainActor (AssistantSettings) -> (system: String, tools: [ToolDefinition]) = { settings in
        (
            PromptBuilder.systemPrompt(userName: settings.userName, customInstructions: settings.customInstructions),
            DeviceTools.localRegistry(settings: settings, handoff: false).definitions
        )
    }

    /// Whether the cloud assistant can be reached, read when an engine reply starts (it decides
    /// what `handoff_to_cloud` does). Launch setup replaces the default.
    static var isOnline: @MainActor () -> Bool = { true }

    private(set) var status: Status = .off
    /// What speculative decoding is doing for the loaded model, for Settings and the benchmark.
    private(set) var speculation = "off"
    /// Seconds the last load took after the download, warm-up included (checks excluded).
    private(set) var lastLoadTime: TimeInterval?
    /// Whether the engine holds the system prompt of the last prewarm or engine reply, so the
    /// next reply needn't prefill it. Stock replies never set it.
    private(set) var isWarm = false
    /// What the host is doing besides downloading and loading, e.g. "Checking the on-device
    /// engine…"; nil otherwise.
    private(set) var statusText: String?
    /// The stored self-test result for the loaded model and this build.
    private(set) var selfTest: EngineSelfTest.Result?
    /// The stored cost curve for the loaded model on this device, if one was measured.
    private(set) var costCurve: CostCurve?
    /// The stored fast-kernel gain for the loaded model on this device (c(8) without ÷ with).
    private(set) var kernelGain: Double?
    /// Whether the loaded checkpoint holds multi-token-prediction weights.
    private(set) var hasMTPWeights = false
    /// The last reply's measurements.
    private(set) var lastStats: LocalGenerationStats?
    /// The engine's live session after its last job (warm or cold, ledger, checkpoints, disk prefix).
    private(set) var sessionDescription: String?
    /// A self-test or speed measurement is running.
    private(set) var isChecking = false

    private struct Key: Equatable {
        var modelID: String
        var draft: Bool
        var fastKernels: Bool
    }

    /// What `applyLive` changes on the running engine without a reload.
    private struct LiveOptions: Equatable {
        var speculation: LocalSpeculationMode
        var prefixCache: Bool
        var extensions: [ObjectIdentifier]
        var curve: CostCurve?
    }

    @ObservationIgnored private var key: Key?
    /// The loaded engine; `engine.loaded.container` serves the stock path.
    @ObservationIgnored private(set) var engine: InferenceEngine?
    /// The draft model, loaded only for a catalog model with one while
    /// `localSpeculativeDecoding` is on.
    @ObservationIgnored private(set) var draftModel: LoadedModel?
    @ObservationIgnored private var loading: Task<Void, Never>?
    @ObservationIgnored private var downloading = false
    /// The settings most recently passed in; a load in progress applies them when it ends.
    @ObservationIgnored private var latest: AssistantSettings?
    @ObservationIgnored private var live: LiveOptions?
    @ObservationIgnored private var prewarmAfterLoad = false
    @ObservationIgnored private var prewarming: Task<Void, Never>?
    /// Set while a check (self-test, cost probe) has the engine to itself.
    @ObservationIgnored private var maintenance: Task<Void, Never>?
    @ObservationIgnored private var pendingCheck: Task<Void, Never>?
    /// Replies and prewarms in progress; a check waits for them.
    @ObservationIgnored private var activeJobs = 0
    @ObservationIgnored private var invalidating: Task<Void, Never>?
    /// Whether the target can roll back rejected drafts (fixed per load).
    @ObservationIgnored private var targetRollback = false
    /// Whether the stock path may draft with the draft model (its cache can be trimmed).
    @ObservationIgnored private var stockDraftAllowed = false
    private let verification = EngineVerificationStore()

    /// The stock path's session, whose cache holds `sessionTurns` after `sessionSystem`. Nil
    /// turns mean the cache can't be trusted (a reply was cut short or failed), so the next reply
    /// rebuilds it.
    @ObservationIgnored private var session: ChatSession?
    @ObservationIgnored private var sessionSystem: String?
    @ObservationIgnored private var sessionTurns: [ChatTurn]?

    /// Whether GPU work may start. iOS refuses GPU work from a background app and MLX aborts when
    /// it does, so replies start only in the foreground, and leaving waits for `inFlight`.
    nonisolated private static let gpuAllowed = OSAllocatedUnfairLock(initialState: true)
    nonisolated private static let inFlight = DispatchGroup()
    /// Counts how often the app stopped being active, to tell whether a check ran uninterrupted.
    nonisolated private static let interruptions = OSAllocatedUnfairLock(initialState: 0)

    /// The engine's GPU guard: the same `gpuAllowed` / `inFlight` pair the stock path uses.
    nonisolated static let engineHooks = EngineHooks(
        beginGPU: { LocalModelHost.beginGPU() },
        endGPU: { LocalModelHost.inFlight.leave() },
        isAllowed: { LocalModelHost.gpuAllowed.withLock { $0 } }
    )

    private static let parameters = GenerateParameters(maxTokens: 1024, temperature: 0.7, topP: 0.8, topK: 20)
    private static let numDraftTokens = 4
    /// Room left for the rest of the app after the weights are loaded.
    private static let memoryMargin = 600_000_000

    private init() {
        let center = NotificationCenter.default
        _ = center.addObserver(forName: UIApplication.willResignActiveNotification, object: nil, queue: .main) { _ in
            Self.gpuAllowed.withLock { $0 = false }
            Self.interruptions.withLock { $0 += 1 }
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

    /// Starts downloading and loading the model the settings ask for, if it isn't already. With
    /// `prewarm`, the system prompt is made resident once the model is ready (see
    /// `prewarm(settings:)`).
    func prepare(_ settings: AssistantSettings, prewarm: Bool = true) {
        guard let option = LocalModelCatalog.option(for: settings.localModelID) else { return }
        latest = settings
        let wanted = Self.key(option, settings)
        if wanted == key, engine != nil || loading != nil {
            if loading != nil {
                prewarmAfterLoad = prewarmAfterLoad || prewarm
            } else if prewarm {
                schedulePrewarm(settings)
            }
            return
        }
        unload(stopDownload: true)
        key = wanted
        prewarmAfterLoad = prewarm
        #if targetEnvironment(simulator)
        status = .failed("On-device models need a real iPhone.")
        #else
        guard UIApplication.shared.applicationState == .active else { return }
        status = .loading(0)
        loading = Task { await load(option, key: wanted) }
        #endif
    }

    /// Returns once the current load, if any, has finished or failed.
    func settle() async {
        await loading?.value
    }

    /// Frees the model. A download in progress carries on unless `stopDownload`.
    func unload(stopDownload: Bool = false) {
        engine = nil
        draftModel = nil
        resetSession()
        prewarming?.cancel()
        prewarming = nil
        live = nil
        guard stopDownload || !downloading else { return }
        loading?.cancel()
        loading = nil
        key = nil
        statusText = nil
        if case .failed = status {} else { status = .off }
        Memory.clearCache()
    }

    /// Forgets the cached session, so the next reply processes the whole conversation again.
    func resetSession() {
        session = nil
        sessionSystem = nil
        sessionTurns = nil
        isWarm = false
        if let engine {
            let previous = invalidating
            invalidating = Task {
                await previous?.value
                await engine.invalidateSession()
            }
        }
    }

    /// Applies changed settings: a different model, draft model setting or fast-kernel setting
    /// unloads the model (the next use loads it again); speculation mode, disk prefix and MTP
    /// change on the running engine. Turning on automatic mode or speculation runs a missing
    /// self-test or speed measurement.
    func apply(_ settings: AssistantSettings) {
        guard let key else {
            // Nothing loaded: show what is stored for the model Settings now selects.
            latest = settings
            readStoredResults(modelID: settings.localModelID)
            return
        }
        guard let option = LocalModelCatalog.option(for: settings.localModelID), Self.key(option, settings) == key else {
            unload(stopDownload: true)
            latest = settings
            readStoredResults(modelID: settings.localModelID)
            return
        }
        latest = settings
        // A load in progress applies `latest` when it ends.
        guard engine != nil, loading == nil else { return }
        Task { await applyLive(settings) }
        updateSpeculation()
        if needsCheck(settings), pendingCheck == nil {
            pendingCheck = Task {
                await runMissingChecks()
                pendingCheck = nil
            }
        }
    }

    /// Makes the system prompt of on-device replies resident before the user asks (plan §4.8):
    /// kept if live, else loaded from disk, else prefilled (and saved). For voice listening
    /// starting and the composer opening. A downloaded model that isn't loaded is loaded first;
    /// a model that isn't downloaded is left alone.
    func prewarm(settings: AssistantSettings) {
        guard let option = LocalModelCatalog.option(for: settings.localModelID) else { return }
        if Self.key(option, settings) == key, engine != nil || loading != nil {
            prepare(settings)
            return
        }
        let needsDraft = settings.localSpeculativeDecoding ? option.draftModelID : nil
        guard isDownloaded(option.id), needsDraft.map({ isDownloaded($0) }) ?? true else { return }
        prepare(settings)
    }

    /// Whether `id`'s files are on the iPhone: its snapshot has `config.json` and its weights.
    /// Looks in the local cache only.
    func isDownloaded(_ id: String) -> Bool {
        Self.downloadedSnapshot(id) != nil
    }

    /// The snapshot directory of `id`'s `main` revision when its files are all on the iPhone.
    nonisolated private static func downloadedSnapshot(_ id: String) -> URL? {
        guard let repo = Repo.ID(rawValue: id) else { return nil }
        let cache = hubCache()
        guard let commit = cache.resolveRevision(repo: repo, kind: .model, ref: "main"),
              let snapshot = try? cache.snapshotPath(repo: repo, kind: .model, commitHash: commit),
              hasModelFiles(in: snapshot)
        else { return nil }
        return snapshot
    }

    private func load(_ option: LocalModelOption, key wanted: Key) async {
        do {
            let downloader = HubDownloader(client: Self.hubClient())
            downloading = true
            // Most of the bytes are the main model's; the draft model follows.
            let mainShare = wanted.draft ? 0.85 : 0.99
            let directory = try await downloader.download(id: option.id, progress: { fraction in
                LocalModelHost.shared.setProgress(fraction * mainShare)
            })
            var draftDirectory: URL?
            if wanted.draft, let draftID = option.draftModelID {
                draftDirectory = try await downloader.download(id: draftID, progress: { fraction in
                    LocalModelHost.shared.setProgress(0.85 + fraction * 0.14)
                })
            }
            downloading = false
            try checkStillWanted()

            let needed = option.approximateBytes + (wanted.draft ? option.draftApproximateBytes : 0) + Self.memoryMargin
            guard os_proc_available_memory() > needed else {
                return fail("Not enough free memory for \(option.displayName) right now. Close other apps, turn off Qwen3-ASR listening, or pick a smaller model.")
            }
            status = .loading(1)
            let started = ContinuousClock.now

            // On the CPU, so it's safe to finish in the background; checkStillWanted drops it then.
            if let draftDirectory, let draftID = option.draftModelID {
                let draft = try await Task.detached {
                    try await Device.withDefaultDevice(.cpu) {
                        try await ModelLoader.load(directory: draftDirectory, id: draftID)
                    }
                }.value
                try checkStillWanted()
                draftModel = draft
            }
            readStoredResults(modelID: option.id, snapshot: directory)
            let settings = latest ?? AssistantSettings()
            let configuration = EngineSetup.configuration(settings: settings, host: self)
            let loaded = try await Task.detached {
                try await Device.withDefaultDevice(.cpu) {
                    try await InferenceEngine.load(directory: directory, modelID: option.id, configuration: configuration)
                }
            }.value
            try checkStillWanted()
            engine = loaded
            live = liveOptions(settings)
            targetRollback = loaded.target.supportsRollback
            // The stock path drafts only when the cache can drop rejected drafts.
            if draftModel != nil {
                stockDraftAllowed = await loaded.loaded.container.perform { context in
                    canTrimPromptCache(context.model.newCache(parameters: nil))
                }
            } else {
                stockDraftAllowed = false
            }

            // Compile the GPU kernels now instead of on the first reply.
            await QwenListener.finishPasses()
            do {
                try await loaded.warmUp()
            } catch EngineError.leftForeground {
                // Leaving the screen: checkStillWanted unloads below. Otherwise the first reply
                // compiles them.
            }
            try checkStillWanted()
            lastLoadTime = Self.seconds(started.duration(to: .now))

            await runMissingChecks(loaded)
            try checkStillWanted()
            statusText = nil
            status = .ready
            loading = nil
            if let latest {
                await applyLive(latest)
            }
            updateSpeculation()
            refreshSessionDescription(loaded)
            if prewarmAfterLoad, let latest {
                schedulePrewarm(latest)
            }
        } catch where error is CancellationError || Task.isCancelled {
            downloading = false
            statusText = nil
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
        engine = nil
        draftModel = nil
        live = nil
        statusText = nil
        status = .failed(message)
        loading = nil
    }

    private func setProgress(_ fraction: Double) {
        if case .loading = status { status = .loading(min(fraction, 0.99)) }
    }

    /// Throws if unload() ran or the app left the foreground while loading.
    private func checkStillWanted() throws {
        try Task.checkCancellation()
        guard UIApplication.shared.applicationState == .active else {
            unload(stopDownload: true)
            throw CancellationError()
        }
    }

    private static func key(_ option: LocalModelOption, _ settings: AssistantSettings) -> Key {
        Key(
            modelID: option.id,
            draft: settings.localSpeculativeDecoding && option.supportsSpeculativeDecoding,
            fastKernels: settings.localFastKernels
        )
    }

    // MARK: - Engine configuration

    private func liveOptions(_ settings: AssistantSettings) -> LiveOptions {
        LiveOptions(
            speculation: settings.localSpeculation,
            prefixCache: settings.localPrefixCache,
            extensions: EngineSetup.extensions(for: settings).map { ObjectIdentifier($0) },
            curve: costCurve
        )
    }

    /// Brings the running engine's speculation mode, disk prefix and extensions in line with
    /// `settings`. The generator factory (and with it the draft statistics) is replaced only when
    /// the mode or the cost curve changed.
    private func applyLive(_ settings: AssistantSettings) async {
        guard let engine else { return }
        let wanted = liveOptions(settings)
        guard let current = live, current != wanted else {
            live = wanted
            return
        }
        live = wanted
        let factory = current.speculation != wanted.speculation || current.curve != wanted.curve
            ? EngineSetup.generatorFactory(settings: settings, host: self) : nil
        let directory = EngineSetup.prefixCacheDirectory(settings: settings)
        let extensions = ExtensionList(EngineSetup.extensions(for: settings))
        await engine.updateConfiguration { configuration in
            if let factory {
                configuration.generatorFactory = factory
            }
            configuration.prefixCacheDirectory = directory
            configuration.extensions = extensions.items
        }
    }

    /// Whether replies under `settings` use the engine rather than the stock session.
    func usesEngine(_ settings: AssistantSettings) -> Bool {
        switch settings.localEngineMode {
        case .alvin:
            return true
        case .stock:
            return false
        case .automatic:
            return selfTest?.passed == true
        }
    }

    private func updateSpeculation() {
        guard engine != nil, let settings = latest else { return }
        if usesEngine(settings) {
            guard settings.localSpeculation != .off else {
                speculation = "off"
                return
            }
            guard targetRollback else {
                speculation = "off: this model's cache can't be rolled back"
                return
            }
            var sources = ["prompt lookup", "past replies", "tool-call skeletons"]
            if let draftModel {
                sources.append("\(Self.shortName(draftModel.id)) draft model")
            }
            let mode = settings.localSpeculation == .toolsOnly ? "tools only" : "automatic"
            speculation = "\(mode), from " + sources.joined(separator: ", ")
        } else if let draftModel {
            speculation = stockDraftAllowed
                ? "on, \(Self.shortName(draftModel.id)) drafting \(Self.numDraftTokens) tokens"
                : "off: this model's cache can't be rolled back"
        } else {
            let option = LocalModelCatalog.option(for: settings.localModelID)
            speculation = option?.supportsSpeculativeDecoding == true ? "off" : "not available for this model"
        }
    }

    // MARK: - Checks

    /// Reads what is stored for `modelID` (its self-test for this build, cost curve and kernel
    /// gain) and whether its checkpoint has MTP weights. `snapshot` is the model's snapshot
    /// directory; without it, the downloaded snapshot is looked up (none: no self-test result).
    private func readStoredResults(modelID: String, snapshot: URL? = nil) {
        let directory = snapshot ?? Self.downloadedSnapshot(modelID)
        selfTest = directory.flatMap { verification.selfTest(for: .init(modelID: modelID, snapshot: $0.lastPathComponent)) }
        costCurve = verification.costCurve(for: .init(modelID: modelID))
        kernelGain = verification.kernelGain(for: .init(modelID: modelID))
        hasMTPWeights = directory.map { EngineSetup.hasMTPWeights(in: $0) } ?? false
    }

    /// Whether a load under `settings` would run a check: the self-test in automatic mode
    /// without a stored result, or the cost probe with speculation on, the engine answering and
    /// no stored curve.
    private func needsCheck(_ settings: AssistantSettings) -> Bool {
        if settings.localEngineMode == .automatic, selfTest == nil { return true }
        return settings.localSpeculation != .off && costCurve == nil && usesEngine(settings)
    }

    /// Steps 5 and 6 of a load (also after settings change): the self-test, then the cost probe,
    /// each only when `needsCheck` says so.
    private func runMissingChecks(_ given: InferenceEngine? = nil) async {
        guard let engine = given ?? engine, let settings = latest, needsCheck(settings) else { return }
        await exclusively {
            if settings.localEngineMode == .automatic, selfTest == nil {
                statusText = "Checking the on-device engine…"
                _ = await performSelfTest(engine)
            }
            if settings.localSpeculation != .off, costCurve == nil, usesEngine(settings) {
                statusText = "Measuring the on-device engine's speed…"
                _ = try? await performCostProbe(engine)
            }
            statusText = nil
        }
        updateSpeculation()
    }

    /// Runs the engine self-test now (Settings, the benchmark) and stores its result. Nil when no
    /// model is loaded, or when the app stopped being active during the run (nothing is stored).
    @discardableResult
    func runSelfTest() async -> EngineSelfTest.Result? {
        guard let engine, status == .ready else { return nil }
        let result: EngineSelfTest.Result? = await exclusively {
            statusText = "Checking the on-device engine…"
            defer { statusText = nil }
            return await performSelfTest(engine)
        }
        updateSpeculation()
        return result
    }

    /// Measures the engine's cost curve on this device now (Settings, the benchmark), stores it
    /// and drafts with it from the next reply.
    @discardableResult
    func measureSpeed() async throws -> CostCurve {
        guard let engine, status == .ready else {
            throw AssistantError.missingConfiguration("Load the on-device model first.")
        }
        return try await exclusively {
            statusText = "Measuring the on-device engine's speed…"
            defer { statusText = nil }
            guard let curve = try await performCostProbe(engine) else { throw Self.leftForeground }
            return curve
        }
    }

    /// Stores how much faster the fast kernels made an 8-row forward on this device (c(8)
    /// without ÷ c(8) with), as measured by whoever compared them. Settings offers the fast-kernel
    /// switch only once it is at least `EngineSetup.kernelGainThreshold`.
    func recordKernelGain(_ gain: Double) {
        guard let engine else { return }
        verification.setKernelGain(gain, for: .init(modelID: engine.loaded.id))
        kernelGain = verification.kernelGain(for: .init(modelID: engine.loaded.id))
    }

    /// Whether the stored measurement says the fast kernels help on this device.
    var fastKernelsHelp: Bool {
        (kernelGain ?? 0) >= EngineSetup.kernelGainThreshold
    }

    private func performSelfTest(_ engine: InferenceEngine) async -> EngineSelfTest.Result? {
        let before = Self.interruptions.withLock { $0 }
        await QwenListener.finishPasses()
        isWarm = false
        let result = await EngineSelfTest.run(engine: engine)
        refreshSessionDescription(engine)
        // A run the app interrupted failed for that reason alone: test again later.
        guard Self.interruptions.withLock({ $0 }) == before, self.engine === engine else { return nil }
        let key = EngineVerificationStore.SelfTestKey(modelID: engine.loaded.id, snapshot: engine.loaded.directory.lastPathComponent)
        verification.setSelfTest(result, for: key)
        selfTest = result
        if !result.passed {
            print("LocalModelHost: the engine self-test failed for \(engine.loaded.id):\n\(result.detail)")
        }
        return result
    }

    private func performCostProbe(_ engine: InferenceEngine) async throws -> CostCurve? {
        let before = Self.interruptions.withLock { $0 }
        await QwenListener.finishPasses()
        let curve = try await CostProbe.measure(engine: engine)
        guard Self.interruptions.withLock({ $0 }) == before, self.engine === engine else { return nil }
        verification.setCostCurve(curve, for: .init(modelID: engine.loaded.id))
        costCurve = curve
        // Draft with the measured curve from the next reply on.
        if let latest {
            await applyLive(latest)
        }
        return curve
    }

    /// Runs `body` with the engine to itself: it waits for any other check, then for replies and
    /// prewarms in progress, and keeps new ones waiting until it returns. A check works in
    /// several engine jobs and needs the session unchanged between them.
    private func exclusively<T>(_ body: () async throws -> T) async rethrows -> T {
        while let other = maintenance {
            await other.value
        }
        let (signal, done) = AsyncStream<Void>.makeStream()
        maintenance = Task {
            for await _ in signal {}
        }
        isChecking = true
        defer {
            done.finish()
            maintenance = nil
            isChecking = false
        }
        while activeJobs > 0 {
            try? await Task.sleep(for: .milliseconds(50))
        }
        return try await body()
    }

    /// Returns once no check is running.
    private func waitForChecks() async {
        while let running = maintenance {
            await running.value
        }
    }

    // MARK: - Prewarm

    private func schedulePrewarm(_ settings: AssistantSettings) {
        guard let engine, status == .ready, prewarming == nil, usesEngine(settings) else { return }
        let provided = Self.systemProvider(settings)
        let handoff = provided.tools.contains { $0.name == HandoffTool.name }
        let system = PromptBuilder.localSystemPrompt(base: provided.system, handoffAvailable: handoff)
        let tools = provided.tools
        prewarming = Task {
            await waitForChecks()
            await invalidating?.value
            if self.engine === engine, !Task.isCancelled {
                activeJobs += 1
                await QwenListener.finishPasses()
                do {
                    try await engine.prewarm(system: system, tools: tools)
                    if self.engine === engine {
                        isWarm = true
                    }
                } catch {
                    // Left the foreground, or the template refused: the next reply prefills.
                }
                activeJobs -= 1
                refreshSessionDescription(engine)
            }
            if self.engine === engine {
                prewarming = nil
            }
        }
    }

    // MARK: - Replies

    /// Streams a reply to the last user turn and returns the visible text and measurements.
    ///
    /// - With the engine (see `usesEngine`): `LocalToolLoop` over the engine, with `tools` (none
    ///   when nil), `handoff_to_cloud` per `handoffAvailable`, side effects waiting on
    ///   `commitGate`, and `PromptBuilder.localSystemPrompt(base: system, …)` as the system prompt.
    ///   Events: the engine's progress marks, text, cues, activity, tool rounds, then
    ///   `.reply(.finished)`. Throws `ReplyHandoff` when the model hands the request to the cloud
    ///   before showing anything.
    /// - With the stock session: no tools; only `.reply` events and progress marks.
    @discardableResult
    func respond(
        system: String,
        turns: [ChatTurn],
        settings: AssistantSettings,
        tools: (any ToolExecutor)? = nil,
        handoffAvailable: Bool = false,
        commitGate: CommitGate? = nil,
        emit: (AssistantEvent) -> Void
    ) async throws -> (text: String, stats: LocalGenerationStats) {
        let requested = ContinuousClock.now
        let showLoading = status != .ready
        if showLoading {
            emit(.cue(.loadingModel))
            emit(.reply(.activity("Loading the on-device model")))
        }
        prepare(settings, prewarm: false)
        await settle()
        if case .failed(let message) = status { throw AssistantError.missingConfiguration(message) }
        try Task.checkCancellation()
        await waitForChecks()
        await invalidating?.value
        try Task.checkCancellation()
        guard let engine, status == .ready else {
            throw AssistantError.missingConfiguration("The on-device model isn't loaded. Try again.")
        }
        if showLoading { emit(.reply(.activity(nil))) }
        activeJobs += 1
        defer { activeJobs -= 1 }
        latest = settings
        await applyLive(settings)
        updateSpeculation()
        // A Qwen3-ASR pass that outlived its turn would otherwise share the GPU with this reply.
        await QwenListener.finishPasses()

        guard usesEngine(settings) else {
            return try await respondWithChatSession(engine, system: system, turns: turns, requested: requested, emit: emit)
        }
        do {
            return try await respondWithEngine(
                engine, system: system, turns: turns, tools: tools, handoffAvailable: handoffAvailable,
                commitGate: commitGate, requested: requested, emit: emit)
        } catch let failure as EngineFailure {
            // Nothing was decoded yet: answer this reply with the stock session over the same
            // model, and let the engine start its session over.
            print("LocalModelHost: the engine failed before its first token (\(failure.error)); answering with the stock session.")
            await engine.invalidateSession()
            isWarm = false
            return try await respondWithChatSession(engine, system: system, turns: turns, requested: requested, emit: emit)
        }
    }

    /// An engine error before the first token, which the stock session retries.
    private struct EngineFailure: Error {
        let error: Error
    }

    private func respondWithEngine(
        _ engine: InferenceEngine,
        system: String,
        turns: [ChatTurn],
        tools: (any ToolExecutor)?,
        handoffAvailable: Bool,
        commitGate: CommitGate?,
        requested: ContinuousClock.Instant,
        emit: (AssistantEvent) -> Void
    ) async throws -> (text: String, stats: LocalGenerationStats) {
        let request = EngineRequest(
            system: PromptBuilder.localSystemPrompt(base: system, handoffAvailable: handoffAvailable),
            tools: tools?.definitions ?? [],
            turns: turns
        )
        let online = Self.isOnline()
        let loop = LocalToolLoop(
            engine: engine,
            executor: tools ?? NoToolExecutor(),
            handoffAvailable: handoffAvailable,
            isOnline: { online },
            toolContext: { ToolContext(commitGate: commitGate) }
        )
        let report = ReportBox()
        var text = ""
        var firstText: TimeInterval?
        // A token was decoded (or text shown, or tools run): too late to retry on the stock path.
        var started = false
        do {
            for try await event in loop.run(request, report: { report.set($0) }) {
                switch event {
                case .progress(.firstToken), .toolRound:
                    started = true
                case .reply(.text(let chunk)):
                    started = true
                    if firstText == nil { firstText = Self.seconds(requested.duration(to: .now)) }
                    text += chunk
                default:
                    break
                }
                emit(event)
            }
        } catch let handoff as ReplyHandoff {
            throw handoff
        } catch is CancellationError {
            // Cancelled by the caller, or stopped because the app is leaving the screen.
            try Task.checkCancellation()
            throw Self.leftForeground
        } catch EngineError.leftForeground {
            throw Self.leftForeground
        } catch {
            if !started, !Task.isCancelled {
                throw EngineFailure(error: error)
            }
            throw error
        }
        try Task.checkCancellation()

        var stats = report.value?.stats ?? LocalGenerationStats(engine: "alvin")
        stats.timeToFirstText = firstText
        lastStats = stats
        isWarm = true
        refreshSessionDescription(engine)
        return (text, stats)
    }

    private func respondWithChatSession(
        _ engine: InferenceEngine,
        system: String,
        turns: [ChatTurn],
        requested: ContinuousClock.Instant,
        emit: (AssistantEvent) -> Void
    ) async throws -> (text: String, stats: LocalGenerationStats) {
        guard let plan = LocalSessionPlan.make(cachedSystem: sessionSystem, cachedTurns: sessionTurns, system: system, turns: turns) else {
            throw AssistantError.invalidResponse("There's no question to answer.")
        }
        // The session runs outside the engine's queue: let any engine job (a prewarm) finish
        // first, so the two never use the GPU at once.
        await prewarming?.value
        await engine.waitUntilIdle()
        try Task.checkCancellation()
        let session: ChatSession
        if plan.action == .append, let cached = self.session {
            session = cached
        } else {
            let history: [ChatTurn]
            if case .rebuild(let turns) = plan.action { history = turns } else { history = [] }
            // The system prompt goes in as the first history message, which the session reads
            // once; as `instructions` it would be prefilled again before every turn (F1).
            session = ChatSession(
                engine.loaded.container,
                instructions: nil,
                history: [.system(system)] + history.map(Self.message),
                speculativeDecoding: stockSpeculation(),
                generateParameters: Self.parameters,
                additionalContext: Self.chatContext
            )
        }
        // Trusted again only once this reply completes.
        self.session = session
        sessionSystem = system
        sessionTurns = nil

        guard Self.beginGPU() else { throw Self.leftForeground }
        Memory.peakMemory = 0
        emit(.progress(.responseStarted))

        var stats = LocalGenerationStats(reusedSession: plan.action == .append, engine: "stock")
        var filter = ThinkingFilter()
        var text = ""
        var firstChunk = true
        var stopReason: GenerateStopReason?
        var failure: Error?
        func show(_ visible: String) {
            guard !visible.isEmpty else { return }
            if stats.timeToFirstText == nil { stats.timeToFirstText = Self.seconds(requested.duration(to: .now)) }
            text += visible
            emit(.reply(.text(visible)))
        }
        do {
            for try await item in session.streamDetails(to: LocalSessionPlan.content(of: plan.newTurn)) {
                // Leaving the foreground: stop before iOS takes the GPU away.
                if Task.isCancelled || !Self.gpuAllowed.withLock({ $0 }) { break }
                switch item {
                case .chunk(let chunk):
                    if firstChunk {
                        firstChunk = false
                        emit(.progress(.firstToken))
                    }
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
            emit(.reply(.finished(.completed)))
        case .length:
            emit(.reply(.finished(.truncated)))
        case .cancelled, nil:
            throw Self.leftForeground
        }
        lastStats = stats
        return (text, stats)
    }

    /// The stock path's speculative decoding: the draft model, when the main model's cache can
    /// drop rejected drafts.
    private func stockSpeculation() -> SpeculativeDecodingConfig? {
        guard stockDraftAllowed, let draftModel else { return nil }
        return SpeculativeDecodingConfig(draftModel: draftModel.container, numDraftTokens: Self.numDraftTokens, memoryPolicy: .recommendedWorkingSet)
    }

    // MARK: - Description

    /// Several lines on the loaded engine, for Settings and the benchmark.
    var engineSummary: String {
        guard let engine else {
            if case .loading = status { return statusText ?? "Loading…" }
            return "Not loaded."
        }
        let info = engine.info
        let settings = latest ?? AssistantSettings()
        var lines: [String] = []
        lines.append("Answers with: " + answererDescription(settings))
        var code = info.forked ? "Alvin fork (exact rollback, last-row logits)" : "stock MLX model code"
        if info.isHybrid { code += ", hybrid (recurrent layers)" }
        lines.append("Model code: \(code)")
        lines.append("Tool calls: \(info.toolCallFormat)")
        lines.append("Speculative decoding: \(speculation)")
        lines.append("Self-test: \(selfTestSummary)")
        lines.append("Cost curve: \(costCurveSummary)")
        lines.append("Multi-token prediction weights: \(hasMTPWeights ? "yes" : "none")")
        if let kernelGain {
            lines.append("Fast kernels: \(String(format: "%.2f", kernelGain))× at 8 rows")
        }
        if info.bytesPerCheckpoint > 0 {
            lines.append("Checkpoint: \(info.bytesPerCheckpoint / (1 << 20)) MiB")
        }
        if let sessionDescription {
            lines.append("Session: \(sessionDescription)")
        }
        return lines.joined(separator: "\n")
    }

    /// "Passed in 4.2 s", "Failed (rollback)", or "Not run for this model and build".
    var selfTestSummary: String {
        guard let selfTest else { return "Not run for this model and build" }
        if selfTest.passed {
            return "Passed in \(String(format: "%.1f", selfTest.seconds)) s"
        }
        let failed = EngineSelfTest.checkNames.filter { selfTest.checks[$0] != true }
        return "Failed (\(failed.joined(separator: ", ")))"
    }

    /// "Measured: c(2) 1.05×, c(4) 1.63×, c(8) 3.07×", or the default curve's note.
    var costCurveSummary: String {
        guard let costCurve else { return "Not measured; using the stock MLX curve" }
        let points = [2, 4, 8].map { "c(\($0)) \(String(format: "%.2f", costCurve.relative($0)))×" }
        return "Measured: " + points.joined(separator: ", ")
    }

    /// The last reply's speculation: tokens per verification round and drafts accepted.
    var lastReplySummary: String {
        guard let lastStats else { return "No reply yet" }
        let engineName = lastStats.engine == "stock" ? "Stock session" : "Alvin engine"
        guard let speculation = lastStats.speculation, speculation.rounds > 0 else {
            if let rate = lastStats.acceptanceRate {
                return "\(engineName): \(Int((rate * 100).rounded()))% of drafts accepted"
            }
            return "\(engineName): no speculation"
        }
        let perRound = speculation.meanTokensPerRound.map { String(format: "%.2f", $0) } ?? "–"
        let accepted = speculation.acceptanceRate.map { "\(Int(($0 * 100).rounded()))%" } ?? "–"
        return "\(engineName): \(perRound) tokens per round, \(accepted) of drafts accepted"
    }

    private func answererDescription(_ settings: AssistantSettings) -> String {
        switch settings.localEngineMode {
        case .alvin:
            return "Alvin engine (chosen in Settings)"
        case .stock:
            return "MLX stock session (chosen in Settings)"
        case .automatic:
            guard let selfTest else { return "MLX stock session until the self-test runs" }
            return selfTest.passed ? "Alvin engine (self-test passed)" : "MLX stock session (self-test failed)"
        }
    }

    private func refreshSessionDescription(_ engine: InferenceEngine) {
        Task { [weak self] in
            let summary = await engine.sessionSummary()
            guard let self, self.engine === engine else { return }
            self.sessionDescription = summary
        }
    }

    /// The number of tokens `text` encodes to with the loaded model's tokenizer.
    func tokenCount(_ text: String) -> Int? {
        engine?.loaded.tokenizer.encode(text: text, addSpecialTokens: false).count
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

    /// "Qwen3-0.6B-4bit" for "mlx-community/Qwen3-0.6B-4bit".
    private static func shortName(_ id: String) -> String {
        String(id.split(separator: "/").last ?? Substring(id))
    }

    /// `Application Support/Models`, created and excluded from backup.
    nonisolated private static func modelsFolder() -> URL {
        var folder = URL.applicationSupportDirectory.appending(path: "Models", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? folder.setResourceValues(values)
        return folder
    }

    nonisolated private static func hubCache() -> HubCache {
        HubCache(cacheDirectory: modelsFolder())
    }

    nonisolated private static func hubClient() -> HubClient {
        // Gigabytes: never over cellular, a hotspot, or in Low Data Mode.
        let configuration = URLSessionConfiguration.default
        configuration.allowsExpensiveNetworkAccess = false
        configuration.allowsConstrainedNetworkAccess = false
        return HubClient(session: URLSession(configuration: configuration), cache: hubCache())
    }

    /// Whether `snapshot` holds `config.json` and every weight file: all shards listed by
    /// `model.safetensors.index.json`, or at least one `.safetensors` file without an index.
    nonisolated static func hasModelFiles(in snapshot: URL) -> Bool {
        let files = FileManager.default
        guard files.fileExists(atPath: snapshot.appending(path: "config.json", directoryHint: .notDirectory).path) else {
            return false
        }
        let index = snapshot.appending(path: "model.safetensors.index.json", directoryHint: .notDirectory)
        if let data = try? Data(contentsOf: index),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let map = json["weight_map"] as? [String: String]
        {
            let shards = Set(map.values)
            return !shards.isEmpty && shards.allSatisfy { shard in
                files.fileExists(atPath: snapshot.appending(path: shard, directoryHint: .notDirectory).path)
            }
        }
        let names = (try? files.contentsOfDirectory(atPath: snapshot.path)) ?? []
        return names.contains { $0.hasSuffix(".safetensors") }
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

/// Receives the tool loop's report, which arrives on the loop's task.
private final class ReportBox: @unchecked Sendable {
    private let lock = NSLock()
    private var report: LocalToolLoop.Report?

    func set(_ report: LocalToolLoop.Report) {
        lock.lock()
        self.report = report
        lock.unlock()
    }

    var value: LocalToolLoop.Report? {
        lock.lock()
        defer { lock.unlock() }
        return report
    }
}

/// Engine extensions handed to the engine queue. They are prepared and used there only.
private struct ExtensionList: @unchecked Sendable {
    let items: [any EngineExtension]

    init(_ items: [any EngineExtension]) {
        self.items = items
    }
}
