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
    /// The stored self-test result for the loaded (else the selected, downloaded) model and this
    /// build.
    private(set) var selfTest: EngineSelfTest.Result?
    /// The stored cost curve for that model on this device (measured with the fast kernels
    /// while the model runs them), if one was measured.
    private(set) var costCurve: CostCurve?
    /// The stored fast-kernel gain for that model on this device: how much cheaper an 8-row
    /// forward was with them (`1 − c(8) with ÷ c(8) without`).
    private(set) var kernelGain: Double?
    /// What the fast kernels last reported for the loaded model (their `prepare` in the load's
    /// warm-up, or `measureFastKernels()`), with the before/after c(8).
    private(set) var kernelSummary: String?
    /// Whether the loaded model runs the fast kernels.
    private(set) var kernelsInstalled = false
    /// Whether that model's checkpoint holds multi-token-prediction weights.
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
        var fastKernelsRequested: Bool
        var curve: CostCurve?
        /// Whether the model runs the fast kernels (more drafts per round may pay).
        var kernelsInstalled: Bool
    }

    @ObservationIgnored private var key: Key?
    /// The loaded engine; `engine.loaded.container` serves the stock path.
    @ObservationIgnored private(set) var engine: InferenceEngine?
    /// The draft model, loaded only for a catalog model with one while
    /// `localSpeculativeDecoding` is on.
    @ObservationIgnored private(set) var draftModel: LoadedModel?
    @ObservationIgnored private var loading: Task<Void, Never>?
    /// A cancelled load that may still be reading weights or checking its engine; the next load
    /// waits for it before downloading.
    @ObservationIgnored private var orphan: Task<Void, Never>?
    /// Ends once nothing uses the engine `unload()` let go of: its pending check (cancelled),
    /// the checks, replies and prewarms in progress, the checks and the prewarm still waiting for
    /// their turn (they return without running), and the jobs on its queue. The next load waits
    /// for it, and for `orphan`, before its memory check, so two models are never resident at
    /// once.
    @ObservationIgnored private var retired: Task<Void, Never>?
    @ObservationIgnored private var downloading = false
    /// The settings most recently passed in; a load in progress applies them when it ends.
    @ObservationIgnored private var latest: AssistantSettings?
    @ObservationIgnored private var live: LiveOptions?
    @ObservationIgnored private var prewarmAfterLoad = false
    /// A scheduled prewarm, from waiting for its turn until it ends.
    @ObservationIgnored private var prewarming: Task<Void, Never>?
    /// The engine work of a prewarm that has started (it counts in `activeJobs`); the stock path
    /// waits for it.
    @ObservationIgnored private var prewarmJob: Task<Void, Never>?
    /// Set while a check (self-test, cost probe, kernel comparison) has the engine to itself.
    @ObservationIgnored private var maintenance: Task<Void, Never>?
    /// Checks in `exclusively`, waiting for their turn or running. Each holds its engine.
    @ObservationIgnored private var checks = 0
    @ObservationIgnored private var pendingCheck: Task<Void, Never>?
    /// Replies and prewarms in progress; a check waits for them. Only `beginJob` and
    /// `beginPrewarmJob` add to it.
    @ObservationIgnored private var activeJobs = 0
    /// The pending session reset (`resetSession()`); nil once a job or check has seen it end.
    @ObservationIgnored private var invalidating: Task<Void, Never>?
    /// Whether the fast kernels' verdict for the loaded engine has been read (their `prepare`
    /// runs in the warm-up, or at the first engine job when the warm-up was cut short).
    @ObservationIgnored private var kernelReportTaken = false
    /// Whether the target can roll back rejected drafts (fixed per load).
    @ObservationIgnored private var targetRollback = false
    /// Whether the stock path may draft with the draft model (its cache can be trimmed).
    @ObservationIgnored private var stockDraftAllowed = false
    /// The model `selfTest`, `costCurve`, `kernelGain` and `hasMTPWeights` were read for.
    @ObservationIgnored private var storedResultsModelID: String?
    private let verification = EngineVerificationStore()

    /// The stock path's session, whose cache holds `sessionTurns` after `sessionSystem`. Nil
    /// turns mean the cache can't be trusted (a reply was cut short or failed), so the next reply
    /// rebuilds it.
    @ObservationIgnored private var session: ChatSession?
    @ObservationIgnored private var sessionSystem: String?
    @ObservationIgnored private var sessionTurns: [ChatTurn]?
    /// Changes whenever the stock session is reset or a stock reply takes it over: a reply keeps
    /// what it built only if it is still the latest.
    @ObservationIgnored private var sessionVersion = 0

    /// Whether GPU work may start. iOS refuses GPU work from a background app and MLX aborts when
    /// it does, so replies start only in the foreground, and leaving waits for `inFlight`.
    nonisolated private static let gpuAllowed = OSAllocatedUnfairLock(initialState: true)
    nonisolated private static let inFlight = DispatchGroup()
    /// Counts how often the app stopped being active, to tell whether a check ran uninterrupted.
    nonisolated private static let interruptions = OSAllocatedUnfairLock(initialState: 0)
    /// Whether the app entered the background since it was last active. While it is merely
    /// inactive, an engine job waits briefly instead of being refused (`beginEngineGPU`).
    nonisolated private static let inBackground = OSAllocatedUnfairLock(initialState: false)
    /// Longest an engine job waits for an inactive app to become active again.
    nonisolated private static let inactiveWait: TimeInterval = 2

    /// The engine's GPU guard: the same `gpuAllowed` / `inFlight` pair the stock path uses.
    nonisolated static let engineHooks = EngineHooks(
        beginGPU: { LocalModelHost.beginEngineGPU() },
        endGPU: { LocalModelHost.inFlight.leave() },
        isAllowed: { LocalModelHost.gpuAllowed.withLock { $0 } }
    )

    nonisolated private static let parameters = GenerateParameters(maxTokens: 1024, temperature: 0.7, topP: 0.8, topK: 20)
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
            Self.inBackground.withLock { $0 = true }
            MainActor.assumeIsolated { LocalModelHost.shared.unload() }
            Self.waitForGPU()
        }
        _ = center.addObserver(forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { LocalModelHost.shared.unload() }
        }
        _ = center.addObserver(forName: UIApplication.didBecomeActiveNotification, object: nil, queue: .main) { _ in
            Self.gpuAllowed.withLock { $0 = true }
            Self.inBackground.withLock { $0 = false }
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
        let previous = orphan
        loading = Task { await load(option, key: wanted, after: previous) }
        #endif
    }

    /// Returns once the current load, if any, has finished or failed.
    func settle() async {
        await loading?.value
    }

    /// Frees the model. A download in progress carries on unless `stopDownload`.
    func unload(stopDownload: Bool = false) {
        retire()
        engine = nil
        draftModel = nil
        kernelsInstalled = false
        resetSession()
        prewarming?.cancel()
        prewarming = nil
        live = nil
        guard stopDownload || !downloading else { return }
        if let loading { orphan = loading }
        loading?.cancel()
        loading = nil
        key = nil
        statusText = nil
        if case .failed = status {} else { status = .off }
        // Off the main thread: it waits for the MLX evaluation in progress, such as a stock
        // reply's prefill chunk, and the background handler must not.
        DispatchQueue.global(qos: .userInitiated).async { Memory.clearCache() }
    }

    /// Sets `retired` for the loaded engine, which is about to be let go of. Its pending check is
    /// cancelled; a check, reply or prewarm in progress ends on its own (a reply when it is done
    /// or the app leaves the screen), a check or prewarm still waiting for its turn returns
    /// without running once it gets it, and the next load waits for all of them.
    private func retire() {
        guard let engine else { return }
        let previous = retired
        let check = pendingCheck
        check?.cancel()
        // unload() cancels it, but while it waits for its turn it still holds the engine.
        let prewarm = prewarming
        let reset = invalidating
        retired = Task {
            await previous?.value
            await check?.value
            await prewarm?.value
            // A check waiting behind another one has captured the engine too.
            while checks > 0 || activeJobs > 0 {
                try? await Task.sleep(for: .milliseconds(50))
            }
            await reset?.value
            await engine.waitUntilIdle()
        }
    }

    /// Forgets the cached session, so the next reply processes the whole conversation again.
    func resetSession() {
        session = nil
        sessionSystem = nil
        sessionTurns = nil
        sessionVersion += 1
        isWarm = false
        if let engine {
            let previous = invalidating
            invalidating = Task {
                await previous?.value
                // Never between the engine jobs of a check, which needs the session unchanged.
                await waitForChecks()
                await engine.invalidateSession()
            }
        }
    }

    /// Applies changed settings: a different model, draft model setting or fast-kernel setting
    /// unloads the model (the next use loads it again); speculation mode, disk prefix and MTP
    /// change on the running engine. Turning on automatic mode or speculation runs a missing
    /// self-test or speed measurement.
    func apply(_ settings: AssistantSettings) {
        guard let key, let option = LocalModelCatalog.option(for: settings.localModelID), Self.key(option, settings) == key else {
            if key != nil {
                unload(stopDownload: true)
            }
            // Nothing (fitting) is loaded: show what is stored for the model Settings selects.
            latest = settings
            if storedResultsModelID != settings.localModelID {
                readStoredResults(modelID: settings.localModelID)
            }
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

    /// Loads `option` once `previous` (a cancelled load) has ended.
    private func load(_ option: LocalModelOption, key wanted: Key, after previous: Task<Void, Never>?) async {
        do {
            if let previous {
                // It keeps reading weights or checking its engine after being cancelled, and
                // holds that model until it ends.
                await previous.value
                try Task.checkCancellation()
            }
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
            // An unloaded engine stays in memory until the check or reply still using it ends.
            await retired?.value
            // The memory an earlier model held (`previous`'s, or one unloaded meanwhile) may still
            // be in MLX's cache, which counts against the app.
            await Task.detached { Memory.clearCache() }.value
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
            kernelReportTaken = false
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

            // Compile the GPU kernels now instead of on the first reply. The warm-up also prepares
            // the extensions: the fast kernels, when requested, test and time themselves and stay
            // only when they help.
            await QwenListener.finishPasses()
            try checkStillWanted()
            do {
                try await loaded.warmUp()
                takeKernelReport(loaded)
            } catch EngineError.leftForeground {
                // Leaving the screen: checkStillWanted unloads below. Otherwise the first reply
                // compiles them and prepares the extensions.
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
        kernelsInstalled = false
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
            fastKernelsRequested: settings.localFastKernels,
            curve: costCurve,
            kernelsInstalled: kernelsInstalled
        )
    }

    /// Brings the running engine's speculation mode, disk prefix and extensions in line with
    /// `settings`. The generator factory (and with it the draft statistics) is replaced only when
    /// the mode, the cost curve or the kernels in the model changed.
    private func applyLive(_ settings: AssistantSettings) async {
        guard let engine else { return }
        let wanted = liveOptions(settings)
        guard let current = live, current != wanted else {
            live = wanted
            return
        }
        live = wanted
        let factory = current.speculation != wanted.speculation || current.curve != wanted.curve
            || current.kernelsInstalled != wanted.kernelsInstalled
            ? EngineSetup.generatorFactory(settings: settings, host: self) : nil
        let directory = EngineSetup.prefixCacheDirectory(settings: settings)
        let extensions = ExtensionList(EngineSetup.extensions(for: settings))
        let kernels = EngineSetup.fastKernels
        let requestKernels = settings.localFastKernels
        await engine.updateConfiguration { configuration in
            if let factory {
                configuration.generatorFactory = factory
            }
            configuration.prefixCacheDirectory = directory
            configuration.extensions = extensions.items
            // Assigning `extensions` dropped the fast-kernel request, which lives among them.
            kernels?.request(requestKernels, in: &configuration)
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
        storedResultsModelID = modelID
        let directory = snapshot ?? Self.downloadedSnapshot(modelID)
        selfTest = directory.flatMap { verification.selfTest(for: .init(modelID: modelID, snapshot: $0.lastPathComponent)) }
        costCurve = verification.costCurve(for: curveKey(modelID))
        kernelGain = verification.kernelGain(for: .init(modelID: modelID))
        kernelSummary = nil
        hasMTPWeights = directory.map { EngineSetup.hasMTPWeights(in: $0) } ?? false
    }

    /// Where `modelID`'s cost curve is stored: the curve with the fast kernels in the model is
    /// another one.
    private func curveKey(_ modelID: String) -> EngineVerificationStore.DeviceKey {
        .init(modelID: modelID, fastKernels: kernelsInstalled)
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
        // Nothing runs when cancelled first (a load that unload() stopped).
        _ = try? await exclusively(engine) {
            // Each check only while `engine` is still the loaded one: an unloaded engine's results
            // aren't kept, and checking it would keep it in memory for seconds.
            if self.engine === engine, settings.localEngineMode == .automatic, selfTest == nil {
                statusText = "Checking the on-device engine…"
                _ = await performSelfTest(engine)
            }
            if self.engine === engine, settings.localSpeculation != .off, costCurve == nil, usesEngine(settings) {
                statusText = "Measuring the on-device engine's speed…"
                _ = try? await performCostProbe(engine)
            }
            statusText = nil
        }
        updateSpeculation()
    }

    /// Runs the engine self-test now (Settings, the benchmark) and stores its result. Nil when no
    /// model is loaded (also when it was unloaded before the test's turn came), when the caller was
    /// cancelled before it started, or when the app stopped being active during the run (nothing
    /// is stored).
    @discardableResult
    func runSelfTest() async -> EngineSelfTest.Result? {
        guard let engine, status == .ready else { return nil }
        let result = try? await exclusively(engine) { () async -> EngineSelfTest.Result? in
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
        guard let engine, status == .ready else { throw Self.notLoaded }
        return try await exclusively(engine) {
            statusText = "Measuring the on-device engine's speed…"
            defer { statusText = nil }
            guard let curve = try await performCostProbe(engine) else { throw Self.leftForeground }
            return curve
        }
    }

    // MARK: - Fast kernels

    /// Whether this build has the fast kernels (launch setup put them in
    /// `EngineSetup.extraExtensions`).
    var fastKernelsAvailable: Bool {
        EngineSetup.fastKernels != nil
    }

    /// Whether the stored comparison says the fast kernels help on this device. Settings offers
    /// the fast-kernel switch only then (and to switch it off).
    var fastKernelsHelp: Bool {
        (kernelGain ?? -1) >= EngineSetup.kernelGainThreshold
    }

    /// Compares the fast kernels with the stock layers on the loaded model now (Settings, the
    /// benchmark): the kernel self-test, then the cost curve without and with them. The model
    /// keeps the layers it has. Stores the gain and returns the comparison's summary line.
    @discardableResult
    func measureFastKernels() async throws -> String {
        guard let kernels = EngineSetup.fastKernels else {
            throw AssistantError.missingConfiguration("This build has no fast kernels.")
        }
        guard let engine, status == .ready else { throw Self.notLoaded }
        return try await exclusively(engine) {
            statusText = "Testing the fast kernels…"
            defer { statusText = nil }
            let before = Self.interruptions.withLock { $0 }
            await QwenListener.finishPasses()
            let verdict = try await kernels.measure(engine)
            // A run the app interrupted timed the interruption: measure again later.
            guard Self.interruptions.withLock({ $0 }) == before, self.engine === engine else { throw Self.leftForeground }
            take(verdict, from: engine, prepared: false)
            return verdict.summary
        }
    }

    /// Reads what the fast kernels' `prepare` found for `engine`, once per load: in the warm-up,
    /// else after the first engine reply (whose job prepared them). Their report describes the
    /// latest `prepare` of any engine, so it is read only when this engine has the kernels among
    /// its extensions (and so prepared them).
    private func takeKernelReport(_ engine: InferenceEngine) {
        guard !kernelReportTaken, self.engine === engine else { return }
        kernelReportTaken = true
        guard let kernels = EngineSetup.fastKernels, live?.extensions.contains(ObjectIdentifier(kernels)) == true,
              let verdict = kernels.report, verdict.modelID == engine.loaded.id
        else { return }
        take(verdict, from: engine, prepared: true)
    }

    /// Stores `verdict`'s gain and shows its summary. A `prepare`'s verdict also says whether the
    /// model runs the kernels; the cost curve then becomes the one measured that way.
    private func take(_ verdict: any FastKernelsVerdict, from engine: InferenceEngine, prepared: Bool) {
        guard self.engine === engine else { return }
        kernelSummary = verdict.summary
        if let gain = verdict.gain {
            let key = EngineVerificationStore.DeviceKey(modelID: engine.loaded.id)
            verification.setKernelGain(gain, for: key)
            kernelGain = verification.kernelGain(for: key)
        }
        if prepared, verdict.isEnabled != kernelsInstalled {
            kernelsInstalled = verdict.isEnabled
            costCurve = verification.costCurve(for: curveKey(engine.loaded.id))
        }
    }

    /// "On; an 8-token check takes 31% less time with them (25% needed)", or that they were
    /// never compared on this iPhone.
    var fastKernelsSummary: String {
        let state = kernelsInstalled ? "On" : "Off"
        guard let kernelGain else { return "\(state); not compared on this iPhone yet" }
        let percent = Int((abs(kernelGain) * 100).rounded())
        let needed = Int((EngineSetup.kernelGainThreshold * 100).rounded())
        let effect = kernelGain >= 0 ? "\(percent)% less time" : "\(percent)% more time"
        return "\(state); an 8-token check takes \(effect) with them (\(needed)% needed)"
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
        verification.setCostCurve(curve, for: curveKey(engine.loaded.id))
        costCurve = curve
        // Draft with the measured curve from the next reply on.
        if let latest {
            await applyLive(latest)
        }
        return curve
    }

    /// Runs `body` with `engine` to itself: it waits for any other check and a pending session
    /// reset, then for replies and prewarms in progress, and keeps new ones (and new resets)
    /// waiting until it returns. A check works in several engine jobs and needs the session
    /// unchanged between them. Throws, without running `body`, `CancellationError` when the caller
    /// is cancelled before `body` starts and `notLoaded` when `engine` was unloaded meanwhile.
    private func exclusively<T>(_ engine: InferenceEngine, _ body: () async throws -> T) async throws -> T {
        // Counted from here: a check waiting for its turn holds `engine` too, and `retire()`
        // waits for it.
        checks += 1
        defer { checks -= 1 }
        // Claimed right after the last wait, with no suspension in between.
        while let next = blocker {
            await wait(for: next)
        }
        try Task.checkCancellation()
        guard self.engine === engine else { throw Self.notLoaded }
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
            // Throws once cancelled; ignoring that would turn this into a busy loop.
            try await Task.sleep(for: .milliseconds(50))
        }
        try Task.checkCancellation()
        guard self.engine === engine else { throw Self.notLoaded }
        return try await body()
    }

    /// Returns once no check is running.
    private func waitForChecks() async {
        while let running = maintenance {
            await running.value
        }
    }

    /// What a reply, prewarm or check waits for before it starts: a running check, else a
    /// pending session reset.
    private var blocker: Task<Void, Never>? {
        maintenance ?? invalidating
    }

    /// Waits for `blocker`, and forgets it if it was the session reset.
    private func wait(for blocker: Task<Void, Never>) async {
        await blocker.value
        if invalidating == blocker {
            invalidating = nil
        }
    }

    /// Waits until no check is running and no session reset is pending, then counts the caller
    /// in `activeJobs` with no suspension in between, so a check can't start in that gap. The
    /// caller takes it out of `activeJobs` when it ends.
    private func beginJob() async {
        while let next = blocker {
            await wait(for: next)
        }
        activeJobs += 1
    }

    /// `beginJob` for a prewarm of `engine`, which also waits until no reply is in progress: a
    /// stock reply waits for a prewarm that has started, so a started prewarm must never wait
    /// for a reply. False, without counting anything, when the prewarm is no longer wanted.
    private func beginPrewarmJob(_ engine: InferenceEngine) async -> Bool {
        while true {
            if let next = blocker {
                await wait(for: next)
            } else if self.engine !== engine || Task.isCancelled {
                return false
            } else if activeJobs > 0 {
                try? await Task.sleep(for: .milliseconds(50))
            } else {
                activeJobs += 1
                return true
            }
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
            if await beginPrewarmJob(engine) {
                let job = Task {
                    await QwenListener.finishPasses()
                    do {
                        try await engine.prewarm(system: system, tools: tools)
                        if self.engine === engine {
                            isWarm = true
                        }
                    } catch {
                        // Left the foreground, or the template refused: the next reply prefills.
                    }
                }
                prewarmJob = job
                await job.value
                if prewarmJob == job {
                    prewarmJob = nil
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
        await beginJob()
        defer { activeJobs -= 1 }
        try Task.checkCancellation()
        guard let engine, status == .ready else {
            throw AssistantError.missingConfiguration("The on-device model isn't loaded. Try again.")
        }
        if showLoading { emit(.reply(.activity(nil))) }
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
        takeKernelReport(engine)
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
        // The session runs outside the engine's queue: let any engine job (a prewarm that has
        // started) finish first, so the two never use the GPU at once. A prewarm that hasn't
        // started waits for this reply.
        await prewarmJob?.value
        await engine.waitUntilIdle()
        try Task.checkCancellation()
        // As on the engine path, no text in the conversation becomes a control token.
        let escaper = SpecialTokenEscaper(renderer: engine.loaded.renderer)
        let prompt = escaper.escape(LocalSessionPlan.content(of: plan.newTurn))
        let source: StockSource
        var speculation: SpeculativeDecodingConfig?
        if plan.action == .append, let cached = session {
            source = .session(cached, prompt: prompt)
        } else {
            let history: [ChatTurn]
            if case .rebuild(let turns) = plan.action { history = turns } else { history = [] }
            // What a ChatSession built from this history would render. The system prompt goes in
            // as the first history message, which the session reads once; as `instructions` it
            // would be prefilled again before every turn (F1).
            let messages: [Chat.Message] = [.system(system)] + history.map { Self.message($0, escaper: escaper) } + [.user(prompt)]
            source = .fresh(engine.loaded, messages: messages)
            speculation = stockSpeculation()
            session = nil
        }
        // Trusted again only once this reply completes.
        sessionVersion += 1
        let version = sessionVersion
        sessionSystem = system
        sessionTurns = nil

        guard Self.beginGPU() else { throw Self.leftForeground }
        Memory.peakMemory = 0
        emit(.progress(.responseStarted))
        // Generated off the main actor, which the background handler blocks while it waits for
        // `inFlight`: `relay` stops the generation on leaving the foreground and leaves `inFlight`.
        let (items, sink) = AsyncThrowingStream<Generation, Error>.makeStream()
        async let relayed = LocalModelHost.relay(source, into: sink)

        var stats = LocalGenerationStats(reusedSession: source.isCached, engine: "stock")
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
            for try await item in items {
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
        // If the loop stopped early, `relay` stops at its next item. Then wait until generation
        // has really stopped using the GPU and the cache.
        sink.finish()
        let outcome = await relayed
        show(filter.finish())
        stats.promptTime += outcome.prefillTime
        stats.peakMemoryBytes = Memory.peakMemory

        if let failure { throw failure }
        try Task.checkCancellation()
        switch stopReason {
        case .stop:
            if sessionVersion == version {
                if let cache = outcome.cache {
                    // Later turns append to this cache as to the cache of a session built from
                    // the history. This reply decoded without the draft model, and the session
                    // gets no draft cache: the draft model starts one at the next turn and drafts
                    // without the earlier context (fewer drafts accepted, still lossless).
                    session = ChatSession(
                        engine.loaded.container,
                        instructions: nil,
                        cache: cache,
                        speculativeDecoding: speculation,
                        generateParameters: Self.parameters,
                        additionalContext: Self.chatContext
                    )
                }
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
        if fastKernelsAvailable || kernelGain != nil {
            // The kernels' own line: their verdict with c(8) before → after.
            lines.append(kernelSummary ?? "Fast kernels: not prepared for this model")
            lines.append("Fast kernels on this iPhone: \(fastKernelsSummary)")
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
    private static let notLoaded = AssistantError.missingConfiguration("Load the on-device model first.")

    /// Qwen-family chat templates: answer directly, without a reasoning block.
    nonisolated private static let chatContext: [String: any Sendable] = ["enable_thinking": false]

    /// A history turn as the stock session reads it, its text escaped as `TurnRenderer` escapes it.
    private static func message(_ turn: ChatTurn, escaper: SpecialTokenEscaper) -> Chat.Message {
        switch turn.role {
        case .user: return .user(escaper.escape(LocalSessionPlan.content(of: turn)))
        case .assistant: return .assistant(escaper.escape(turn.text))
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

    /// `beginGPU` for engine jobs, which ask on the engine queue. While the app is inactive but
    /// not in the background (a permission alert a tool raised, Control Center), it waits up to
    /// `inactiveWait` for the app to become active again instead of refusing at once: the job
    /// after a tool round would otherwise fail although the app never left the screen. The stock
    /// path asks on the main thread, which delivers `didBecomeActive`, so it can't wait.
    nonisolated private static func beginEngineGPU() -> Bool {
        let deadline = DispatchTime.now() + inactiveWait
        while !beginGPU() {
            guard !inBackground.withLock({ $0 }), DispatchTime.now() < deadline else { return false }
            Thread.sleep(forTimeInterval: 0.02)
        }
        return true
    }

    /// Generates a stock reply off the main actor, passing it on to `sink`, and stops it at the
    /// first item or prefill chunk after the app stopped being active (or once cancelled, or once
    /// `sink` is finished). Then waits until generation has really stopped using the GPU and the
    /// cache, and leaves `inFlight`, which the caller entered.
    ///
    /// A cached session answers through `ChatSession`; only the new turn is prefilled there, which
    /// can't be stopped. A fresh or rebuilt session prefills the whole conversation, so its first
    /// reply is generated by `generateFresh` instead, whose prefill can be.
    nonisolated private static func relay(
        _ source: StockSource,
        into sink: AsyncThrowingStream<Generation, Error>.Continuation
    ) async -> StockRelayResult {
        var result = StockRelayResult()
        switch source {
        case .session(let session, let prompt):
            do {
                // The loop holds the only reference to the session's stream: leaving it drops the
                // stream, which stops the generation.
                for try await item in session.streamDetails(to: prompt) {
                    guard pass(item, to: sink) else { break }
                }
                sink.finish()
            } catch {
                sink.finish(throwing: error)
            }
            await session.synchronize()
        case .fresh(let loaded, let messages):
            do {
                result = try await generateFresh(messages, with: loaded, into: sink)
                sink.finish()
            } catch {
                sink.finish(throwing: error)
            }
            // Whatever ended it, no work of this reply may still be on the GPU.
            Stream().synchronize()
        }
        inFlight.leave()
        return result
    }

    /// Passes `item` on to `sink`; false once generation should stop instead.
    nonisolated private static func pass(_ item: Generation, to sink: AsyncThrowingStream<Generation, Error>.Continuation) -> Bool {
        guard !Task.isCancelled, gpuAllowed.withLock({ $0 }) else { return false }
        if case .terminated = sink.yield(item) { return false }
        return true
    }

    /// The first reply of a fresh or rebuilt stock session, generated as a `ChatSession` built
    /// from `messages` (system prompt, history, new user turn) would: the same rendering and the
    /// same `TokenIterator`. All but the last prompt token are prefilled first in `StockPrefill`'s
    /// chunks, which stop once the app stops being active; the iterator starts from that token.
    /// The cache then holds what that session's cache would, and is returned with the prefill's
    /// seconds. Returns no cache when the prefill was stopped.
    nonisolated private static func generateFresh(
        _ messages: [Chat.Message],
        with loaded: LoadedModel,
        into sink: AsyncThrowingStream<Generation, Error>.Continuation
    ) async throws -> StockRelayResult {
        let container = loaded.container
        let processor = await container.processor
        let input = try await processor.prepare(input: UserInput(
            chat: messages, processing: .init(resize: CGSize(width: 512, height: 512)), tools: nil,
            additionalContext: chatContext))
        let tokens = input.text.tokens
        let count = tokens.size
        guard count > 0 else { throw AssistantError.invalidResponse("There's no question to answer.") }
        let started = ContinuousClock.now
        let cache: [KVCache]
        do {
            cache = try StockPrefill.run(tokens[..<(count - 1)], model: loaded.model, parameters: parameters) {
                !Task.isCancelled && gpuAllowed.withLock { $0 }
            }
        } catch EngineError.leftForeground {
            return StockRelayResult()
        }
        let prefillTime = seconds(started.duration(to: .now))
        // `parameters` has no penalties, whose processor would see only this token of the prompt.
        let iterator = try TokenIterator(
            input: LMInput(tokens: tokens[(count - 1)...]), model: loaded.model, cache: cache, parameters: parameters)
        let (items, generation) = MLXLMCommon.generateTask(
            promptTokenCount: count, modelConfiguration: await container.configuration,
            tokenizer: await container.tokenizer, iterator: iterator, tools: nil)
        for await item in items {
            guard pass(item, to: sink) else { break }
        }
        // Stops a generation left early, and waits until it no longer uses the cache.
        generation.cancel()
        await generation.value
        return StockRelayResult(cache: cache, prefillTime: prefillTime)
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

/// What a stock reply generates from, handed to the task that generates it (`relay`); nothing
/// else uses it until that task ends: the cached session and the new turn's text, or the
/// loaded model and the whole conversation for a fresh or rebuilt session.
private enum StockSource: @unchecked Sendable {
    case session(ChatSession, prompt: String)
    case fresh(LoadedModel, messages: [Chat.Message])

    var isCached: Bool {
        if case .session = self { return true }
        return false
    }
}

/// What `relay` hands back: a fresh session's cache once its reply has been generated (the
/// caller keeps it only if the reply completed), and how long its prefill took.
private struct StockRelayResult: @unchecked Sendable {
    var cache: [KVCache]?
    var prefillTime: TimeInterval = 0
}

/// Engine extensions handed to the engine queue. They are prepared and used there only.
private struct ExtensionList: @unchecked Sendable {
    let items: [any EngineExtension]

    init(_ items: [any EngineExtension]) {
        self.items = items
    }
}
