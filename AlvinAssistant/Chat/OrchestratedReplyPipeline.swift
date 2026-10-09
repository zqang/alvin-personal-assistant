import AssistantKit
import Foundation

/// The app's reply pipeline: routing between Claude and the on-device model, client tools, deep
/// mode, early start and connection prewarm (plan §5).
///
/// Each reply:
/// 1. reads the conversation (plus `pendingUser`, for a voice reply started early) into turns;
/// 2. builds the engines: Claude with the device tools, deep mode, the on-device model with its
///    tool subset (when downloaded), or the OpenAI-compatible service;
/// 3. decides the route: with a single provider, that provider (Claude may go deep); with
///    automatic routing (Claude as the provider), `RouteDecider` with live signals;
/// 4. runs it through `Orchestrator`, which yields `.routed` first and falls back once before the
///    reply commits;
/// 5. records deep-mode use and each engine's time to first text in the `SettingsStore`.
///
/// Side-effect tools wait on the request's commit gate, so a reply started early acts only once
/// its turn is final. A deep reply started early doesn't start at all before then: a discarded one
/// would spend deep-mode work and budget.
@MainActor
final class OrchestratedReplyPipeline: ReplyPipeline {
    private let store: SettingsStore

    /// The on-device model's download state, read at most every few seconds: `missingSetup()` runs
    /// in the chat screen's body, which streaming redraws for every chunk.
    private var downloadCheck: (modelID: String, downloaded: Bool, at: Date)?

    /// Warms the connection replies use (the same transport), at most every 30 s.
    private static let prewarmer = ConnectionPrewarmer(transport: ReplyService.transport)

    init(store: SettingsStore) {
        self.store = store
    }

    var supportsEarlyStart: Bool { true }

    // MARK: - Replies

    func stream(_ request: ReplyRequest) -> AsyncThrowingStream<AssistantEvent, Error> {
        let settings = store.settings
        var messages = request.conversation.orderedMessages.map(\.stored)
        if let pendingUser = request.pendingUser {
            messages.append(pendingUser)
        }
        let turns = PromptBuilder.turns(from: messages)
        let system = PromptBuilder.systemPrompt(userName: settings.userName, customInstructions: settings.customInstructions)

        let engines: Orchestrator.Engines
        do {
            engines = try makeEngines(settings: settings, turns: turns, request: request)
        } catch {
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: error)
            }
        }
        let decision = decide(settings: settings, turns: turns, request: request)
        // Voice waits less for a silent cloud when the on-device model can answer at once.
        let watchdog: Duration = request.inputIsVoice && LocalModelHost.shared.status == .ready ? .seconds(3) : .seconds(8)
        let orchestrator = Orchestrator(decision: decision, engines: engines, firstEventTimeout: watchdog)
        // The OpenAI-compatible service isn't one of the router's engines; its times would skew
        // Claude's.
        let recordsLatency = settings.provider != .openAICompatible
        return relay(orchestrator, system: system, turns: turns, deep: decision.mode == .deep, gate: request.commitGate, recordsLatency: recordsLatency)
    }

    /// The engines for one reply under `settings`.
    private func makeEngines(settings: AssistantSettings, turns: [ChatTurn], request: ReplyRequest) throws -> Orchestrator.Engines {
        let gate = request.commitGate
        let toolContext: @Sendable () -> ToolContext = { ToolContext(commitGate: gate) }
        switch settings.provider {
        case .anthropic:
            let claude = ReplyService.claudeConfiguration(settings: settings, store: store)
            let tools = DeviceTools.registry(settings: settings)
            let cloud = ClaudeProvider(
                configuration: claude,
                transport: ReplyService.transport,
                tools: ToolRunner(registry: tools.covering(turns)),
                options: .init(),
                toolContext: toolContext
            )
            var deep = request.inputIsVoice ? DeliberationConfiguration.voice() : DeliberationConfiguration.typed()
            deep.strategy = settings.deepStrategy
            let workerModel = settings.deepWorkerModel.trimmed
            deep.workerModel = workerModel.isEmpty ? nil : workerModel
            let deliberation = Deliberation(
                configuration: deep,
                claude: claude,
                transport: ReplyService.transport,
                tools: tools,
                toolContext: toolContext
            )
            // With a single provider, Claude answers alone (plan §5.2: no on-device fallback).
            let local = settings.routingMode == .automatic && localAvailable(settings) ? localProvider(settings: settings, gate: gate) : nil
            return Orchestrator.Engines(cloud: cloud, deep: deliberation, local: local)
        case .onDevice:
            // Downloaded on first use, as before routing existed.
            return Orchestrator.Engines(local: localProvider(settings: settings, gate: gate))
        case .openAICompatible:
            let compatible = try ReplyService.compatibleProvider(settings: settings, store: store)
            return Orchestrator.Engines(cloud: LegacyProviderAdapter(compatible))
        }
    }

    /// The on-device model with the tools and handoff `LocalPromptFactory` gives it, which the
    /// prewarm uses too.
    private func localProvider(settings: AssistantSettings, gate: CommitGate?) -> LocalProvider {
        let prompt = LocalPromptFactory.make(settings: settings, store: store)
        let tools: (any ToolExecutor)? = prompt.tools.isEmpty ? nil : ToolRunner(registry: prompt.tools, lenientInput: true)
        return LocalProvider(settings: settings, tools: tools, handoffAvailable: prompt.handoff, commitGate: gate)
    }

    /// The route of one reply.
    private func decide(settings: AssistantSettings, turns: [ChatTurn], request: ReplyRequest) -> RouteDecision {
        switch settings.provider {
        case .onDevice:
            return RouteDecision(engine: .local, reason: .userChoice)
        case .openAICompatible:
            return RouteDecision(engine: .cloud, reason: .userChoice)
        case .anthropic:
            break
        }
        let host = LocalModelHost.shared
        let automatic = settings.routingMode == .automatic
        var signals = RouteSignals(
            policy: automatic ? .automatic(preferLocal: settings.preferOnDevice) : .cloudOnly,
            isOnline: Connectivity.shared.isOnline,
            cloudConfigured: !store.secret(.anthropic).isEmpty,
            // With a single provider there is no on-device fallback.
            localAvailable: automatic && localAvailable(settings),
            localReady: automatic && host.status == .ready,
            inputIsVoice: request.inputIsVoice,
            deepRequested: request.deep && settings.deepMode != .off,
            deepMode: settings.deepMode,
            deepBudgetLeft: store.deepRunsLeft() > 0,
            fastLocalSmallTalk: settings.fastLocalSmallTalk
        )
        signals.setExpectations(from: store.latency)
        let utterance = turns.last(where: { $0.role == .user })?.text ?? ""
        let decider = RouteDecider()
        let intents = decider.classifier.classify(utterance)
        // A device action goes on device only when the on-device reply can run the tools: the
        // Alvin engine with the device tools on. MLX's stock session has no tools.
        if intents.contains(.deviceAction), !(settings.deviceToolsEnabled && host.usesEngine(settings)) {
            signals.localReady = false
        }
        return decider.decide(intents: intents, signals: signals)
    }

    /// Streams `orchestrator`'s events, recording deep-mode use and the time to first text.
    private func relay(
        _ orchestrator: Orchestrator,
        system: String,
        turns: [ChatTurn],
        deep: Bool,
        gate: CommitGate?,
        recordsLatency: Bool
    ) -> AsyncThrowingStream<AssistantEvent, Error> {
        let store = store
        return AsyncThrowingStream { continuation in
            let task = Task { @MainActor in
                do {
                    if deep {
                        // A reply started early: deep mode begins only once the turn is final
                        // (the wait throws when the reply is discarded).
                        try await gate?.wait()
                        store.recordDeepRun()
                    }
                    var firstText = FirstTextTap()
                    for try await event in orchestrator.streamEvents(system: system, turns: turns) {
                        if recordsLatency, let sample = firstText.observe(event, at: .now) {
                            store.recordFirstText(engine: sample.engine, mode: sample.mode, seconds: sample.seconds)
                        }
                        continuation.yield(event)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Setup, prewarm, names

    func missingSetup() -> String? {
        let settings = store.settings
        guard settings.provider == .anthropic, settings.routingMode == .automatic else {
            return ReplyService.missingSetup(settings: settings, store: store)
        }
        let downloaded = localAvailable(settings)
        if !Connectivity.shared.isOnline, !downloaded {
            return "You're offline. Download an on-device model on Wi-Fi to talk offline."
        }
        if !store.secret(.anthropic).isEmpty || downloaded {
            return nil
        }
        return "Add your Anthropic API key in Settings, or download an on-device model, to start."
    }

    func prewarm(inputIsVoice: Bool) {
        let settings = store.settings
        switch settings.provider {
        case .anthropic:
            if Connectivity.shared.isOnline, !store.secret(.anthropic).isEmpty,
               let request = try? ClaudeProvider.prewarmRequest(configuration: ReplyService.claudeConfiguration(settings: settings, store: store))
            {
                let prewarmer = Self.prewarmer
                Task { await prewarmer.prewarm(request) }
            }
            if settings.routingMode == .automatic {
                // Loads a downloaded model and makes its system prompt resident; never downloads.
                LocalModelHost.shared.prewarm(settings: settings)
            }
        case .onDevice:
            if inputIsVoice {
                // As before: voice mode gets the model ready, downloading it if needed.
                LocalModelHost.shared.prepare(settings)
            } else {
                LocalModelHost.shared.prewarm(settings: settings)
            }
        case .openAICompatible:
            break
        }
    }

    func displayName(for decision: RouteDecision?) -> String {
        let settings = store.settings
        let engine = decision?.engine ?? (settings.provider == .onDevice ? .local : .cloud)
        switch engine {
        case .local:
            let model = LocalModelCatalog.option(for: settings.localModelID)?.displayName ?? "model"
            return "On device · \(model)"
        case .cloud:
            if settings.provider == .openAICompatible {
                return settings.compatibleModel
            }
            let name = ClaudeModelCatalog.displayName(for: settings.claudeModel)
            return decision?.mode == .deep ? "\(name) · Deep" : name
        }
    }

    // MARK: - Helpers

    /// Whether the on-device model `settings` selects can answer without a download: it is
    /// loaded, or its files are on the iPhone.
    private func localAvailable(_ settings: AssistantSettings) -> Bool {
        guard LocalModelCatalog.option(for: settings.localModelID) != nil else { return false }
        // Read first, so a view calling this redraws when a download finishes.
        if LocalModelHost.shared.status == .ready { return true }
        let now = Date()
        if let check = downloadCheck, check.modelID == settings.localModelID, now.timeIntervalSince(check.at) < 5 {
            return check.downloaded
        }
        let downloaded = LocalModelHost.shared.isDownloaded(settings.localModelID)
        downloadCheck = (settings.localModelID, downloaded, now)
        return downloaded
    }
}

/// Measures a reply's time to first text for `LatencyEstimator`: from the latest `.routed` event
/// (the route, or a fallback's) to the first non-empty text. A reply that loaded the on-device
/// model first or ran tools before its text measures that, not the engine, so it gives no sample.
struct FirstTextTap {
    private var route: RouteDecision?
    private var since = ContinuousClock.now
    private var counts = true
    private var done = false

    mutating func observe(_ event: AssistantEvent, at now: ContinuousClock.Instant) -> (engine: ReplyEngine, mode: ReplyMode, seconds: TimeInterval)? {
        switch event {
        case .routed(let decision):
            route = decision
            since = now
            counts = true
        case .cue(.loadingModel), .toolRound:
            counts = false
        case .reply(.text(let text)) where !text.isEmpty:
            guard !done else { return nil }
            done = true
            guard counts, let route else { return nil }
            let elapsed = since.duration(to: now).components
            let seconds = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
            return (route.engine, route.mode, seconds)
        default:
            break
        }
        return nil
    }
}
