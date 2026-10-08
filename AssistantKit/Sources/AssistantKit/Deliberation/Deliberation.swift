import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// One parallel worker of deep mode: a same-model Claude call with its own brief and effort,
/// whose notes go to the merger. The standard briefs are in `WorkerBriefs.swift`.
public struct WorkerBrief: Equatable, Sendable {
    /// Names the worker. The standard ids also name its section of the merger's notes
    /// (`researcher` → `<research>`, `reasoner` → `<reasoning>`, `critic` → `<critique>`).
    public var id: String
    /// The worker's brief, sent after the user's last message.
    public var instruction: String
    /// The worker's effort, sent as a per-message effort change where the model accepts one.
    public var effort: String
    /// Whether the worker's read-only tool calls run. Every worker declares the same tools, so all
    /// requests share one cached prefix; a worker without tools gets "Not available here." for any
    /// call. Side-effect tools never run in a worker.
    public var usesTools: Bool
    /// Whether the worker's activity lines (e.g. "Searching the web for …") are shown to the user,
    /// at most one new line every 3 s.
    public var forwardsActivity: Bool

    public init(id: String, instruction: String, effort: String, usesTools: Bool, forwardsActivity: Bool) {
        self.id = id
        self.instruction = instruction
        self.effort = effort
        self.usesTools = usesTools
        self.forwardsActivity = forwardsActivity
    }
}

/// How a deep-mode reply runs.
public struct DeliberationConfiguration: Equatable, Sendable {
    /// `.single`: one same-model call at `singleEffort`. `.parallel`: `workers`, then a merger.
    public var strategy: DeepStrategy = .single
    /// The effort of the `.single` call.
    public var singleEffort = "high"
    /// The `.parallel` workers, run at the same time.
    public var workers: [WorkerBrief] = [.researcher, .reasoner, .critic]
    /// The workers' model. Nil (or blank) uses the conversation's model, which shares its cached
    /// prefix; another model doesn't.
    public var workerModel: String? = nil
    /// When `.parallel` stops waiting for workers; workers still running are cut off and keep what
    /// they wrote. It is also how long a call the user waits on (the `.single` call, the merger or
    /// the plain answer) may go without a word from the server before the reply gives up.
    /// `.cue(.stillThinking)` plays at half of it if no text has arrived. Zero or less means no
    /// deadline, no time limit and no cue.
    public var deadline: Duration
    /// Output cap of each worker; thinking counts toward it.
    public var workerMaxTokens = 12_000
    /// Output cap of the merger.
    public var mergerMaxTokens: Int
    /// The merger's brief.
    public var mergerInstruction: String

    public init(
        strategy: DeepStrategy = .single,
        singleEffort: String = "high",
        workers: [WorkerBrief] = [.researcher, .reasoner, .critic],
        workerModel: String? = nil,
        deadline: Duration,
        workerMaxTokens: Int = 12_000,
        mergerMaxTokens: Int,
        mergerInstruction: String = DeliberationConfiguration.defaultMergerInstruction
    ) {
        self.strategy = strategy
        self.singleEffort = singleEffort
        self.workers = workers
        self.workerModel = workerModel
        self.deadline = deadline
        self.workerMaxTokens = workerMaxTokens
        self.mergerMaxTokens = mergerMaxTokens
        self.mergerInstruction = mergerInstruction
    }

    /// For spoken input: a 25 s deadline and a 4,000-token merger.
    public static func voice() -> Self {
        Self(deadline: .seconds(25), mergerMaxTokens: 4_000)
    }

    /// For typed input: a 60 s deadline and a 16,000-token merger.
    public static func typed() -> Self {
        Self(deadline: .seconds(60), mergerMaxTokens: 16_000)
    }
}

/// Deep mode: a Claude reply that takes longer to answer better.
///
/// Both strategies first yield `.cue(.deepThinking)` and `.reply(.activity("Thinking it through"))`,
/// before any request is sent. If no text has arrived at half the deadline, `.cue(.stillThinking)`
/// follows.
///
/// - `.single`: one call with `singleEffort` and the full tools; its events pass through.
/// - `.parallel`:
///   - The workers run at the same time. Each declares the same tools as the merger, so every
///     request reads the same cached prefix, but side-effect tools are blocked
///     (`ToolRegistry.readOnly()`), and only a worker with `usesTools` runs the read-only ones.
///   - Each worker collects its text. A refusal or an error drops the worker; a truncated answer,
///     or one cut off by the deadline, keeps its partial text. Worker events are not passed on,
///     except a forwarding worker's activity lines.
///   - With notes from at least one worker, the merger answers with the full tools, the notes as an
///     `<analyst_notes>` block at the end of the user's last message, and `mergerInstruction`; its
///     events pass through. Without notes, a plain call answers instead, unless the deadline passed
///     without any worker reaching the server: then the reply throws `URLError(.timedOut)` at
///     once rather than send another request into a network that has stayed silent.
///
/// The cue counts as the reply's first event, so the orchestrator's first-event watchdog never
/// fires for deep mode, and deep mode keeps its own: if the call the user waits on (the `.single`
/// call, the merger or the plain answer) has heard nothing from the server by the deadline, not
/// even the start of a response, the network is presumed down (even if it looks up). The call is
/// cancelled and the reply throws `URLError(.timedOut)`, so the orchestrator can fall back to the
/// on-device model before the answer began, rather than after URLSession's own timeout and
/// retries.
///
/// Ending or cancelling the returned stream cancels every request in flight.
public struct Deliberation: AssistantProvider {
    /// The activity line while deep mode works.
    public static let thinkingActivity = "Thinking it through"
    /// The shortest time between two new worker activity lines, in seconds.
    static let activityInterval: TimeInterval = 3

    public let configuration: DeliberationConfiguration
    public let claude: ClaudeConfiguration
    private let transport: HTTPStreamingTransport
    private let tools: ToolRegistry
    private let toolContext: @Sendable () -> ToolContext
    private let hooks: Hooks

    /// - Parameters:
    ///   - claude: the conversation's Claude configuration; its effort is the merger's.
    ///   - tools: the full tools. Tools that `turns` used but `tools` lacks are stubbed.
    ///   - toolContext: the context of each tool round (e.g. the commit gate).
    public init(
        configuration: DeliberationConfiguration,
        claude: ClaudeConfiguration,
        transport: HTTPStreamingTransport,
        tools: ToolRegistry,
        toolContext: @escaping @Sendable () -> ToolContext = { ToolContext() }
    ) {
        self.init(configuration: configuration, claude: claude, transport: transport, tools: tools, toolContext: toolContext, hooks: Hooks())
    }

    init(
        configuration: DeliberationConfiguration,
        claude: ClaudeConfiguration,
        transport: HTTPStreamingTransport,
        tools: ToolRegistry,
        toolContext: @escaping @Sendable () -> ToolContext = { ToolContext() },
        hooks: Hooks
    ) {
        self.configuration = configuration
        self.claude = claude
        self.transport = transport
        self.tools = tools
        self.toolContext = toolContext
        self.hooks = hooks
    }

    /// Seams for tests.
    struct Hooks: Sendable {
        /// Waits for the deadline and its halfway point.
        var sleep: @Sendable (Duration) async throws -> Void
        /// Monotonic seconds, for spacing worker activity lines.
        var uptime: @Sendable () -> TimeInterval
        /// Called with a worker's id when it has ended and its outcome is settled.
        var workerEnded: @Sendable (String) -> Void

        init(
            sleep: @escaping @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) },
            uptime: @escaping @Sendable () -> TimeInterval = { ProcessInfo.processInfo.systemUptime },
            workerEnded: @escaping @Sendable (String) -> Void = { _ in }
        ) {
            self.sleep = sleep
            self.uptime = uptime
            self.workerEnded = workerEnded
        }
    }

    public func streamEvents(system: String, turns: [ChatTurn]) -> AsyncThrowingStream<AssistantEvent, Error> {
        AsyncThrowingStream { continuation in
            let output = Output(continuation: continuation, uptime: hooks.uptime)
            // The cue is local, so the user hears it at once, before any request is sent.
            output.begin()
            let task = Task {
                do {
                    try await run(system: system, turns: turns, output: output)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Running

    private func run(system: String, turns: [ChatTurn], output: Output) async throws {
        // Every request of this reply declares the same tools, including stubs for tools the
        // history used, so the API accepts the history and the prefix stays cached.
        let registry = tools.covering(turns)
        let deadline = configuration.deadline
        let sleep = hooks.sleep
        try await withThrowingTaskGroup(of: Void.self) { group in
            if deadline > .zero {
                group.addTask {
                    do {
                        try await sleep(deadline / 2)
                    } catch {
                        return
                    }
                    guard !Task.isCancelled else { return }
                    output.stillThinking()
                }
            }
            switch configuration.strategy {
            case .single:
                let options = ClaudeTurnOptions(effort: configuration.singleEffort)
                try await relay(answer(registry, options: options), system: system, turns: turns, output: output)
            case .parallel:
                try await deliberate(system: system, turns: turns, registry: registry, output: output)
            }
            group.cancelAll()
        }
    }

    /// A reply that the user sees: the conversation's model with the full tools.
    private func answer(_ registry: ToolRegistry, options: ClaudeTurnOptions) -> ClaudeProvider {
        ClaudeProvider(
            configuration: claude,
            transport: transport,
            tools: ToolRunner(registry: registry),
            options: options,
            toolContext: toolContext
        )
    }

    private enum AnswerSignal: Sendable {
        case ended
        case deadline
        case timerStopped
    }

    /// Streams an answer the user sees. If the call has produced no event by the deadline, it is
    /// cancelled and this throws `URLError(.timedOut)`; after its first event it may take as long
    /// as it needs.
    private func relay(_ provider: ClaudeProvider, system: String, turns: [ChatTurn], output: Output) async throws {
        let deadline = configuration.deadline
        guard deadline > .zero else {
            for try await event in provider.streamEvents(system: system, turns: turns) {
                output.pass(event)
            }
            return
        }
        let contact = ServerContact()
        let sleep = hooks.sleep
        try await withThrowingTaskGroup(of: AnswerSignal.self) { group in
            group.addTask {
                for try await event in provider.streamEvents(system: system, turns: turns) {
                    // Nothing more is passed on once the deadline has given up on the call.
                    guard contact.hear() else { break }
                    output.pass(event)
                }
                return .ended
            }
            group.addTask {
                do {
                    try await sleep(deadline)
                } catch {
                    return .timerStopped
                }
                return .deadline
            }
            while let signal = try await group.next() {
                switch signal {
                case .ended:
                    // Stops the timer.
                    group.cancelAll()
                    return
                case .deadline:
                    // The server spoke in time; let the call run.
                    guard contact.expireIfSilent() else { continue }
                    // Leaving the group cancels the call and waits for it to end.
                    throw URLError(.timedOut)
                case .timerStopped:
                    continue
                }
            }
        }
    }

    private func deliberate(system: String, turns: [ChatTurn], registry: ToolRegistry, output: Output) async throws {
        let round = await runWorkers(system: system, turns: turns, registry: registry, output: output)
        output.endWorkers()
        try Task.checkCancellation()

        var notes: [(id: String, text: String)] = []
        for (brief, outcome) in zip(configuration.workers, round.outcomes) {
            if let text = outcome.notes {
                notes.append((id: brief.id, text: text))
            }
        }
        guard !notes.isEmpty else {
            if round.deadlinePassed, !round.outcomes.contains(where: \.reachedServer) {
                throw URLError(.timedOut)
            }
            try await relay(answer(registry, options: ClaudeTurnOptions()), system: system, turns: turns, output: output)
            return
        }
        let options = ClaudeTurnOptions(
            instruction: configuration.mergerInstruction,
            userData: AnalystNotes.block(notes),
            maxTokens: configuration.mergerMaxTokens
        )
        try await relay(answer(registry, options: options), system: system, turns: turns, output: output)
    }

    // MARK: - Workers

    private struct WorkerOutcome: Sendable {
        /// The worker's notes; nil when it was dropped or wrote nothing.
        var notes: String?
        /// Whether anything came back from the server: an event, or an error other than a
        /// connectivity failure.
        var reachedServer: Bool

        static let none = WorkerOutcome(notes: nil, reachedServer: false)
    }

    private enum WorkerSignal: Sendable {
        case worker(Int, WorkerOutcome)
        case deadline
        case timerStopped
    }

    /// Runs every worker until all have ended or the deadline passes, then cuts off the rest.
    /// The outcomes follow `configuration.workers`' order.
    private func runWorkers(
        system: String,
        turns: [ChatTurn],
        registry: ToolRegistry,
        output: Output
    ) async -> (outcomes: [WorkerOutcome], deadlinePassed: Bool) {
        let workers = configuration.workers
        let deadline = configuration.deadline
        let sleep = hooks.sleep
        return await withTaskGroup(of: WorkerSignal.self) { group in
            for (index, brief) in workers.enumerated() {
                group.addTask {
                    .worker(index, await runWorker(brief, system: system, turns: turns, registry: registry, output: output))
                }
            }
            if deadline > .zero {
                group.addTask {
                    do {
                        try await sleep(deadline)
                    } catch {
                        return .timerStopped
                    }
                    return .deadline
                }
            }

            var outcomes = [WorkerOutcome](repeating: .none, count: workers.count)
            var pending = workers.count
            var deadlinePassed = false
            while pending > 0, let signal = await group.next() {
                switch signal {
                case .worker(let index, let outcome):
                    outcomes[index] = outcome
                    pending -= 1
                case .deadline:
                    // Cut off the workers still running; each returns what it has written so far.
                    deadlinePassed = true
                    group.cancelAll()
                case .timerStopped:
                    break
                }
            }
            // Stops the deadline timer once every worker has ended.
            group.cancelAll()
            return (outcomes, deadlinePassed)
        }
    }

    private func runWorker(
        _ brief: WorkerBrief,
        system: String,
        turns: [ChatTurn],
        registry: ToolRegistry,
        output: Output
    ) async -> WorkerOutcome {
        var workerClaude = claude
        if let model = configuration.workerModel?.trimmed, !model.isEmpty {
            workerClaude.model = model
        }
        let allowed = registry.readOnly()
        let tools: any ToolExecutor = brief.usesTools ? ToolRunner(registry: allowed) : DeclaredOnlyTools(registry: allowed)
        let options = ClaudeTurnOptions(
            instruction: brief.instruction,
            effort: brief.effort,
            maxTokens: configuration.workerMaxTokens
        )
        let provider = ClaudeProvider(
            configuration: workerClaude,
            transport: transport,
            tools: tools,
            options: options,
            toolContext: toolContext
        )

        var text = ""
        var reachedServer = false
        var dropped = false
        do {
            for try await event in provider.streamEvents(system: system, turns: turns) {
                reachedServer = true
                switch event {
                case .reply(.text(let chunk)):
                    text += chunk
                case .reply(.finished(.refused)):
                    dropped = true
                case .reply(.activity(let activity)):
                    if brief.forwardsActivity { output.worker(brief.id, activity: activity) }
                case .reply(.finished), .cue, .toolRound, .routed, .progress:
                    break
                }
            }
        } catch {
            // A failed worker is dropped. One cut off by the deadline (or by the reply being
            // cancelled) keeps what it wrote.
            if !Task.isCancelled {
                dropped = true
                if !error.isConnectivityFailure { reachedServer = true }
            }
        }
        if brief.forwardsActivity { output.worker(brief.id, activity: nil) }

        let notes = text.trimmed
        let outcome = WorkerOutcome(notes: dropped || notes.isEmpty ? nil : notes, reachedServer: reachedServer)
        hooks.workerEnded(brief.id)
        return outcome
    }

    // MARK: - Output

    /// The reply's events as the consumer sees them, with the state that decides the cues and the
    /// activity line. Workers and timers report here from their own tasks, so it is locked.
    final class Output: @unchecked Sendable {
        typealias Continuation = AsyncThrowingStream<AssistantEvent, Error>.Continuation

        private let lock = NSLock()
        private let continuation: Continuation
        private let uptime: @Sendable () -> TimeInterval
        /// Text has been passed on, or the answer finished: no cue may follow.
        private var answered = false
        /// The last activity event set a line rather than clearing it.
        private var activityShown = false
        /// The worker activity on screen and the worker it came from, if the line shows one.
        private var workerLine: (id: String, text: String)?
        /// When the last new worker line was shown.
        private var workerLineTime: TimeInterval?
        private var workersEnded = false

        init(continuation: Continuation, uptime: @escaping @Sendable () -> TimeInterval) {
            self.continuation = continuation
            self.uptime = uptime
        }

        func begin() {
            lock.lock()
            defer { lock.unlock() }
            continuation.yield(.cue(.deepThinking))
            setActivity(Deliberation.thinkingActivity)
        }

        /// Passes on an event of the answer, clearing our activity line before its first text.
        func pass(_ event: AssistantEvent) {
            lock.lock()
            defer { lock.unlock() }
            switch event {
            case .reply(.text(let text)) where !text.isEmpty:
                if activityShown { setActivity(nil) }
                workerLine = nil
                answered = true
            case .reply(.finished):
                answered = true
            case .reply(.activity(let activity)):
                activityShown = activity != nil
                workerLine = nil
            default:
                break
            }
            continuation.yield(event)
        }

        /// `.cue(.stillThinking)`, unless the answer has begun.
        func stillThinking() {
            lock.lock()
            defer { lock.unlock() }
            guard !answered else { return }
            continuation.yield(.cue(.stillThinking))
        }

        /// The activity of the forwarding worker `id`. A new line shows at most every
        /// `activityInterval` seconds, but one that extends the worker's own line on screen (a
        /// search gaining its query) shows at once. When the worker's line goes away (nil, also
        /// sent when the worker ends), "Thinking it through" returns.
        func worker(_ id: String, activity: String?) {
            lock.lock()
            defer { lock.unlock() }
            guard !workersEnded, !answered else { return }
            let ownLine = workerLine?.id == id ? workerLine?.text : nil
            guard let activity, !activity.isEmpty else {
                if ownLine != nil { setActivity(Deliberation.thinkingActivity) }
                return
            }
            guard activity != workerLine?.text else { return }
            let extends = ownLine.map { activity.hasPrefix($0) } ?? false
            if !extends {
                let now = uptime()
                if let last = workerLineTime, now - last < Deliberation.activityInterval { return }
                workerLineTime = now
            }
            setActivity(activity)
            workerLine = (id: id, text: activity)
        }

        /// The workers are done: later worker activity is dropped, and a worker line still on
        /// screen gives way to "Thinking it through" while the answer starts.
        func endWorkers() {
            lock.lock()
            defer { lock.unlock() }
            workersEnded = true
            if workerLine != nil, !answered { setActivity(Deliberation.thinkingActivity) }
        }

        /// Call with the lock held.
        private func setActivity(_ activity: String?) {
            continuation.yield(.reply(.activity(activity)))
            activityShown = activity != nil
            workerLine = nil
        }
    }

    /// Whether an answer call has heard from the server. The call's events and the deadline settle
    /// under one lock, so an event is either passed on or dropped for good.
    private final class ServerContact: @unchecked Sendable {
        private let lock = NSLock()
        private var heard = false
        private var expired = false

        /// Records an event of the call. Returns false once the deadline has given up on it.
        func hear() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            guard !expired else { return false }
            heard = true
            return true
        }

        /// Gives up on the call if it hasn't produced an event yet. Returns whether it gave up.
        func expireIfSilent() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            guard !heard else { return false }
            expired = true
            return true
        }
    }
}

/// Declares a registry's tools without running any: every call gets
/// `ToolRegistry.unavailableMessage`. A worker without tools thereby keeps the same tool list,
/// and so the same cached prefix, as the others.
private struct DeclaredOnlyTools: ToolExecutor {
    let registry: ToolRegistry

    var definitions: [ToolDefinition] {
        registry.definitions
    }

    func presentation(for name: String) -> ToolPresentation {
        registry.presentation(for: name)
    }

    func run(_ calls: [PendingToolCall], context: ToolContext) async -> ToolRound {
        ToolRound(calls: calls.map { ToolCallRecord(call: $0, output: .error(ToolRegistry.unavailableMessage)) })
    }
}
