import AssistantKit
import Foundation

/// Answers a request on device with tools (plan §4.9, WP31): it runs `InferenceEngine.reply`,
/// runs the tool calls the model makes through a `ToolExecutor`, hands their results back with
/// `InferenceEngine.continueReply(after:)`, and repeats until the model answers, for at most
/// `maxRounds` rounds.
///
/// Events, in order:
/// - the engine's `.progress` marks, passed on unchanged;
/// - visible text as `.reply(.text)`. Leading whitespace is held back, so the first text event
///   is the first visible character;
/// - for each round: `.cue` (the first runnable call's cue, if it has one), then
///   `.reply(.activity)`, then the round's `.toolRound` once every call has a result; the next
///   text clears the activity with `.reply(.activity(nil))`;
/// - `.reply(.finished)` last: `.completed` after a stop, `.truncated` at the length limit, and
///   `.other("tool_limit")` when the model still asks for tools after `maxRounds` rounds.
///
/// A reply the engine cancels (the stream was terminated, or the GPU stopped being allowed)
/// throws `CancellationError`.
///
/// **`handoff_to_cloud`** is intercepted and never run. While the reply is uncommitted (no
/// visible text and no tool round yet), `handoffAvailable` and `isOnline()`:
/// - as soon as the model has written the call's name, generation stops and the stream throws
///   `ReplyHandoff(reason: "early")`, which saves decoding the call's reason;
/// - a parsed call that slipped through throws `ReplyHandoff(reason:)` with the model's reason.
///
/// Otherwise the call gets a tool result instead: the cloud isn't available, it is unreachable
/// (offline), or "Handoff unavailable now; finish your answer." once the reply is committed.
/// The orchestrator falls back to the cloud on `ReplyHandoff` only before commit, which is why
/// a reply that showed text or ran tools never throws it.
///
/// **Plausibility guard.** Before a round runs, `profile.callGuard` checks each call against the
/// user's words (`ToolCallGuard.userText(of:)`): an on-device model sometimes reaches for the wrong
/// tool (asked how long a timer had left, Woof started a one-second timer). A call the guard objects
/// to never runs: the request is handed off when it still can be, else the call gets an error
/// result that tells the model to ask the user.
///
/// **Claim guard.** On an action request (`.deviceAction` for the user's words, from
/// `profile.intents`) with tools offered, Woof sometimes answered in words that the action was done ("已经为
/// 你设置好了明天的提醒") without calling a tool. For such a request:
/// - the first reply asks the engine for `.requiredIfLikely(likelyToolCallFloor)`;
/// - its visible text is held back until its first sentence ends (or the reply does), while no
///   tool call has started; a hold that lasts `holdActivityDelay` shows `holdActivity`;
/// - held text that claims an action (`profile.claims`) is dropped with its generation,
///   and the request is asked again with `.required`, once per run (`Report.claimRetries`);
/// - a retry that writes no call with readable input throws `ReplyHandoff(reason:
///   claimHandoffReason)` when it can, else ends with `SuccessClaim.actionFailedMessage`, which
///   the engine didn't write, so the next request rebuilds (`diverged`).
///
/// The guard lasts while every tool round has only gathered context for the action
/// (`profile.contextLookups`, such as the time) and no side-effect call
/// (`ToolExecutor.effect(of:)`) has succeeded: Woof once looked up the time for "next Monday", then said the meeting was in the
/// calendar. After any other lookup the model is answering a question ("What's on my calendar
/// tomorrow?"), whose true answer can read as a claim ("你明天已经有两个会议"), so nothing more is
/// forced or held: a forced call there could create an event nobody asked for. While a round is left, the continuation after
/// such a round opens with `.requiredIfLikely(likelyToolCallFloor)` and, if the run hasn't
/// retried yet, its text is held the same way: a claim is dropped and the continuation decoded
/// again from the end of the round's results with `.required`
/// (`InferenceEngine.retryContinuation`), so the cache and the reply hold only what is shown.
/// Without a retry left the text goes out as written: an answer from a lookup ("You have a
/// meeting scheduled at 10") can read as a claim, and mustn't end in `actionFailedMessage`.
/// Once a side effect has succeeded, the model is confirming a real action: nothing is forced
/// or held.
///
/// A request with `ToolChoice.native` is left to the model: no claim guard (the benchmark's
/// "Compare tool modes").
///
/// The engine's cache stays exact whatever happens (plan §4.4): an aborted reply is recorded as
/// unanswered, so the next request rewinds past it (`replaceLastUserTurn` or `diverged`).
public struct LocalToolLoop: Sendable {
    /// The reason of the `ReplyHandoff` thrown as soon as the model names `handoff_to_cloud`.
    public static let earlyHandoffReason = "early"
    /// The reason of the `ReplyHandoff` thrown when a reply claimed an action and its retry
    /// still wrote no tool call.
    public static let claimHandoffReason = "claimed an action without a tool call"
    /// A device action's first reply, and its continuations after lookups only, open with a tool
    /// call when the model gives one at least this probability at the first token: a call it
    /// considered at all.
    public static let likelyToolCallFloor = 0.05
    /// The activity shown while held-back text has lasted `holdActivityDelay` seconds.
    public static let holdActivity = "…"
    static let holdActivityDelay: TimeInterval = 1
    /// Where held-back text's first sentence ends.
    static let sentenceEnds: Set<Character> = Set("。！？.!?\n")
    /// Letters and digits held-back text needs before a sentence end releases it, so an opener
    /// such as "OK!" or "好的！" doesn't let the claim after it ("好的！已经为你设置好了…") through.
    static let shortOpenerLength = 10

    /// Whether held-back text has ended a first sentence that isn't just a short opener.
    public static func endsFirstSentence(_ text: Substring) -> Bool {
        guard let end = text.lastIndex(where: { sentenceEnds.contains($0) }) else { return false }
        return text[..<end].filter { $0.isLetter || $0.isNumber }.count >= shortOpenerLength
    }
    /// The reason `.reply(.finished(.other(_)))` carries when `maxRounds` is used up.
    public static let toolLimitReason = "tool_limit"
    /// `handoff_to_cloud`'s result while offline.
    public static let offlineMessage = "The cloud assistant is unreachable (offline). Answer as best you can and say so if you can't."
    /// `handoff_to_cloud`'s result once the reply has shown text or run tools.
    public static let lateHandoffMessage = "Handoff unavailable now; finish your answer."
    /// `handoff_to_cloud`'s result when this loop may not hand off at all (no cloud assistant).
    public static let handoffUnavailableMessage = "The cloud assistant isn't available here. Answer as best you can and say so if you can't."
    /// The result of a call that returned no record (an executor that broke its contract).
    static let missingResultMessage = "The tool returned no result."

    public let engine: InferenceEngine
    public let executor: any ToolExecutor
    /// Whether `handoff_to_cloud` may hand the request to the cloud (a cloud key exists).
    public let handoffAvailable: Bool
    /// Most tool rounds one reply runs.
    public let maxRounds: Int
    /// The app's words, claims, context lookups and call guard (`AssistantProfile`).
    public let profile: AssistantProfile
    private let isOnline: @Sendable () -> Bool
    private let toolContext: @Sendable () -> ToolContext

    /// - Parameters:
    ///   - executor: runs the tools; pass one with lenient input (`ToolRunner(..., lenientInput: true)`)
    ///     for on-device models, which write numbers and booleans as strings.
    ///   - handoffAvailable: whether `handoff_to_cloud` may hand off (a cloud key exists).
    ///   - isOnline: read when a handoff is decided.
    ///   - toolContext: read once per round; carries the commit gate side effects wait for.
    ///   - maxRounds: most tool rounds per reply.
    ///   - profile: the app's profile, the same value its router and prompts use.
    public init(
        engine: InferenceEngine,
        executor: any ToolExecutor,
        handoffAvailable: Bool,
        isOnline: @escaping @Sendable () -> Bool,
        toolContext: @escaping @Sendable () -> ToolContext,
        profile: AssistantProfile,
        maxRounds: Int = 3
    ) {
        self.engine = engine
        self.executor = executor
        self.handoffAvailable = handoffAvailable
        self.isOnline = isOnline
        self.toolContext = toolContext
        self.maxRounds = max(0, maxRounds)
        self.profile = profile
    }

    /// Answers `request`. Ending or cancelling the stream stops the engine at its next step.
    /// `report`, when given, is called once with what the run did, before the stream ends.
    public func run(_ request: EngineRequest, report: (@Sendable (Report) -> Void)? = nil) -> AsyncThrowingStream<AssistantEvent, Error> {
        let loop = self
        return AsyncThrowingStream { continuation in
            let task = Task {
                var state = RunState(start: Date())
                do {
                    try await loop.drive(request, state: &state) { continuation.yield($0) }
                    report?(state.report)
                    continuation.finish()
                } catch {
                    if let handoff = error as? ReplyHandoff {
                        state.handoffReason = handoff.reason
                    }
                    report?(state.report)
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: Report

    /// What one run did, for telemetry.
    public struct Report: Sendable {
        /// Each engine generation's statistics, in order: the reply, then one per tool round. A
        /// reply dropped for claiming an action is there only if it had finished; its retry follows.
        public var generations: [LocalGenerationStats]
        /// Tool rounds run.
        public var rounds: Int
        /// From the start of the run to its first visible text; nil if it showed none.
        public var timeToFirstText: TimeInterval?
        /// The reason of the `ReplyHandoff` the run threw, if it did.
        public var handoffReason: String?
        /// Times the reply was asked again because it claimed an action without a tool call.
        public var claimRetries: Int

        public init(
            generations: [LocalGenerationStats] = [], rounds: Int = 0, timeToFirstText: TimeInterval? = nil, handoffReason: String? = nil,
            claimRetries: Int = 0
        ) {
            self.generations = generations
            self.rounds = rounds
            self.timeToFirstText = timeToFirstText
            self.handoffReason = handoffReason
            self.claimRetries = claimRetries
        }

        /// The generations combined into one reply's statistics (see `LocalToolLoop.combine`),
        /// with `claimRetries`.
        public var stats: LocalGenerationStats {
            var stats = LocalToolLoop.combine(generations, timeToFirstText: timeToFirstText)
            stats.claimRetries = claimRetries
            return stats
        }
    }

    /// One reply's statistics from its generations: token counts, times, drafts and prefilled
    /// tokens add up; peak memory is the largest; confidence is token-weighted (its p10 is the
    /// lowest p10, a lower bound); speculation adds up per source; a forced tool call in any
    /// generation counts. The plan, phases, reused tokens, tool-call probability and engine are
    /// the first generation's, which answered the request.
    /// `timeToFirstText` replaces the first generation's, which only covers that generation.
    public static func combine(_ generations: [LocalGenerationStats], timeToFirstText: TimeInterval?) -> LocalGenerationStats {
        guard var combined = generations.first else {
            return LocalGenerationStats(timeToFirstText: timeToFirstText, engine: "alvin")
        }
        combined.timeToFirstText = timeToFirstText
        for next in generations.dropFirst() {
            combined.promptTokens += next.promptTokens
            combined.promptTime += next.promptTime
            combined.generatedTokens += next.generatedTokens
            combined.generateTime += next.generateTime
            combined.draftTokens = sum(combined.draftTokens, next.draftTokens)
            combined.acceptedDraftTokens = sum(combined.acceptedDraftTokens, next.acceptedDraftTokens)
            combined.peakMemoryBytes = largest(combined.peakMemoryBytes, next.peakMemoryBytes)
            combined.prefilledTokens = sum(combined.prefilledTokens, next.prefilledTokens)
            combined.speculation = merge(combined.speculation, next.speculation)
            combined.confidence = merge(combined.confidence, next.confidence)
            combined.forcedToolCall = combined.forcedToolCall || next.forcedToolCall
        }
        return combined
    }

    private static func sum(_ a: Int?, _ b: Int?) -> Int? {
        guard let a else { return b }
        guard let b else { return a }
        return a + b
    }

    private static func largest(_ a: Int?, _ b: Int?) -> Int? {
        guard let a else { return b }
        guard let b else { return a }
        return max(a, b)
    }

    private static func merge(_ a: SpeculationStats?, _ b: SpeculationStats?) -> SpeculationStats? {
        guard let a else { return b }
        guard let b else { return a }
        return SpeculationStats(
            rounds: a.rounds + b.rounds,
            plainTokens: a.plainTokens + b.plainTokens,
            drafted: a.drafted.merging(b.drafted, uniquingKeysWith: +),
            accepted: a.accepted.merging(b.accepted, uniquingKeysWith: +),
            tokensPerRound: a.tokensPerRound.merging(b.tokensPerRound, uniquingKeysWith: +))
    }

    private static func merge(_ a: TokenConfidence?, _ b: TokenConfidence?) -> TokenConfidence? {
        guard let a, a.tokens > 0 else { return b ?? a }
        guard let b, b.tokens > 0 else { return a }
        let tokens = a.tokens + b.tokens
        let mean = (a.meanTop1 * Double(a.tokens) + b.meanTop1 * Double(b.tokens)) / Double(tokens)
        return TokenConfidence(meanTop1: mean, p10Top1: min(a.p10Top1, b.p10Top1), tokens: tokens)
    }

    // MARK: Plausibility guard

    /// The result of a call the guard objected to, when the request can't be handed off.
    static func guardMessage(_ objection: String) -> String {
        "Not done: \(objection). Don't guess: ask the user, or say you can't do that here."
    }

    // MARK: Running

    /// What the loop knows about the reply so far.
    struct RunState {
        let start: Date
        var generations: [LocalGenerationStats] = []
        var rounds = 0
        /// Visible text or a tool round has gone out: the orchestrator can no longer fall back.
        var committed = false
        var shownText = false
        var activityShown = false
        var firstText: Date?
        var handoffReason: String?
        var claimRetries = 0
        /// A side-effect call has succeeded: the claim guard is over.
        var acted = false
        /// Every round so far only gathered context for the action (`profile.contextLookups`).
        var contextOnly = true
        /// Times `show` emitted text.
        var shownTexts = 0
        /// Visible text is held back (a device action's reply before its first sentence end).
        var holding = false
        var held = ""
        var heldSince: Date?

        init(start: Date) {
            self.start = start
        }

        var report: Report {
            Report(
                generations: generations, rounds: rounds, timeToFirstText: firstText.map { $0.timeIntervalSince(start) },
                handoffReason: handoffReason, claimRetries: claimRetries)
        }

        /// Ends a hold: the held text, if any, goes out (see `show`).
        mutating func release(emit: (AssistantEvent) -> Void) {
            guard holding else { return }
            holding = false
            if !held.isEmpty {
                show(held, emit: emit)
            }
            held = ""
        }

        /// Emits visible text: leading whitespace is held back (dropped) until the first visible
        /// character, and a shown activity is cleared first.
        mutating func show(_ text: String, emit: (AssistantEvent) -> Void) {
            var text = text
            if !shownText {
                text = String(text.drop(while: { $0.isWhitespace }))
                guard !text.isEmpty else { return }
                shownText = true
                committed = true
                firstText = Date()
            }
            if activityShown {
                activityShown = false
                emit(.reply(.activity(nil)))
            }
            shownTexts += 1
            emit(.reply(.text(text)))
        }
    }

    /// How one engine generation ended.
    enum Generation {
        case finished(EngineFinish.Reason)
        case toolCalls([PendingToolCall])
        /// The model named `handoff_to_cloud` while a handoff was possible.
        case handoff(String)
        /// Held-back text claimed an action before a tool call of its own; the generation was dropped.
        case claimed
    }

    func drive(_ request: EngineRequest, state: inout RunState, emit: (AssistantEvent) -> Void) async throws {
        let userText = ToolCallGuard.userText(of: request.turns)
        // An action with tools: the claim guard (see the type's documentation). Unlike the router,
        // which reads only the newest user turn, this also reads the turn an assistant question
        // was answering, so "Ten minutes" after "How long should the timer run?" stays an action.
        var first = request
        let guarded = !request.tools.isEmpty && request.toolChoice != .native && profile.intents.classify(userText).contains(.deviceAction)
        if guarded {
            if first.toolChoice == .auto {
                first.toolChoice = .requiredIfLikely(Self.likelyToolCallFloor)
            }
            state.holding = true
        }
        var generation = try await consume(engine.reply(first), state: &state, emit: emit)
        while true {
            switch generation {
            case .claimed:
                // Once: a retry never ends `.claimed`.
                generation = try await retryClaim(request, userText: userText, state: &state, emit: emit)
            case .handoff(let reason):
                // The engine stream is gone, so the engine stops at its next step; wait for it,
                // so the cache and its record of the reply are settled before anyone else asks.
                await engine.waitUntilIdle()
                throw ReplyHandoff(reason: reason)
            case .finished(let reason):
                switch reason {
                case .stop, .toolCalls:
                    emit(.reply(.finished(.completed)))
                case .length:
                    emit(.reply(.finished(.truncated)))
                case .cancelled:
                    throw CancellationError()
                }
                return
            case .toolCalls(let calls):
                guard !calls.isEmpty else {
                    emit(.reply(.finished(.other("tool_use"))))
                    return
                }
                guard state.rounds < maxRounds else {
                    emit(.reply(.finished(.other(Self.toolLimitReason))))
                    return
                }
                let round = try await runRound(calls, userText: userText, state: &state, emit: emit)
                state.rounds += 1
                state.committed = true
                state.acted = state.acted || round.calls.contains { !$0.isError && executor.effect(of: $0.name) == .sideEffect }
                state.contextOnly = state.contextOnly && round.calls.allSatisfy { profile.contextLookups.contains($0.name) }
                emit(.toolRound(round))
                try Task.checkCancellation()
                // After lookups only, a call is still due (see the type's documentation); a call
                // past `maxRounds` couldn't run, so none is forced or retried then.
                var choice = ToolChoice.native
                if guarded, !state.acted, state.contextOnly, state.rounds < maxRounds {
                    choice = .requiredIfLikely(Self.likelyToolCallFloor)
                    state.holding = state.claimRetries == 0
                    state.heldSince = nil
                }
                generation = try await consume(engine.continueReply(after: round, toolChoice: choice), state: &state, emit: emit)
            }
        }
    }

    /// Asks again, with a required tool call, for a generation whose held-back text claimed an
    /// action: the reply, with the same turns, or after a tool round the continuation, from the
    /// end of the round's results. Returns the retry's generation when it wrote a call with
    /// readable input, handed off, showed text of its own or was cancelled. Otherwise the request
    /// is handed off when it can be (throws), else the reply ends with
    /// `SuccessClaim.actionFailedMessage` (as `.finished(.stop)`).
    private func retryClaim(
        _ request: EngineRequest, userText: String, state: inout RunState, emit: (AssistantEvent) -> Void
    ) async throws -> Generation {
        try Task.checkCancellation()
        state.claimRetries += 1
        state.held = ""
        state.heldSince = nil
        let shownTexts = state.shownTexts
        var retry = request
        retry.toolChoice = .required
        let events = state.rounds == 0 ? engine.reply(retry) : engine.retryContinuation(toolChoice: .required)
        let generation = try await consume(events, state: &state, emit: emit, retrying: true)
        switch generation {
        case .toolCalls(let calls) where calls.contains { $0.input != nil }:
            return generation
        case .handoff, .finished(.cancelled):
            return generation
        case .finished where state.shownTexts > shownTexts:
            return generation
        default:
            break
        }
        if handoffBlocker(state) == nil {
            await engine.waitUntilIdle()
            throw ReplyHandoff(reason: Self.claimHandoffReason)
        }
        state.show(SuccessClaim.actionFailedMessage(for: userText), emit: emit)
        return .finished(.stop)
    }

    /// Relays one engine generation. Returning early (an early handoff, a claim) drops the
    /// stream, which terminates it: the engine stops at its next step, with an exact ledger.
    /// A retry's (`retrying`) start marks aren't passed on: the first try reported them.
    private func consume(
        _ events: AsyncThrowingStream<EngineEvent, Error>, state: inout RunState, emit: (AssistantEvent) -> Void,
        retrying: Bool = false
    ) async throws -> Generation {
        var calls: [PendingToolCall] = []
        for try await event in events {
            switch event {
            case .progress(let mark):
                if case .toolCallStarted = mark {
                    state.release(emit: emit)
                } else if retrying {
                    continue
                }
                emit(.progress(mark))
                if case .toolCallStarted(let name?) = mark, name == HandoffTool.name, handoffBlocker(state) == nil {
                    return .handoff(Self.earlyHandoffReason)
                }
            case .text(let text):
                guard state.holding else {
                    state.show(text, emit: emit)
                    continue
                }
                state.held += text
                let visible = state.held.drop(while: { $0.isWhitespace })
                if Self.endsFirstSentence(visible) {
                    if profile.claims.claimsDone(String(visible)) {
                        return .claimed
                    }
                    state.release(emit: emit)
                } else if !visible.isEmpty, !state.activityShown {
                    let since = state.heldSince ?? Date()
                    state.heldSince = since
                    if Date().timeIntervalSince(since) >= Self.holdActivityDelay {
                        emit(.reply(.activity(Self.holdActivity)))
                        state.activityShown = true
                    }
                }
            case .toolCalls(let found):
                calls += found
            case .finished(let finish):
                state.generations.append(finish.stats)
                if state.holding, finish.reason != .cancelled {
                    if calls.isEmpty, profile.claims.claimsDone(state.held) {
                        return .claimed
                    }
                    state.release(emit: emit)
                }
                if finish.reason == .toolCalls {
                    return .toolCalls(calls)
                }
                return .finished(finish.reason)
            }
        }
        // Only a terminated stream ends without `.finished`.
        throw CancellationError()
    }

    /// Why `handoff_to_cloud` can't hand off now, as the result to give the call; nil when it can.
    private func handoffBlocker(_ state: RunState) -> String? {
        guard handoffAvailable else { return Self.handoffUnavailableMessage }
        guard isOnline() else { return Self.offlineMessage }
        guard !state.committed else { return Self.lateHandoffMessage }
        return nil
    }

    /// Runs one round: intercepts `handoff_to_cloud`, applies the guard, runs the rest through
    /// the executor and returns one record per call, in the calls' order. Throws `ReplyHandoff`
    /// when the round hands the request off (nothing has run then).
    private func runRound(
        _ calls: [PendingToolCall], userText: String, state: inout RunState, emit: (AssistantEvent) -> Void
    ) async throws -> ToolRound {
        var answers: [Int: ToolOutput] = [:]
        let handoffs = calls.indices.filter { calls[$0].name == HandoffTool.name }
        // Calls whose input didn't parse never run (the executor answers them with an error).
        let objections = calls.indices.compactMap { index -> (index: Int, why: String)? in
            guard calls[index].name != HandoffTool.name, calls[index].input != nil,
                  let why = profile.callGuard.check(calls[index], userText) else { return nil }
            return (index, why)
        }

        if !handoffs.isEmpty || !objections.isEmpty {
            if let blocker = handoffBlocker(state) {
                for index in handoffs {
                    answers[index] = .error(blocker)
                }
            } else {
                let reason: String
                if let index = handoffs.first {
                    reason = Self.handoffReason(of: calls[index])
                } else {
                    let objection = objections[0]
                    reason = "\(calls[objection.index].name) doesn't fit the request: \(objection.why)"
                }
                await engine.waitUntilIdle()
                throw ReplyHandoff(reason: reason)
            }
        }
        for objection in objections {
            answers[objection.index] = .error(Self.guardMessage(objection.why))
        }

        let runnable = calls.indices.filter { answers[$0] == nil }
        if let first = runnable.first {
            let presentation = executor.presentation(for: calls[first].name)
            if let cue = presentation.cue {
                emit(.cue(cue))
            }
            emit(.reply(.activity(presentation.activity)))
            state.activityShown = true
        }
        var executed: [ToolCallRecord] = []
        if !runnable.isEmpty {
            executed = await executor.run(runnable.map { calls[$0] }, context: toolContext()).calls
        }

        var records: [ToolCallRecord] = []
        for (index, call) in calls.enumerated() {
            if let answer = answers[index] {
                records.append(ToolCallRecord(call: call, output: answer))
            } else if let found = executed.firstIndex(where: { $0.id == call.id }) {
                records.append(executed.remove(at: found))
            } else {
                // Every call needs a result, or the model's next turn reads a broken history.
                records.append(ToolCallRecord(call: call, output: .error(Self.missingResultMessage)))
            }
        }
        return ToolRound(calls: records)
    }

    /// The reason a `handoff_to_cloud` call gives, or "handoff" when it gives none.
    static func handoffReason(of call: PendingToolCall) -> String {
        let reason = call.input?["reason"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return reason.isEmpty ? "handoff" : reason
    }
}
