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
/// **Plausibility guard.** Before a round runs, `callGuard` checks each call against the user's
/// words (`CallGuard.userText(of:)`): an on-device model sometimes reaches for the wrong tool
/// (asked how long a timer had left, Woof started a one-second timer). A call the guard objects
/// to never runs: the request is handed off when it still can be, else the call gets an error
/// result that tells the model to ask the user.
///
/// The engine's cache stays exact whatever happens (plan §4.4): an aborted reply is recorded as
/// unanswered, so the next request rewinds past it (`replaceLastUserTurn` or `diverged`).
public struct LocalToolLoop: Sendable {
    /// The reason of the `ReplyHandoff` thrown as soon as the model names `handoff_to_cloud`.
    public static let earlyHandoffReason = "early"
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
    public let callGuard: CallGuard
    private let isOnline: @Sendable () -> Bool
    private let toolContext: @Sendable () -> ToolContext

    /// - Parameters:
    ///   - executor: runs the tools; pass one with lenient input (`ToolRunner(..., lenientInput: true)`)
    ///     for on-device models, which write numbers and booleans as strings.
    ///   - handoffAvailable: whether `handoff_to_cloud` may hand off (a cloud key exists).
    ///   - isOnline: read when a handoff is decided.
    ///   - toolContext: read once per round; carries the commit gate side effects wait for.
    ///   - maxRounds: most tool rounds per reply.
    ///   - callGuard: checks calls against the user's words before they run.
    public init(
        engine: InferenceEngine,
        executor: any ToolExecutor,
        handoffAvailable: Bool,
        isOnline: @escaping @Sendable () -> Bool,
        toolContext: @escaping @Sendable () -> ToolContext,
        maxRounds: Int = 3,
        callGuard: CallGuard = .standard
    ) {
        self.engine = engine
        self.executor = executor
        self.handoffAvailable = handoffAvailable
        self.isOnline = isOnline
        self.toolContext = toolContext
        self.maxRounds = max(0, maxRounds)
        self.callGuard = callGuard
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
        /// Each engine generation's statistics, in order: the reply, then one per tool round.
        public var generations: [LocalGenerationStats]
        /// Tool rounds run.
        public var rounds: Int
        /// From the start of the run to its first visible text; nil if it showed none.
        public var timeToFirstText: TimeInterval?
        /// The reason of the `ReplyHandoff` the run threw, if it did.
        public var handoffReason: String?

        public init(generations: [LocalGenerationStats] = [], rounds: Int = 0, timeToFirstText: TimeInterval? = nil, handoffReason: String? = nil) {
            self.generations = generations
            self.rounds = rounds
            self.timeToFirstText = timeToFirstText
            self.handoffReason = handoffReason
        }

        /// The generations combined into one reply's statistics (see `LocalToolLoop.combine`).
        public var stats: LocalGenerationStats {
            LocalToolLoop.combine(generations, timeToFirstText: timeToFirstText)
        }
    }

    /// One reply's statistics from its generations: token counts, times, drafts and prefilled
    /// tokens add up; peak memory is the largest; confidence is token-weighted (its p10 is the
    /// lowest p10, a lower bound); speculation adds up per source. The plan, phases, reused
    /// tokens and engine are the first generation's, which answered the request.
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

    /// Checks a tool call against the user's words before it runs.
    public struct CallGuard: Sendable {
        /// Why `call` doesn't fit `userText` (`userText(of:)` of the request), or nil when it may run.
        public let check: @Sendable (_ call: PendingToolCall, _ userText: String) -> String?

        public init(_ check: @escaping @Sendable (_ call: PendingToolCall, _ userText: String) -> String?) {
            self.check = check
        }

        /// Lets every call run.
        public static let none = CallGuard { _, _ in nil }

        /// Stops the measured failure (plan §9): asked how long a timer had left, Woof started a
        /// one-second timer. A `set_timer` call runs when the user stated an amount the guard can
        /// read (`statesAnAmount`). Without one, it doesn't run when the user's words ask about a
        /// timer that is already running (`asksAboutRunningTimer`), or when `seconds` is under
        /// `shortestUnstatedSeconds`, a duration nobody wants without saying so (in any
        /// language).
        ///
        /// Any other call runs. Amounts are read only in English, Chinese and digits, and the
        /// app takes typed and spoken requests in every language speech recognition offers:
        /// objecting to every call without a readable amount would refuse "pon un temporizador
        /// de diez minutos" every time, and offline it could never be set at all.
        public static let standard = CallGuard { call, userText in
            switch call.name {
            case CallGuard.setTimerName:
                return CallGuard.timerObjection(seconds: CallGuard.seconds(of: call), userText: userText)
            default:
                return nil
            }
        }

        static let setTimerName = "set_timer"
        /// A timer shorter than this (in seconds) runs only when the user stated an amount.
        public static let shortestUnstatedSeconds = 5

        /// Why a `set_timer` call for `seconds` (nil when unreadable) doesn't fit `userText`.
        static func timerObjection(seconds: Int?, userText: String) -> String? {
            if statesAnAmount(userText) {
                return nil
            }
            if asksAboutRunningTimer(userText) {
                return "the user asked about a timer that is already running, and set_timer only starts a new one"
            }
            if let seconds, seconds < shortestUnstatedSeconds {
                return "the user didn't say how long the timer should run"
            }
            return nil
        }

        /// The call's `seconds` as a whole number, also when the model wrote it as a string
        /// ("600"), as on-device models do.
        static func seconds(of call: PendingToolCall) -> Int? {
            guard let value = call.input?["seconds"] else { return nil }
            if let whole = value.intValue {
                return whole
            }
            guard let text = value.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines) else { return nil }
            if let whole = Int(text) {
                return whole
            }
            guard let number = Double(text), number.isFinite, abs(number) < 9e15 else { return nil }
            return Int(number.rounded(.down))
        }

        /// Whether `text` asks about a timer that is already running: in English "left",
        /// "remaining", "how long", "how much"; in Chinese 剩 ("还剩", "剩下"), 多久, 多长时间,
        /// 多少时间 or 几 before a unit of time ("还有几分钟"). Other languages aren't recognised.
        public static func asksAboutRunningTimer(_ text: String) -> Bool {
            let words = text.lowercased().split { !$0.isLetter }.map(String.init)
            if words.contains(where: { runningTimerWords.contains($0) }) {
                return true
            }
            for (index, word) in words.enumerated().dropLast() {
                if word == "how", runningTimerFollowers.contains(words[index + 1]) {
                    return true
                }
            }
            if runningTimerChinese.contains(where: { text.contains($0) }) {
                return true
            }
            let characters = Array(text)
            for (index, character) in characters.enumerated() where character == "几" || character == "幾" {
                var next = index + 1
                while next < characters.count, chineseBetween.contains(characters[next]) {
                    next += 1
                }
                if next < characters.count, chineseTimeUnits.contains(characters[next]) {
                    return true
                }
            }
            return false
        }

        private static let runningTimerWords: Set<String> = ["left", "remaining", "remain", "remains", "elapsed"]
        /// Words after "how" that ask for a duration: "how long", "how much (time)".
        private static let runningTimerFollowers: Set<String> = ["long", "much"]
        private static let runningTimerChinese = ["剩", "多久", "多长", "多長", "多少时间", "多少時間"]

        /// The user's words a call should fit: the newest user turn's text and, when the reply
        /// before it asked the user something ("What should I call it?"), the user turn that
        /// reply answered, so an amount given there still counts.
        public static func userText(of turns: [ChatTurn]) -> String {
            guard let last = turns.lastIndex(where: { $0.role == .user }) else { return "" }
            var texts = [turns[last].text]
            if last >= 2, turns[last - 1].role == .assistant, turns[last - 2].role == .user,
               turns[last - 1].text.contains(where: { $0 == "?" || $0 == "？" }) {
                texts.insert(turns[last - 2].text, at: 0)
            }
            return texts.joined(separator: "\n")
        }

        /// Whether `text` states an amount: a decimal digit in any script ("10", "３"), a Chinese
        /// numeral before a unit of time ("二十分钟", "一个半小时"; not the "一" of "一下" or
        /// "一个计时器"), an English number word ("ten", "half", "couple"), or "a"/"an" before a
        /// unit of time ("an hour").
        public static func statesAnAmount(_ text: String) -> Bool {
            if text.unicodeScalars.contains(where: { $0.properties.numericType == .decimal }) || hasChineseDuration(text) {
                return true
            }
            let words = text.lowercased().split { !$0.isLetter }.map(String.init)
            for (index, word) in words.enumerated() {
                if numberWords.contains(word) {
                    return true
                }
                if word == "a" || word == "an", index + 1 < words.count, timeUnits.contains(words[index + 1]) {
                    return true
                }
            }
            return false
        }

        /// A Chinese numeral followed by a unit of time, possibly after more numerals or a
        /// measure word ("一个半小时", "十多分钟", "一百二十秒"). "几分钟" doesn't count: it is
        /// how a question about a running timer asks ("还剩几分钟").
        private static func hasChineseDuration(_ text: String) -> Bool {
            let characters = Array(text)
            for (index, character) in characters.enumerated() where chineseNumerals.contains(character) {
                var next = index + 1
                while next < characters.count, chineseNumerals.contains(characters[next]) || chineseBetween.contains(characters[next]) {
                    next += 1
                }
                if next < characters.count, chineseTimeUnits.contains(characters[next]) {
                    return true
                }
            }
            return false
        }

        private static let chineseNumerals: Set<Character> = Set("〇零一二两兩三四五六七八九十百千万萬半廿卅")
        /// Measure words and "more" that may sit between a numeral and its unit.
        private static let chineseBetween: Set<Character> = Set("个個多来來")
        /// Seconds, minutes, hours (小时/小時, 钟头/鐘頭), quarter hours.
        private static let chineseTimeUnits: Set<Character> = Set("秒分小钟鐘刻")

        private static let numberWords: Set<String> = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
            "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
            "half", "quarter", "couple", "few", "several", "dozen",
        ]

        private static let timeUnits: Set<String> = [
            "second", "sec", "minute", "min", "hour", "hr",
        ]
    }

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

        init(start: Date) {
            self.start = start
        }

        var report: Report {
            Report(
                generations: generations, rounds: rounds, timeToFirstText: firstText.map { $0.timeIntervalSince(start) },
                handoffReason: handoffReason)
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
            emit(.reply(.text(text)))
        }
    }

    /// How one engine generation ended.
    enum Generation {
        case finished(EngineFinish.Reason)
        case toolCalls([PendingToolCall])
        /// The model named `handoff_to_cloud` while a handoff was possible.
        case handoff(String)
    }

    func drive(_ request: EngineRequest, state: inout RunState, emit: (AssistantEvent) -> Void) async throws {
        let userText = CallGuard.userText(of: request.turns)
        var generation = try await consume(engine.reply(request), state: &state, emit: emit)
        while true {
            switch generation {
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
                emit(.toolRound(round))
                try Task.checkCancellation()
                generation = try await consume(engine.continueReply(after: round), state: &state, emit: emit)
            }
        }
    }

    /// Relays one engine generation. Returning early (an early handoff) drops the stream, which
    /// terminates it: the engine stops at its next step, with an exact ledger.
    private func consume(
        _ events: AsyncThrowingStream<EngineEvent, Error>, state: inout RunState, emit: (AssistantEvent) -> Void
    ) async throws -> Generation {
        var calls: [PendingToolCall] = []
        for try await event in events {
            switch event {
            case .progress(let mark):
                emit(.progress(mark))
                if case .toolCallStarted(let name?) = mark, name == HandoffTool.name, handoffBlocker(state) == nil {
                    return .handoff(Self.earlyHandoffReason)
                }
            case .text(let text):
                state.show(text, emit: emit)
            case .toolCalls(let found):
                calls += found
            case .finished(let finish):
                state.generations.append(finish.stats)
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
                  let why = callGuard.check(calls[index], userText) else { return nil }
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
