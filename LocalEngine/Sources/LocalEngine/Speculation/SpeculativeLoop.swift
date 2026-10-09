import AssistantKit
import Foundation
import MLX
import MLXLMCommon

/// The draft policy shared by the replies of one engine, so acceptance statistics and backoff
/// carry over from reply to reply. Engine queue only.
public final class SharedDraftPolicy {
    public var policy: DraftPolicy

    public init(_ policy: DraftPolicy) {
        self.policy = policy
    }

    public convenience init(curve: CostCurve, maxDraft: Int = 4) {
        var configuration = DraftPolicy.Configuration()
        configuration.maxDraft = maxDraft
        self.init(DraftPolicy(curve: curve, configuration: configuration))
    }
}

/// Speculative decoding with sample-match verification (plan §4.7, WP30 instruction 2).
///
/// **Plain mode** decodes like `DecodeLoop`, pipelined: the next token `y` is sampled lazily
/// and evaluated while the host handles the current one. After each synced token the drafters
/// are asked (in the order given: the exemplar only inside a tool call, then prompt lookup,
/// the corpus, extension drafters and the draft model) whether a round could pay.
///
/// **Round mode**, when the policy picks K > 0 drafts:
/// 1. drain the pipeline: sync the in-flight `y` (that step's work is used, not wasted);
/// 2. feed `[y] + drafts` in one forward with logits for every row and a rollback capture;
/// 3. sample every row with one `FastSampler` call and sync once;
/// 4. `AcceptanceRule.resolve`: the leading drafts that equal the target's own samples are
///    accepted, then the target's sample at the first mismatch (or after the last draft) is
///    emitted too;
/// 5. `session.commit` keeps `y` and the accepted drafts (exact rollback of the rest), the
///    drafters observe the round and the policy records it;
/// 6. the emitted bonus or correction token becomes the next `y`, not fed yet.
///
/// Every row is sampled exactly as plain decoding would sample it (with a seed, position-keyed
/// like `DecodeLoop`), and drafts never depend on the target's random draws, so the emitted
/// tokens have exactly the plain distribution at any temperature.
///
/// Drafts never contain special tokens (turn markers, role tokens, `<|endoftext|>`, think and
/// tool markup), except in the exemplar drafter's tool-call structure; a stop token such as
/// `<|im_end|>` may be drafted and ends the draft. A round never verifies more than
/// `remaining − 1` drafts.
///
/// `flush()` feeds a pending `y` (an emitted token, or the stop token the target sampled)
/// cache-only, so the cache holds everything emitted and, after a stop, the stop token, as
/// plain decoding leaves it (F4). Not thread-safe: engine queue only.
public final class SpeculativeLoop: TokenGenerator {
    public struct Options {
        /// `.toolsOnly` drafts only inside a tool call or for matches of at least 3 tokens.
        /// (`.off` is handled by the factory, which then uses `DecodeLoop`.)
        public var mode: LocalSpeculationMode
        /// Tests and the self-test: verify this many drafts (fewer when the proposal or the
        /// remaining length is shorter) whenever a drafter has a proposal, ignoring the policy.
        public var forcedDraftLength: Int?
        /// How often, in emitted tokens, the thermal state and Low Power Mode are read.
        public var deviceStateInterval: Int
        /// Reads the device state; nil reads `ProcessInfo` (nominal off Apple platforms).
        public var deviceState: (@Sendable () -> (thermal: ThermalLevel, lowPower: Bool))?

        public init(
            mode: LocalSpeculationMode = .automatic, forcedDraftLength: Int? = nil, deviceStateInterval: Int = 16,
            deviceState: (@Sendable () -> (thermal: ThermalLevel, lowPower: Bool))? = nil
        ) {
            self.mode = mode
            self.forcedDraftLength = forcedDraftLength
            self.deviceStateInterval = deviceStateInterval
            self.deviceState = deviceState
        }
    }

    /// One verified round.
    public struct Round: Equatable, Sendable {
        public let source: DraftSource
        /// Drafts verified (K).
        public let proposed: Int
        /// Leading drafts accepted (an accepted stop draft included).
        public let accepted: Int
        /// Tokens the round emitted.
        public let emitted: Int
        /// Rows kept in the cache, `y` included.
        public let kept: Int
        /// Whether the round ended the reply at a stop token.
        public let stopped: Bool

        /// Drafts discarded: rejected, or not looked at after the first rejection or an accepted
        /// stop draft.
        public var rejected: Int { proposed - accepted }
    }

    /// Acceptance at one draft depth over a reply's rounds.
    public struct DepthAcceptance: Equatable, Sendable {
        /// Rounds that verified a draft at this depth with every earlier draft accepted.
        public var observed = 0
        /// Of those, the rounds whose draft at this depth was accepted.
        public var accepted = 0

        public var rate: Double? { observed > 0 ? Double(accepted) / Double(observed) : nil }
    }

    /// The next token to feed.
    private enum Next {
        /// Sampled but not synced, emitted or fed (the plain pipeline's in-flight token).
        case lazy(token: MLXArray, top1: MLXArray)
        /// Synced, not fed: either emitted already, or a stop token the target sampled.
        case known(token: Int, emitted: Bool)
        case none
    }

    private struct Choice {
        let drafter: any Drafter
        let proposal: DraftProposal
        let k: Int
        let expected: Double
    }

    public let context: GeneratorContext
    public let options: Options
    /// The drafters, in the order they are asked.
    public let drafters: [any Drafter]
    public let sharedPolicy: SharedDraftPolicy

    public private(set) var rounds: [Round] = []
    private var stats = SpeculationStats()
    private var next: Next = .none
    private var finished: EngineFinish.Reason?
    private var emittedCount = 0
    private var emittedTokens: [Int] = []
    private var nextDeviceCheck = 0
    private let canRollback: Bool
    private let wantsHidden: Bool
    /// Tokens no draft may contain (but the exemplar's).
    private let blockedTokens: Set<Int>
    private let toolCallStart: Int?
    private let toolCallEnd: Int?

    /// - Parameters:
    ///   - context: the reply; its `drafters` are not added (pass the ones to use in `drafters`).
    ///   - drafters: the drafters to ask, in order, already `reset` to the ledger.
    ///   - policy: the shared draft policy (updated by every round and plain token).
    public init(_ context: GeneratorContext, drafters: [any Drafter], policy: SharedDraftPolicy, options: Options = Options()) {
        self.context = context
        self.options = options
        self.drafters = drafters
        self.sharedPolicy = policy
        self.canRollback = context.session.target.supportsRollback
        self.wantsHidden = drafters.contains { $0.wantsHidden }
        self.blockedTokens = Self.blockedDraftTokens(renderer: context.renderer, stopTokens: context.stopTokens)
        self.toolCallStart = context.renderer.tokenID("<tool_call>")
        self.toolCallEnd = context.renderer.tokenID("</tool_call>")

        let position = context.session.cachedCount
        let (token, top1) = context.sampler.sample(context.firstLogits, positions: position ..< (position + 1))
        asyncEval(token, top1)
        next = .lazy(token: token, top1: top1)
        if context.maxTokens <= 0 {
            finished = .length
            next = .none
        }
        refreshDeviceState()
    }

    // MARK: Special tokens

    /// The chat format's control tokens found in the vocabulary (turn markers, think and tool
    /// markup), plus the stop tokens.
    public static func specialTokens(renderer: any ChatTemplateRendering, stopTokens: Set<Int>) -> Set<Int> {
        let names = [
            "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
            "<tool_response>", "</tool_response>",
        ]
        return Set(names.compactMap { renderer.tokenID($0) }).union(stopTokens)
    }

    /// Tokens no draft but the exemplar's may contain: the special tokens and the role tokens,
    /// except the stop tokens other than `<|endoftext|>` (a drafted `<|im_end|>` is verified
    /// like any token and, when accepted, ends the reply exactly where plain decoding would).
    public static func blockedDraftTokens(renderer: any ChatTemplateRendering, stopTokens: Set<Int>) -> Set<Int> {
        let endOfText = renderer.tokenID("<|endoftext|>")
        let draftableStops = stopTokens.filter { $0 != endOfText }
        return specialTokens(renderer: renderer, stopTokens: stopTokens)
            .union(PromptLookupDrafter.excludedTokens(renderer: renderer))
            .subtracting(draftableStops)
    }

    // MARK: TokenGenerator

    public var speculation: SpeculationStats? { stats }

    public func step() throws -> (emitted: [Int], finished: EngineFinish.Reason?) {
        if let finished {
            return ([], finished)
        }
        guard context.isAllowed() else {
            finish(.cancelled)
            return ([], .cancelled)
        }
        if emittedCount >= nextDeviceCheck {
            refreshDeviceState()
        }

        switch next {
        case .none:
            finish(.stop)
            return ([], .stop)
        case .lazy(let token, let top1):
            if canRollback && worthDraining() {
                // Drain: the in-flight token becomes the next round's `y`.
                let value = token.item(Int.self)
                context.confidence.record(top1.item(Float.self))
                next = .known(token: value, emitted: false)
                return try stepWithKnown()
            }
            return plainStep(token: token, top1: top1)
        case .known:
            return try stepWithKnown()
        }
    }

    public func flush() throws {
        guard case .known(let token, _) = next else { return }
        // An emitted token the cache doesn't hold yet, or the stop token the target sampled.
        context.session.feed([token], rows: .none)
        asyncEval(context.session.target.cache)
        next = .none
        if finished == nil {
            finish(.cancelled)
        }
    }

    // MARK: Statistics

    /// The acceptance at each draft depth (1-based) over the rounds so far.
    public var acceptanceByDepth: [Int: DepthAcceptance] {
        Self.acceptanceByDepth(rounds)
    }

    public static func acceptanceByDepth(_ rounds: [Round]) -> [Int: DepthAcceptance] {
        var depths: [Int: DepthAcceptance] = [:]
        for round in rounds where round.proposed > 0 {
            let observed = min(round.accepted + 1, round.proposed)
            for depth in 1 ... observed {
                depths[depth, default: DepthAcceptance()].observed += 1
                if depth <= round.accepted {
                    depths[depth, default: DepthAcceptance()].accepted += 1
                }
            }
        }
        return depths
    }

    // MARK: Plain steps

    /// The pipelined plain step of `DecodeLoop`: feed the lazy `y`, start sampling the next
    /// token, then sync `y` and emit it.
    private func plainStep(token: MLXArray, top1: MLXArray) -> (emitted: [Int], finished: EngineFinish.Reason?) {
        let session = context.session
        let position = session.cachedCount
        let result = session.feed(token, count: 1, rows: .last, hidden: wantsHidden)
        guard let logits = result.logits else {
            preconditionFailure("A `.last` forward returned no logits.")
        }
        let (following, followingTop1) = context.sampler.sample(logits, positions: (position + 1) ..< (position + 2))
        asyncEval(following, followingTop1)

        let value = token.item(Int.self)
        session.resolvePending([value])
        context.confidence.record(top1.item(Float.self))
        next = .lazy(token: following, top1: followingTop1)

        if context.stopTokens.contains(value) {
            observe(committed: [value], emitted: [], hidden: result.hidden)
            next = .none
            finish(.stop)
            return ([], .stop)
        }
        observe(committed: [value], emitted: [value], hidden: result.hidden)
        return emitPlain(value)
    }

    /// Counts a token decoded without drafts and emits it.
    private func emitPlain(_ value: Int) -> (emitted: [Int], finished: EngineFinish.Reason?) {
        emittedCount += 1
        emittedTokens.append(value)
        stats.recordPlain(1)
        sharedPolicy.policy.didDecodePlain(1)
        if emittedCount >= context.maxTokens {
            if case .lazy = next { next = .none }
            finish(.length)
            return ([value], .length)
        }
        return ([value], nil)
    }

    /// A step whose `y` is known: emit it if it wasn't, then run a round if one pays, else
    /// feed `y` and start the next token (plain).
    private func stepWithKnown() throws -> (emitted: [Int], finished: EngineFinish.Reason?) {
        guard case .known(let y, let alreadyEmitted) = next else {
            preconditionFailure("No known token to continue from.")
        }
        var output: [Int] = []
        if !alreadyEmitted {
            if context.stopTokens.contains(y) {
                // The reply ends here; `flush` feeds the stop token, as plain decoding would.
                finish(.stop)
                return ([], .stop)
            }
            next = .known(token: y, emitted: true)
            let (emitted, finished) = emitPlain(y)
            output = emitted
            if let finished {
                return (output, finished)
            }
        }

        let remaining = context.maxTokens - emittedCount
        if canRollback, remaining >= 2, let choice = chooseDraft(after: y, remaining: remaining) {
            let (emitted, finished) = try round(y: y, choice: choice)
            return (output + emitted, finished)
        }

        // Plain: feed `y` and start sampling the token after it.
        let session = context.session
        let position = session.cachedCount
        let result = session.feed([y], rows: .last, hidden: wantsHidden)
        guard let logits = result.logits else {
            preconditionFailure("A `.last` forward returned no logits.")
        }
        let (following, top1) = context.sampler.sample(logits, positions: (position + 1) ..< (position + 2))
        asyncEval(following, top1)
        next = .lazy(token: following, top1: top1)
        observe(committed: [y], emitted: [], hidden: result.hidden)
        return (output, nil)
    }

    // MARK: Rounds

    /// One verify round over `[y] + drafts` (y emitted, not fed).
    private func round(y: Int, choice: Choice) throws -> (emitted: [Int], finished: EngineFinish.Reason?) {
        let session = context.session
        let drafts = Array(choice.proposal.tokens.prefix(choice.k))
        let input = [y] + drafts
        let rows = input.count
        let position = session.cachedCount

        let started = ProcessInfo.processInfo.systemUptime
        let result = session.feed(input, rows: .all, capture: true, hidden: wantsHidden)
        guard let logits = result.logits else {
            preconditionFailure("An `.all` forward returned no logits.")
        }
        let (tokens, top1) = context.sampler.sample(logits, positions: (position + 1) ..< (position + 1 + rows))
        try checkedEval(tokens, top1)
        let verifySeconds = ProcessInfo.processInfo.systemUptime - started
        let sampled = tokens.asArray(Int32.self).map { Int($0) }
        let probabilities = top1.asArray(Float.self)

        let remaining = context.maxTokens - emittedCount
        let outcome = AcceptanceRule.resolve(drafts: drafts, sampled: sampled, stopTokens: context.stopTokens, remaining: remaining)
        session.commit(result.capture, keep: outcome.keep, of: rows)
        observe(committed: Array(input[..<outcome.keep]), emitted: outcome.emitted, hidden: result.hidden)

        sharedPolicy.policy.record(choice.proposal, proposed: drafts.count, round: outcome)
        stats.recordRound(
            source: choice.proposal.source.rawValue, drafted: drafts.count, accepted: outcome.acceptedDrafts,
            emitted: outcome.emitted.count)
        rounds.append(Round(
            source: choice.proposal.source, proposed: drafts.count, accepted: outcome.acceptedDrafts,
            emitted: outcome.emitted.count, kept: outcome.keep, stopped: outcome.stopped))
        let sampledRows = outcome.stoppedOnDraft ? outcome.acceptedDrafts : outcome.emitted.count + (outcome.stopped ? 1 : 0)
        context.confidence.record(contentsOf: probabilities.prefix(sampledRows))
        if let drafter = choice.drafter as? DraftModelDrafter, !drafter.calibrated {
            drafter.calibrate(verifySeconds: verifySeconds, rows: rows, curve: sharedPolicy.policy.curve)
        }

        emittedCount += outcome.emitted.count
        emittedTokens += outcome.emitted
        if outcome.stopped {
            if outcome.stoppedOnDraft {
                // The accepted stop draft is in the cache.
                next = .none
            } else {
                // The target sampled the stop token in place of a draft; `flush` feeds it.
                next = .known(token: sampled[outcome.acceptedDrafts], emitted: false)
            }
            finish(.stop)
            return (outcome.emitted, .stop)
        }
        guard let last = outcome.emitted.last else {
            preconditionFailure("A round without a stop emits at least one token.")
        }
        // Every emitted token but the last is in the cache; the last is the next `y`.
        next = .known(token: last, emitted: true)
        if emittedCount >= context.maxTokens {
            finish(.length)
            return (outcome.emitted, .length)
        }
        return (outcome.emitted, nil)
    }

    // MARK: Drafting decisions

    private var maxProposal: Int {
        options.forcedDraftLength ?? sharedPolicy.policy.maxDraft
    }

    /// Whether `drafter` may be asked at all now.
    private func mayAsk(_ drafter: any Drafter, insideToolCall: Bool) -> Bool {
        if drafter.source == .exemplar && !insideToolCall { return false }
        if options.forcedDraftLength == nil {
            let policy = sharedPolicy.policy
            guard policy.isSpeculationEnabled, !policy.isBackedOff(drafter.source) else { return false }
        }
        return true
    }

    /// A drafter whose proposals cost model work (a draft model, MTP): asked only when a round
    /// can follow, never for a look ahead.
    private func isExpensive(_ drafter: any Drafter) -> Bool {
        drafter.costPerToken > 0
    }

    /// Drafts to verify for `proposal` (0 = no round), with `remaining` tokens left to emit after
    /// `y`.
    private func draftLength(_ proposal: DraftProposal, drafter: any Drafter, remaining: Int) -> Int {
        if let forced = options.forcedDraftLength {
            return max(0, min(forced, proposal.tokens.count, remaining - 1))
        }
        return sharedPolicy.policy.draftLength(for: proposal, remaining: remaining, draftCostPerToken: drafter.costPerToken)
    }

    /// The most drafts an expensive drafter's best possible proposal could get.
    private func potentialDraftLength(_ drafter: any Drafter, remaining: Int) -> Int {
        let full = DraftProposal(tokens: Array(repeating: 0, count: max(1, maxProposal)), source: drafter.source, matchLength: 0)
        return draftLength(full, drafter: drafter, remaining: remaining)
    }

    /// With `y` still in flight: whether some drafter is likely to have a round worth draining
    /// the pipeline for. Cheap drafters are asked about the ledger (their first token guesses
    /// `y`); expensive ones only need to be allowed to draft.
    private func worthDraining() -> Bool {
        let remaining = context.maxTokens - emittedCount - 1
        guard remaining >= 2 else { return false }
        let insideToolCall = context.insideToolCall()
        let ledger = context.session.ledger
        for drafter in drafters where mayAsk(drafter, insideToolCall: insideToolCall) {
            if isExpensive(drafter) {
                if options.mode == .toolsOnly && !insideToolCall { continue }
                if potentialDraftLength(drafter, remaining: remaining) > 0 { return true }
                continue
            }
            guard let raw = drafter.propose(context: ledger[...], maxTokens: min(maxProposal, remaining - 1) + 1),
                  let proposal = sanitize(raw), proposal.tokens.count >= 2
            else { continue }
            let shifted = DraftProposal(tokens: Array(proposal.tokens.dropFirst()), source: proposal.source, matchLength: proposal.matchLength)
            if options.mode == .toolsOnly && !insideToolCall && shifted.matchLength < 3 { continue }
            if draftLength(shifted, drafter: drafter, remaining: remaining) > 0 { return true }
        }
        return false
    }

    /// The best proposal for the tokens after `y` (emitted, not fed) by
    /// `DraftPolicy.expectedTokens`, with its draft length; nil for a plain step.
    private func chooseDraft(after y: Int, remaining: Int) -> Choice? {
        let insideToolCall = isInsideToolCall(after: y)
        var tokens = context.session.ledger
        tokens.append(y)
        let policy = sharedPolicy.policy
        var best: Choice?
        for drafter in drafters where mayAsk(drafter, insideToolCall: insideToolCall) {
            var limit = min(maxProposal, remaining - 1)
            if isExpensive(drafter) {
                if options.mode == .toolsOnly && !insideToolCall { continue }
                let potential = potentialDraftLength(drafter, remaining: remaining)
                guard potential > 0 else { continue }
                if let best, best.expected >= policy.expectedTokens(source: drafter.source, matchLength: 0, k: potential) {
                    continue
                }
                // Drafting costs model work per token: ask for no more than a round could use.
                limit = min(limit, potential)
            }
            guard let raw = drafter.propose(context: tokens[...], maxTokens: limit),
                  let proposal = sanitize(raw)
            else { continue }
            if options.mode == .toolsOnly && !insideToolCall && proposal.matchLength < 3 { continue }
            let k = draftLength(proposal, drafter: drafter, remaining: remaining)
            guard k > 0 else { continue }
            let expected = policy.expectedTokens(source: proposal.source, matchLength: proposal.matchLength, k: k)
            if best == nil || expected > best!.expected {
                best = Choice(drafter: drafter, proposal: proposal, k: k, expected: expected)
            }
        }
        return best
    }

    /// Whether the text, `y` included, is inside a tool call (the streamer hasn't seen `y`).
    private func isInsideToolCall(after y: Int) -> Bool {
        if let toolCallStart, y == toolCallStart { return true }
        if let toolCallEnd, y == toolCallEnd { return false }
        return context.insideToolCall()
    }

    /// The proposal without what drafts may not contain: cut before a blocked token (except in
    /// exemplar structure) and after a stop token. Nil when nothing is left.
    private func sanitize(_ proposal: DraftProposal) -> DraftProposal? {
        let structural = proposal.source == .exemplar
        var tokens: [Int] = []
        for token in proposal.tokens {
            if !structural && blockedTokens.contains(token) { break }
            tokens.append(token)
            if context.stopTokens.contains(token) { break }
        }
        guard !tokens.isEmpty else { return nil }
        return DraftProposal(tokens: tokens, source: proposal.source, matchLength: proposal.matchLength)
    }

    // MARK: Helpers

    private func observe(committed: [Int], emitted: [Int], hidden: MLXArray?) {
        guard !drafters.isEmpty else { return }
        let observation = RoundObservation(committed: committed, emitted: emitted, hidden: hidden)
        for drafter in drafters {
            drafter.observe(observation)
        }
    }

    private func finish(_ reason: EngineFinish.Reason) {
        guard finished == nil else { return }
        finished = reason
        for case let drafter as ReplyObservingDrafter in drafters {
            drafter.replyFinished(emittedTokens)
        }
    }

    private func refreshDeviceState() {
        let state = options.deviceState?() ?? Self.currentDeviceState()
        sharedPolicy.policy.adjust(thermal: state.thermal, lowPower: state.lowPower)
        nextDeviceCheck = emittedCount + max(1, options.deviceStateInterval)
    }

    /// `ProcessInfo`'s thermal state and Low Power Mode.
    public static func currentDeviceState() -> (thermal: ThermalLevel, lowPower: Bool) {
        #if canImport(Darwin)
        let info = ProcessInfo.processInfo
        let thermal: ThermalLevel
        switch info.thermalState {
        case .nominal: thermal = .nominal
        case .fair: thermal = .fair
        case .serious: thermal = .serious
        case .critical: thermal = .critical
        @unknown default: thermal = .nominal
        }
        return (thermal, info.isLowPowerModeEnabled)
        #else
        return (.nominal, false)
        #endif
    }
}
