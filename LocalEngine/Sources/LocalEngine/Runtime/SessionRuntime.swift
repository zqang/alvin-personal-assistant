import AssistantKit
import Foundation
import MLX
import MLXLMCommon

/// Brings the live session to a request and keeps its conversation snapshot (plan §4.4–§4.5,
/// WP20 instruction 4).
///
/// `prepare`:
/// 1. computes the prefix key;
/// 2. plans with `SessionPlanner`;
/// 3. executes the base: rewind (`keep`), the disk prefix (`persistedPrefix`) or an empty cache;
/// 4. renders the pieces; 5. applies the overlap rule to each continuation;
/// 6. places the checkpoints (`systemEnd`, the newest user turn's start, `replyStart`);
/// 7. prefills; 8. updates the snapshot and applies `CheckpointPolicy`; 9. times the phases.
///
/// A render failure falls back to rendering the whole conversation into an empty cache for
/// that reply. Not thread-safe: engine queue only.
final class SessionRuntime {
    let session: LiveSession
    /// Nil when the tokenizer has no ChatML markers (every reply renders everything).
    private(set) var renderer: TurnRenderer?
    let templates: any ChatTemplateRendering
    let modelID: String
    /// The snapshot directory's name, part of the prefix key.
    let revision: String
    var configuration: EngineConfiguration
    var prefixStore: PrefixCacheStore?

    /// What `prepare` did.
    struct Prepared {
        /// `[1, V]` logits predicting the reply's first token (evaluated).
        let firstLogits: MLXArray
        let reason: String
        let phases: EnginePhaseTimes
        let prefilledTokens: Int
        let reusedTokens: Int
    }

    /// The reply being generated: across tool rounds it stays one assistant turn.
    struct ReplyState {
        var request: EngineRequest
        var rounds: [ToolRound] = []
        /// Visible text of every generation of this reply so far.
        var text = ""
        /// Calls the last generation asked for, waiting for `continueReply`.
        var pendingCalls: [PendingToolCall]?
        /// Whether this reply renders everything into an empty cache (no reuse).
        var fullRender = false
    }

    private(set) var reply: ReplyState?

    init(session: LiveSession, renderer: TurnRenderer?, templates: any ChatTemplateRendering, modelID: String, revision: String,
         configuration: EngineConfiguration) {
        self.session = session
        self.renderer = renderer
        self.templates = templates
        self.modelID = modelID
        self.revision = revision
        self.configuration = configuration
        session.prefillChunk = configuration.prefillChunk
        self.prefixStore = configuration.prefixCacheDirectory.map(PrefixCacheStore.init(directory:))
        prefixStore?.pruneOtherModels(modelID: modelID)
    }

    func apply(_ configuration: EngineConfiguration) {
        let directoryChanged = configuration.prefixCacheDirectory != self.configuration.prefixCacheDirectory
        if configuration.chatContext != self.configuration.chatContext, renderer != nil {
            renderer = try? TurnRenderer(renderer: templates, chatContext: configuration.chatContext)
        }
        self.configuration = configuration
        session.prefillChunk = configuration.prefillChunk
        if directoryChanged {
            prefixStore = configuration.prefixCacheDirectory.map(PrefixCacheStore.init(directory:))
            prefixStore?.pruneOtherModels(modelID: modelID)
        }
        // The chat context is part of the prefix key, so a change makes the planner rebuild.
        applyCheckpointPolicy()
    }

    var chatContext: [String: Bool] { configuration.chatContext }

    func prefixKey(system: String, tools: [ToolDefinition]) -> String {
        PrefixKey.make(
            modelID: modelID, revision: revision, system: system, toolsJSON: PrefixKey.canonicalToolsJSON(tools),
            context: PrefixKey.canonicalContext(chatContext), formatVersion: EngineInfo.formatVersion)
    }

    // MARK: Preparing a reply

    /// Throws `renderFailed` for a request `prepare` can't answer (one that doesn't end with a
    /// user turn). Checked before anything changes, so a refused request leaves the session and
    /// a reply waiting for tool results as they were.
    static func validate(_ request: EngineRequest) throws {
        guard request.turns.last?.role == .user else {
            throw EngineError.renderFailed("the request doesn't end with a user turn")
        }
    }

    func prepare(_ request: EngineRequest, isAllowed: () -> Bool) throws -> Prepared {
        try Self.validate(request)
        reply = ReplyState(request: request)
        let key = prefixKey(system: request.system, tools: request.tools)
        guard let renderer, !session.noReuse else {
            return try prepareFullRender(request, extra: nil, reason: "noReuse", isAllowed: isAllowed)
        }
        do {
            return try prepareReusing(request, key: key, renderer: renderer, isAllowed: isAllowed)
        } catch EngineError.renderFailed {
            return try prepareFullRender(request, extra: nil, reason: "renderFailed", isAllowed: isAllowed)
        }
    }

    private func prepareReusing(_ request: EngineRequest, key: String, renderer: TurnRenderer, isAllowed: () -> Bool) throws -> Prepared {
        var phases = EnginePhaseTimes()
        var clock = Date()
        func lap() -> TimeInterval {
            let now = Date()
            defer { clock = now }
            return now.timeIntervalSince(clock)
        }

        // A snapshot may describe fewer tokens than the ledger holds (after an interrupted
        // prefill); every plan then rewinds to a position it describes.
        let live = session.snapshot.flatMap { $0.tokenCount <= session.ledger.count ? $0 : nil }
        let hasPersistedPrefix = prefixStore?.contains(key: key) ?? false
        guard let plan = SessionPlanner.plan(
            live: live, prefixKey: key, hasPersistedPrefix: hasPersistedPrefix, turns: request.turns, limits: configuration.limits)
        else {
            throw EngineError.renderFailed("the request doesn't end with a user turn")
        }
        phases.plan = lap()
        // Anything below can leave the cache mid-change; the snapshot is rebuilt at the end.
        session.snapshot = nil

        // The base.
        var pieces = plan.pieces
        var base = plan.base
        var systemEnd = 0
        var reused = 0
        switch plan.base {
        case .keep(let position):
            let position = min(position, session.ledger.count)
            do {
                try session.rewind(to: position, isAllowed: isAllowed)
            } catch let interrupted as PrefillInterrupted {
                // The re-feed stopped short of `position`: keep the deepest cut of the
                // conversation the ledger still holds.
                session.snapshot = Self.snapshot(of: live, describingAtMost: session.ledger.count, key: key)
                throw interrupted
            }
            systemEnd = min(live?.systemEnd ?? 0, position)
            reused = position
        case .persistedPrefix:
            if let prefix = renderer.systemPrefix(system: request.system, tools: request.tools), !prefix.isEmpty,
               let cache = prefixStore?.load(key: key, expectedTokens: prefix, layout: session.layout),
               (try? session.adopt(cache, holding: prefix)) != nil
            {
                session.checkpoint(.systemEnd)
                systemEnd = prefix.count
                reused = prefix.count
            } else {
                session.reset()
                base = .empty
                pieces = [.systemPrefix] + pieces
            }
        case .empty:
            session.reset()
        }
        phases.rewind = lap()
        phases.start = EnginePhaseTimes.start(for: base)

        // The pieces, with the overlap rule for continuations.
        var tokens: [Int] = []
        for piece in pieces {
            switch piece {
            case .systemPrefix:
                if session.ledger.isEmpty && tokens.isEmpty,
                   let prefix = renderer.systemPrefix(system: request.system, tools: request.tools)
                {
                    tokens += prefix
                    systemEnd = prefix.count
                }
            case .firstTurns(let turns):
                tokens += try renderer.firstTurns(
                    system: request.system, tools: request.tools, turns: turns, after: session.ledger + tokens)
            case .continuation(let turns):
                let delta = try renderer.continuation(system: request.system, tools: request.tools, turns: turns)
                tokens += dropOverlap(delta, after: tokens)
            case .assistantText(let text):
                tokens += renderer.assistantText(text)
            }
        }
        phases.render = lap()

        // Checkpoints: the system prefix's end, the newest user turn's start, the reply's start.
        let start = session.ledger.count
        let replyStart = start + tokens.count
        let combined = session.ledger + tokens
        let userStart = TurnDelta.lastUserTurnStart(in: combined[...], turnStart: renderer.turnStart)
        var marks: [(CheckpointMark, Int)] = [(.replyStart, replyStart)]
        if systemEnd > 0 && systemEnd >= start {
            marks.append((.systemEnd, systemEnd))
        }
        if let userStart, userStart >= start {
            marks.append((.lastUserStart, userStart))
        }

        // An interrupted prefill keeps what the cache still holds reusable: the base, which the
        // next plan rewinds to. Hold the state there until the prefill completes, so that rewind
        // doesn't re-feed from an earlier checkpoint.
        let resumable = Self.snapshot(of: live, base: base, kept: start, key: key, systemEnd: systemEnd)
        if resumable != nil {
            session.holdResumePoint()
        }
        let logits: MLXArray
        do {
            logits = try prefill(tokens, marks: marks, isAllowed: isAllowed)
        } catch is PrefillInterrupted {
            // Checkpoints taken past the start are stale: no plan keeps more than `resumable`.
            session.checkpoints.drop(above: start)
            session.snapshot = resumable
            throw PrefillInterrupted()
        }
        session.releaseResumePoint()
        phases.prefill = lap()

        session.snapshot = SessionSnapshot.afterPrefill(
            prefixKey: key, systemEnd: systemEnd, plan: plan, turns: request.turns, userStart: userStart, replyStart: replyStart)
        applyCheckpointPolicy()
        return Prepared(
            firstLogits: logits, reason: plan.reason.rawValue, phases: phases, prefilledTokens: tokens.count, reusedTokens: reused)
    }

    /// The snapshot describing the first `kept` tokens of the ledger after a plan's base ran:
    /// the live snapshot cut back to `kept`, or the system prefix alone. Nil if no snapshot
    /// describes them.
    static func snapshot(of live: SessionSnapshot?, base: SessionPlan.Base, kept: Int, key: String, systemEnd: Int) -> SessionSnapshot? {
        switch base {
        case .keep:
            guard let live else { return nil }
            if kept == live.tokenCount { return live }
            if kept == live.systemEnd && kept > 0 {
                return SessionSnapshot(prefixKey: key, systemEnd: live.systemEnd, firstTurnIndex: live.firstTurnIndex, turns: [], tokenCount: kept)
            }
            guard let index = live.newestUserTurnIndex else { return nil }
            var cut = live
            if kept == live.turns[index].replyStart {
                cut.turns = Array(live.turns[...index])
            } else if kept == live.turns[index].start {
                cut.turns = Array(live.turns[..<index])
            } else {
                return nil
            }
            cut.tokenCount = kept
            return cut
        case .persistedPrefix, .empty:
            guard kept > 0, kept == systemEnd else { return nil }
            return SessionSnapshot(prefixKey: key, systemEnd: systemEnd, firstTurnIndex: 0, turns: [], tokenCount: kept)
        }
    }

    /// The deepest cut of `live` (`snapshot(of:base:kept:...)` with a `.keep` base) that describes
    /// at most `limit` tokens: what stays reusable after a rewind stopped short. Nil if none does.
    static func snapshot(of live: SessionSnapshot?, describingAtMost limit: Int, key: String) -> SessionSnapshot? {
        guard let live else { return nil }
        var cuts = [live.tokenCount, live.systemEnd]
        if let index = live.newestUserTurnIndex {
            cuts += [live.turns[index].replyStart, live.turns[index].start].compactMap { $0 }
        }
        for kept in cuts.filter({ $0 <= limit }).sorted(by: >) {
            if let cut = snapshot(of: live, base: .keep(kept), kept: kept, key: key, systemEnd: live.systemEnd) {
                return cut
            }
        }
        return nil
    }

    /// Renders `[system] + window` (+ `extra`, the reply in progress) into an empty cache. The
    /// session then holds no reusable snapshot.
    private func prepareFullRender(_ request: EngineRequest, extra: ChatTurn?, reason: String, isAllowed: () -> Bool) throws -> Prepared {
        var phases = EnginePhaseTimes(start: "cold")
        var clock = Date()
        func lap() -> TimeInterval {
            let now = Date()
            defer { clock = now }
            return now.timeIntervalSince(clock)
        }
        reply?.fullRender = true
        session.reset()
        phases.rewind = lap()
        let window = SessionPlanner.windowStart(request.turns, keepTurns: configuration.limits.keepTurns)
        var turns = Array(request.turns[min(window, request.turns.count)...])
        if let extra { turns.append(extra) }
        let tokens = try TurnRenderer.fullRender(
            renderer: templates, chatContext: chatContext, system: request.system, tools: request.tools, turns: turns)
        phases.render = lap()
        let logits = try prefill(tokens, marks: [], isAllowed: isAllowed)
        phases.prefill = lap()
        return Prepared(firstLogits: logits, reason: reason, phases: phases, prefilledTokens: tokens.count, reusedTokens: 0)
    }

    /// Prefills `tokens` and returns the evaluated logits after the last one. With nothing to
    /// feed, the last ledger token is fed again (after a one-token rewind) for its logits.
    private func prefill(_ tokens: [Int], marks: [(CheckpointMark, Int)], isAllowed: () -> Bool) throws -> MLXArray {
        var tokens = tokens
        if tokens.isEmpty {
            guard let last = session.ledger.last else {
                throw EngineError.renderFailed("nothing to prefill")
            }
            try session.rewind(to: session.ledger.count - 1, isAllowed: isAllowed)
            tokens = [last]
        }
        guard let logits = try Prefill.run(
            session, tokens: tokens, chunk: configuration.prefillChunk, marks: marks, wantLogits: true, isAllowed: isAllowed)
        else {
            throw EngineError.renderFailed("the prefill returned no logits")
        }
        eval(logits)
        return logits
    }

    /// `delta` without its longest prefix (≤ 4 tokens) that the cache already ends with (L3).
    private func dropOverlap(_ delta: [Int], after tokens: [Int]) -> [Int] {
        let tail = Array((session.ledger + tokens).suffix(4))
        let overlap = TurnDelta.overlap(ledgerTail: tail[...], delta: delta)
        return Array(delta.dropFirst(overlap))
    }

    // MARK: Tool rounds

    /// Whether a reply is waiting for tool results.
    var awaitingToolResults: Bool {
        reply?.pendingCalls != nil
    }

    /// Appends `round`'s results to the reply in progress (whose last generation asked for the
    /// tools) and the generation prompt that follows. Throws `busy` when no reply is waiting.
    func prepareContinuation(after round: ToolRound, isAllowed: () -> Bool) throws -> Prepared {
        guard var state = reply, state.pendingCalls != nil else {
            throw EngineError.busy
        }
        state.rounds.append(round)
        state.pendingCalls = nil
        reply = state
        let request = state.request
        let inProgress = ChatTurn(role: .assistant, text: "", toolRounds: state.rounds)

        guard let renderer, !session.noReuse, !state.fullRender, session.snapshot != nil else {
            return try prepareFullRender(request, extra: inProgress, reason: "toolRound", isAllowed: isAllowed)
        }
        var phases = EnginePhaseTimes(start: "warm")
        var clock = Date()
        func lap() -> TimeInterval {
            let now = Date()
            defer { clock = now }
            return now.timeIntervalSince(clock)
        }
        let delta: [Int]
        do {
            delta = try renderer.toolRoundContinuation(system: request.system, tools: request.tools, round: round)
        } catch EngineError.renderFailed {
            return try prepareFullRender(request, extra: inProgress, reason: "renderFailed", isAllowed: isAllowed)
        }
        let tokens = dropOverlap(delta, after: [])
        phases.render = lap()
        let reused = session.ledger.count
        let logits = try prefill(tokens, marks: [], isAllowed: isAllowed)
        phases.prefill = lap()
        recordReply(unanswered: false)
        return Prepared(firstLogits: logits, reason: "toolRound", phases: phases, prefilledTokens: tokens.count, reusedTokens: reused)
    }

    // MARK: After a generation

    /// Records what a generation of the reply produced: its visible text, and the tool calls it
    /// asks for (`calls`, waiting for `continueReply`). `startedCall` tells that a tool call
    /// began, even if it didn't parse.
    /// `incomplete` tells that some emitted tokens couldn't be fed (the GPU stopped being
    /// allowed before `flush`).
    func finishGeneration(text: String, calls: [PendingToolCall], startedCall: Bool, reason: EngineFinish.Reason, incomplete: Bool = false) {
        guard reply != nil else { return }
        reply?.text += text
        reply?.pendingCalls = (!calls.isEmpty && reason != .cancelled) ? calls : nil
        recordReply(unanswered: startedCall || !calls.isEmpty, incomplete: incomplete)
    }

    /// Marks a recorded reply whose last emitted tokens aren't in the cache: no stored text
    /// equals it, so the planner replaces the reply instead of extending it.
    static let incompleteMarker = "\u{F8FF}"

    /// Stores the reply in the snapshot, as an assistant turn after the request's user turn.
    /// A tool call without results is recorded as a round no stored turn can equal, so a later
    /// request never extends a cache that ends in an unanswered call as if it were plain text.
    private func recordReply(unanswered: Bool, incomplete: Bool = false) {
        guard let state = reply, session.snapshot != nil else { return }
        var rounds = state.rounds
        if unanswered {
            rounds.append(ToolRound(calls: [ToolCallRecord(id: "", name: "", input: .null, result: "", isError: true)]))
        }
        var text = state.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if incomplete {
            text += Self.incompleteMarker
        }
        session.snapshot?.recordReply(ChatTurn(role: .assistant, text: text, toolRounds: rounds), tokenCount: session.ledger.count)
    }

    /// Forgets the conversation after a failure part-way through a job. The cache may then hold
    /// tokens the ledger doesn't (a generator that threw with tokens still pending, a forward
    /// cut short by an MLX error), so it starts over empty, which keeps L1 for every later job
    /// (scratch work, checks). Nothing is lost: without a snapshot no plan reuses the cache.
    func abandon() {
        session.reset()
        reply = nil
    }

    /// After an interrupted prefill: the snapshot already describes what stays reusable.
    func interrupted() {
        reply = nil
    }

    // MARK: Checkpoints and the disk prefix

    func applyCheckpointPolicy() {
        let kept = CheckpointPolicy.marksToKeep(
            session.checkpoints.marks, bytesPerMark: session.bytesPerCheckpoint, budgetBytes: configuration.checkpointBudgetBytes)
        session.checkpoints.keep(kept)
    }

    /// Saves the cache's system prefix to disk if it isn't there yet.
    func persistPrefixIfNeeded(isAllowed: () -> Bool) {
        guard let store = prefixStore, let snapshot = session.snapshot, snapshot.systemEnd > 0,
              session.ledger.count >= snapshot.systemEnd,
              let checkpoint = session.checkpoints.snapshot(for: .systemEnd), checkpoint.position == snapshot.systemEnd,
              !store.contains(key: snapshot.prefixKey), isAllowed()
        else { return }
        let position = snapshot.systemEnd
        let layout = session.layout
        let cache = session.target.cache
        var kvState: [Int: [MLXArray]] = [:]
        for index in layout.attention {
            kvState[index] = cache[index].state
        }
        do {
            try store.save(
                snapshot: checkpoint, position: position, kvState: kvState, layout: layout, key: snapshot.prefixKey,
                tokens: Array(session.ledger[..<position]), modelID: modelID)
        } catch {
            print("LocalEngine: couldn't save the system prefix: \(error)")
        }
    }

    /// Makes the system prefix of `system` and `tools` resident: kept if live, else loaded from
    /// disk, else prefilled (and saved).
    func prewarm(system: String, tools: [ToolDefinition], isAllowed: () -> Bool) throws {
        guard let renderer, !session.noReuse else { return }
        let key = prefixKey(system: system, tools: tools)
        // The live conversation already starts with this prefix: keep it (and a reply waiting
        // for tool results). Its `systemEnd` checkpoint may be gone (`dropCheckpoints`); a later
        // rewind to it then re-feeds from the deepest checkpoint left, which stays exact.
        if let snapshot = session.snapshot, snapshot.prefixKey == key, snapshot.systemEnd > 0,
           session.ledger.count >= snapshot.systemEnd
        {
            return
        }
        guard let prefix = renderer.systemPrefix(system: system, tools: tools), !prefix.isEmpty else { return }
        reply = nil
        if let cache = prefixStore?.load(key: key, expectedTokens: prefix, layout: session.layout),
           (try? session.adopt(cache, holding: prefix)) != nil
        {
            session.checkpoint(.systemEnd)
        } else {
            session.reset()
            _ = try Prefill.run(
                session, tokens: prefix, chunk: configuration.prefillChunk, marks: [(.systemEnd, prefix.count)], wantLogits: false,
                isAllowed: isAllowed)
        }
        session.snapshot = SessionSnapshot(prefixKey: key, systemEnd: prefix.count, firstTurnIndex: 0, turns: [], tokenCount: prefix.count)
        persistPrefixIfNeeded(isAllowed: isAllowed)
    }

    /// Drops the live conversation (the cache starts empty next time).
    func invalidate() {
        session.reset()
        reply = nil
    }
}
