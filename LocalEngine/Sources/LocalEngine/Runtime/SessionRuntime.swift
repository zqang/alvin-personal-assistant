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

    func prepare(_ request: EngineRequest, isAllowed: () -> Bool) throws -> Prepared {
        guard request.turns.last?.role == .user else {
            throw EngineError.renderFailed("the request doesn't end with a user turn")
        }
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

        let live = session.snapshot.flatMap { $0.tokenCount == session.ledger.count ? $0 : nil }
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
            session.rewind(to: position)
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

        let logits = try prefill(tokens, marks: marks, isAllowed: isAllowed)
        phases.prefill = lap()

        session.snapshot = SessionSnapshot.afterPrefill(
            prefixKey: key, systemEnd: systemEnd, plan: plan, turns: request.turns, userStart: userStart, replyStart: replyStart)
        applyCheckpointPolicy()
        return Prepared(
            firstLogits: logits, reason: plan.reason.rawValue, phases: phases, prefilledTokens: tokens.count, reusedTokens: reused)
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
            session.rewind(to: session.ledger.count - 1)
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
    func finishGeneration(text: String, calls: [PendingToolCall], startedCall: Bool, reason: EngineFinish.Reason) {
        guard reply != nil else { return }
        reply?.text += text
        reply?.pendingCalls = (!calls.isEmpty && reason != .cancelled) ? calls : nil
        recordReply(unanswered: startedCall || !calls.isEmpty)
    }

    /// Stores the reply in the snapshot, as an assistant turn after the request's user turn.
    /// A tool call without results is recorded as a round no stored turn can equal, so a later
    /// request never extends a cache that ends in an unanswered call as if it were plain text.
    private func recordReply(unanswered: Bool) {
        guard let state = reply, session.snapshot != nil else { return }
        var rounds = state.rounds
        if unanswered {
            rounds.append(ToolRound(calls: [ToolCallRecord(id: "", name: "", input: .null, result: "", isError: true)]))
        }
        let text = state.text.trimmingCharacters(in: .whitespacesAndNewlines)
        session.snapshot?.recordReply(ChatTurn(role: .assistant, text: text, toolRounds: rounds), tokenCount: session.ledger.count)
    }

    /// Forgets the conversation in the cache after a failure (the ledger stays exact).
    func abandon() {
        session.snapshot = nil
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
        if let snapshot = session.snapshot, snapshot.prefixKey == key, snapshot.systemEnd > 0,
           session.checkpoints.marks[.systemEnd] == snapshot.systemEnd, session.ledger.count >= snapshot.systemEnd
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
