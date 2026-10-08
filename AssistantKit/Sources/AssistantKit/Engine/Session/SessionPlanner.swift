import Foundation

/// Plans how the engine reuses its live cache for a request, so that only tokens the cache doesn't
/// already hold are prefilled. Pure: it reads a snapshot of the cache and the request's turns.
///
/// The cache keeps replies exactly as they were generated ("append as generated"); a request that
/// extends the cached conversation only appends its delta. The first matching rule wins:
///
/// | # | Condition | Base | Pieces | Reason |
/// |---|---|---|---|---|
/// | 0 | The request doesn't end with a user turn | — | returns nil | — |
/// | 1 | No live session | persisted prefix if one exists, else empty | `[.systemPrefix]` (empty only) + `[.firstTurns(window)]` | `newSession` |
/// | 2 | The prefix key changed | as rule 1 | as rule 1 | `prefixChanged` |
/// | 3 | Every cached turn matches the request, the cache ends on a reply, and more turns follow | keep everything | `[.continuation(newTurns)]` | `append` |
/// | 4 | Only the last cached reply differs, neither version has tool rounds, and more turns follow | keep to the reply's start | `[.assistantText(text), .continuation(rest)]` | `replaceLastReply` |
/// | 5 | The turns before the newest cached user turn match, and the request ends with a replacement for that turn (a tentative turn, an edit, or a retry); any reply after it is dropped | keep to the turn's start (must be after the system prefix) | `[.continuation([user])]` | `replaceLastUserTurn` |
/// | 6 | Anything else, including another conversation | keep the system prefix | `[.firstTurns(window)]` | `diverged` |
/// | 7 | Rules 3–5 would grow the cache past `maxTokens` | as rule 6 | as rule 6 | `overBudget` |
///
/// The window is the last `keepTurns` turns, moved forward to start on a user turn.
public enum SessionPlanner {
    public struct Limits: Equatable, Sendable {
        /// Most tokens the cache may hold before it is rebuilt from the window.
        public var maxTokens = 6_144
        /// Turns a rebuilt cache starts from (the request's last turn included).
        public var keepTurns = 12

        public init(maxTokens: Int = 6_144, keepTurns: Int = 12) {
            self.maxTokens = maxTokens
            self.keepTurns = keepTurns
        }
    }

    public static func plan(
        live: SessionSnapshot?,
        prefixKey: String,
        hasPersistedPrefix: Bool,
        turns: [ChatTurn],
        limits: Limits = .init()
    ) -> SessionPlan? {
        // Rule 0.
        guard let last = turns.last, last.role == .user else { return nil }
        let window = windowStart(turns, keepTurns: limits.keepTurns)

        // Rules 1 and 2.
        guard let live else {
            return fresh(turns: turns, window: window, hasPersistedPrefix: hasPersistedPrefix, reason: .newSession)
        }
        guard live.prefixKey == prefixKey else {
            return fresh(turns: turns, window: window, hasPersistedPrefix: hasPersistedPrefix, reason: .prefixChanged)
        }

        func rebuild(_ reason: SessionPlan.Reason) -> SessionPlan {
            SessionPlan(
                base: .keep(min(max(0, live.systemEnd), live.tokenCount)),
                pieces: [.firstTurns(Array(turns[window...]))],
                firstTurnIndex: window,
                reason: reason
            )
        }

        // A reuse plan, unless it would take the cache past the budget (rule 7).
        func reuse(keep position: Int, adding added: ArraySlice<ChatTurn>, pieces: [FeedPiece], reason: SessionPlan.Reason) -> SessionPlan {
            guard position + estimatedTokens(added) <= limits.maxTokens else { return rebuild(.overBudget) }
            return SessionPlan(base: .keep(position), pieces: pieces, firstTurnIndex: live.firstTurnIndex, reason: reason)
        }

        let first = live.firstTurnIndex
        let cached = live.turns
        guard first >= 0, first <= turns.count else { return rebuild(.diverged) }
        let request = turns[first...]
        let positions = 0 ... live.tokenCount

        /// Whether the first `count` cached turns equal the request's turns from `first` on.
        func cachedPrefixMatches(_ count: Int) -> Bool {
            count <= request.count && zip(cached.prefix(count), request.prefix(count)).allSatisfy { LocalSessionPlan.sameTurn($0.turn, $1) }
        }

        // Rule 3: the request extends the cached conversation.
        if let lastCached = cached.last, lastCached.turn.role == .assistant,
           request.count > cached.count, cachedPrefixMatches(cached.count)
        {
            let newTurns = request.dropFirst(cached.count)
            return reuse(keep: live.tokenCount, adding: newTurns, pieces: [.continuation(Array(newTurns))], reason: .append)
        }

        // Rule 4: the last cached reply was interrupted, shortened or edited; more turns follow.
        if cached.count >= 2, request.count > cached.count {
            let replyIndex = cached.count - 1
            let cachedReply = cached[replyIndex]
            let cachedUser = cached[replyIndex - 1]
            let newReply = request[request.startIndex + replyIndex]
            if cachedReply.turn.role == .assistant, newReply.role == .assistant,
               cachedReply.turn.toolRounds.isEmpty, newReply.toolRounds.isEmpty,
               cachedUser.turn.role == .user, let replyStart = cachedUser.replyStart, positions.contains(replyStart),
               cachedPrefixMatches(replyIndex)
            {
                let rest = request.dropFirst(replyIndex + 1)
                return reuse(
                    keep: replyStart,
                    adding: request.dropFirst(replyIndex),
                    pieces: [.assistantText(newReply.text.trimmed), .continuation(Array(rest))],
                    reason: .replaceLastReply
                )
            }
        }

        // Rule 5: the newest cached user turn was replaced (a tentative turn, an edit, or a retry),
        // and the request ends with its replacement. The continuation needs a turn end to follow,
        // so a turn at the very start of the ledger (no system prefix) is rebuilt instead.
        if let userIndex = live.newestUserTurnIndex, request.count == userIndex + 1,
           let start = cached[userIndex].start, start > 0, positions.contains(start),
           cachedPrefixMatches(userIndex)
        {
            return reuse(keep: start, adding: request.suffix(1), pieces: [.continuation([last])], reason: .replaceLastUserTurn)
        }

        // Rule 6.
        return rebuild(.diverged)
    }

    /// The index the window of the last `keepTurns` turns starts at, moved forward to a user turn.
    /// `turns.count` when no user turn is left.
    public static func windowStart(_ turns: [ChatTurn], keepTurns: Int) -> Int {
        var start = max(0, turns.count - max(1, keepTurns))
        while start < turns.count, turns[start].role != .user { start += 1 }
        return start
    }

    /// A rough token count for `turns`: UTF-8 bytes / 3, plus 8 per turn and 64 per tool round.
    /// It errs high for English (about 4 bytes per token) and is close for Chinese.
    public static func estimatedTokens<C: Collection>(_ turns: C) -> Int where C.Element == ChatTurn {
        var bytes = 0
        var overhead = 0
        for turn in turns {
            bytes += LocalSessionPlan.content(of: turn).utf8.count
            overhead += 8 + 64 * turn.toolRounds.count
        }
        return bytes / 3 + overhead
    }

    private static func fresh(turns: [ChatTurn], window: Int, hasPersistedPrefix: Bool, reason: SessionPlan.Reason) -> SessionPlan {
        let firstTurns = FeedPiece.firstTurns(Array(turns[window...]))
        return SessionPlan(
            base: hasPersistedPrefix ? .persistedPrefix : .empty,
            pieces: hasPersistedPrefix ? [firstTurns] : [.systemPrefix, firstTurns],
            firstTurnIndex: window,
            reason: reason
        )
    }
}
