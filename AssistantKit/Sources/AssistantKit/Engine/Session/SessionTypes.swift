import Foundation

/// One conversation turn held in the engine's live cache, with the ledger positions the session
/// planner needs to rewind into it.
///
/// Positions are indices into the token ledger (the exact tokens in the cache). Only the newest
/// user turn carries them; older turns keep `nil`.
public struct CachedTurn: Equatable, Sendable {
    public var turn: ChatTurn
    /// For the newest user turn: the index of its `<|im_start|>`.
    public var start: Int?
    /// For the newest user turn: the position right after its generation prompt, where the reply's
    /// tokens begin.
    public var replyStart: Int?

    public init(turn: ChatTurn, start: Int? = nil, replyStart: Int? = nil) {
        self.turn = turn
        self.start = start
        self.replyStart = replyStart
    }
}

/// What the engine's live cache holds, in conversation terms.
///
/// The cache holds the system prefix (`0 ..< systemEnd`), then the turns a request sent from
/// `firstTurnIndex` on, each exactly as it was fed: user turns as rendered, replies as generated.
public struct SessionSnapshot: Equatable, Sendable {
    /// `PrefixKey.make(...)` of the model, system prompt, tools and chat context the cache was built with.
    public var prefixKey: String
    /// Ledger position where the system prefix ends; 0 when the template gave no usable prefix.
    public var systemEnd: Int
    /// Index, in the request's `turns`, of the first turn held in the cache.
    public var firstTurnIndex: Int
    /// The cached turns, in order.
    public var turns: [CachedTurn]
    /// Number of tokens in the cache (the ledger length).
    public var tokenCount: Int

    public init(prefixKey: String, systemEnd: Int, firstTurnIndex: Int, turns: [CachedTurn], tokenCount: Int) {
        self.prefixKey = prefixKey
        self.systemEnd = systemEnd
        self.firstTurnIndex = firstTurnIndex
        self.turns = turns
        self.tokenCount = tokenCount
    }

    /// Index in `turns` of the newest cached user turn.
    public var newestUserTurnIndex: Int? {
        turns.lastIndex { $0.turn.role == .user }
    }

    /// The snapshot after `plan` has been fed: the request's turns from `plan.firstTurnIndex` on,
    /// with the newest user turn (the request's last turn) marked at `userStart` and `replyStart`.
    /// `replyStart` is also the token count, since the cache now ends with the generation prompt.
    public static func afterPrefill(
        prefixKey: String,
        systemEnd: Int,
        plan: SessionPlan,
        turns: [ChatTurn],
        userStart: Int?,
        replyStart: Int
    ) -> SessionSnapshot {
        let first = min(max(0, plan.firstTurnIndex), turns.count)
        var cached = turns[first...].map { CachedTurn(turn: $0) }
        if !cached.isEmpty {
            cached[cached.count - 1].start = userStart
            cached[cached.count - 1].replyStart = replyStart
        }
        return SessionSnapshot(prefixKey: prefixKey, systemEnd: systemEnd, firstTurnIndex: first, turns: cached, tokenCount: replyStart)
    }

    /// Records the reply being generated (its visible text, trimmed, and its tool rounds so far)
    /// and the new cache length. Call it after each tool round and once the reply ends: the first
    /// call appends an assistant turn after the request's user turn, later calls replace it.
    public mutating func recordReply(_ reply: ChatTurn, tokenCount: Int) {
        if let last = turns.last, last.turn.role == .assistant {
            turns[turns.count - 1] = CachedTurn(turn: reply)
        } else {
            turns.append(CachedTurn(turn: reply))
        }
        self.tokenCount = tokenCount
    }
}

/// A piece of tokens to feed after a plan's base, in order.
public enum FeedPiece: Equatable, Sendable {
    /// The system prefix: the tokens of `[system, user]` (with tools) before the user turn's `<|im_start|>`.
    case systemPrefix
    /// The canonical render of `[system] + turns` with the generation prompt, minus the system
    /// prefix the ledger holds at this point (all of it when the ledger is empty, as after
    /// `.keep(0)` when the template gave no usable prefix).
    case firstTurns([ChatTurn])
    /// The sentinel delta of `turns` (`TurnDelta`); starts with a turn end and ends with the
    /// generation prompt.
    case continuation([ChatTurn])
    /// Raw reply text, fed after a generation prompt without a turn end.
    case assistantText(String)
}

/// How to bring the live cache to a request: where to rewind to, then what to feed.
public struct SessionPlan: Equatable, Sendable {
    public enum Base: Equatable, Sendable {
        /// Rewind the live cache to this ledger position (`tokenCount` means keep everything).
        case keep(Int)
        /// Load the system prefix saved on disk for the request's prefix key.
        case persistedPrefix
        /// Start from an empty cache.
        case empty
    }

    /// Which planner rule produced the plan.
    public enum Reason: String, Equatable, Sendable {
        case newSession, prefixChanged, append, replaceLastReply, replaceLastUserTurn, diverged, overBudget
    }

    public var base: Base
    public var pieces: [FeedPiece]
    /// Index, in the request's turns, of the first turn the cache holds once the plan has run.
    public var firstTurnIndex: Int
    public var reason: Reason

    public init(base: Base, pieces: [FeedPiece], firstTurnIndex: Int, reason: Reason) {
        self.base = base
        self.pieces = pieces
        self.firstTurnIndex = firstTurnIndex
        self.reason = reason
    }

    /// Ledger tokens the plan keeps from the live cache (0 unless the base is `.keep`).
    public var keptTokens: Int {
        if case .keep(let position) = base { return position }
        return 0
    }
}
