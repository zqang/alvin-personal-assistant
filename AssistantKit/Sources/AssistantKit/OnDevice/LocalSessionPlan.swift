import Foundation

/// Decides whether an on-device reply can continue the model's cached session or has to rebuild
/// it, the way Husky avoids re-reading a conversation it has already processed.
///
/// A cached session holds the system prompt and every turn it has seen, including the replies it
/// generated. When a request is exactly that history plus one new user turn, only the new turn
/// needs a prefill. Anything else (an interrupted or edited reply, another conversation, a
/// changed system prompt, a history past `maxTurns`) rebuilds from the last `keepTurns` turns.
public struct LocalSessionPlan: Equatable, Sendable {
    public enum Action: Equatable, Sendable {
        /// Append `newTurn` to the cached session.
        case append
        /// Start a new session primed with `history`.
        case rebuild(history: [ChatTurn])
    }

    public let action: Action
    /// The user turn to answer.
    public let newTurn: ChatTurn

    /// Most turns a session may hold before it is rebuilt from a shorter window.
    public static let maxTurns = 24
    /// Turns a rebuilt session starts from (the new user turn included).
    public static let keepTurns = 12

    /// Nil if the request doesn't end with a user turn.
    public static func make(
        cachedSystem: String?,
        cachedTurns: [ChatTurn]?,
        system: String,
        turns: [ChatTurn],
        maxTurns: Int = maxTurns,
        keepTurns: Int = keepTurns
    ) -> LocalSessionPlan? {
        guard let last = turns.last, last.role == .user else { return nil }
        if let cachedTurns, cachedSystem == system,
           turns.count == cachedTurns.count + 1, turns.count <= maxTurns,
           zip(cachedTurns, turns).allSatisfy(sameTurn)
        {
            return LocalSessionPlan(action: .append, newTurn: last)
        }
        var history = Array(turns.dropLast().suffix(max(0, keepTurns - 1)))
        // Start on a user turn, as the chat template expects.
        while history.first?.role == .assistant { history.removeFirst() }
        return LocalSessionPlan(action: .rebuild(history: history), newTurn: last)
    }

    /// Generated replies are compared without surrounding whitespace, which storage trims.
    /// Tool rounds must match exactly: they are part of what the model saw.
    static func sameTurn(_ a: ChatTurn, _ b: ChatTurn) -> Bool {
        a.role == b.role && a.context == b.context && a.text.trimmed == b.text.trimmed && a.toolRounds == b.toolRounds
    }

    /// The text a turn is sent to the model as: its context tag, then what was said.
    public static func content(of turn: ChatTurn) -> String {
        guard turn.role == .user, let context = turn.context, !context.isEmpty else { return turn.text }
        return context + "\n\n" + turn.text
    }
}

extension String {
    var trimmed: String { trimmingCharacters(in: .whitespacesAndNewlines) }
}
