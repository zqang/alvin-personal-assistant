import Foundation

/// Token-exact deltas between a cached conversation and a request, worked out from chat-template
/// renders without knowing the template.
///
/// **Sentinel render.** To get the tokens that follow a cached reply, render
/// `[system, user(userSentinel), assistant(assistantSentinel)] + newTurns` with the generation
/// prompt, decode with special tokens kept, and cut at the first turn end after the assistant
/// sentinel (`continuation(in:after:turnEnd:)`). The cut starts with a special token, where BPE
/// always splits, so encoding it gives exactly that segment of a real render. For a tool round,
/// render `assistant(assistantSentinel, tool_calls:)` and the `tool` messages instead; the delta
/// again starts at the first turn end after the sentinel.
///
/// **Overlap rule.** A cache that stopped normally ends with the turn end it generated (fed, as
/// `TokenIterator` does); one that was cancelled ends at its last content token; one rewound to a
/// user turn ends with `<|im_end|>\n`. Before appending a delta, drop its longest prefix (up to
/// four tokens) that the ledger already ends with (`overlap(ledgerTail:delta:)`).
public enum TurnDelta {
    public static let userSentinel = "ALVIN_SPLIT_U_7F3A9C"
    public static let assistantSentinel = "ALVIN_SPLIT_A_7F3A9C"

    /// The turns of a sentinel render for `newTurns`: a placeholder user turn and reply, then `newTurns`.
    public static func sentinelTurns<C: Collection>(followedBy newTurns: C) -> [ChatTurn] where C.Element == ChatTurn {
        [ChatTurn(role: .user, text: userSentinel), ChatTurn(role: .assistant, text: assistantSentinel)] + Array(newTurns)
    }

    /// The text of `rendered` from the first `turnEnd` after `sentinel` (inclusive) to the end.
    /// Nil unless `sentinel` occurs exactly once and a `turnEnd` follows it.
    public static func continuation(in rendered: String, after sentinel: String, turnEnd: String) -> String? {
        guard !sentinel.isEmpty, !turnEnd.isEmpty,
              let found = rendered.range(of: sentinel, options: .literal),
              rendered.range(of: sentinel, options: .literal, range: found.upperBound ..< rendered.endIndex) == nil,
              let end = rendered.range(of: turnEnd, options: .literal, range: found.upperBound ..< rendered.endIndex)
        else { return nil }
        return String(rendered[end.lowerBound...])
    }

    /// The length of the longest prefix of `delta`, at most `maxOverlap` tokens, that `ledgerTail` ends with.
    public static func overlap(ledgerTail: ArraySlice<Int>, delta: [Int], maxOverlap: Int = 4) -> Int {
        let limit = min(maxOverlap, delta.count, ledgerTail.count)
        guard limit > 0 else { return 0 }
        for length in stride(from: limit, through: 1, by: -1)
        where ledgerTail.suffix(length).elementsEqual(delta.prefix(length)) {
            return length
        }
        return 0
    }

    /// `tokens` without `prefix`, or nil if `tokens` doesn't start with it.
    public static func dropPrefix<P: Collection>(_ prefix: P, from tokens: [Int]) -> [Int]? where P.Element == Int {
        guard tokens.starts(with: prefix) else { return nil }
        return Array(tokens.dropFirst(prefix.count))
    }

    /// The system prefix of a `[system, user]` render without the generation prompt: the tokens
    /// before its last `turnStart`. Nil when there is no `turnStart` or nothing precedes it.
    public static func systemPrefix(in rendered: [Int], turnStart: Int) -> [Int]? {
        guard let index = rendered.lastIndex(of: turnStart), index > 0 else { return nil }
        return Array(rendered[..<index])
    }

    /// The absolute index of the second-to-last `turnStart` in `tokens`: after a request has been
    /// fed (ending with the generation prompt's `<|im_start|>assistant…`), the start of its user turn.
    public static func lastUserTurnStart(in tokens: ArraySlice<Int>, turnStart: Int) -> Int? {
        guard let last = tokens.lastIndex(of: turnStart) else { return nil }
        return tokens[tokens.startIndex ..< last].lastIndex(of: turnStart)
    }
}
