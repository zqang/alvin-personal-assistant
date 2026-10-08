import Foundation

/// Prompt-lookup drafting: finds the session's latest n tokens earlier in the session and proposes
/// what followed them there. Replies that repeat the prompt (edits, quotes, tool arguments copied
/// from a tool result) get long runs of accepted drafts this way, at no model cost.
///
/// The index is incremental: appending a token indexes the n-grams it completes, and `truncate`
/// undoes exactly the entries of the removed tokens, so after any sequence of appends and
/// truncations it proposes the same as an index built from scratch over the same tokens.
public struct NGramIndex: Sendable {
    /// Shortest suffix that may match.
    public let minN: Int
    /// Longest suffix that is tried first.
    public let maxN: Int
    /// Tokens a continuation never includes (e.g. `<|im_start|>`, role tokens, `<|endoftext|>`).
    public let excluded: Set<Int>
    /// The indexed tokens, in order.
    public private(set) var tokens: [Int] = []

    /// `tables[n - minN]` maps the hash of an n-gram to the start positions (ascending) of its
    /// occurrences that are followed by at least one token: an n-gram starting at `s` is indexed
    /// once `s + n < count`. Entries are added and removed in stack order.
    private var tables: [[UInt64: [Int]]]

    public init(minN: Int = 2, maxN: Int = 4, excluded: Set<Int>) {
        let lower = max(1, minN)
        self.minN = lower
        self.maxN = max(lower, maxN)
        self.excluded = excluded
        tables = Array(repeating: [:], count: self.maxN - lower + 1)
    }

    /// The number of indexed tokens.
    public var count: Int { tokens.count }

    public mutating func append<S: Sequence>(contentsOf newTokens: S) where S.Element == Int {
        for token in newTokens {
            tokens.append(token)
            // The n-grams that end just before the new token now have a continuation.
            let end = tokens.count - 1
            for n in minN...maxN where end - n >= 0 {
                let hash = Self.hash(tokens, end: end, n: n)
                tables[n - minN][hash, default: []].append(end - n)
            }
        }
    }

    /// Keeps the first `count` tokens (no-op when `count` is not smaller than the current count).
    public mutating func truncate(to count: Int) {
        let target = max(0, count)
        guard target < tokens.count else { return }
        // Undo `append` in reverse order: the n-grams whose continuation is a removed token.
        for end in stride(from: tokens.count - 1, through: target, by: -1) {
            for n in stride(from: maxN, through: minN, by: -1) where end - n >= 0 {
                let hash = Self.hash(tokens, end: end, n: n)
                let table = n - minN
                guard var starts = tables[table].removeValue(forKey: hash) else {
                    assertionFailure("NGramIndex: missing entry")
                    continue
                }
                let removed = starts.removeLast()
                assert(removed == end - n, "NGramIndex: entries out of order")
                if !starts.isEmpty { tables[table][hash] = starts }
            }
        }
        tokens.removeLast(tokens.count - target)
    }

    /// Makes the index hold exactly `ledger`, keeping the work proportional to what changed: it
    /// truncates to the common prefix and appends the rest.
    public mutating func sync(to ledger: [Int]) {
        var common = 0
        let limit = min(ledger.count, tokens.count)
        while common < limit, ledger[common] == tokens[common] { common += 1 }
        truncate(to: common)
        if common < ledger.count { append(contentsOf: ledger[common...]) }
    }

    /// The continuation of the longest suffix n-gram (`maxN` down to `minN`) that occurred earlier;
    /// the most recent earlier occurrence wins. The continuation stops at an excluded token, at the
    /// current end, or after `maxTokens` tokens. Nil when nothing matches or the winning
    /// occurrence is followed by an excluded token.
    public func propose(maxTokens: Int) -> DraftProposal? {
        guard maxTokens > 0 else { return nil }
        let m = tokens.count
        for n in stride(from: maxN, through: minN, by: -1) where m >= n + 1 {
            guard let starts = tables[n - minN][Self.hash(tokens, end: m, n: n)] else { continue }
            let suffix = tokens[(m - n)..<m]
            guard let start = starts.last(where: { tokens[$0..<($0 + n)] == suffix }) else { continue }
            var draft: [Int] = []
            var j = start + n
            while draft.count < maxTokens, j < m, !excluded.contains(tokens[j]) {
                draft.append(tokens[j])
                j += 1
            }
            return draft.isEmpty ? nil : DraftProposal(tokens: draft, source: .promptLookup, matchLength: n)
        }
        return nil
    }

    /// Hash of `tokens[(end - n)..<end]`, folded from the end backwards.
    private static func hash(_ tokens: [Int], end: Int, n: Int) -> UInt64 {
        var hash = DraftTokenHash.seed
        for i in stride(from: end - 1, through: end - n, by: -1) {
            hash = DraftTokenHash.mix(hash, tokens[i])
        }
        return hash
    }
}
