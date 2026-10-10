import Foundation

/// Drafting from fixed exemplar sequences, such as one tool-call skeleton per tool in the model's
/// format (`<tool_call>\n<function=NAME>\n<parameter=P>\n`). The structure of a tool call is
/// predictable once it starts, so these drafts are accepted at high rates.
///
/// Matching is the same as `SuffixCorpus`: the longest context suffix (at most
/// `SuffixCorpus.maxMatch` tokens) found in a sequence with a following token, continued with the
/// most frequent next token (ties go to the most recent sequence). A match needs 2 tokens, except
/// that an anchor token (like `<tool_call>`) matches on its own.
public struct ExemplarSeeds: Sendable {
    /// Tokens that may match on their own.
    public let anchors: Set<Int>
    private var matcher: ContinuationMatcher

    /// Sequences may contain special tokens. The first token of each sequence is an anchor, so a
    /// skeleton starts drafting right after its opening token is sampled.
    public init(sequences: [[Int]]) {
        self.init(sequences: sequences, anchors: Set(sequences.compactMap(\.first)))
    }

    /// Sequences with an explicit set of anchor tokens (pass an empty set to require 2-token matches).
    public init(sequences: [[Int]], anchors: Set<Int>) {
        self.anchors = anchors
        var matcher = ContinuationMatcher(keyLength: 1, maxMatch: SuffixCorpus.maxMatch)
        for sequence in sequences where sequence.count >= 2 {
            matcher.add(sequence)
        }
        self.matcher = matcher
    }

    /// The sequences kept (those with at least 2 tokens), in order.
    public var sequences: [[Int]] { matcher.documents }

    public func propose(context: ArraySlice<Int>, maxTokens: Int) -> DraftProposal? {
        matcher.propose(context: context, maxTokens: maxTokens, minMatch: 2, anchors: anchors, excluded: [])
            .map { DraftProposal(tokens: $0.tokens, source: .exemplar, matchLength: $0.matchLength) }
    }
}
