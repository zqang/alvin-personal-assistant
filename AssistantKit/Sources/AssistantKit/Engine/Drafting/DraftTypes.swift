import Foundation

/// Where a speculative draft came from. The draft policy keeps separate acceptance statistics and
/// backoff state per source.
public enum DraftSource: String, Codable, Sendable, CaseIterable {
    /// N-gram lookup over the live session's tokens (`NGramIndex`).
    case promptLookup
    /// Continuations of past replies and tool results (`SuffixCorpus`).
    case corpus
    /// Tool-call skeletons in the model's format (`ExemplarSeeds`).
    case exemplar
    /// The model's own multi-token-prediction head.
    case mtp
    /// A small draft model with the target's vocabulary.
    case draftModel
}

/// Tokens a drafter guesses will come next.
public struct DraftProposal: Equatable, Sendable {
    /// The guessed tokens, in order. Never empty when produced by the drafting types.
    public var tokens: [Int]
    public var source: DraftSource
    /// How many context tokens matched to produce the guess (0 when it does not apply, as for a
    /// draft model). The draft policy buckets its acceptance statistics by this length.
    public var matchLength: Int

    public init(tokens: [Int], source: DraftSource, matchLength: Int) {
        self.tokens = tokens
        self.source = source
        self.matchLength = matchLength
    }
}

/// The device's thermal state, mirrored from `ProcessInfo.ThermalState` so the policy stays
/// platform independent.
public enum ThermalLevel: Int, Codable, Sendable {
    case nominal, fair, serious, critical
}

/// A 64-bit mix of token ids for the drafting indexes. Every hit is verified against the tokens,
/// so collisions only cost a comparison.
enum DraftTokenHash {
    static let seed: UInt64 = 0x9E37_79B9_7F4A_7C15

    @inline(__always)
    static func mix(_ hash: UInt64, _ token: Int) -> UInt64 {
        var h = (hash ^ UInt64(bitPattern: Int64(token))) &* 0xBF58_476D_1CE4_E5B9
        h ^= h >> 31
        h = h &* 0x94D0_49BB_1331_11EB
        return h ^ (h >> 29)
    }
}
