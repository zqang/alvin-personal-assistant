import Foundation

/// Where the time to the first token of an engine reply went. The phases run one after another,
/// so `total` is the engine's time to first token.
public struct EnginePhaseTimes: Codable, Equatable, Sendable {
    /// Seconds spent planning the cache reuse (`SessionPlanner`).
    public var plan: TimeInterval
    /// Seconds spent rendering the chat template and tokenizing the pieces.
    public var render: TimeInterval
    /// Seconds spent rewinding the cache (restoring a checkpoint and re-feeding up to the base).
    public var rewind: TimeInterval
    /// Seconds spent prefilling the new tokens.
    public var prefill: TimeInterval
    /// Seconds from the end of the prefill to the first sampled token.
    public var firstToken: TimeInterval
    /// How the cache started: `warm` (kept the live cache), `disk` (loaded the saved system prefix)
    /// or `cold` (empty); see `start(for:)`.
    public var start: String

    public init(plan: TimeInterval = 0, render: TimeInterval = 0, rewind: TimeInterval = 0, prefill: TimeInterval = 0, firstToken: TimeInterval = 0, start: String = "") {
        self.plan = plan
        self.render = render
        self.rewind = rewind
        self.prefill = prefill
        self.firstToken = firstToken
        self.start = start
    }

    public var total: TimeInterval {
        plan + render + rewind + prefill + firstToken
    }

    /// `start` for a plan's base.
    public static func start(for base: SessionPlan.Base) -> String {
        switch base {
        case .keep: return "warm"
        case .persistedPrefix: return "disk"
        case .empty: return "cold"
        }
    }
}

/// Speculative decoding in one reply.
public struct SpeculationStats: Codable, Equatable, Sendable {
    /// Verify rounds (one target forward over `[y] + drafts` each).
    public var rounds: Int
    /// Tokens decoded one at a time, without drafts.
    public var plainTokens: Int
    /// Drafted tokens sent to verification, per draft source (`DraftSource` raw values).
    public var drafted: [String: Int]
    /// Drafted tokens the target accepted, per draft source.
    public var accepted: [String: Int]
    /// Histogram of tokens emitted per round: tokens → rounds.
    public var tokensPerRound: [Int: Int]

    public init(rounds: Int = 0, plainTokens: Int = 0, drafted: [String: Int] = [:], accepted: [String: Int] = [:], tokensPerRound: [Int: Int] = [:]) {
        self.rounds = rounds
        self.plainTokens = plainTokens
        self.drafted = drafted
        self.accepted = accepted
        self.tokensPerRound = tokensPerRound
    }

    /// Counts one verify round.
    public mutating func recordRound(source: String, drafted draftedTokens: Int, accepted acceptedTokens: Int, emitted: Int) {
        rounds += 1
        drafted[source, default: 0] += draftedTokens
        accepted[source, default: 0] += acceptedTokens
        tokensPerRound[emitted, default: 0] += 1
    }

    /// Counts tokens decoded without drafts.
    public mutating func recordPlain(_ tokens: Int = 1) {
        plainTokens += tokens
    }

    public var totalDrafted: Int { drafted.values.reduce(0, +) }
    public var totalAccepted: Int { accepted.values.reduce(0, +) }

    /// Tokens emitted by verify rounds.
    public var roundTokens: Int { tokensPerRound.reduce(0) { $0 + $1.key * $1.value } }

    /// Share of drafted tokens accepted; nil when nothing was drafted.
    public var acceptanceRate: Double? {
        totalDrafted > 0 ? Double(totalAccepted) / Double(totalDrafted) : nil
    }

    /// Mean tokens emitted per verify round; nil without rounds.
    public var meanTokensPerRound: Double? {
        rounds > 0 ? Double(roundTokens) / Double(rounds) : nil
    }
}

/// How sure the model was of the tokens it sampled: the top-1 probability per token.
/// Telemetry only; a confidence-based escalation policy needs on-device calibration first.
public struct TokenConfidence: Codable, Equatable, Sendable {
    public var meanTop1: Double
    /// 10th percentile (nearest rank) of the top-1 probabilities.
    public var p10Top1: Double
    public var tokens: Int

    public init(meanTop1: Double, p10Top1: Double, tokens: Int) {
        self.meanTop1 = meanTop1
        self.p10Top1 = p10Top1
        self.tokens = tokens
    }

    /// Summarizes per-token top-1 probabilities; nil when there are none.
    public init?<S: Sequence>(top1 samples: S) where S.Element: BinaryFloatingPoint {
        let values = samples.map { Double($0) }.filter(\.isFinite).sorted()
        guard !values.isEmpty else { return nil }
        let rank = Int((0.1 * Double(values.count)).rounded(.up))
        self.init(
            meanTop1: values.reduce(0, +) / Double(values.count),
            p10Top1: values[min(values.count - 1, max(0, rank - 1))],
            tokens: values.count
        )
    }
}
