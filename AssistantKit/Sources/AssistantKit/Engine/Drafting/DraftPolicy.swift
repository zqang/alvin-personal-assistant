import Foundation

/// Decides how many drafted tokens to verify in the next round, if any.
///
/// For K drafts the round verifies K + 1 rows and yields E(K) tokens on average, where
/// E(K) = 1 + Σ_{j≤K} Π_{i≤j} a(i) and a(i) is the acceptance rate at depth i given that every
/// earlier draft was accepted. The policy picks the K that maximizes
/// E(K) / (c(K+1) + K·draftCost + roundOverhead) and speculates only when that beats plain
/// decoding (worth 1 token per unit of cost) by more than the margin. Acceptance rates are
/// exponentially weighted averages kept per source, per match-length bucket and per depth.
///
/// Prompt-lookup matches shorter than `Configuration.minPromptLookupMatch` (3 by default) never
/// speculate: measured on the target model, their first draft is accepted only about 43% of the
/// time, against about 90% for matches of 3 or more.
///
/// The decision depends only on the cost curve and the recorded rounds, never on wall time.
public struct DraftPolicy: Sendable {
    public struct Configuration: Sendable {
        /// Most drafts per round. Values above 4 apply only when the curve gives c(9)/c(1) ≤ 2, and
        /// never above 8.
        public var maxDraft = 4
        /// Fixed cost of a speculative round (pipeline drain, host sync), relative to one row.
        public var roundOverhead = 0.04
        /// Speculate only when the best value exceeds 1 + margin.
        public var margin = 0.05
        /// Plain tokens of the first backoff; each further backoff without an accepted draft in
        /// between doubles it.
        public var backoffTokens = 16
        /// Longest backoff, in plain tokens.
        public var maxBackoff = 256
        /// Prior acceptance per depth for matches of 3 or more tokens, and for exemplar, MTP and
        /// draft-model proposals.
        public var prior = 0.6
        /// Shortest prompt-lookup match that may speculate. Shorter matches always decode plainly
        /// and promise no gain in `expectedTokens`.
        public var minPromptLookupMatch = 3
        /// Prior acceptance per depth for corpus matches shorter than 3 tokens, and for prompt-lookup
        /// matches shorter than 3 when `minPromptLookupMatch` lets them speculate.
        public var shortMatchPrior = 0.3
        /// Weight of each new observation in the acceptance averages.
        public var ewma = 0.3
        /// Rounds in a row with no accepted draft that back a source off.
        public var zeroAcceptanceRounds = 3

        public init() {}
    }

    public let curve: CostCurve
    public let configuration: Configuration
    /// The effective most drafts per round (see `Configuration.maxDraft`).
    public let maxDraft: Int
    public private(set) var thermal: ThermalLevel = .nominal
    public private(set) var lowPower = false

    private struct Bucket: Hashable, Sendable {
        var source: DraftSource
        /// 1, 2, 3, or 4 for 4 and more matched tokens.
        var length: Int
    }

    private struct Backoff: Sendable {
        var zeroStreak = 0
        /// Backoffs since the last accepted draft; sets the doubling.
        var level = 0
        var remaining = 0
    }

    /// `costs[s]` is c(s) relative to one row, for s in 0...maxDraft + 1 (index 0 unused).
    private let costs: [Double]
    private var acceptance: [Bucket: [Double]] = [:]
    private var backoffs: [DraftSource: Backoff] = [:]

    public init(curve: CostCurve, configuration: Configuration = .init()) {
        self.curve = curve
        self.configuration = configuration
        let requested = max(0, configuration.maxDraft)
        if requested <= 4 {
            maxDraft = requested
        } else {
            maxDraft = curve.relative(9) <= 2 ? min(requested, 8) : 4
        }
        costs = (0...(maxDraft + 1)).map { curve.relative(max($0, 1)) }
    }

    /// False while the device is critically hot or in Low Power Mode.
    public var isSpeculationEnabled: Bool {
        !(thermal == .critical || lowPower)
    }

    /// The margin in force: tripled while the device is seriously hot.
    public var effectiveMargin: Double {
        thermal == .serious ? configuration.margin * 3 : configuration.margin
    }

    /// Whether proposals from `source` with this match length may speculate at all. Only prompt
    /// lookup is gated, by `Configuration.minPromptLookupMatch`.
    public func isEligible(source: DraftSource, matchLength: Int) -> Bool {
        source != .promptLookup || matchLength >= configuration.minPromptLookupMatch
    }

    /// The number of drafts to verify next (0 means a plain step). Never more than
    /// `min(maxDraft, remaining − 1, proposal.tokens.count)`.
    /// - Parameters:
    ///   - remaining: tokens the reply may still emit.
    ///   - draftCostPerToken: the drafter's cost per drafted token, relative to one target row.
    public func draftLength(for proposal: DraftProposal, remaining: Int, draftCostPerToken: Double) -> Int {
        guard isSpeculationEnabled,
              isEligible(source: proposal.source, matchLength: proposal.matchLength),
              !isBackedOff(proposal.source),
              draftCostPerToken.isFinite
        else { return 0 }
        let cap = min(maxDraft, remaining - 1, proposal.tokens.count)
        guard cap >= 1 else { return 0 }
        let draftCost = max(0, draftCostPerToken)
        let overhead = max(0, configuration.roundOverhead)
        let bucket = Self.bucket(proposal.source, proposal.matchLength)
        var best = 0
        var bestValue = 1 + effectiveMargin
        var expected = 1.0
        var running = 1.0
        for k in 1...cap {
            running *= rate(bucket, depth: k)
            expected += running
            let value = expected / (costs[k + 1] + Double(k) * draftCost + overhead)
            if value > bestValue {
                best = k
                bestValue = value
            }
        }
        return best
    }

    /// E(k): the expected tokens of a round with `k` drafts from this source and match length,
    /// counting the bonus token. Use it to compare proposals from different drafters. A proposal
    /// that may not speculate (see `isEligible`) gets 1, the yield of a plain step.
    public func expectedTokens(source: DraftSource, matchLength: Int, k: Int) -> Double {
        guard k >= 1, isEligible(source: source, matchLength: matchLength) else { return 1 }
        let bucket = Self.bucket(source, matchLength)
        var expected = 1.0
        var running = 1.0
        for depth in 1...k {
            running *= rate(bucket, depth: depth)
            expected += running
        }
        return expected
    }

    /// The current acceptance estimate at `depth` (1-based), given every earlier draft was accepted.
    public func acceptance(source: DraftSource, matchLength: Int, depth: Int) -> Double {
        rate(Self.bucket(source, matchLength), depth: max(1, depth))
    }

    /// Records a round resolved by `AcceptanceRule.resolve` from `proposed` drafts of `proposal`.
    /// Use this rather than `record(_:proposed:accepted:)` with `round.acceptedDrafts`: when the
    /// round ended at an accepted stop draft (`RoundResult.stoppedOnDraft`), the drafts after the
    /// stop are left out instead of counting as a rejection.
    public mutating func record(_ proposal: DraftProposal, proposed: Int, round: RoundResult) {
        let observed = round.stoppedOnDraft ? min(proposed, round.acceptedDrafts) : proposed
        record(proposal, proposed: observed, accepted: round.acceptedDrafts)
    }

    /// Records a verified round: `proposed` drafts of `proposal` were verified and the first
    /// `accepted` of them matched. Depths up to `accepted` count as accepted and depth
    /// `accepted + 1`, if proposed, as rejected; deeper drafts were not observed.
    ///
    /// A round that ended at an accepted stop draft had no rejection, and `acceptedDrafts` does not
    /// say whether the drafts after the stop matched. For such a round pass `proposed: accepted`, or
    /// use `record(_:proposed:round:)`, which does so.
    public mutating func record(_ proposal: DraftProposal, proposed: Int, accepted: Int) {
        guard proposed > 0 else { return }
        let accepted = min(max(0, accepted), proposed)
        let bucket = Self.bucket(proposal.source, proposal.matchLength)
        let weight = min(max(configuration.ewma, 0), 1)
        var rates = acceptance[bucket] ?? []
        // Depths past the first rejection were not observed.
        let observed = min(accepted + 1, proposed)
        while rates.count < observed { rates.append(prior(bucket)) }
        for depth in 1...observed {
            let outcome: Double = depth <= accepted ? 1 : 0
            rates[depth - 1] = (1 - weight) * rates[depth - 1] + weight * outcome
        }
        acceptance[bucket] = rates

        var state = backoffs[proposal.source] ?? Backoff()
        if accepted == 0 {
            state.zeroStreak += 1
            if state.zeroStreak >= max(1, configuration.zeroAcceptanceRounds) {
                state.remaining = backoffLength(level: state.level)
                state.level += 1
                state.zeroStreak = 0
            }
        } else {
            state.zeroStreak = 0
            state.level = 0
        }
        backoffs[proposal.source] = state
    }

    /// Counts plain (non-speculative) tokens against every active backoff.
    public mutating func didDecodePlain(_ tokens: Int) {
        guard tokens > 0 else { return }
        for (source, state) in backoffs where state.remaining > 0 {
            backoffs[source]?.remaining = max(0, state.remaining - tokens)
        }
    }

    /// Applies the device state: `serious` triples the margin; `critical` or Low Power Mode turns
    /// speculation off.
    public mutating func adjust(thermal: ThermalLevel, lowPower: Bool) {
        self.thermal = thermal
        self.lowPower = lowPower
    }

    /// Plain tokens left before `source` may draft again.
    public func backoffRemaining(for source: DraftSource) -> Int {
        backoffs[source]?.remaining ?? 0
    }

    public func isBackedOff(_ source: DraftSource) -> Bool {
        backoffRemaining(for: source) > 0
    }

    // MARK: - Private

    private static func bucket(_ source: DraftSource, _ matchLength: Int) -> Bucket {
        Bucket(source: source, length: min(max(matchLength, 1), 4))
    }

    private func prior(_ bucket: Bucket) -> Double {
        switch bucket.source {
        case .promptLookup, .corpus:
            return bucket.length >= 3 ? configuration.prior : configuration.shortMatchPrior
        case .exemplar, .mtp, .draftModel:
            return configuration.prior
        }
    }

    private func rate(_ bucket: Bucket, depth: Int) -> Double {
        if let rates = acceptance[bucket], depth <= rates.count { return rates[depth - 1] }
        return prior(bucket)
    }

    private func backoffLength(level: Int) -> Int {
        let cap = max(0, configuration.maxBackoff)
        var length = min(max(0, configuration.backoffTokens), cap)
        var doublings = level
        while doublings > 0, length < cap {
            length = min(length * 2, cap)
            doublings -= 1
        }
        return length
    }
}
