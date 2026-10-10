import XCTest
@testable import AssistantKit

final class DraftPolicyTests: XCTestCase {
    private func proposal(_ source: DraftSource = .promptLookup, length: Int = 8, match: Int = 3) -> DraftProposal {
        DraftProposal(tokens: Array(100..<(100 + length)), source: source, matchLength: match)
    }

    private func policy(prior: Double = 0.6, curve: CostCurve = .stockMLXDefault, _ configure: (inout DraftPolicy.Configuration) -> Void = { _ in }) -> DraftPolicy {
        var configuration = DraftPolicy.Configuration()
        configuration.prior = prior
        configure(&configuration)
        return DraftPolicy(curve: curve, configuration: configuration)
    }

    func testDefaults() {
        let configuration = DraftPolicy.Configuration()
        XCTAssertEqual(configuration.maxDraft, 4)
        XCTAssertEqual(configuration.roundOverhead, 0.04)
        XCTAssertEqual(configuration.margin, 0.05)
        XCTAssertEqual(configuration.backoffTokens, 16)
        XCTAssertEqual(configuration.maxBackoff, 256)
        XCTAssertEqual(configuration.prior, 0.6)
        XCTAssertEqual(configuration.ewma, 0.3)
        XCTAssertEqual(configuration.minPromptLookupMatch, 3)
        XCTAssertEqual(configuration.shortMatchPrior, 0.3)
        let policy = DraftPolicy(curve: .stockMLXDefault)
        XCTAssertEqual(policy.maxDraft, 4)
        XCTAssertTrue(policy.isSpeculationEnabled)
        XCTAssertEqual(policy.acceptance(source: .promptLookup, matchLength: 3, depth: 1), 0.6)
        XCTAssertEqual(policy.acceptance(source: .promptLookup, matchLength: 2, depth: 1), 0.3)
        XCTAssertEqual(policy.acceptance(source: .corpus, matchLength: 1, depth: 2), 0.3)
        XCTAssertEqual(policy.acceptance(source: .exemplar, matchLength: 1, depth: 1), 0.6)
        XCTAssertEqual(policy.acceptance(source: .draftModel, matchLength: 0, depth: 3), 0.6)
    }

    /// Value(K) = E(K) / (c(K+1) + K·draftCost + 0.04) with a constant per-depth acceptance a, on
    /// the stock curve. Hand-computed:
    ///
    /// | a    | draftCost | V(1)   | V(2)   | V(3)   | V(4)   | K* |
    /// |------|-----------|--------|--------|--------|--------|----|
    /// | 0.10 | 0         | 1.0092 | 0.7986 | 0.6653 | 0.5583 | 0  |
    /// | 0.20 | 0         | 1.1009 | 0.8921 | 0.7473 | 0.6279 | 1  |
    /// | 0.60 | 0         | 1.4679 | 1.4101 | 1.3030 | 1.1586 | 1  |
    /// | 0.70 | 0         | 1.5596 | 1.5755 | 1.5168 | 1.3935 | 2  |
    /// | 0.80 | 0         | 1.6514 | 1.7554 | 1.7677 | 1.6892 | 3  |
    /// | 0.90 | 0         | 1.7431 | 1.9496 | 2.0593 | 2.0578 | 3  |
    /// | 0.95 | 0         | 1.7890 | 2.0522 | 2.2215 | 2.2736 | 4  |
    /// | 0.30 | 0.15      | 1.0484 | 0.8225 | 0.6684 | 0.5502 | 0  |
    /// | 0.50 | 0.15      | 1.2097 | 1.0355 | 0.8844 | 0.7481 | 1  |
    /// | 0.80 | 0.15      | 1.4516 | 1.4438 | 1.3925 | 1.2979 | 1  |
    /// | 0.85 | 0.15      | 1.4919 | 1.5222 | 1.5031 | 1.4319 | 2  |
    /// | 0.90 | 0.15      | 1.5323 | 1.6036 | 1.6222 | 1.5811 | 3  |
    /// | 1.00 | 0.15      | 1.6129 | 1.7751 | 1.8868 | 1.9305 | 4  |
    func testDraftLengthMatchesHandComputedTable() {
        let table: [(a: Double, cost: Double, k: Int)] = [
            (0.10, 0, 0), (0.20, 0, 1), (0.60, 0, 1), (0.70, 0, 2), (0.80, 0, 3), (0.90, 0, 3), (0.95, 0, 4),
            (0.30, 0.15, 0), (0.50, 0.15, 1), (0.80, 0.15, 1), (0.85, 0.15, 2), (0.90, 0.15, 3), (1.00, 0.15, 4),
        ]
        for row in table {
            let policy = policy(prior: row.a)
            XCTAssertEqual(policy.draftLength(for: proposal(), remaining: 100, draftCostPerToken: row.cost), row.k, "a \(row.a) cost \(row.cost)")
            XCTAssertEqual(policy.draftLength(for: proposal(.draftModel, match: 0), remaining: 100, draftCostPerToken: row.cost), row.k)
        }
    }

    func testExpectedTokens() {
        let policy = policy(prior: 0.6)
        XCTAssertEqual(policy.expectedTokens(source: .promptLookup, matchLength: 3, k: 0), 1)
        XCTAssertEqual(policy.expectedTokens(source: .promptLookup, matchLength: 3, k: 2), 1.96, accuracy: 1e-12)
        XCTAssertEqual(policy.expectedTokens(source: .promptLookup, matchLength: 5, k: 4), 2.3056, accuracy: 1e-12)
        // Prompt-lookup matches shorter than 3 never speculate by default, so they promise no gain.
        XCTAssertEqual(policy.expectedTokens(source: .promptLookup, matchLength: 2, k: 2), 1)
        XCTAssertEqual(policy.expectedTokens(source: .promptLookup, matchLength: 1, k: 4), 1)
        // Short corpus matches are not gated; they start from the lower prior.
        XCTAssertEqual(policy.expectedTokens(source: .corpus, matchLength: 2, k: 2), 1.39, accuracy: 1e-12)
    }

    func testShortPromptLookupMatchesDecodePlainlyByDefault() {
        var policy = DraftPolicy(curve: .stockMLXDefault)
        XCTAssertFalse(policy.isEligible(source: .promptLookup, matchLength: 2))
        XCTAssertTrue(policy.isEligible(source: .promptLookup, matchLength: 3))
        XCTAssertTrue(policy.isEligible(source: .corpus, matchLength: 2))
        XCTAssertTrue(policy.isEligible(source: .exemplar, matchLength: 1))
        XCTAssertTrue(policy.isEligible(source: .draftModel, matchLength: 0))
        // Without the gate, a = 0.3 would give V(1) = 1.3 / 1.09 = 1.1927 > 1.05.
        XCTAssertEqual(policy.draftLength(for: proposal(match: 2), remaining: 100, draftCostPerToken: 0), 0)
        XCTAssertEqual(policy.draftLength(for: proposal(match: 1), remaining: 100, draftCostPerToken: 0), 0)
        XCTAssertEqual(policy.draftLength(for: proposal(match: 3), remaining: 100, draftCostPerToken: 0), 1)
        // The gate is not learned away: short matches stay plain however well they did.
        for _ in 0..<20 { policy.record(proposal(match: 2), proposed: 4, accepted: 4) }
        XCTAssertEqual(policy.draftLength(for: proposal(match: 2), remaining: 100, draftCostPerToken: 0), 0)
        XCTAssertEqual(policy.expectedTokens(source: .promptLookup, matchLength: 2, k: 4), 1)
        // Short matches of other drafters still speculate: a = 0.3, V(1) = 1.1927, V(2) = 1.39 / 1.39 = 1.
        XCTAssertEqual(policy.draftLength(for: proposal(.corpus, match: 2), remaining: 100, draftCostPerToken: 0), 1)
    }

    func testLoweringTheGateAppliesTheShortMatchPrior() {
        let policy = policy { $0.minPromptLookupMatch = 2 }
        XCTAssertTrue(policy.isEligible(source: .promptLookup, matchLength: 2))
        // a = 0.3: V(1) = 1.3 / 1.09 = 1.1927, V(2) = 1.39 / 1.39 = 1.
        XCTAssertEqual(policy.draftLength(for: proposal(match: 2), remaining: 100, draftCostPerToken: 0), 1)
        XCTAssertEqual(policy.expectedTokens(source: .promptLookup, matchLength: 2, k: 2), 1.39, accuracy: 1e-12)
        // With a draft cost of 0.2, V(1) = 1.3 / 1.29 = 1.0078: not worth it.
        XCTAssertEqual(policy.draftLength(for: proposal(match: 2), remaining: 100, draftCostPerToken: 0.2), 0)
        // Below the lowered gate, still plain.
        XCTAssertEqual(policy.draftLength(for: proposal(match: 1), remaining: 100, draftCostPerToken: 0), 0)
        XCTAssertEqual(policy.expectedTokens(source: .promptLookup, matchLength: 1, k: 2), 1)
    }

    func testCurveShapesTheChoice() {
        let flat = CostCurve(seconds: [1: 1, 9: 1])
        let linear = CostCurve(seconds: [1: 1, 9: 9])
        // Flat: V(4) = 2.3056 / 1.04 = 2.2169 is the best.
        XCTAssertEqual(policy(curve: flat).draftLength(for: proposal(), remaining: 100, draftCostPerToken: 0), 4)
        // Flat but costly drafts: V(1) = 1.6 / 1.54 = 1.039 does not clear the margin.
        XCTAssertEqual(policy(curve: flat).draftLength(for: proposal(), remaining: 100, draftCostPerToken: 0.5), 0)
        // Linear cost: speculation never pays, whatever the acceptance.
        XCTAssertEqual(policy(prior: 0.99, curve: linear).draftLength(for: proposal(), remaining: 100, draftCostPerToken: 0), 0)
    }

    func testNeverExceedsTheCaps() {
        let flat = policy(prior: 0.95, curve: CostCurve(seconds: [1: 1, 9: 1]))
        XCTAssertEqual(flat.draftLength(for: proposal(), remaining: 100, draftCostPerToken: 0), 4)
        XCTAssertEqual(flat.draftLength(for: proposal(), remaining: 3, draftCostPerToken: 0), 2)
        XCTAssertEqual(flat.draftLength(for: proposal(), remaining: 1, draftCostPerToken: 0), 0)
        XCTAssertEqual(flat.draftLength(for: proposal(), remaining: 0, draftCostPerToken: 0), 0)
        XCTAssertEqual(flat.draftLength(for: proposal(length: 2), remaining: 100, draftCostPerToken: 0), 2)
        XCTAssertEqual(flat.draftLength(for: proposal(length: 0), remaining: 100, draftCostPerToken: 0), 0)
        XCTAssertEqual(flat.draftLength(for: proposal(), remaining: 100, draftCostPerToken: .nan), 0)
        XCTAssertEqual(flat.draftLength(for: proposal(), remaining: 100, draftCostPerToken: -1), 4)
    }

    func testMaxDraftAboveFourNeedsACheapNineRowForward() {
        func configured(_ maxDraft: Int, _ curve: CostCurve) -> DraftPolicy {
            policy(prior: 0.99, curve: curve) { $0.maxDraft = maxDraft }
        }
        // Stock: c(9)/c(1) = 3.4 > 2, so Kmax stays 4.
        XCTAssertEqual(configured(8, .stockMLXDefault).maxDraft, 4)
        let cheap = CostCurve(seconds: [1: 1, 4: 1.3, 9: 2])
        XCTAssertEqual(configured(8, cheap).maxDraft, 8)
        XCTAssertEqual(configured(6, cheap).maxDraft, 6)
        XCTAssertEqual(configured(12, cheap).maxDraft, 8)
        XCTAssertEqual(configured(2, .stockMLXDefault).maxDraft, 2)
        XCTAssertEqual(configured(-1, cheap).maxDraft, 0)
        XCTAssertEqual(configured(8, cheap).draftLength(for: proposal(length: 10), remaining: 100, draftCostPerToken: 0), 8)
        XCTAssertEqual(configured(8, .stockMLXDefault).draftLength(for: proposal(length: 10), remaining: 100, draftCostPerToken: 0), 4)
        XCTAssertEqual(configured(0, cheap).draftLength(for: proposal(), remaining: 100, draftCostPerToken: 0), 0)
    }

    func testRecordUpdatesPerDepthAverages() {
        var policy = DraftPolicy(curve: .stockMLXDefault)
        policy.record(proposal(), proposed: 2, accepted: 2)
        XCTAssertEqual(policy.acceptance(source: .promptLookup, matchLength: 3, depth: 1), 0.72, accuracy: 1e-12)
        XCTAssertEqual(policy.acceptance(source: .promptLookup, matchLength: 3, depth: 2), 0.72, accuracy: 1e-12)
        XCTAssertEqual(policy.acceptance(source: .promptLookup, matchLength: 3, depth: 3), 0.6, accuracy: 1e-12)
        // Accepted 1 of 3: depth 1 succeeds, depth 2 fails, depth 3 is not observed.
        policy.record(proposal(), proposed: 3, accepted: 1)
        XCTAssertEqual(policy.acceptance(source: .promptLookup, matchLength: 3, depth: 1), 0.804, accuracy: 1e-12)
        XCTAssertEqual(policy.acceptance(source: .promptLookup, matchLength: 3, depth: 2), 0.504, accuracy: 1e-12)
        XCTAssertEqual(policy.acceptance(source: .promptLookup, matchLength: 3, depth: 3), 0.6, accuracy: 1e-12)
        // Other buckets and sources are untouched.
        XCTAssertEqual(policy.acceptance(source: .promptLookup, matchLength: 4, depth: 1), 0.6)
        XCTAssertEqual(policy.acceptance(source: .corpus, matchLength: 3, depth: 1), 0.6)
        // Match lengths of 4 and more share a bucket.
        policy.record(proposal(match: 9), proposed: 1, accepted: 0)
        XCTAssertEqual(policy.acceptance(source: .promptLookup, matchLength: 4, depth: 1), 0.42, accuracy: 1e-12)
        // Out-of-range counts are clamped; nothing proposed records nothing.
        policy.record(proposal(.corpus), proposed: 0, accepted: 0)
        XCTAssertEqual(policy.acceptance(source: .corpus, matchLength: 3, depth: 1), 0.6)
        policy.record(proposal(.corpus), proposed: 1, accepted: 5)
        XCTAssertEqual(policy.acceptance(source: .corpus, matchLength: 3, depth: 1), 0.72, accuracy: 1e-12)
    }

    func testRoundEndingAtAnAcceptedStopDraftRecordsNoRejection() {
        let stop = 99
        func resolve(_ drafts: [Int], _ sampled: [Int]) -> RoundResult {
            AcceptanceRule.resolve(drafts: drafts, sampled: sampled, stopTokens: [stop], remaining: 100)
        }
        func rates(_ policy: DraftPolicy) -> [Double] {
            (1...4).map { policy.acceptance(source: .promptLookup, matchLength: 3, depth: $0) }
        }

        // The reply ends at the accepted stop draft; the target even agreed with the draft after it.
        let atStop = resolve([10, stop, 12, 13], [10, stop, 12, 7, 8])
        XCTAssertTrue(atStop.stoppedOnDraft)
        XCTAssertEqual(atStop.acceptedDrafts, 2)
        var policy = DraftPolicy(curve: .stockMLXDefault)
        policy.record(proposal(), proposed: 4, round: atStop)
        for (rate, expected) in zip(rates(policy), [0.72, 0.72, 0.6, 0.6]) { XCTAssertEqual(rate, expected, accuracy: 1e-12) }
        // Passing the bare counts would score depth 3 as a rejection.
        var counted = DraftPolicy(curve: .stockMLXDefault)
        counted.record(proposal(), proposed: 4, accepted: atStop.acceptedDrafts)
        XCTAssertEqual(counted.acceptance(source: .promptLookup, matchLength: 3, depth: 3), 0.42, accuracy: 1e-12)

        // A stop sampled in place of draft 2 is a real rejection of draft 2.
        let corrected = resolve([10, 11, 12], [10, stop, 12, 13])
        XCTAssertTrue(corrected.stopped)
        XCTAssertFalse(corrected.stoppedOnDraft)
        var other = DraftPolicy(curve: .stockMLXDefault)
        other.record(proposal(), proposed: 3, round: corrected)
        for (rate, expected) in zip(rates(other), [0.72, 0.42, 0.6, 0.6]) { XCTAssertEqual(rate, expected, accuracy: 1e-12) }
        // Rounds without a stop record exactly as the counts say.
        other.record(proposal(), proposed: 2, round: resolve([10, 11], [10, 11, 12]))
        other.record(proposal(), proposed: 2, round: resolve([10, 11], [10, 5, 12]))
        var same = DraftPolicy(curve: .stockMLXDefault)
        same.record(proposal(), proposed: 3, accepted: 1)
        same.record(proposal(), proposed: 2, accepted: 2)
        same.record(proposal(), proposed: 2, accepted: 1)
        for (rate, expected) in zip(rates(other), rates(same)) { XCTAssertEqual(rate, expected, accuracy: 1e-12) }
    }

    func testLearnedAcceptanceRaisesK() {
        var policy = DraftPolicy(curve: .stockMLXDefault)
        XCTAssertEqual(policy.draftLength(for: proposal(), remaining: 100, draftCostPerToken: 0), 1)
        for _ in 0..<20 { policy.record(proposal(), proposed: 4, accepted: 4) }
        // a ≈ 1 at every depth: V(4) = 5 / 1.99 = 2.51 is the best.
        XCTAssertEqual(policy.draftLength(for: proposal(), remaining: 100, draftCostPerToken: 0), 4)
        for _ in 0..<20 { policy.record(proposal(), proposed: 1, accepted: 0) }
        // a(1) ≈ 0 now, but the backoff decides first.
        XCTAssertTrue(policy.isBackedOff(.promptLookup))
        policy.didDecodePlain(1_000)
        XCTAssertEqual(policy.draftLength(for: proposal(), remaining: 100, draftCostPerToken: 0), 0)
    }

    func testBackoffDoublesAndRecovers() {
        var policy = DraftPolicy(curve: .stockMLXDefault)
        let lookup = proposal()

        func zeroRounds(_ count: Int) {
            for _ in 0..<count { policy.record(lookup, proposed: 1, accepted: 0) }
        }

        zeroRounds(2)
        XCTAssertFalse(policy.isBackedOff(.promptLookup))
        zeroRounds(1)
        XCTAssertEqual(policy.backoffRemaining(for: .promptLookup), 16)
        XCTAssertEqual(policy.draftLength(for: lookup, remaining: 100, draftCostPerToken: 0), 0)
        // Other sources keep drafting.
        XCTAssertFalse(policy.isBackedOff(.corpus))
        XCTAssertEqual(policy.draftLength(for: proposal(.corpus), remaining: 100, draftCostPerToken: 0), 1)
        policy.didDecodePlain(15)
        XCTAssertEqual(policy.backoffRemaining(for: .promptLookup), 1)
        policy.didDecodePlain(1)
        XCTAssertFalse(policy.isBackedOff(.promptLookup))

        // Each further backoff without an accepted draft doubles, up to 256.
        var lengths: [Int] = []
        for _ in 0..<6 {
            zeroRounds(3)
            lengths.append(policy.backoffRemaining(for: .promptLookup))
            policy.didDecodePlain(1_000)
        }
        XCTAssertEqual(lengths, [32, 64, 128, 256, 256, 256])

        // An accepted draft resets the doubling and the streak.
        policy.record(lookup, proposed: 2, accepted: 1)
        zeroRounds(2)
        policy.record(lookup, proposed: 2, accepted: 2)
        zeroRounds(2)
        XCTAssertFalse(policy.isBackedOff(.promptLookup))
        zeroRounds(1)
        XCTAssertEqual(policy.backoffRemaining(for: .promptLookup), 16)

        // Plain tokens of zero or less change nothing.
        policy.didDecodePlain(0)
        policy.didDecodePlain(-5)
        XCTAssertEqual(policy.backoffRemaining(for: .promptLookup), 16)
    }

    func testThermalAndLowPower() {
        var policy = DraftPolicy(curve: .stockMLXDefault)
        // A short corpus match, a = 0.3 with draft cost 0.08: V(1) = 1.3 / 1.17 = 1.111, between
        // 1.05 and 1.15.
        let short = proposal(.corpus, match: 2)
        XCTAssertEqual(policy.draftLength(for: short, remaining: 100, draftCostPerToken: 0.08), 1)
        policy.adjust(thermal: .fair, lowPower: false)
        XCTAssertEqual(policy.draftLength(for: short, remaining: 100, draftCostPerToken: 0.08), 1)
        policy.adjust(thermal: .serious, lowPower: false)
        XCTAssertEqual(policy.effectiveMargin, 0.15, accuracy: 1e-12)
        XCTAssertEqual(policy.draftLength(for: short, remaining: 100, draftCostPerToken: 0.08), 0)
        // A clearly better proposal still speculates when seriously hot.
        XCTAssertEqual(policy.draftLength(for: proposal(), remaining: 100, draftCostPerToken: 0), 1)
        policy.adjust(thermal: .critical, lowPower: false)
        XCTAssertFalse(policy.isSpeculationEnabled)
        XCTAssertEqual(policy.draftLength(for: proposal(), remaining: 100, draftCostPerToken: 0), 0)
        policy.adjust(thermal: .nominal, lowPower: true)
        XCTAssertFalse(policy.isSpeculationEnabled)
        XCTAssertEqual(policy.draftLength(for: proposal(), remaining: 100, draftCostPerToken: 0), 0)
        policy.adjust(thermal: .nominal, lowPower: false)
        XCTAssertTrue(policy.isSpeculationEnabled)
        XCTAssertEqual(policy.draftLength(for: short, remaining: 100, draftCostPerToken: 0.08), 1)
    }
}
