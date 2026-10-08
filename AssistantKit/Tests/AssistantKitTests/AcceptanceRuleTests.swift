import XCTest
@testable import AssistantKit

final class AcceptanceRuleTests: XCTestCase {
    private let stops: Set<Int> = [99]

    private func resolve(_ drafts: [Int], _ sampled: [Int], remaining: Int = 100) -> RoundResult {
        AcceptanceRule.resolve(drafts: drafts, sampled: sampled, stopTokens: stops, remaining: remaining)
    }

    // MARK: - Truth table

    func testAllDraftsAccepted() {
        XCTAssertEqual(resolve([10, 11, 12], [10, 11, 12, 13]), RoundResult(emitted: [10, 11, 12, 13], acceptedDrafts: 3, keep: 4, stopped: false))
    }

    func testRejectAtFirstDraft() {
        XCTAssertEqual(resolve([10, 11], [20, 11, 12]), RoundResult(emitted: [20], acceptedDrafts: 0, keep: 1, stopped: false))
    }

    func testRejectMidway() {
        XCTAssertEqual(resolve([10, 11, 12], [10, 21, 12, 13]), RoundResult(emitted: [10, 21], acceptedDrafts: 1, keep: 2, stopped: false))
        // A later row that happens to agree does not count after a rejection.
        XCTAssertEqual(resolve([10, 11, 12], [10, 11, 30, 13]), RoundResult(emitted: [10, 11, 30], acceptedDrafts: 2, keep: 3, stopped: false))
    }

    func testAcceptedStopDraftEndsTheRoundWithTheStopKept() {
        XCTAssertEqual(resolve([10, 99, 12], [10, 99, 12, 13]), RoundResult(emitted: [10], acceptedDrafts: 2, keep: 3, stopped: true))
        XCTAssertEqual(resolve([99, 1], [99, 1, 2]), RoundResult(emitted: [], acceptedDrafts: 1, keep: 2, stopped: true))
        // A rejected stop draft is just a rejected draft.
        XCTAssertEqual(resolve([99], [5, 6]), RoundResult(emitted: [5], acceptedDrafts: 0, keep: 1, stopped: false))
    }

    func testStopAsBonusOrCorrection() {
        XCTAssertEqual(resolve([10, 11], [10, 11, 99]), RoundResult(emitted: [10, 11], acceptedDrafts: 2, keep: 3, stopped: true))
        XCTAssertEqual(resolve([10, 11], [10, 99, 5]), RoundResult(emitted: [10], acceptedDrafts: 1, keep: 2, stopped: true))
        XCTAssertEqual(resolve([10], [99, 5]), RoundResult(emitted: [], acceptedDrafts: 0, keep: 1, stopped: true))
    }

    func testRemainingCapsEmittedTokens() {
        XCTAssertEqual(resolve([10, 11, 12], [10, 11, 12, 13], remaining: 2), RoundResult(emitted: [10, 11], acceptedDrafts: 3, keep: 2, stopped: false))
        XCTAssertEqual(resolve([10, 11, 12], [10, 11, 12, 13], remaining: 0), RoundResult(emitted: [], acceptedDrafts: 3, keep: 1, stopped: false))
        // A stop beyond the cap does not end the reply as a stop.
        XCTAssertEqual(resolve([10, 11, 99], [10, 11, 99, 1], remaining: 1), RoundResult(emitted: [10], acceptedDrafts: 3, keep: 1, stopped: false))
        // When everything fits, the cap changes nothing.
        XCTAssertEqual(resolve([10, 11], [10, 11, 99], remaining: 2), RoundResult(emitted: [10, 11], acceptedDrafts: 2, keep: 3, stopped: true))
        XCTAssertEqual(resolve([10, 11], [10, 11, 12], remaining: 3), RoundResult(emitted: [10, 11, 12], acceptedDrafts: 2, keep: 3, stopped: false))
    }

    func testStoppedOnDraftTellsAnAcceptedStopDraftFromASampledStop() {
        // An accepted stop draft, even when the target also agrees with the drafts after it.
        XCTAssertTrue(resolve([10, 99, 12], [10, 99, 12, 13]).stoppedOnDraft)
        XCTAssertTrue(resolve([99, 1], [99, 1, 2]).stoppedOnDraft)
        XCTAssertTrue(resolve([10, 99], [10, 99, 99]).stoppedOnDraft)
        // A stop the target sampled, as the bonus or in place of a rejected draft.
        XCTAssertFalse(resolve([10, 11], [10, 11, 99]).stoppedOnDraft)
        XCTAssertFalse(resolve([10, 11], [10, 99, 5]).stoppedOnDraft)
        XCTAssertFalse(resolve([10], [99, 5]).stoppedOnDraft)
        XCTAssertFalse(resolve([], [99]).stoppedOnDraft)
        // No stop at all, or a rejected stop draft.
        XCTAssertFalse(resolve([10, 11, 12], [10, 11, 12, 13]).stoppedOnDraft)
        XCTAssertFalse(resolve([10, 11, 12], [10, 21, 12, 13]).stoppedOnDraft)
        XCTAssertFalse(resolve([99], [5, 6]).stoppedOnDraft)
        XCTAssertFalse(resolve([], [7]).stoppedOnDraft)
    }

    func testNoDrafts() {
        XCTAssertEqual(resolve([], [7]), RoundResult(emitted: [7], acceptedDrafts: 0, keep: 1, stopped: false))
        XCTAssertEqual(resolve([], [99]), RoundResult(emitted: [], acceptedDrafts: 0, keep: 1, stopped: true))
    }

    func testDraftsWithoutASampledRowAreIgnored() {
        XCTAssertEqual(resolve([10, 11, 12], [10, 11]), RoundResult(emitted: [10, 11], acceptedDrafts: 1, keep: 2, stopped: false))
        XCTAssertEqual(resolve([10], [10, 11, 12]), RoundResult(emitted: [10, 11], acceptedDrafts: 1, keep: 2, stopped: false))
    }

    // MARK: - Losslessness

    /// A tiny Markov "model" over 6 tokens: `model[t]` is the next-token distribution after token t.
    private static let model: [[Double]] = [
        [0.10, 0.05, 0.15, 0.35, 0.20, 0.15],
        [0.30, 0.10, 0.10, 0.10, 0.25, 0.15],
        [0.05, 0.40, 0.05, 0.20, 0.10, 0.20],
        [0.20, 0.20, 0.20, 0.05, 0.05, 0.30],
        [0.15, 0.15, 0.30, 0.10, 0.20, 0.10],
        [0.25, 0.05, 0.10, 0.30, 0.10, 0.20],
    ]

    /// Deterministic drafters: their drafts depend only on the context, never on the target's draws.
    private enum Drafter {
        case constant(Int)
        case argmax

        func drafts(after token: Int, k: Int) -> [Int] {
            var drafts: [Int] = []
            var last = token
            for _ in 0..<k {
                switch self {
                case .constant(let value):
                    last = value
                case .argmax:
                    let row = AcceptanceRuleTests.model[last]
                    last = row.indices.max { row[$0] < row[$1] }!
                }
                drafts.append(last)
            }
            return drafts
        }
    }

    /// One speculative round after `y`: the target samples every input row as plain decoding
    /// would, then the rule decides.
    private static func round(after y: Int, drafter: Drafter, k: Int, stops: Set<Int>, rng: inout DraftingTestRNG) -> (drafts: [Int], result: RoundResult) {
        let drafts = drafter.drafts(after: y, k: k)
        let inputs = [y] + drafts
        let sampled = inputs.map { rng.categorical(model[$0]) }
        return (drafts, AcceptanceRule.resolve(drafts: drafts, sampled: sampled, stopTokens: stops, remaining: 1_000))
    }

    func testFirstTokenFollowsTheTargetDistribution() {
        let rounds = 200_000
        let cases: [(Drafter, Int, Set<Int>)] = [
            (.constant(2), 1, []),
            (.constant(2), 3, []),
            (.argmax, 2, []),
            (.argmax, 4, []),
            // Adversarial: the draft is the stop token.
            (.constant(5), 2, [5]),
        ]
        var rng = DraftingTestRNG(seed: 0x5EED)
        for (drafter, k, stopSet) in cases {
            var first = [Int](repeating: 0, count: 6)
            var afterRejection = [Int](repeating: 0, count: 6)
            var secondAfterRejection = [Int](repeating: 0, count: 6)
            var accepted = 0
            let drafts = drafter.drafts(after: 0, k: k)
            for _ in 0..<rounds {
                let (_, result) = Self.round(after: 0, drafter: drafter, k: k, stops: stopSet, rng: &rng)
                // A stop shows up as no emitted token; for the distribution it is that token.
                let outcome = result.emitted.first ?? stopSet.first!
                first[outcome] += 1
                accepted += result.acceptedDrafts
                if result.acceptedDrafts == 0 { afterRejection[outcome] += 1 }
                if result.acceptedDrafts == 1, k >= 2, result.emitted.count >= 2 { secondAfterRejection[result.emitted[1]] += 1 }
            }
            let label = "drafter \(drafter) k \(k) stops \(stopSet)"
            XCTAssertGreaterThan(accepted, 0, label)
            let p = Self.model[0]
            XCTAssertGreaterThan(ChiSquare.pValue(observed: first, probabilities: p), 0.001, "\(label): \(first)")

            // After a rejection at depth 1, the emitted token is the target's sample conditioned on
            // differing from the draft.
            XCTAssertEqual(afterRejection[drafts[0]], 0, label)
            XCTAssertGreaterThan(ChiSquare.pValue(observed: afterRejection, probabilities: Self.excluding(drafts[0], from: p)), 0.001, "\(label): \(afterRejection)")

            // After accepting draft 1 and rejecting draft 2, the second token is the target's sample
            // after draft 1, conditioned on differing from draft 2.
            if k >= 2, stopSet.isEmpty {
                XCTAssertEqual(secondAfterRejection[drafts[1]], 0, label)
                let conditional = Self.excluding(drafts[1], from: Self.model[drafts[0]])
                XCTAssertGreaterThan(ChiSquare.pValue(observed: secondAfterRejection, probabilities: conditional), 0.001, "\(label): \(secondAfterRejection)")
            }
        }
    }

    func testTwoTokenSequencesFollowPlainDecoding() {
        let runs = 200_000
        var rng = DraftingTestRNG(seed: 0xA11CE)
        for (drafter, k) in [(Drafter.argmax, 3), (Drafter.constant(1), 2)] {
            var joint = [Int](repeating: 0, count: 36)
            var inconsistent = 0
            var speculativeRounds = 0
            for _ in 0..<runs {
                var emitted: [Int] = []
                var y = 0
                while emitted.count < 2 {
                    let (_, result) = Self.round(after: y, drafter: drafter, k: k, stops: [], rng: &rng)
                    // Without stops every round emits its accepted drafts plus one sampled token, and
                    // keeps y plus the accepted drafts.
                    if result.keep != result.acceptedDrafts + 1 || result.emitted.count != result.acceptedDrafts + 1 {
                        inconsistent += 1
                    }
                    if result.acceptedDrafts > 0 { speculativeRounds += 1 }
                    emitted += result.emitted
                    y = emitted[emitted.count - 1]
                }
                joint[emitted[0] * 6 + emitted[1]] += 1
            }
            XCTAssertEqual(inconsistent, 0)
            XCTAssertGreaterThan(speculativeRounds, runs / 20)
            let plain: [Double] = (0..<36).map { (cell: Int) -> Double in
                let a = cell / 6, b = cell % 6
                return Self.model[0][a] * Self.model[a][b]
            }
            XCTAssertGreaterThan(ChiSquare.pValue(observed: joint, probabilities: plain), 0.001, "drafter \(drafter) k \(k)")
        }
    }

    func testChiSquareHelperDetectsAWrongDistribution() {
        // The test would catch a rule that, say, always emitted an accepted draft.
        var rng = DraftingTestRNG(seed: 1)
        var biased = [Int](repeating: 0, count: 6)
        for _ in 0..<20_000 {
            let token = rng.categorical(Self.model[0])
            biased[token == 2 || rng.unit() < 0.05 ? 2 : token] += 1
        }
        XCTAssertLessThan(ChiSquare.pValue(observed: biased, probabilities: Self.model[0]), 0.001)
        XCTAssertEqual(ChiSquare.pValue(statistic: 20.515, degreesOfFreedom: 5), 0.001, accuracy: 2e-5)
        XCTAssertEqual(ChiSquare.pValue(statistic: 66.619, degreesOfFreedom: 35), 0.001, accuracy: 2e-5)
        XCTAssertEqual(ChiSquare.pValue(statistic: 4.351, degreesOfFreedom: 5), 0.5, accuracy: 1e-3)
        XCTAssertEqual(ChiSquare.pValue(statistic: 0, degreesOfFreedom: 3), 1)
    }

    private static func excluding(_ token: Int, from p: [Double]) -> [Double] {
        let rest = 1 - p[token]
        return p.indices.map { $0 == token ? 0 : p[$0] / rest }
    }
}

/// Pearson's chi-square goodness-of-fit test.
private enum ChiSquare {
    /// The p-value of `observed` counts against `probabilities`; cells with probability 0 are left
    /// out (and must be empty).
    static func pValue(observed: [Int], probabilities: [Double]) -> Double {
        let total = Double(observed.reduce(0, +))
        var statistic = 0.0
        var cells = 0
        for (count, p) in zip(observed, probabilities) where p > 0 {
            let expected = total * p
            statistic += (Double(count) - expected) * (Double(count) - expected) / expected
            cells += 1
        }
        return pValue(statistic: statistic, degreesOfFreedom: cells - 1)
    }

    /// Q(df/2, x/2), the regularized upper incomplete gamma function.
    static func pValue(statistic: Double, degreesOfFreedom: Int) -> Double {
        let a = Double(degreesOfFreedom) / 2
        let x = statistic / 2
        guard x > 0 else { return 1 }
        let logPrefix = -x + a * log(x) - lgamma(a)
        if x < a + 1 {
            // Series for the lower function P.
            var term = 1 / a
            var sum = term
            var n = a
            for _ in 0..<10_000 {
                n += 1
                term *= x / n
                sum += term
                if abs(term) < abs(sum) * 1e-15 { break }
            }
            return max(0, 1 - sum * exp(logPrefix))
        }
        // Continued fraction for Q (modified Lentz).
        let tiny = 1e-300
        var b = x + 1 - a
        var c = 1 / tiny
        var d = 1 / b
        var h = d
        for i in 1..<10_000 {
            let an = -Double(i) * (Double(i) - a)
            b += 2
            d = an * d + b
            if abs(d) < tiny { d = tiny }
            c = b + an / c
            if abs(c) < tiny { c = tiny }
            d = 1 / d
            let delta = d * c
            h *= delta
            if abs(delta - 1) < 1e-15 { break }
        }
        return exp(logPrefix) * h
    }
}
