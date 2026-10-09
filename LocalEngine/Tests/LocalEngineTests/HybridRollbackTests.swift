import AssistantKit
import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

/// Rolling back a verified round on the hybrid (gated-delta capture and replay) and on Qwen3
/// (attention trim): after keeping any m of the drafts, the cache is the one a session that never
/// saw the rejected rows holds (plan §4.3, §6.4).
final class HybridRollbackTests: XCTestCase {
    override func setUpWithError() throws {
        try MetalAvailability.require()
    }

    /// Forced partial acceptance at every m: feed `[y] + drafts` (the first m drafts right, the
    /// rest wrong) with a capture, commit `keep = m + 1`, then feed the bonus token. The next
    /// logits and the recurrent state equal a fresh session fed only the kept tokens and the
    /// bonus, and the session passes `assertConsistent`.
    func testCommitAtEveryAcceptedCountMatchesAReference() throws {
        var rows: [[String]] = []
        for tiny in EngineTestHarness.Tiny.allCases {
            let model = try EngineTestHarness.makeModel(tiny, seed: 400)
            let prompt = TinyModels.tokens(11, seed: 401)
            let greedy = try SpecHarness.plain(model: model, prompt: prompt, maxTokens: 12, stops: []).tokens
            let y = greedy[0]
            var worst: Float = 0
            for drafts in [4, 8] {
                for m in 0 ... drafts {
                    let label = "\(tiny) K=\(drafts) m=\(m)"
                    let truth = Array(greedy[1 ... drafts])
                    var proposal = Array(truth.prefix(m))
                    while proposal.count < drafts {
                        // Wrong at depth m, arbitrary after it.
                        proposal.append((truth[proposal.count] + 1 + proposal.count) % TinyModels.vocabularySize)
                    }
                    let session = LiveSession(target: SpecHarness.target(for: model))
                    session.feed(prompt, rows: .last)
                    let result = session.feed([y] + proposal, rows: .all, capture: true)
                    let sampled = result.logits!.argMax(axis: -1).asArray(Int32.self).map { Int($0) }
                    let outcome = AcceptanceRule.resolve(drafts: proposal, sampled: sampled, stopTokens: [], remaining: 100)
                    XCTAssertEqual(outcome.acceptedDrafts, m, "\(label): the sharpened model agrees with its plain run")
                    session.commit(result.capture, keep: outcome.keep, of: proposal.count + 1)
                    let kept = prompt + [y] + Array(proposal.prefix(outcome.acceptedDrafts))
                    XCTAssertEqual(session.ledger, kept, label)

                    let bonus = sampled[outcome.acceptedDrafts]
                    let live = session.feed([bonus], rows: .last).logits!.reshaped(-1).asType(.float32)

                    let reference = LiveSession(target: SpecHarness.target(for: model))
                    let expected = reference.feed(kept + [bonus], rows: .last).logits!.reshaped(-1).asType(.float32)
                    let scale = max(1, MLX.abs(expected).max().item(Float.self))
                    let difference = LogitCheck.maxAbsDifference(live, expected)
                    worst = max(worst, difference / scale)
                    XCTAssertLessThanOrEqual(difference, 1e-4 * scale, "\(label): next logits after the commit")

                    // The recurrent state is the reference's (it saw the same tokens, one block).
                    for index in session.layout.recurrent {
                        let liveLayer = session.target.cache[index] as! ArraysCache
                        let referenceLayer = reference.target.cache[index] as! ArraysCache
                        for slot in 0 ..< 2 {
                            let a = try XCTUnwrap(liveLayer[slot], "\(label) layer \(index) slot \(slot)")
                            let b = try XCTUnwrap(referenceLayer[slot], "\(label) layer \(index) slot \(slot)")
                            XCTAssertTrue(LogitCheck.isClose(a, b, rtol: 1e-4, atol: 1e-5), "\(label) layer \(index) slot \(slot)")
                        }
                    }
                    let report = session.assertConsistent()
                    XCTAssertTrue(report.isConsistent(), "\(label): \(report)")
                }
            }
            rows.append([tiny.rawValue, "K ∈ {4, 8}, every m", String(format: "%.2e", worst)])
        }
        EngineReport.appendTable(
            title: "Rollback after partial acceptance vs a session that never saw the rejected rows",
            header: ["Model", "Cases", "Worst max |Δ| / scale"], rows: rows)
    }

    /// Keeping any m of a fed block (m = 0 included) leaves exactly m of its tokens: in the
    /// ledger, in every attention layer's offset and in the recurrent state (`assertConsistent`
    /// rebuilds the ledger from scratch); checkpoints at or below the kept end survive.
    func testCommitLeavesNoTraceOfRejectedRows() throws {
        let model = try EngineTestHarness.makeModel(.hybrid, seed: 410)
        let prompt = TinyModels.tokens(9, seed: 411)
        let drafts = TinyModels.tokens(5, seed: 412)
        for m in 0 ... drafts.count {
            let session = LiveSession(target: SpecHarness.target(for: model))
            session.feed(prompt, rows: .last)
            session.checkpoint(.replyStart)
            let before = session.target.cache.map { $0.offset }
            let result = session.feed([drafts[0]] + Array(drafts[1...]), rows: .all, capture: true)
            session.commit(result.capture, keep: m, of: drafts.count)
            XCTAssertEqual(session.ledger, prompt + Array(drafts.prefix(m)), "m=\(m)")
            for index in session.layout.attention {
                XCTAssertEqual(session.target.cache[index].offset, before[index] + m, "m=\(m) layer \(index)")
            }
            XCTAssertEqual(session.checkpoints.marks[.replyStart], prompt.count, "checkpoints at or below the kept end stay")
            let report = session.assertConsistent()
            XCTAssertTrue(report.isConsistent(), "m=\(m): \(report)")
        }
    }

    /// Through the loop: a drafter that is right up to depth m and wrong at m + 1 (m cycling
    /// 0…K−1 from round to round) leaves output and cache equal to plain decoding.
    func testLoopWithPartialAcceptanceAtEveryDepth() throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let model = try EngineTestHarness.makeModel(tiny, seed: 420)
            let prompt = TinyModels.tokens(13, seed: 421)
            let reference = try SpecHarness.plain(model: model, prompt: prompt, maxTokens: 40, stops: [])
            for k in [2, 4, 8] {
                let drafter = DepthCyclingDrafter(sequence: prompt + reference.tokens, depths: k)
                let run = try SpecHarness.speculative(
                    model: model, prompt: prompt, maxTokens: 40, stops: [], drafters: [drafter],
                    options: SpeculativeLoop.Options(forcedDraftLength: k))
                let label = "\(tiny) K=\(k)"
                XCTAssertEqual(run.tokens, reference.tokens, label)
                XCTAssertEqual(run.session.ledger, reference.session.ledger, label)
                let accepted = Set(run.loop.rounds.filter { $0.proposed == k }.map(\.accepted))
                XCTAssertGreaterThan(accepted.count, 1, "\(label): several acceptance depths were exercised: \(accepted.sorted())")
                try SpecHarness.assertExact(run, prompt: prompt, maxTokens: 40, stops: [], label: label)
            }
        }
    }
}

/// Proposes the reference continuation with a wrong token at depth `m`, where m cycles through
/// 0…depths−1 from one proposal to the next (and `depths` means all right).
final class DepthCyclingDrafter: Drafter {
    let sequence: [Int]
    let depths: Int
    private var counter = 0
    var source: DraftSource { .mtp }
    var costPerToken: Double { 0 }
    var wantsHidden: Bool { false }

    init(sequence: [Int], depths: Int) {
        self.sequence = sequence
        self.depths = depths
    }

    func reset(ledger: [Int], request: EngineRequest) {}

    func propose(context: ArraySlice<Int>, maxTokens: Int) -> DraftProposal? {
        let start = context.count
        guard maxTokens > 0, start < sequence.count, context.elementsEqual(sequence[..<start]) else { return nil }
        var tokens = Array(sequence[start...].prefix(maxTokens))
        let wrongAt = counter % (depths + 1)
        counter += 1
        if wrongAt < tokens.count {
            tokens[wrongAt] = (tokens[wrongAt] + 1) % TinyModels.vocabularySize
        }
        return DraftProposal(tokens: tokens, source: .mtp, matchLength: 4)
    }

    func observe(_ round: RoundObservation) {}
}
