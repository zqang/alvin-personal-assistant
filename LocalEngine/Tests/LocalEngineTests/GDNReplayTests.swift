import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

/// Exact rollback of a verified block (plan §4.3): after a 6-token verify pass with capture,
/// `HybridTarget.commit(keep: m)` leaves the recurrent layers in exactly the state the pass had
/// after `m` steps, and the next logits match a fresh run over the accepted prefix.
final class GDNReplayTests: XCTestCase {
    private static let promptCount = 9
    private static let verifyCount = 6

    override func setUpWithError() throws {
        try MetalAvailability.require()
    }

    private struct Round {
        let target: HybridTarget
        let capture: GDNCaptureSink
        /// `(cache[0], cache[1])` of each recurrent layer before the verify pass.
        let before: [(conv: MLXArray?, state: MLXArray?)]
    }

    /// Prefills the prompt (cache only), then verifies the drafts with capture. Re-run for each
    /// `m` so every commit starts from the same pre-round state.
    private func makeRound(_ model: any HybridQwen35Forwarding, prompt: [Int], drafts: [Int]) throws -> Round {
        let target = HybridTarget(model: model)
        let prefill = target.forward(Self.int32(prompt), rows: .none, captureForRollback: false, wantHidden: false)
        XCTAssertNil(prefill.logits)
        eval(target.cache)

        let before = target.layout.recurrent.map { (index: Int) -> (conv: MLXArray?, state: MLXArray?) in
            let layer = target.cache[index] as! MambaCache
            return (conv: layer[0], state: layer[1])
        }

        let verify = target.forward(Self.int32(drafts), rows: .all, captureForRollback: true, wantHidden: false)
        let logits = try XCTUnwrap(verify.logits)
        let capture = try XCTUnwrap(verify.capture as? GDNCaptureSink)
        eval(logits)
        eval(target.cache)
        return Round(target: target, capture: capture, before: before)
    }

    func testCommitReplaysTheRecurrentStateExactly() throws {
        try assertReplay(model: TinyForkModels.makeForkHybrid(seed: 41), label: "text")
    }

    func testCommitReplaysTheRecurrentStateExactlyInTheWrapper() throws {
        try assertReplay(model: TinyForkModels.makeForkHybridWrapper(seed: 42), label: "wrapper")
    }

    func testCommitReplaysTheRecurrentStateExactlyIn4Bit() throws {
        let model = try TinyForkModels.makeForkHybrid(seed: 43)
        TinyModels.quantize4bit(model)
        try assertReplay(model: model, label: "text, 4-bit", tolerance: 1e-4)
    }

    /// Keeping every verified token changes nothing: the slots are the pass's own arrays.
    func testCommitKeepingEverythingIsANoOp() throws {
        let model = try TinyForkModels.makeForkHybrid(seed: 44)
        let round = try makeRound(model, prompt: TinyModels.tokens(Self.promptCount, seed: 45), drafts: TinyModels.tokens(Self.verifyCount, seed: 46))
        let target = round.target
        let slots = target.layout.recurrent.map { (index: Int) -> (MLXArray?, MLXArray?) in
            let layer = target.cache[index] as! MambaCache
            return (layer[0], layer[1])
        }
        let offsets = target.cache.map { $0.offset }

        target.commit(round.capture, keep: Self.verifyCount, of: Self.verifyCount)

        XCTAssertEqual(target.cache.map { $0.offset }, offsets)
        for (index, slot) in zip(target.layout.recurrent, slots) {
            let layer = target.cache[index] as! MambaCache
            XCTAssertTrue(layer[0] === slot.0, "layer \(index) conv")
            XCTAssertTrue(layer[1] === slot.1, "layer \(index) state")
        }
    }

    /// Keeping none of the verified tokens puts back the pre-round slots.
    func testCommitKeepingNothingRestoresThePreRoundState() throws {
        let model = try TinyForkModels.makeForkHybrid(seed: 47)
        let round = try makeRound(model, prompt: TinyModels.tokens(Self.promptCount, seed: 48), drafts: TinyModels.tokens(Self.verifyCount, seed: 49))
        let target = round.target
        target.commit(round.capture, keep: 0, of: Self.verifyCount)
        eval(target.cache)

        XCTAssertEqual(target.layout.attention.map { target.cache[$0].offset }, [Self.promptCount, Self.promptCount])
        for (position, index) in target.layout.recurrent.enumerated() {
            let layer = target.cache[index] as! MambaCache
            let conv = try XCTUnwrap(layer[0])
            let beforeConv = try XCTUnwrap(round.before[position].conv)
            XCTAssertTrue(LogitCheck.isExactlyEqual(conv, beforeConv), "layer \(index) conv")
            XCTAssertTrue(layer[1] === round.before[position].state, "layer \(index) state")
        }
    }

    // MARK: Helpers

    private func assertReplay(model: any HybridQwen35Forwarding, label: String, tolerance: Double = 1e-5) throws {
        let prompt = TinyModels.tokens(Self.promptCount, seed: 50)
        let drafts = TinyModels.tokens(Self.verifyCount, seed: 51)
        let next = 77
        var worst: Float = 0

        for m in 1 ..< Self.verifyCount {
            let round = try makeRound(model, prompt: prompt, drafts: drafts)
            let target = round.target
            let capture = round.capture
            XCTAssertEqual(capture.entries.map { $0.layer }, target.layout.recurrent, label)

            target.commit(capture, keep: m, of: Self.verifyCount)
            eval(target.cache)

            // Attention layers hold the prompt and the first m drafts.
            XCTAssertEqual(
                target.layout.attention.map { target.cache[$0].offset },
                target.layout.attention.map { _ in Self.promptCount + m }, "\(label), m = \(m)")

            let mask = MLXArray((0 ..< Self.verifyCount).map { $0 < m }, [1, Self.verifyCount])
            for (position, (index, entry)) in zip(target.layout.recurrent, capture.entries).enumerated() {
                let layer = target.cache[index] as! MambaCache

                // Gated-delta state: bitwise the masked full run on the captured inputs.
                let (_, reference) = gatedDeltaUpdate(
                    q: entry.q, k: entry.k, v: entry.v, a: entry.a, b: entry.b,
                    aLog: entry.aLog, dtBias: entry.dtBias, state: entry.initialState, mask: mask)
                let state = try XCTUnwrap(layer[1])
                eval(reference, state)
                XCTAssertTrue(
                    LogitCheck.isExactlyEqual(state, reference),
                    "\(label), m = \(m), layer \(index) state: max |Δ| = \(LogitCheck.maxAbsDifference(state, reference))")

                // Conv state: the last K − 1 rows of concat(pre-round conv state, qkv of the first
                // m drafts).
                let convRows = entry.convStateRows
                let beforeConv = try XCTUnwrap(round.before[position].conv)
                let qkv = entry.convInput[0..., convRows ..< (convRows + m), 0...]
                let joined = concatenated([beforeConv, qkv], axis: 1)
                let expectedConv = joined[0..., (joined.dim(1) - convRows)..., 0...]
                let conv = try XCTUnwrap(layer[0])
                eval(expectedConv, conv)
                XCTAssertEqual(conv.shape, beforeConv.shape, "\(label), m = \(m), layer \(index)")
                XCTAssertTrue(
                    LogitCheck.isExactlyEqual(conv, expectedConv),
                    "\(label), m = \(m), layer \(index) conv: max |Δ| = \(LogitCheck.maxAbsDifference(conv, expectedConv))")
            }

            // Next-step logits: allClose to a fresh run over prompt + the first m drafts. One
            // evaluation holds 9-, m- and 1-step gated-delta calls, as the engine's lazy rounds
            // do (LibraryAssumptionTests 9).
            let afterCommit = try XCTUnwrap(
                target.forward(Self.int32([next]), rows: .last, captureForRollback: false, wantHidden: false).logits)

            let fresh = HybridTarget(model: model)
            _ = fresh.forward(Self.int32(prompt), rows: .none, captureForRollback: false, wantHidden: false)
            _ = fresh.forward(Self.int32(Array(drafts.prefix(m))), rows: .none, captureForRollback: false, wantHidden: false)
            let reference = try XCTUnwrap(
                fresh.forward(Self.int32([next]), rows: .last, captureForRollback: false, wantHidden: false).logits)
            eval(afterCommit, reference)

            let difference = LogitCheck.maxAbsDifference(afterCommit, reference)
            worst = max(worst, difference)
            XCTAssertTrue(
                LogitCheck.isClose(afterCommit, reference, rtol: tolerance, atol: tolerance),
                "\(label), m = \(m): next-step logits max |Δ| = \(difference)")
        }
        EngineReport.append("- GDN replay (\(label)): recurrent state bitwise equal for m = 1…\(Self.verifyCount - 1); next-step logits max |Δ| = \(worst)")
    }

    private static func int32(_ tokens: [Int]) -> MLXArray {
        MLXArray(tokens.map { Int32($0) })
    }
}
