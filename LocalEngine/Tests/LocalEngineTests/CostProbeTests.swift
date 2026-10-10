import AssistantKit
import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLMCommon
import XCTest

/// `CostProbe`: measures a curve over the requested widths on a scratch extension of the live
/// session, and leaves the session exactly as it was.
final class CostProbeTests: XCTestCase {
    override func setUpWithError() throws {
        try MetalAvailability.require()
    }

    /// The logits after feeding `probe` on a scratch extension of the session (`[V]` float32),
    /// and the ledger.
    private func state(_ engine: InferenceEngine, probe: Int = 65) async throws -> (ledger: [Int], logits: MLXArray) {
        try await engine.withSession { session in
            let logits = session.withScratch { () -> MLXArray in
                let row = session.feed([probe], rows: .last).logits!.reshaped(-1).asType(.float32)
                eval(row)
                return row
            }
            return (session.ledger, logits)
        }
    }

    func testProbeReturnsACurveAndLeavesTheSessionUnchanged() async throws {
        var rows: [[String]] = []
        for tiny in EngineTestHarness.Tiny.allCases {
            let engine = try EngineTestHarness.makeEngine(tiny: tiny, seed: 600)
            let request = EngineRequest(system: "You are Alvin.", turns: [ChatTurn(role: .user, text: "Hello there, how are you?")])
            _ = try await EngineTestHarness.collect(engine.reply(request))
            let before = try await state(engine)
            let summary = await engine.sessionSummary()

            let curve = try await CostProbe.measure(engine: engine)
            XCTAssertEqual(Set(curve.seconds.keys), [1, 2, 3, 4, 6, 8], "\(tiny)")
            XCTAssertTrue(curve.seconds.values.allSatisfy { $0.isFinite && $0 > 0 }, "\(tiny): \(curve.seconds)")
            XCTAssertEqual(curve.relative(1), 1)
            XCTAssertGreaterThanOrEqual(curve.relative(8), 1)

            let after = try await state(engine)
            XCTAssertEqual(after.ledger, before.ledger, "\(tiny): the ledger is unchanged")
            XCTAssertTrue(LogitCheck.isExactlyEqual(after.logits, before.logits), "\(tiny): the next logits are unchanged")
            let summaryAfter = await engine.sessionSummary()
            XCTAssertEqual(summaryAfter, summary, "\(tiny)")
            let report = try await engine.withSession { $0.assertConsistent() }
            XCTAssertTrue(report.isConsistent(), "\(tiny): \(report)")

            // The session still continues as before.
            let next = EngineRequest(
                system: "You are Alvin.",
                turns: request.turns + [ChatTurn(role: .assistant, text: "x"), ChatTurn(role: .user, text: "And now?")])
            _ = try await EngineTestHarness.collect(engine.reply(next))

            rows.append([tiny.rawValue] + [1, 2, 3, 4, 6, 8].map { String(format: "%.2f", curve.relative($0)) })
        }
        EngineReport.appendTable(
            title: "Cost probe on the tiny models (relative c(S); not representative of real models)",
            header: ["Model", "1", "2", "3", "4", "6", "8"], rows: rows)
    }

    func testProbeOnAColdSessionAndCustomWidths() async throws {
        let engine = try EngineTestHarness.makeEngine(tiny: .hybrid, seed: 610)
        let curve = try await CostProbe.measure(engine: engine, widths: [3, 1, 3, 0])
        XCTAssertEqual(Set(curve.seconds.keys), [1, 3])
        let ledger = try await engine.withSession { $0.ledger }
        XCTAssertEqual(ledger, [], "a cold session stays cold")
        let summary = await engine.sessionSummary()
        XCTAssertTrue(summary.hasPrefix("cold"), summary)
    }

    func testProbeRespectsTheGPUHooks() async throws {
        var configuration = EngineTestHarness.testConfiguration()
        configuration.hooks = EngineHooks(beginGPU: { false }, endGPU: {}, isAllowed: { true })
        let refused = try EngineTestHarness.makeEngine(tiny: .qwen3, seed: 620, configuration: configuration)
        do {
            _ = try await CostProbe.measure(engine: refused)
            XCTFail("the probe ran without the GPU")
        } catch {
            XCTAssertEqual(error as? EngineError, .leftForeground)
        }

        // The GPU stops being allowed part-way: the probe stops and the session is restored.
        let engine = try EngineTestHarness.makeEngine(tiny: .qwen3, seed: 621)
        _ = try await EngineTestHarness.collect(engine.reply(
            EngineRequest(system: "You are Alvin.", turns: [ChatTurn(role: .user, text: "Hi")])))
        let before = try await state(engine)
        let counter = ProbeCallCounter()
        await engine.updateConfiguration {
            $0.hooks = EngineHooks(beginGPU: { true }, endGPU: {}, isAllowed: { counter.next() < 6 })
        }
        do {
            _ = try await CostProbe.measure(engine: engine)
            XCTFail("the probe should stop when the GPU is no longer allowed")
        } catch {
            XCTAssertEqual(error as? EngineError, .leftForeground)
        }
        await engine.updateConfiguration { $0.hooks = .alwaysAllowed }
        let after = try await state(engine)
        XCTAssertEqual(after.ledger, before.ledger)
        XCTAssertTrue(LogitCheck.isExactlyEqual(after.logits, before.logits))
    }
}

/// A thread-safe call counter.
final class ProbeCallCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0

    func next() -> Int {
        lock.lock()
        defer { lock.unlock() }
        value += 1
        return value
    }
}
