import AssistantKit
import Foundation
@testable import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLMCommon
import XCTest

/// `EngineSelfTest`: passes on the tiny models (plain and speculative engines), fails when the
/// rollback is broken on purpose, and leaves the session invalidated.
final class SelfTestTests: XCTestCase {
    override func setUpWithError() throws {
        try MetalAvailability.require()
    }

    override func tearDown() {
        EngineSelfTest.debugBreakRollback = false
        super.tearDown()
    }

    func testPassesOnTheTinyModels() async throws {
        var rows: [[String]] = []
        for tiny in EngineTestHarness.Tiny.allCases {
            for speculative in [false, true] {
                var configuration = EngineTestHarness.testConfiguration()
                configuration.temperature = 0.7  // the self-test decodes greedily whatever the configuration says
                if speculative {
                    configuration.generatorFactory = SpeculativeGeneratorFactory(mode: .automatic, curve: .stockMLXDefault, corpusURL: nil)
                }
                let engine = try EngineTestHarness.makeEngine(tiny: tiny, seed: 700, configuration: configuration)
                let result = await EngineSelfTest.run(engine: engine)
                let label = "\(tiny)\(speculative ? " speculative" : "")"
                XCTAssertTrue(result.passed, "\(label):\n\(result.detail)")
                XCTAssertEqual(result.checks, ["sessionReuse": true, "speculation": true, "rollback": true], label)
                XCTAssertEqual(result.formatVersion, EngineInfo.formatVersion)
                XCTAssertGreaterThan(result.seconds, 0)
                XCTAssertEqual(result.detail.split(separator: "\n").count, 3, result.detail)
                let summary = await engine.sessionSummary()
                XCTAssertTrue(summary.hasPrefix("cold"), "\(label): the session is invalidated afterwards: \(summary)")
                let ledger = try await engine.withSession { $0.ledger }
                XCTAssertEqual(ledger, [])

                // The engine works normally afterwards.
                let events = try await EngineTestHarness.collect(engine.reply(
                    EngineRequest(system: "You are Alvin.", turns: [ChatTurn(role: .user, text: "Hello")])))
                XCTAssertNotNil(EngineTestHarness.finish(events))
                rows.append([label, result.passed ? "passed" : "FAILED", String(format: "%.2f s", result.seconds),
                             result.detail.replacingOccurrences(of: "\n", with: "<br>").replacingOccurrences(of: "|", with: "/")])
            }
        }
        EngineReport.appendTable(title: "Self-test on the tiny models", header: ["Engine", "Result", "Time", "Detail"], rows: rows)
    }

    func testFailsWithABrokenRollback() async throws {
        EngineSelfTest.debugBreakRollback = true
        for tiny in EngineTestHarness.Tiny.allCases {
            let engine = try EngineTestHarness.makeEngine(tiny: tiny, seed: 710)
            let result = await EngineSelfTest.run(engine: engine)
            XCTAssertFalse(result.passed, "\(tiny):\n\(result.detail)")
            XCTAssertEqual(result.checks["rollback"], false, "\(tiny):\n\(result.detail)")
            XCTAssertEqual(result.checks["sessionReuse"], true, "\(tiny):\n\(result.detail)")
            XCTAssertEqual(result.checks["speculation"], true, "\(tiny):\n\(result.detail)")
            EngineReport.append("- Self-test with debugBreakRollback (\(tiny)): " + (result.detail.split(separator: "\n").last.map(String.init) ?? ""))

            // The fault left nothing behind: the session was invalidated and is exact.
            let events = try await EngineTestHarness.collect(engine.reply(
                EngineRequest(system: "You are Alvin.", turns: [ChatTurn(role: .user, text: "Hello")])))
            XCTAssertNotNil(EngineTestHarness.finish(events))
            let report = try await engine.withSession { $0.assertConsistent() }
            XCTAssertTrue(report.isConsistent(), "\(tiny): \(report)")
        }
    }

    /// Random tiny models repeat one token, which prompt lookup can't draft ahead of, so check 2
    /// is also run on a scripted model that copies the prompt: it speculates and passes, and the
    /// broken rollback fails check 3 there too.
    func testSpeculationCheckSpeculatesOnACopyingModel() async throws {
        let tokenizer = FakeChatMLTokenizer()
        let copy = tokenizer.encodeRaw("The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.")
        for tiny in EngineTestHarness.Tiny.allCases {
            for broken in [false, true] {
                EngineSelfTest.debugBreakRollback = broken
                let (engine, _) = try EngineTestHarness.makeScriptedEngine(tiny: tiny, scripts: [copy])
                let (speculated, line, rolledBack, rollbackLine) = try await engine.onQueue {
                    try EngineSelfTest.checkSpeculationAndRollback(engine)
                }
                XCTAssertTrue(speculated, "\(tiny): \(line)")
                XCTAssertFalse(line.contains("; 0 rounds"), "\(tiny): \(line)")
                XCTAssertEqual(rolledBack, !broken, "\(tiny) broken=\(broken): \(rollbackLine)")
                EngineReport.append("- Self-test checks 2–3 on a copying scripted \(tiny)\(broken ? " (broken rollback)" : ""): \(line); \(rollbackLine)")
            }
        }
    }

    func testRefusedGPUFails() async throws {
        var configuration = EngineTestHarness.testConfiguration()
        configuration.hooks = EngineHooks(beginGPU: { false }, endGPU: {}, isAllowed: { true })
        let engine = try EngineTestHarness.makeEngine(tiny: .qwen3, seed: 720, configuration: configuration)
        let result = await EngineSelfTest.run(engine: engine)
        XCTAssertFalse(result.passed)
        XCTAssertTrue(result.detail.contains("stopped"), result.detail)
    }

    func testResultRoundTrips() throws {
        let result = EngineSelfTest.Result(
            passed: true, checks: ["sessionReuse": true, "speculation": true, "rollback": true], detail: "ok", seconds: 3.5,
            formatVersion: EngineInfo.formatVersion)
        let decoded = try JSONDecoder().decode(EngineSelfTest.Result.self, from: JSONEncoder().encode(result))
        XCTAssertEqual(decoded, result)
    }

    /// The sequence comparison of check 2 follows the near-tie rule.
    func testSequenceComparison() {
        XCTAssertTrue(EngineSelfTest.compareSequences(plain: [1, 2, 3], margins: [1, 1, 1], speculative: [1, 2, 3]).0)
        XCTAssertTrue(EngineSelfTest.compareSequences(plain: [1, 2, 3], margins: [1, 0.1, 1], speculative: [1, 5, 3]).0)
        XCTAssertFalse(EngineSelfTest.compareSequences(plain: [1, 2, 3], margins: [1, 0.5, 1], speculative: [1, 5, 3]).0)
        XCTAssertFalse(EngineSelfTest.compareSequences(plain: [1, 2, 3], margins: [1, 1, 1, 1], speculative: [1, 2]).0)
    }
}
