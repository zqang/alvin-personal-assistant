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

    /// The self-test leaves nothing behind: no system prefix saved to disk (the disk prefix is
    /// on), and the configured generator factory, whose corpus is persisted and whose draft
    /// statistics carry over, is never used. A normal reply afterwards does both, so the checks
    /// can see them.
    func testLeavesNoTracesBehind() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("self-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        for tiny in EngineTestHarness.Tiny.allCases {
            let directory = root.appendingPathComponent(tiny.rawValue)
            let factory = CountingGeneratorFactory()
            var configuration = EngineTestHarness.testConfiguration()
            configuration.prefixCacheDirectory = directory
            configuration.generatorFactory = factory
            let engine = try EngineTestHarness.makeEngine(tiny: tiny, seed: 730, configuration: configuration)

            let result = await EngineSelfTest.run(engine: engine)
            await engine.waitUntilIdle()
            XCTAssertTrue(result.passed, "\(tiny):\n\(result.detail)")
            XCTAssertEqual(try prefixFiles(in: directory), [], "\(tiny): the self-test saved no system prefix")
            XCTAssertEqual(factory.made, 0, "\(tiny): the self-test didn't use the configured generator factory")

            _ = try await EngineTestHarness.collect(engine.reply(
                EngineRequest(system: "You are Alvin.", turns: [ChatTurn(role: .user, text: "Hello")])))
            await engine.waitUntilIdle()
            XCTAssertEqual(try prefixFiles(in: directory).count, 1, "\(tiny): a reply saves its system prefix")
            XCTAssertEqual(factory.made, 1, "\(tiny): a reply uses the factory")
        }
    }

    private func prefixFiles(in directory: URL) throws -> [String] {
        guard FileManager.default.fileExists(atPath: directory.path) else { return [] }
        return try FileManager.default.contentsOfDirectory(atPath: directory.path).filter { $0.hasSuffix(".safetensors") }
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
        let tie = EngineSelfTest.nearTieMargin
        XCTAssertEqual(tie, 0.5, "the 4-bit near-tie margin the engine tests use (plan §10)")
        XCTAssertTrue(EngineSelfTest.compareSequences(plain: [1, 2, 3], margins: [1, 1, 1], speculative: [1, 2, 3]).0)
        XCTAssertTrue(EngineSelfTest.compareSequences(plain: [1, 2, 3], margins: [1, tie * 0.8, 1], speculative: [1, 5, 3]).0)
        XCTAssertFalse(EngineSelfTest.compareSequences(plain: [1, 2, 3], margins: [1, tie, 1], speculative: [1, 5, 3]).0)
        XCTAssertFalse(EngineSelfTest.compareSequences(plain: [1, 2, 3], margins: [1, tie * 1.2, 1], speculative: [1, 5, 3]).0)
        XCTAssertFalse(EngineSelfTest.compareSequences(plain: [1, 2, 3], margins: [1, 1, 1, 1], speculative: [1, 2]).0)
    }

    /// The logits comparison of checks 1 and 3: a different argmax agrees only when the candidate
    /// picked one of the reference's near-tied tokens, not merely when the reference is close.
    func testLogitsComparison() {
        let reference = MLXArray([5.0, 4.8, 0.0, -1.0] as [Float])
        XCTAssertTrue(EngineSelfTest.compare(MLXArray([5.0, 4.8, 0.1, -1.0] as [Float]), reference).agrees, "same argmax")
        let flipped = EngineSelfTest.compare(MLXArray([4.8, 5.0, 0.0, -1.0] as [Float]), reference)
        XCTAssertTrue(flipped.agrees, "a near-tie flip: \(flipped)")
        XCTAssertEqual(flipped.candidateGap, 0.2, accuracy: 1e-5)
        let unrelated = EngineSelfTest.compare(MLXArray([0.0, 0.0, 9.0, 0.0] as [Float]), reference)
        XCTAssertEqual(unrelated.referenceMargin, 0.2, accuracy: 1e-5)
        XCTAssertFalse(unrelated.agrees, "the reference is close, but the candidate's choice isn't one of its near-ties: \(unrelated)")
        let wide = MLXArray([5.0, 3.0, 0.0, -1.0] as [Float])
        XCTAssertFalse(EngineSelfTest.compare(MLXArray([3.0, 5.0, 0.0, -1.0] as [Float]), wide).agrees, "a margin of 2 is no near-tie")
    }
}

/// A plain generator factory that counts the generators it makes.
final class CountingGeneratorFactory: GeneratorFactory, @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    var made: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    func makeGenerator(_ context: GeneratorContext) -> TokenGenerator {
        lock.lock()
        count += 1
        lock.unlock()
        return DecodeLoop(context)
    }
}
