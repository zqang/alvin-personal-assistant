import AssistantKit
import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

/// `DecodeLoop`: greedy decoding equals the stock `TokenIterator` (up to near-ties, plan §6.4) on
/// the tiny Qwen3 (`StockTarget`) and the tiny hybrid (`HybridTarget`), and it leaves the ledger
/// exact at every way of stopping.
final class DecodeLoopTests: XCTestCase {
    override func setUpWithError() throws {
        try MetalAvailability.require()
    }

    private static func target(for model: any LanguageModel) -> any TargetModel {
        if let fork = model as? any HybridQwen35Forwarding {
            return HybridTarget(model: fork)
        }
        return StockTarget(model: model)
    }

    /// A decode loop over a fresh session holding `prompt`.
    private static func decodeLoop(
        model: any LanguageModel, prompt: [Int], maxTokens: Int, stopTokens: Set<Int> = [],
        isAllowed: @escaping () -> Bool = { true }
    ) -> (DecodeLoop, LiveSession, ConfidenceRecorder) {
        let session = LiveSession(target: target(for: model))
        let logits = session.feed(prompt, rows: .last).logits!
        let confidence = ConfidenceRecorder()
        let context = GeneratorContext(
            session: session, sampler: FastSampler(temperature: 0), firstLogits: logits, stopTokens: stopTokens,
            maxTokens: maxTokens, request: EngineRequest(system: "", turns: []), drafters: [], insideToolCall: { false },
            isAllowed: isAllowed, renderer: FakeChatMLTokenizer(), toolCallFormat: .json, confidence: confidence)
        return (DecodeLoop(context), session, confidence)
    }

    private static func run(_ loop: DecodeLoop) throws -> (tokens: [Int], reason: EngineFinish.Reason) {
        var tokens: [Int] = []
        while true {
            let (emitted, finished) = try loop.step()
            tokens += emitted
            if let finished { return (tokens, finished) }
        }
    }

    private static func stockGreedy(model: any LanguageModel, prompt: [Int], count: Int) throws -> [Int] {
        let iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map { Int32($0) })), model: model, cache: nil,
            parameters: GenerateParameters(maxTokens: count, temperature: 0))
        return Array(iterator)
    }

    func testGreedyEqualsTokenIterator() throws {
        var rows: [[String]] = []
        for tiny in EngineTestHarness.Tiny.allCases {
            var tally = NearTieTally(tolerance: NearTieTally.float32Tolerance)
            for seed in UInt64(1) ... 4 {
                let model = try EngineTestHarness.makeModel(tiny, seed: 100 + seed)
                let prompt = TinyModels.tokens(9 + Int(seed), seed: 200 + seed)
                let expected = try Self.stockGreedy(model: model, prompt: prompt, count: 64)
                XCTAssertEqual(expected.count, 64)

                let (loop, session, confidence) = Self.decodeLoop(model: model, prompt: prompt, maxTokens: 64)
                let (tokens, reason) = try Self.run(loop)
                XCTAssertEqual(reason, .length)
                XCTAssertEqual(tokens.count, 64)
                XCTAssertEqual(session.ledger, prompt + tokens, "every emitted token is in the ledger")
                XCTAssertEqual(session.pendingCount, 0)
                XCTAssertEqual(confidence.samples.count, 64)

                let margins = TeacherForcing.run(model: model, prompt: prompt, continuation: expected).map(\.margin)
                tally.record("\(tiny) seed \(seed)", reference: expected, candidate: tokens, referenceMargins: margins)
                XCTAssertTrue(session.assertConsistent().isConsistent(), "\(tiny) seed \(seed)")
            }
            tally.assertPassed()
            rows.append([tiny.rawValue, tally.summary])
        }
        EngineReport.appendTable(title: "DecodeLoop greedy vs TokenIterator (tiny, 64 tokens)", header: ["Model", "Result"], rows: rows)
    }

    /// A stop token ends the reply without being emitted, and stays in the cache (F4).
    func testStopTokenIsFedButNotEmitted() throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let model = try EngineTestHarness.makeModel(tiny, seed: 31)
            let prompt = TinyModels.tokens(8, seed: 32)
            let expected = try Self.stockGreedy(model: model, prompt: prompt, count: 12)
            let stop = expected[5]
            let firstStop = expected.firstIndex(of: stop)!
            let (loop, session, _) = Self.decodeLoop(model: model, prompt: prompt, maxTokens: 64, stopTokens: [stop])
            let (tokens, reason) = try Self.run(loop)
            XCTAssertEqual(reason, .stop, "\(tiny)")
            XCTAssertEqual(tokens, Array(expected[..<firstStop]), "\(tiny)")
            XCTAssertEqual(session.ledger, prompt + Array(expected[...firstStop]), "\(tiny): the stop token is fed")
            XCTAssertTrue(session.assertConsistent().isConsistent(), "\(tiny)")
            // Stepping after the end changes nothing.
            let (after, finished) = try loop.step()
            XCTAssertEqual(after, [])
            XCTAssertEqual(finished, .stop)
        }
    }

    /// When the GPU stops being allowed, the loop stops before feeding: the ledger ends with the
    /// last emitted token and nothing is pending.
    func testDisallowedStopsWithAnExactLedger() throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let model = try EngineTestHarness.makeModel(tiny, seed: 41)
            let prompt = TinyModels.tokens(10, seed: 42)
            var calls = 0
            let (loop, session, _) = Self.decodeLoop(model: model, prompt: prompt, maxTokens: 64) {
                calls += 1
                return calls <= 5
            }
            let (tokens, reason) = try Self.run(loop)
            XCTAssertEqual(reason, .cancelled, "\(tiny)")
            XCTAssertEqual(tokens.count, 5, "\(tiny)")
            XCTAssertEqual(session.ledger, prompt + tokens, "\(tiny)")
            XCTAssertEqual(session.pendingCount, 0)
            XCTAssertTrue(session.assertConsistent().isConsistent(), "\(tiny)")
            try loop.flush()
            XCTAssertEqual(session.ledger, prompt + tokens, "flush has nothing to feed")
        }
    }
}
