import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

/// `StockPrefill`: feeding all but the last prompt token in chunks and starting MLX's stock
/// `TokenIterator` from that token generates what one `TokenIterator` over the whole prompt
/// does, and leaves the same cache; a refusal stops it between chunks.
final class StockPrefillTests: XCTestCase {
    override func setUpWithError() throws {
        try MetalAvailability.require()
    }

    /// Plain attention, the stock hybrid and the engine's fork of it, all sharpened.
    private static func models() throws -> [(label: String, model: any LanguageModel)] {
        let qwen3 = try TinyModels.makeStockQwen3(seed: 11)
        let hybrid = try TinyModels.makeStockHybrid(seed: 12)
        TinyModels.sharpen(qwen3)
        TinyModels.sharpen(hybrid)
        return [("qwen3", qwen3), ("hybrid", hybrid), ("hybrid fork", try EngineTestHarness.makeModel(.hybrid, seed: 13))]
    }

    private static func array(_ tokens: [Int]) -> MLXArray {
        MLXArray(tokens.map { Int32($0) })
    }

    /// The offset of every attention layer's cache (the recurrent layers keep no count).
    private static func attentionOffsets(_ cache: [KVCache]) -> [Int] {
        cache.compactMap { $0 as? KVCacheSimple }.map(\.offset)
    }

    func testChunkedPrefillThenTheLastTokenGeneratesWhatOneShotGenerationDoes() throws {
        for (label, model) in try Self.models() {
            // One short chunk, several ending in a partial one, and exactly one full chunk.
            for (count, step) in [(6, 512), (40, 8), (45, 16), (33, 32)] {
                let name = "\(label), \(count) tokens in chunks of \(step)"
                let prompt = TinyModels.tokens(count, seed: UInt64(count * 31 + step))
                let head = Self.array(Array(prompt.dropLast()))
                let last = Self.array([prompt[count - 1]])
                let parameters = GenerateParameters(maxTokens: 24, temperature: 0, prefillStepSize: step)

                // The logits that predict the first reply token.
                let oneShotLogits = TinyModels.logits(model, prompt, cache: model.newCache(parameters: nil))[(count - 1)...]
                let prefilled = try StockPrefill.run(head, model: model, parameters: parameters) { true }
                let offsets = Self.attentionOffsets(prefilled)
                XCTAssertFalse(offsets.isEmpty, name)
                XCTAssertEqual(offsets, Array(repeating: count - 1, count: offsets.count), name)
                let chunkedLogits = TinyModels.logits(model, [prompt[count - 1]], cache: prefilled)
                XCTAssertTrue(
                    LogitCheck.isClose(chunkedLogits, oneShotLogits, rtol: 1e-4, atol: 1e-4),
                    "\(name), first step: max |Δ| = \(LogitCheck.maxAbsDifference(chunkedLogits, oneShotLogits))")

                // Greedy generation, then the caches both runs leave.
                let oneShotCache = model.newCache(parameters: parameters)
                let expected = Array(try TokenIterator(
                    input: LMInput(tokens: Self.array(prompt)), model: model, cache: oneShotCache, parameters: parameters))
                XCTAssertEqual(expected.count, 24, name)
                let cache = try StockPrefill.run(head, model: model, parameters: parameters) { true }
                let tokens = Array(try TokenIterator(input: LMInput(tokens: last), model: model, cache: cache, parameters: parameters))
                XCTAssertEqual(tokens, expected, name)
                XCTAssertEqual(Self.attentionOffsets(cache), Self.attentionOffsets(oneShotCache), name)
                let probe = [65, 66, 67]
                let after = TinyModels.logits(model, probe, cache: cache)
                let oneShotAfter = TinyModels.logits(model, probe, cache: oneShotCache)
                XCTAssertTrue(
                    LogitCheck.isClose(after, oneShotAfter, rtol: 1e-4, atol: 1e-4),
                    "\(name), after the reply: max |Δ| = \(LogitCheck.maxAbsDifference(after, oneShotAfter))")
            }
        }
    }

    func testARefusalStopsBetweenChunks() throws {
        for (label, model) in try Self.models() {
            let tokens = Self.array(TinyModels.tokens(40, seed: 5))
            let parameters = GenerateParameters(prefillStepSize: 8)
            for allowed in [0, 2, 4] {
                let name = "\(label), \(allowed) chunks allowed"
                let observed = ObservedModel(model)
                var checks = 0
                XCTAssertThrowsError(try StockPrefill.run(tokens, model: observed, parameters: parameters) {
                    checks += 1
                    return checks <= allowed
                }, name) { error in
                    XCTAssertEqual(error as? EngineError, .leftForeground, name)
                }
                XCTAssertEqual(checks, allowed + 1, name)
                XCTAssertEqual(observed.chunks, Array(repeating: 8, count: allowed), "\(name): nothing is fed after the refusal")
                let offsets = Self.attentionOffsets(observed.caches)
                XCTAssertFalse(offsets.isEmpty, name)
                XCTAssertEqual(offsets, Array(repeating: allowed * 8, count: offsets.count), name)
            }

            // Uninterrupted: one question per chunk, none after the last.
            let observed = ObservedModel(model)
            var checks = 0
            _ = try StockPrefill.run(tokens, model: observed, parameters: parameters) {
                checks += 1
                return true
            }
            XCTAssertEqual(checks, 5, label)
            XCTAssertEqual(observed.chunks, [8, 8, 8, 8, 8], label)
        }
    }
}

/// Passes everything to `inner`, recording the caches it makes and the size of each forward.
private final class ObservedModel: Module, LanguageModel {
    private let inner: any LanguageModel
    private(set) var caches: [KVCache] = []
    private(set) var chunks: [Int] = []

    init(_ inner: any LanguageModel) {
        self.inner = inner
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        try inner.prepare(input, cache: cache, windowSize: windowSize)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        chunks.append(input.tokens.dim(-1))
        return inner(input, cache: cache, state: state)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        caches = inner.newCache(parameters: parameters)
        return caches
    }
}
