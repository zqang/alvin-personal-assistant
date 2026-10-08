import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

/// The `HybridQwen35` fork computes exactly what the stock Qwen3.5 model computes (plan §6.4:
/// `max |Δ| == 0`), through its stock entry point and through `engineForward(rows: .all)` with
/// capture on, in float32 and 4-bit, for the text model and the wrapper.
final class HybridParityTests: XCTestCase {
    override func setUpWithError() throws {
        try MetalAvailability.require()
    }

    /// The fork's module tree has the stock keys and shapes, so stock checkpoints load unchanged.
    func testParameterKeysAndShapesMatchStock() throws {
        let stockText = try TinyModels.makeStockHybrid(seed: 1)
        let forkText = try TinyForkModels.makeForkHybrid(seed: 2)
        assertSameParameters(stock: stockText, fork: forkText, label: "text")

        let stockWrapper = try TinyModels.makeStockHybridWrapper(seed: 1)
        let forkWrapper = try TinyForkModels.makeForkHybridWrapper(seed: 2)
        assertSameParameters(stock: stockWrapper, fork: forkWrapper, label: "wrapper")
        XCTAssertTrue(shapes(forkWrapper).keys.allSatisfy { $0.hasPrefix("language_model.") })
    }

    func testTextModelMatchesStockExactlyInFloat32() throws {
        let stock = try TinyModels.makeStockHybrid(seed: 11)
        let fork = try TinyForkModels.fork(of: stock)
        try assertExactParity(stock: stock, fork: fork, label: "text, float32")
    }

    func testTextModelMatchesStockExactlyIn4Bit() throws {
        let stock = try TinyModels.makeStockHybrid(seed: 12)
        let fork = try TinyForkModels.makeForkHybrid(seed: 99)
        TinyModels.quantize4bit(stock)
        TinyModels.quantize4bit(fork)
        try TinyForkModels.copyWeights(from: stock, to: fork)
        try assertExactParity(stock: stock, fork: fork, label: "text, 4-bit")
    }

    func testWrapperMatchesStockExactlyInFloat32() throws {
        let stock = try TinyModels.makeStockHybridWrapper(seed: 13)
        let fork = try TinyForkModels.fork(of: stock)
        try assertExactParity(stock: stock, fork: fork, label: "wrapper, float32")
    }

    func testWrapperMatchesStockExactlyIn4Bit() throws {
        let stock = try TinyModels.makeStockHybridWrapper(seed: 14)
        let fork = try TinyForkModels.makeForkHybridWrapper(seed: 98)
        TinyModels.quantize4bit(stock)
        TinyModels.quantize4bit(fork)
        try TinyForkModels.copyWeights(from: stock, to: fork)
        try assertExactParity(stock: stock, fork: fork, label: "wrapper, 4-bit")
    }

    /// After the same tokens, the fork's cache (filled by a `.none` engine forward with capture)
    /// holds exactly what the stock cache holds.
    func testCacheContentsMatchStockExactly() throws {
        let stock = try TinyModels.makeStockHybrid(seed: 15)
        let fork = try TinyForkModels.fork(of: stock)
        let stockCache = stock.newCache(parameters: nil)
        let forkCache = fork.newCache(parameters: nil)
        XCTAssertEqual(CacheLayout.detect(forkCache), CacheLayout.detect(stockCache))

        let tokens = TinyModels.tokens(9, seed: 16)
        eval(TinyModels.logits(stock, tokens, cache: stockCache))
        let result = TinyForkModels.engineLogits(fork, tokens, cache: forkCache, rows: .none, capture: GDNCaptureSink())
        XCTAssertNil(result.logits)
        eval(forkCache)
        eval(stockCache)

        XCTAssertEqual(forkCache.map { $0.offset }, stockCache.map { $0.offset })
        for index in 0 ..< stockCache.count {
            let forkState = forkCache[index].state
            let stockState = stockCache[index].state
            XCTAssertEqual(forkState.count, stockState.count, "layer \(index)")
            for slot in 0 ..< min(forkState.count, stockState.count) {
                let a = forkState[slot]
                let b = stockState[slot]
                XCTAssertTrue(
                    LogitCheck.isExactlyEqual(a, b),
                    "layer \(index) slot \(slot): max |Δ| = \(LogitCheck.maxAbsDifference(a, b))")
            }
        }
    }

    // MARK: Helpers

    private func shapes(_ model: Module) -> [String: [Int]] {
        var result: [String: [Int]] = [:]
        for (key, value) in model.parameters().flattened() {
            result[key] = value.shape
        }
        return result
    }

    /// Same keys and shapes in float32, and again after quantizing both to 4 bits.
    private func assertSameParameters(stock: Module, fork: Module, label: String) {
        XCTAssertEqual(shapes(fork), shapes(stock), label)
        TinyModels.quantize4bit(stock)
        TinyModels.quantize4bit(fork)
        XCTAssertEqual(shapes(fork), shapes(stock), "\(label), 4-bit")
    }

    /// Feeds 7, then 1, then 5 tokens through each model's own fresh cache: the stock model, the
    /// fork's `callAsFunction`, and the fork's `engineForward(rows: .all)` with capture on. All
    /// logits must be bitwise equal.
    private func assertExactParity(stock: any LanguageModel, fork: any HybridQwen35Forwarding, label: String) throws {
        let stockCache = stock.newCache(parameters: nil)
        let callCache = fork.newCache(parameters: nil)
        let engineCache = fork.newCache(parameters: nil)

        for (step, count) in [7, 1, 5].enumerated() {
            let tokens = TinyModels.tokens(count, seed: UInt64(30 + step))
            let expected = TinyModels.logits(stock, tokens, cache: stockCache)
            let viaCall = TinyModels.logits(fork, tokens, cache: callCache)
            let sink = GDNCaptureSink()
            let viaEngine = try XCTUnwrap(
                TinyForkModels.engineLogits(fork, tokens, cache: engineCache, rows: .all, capture: sink).logits)
            eval(expected, viaCall, viaEngine)

            XCTAssertEqual(expected.shape, [count, TinyModels.vocabularySize], label)
            XCTAssertTrue(
                LogitCheck.isExactlyEqual(viaCall, expected),
                "\(label), chunk \(step) (\(count) tokens), callAsFunction: max |Δ| = \(LogitCheck.maxAbsDifference(viaCall, expected))")
            XCTAssertTrue(
                LogitCheck.isExactlyEqual(viaEngine, expected),
                "\(label), chunk \(step) (\(count) tokens), engineForward(.all): max |Δ| = \(LogitCheck.maxAbsDifference(viaEngine, expected))")
            XCTAssertEqual(sink.entries.map { $0.layer }, [0, 2], label)
        }
    }
}
