import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

/// `HybridTarget`: row selection, hidden states, capture, cache handling and the registry that
/// creates it.
final class HybridTargetTests: XCTestCase {
    override func setUpWithError() throws {
        try MetalAvailability.require()
    }

    private static func int32(_ tokens: [Int]) -> MLXArray {
        MLXArray(tokens.map { Int32($0) })
    }

    func testTargetDescribesTheFork() throws {
        let target = HybridTarget(model: try TinyForkModels.makeForkHybrid(seed: 61))
        XCTAssertTrue(target.supportsRollback)
        XCTAssertTrue(target.supportsRowSelection)
        XCTAssertEqual(target.vocabularySize, TinyModels.vocabularySize)
        XCTAssertEqual(target.layout, CacheLayout(attention: [1, 3], recurrent: [0, 2]))
        XCTAssertEqual(target.cache.count, 4)
        XCTAssertTrue(target.model is HybridQwen35TextModel)

        let wrapper = HybridTarget(model: try TinyForkModels.makeForkHybridWrapper(seed: 61))
        XCTAssertEqual(wrapper.layout, CacheLayout(attention: [1, 3], recurrent: [0, 2]))
        XCTAssertTrue(wrapper.model is HybridQwen35Model)
    }

    func testRowModesGiveTheRightShapes() throws {
        let model = try TinyForkModels.makeForkHybrid(seed: 62)
        let tokens = TinyModels.tokens(5, seed: 63)
        let vocabulary = TinyModels.vocabularySize
        let hiddenSize = 64

        for wantHidden in [false, true] {
            let all = HybridTarget(model: model).forward(Self.int32(tokens), rows: .all, captureForRollback: false, wantHidden: wantHidden)
            let last = HybridTarget(model: model).forward(Self.int32(tokens), rows: .last, captureForRollback: false, wantHidden: wantHidden)
            let none = HybridTarget(model: model).forward(Self.int32(tokens), rows: .none, captureForRollback: false, wantHidden: wantHidden)

            XCTAssertEqual(all.logits?.shape, [5, vocabulary])
            XCTAssertEqual(last.logits?.shape, [1, vocabulary])
            XCTAssertNil(none.logits)
            for result in [all, last, none] {
                XCTAssertNil(result.capture)
                if wantHidden {
                    XCTAssertEqual(result.hidden?.shape, [5, hiddenSize])
                } else {
                    XCTAssertNil(result.hidden)
                }
            }
        }

        // A `[1, S]` input works too.
        let batched = HybridTarget(model: model).forward(Self.int32(tokens)[.newAxis], rows: .all, captureForRollback: false, wantHidden: false)
        XCTAssertEqual(batched.logits?.shape, [5, vocabulary])
    }

    /// `.last` (the hidden states sliced before the head) is allClose to the last row of `.all`,
    /// on an empty cache and on a filled one; the hidden states are the same whatever the rows.
    func testLastRowMatchesTheLastRowOfAll() throws {
        let model = try TinyForkModels.makeForkHybrid(seed: 64)
        let all = HybridTarget(model: model)
        let last = HybridTarget(model: model)
        for (step, count) in [7, 3, 1].enumerated() {
            let tokens = Self.int32(TinyModels.tokens(count, seed: UInt64(65 + step)))
            let full = all.forward(tokens, rows: .all, captureForRollback: false, wantHidden: true)
            let sliced = last.forward(tokens, rows: .last, captureForRollback: false, wantHidden: true)
            let fullLogits = try XCTUnwrap(full.logits)
            let lastLogits = try XCTUnwrap(sliced.logits)
            let expected = fullLogits[(count - 1)..., 0...]
            eval(expected, lastLogits)
            XCTAssertTrue(
                LogitCheck.isClose(lastLogits, expected, rtol: 1e-5, atol: 1e-5),
                "chunk \(step): max |Δ| = \(LogitCheck.maxAbsDifference(lastLogits, expected))")

            let fullHidden = try XCTUnwrap(full.hidden)
            let lastHidden = try XCTUnwrap(sliced.hidden)
            eval(fullHidden, lastHidden)
            XCTAssertTrue(LogitCheck.isExactlyEqual(lastHidden, fullHidden), "chunk \(step) hidden")
        }
    }

    /// `.none` computes no logits but still appends the tokens to the cache.
    func testNoneAdvancesTheCacheWithoutLogits() throws {
        let model = try TinyForkModels.makeForkHybrid(seed: 68)
        let prompt = TinyModels.tokens(8, seed: 69)
        let next = TinyModels.tokens(2, seed: 70)

        let cacheOnly = HybridTarget(model: model)
        let prefill = cacheOnly.forward(Self.int32(prompt), rows: .none, captureForRollback: false, wantHidden: false)
        XCTAssertNil(prefill.logits)
        XCTAssertNil(prefill.hidden)
        eval(cacheOnly.cache)
        XCTAssertEqual(cacheOnly.layout.attention.map { cacheOnly.cache[$0].offset }, [8, 8])
        for index in cacheOnly.layout.recurrent {
            let layer = try XCTUnwrap(cacheOnly.cache[index] as? MambaCache)
            XCTAssertNotNil(layer[0])
            XCTAssertNotNil(layer[1])
        }

        let withLogits = HybridTarget(model: model)
        _ = withLogits.forward(Self.int32(prompt), rows: .all, captureForRollback: false, wantHidden: false)

        let actual = try XCTUnwrap(cacheOnly.forward(Self.int32(next), rows: .all, captureForRollback: false, wantHidden: false).logits)
        let expected = try XCTUnwrap(withLogits.forward(Self.int32(next), rows: .all, captureForRollback: false, wantHidden: false).logits)
        eval(actual, expected)
        XCTAssertTrue(
            LogitCheck.isClose(actual, expected, rtol: 1e-5, atol: 1e-5),
            "max |Δ| = \(LogitCheck.maxAbsDifference(actual, expected))")
    }

    /// With capture on, the forward returns one entry per recurrent layer, in layer order, with
    /// the shapes `commit` relies on.
    func testCaptureRecordsEveryRecurrentLayer() throws {
        let target = HybridTarget(model: try TinyForkModels.makeForkHybrid(seed: 71))
        let first = target.forward(Self.int32(TinyModels.tokens(4, seed: 72)), rows: .last, captureForRollback: true, wantHidden: false)
        let firstSink = try XCTUnwrap(first.capture as? GDNCaptureSink)
        XCTAssertEqual(firstSink.entries.map { $0.layer }, [0, 2])
        for entry in firstSink.entries {
            XCTAssertNil(entry.initialState, "an empty cache starts from zeros")
            XCTAssertNil(entry.mask)
            XCTAssertEqual(entry.steps, 4)
            XCTAssertEqual(entry.convStateRows, 3)
            XCTAssertEqual(entry.convInput.shape, [1, 3 + 4, 2 * 64 + 128])
            XCTAssertEqual(entry.q.shape, [1, 4, 2, 32])
            XCTAssertEqual(entry.k.shape, [1, 4, 2, 32])
            XCTAssertEqual(entry.v.shape, [1, 4, 4, 32])
            XCTAssertEqual(entry.a.shape, [1, 4, 4])
            XCTAssertEqual(entry.b.shape, [1, 4, 4])
        }
        eval(target.cache)

        let second = target.forward(Self.int32(TinyModels.tokens(3, seed: 73)), rows: .all, captureForRollback: true, wantHidden: false)
        let secondSink = try XCTUnwrap(second.capture as? GDNCaptureSink)
        for (index, entry) in zip(target.layout.recurrent, secondSink.entries) {
            XCTAssertEqual(entry.steps, 3)
            let initial = try XCTUnwrap(entry.initialState)
            XCTAssertEqual(initial.dtype, .float32)
            XCTAssertEqual(initial.shape, [1, 4, 32, 32], "layer \(index)")
        }

        let plain = target.forward(Self.int32([65]), rows: .last, captureForRollback: false, wantHidden: false)
        XCTAssertNil(plain.capture)
    }

    /// `commit` trims the attention layers to the kept prefix.
    func testCommitTrimsAttentionLayers() throws {
        let target = HybridTarget(model: try TinyForkModels.makeForkHybrid(seed: 74))
        _ = target.forward(Self.int32(TinyModels.tokens(6, seed: 75)), rows: .none, captureForRollback: false, wantHidden: false)
        let verify = target.forward(Self.int32(TinyModels.tokens(5, seed: 76)), rows: .all, captureForRollback: true, wantHidden: false)
        eval(target.cache)
        target.commit(verify.capture, keep: 2, of: 5)
        eval(target.cache)
        XCTAssertEqual(target.layout.attention.map { target.cache[$0].offset }, [8, 8])
    }

    func testResetCacheStartsOver() throws {
        let model = try TinyForkModels.makeForkHybrid(seed: 77)
        let tokens = TinyModels.tokens(5, seed: 78)
        let target = HybridTarget(model: model)
        _ = target.forward(Self.int32(TinyModels.tokens(4, seed: 79)), rows: .none, captureForRollback: false, wantHidden: false)
        eval(target.cache)

        target.resetCache()
        XCTAssertEqual(target.cache.map { $0.offset }, [0, 0, 0, 0])
        for index in target.layout.recurrent {
            let layer = try XCTUnwrap(target.cache[index] as? MambaCache)
            XCTAssertNil(layer[0])
            XCTAssertNil(layer[1])
        }

        let actual = try XCTUnwrap(target.forward(Self.int32(tokens), rows: .all, captureForRollback: false, wantHidden: false).logits)
        let expected = try XCTUnwrap(HybridTarget(model: model).forward(Self.int32(tokens), rows: .all, captureForRollback: false, wantHidden: false).logits)
        eval(actual, expected)
        XCTAssertTrue(LogitCheck.isExactlyEqual(actual, expected), "max |Δ| = \(LogitCheck.maxAbsDifference(actual, expected))")
    }

    func testAdoptTakesAMatchingCacheAndRefusesOthers() throws {
        let model = try TinyForkModels.makeForkHybrid(seed: 80)
        let prompt = TinyModels.tokens(6, seed: 81)
        let next = TinyModels.tokens(2, seed: 82)

        let source = HybridTarget(model: model)
        _ = source.forward(Self.int32(prompt), rows: .none, captureForRollback: false, wantHidden: false)
        eval(source.cache)

        let reference = HybridTarget(model: model)
        _ = reference.forward(Self.int32(prompt), rows: .none, captureForRollback: false, wantHidden: false)
        let expected = try XCTUnwrap(reference.forward(Self.int32(next), rows: .all, captureForRollback: false, wantHidden: false).logits)

        let target = HybridTarget(model: model)
        try target.adopt(source.cache)
        let actual = try XCTUnwrap(target.forward(Self.int32(next), rows: .all, captureForRollback: false, wantHidden: false).logits)
        eval(actual, expected)
        XCTAssertTrue(LogitCheck.isExactlyEqual(actual, expected), "max |Δ| = \(LogitCheck.maxAbsDifference(actual, expected))")

        let expectedLayout = CacheLayout(attention: [1, 3], recurrent: [0, 2])
        let qwen3Cache = try TinyModels.makeStockQwen3(seed: 83).newCache(parameters: nil)
        XCTAssertThrowsError(try target.adopt(qwen3Cache)) { error in
            XCTAssertEqual(
                error as? TargetCacheError,
                .layoutMismatch(expected: expectedLayout, found: CacheLayout(attention: [0, 1, 2, 3], recurrent: [])))
        }
        XCTAssertThrowsError(try target.adopt([MambaCache(), KVCacheSimple()]))
        XCTAssertThrowsError(try target.adopt([MambaCache(), RotatingKVCache(maxSize: 16), MambaCache(), KVCacheSimple()])) { error in
            XCTAssertEqual(error as? TargetCacheError, .layoutMismatch(expected: expectedLayout, found: nil))
        }
    }

    /// The registry builds fork models from `config.json` data, and `makeTarget` gives a
    /// `HybridTarget` only for them.
    func testRegistryCreatesForkModelsAndTargets() async throws {
        let registry = EngineModelRegistry.makeTypeRegistry()
        let containsText = await registry.contains("qwen3_5_text")
        let containsWrapper = await registry.contains("qwen3_5")
        let containsMoE = await registry.contains("qwen3_5_moe")
        XCTAssertTrue(containsText)
        XCTAssertTrue(containsWrapper)
        XCTAssertFalse(containsMoE)

        let text = try await registry.createModel(configuration: Data(TinyModels.hybridTextConfigJSON.utf8), modelType: "qwen3_5_text")
        let wrapper = try await registry.createModel(configuration: Data(TinyModels.hybridWrapperConfigJSON.utf8), modelType: "qwen3_5")
        XCTAssertTrue(text is HybridQwen35TextModel)
        XCTAssertTrue(wrapper is HybridQwen35Model)
        XCTAssertTrue(EngineModelRegistry.isFork(text))
        XCTAssertTrue(EngineModelRegistry.isFork(wrapper))
        XCTAssertFalse(EngineModelRegistry.isFork(try TinyModels.makeStockHybrid(seed: 84)))
        XCTAssertFalse(EngineModelRegistry.isFork(try TinyModels.makeStockQwen3(seed: 84)))

        let forkTarget = EngineModelRegistry.makeTarget(for: try Self.loaded(text))
        XCTAssertTrue(forkTarget is HybridTarget)
        XCTAssertEqual(forkTarget?.layout, CacheLayout(attention: [1, 3], recurrent: [0, 2]))
        XCTAssertTrue(EngineModelRegistry.makeTarget(for: try Self.loaded(wrapper)) is HybridTarget)
        XCTAssertNil(EngineModelRegistry.makeTarget(for: try Self.loaded(TinyModels.makeStockHybrid(seed: 85))))
    }

    /// A `LoadedModel` around a tiny model, with the fake tokenizer.
    private static func loaded(_ model: any LanguageModel) throws -> LoadedModel {
        let directory = FileManager.default.temporaryDirectory
        let context = ModelContext(
            configuration: ModelConfiguration(directory: directory), model: model,
            processor: UnusedInputProcessor(), tokenizer: FakeChatMLTokenizer())
        return try ModelLoader.makeLoadedModel(context: context, id: "tiny", directory: directory, modelType: "qwen3_5_text")
    }
}

/// The engine renders prompts itself, so the tiny `LoadedModel`s never use their processor.
private struct UnusedInputProcessor: UserInputProcessor {
    struct Unused: Error {}

    func prepare(input: UserInput) async throws -> LMInput {
        throw Unused()
    }
}
