import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

final class EngineCacheTests: XCTestCase {
    override func setUpWithError() throws {
        try MetalAvailability.require()
    }

    func testDetectFindsAttentionAndRecurrentLayers() throws {
        let hybridCache = try TinyModels.makeStockHybrid(seed: 1).newCache(parameters: nil)
        let hybrid = try XCTUnwrap(CacheLayout.detect(hybridCache))
        XCTAssertEqual(hybrid.attention, [1, 3])
        XCTAssertEqual(hybrid.recurrent, [0, 2])
        XCTAssertTrue(hybrid.isHybrid)

        let qwen3Cache = try TinyModels.makeStockQwen3(seed: 1).newCache(parameters: nil)
        let qwen3 = try XCTUnwrap(CacheLayout.detect(qwen3Cache))
        XCTAssertEqual(qwen3, CacheLayout(attention: [0, 1, 2, 3], recurrent: []))
        XCTAssertFalse(qwen3.isHybrid)

        XCTAssertEqual(CacheLayout.detect([MambaCache(), KVCacheSimple()]), CacheLayout(attention: [1], recurrent: [0]))
    }

    func testDetectRejectsCachesThatMoveOrRewriteTheirContents() {
        XCTAssertNil(CacheLayout.detect([KVCacheSimple(), RotatingKVCache(maxSize: 16)]))
        XCTAssertNil(CacheLayout.detect([RotatingKVCache(maxSize: 16, keep: 4)]))
        XCTAssertNil(CacheLayout.detect([QuantizedKVCache()]))
        XCTAssertNil(CacheLayout.detect([ChunkedKVCache(chunkSize: 8)]))
        XCTAssertNil(CacheLayout.detect([ArraysCache(size: 2)]))
    }

    /// Snapshot, advance 5 tokens, restore: the next logits equal those at the snapshot point
    /// exactly, for the hybrid and for Qwen3.
    func testSnapshotAdvanceRestoreIsExact() throws {
        let models: [any LanguageModel] = try [TinyModels.makeStockHybrid(seed: 21), TinyModels.makeStockQwen3(seed: 21)]
        for model in models {
            let name = "\(type(of: model))"
            let cache = model.newCache(parameters: nil)
            let layout = try XCTUnwrap(CacheLayout.detect(cache))
            let prompt = TinyModels.tokens(7, seed: 22)
            eval(TinyModels.logits(model, prompt, cache: cache))

            let snapshot = EngineCacheOps.snapshot(cache, layout: layout, position: prompt.count)
            XCTAssertEqual(snapshot.position, prompt.count)
            XCTAssertEqual(snapshot.bytes, EngineCacheOps.recurrentBytes(cache, layout: layout))
            if layout.isHybrid {
                XCTAssertGreaterThan(snapshot.bytes, 0, name)
            } else {
                XCTAssertEqual(snapshot.bytes, 0, name)
            }

            // Advance 5 tokens; the first one's logits are the reference.
            let advance = TinyModels.tokens(5, seed: 23)
            let expected = TinyModels.logits(model, [advance[0]], cache: cache)
            eval(expected)
            eval(TinyModels.logits(model, Array(advance[1...]), cache: cache))
            for index in layout.attention {
                XCTAssertEqual(cache[index].offset, prompt.count + advance.count, name)
            }

            EngineCacheOps.restore(cache, layout: layout, to: snapshot, currentPosition: prompt.count + advance.count)
            for index in layout.attention {
                XCTAssertEqual(cache[index].offset, prompt.count, name)
            }
            let actual = TinyModels.logits(model, [advance[0]], cache: cache)
            XCTAssertEqual(LogitCheck.maxAbsDifference(actual, expected), 0, name)
        }
    }

    /// A snapshot of an empty cache (no recurrent state yet) restores to a cache that behaves
    /// exactly like a new one.
    func testRestoringAnEmptySnapshotEqualsAFreshCache() throws {
        let models: [any LanguageModel] = try [TinyModels.makeStockHybrid(seed: 31), TinyModels.makeStockQwen3(seed: 31)]
        for model in models {
            let name = "\(type(of: model))"
            let prompt = TinyModels.tokens(6, seed: 32)
            let expected = TinyModels.logits(model, prompt, cache: model.newCache(parameters: nil))
            eval(expected)

            let cache = model.newCache(parameters: nil)
            let layout = try XCTUnwrap(CacheLayout.detect(cache))
            let empty = EngineCacheOps.snapshot(cache, layout: layout, position: 0)
            XCTAssertEqual(empty.bytes, 0, name)
            eval(TinyModels.logits(model, TinyModels.tokens(5, seed: 33), cache: cache))
            EngineCacheOps.restore(cache, layout: layout, to: empty, currentPosition: 5)
            XCTAssertEqual(EngineCacheOps.recurrentBytes(cache, layout: layout), 0, name)

            let actual = TinyModels.logits(model, prompt, cache: cache)
            XCTAssertEqual(LogitCheck.maxAbsDifference(actual, expected), 0, name)
        }
    }

    func testTrimAttentionLeavesRecurrentSlotsAlone() throws {
        let model = try TinyModels.makeStockHybrid(seed: 41)
        let cache = model.newCache(parameters: nil)
        let layout = try XCTUnwrap(CacheLayout.detect(cache))
        eval(TinyModels.logits(model, TinyModels.tokens(6, seed: 42), cache: cache))
        let recurrent = try XCTUnwrap(cache[0] as? MambaCache)
        let state = try XCTUnwrap(recurrent[1])
        let bytes = EngineCacheOps.recurrentBytes(cache, layout: layout)
        XCTAssertGreaterThan(bytes, 0)

        EngineCacheOps.trimAttention(cache, layout: layout, by: 2)
        XCTAssertEqual(cache.map { $0.offset }, [0, 4, 0, 4])
        XCTAssertTrue(recurrent[1] === state)
        XCTAssertEqual(EngineCacheOps.recurrentBytes(cache, layout: layout), bytes)

        EngineCacheOps.trimAttention(cache, layout: layout, by: 0)
        XCTAssertEqual(cache.map { $0.offset }, [0, 4, 0, 4])
    }
}
