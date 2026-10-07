import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

/// One test per library fact the engine builds on (plan §2: F5, F6, F10), on tiny float32
/// models. If an upgrade of mlx-swift-lm changes one of these, this suite says which.
final class LibraryAssumptionTests: XCTestCase {
    override func setUpWithError() throws {
        try MetalAvailability.require()
    }

    /// 1. The hybrid's cache alternates `MambaCache` (gated-delta layers) and `KVCacheSimple`
    /// (every `full_attention_interval`-th layer), in both the text and the wrapper model.
    func testHybridCacheLayoutFollowsFullAttentionInterval() throws {
        let hybrids: [any LanguageModel] = try [TinyModels.makeStockHybrid(seed: 1), TinyModels.makeStockHybridWrapper(seed: 1)]
        for model in hybrids {
            let cache = model.newCache(parameters: nil)
            XCTAssertEqual(cache.count, 4)
            for (index, layer) in cache.enumerated() {
                if index % 2 == 0 {
                    XCTAssertTrue(type(of: layer) == MambaCache.self, "layer \(index) is \(type(of: layer))")
                } else {
                    XCTAssertTrue(type(of: layer) == KVCacheSimple.self, "layer \(index) is \(type(of: layer))")
                }
            }
            XCTAssertEqual(CacheLayout.detect(cache), CacheLayout(attention: [1, 3], recurrent: [0, 2]))
        }
        let qwen3 = try TinyModels.makeStockQwen3(seed: 1).newCache(parameters: nil)
        XCTAssertTrue(qwen3.allSatisfy { type(of: $0) == KVCacheSimple.self })
    }

    /// 2. `MambaCache.offset` stays 0 after forwards (positions must come from the ledger); an
    /// attention layer's offset counts the tokens fed.
    func testMambaOffsetStaysZeroWhileAttentionOffsetCountsTokens() throws {
        let model = try TinyModels.makeStockHybrid(seed: 2)
        let cache = model.newCache(parameters: nil)
        eval(TinyModels.logits(model, TinyModels.tokens(7, seed: 1), cache: cache))
        XCTAssertEqual(cache.map { $0.offset }, [0, 7, 0, 7])
        eval(TinyModels.logits(model, [65], cache: cache))
        XCTAssertEqual(cache.map { $0.offset }, [0, 8, 0, 8])
    }

    /// 3. A forward replaces both recurrent slots with new arrays and leaves the old ones as they
    /// were, so holding references is a valid snapshot.
    func testRecurrentSlotsAreReplacedNotMutated() throws {
        let model = try TinyModels.makeStockHybrid(seed: 3)
        let cache = model.newCache(parameters: nil)
        eval(TinyModels.logits(model, TinyModels.tokens(6, seed: 2), cache: cache))
        let layer = try XCTUnwrap(cache[0] as? MambaCache)
        let conv = try XCTUnwrap(layer[0])
        let state = try XCTUnwrap(layer[1])
        let convCopy = conv[.ellipsis]
        let stateCopy = state[.ellipsis]
        eval(convCopy, stateCopy)

        eval(TinyModels.logits(model, TinyModels.tokens(3, seed: 3), cache: cache))
        let newConv = try XCTUnwrap(layer[0])
        let newState = try XCTUnwrap(layer[1])
        XCTAssertFalse(newConv === conv)
        XCTAssertFalse(newState === state)
        XCTAssertTrue(LogitCheck.isExactlyEqual(conv, convCopy))
        XCTAssertTrue(LogitCheck.isExactlyEqual(state, stateCopy))
        XCTAssertGreaterThan(LogitCheck.maxAbsDifference(newState, state), 0)
    }

    /// 4. Trimming `KVCacheSimple` and re-feeding the same tokens gives exactly the logits of
    /// never trimming.
    func testTrimThenRefeedEqualsNeverTrimmingExactly() throws {
        let model = try TinyModels.makeStockQwen3(seed: 4)
        let prompt = TinyModels.tokens(8, seed: 4)
        let next = TinyModels.tokens(3, seed: 5)

        let plain = model.newCache(parameters: nil)
        eval(TinyModels.logits(model, prompt, cache: plain))
        let expected = TinyModels.logits(model, next, cache: plain)
        eval(expected)

        let trimmed = model.newCache(parameters: nil)
        eval(TinyModels.logits(model, prompt, cache: trimmed))
        eval(TinyModels.logits(model, next, cache: trimmed))
        for layer in trimmed {
            XCTAssertEqual(layer.trim(next.count), next.count)
        }
        XCTAssertEqual(trimmed.map { $0.offset }, [8, 8, 8, 8])
        let actual = TinyModels.logits(model, next, cache: trimmed)
        XCTAssertEqual(LogitCheck.maxAbsDifference(actual, expected), 0)
    }

    /// 5. One multi-token forward over a filled cache matches single-token forwards (allClose
    /// 1e-4), for the hybrid and for Qwen3.
    func testMultiTokenForwardMatchesSequentialForwards() throws {
        let models: [any LanguageModel] = try [TinyModels.makeStockHybrid(seed: 5), TinyModels.makeStockQwen3(seed: 5)]
        for model in models {
            let prefix = TinyModels.tokens(6, seed: 6)
            let next = TinyModels.tokens(5, seed: 7)

            let batched = model.newCache(parameters: nil)
            eval(TinyModels.logits(model, prefix, cache: batched))
            let together = TinyModels.logits(model, next, cache: batched)

            let sequential = model.newCache(parameters: nil)
            eval(TinyModels.logits(model, prefix, cache: sequential))
            var rows: [MLXArray] = []
            for token in next {
                let row = TinyModels.logits(model, [token], cache: sequential)
                eval(row)
                rows.append(row)
            }
            let oneByOne = concatenated(rows, axis: 0)
            XCTAssertTrue(
                LogitCheck.isClose(together, oneByOne, rtol: 1e-4, atol: 1e-4),
                "\(type(of: model)): max |Δ| = \(LogitCheck.maxAbsDifference(together, oneByOne))")
        }
    }

    /// 6. Stock speculative decoding refuses the hybrid: its cache can't be trimmed.
    func testOnlyPureAttentionCachesAreTrimmable() throws {
        let hybrid = try TinyModels.makeStockHybrid(seed: 6).newCache(parameters: nil)
        let qwen3 = try TinyModels.makeStockQwen3(seed: 6).newCache(parameters: nil)
        XCTAssertFalse(canTrimPromptCache(hybrid))
        XCTAssertTrue(canTrimPromptCache(qwen3))
    }

    /// 7. `savePromptCache` then `loadPromptCache` round-trips the hybrid's cache: the next
    /// step's logits are exactly equal.
    func testPromptCacheFileRoundTripIsExact() throws {
        let model = try TinyModels.makeStockHybrid(seed: 7)
        let cache = model.newCache(parameters: nil)
        eval(TinyModels.logits(model, TinyModels.tokens(9, seed: 8), cache: cache))

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("local-engine-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: cache, metadata: ["purpose": "round trip"])
        let (loaded, metadata) = try loadPromptCache(url: url)

        XCTAssertEqual(metadata["purpose"], "round trip")
        XCTAssertEqual(CacheLayout.detect(loaded), CacheLayout(attention: [1, 3], recurrent: [0, 2]))
        XCTAssertEqual(loaded.map { $0.offset }, cache.map { $0.offset })

        let expected = TinyModels.logits(model, [77], cache: cache)
        let actual = TinyModels.logits(model, [77], cache: loaded)
        XCTAssertEqual(LogitCheck.maxAbsDifference(actual, expected), 0)
    }

    /// 8. `gatedDeltaUpdate` over a prefix of length m gives bitwise the state of the masked full
    /// run (`mask = t < m`), on random bf16 inputs. This is what makes WP16's replay exact (F5).
    func testGatedDeltaPrefixSliceEqualsMaskedRunBitwise() throws {
        let steps = 6
        let keys = MLXRandom.split(key: MLXRandom.key(8), into: 8)
        let q = MLXRandom.normal([1, steps, 2, 32], key: keys[0]).asType(.bfloat16)
        let k = MLXRandom.normal([1, steps, 2, 32], key: keys[1]).asType(.bfloat16)
        let v = MLXRandom.normal([1, steps, 4, 32], key: keys[2]).asType(.bfloat16)
        let a = MLXRandom.normal([1, steps, 4], key: keys[3]).asType(.bfloat16)
        let b = MLXRandom.normal([1, steps, 4], key: keys[4]).asType(.bfloat16)
        let aLog = MLXRandom.normal([4], key: keys[5]).asType(.bfloat16)
        let dtBias = MLXRandom.normal([4], key: keys[6]).asType(.bfloat16)
        let initial = MLXRandom.normal([1, 4, 32, 32], key: keys[7]) * 0.1
        eval(q, k, v, a, b, aLog, dtBias, initial)

        for m in 1 ..< steps {
            let (prefixOut, prefixState) = gatedDeltaUpdate(
                q: q[0..., ..<m], k: k[0..., ..<m], v: v[0..., ..<m], a: a[0..., ..<m], b: b[0..., ..<m],
                aLog: aLog, dtBias: dtBias, state: initial)
            let mask = MLXArray((0 ..< steps).map { $0 < m }, [1, steps])
            let (maskedOut, maskedState) = gatedDeltaUpdate(
                q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: initial, mask: mask)
            eval(prefixOut, prefixState, maskedOut, maskedState)

            XCTAssertEqual(prefixState.dtype, .float32)
            XCTAssertTrue(LogitCheck.isExactlyEqual(prefixState, maskedState), "state, m = \(m): max |Δ| = \(LogitCheck.maxAbsDifference(prefixState, maskedState))")
            XCTAssertTrue(LogitCheck.isExactlyEqual(prefixOut, maskedOut[0..., ..<m]), "output, m = \(m)")
        }
    }
}
