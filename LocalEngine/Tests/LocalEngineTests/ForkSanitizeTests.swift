import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

/// Loading the `HybridQwen35` fork: configuration decoding, the 3.32.3 sanitize rule (F7) and
/// the registry's refusal of mixture-of-experts checkpoints.
final class ForkSanitizeTests: XCTestCase {
    override func setUpWithError() throws {
        try MetalAvailability.require()
    }

    private static let convChannels = 256  // 2 × (2 × 32) + 4 × 32 in the tiny config
    private static let normKeys = [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.1.self_attn.q_norm.weight",
        "model.layers.1.self_attn.k_norm.weight",
        "model.norm.weight",
    ]

    // MARK: Sanitize

    /// An MLX checkpoint that kept `mtp.*` tensors (conv1d already `[C, K, 1]`): the fork drops
    /// the `mtp.*` tensors and leaves the norms alone. Stock 3.31.4 shifts them (F7).
    func testMTPWithSanitizedConvKeepsNormsAndDropsMTP() throws {
        let weights = Self.syntheticWeights(sanitizedConv: true, includeMTP: true)
        let fork = try TinyForkModels.makeForkHybrid(seed: 1)
        let sanitized = fork.sanitize(weights: weights)

        XCTAssertFalse(sanitized.keys.contains { $0.contains("mtp.") })
        for key in Self.normKeys {
            let value = try XCTUnwrap(sanitized[key], key)
            XCTAssertTrue(LogitCheck.isExactlyEqual(value, try XCTUnwrap(weights[key])), "\(key) changed")
        }
        let conv = try XCTUnwrap(sanitized["model.layers.0.linear_attn.conv1d.weight"])
        XCTAssertEqual(conv.shape, [Self.convChannels, 4, 1])
        XCTAssertTrue(LogitCheck.isExactlyEqual(conv, try XCTUnwrap(weights["model.layers.0.linear_attn.conv1d.weight"])))

        // The stock model shifts the norms of the same checkpoint by +1 (the F7 bug).
        let stock = try TinyModels.makeStockHybrid(seed: 1)
        let stockSanitized = stock.sanitize(weights: weights)
        let stockNorm = try XCTUnwrap(stockSanitized["model.norm.weight"])
        let original = try XCTUnwrap(weights["model.norm.weight"])
        XCTAssertTrue(LogitCheck.isExactlyEqual(stockNorm, Self.shifted(original)), "stock 3.31.4 no longer shifts: F7 may be fixed upstream")
    }

    /// An unconverted checkpoint (conv1d `[C, 1, K]`): the kernel axis moves and the norms
    /// shift by +1, as in stock.
    func testUnsanitizedConvIsMovedAndNormsAreShifted() throws {
        for includeMTP in [false, true] {
            let weights = Self.syntheticWeights(sanitizedConv: false, includeMTP: includeMTP)
            let fork = try TinyForkModels.makeForkHybrid(seed: 2)
            let sanitized = fork.sanitize(weights: weights)

            XCTAssertFalse(sanitized.keys.contains { $0.contains("mtp.") })
            let raw = try XCTUnwrap(weights["model.layers.0.linear_attn.conv1d.weight"])
            XCTAssertEqual(raw.shape, [Self.convChannels, 1, 4])
            let conv = try XCTUnwrap(sanitized["model.layers.0.linear_attn.conv1d.weight"])
            XCTAssertEqual(conv.shape, [Self.convChannels, 4, 1])
            XCTAssertTrue(LogitCheck.isExactlyEqual(conv, raw.movedAxis(source: 2, destination: 1)))
            // Element [c, 0, k] moved to [c, k, 0].
            XCTAssertEqual(conv[5, 3, 0].item(Float.self), raw[5, 0, 3].item(Float.self))

            for key in Self.normKeys {
                let value = try XCTUnwrap(sanitized[key], key)
                let original = try XCTUnwrap(weights[key])
                XCTAssertTrue(LogitCheck.isExactlyEqual(value, Self.shifted(original)), "\(key) not shifted (mtp: \(includeMTP))")
            }
            // Other one-dimensional weights are left alone.
            let dtBias = try XCTUnwrap(sanitized["model.layers.0.linear_attn.dt_bias"])
            XCTAssertTrue(LogitCheck.isExactlyEqual(dtBias, try XCTUnwrap(weights["model.layers.0.linear_attn.dt_bias"])))
        }
    }

    /// Tied embeddings drop every `lm_head` tensor (weights, scales and biases).
    func testTiedEmbeddingsDropLMHead() throws {
        var weights = Self.syntheticWeights(sanitizedConv: true, includeMTP: false)
        weights["lm_head.weight"] = MLXArray.zeros([128, 8], dtype: .uint32)
        weights["lm_head.scales"] = MLXArray.zeros([128, 1])
        weights["lm_head.biases"] = MLXArray.zeros([128, 1])
        let sanitized = try TinyForkModels.makeForkHybrid(seed: 3).sanitize(weights: weights)
        XCTAssertFalse(sanitized.keys.contains { $0.hasPrefix("lm_head") })

        let wrapped = try TinyForkModels.makeForkHybridWrapper(seed: 3).sanitize(weights: weights)
        XCTAssertFalse(wrapped.keys.contains { $0.contains("lm_head") })
    }

    /// The wrapper maps keys exactly like the stock `Qwen35Model` and drops vision tensors.
    func testWrapperRemapsKeysLikeStockAndDropsVision() throws {
        var weights: [String: MLXArray] = [:]
        for (key, value) in Self.syntheticWeights(sanitizedConv: true, includeMTP: true) {
            if key.hasPrefix("model.") {
                weights["model.language_model." + String(key.dropFirst("model.".count))] = value
            } else {
                weights[key] = value
            }
        }
        weights["vision_tower.blocks.0.weight"] = MLXArray.ones([4])
        weights["model.visual.patch_embed.weight"] = MLXArray.ones([4])
        weights["language_model.model.layers.2.input_layernorm.weight"] = MLXArray.ones([64])

        let fork = try TinyForkModels.makeForkHybridWrapper(seed: 4)
        let sanitized = fork.sanitize(weights: weights)
        let stock = try TinyModels.makeStockHybridWrapper(seed: 4)
        let stockSanitized = stock.sanitize(weights: weights)

        // Same keys as stock (stock keeps nothing the fork drops here, since stock drops mtp.*
        // too), and every key under `language_model.`.
        XCTAssertEqual(Set(sanitized.keys), Set(stockSanitized.keys))
        XCTAssertTrue(sanitized.keys.allSatisfy { $0.hasPrefix("language_model.") })
        XCTAssertFalse(sanitized.keys.contains { $0.contains("visual") || $0.contains("vision_tower") || $0.contains("mtp.") })
        XCTAssertNotNil(sanitized["language_model.model.layers.0.linear_attn.conv1d.weight"])
        XCTAssertNotNil(sanitized["language_model.model.norm.weight"])
        XCTAssertNotNil(sanitized["language_model.model.layers.2.input_layernorm.weight"])

        // Norms unchanged in the fork (sanitized conv1d), shifted by stock (F7).
        let original = try XCTUnwrap(weights["model.language_model.norm.weight"])
        XCTAssertTrue(LogitCheck.isExactlyEqual(try XCTUnwrap(sanitized["language_model.model.norm.weight"]), original))
        XCTAssertTrue(LogitCheck.isExactlyEqual(try XCTUnwrap(stockSanitized["language_model.model.norm.weight"]), Self.shifted(original)))
    }

    /// End to end: an MLX checkpoint of the stock model with `mtp.*` tensors added loads into
    /// the fork (every key used and set) and computes exactly the stock logits.
    func testCheckpointWithMTPLoadsIntoTheForkExactly() throws {
        let stock = try TinyModels.makeStockHybrid(seed: 5)
        var checkpoint: [String: MLXArray] = [:]
        for (key, value) in stock.parameters().flattened() {
            checkpoint[key] = value
        }
        checkpoint["mtp.fc.weight"] = MLXArray.ones([64, 128])
        checkpoint["mtp.layers.0.input_layernorm.weight"] = MLXArray.zeros([64])
        checkpoint["mtp.norm.weight"] = MLXArray.zeros([64])

        let fork = try TinyForkModels.makeForkHybrid(seed: 6)
        let sanitized = fork.sanitize(weights: checkpoint)
        try fork.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        eval(fork)

        let tokens = TinyModels.tokens(7, seed: 7)
        let expected = TinyModels.logits(stock, tokens, cache: stock.newCache(parameters: nil))
        let actual = TinyModels.logits(fork, tokens, cache: fork.newCache(parameters: nil))
        eval(expected, actual)
        XCTAssertTrue(LogitCheck.isExactlyEqual(actual, expected), "max |Δ| = \(LogitCheck.maxAbsDifference(actual, expected))")
    }

    // MARK: Configuration

    func testConfigurationDecodesTheTinyConfigLikeStock() throws {
        let text = try TinyForkModels.hybridTextConfiguration()
        XCTAssertEqual(text.modelType, "qwen3_5_text")
        XCTAssertEqual(text.hiddenSize, 64)
        XCTAssertEqual(text.hiddenLayers, 4)
        XCTAssertEqual(text.fullAttentionInterval, 2)
        XCTAssertEqual(text.attentionHeads, 2)
        XCTAssertEqual(text.kvHeads, 1)
        XCTAssertEqual(text.headDim, 32)
        XCTAssertEqual(text.linearNumKeyHeads, 2)
        XCTAssertEqual(text.linearNumValueHeads, 4)
        XCTAssertEqual(text.linearKeyHeadDim, 32)
        XCTAssertEqual(text.linearValueHeadDim, 32)
        XCTAssertEqual(text.linearConvKernelDim, 4)
        XCTAssertEqual(text.vocabularySize, 128)
        XCTAssertTrue(text.tieWordEmbeddings)
        XCTAssertEqual(text.numExperts, 0)
        // rope_parameters: theta and the partial factor come from it, and `rope_type` is copied
        // to `type`.
        XCTAssertEqual(text.ropeTheta, 10_000_000)
        XCTAssertEqual(text.partialRotaryFactor, 0.25)
        XCTAssertEqual(text.ropeScaling?["type"], .string("default"))
        XCTAssertEqual(text.ropeScaling?["rope_type"], .string("default"))

        let wrapper = try TinyForkModels.hybridWrapperConfiguration()
        XCTAssertEqual(wrapper.modelType, "qwen3_5")
        XCTAssertEqual(wrapper.textConfig.hiddenSize, 64)
        XCTAssertEqual(wrapper.textConfig.ropeTheta, 10_000_000)

        // A flat `qwen3_5` configuration (no text_config) is the text configuration itself.
        let flat = try JSONDecoder.json5().decode(
            HybridQwen35Configuration.self,
            from: Data(TinyModels.hybridTextConfigJSON.replacingOccurrences(of: "\"qwen3_5_text\"", with: "\"qwen3_5\"").utf8))
        XCTAssertEqual(flat.modelType, "qwen3_5")
        XCTAssertEqual(flat.textConfig.hiddenLayers, 4)
        XCTAssertEqual(flat.textConfig.linearNumValueHeads, 4)
    }

    func testConfigurationWithoutRopeParametersUsesTopLevelFieldsAndDefaults() throws {
        let json = """
            {"model_type": "qwen3_5_text", "hidden_size": 64, "num_attention_heads": 2, "rope_theta": 5000, "partial_rotary_factor": 0.5}
            """
        let configuration = try JSONDecoder.json5().decode(HybridQwen35TextConfiguration.self, from: Data(json.utf8))
        XCTAssertEqual(configuration.ropeTheta, 5000)
        XCTAssertEqual(configuration.partialRotaryFactor, 0.5)
        XCTAssertEqual(configuration.ropeScaling?["type"], .string("default"))
        XCTAssertEqual(configuration.ropeScaling?["mrope_section"], .ints([11, 11, 10]))
        XCTAssertEqual(configuration.headDim, 32)
        XCTAssertEqual(configuration.fullAttentionInterval, 4)
    }

    func testMixtureOfExpertsConfigurationIsRefused() async throws {
        let moe = TinyModels.hybridTextConfigJSON.replacingOccurrences(
            of: "\"num_hidden_layers\": 4,",
            with: "\"num_hidden_layers\": 4, \"num_experts\": 4, \"num_experts_per_tok\": 2, \"moe_intermediate_size\": 32, \"shared_expert_intermediate_size\": 32,")
        XCTAssertNotEqual(moe, TinyModels.hybridTextConfigJSON, "the tiny config changed; update this test")

        XCTAssertThrowsError(try JSONDecoder.json5().decode(HybridQwen35TextConfiguration.self, from: Data(moe.utf8))) { error in
            XCTAssertEqual(error as? EngineModelError, .unsupported("moe"))
        }
        let wrapped = "{\"model_type\": \"qwen3_5\", \"text_config\": \(moe)}"
        XCTAssertThrowsError(try JSONDecoder.json5().decode(HybridQwen35Configuration.self, from: Data(wrapped.utf8))) { error in
            XCTAssertEqual(error as? EngineModelError, .unsupported("moe"))
        }

        // Through the registry, the error reaches `ModelLoader` unchanged, so it falls back to
        // the stock factory.
        let registry = EngineModelRegistry.makeTypeRegistry()
        do {
            _ = try await registry.createModel(configuration: Data(wrapped.utf8), modelType: "qwen3_5")
            XCTFail("a MoE configuration was accepted")
        } catch {
            XCTAssertEqual(error as? EngineModelError, .unsupported("moe"))
        }
    }

    // MARK: Helpers

    /// A norm weight shifted by +1 in its own dtype, as sanitize does.
    private static func shifted(_ weight: MLXArray) -> MLXArray {
        weight + MLXArray(1, dtype: weight.dtype)
    }

    /// A few tensors with the tiny model's names and shapes, in an MLX checkpoint layout
    /// (`sanitizedConv`: conv1d `[C, K, 1]`) or the original layout (`[C, 1, K]`).
    private static func syntheticWeights(sanitizedConv: Bool, includeMTP: Bool) -> [String: MLXArray] {
        let key = MLXRandom.key(17)
        let conv = MLXRandom.normal(sanitizedConv ? [convChannels, 4, 1] : [convChannels, 1, 4], key: key)
        var weights: [String: MLXArray] = [
            "model.layers.0.linear_attn.conv1d.weight": conv,
            "model.layers.0.linear_attn.dt_bias": MLXArray([0.5, 0.25, -0.5, 1] as [Float]),
            "model.embed_tokens.weight": MLXArray.ones([128, 64]),
        ]
        for (index, normKey) in normKeys.enumerated() {
            let size = normKey.contains("_norm.weight") && normKey.contains("self_attn") ? 32 : 64
            weights[normKey] = MLXArray(Array(repeating: Float(index + 1) / 10, count: size))
        }
        if includeMTP {
            weights["mtp.fc.weight"] = MLXArray.ones([64, 128])
            weights["mtp.layers.0.input_layernorm.weight"] = MLXArray.zeros([64])
            weights["mtp.norm.weight"] = MLXArray.zeros([64])
        }
        eval(Array(weights.values))
        return weights
    }
}
