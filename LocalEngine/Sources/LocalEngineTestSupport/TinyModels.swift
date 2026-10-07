import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Random-weight float32 models small enough for unit tests, built with the stock mlx-swift-lm
/// model code.
public enum TinyModels {
    public static let vocabularySize = 128

    /// Qwen3.5 text model: layers alternate gated-delta (even) and attention (odd). The
    /// gated-delta kernel needs key and value head dims that are multiples of 32.
    public static let hybridTextConfigJSON = """
        {
          "model_type": "qwen3_5_text",
          "hidden_size": 64,
          "num_hidden_layers": 4,
          "full_attention_interval": 2,
          "intermediate_size": 128,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 32,
          "linear_num_key_heads": 2,
          "linear_num_value_heads": 4,
          "linear_key_head_dim": 32,
          "linear_value_head_dim": 32,
          "linear_conv_kernel_dim": 4,
          "rms_norm_eps": 1e-6,
          "vocab_size": 128,
          "max_position_embeddings": 4096,
          "tie_word_embeddings": true,
          "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25
          }
        }
        """

    /// The same model in the `qwen3_5` wrapper form (`language_model` + `text_config`).
    public static var hybridWrapperConfigJSON: String {
        """
        {
          "model_type": "qwen3_5",
          "text_config": \(hybridTextConfigJSON)
        }
        """
    }

    /// A Qwen3 model of the same size: plain attention only, so its cache is trimmable.
    public static let qwen3ConfigJSON = """
        {
          "model_type": "qwen3",
          "hidden_size": 64,
          "num_hidden_layers": 4,
          "intermediate_size": 128,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 32,
          "rms_norm_eps": 1e-6,
          "vocab_size": 128,
          "max_position_embeddings": 4096,
          "rope_theta": 1000000,
          "tie_word_embeddings": true
        }
        """

    public static func hybridTextConfiguration() throws -> Qwen35TextConfiguration {
        try JSONDecoder.json5().decode(Qwen35TextConfiguration.self, from: Data(hybridTextConfigJSON.utf8))
    }

    public static func hybridWrapperConfiguration() throws -> Qwen35Configuration {
        try JSONDecoder.json5().decode(Qwen35Configuration.self, from: Data(hybridWrapperConfigJSON.utf8))
    }

    public static func qwen3Configuration() throws -> Qwen3Configuration {
        try JSONDecoder.json5().decode(Qwen3Configuration.self, from: Data(qwen3ConfigJSON.utf8))
    }

    /// The stock hybrid (`Qwen35TextModel`) with weights drawn from `seed`.
    public static func makeStockHybrid(seed: UInt64 = 0) throws -> Qwen35TextModel {
        let configuration = try hybridTextConfiguration()
        return build(seed: seed) { Qwen35TextModel(configuration) }
    }

    /// The stock hybrid in wrapper form (`Qwen35Model`) with weights drawn from `seed`.
    public static func makeStockHybridWrapper(seed: UInt64 = 0) throws -> Qwen35Model {
        let configuration = try hybridWrapperConfiguration()
        return build(seed: seed) { Qwen35Model(configuration) }
    }

    /// The stock Qwen3 model with weights drawn from `seed`.
    public static func makeStockQwen3(seed: UInt64 = 0) throws -> Qwen3Model {
        let configuration = try qwen3Configuration()
        return build(seed: seed) { Qwen3Model(configuration) }
    }

    /// Multiplies the token embedding (also the head, since the tiny models tie them) by
    /// `factor`, so greedy margins are large and near-ties rare. Call before `quantize4bit`.
    public static func sharpen(_ model: Module, factor: Float = 8) {
        let updates = model.parameters().flattened().compactMap { (key, value) -> (String, MLXArray)? in
            guard key.hasSuffix("embed_tokens.weight") else { return nil }
            precondition(value.dtype.isFloatingPoint, "Sharpen before quantizing: \(key) is \(value.dtype).")
            return (key, value * factor)
        }
        precondition(!updates.isEmpty, "The model has no embed_tokens.weight to sharpen.")
        model.update(parameters: ModuleParameters.unflattened(updates))
        eval(model)
    }

    /// Quantizes every `Linear` and `Embedding` to 4 bits with group size 64, as the MLX
    /// community checkpoints are.
    public static func quantize4bit(_ model: Module) {
        quantize(model: model, groupSize: 64, bits: 4)
        eval(model)
    }

    /// Deterministic token ids in the printable ASCII range (32...126), which the fake tokenizer
    /// decodes as text.
    public static func tokens(_ count: Int, seed: UInt64) -> [Int] {
        var state = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return (0 ..< count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return 32 + Int((state >> 33) % 95)
        }
    }

    /// One batch-1 forward of `tokens` through `model`, appending to `cache`. Returns float32
    /// logits `[S, V]`.
    public static func logits(_ model: any LanguageModel, _ tokens: [Int], cache: [KVCache]?) -> MLXArray {
        let input = MLXArray(tokens.map { Int32($0) })[.newAxis]
        return model(input, cache: cache)[0].asType(.float32)
    }

    private static func build<M: Module>(seed: UInt64, _ make: () -> M) -> M {
        MLXRandom.seed(seed)
        let model = make()
        eval(model)
        return model
    }
}
