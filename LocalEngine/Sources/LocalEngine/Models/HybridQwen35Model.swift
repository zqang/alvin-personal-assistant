// The engine's fork of the Qwen3.5 text model ("HybridQwen35"): the models.
//
// Copied from mlx-swift-lm 3.31.4 (tag 3.31.4), Libraries/MLXLLM/Models/Qwen35.swift
// (`Qwen35TextModelInner`, `Qwen35TextModel`, `Qwen35Model`).
// Changes:
// - `engineForward` computes the head only for the rows the engine needs (none, the last, or
//   all), can return the post-final-norm hidden states, and can record the gated-delta inputs
//   (`GDNCaptureSink`) so the recurrent state can be rolled back exactly;
// - `sanitize` uses the 3.32.3 rule: norm weights are shifted only for an unsanitized conv1d, not
//   whenever `mtp.*` tensors are present (on 3.31.4 an MLX checkpoint that kept `mtp.*` loads as
//   garbage, F7). Tied checkpoints drop every `lm_head` tensor, as 3.32.3 does.
// `callAsFunction`, `newCache` and every module key are unchanged, so the stock `TokenIterator`,
// `ChatSession`, `loadWeights` and quantize-by-path work with the fork as with the stock model.
//
// Original code under the MIT License:
//
// Copyright (c) 2024 ml-explore
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// A model built from the `HybridQwen35` fork: the stock interface plus the engine's forward.
public protocol HybridQwen35Forwarding: LanguageModel {
    var vocabularySize: Int { get }

    /// One forward pass for the engine.
    ///
    /// - Parameters:
    ///   - inputs: token ids `[1, S]`.
    ///   - cache: the model's cache (from `newCache`); the tokens are appended to it.
    ///   - rows: which rows get logits. `.last` slices the hidden states to the last row
    ///     **before** the head; `.none` skips the head entirely.
    ///   - capture: when given, every gated-delta layer records its inputs into it.
    ///   - wantHidden: whether to return the post-final-norm hidden states.
    /// - Returns: `logits` `[1, R, V]` (R = S for `.all`, 1 for `.last`; nil for `.none`) and
    ///   `hidden` `[1, S, H]` for every input row whatever `rows` is (nil unless `wantHidden`).
    ///   Everything stays lazy.
    func engineForward(
        _ inputs: MLXArray, cache: [KVCache], rows: LogitRows, capture: GDNCaptureSink?,
        wantHidden: Bool
    ) -> (logits: MLXArray?, hidden: MLXArray?)
}

// MARK: - Text model

/// The decoder stack (`model` in checkpoints): embedding, layers and final norm.
public final class HybridQwen35TextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [HQDecoderLayer]
    let norm: RMSNorm

    let ssmIdx: Int
    let faIdx: Int

    init(_ args: HybridQwen35TextConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0 ..< args.hiddenLayers).map { layerIdx in
            HQDecoderLayer(args, layerIdx: layerIdx)
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.ssmIdx = 0
        self.faIdx = args.fullAttentionInterval - 1

        super.init()
    }

    /// The stock forward: post-final-norm hidden states `[B, S, H]`.
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
        norm(layerOutputs(inputs, cache: cache, capture: nil))
    }

    /// The hidden states after the last decoder layer, before the final norm: `[B, S, H]`.
    func layerOutputs(_ inputs: MLXArray, cache: [KVCache?]?, capture: GDNCaptureSink?) -> MLXArray {
        var hiddenStates = embedTokens(inputs)

        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let faMask = createAttentionMask(h: hiddenStates, cache: cacheArray?[faIdx])
        let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

        for (i, layer) in layers.enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask =
                layer.isLinear
                ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = layer(
                hiddenStates, attentionMask: attnMask, ssmMask: mask, cache: cacheArray?[i],
                capture: capture)
        }

        return hiddenStates
    }
}

/// The Qwen3.5 text model (`model_type` `qwen3_5_text`), forked for the engine.
public final class HybridQwen35TextModel: Module, LLMModel, KVCacheDimensionProvider, HybridQwen35Forwarding {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: HybridQwen35TextModelInner
    public let configuration: HybridQwen35TextConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: HybridQwen35TextConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = HybridQwen35TextModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    /// The output head: `lm_head`, or the tied embedding.
    func head(_ hidden: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hidden)
        } else {
            return model.embedTokens.asLinear(hidden)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        head(model(inputs, cache: cache))
    }

    public func engineForward(
        _ inputs: MLXArray, cache: [KVCache], rows: LogitRows, capture: GDNCaptureSink?,
        wantHidden: Bool
    ) -> (logits: MLXArray?, hidden: MLXArray?) {
        let outputs = model.layerOutputs(inputs, cache: cache, capture: capture)
        let normed: MLXArray? = (wantHidden || rows == .all) ? model.norm(outputs) : nil

        let logits: MLXArray?
        switch rows {
        case .none:
            logits = nil
        case .all:
            logits = head(normed!)
        case .last:
            let last = outputs.dim(1) - 1
            if let normed {
                logits = head(normed[0..., last..., 0...])
            } else {
                logits = head(model.norm(outputs[0..., last..., 0...]))
            }
        }
        return (logits, wantHidden ? normed : nil)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return model.layers.map { layer in
            if layer.isLinear {
                return MambaCache()
            }
            return KVCacheSimple()
        }
    }

    /// Prepares checkpoint weights:
    /// - drops `mtp.*` tensors, and every `lm_head` tensor when the embeddings are tied;
    /// - when some `conv1d.weight` is in the original (unsanitized) layout, moves its kernel axis
    ///   and shifts the norm weights by +1. Only that layout marks an unconverted checkpoint; an
    ///   MLX checkpoint may keep `mtp.*` tensors, and shifting its norms again would ruin it (F7).
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let hasUnsanitizedConv1d = weights.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }
        let shouldShiftNormWeights = hasUnsanitizedConv1d

        var weights = weights.filter { !$0.key.contains("mtp.") }

        if configuration.tieWordEmbeddings {
            weights = weights.filter { key, _ in
                !key.split(separator: ".").contains("lm_head")
            }
        }

        let normKeys = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        for k in Array(weights.keys) {
            guard let v = weights[k] else { continue }
            if k.contains("conv1d.weight") && v.dim(-1) != 1 {
                weights[k] = v.movedAxis(source: 2, destination: 1)
                continue
            }
            if shouldShiftNormWeights
                && normKeys.contains(where: { k.hasSuffix($0) })
                && v.ndim == 1
            {
                weights[k] = v + MLXArray(1, dtype: v.dtype)
            }
        }

        return weights
    }
}

extension HybridQwen35TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Top-level model

/// The `qwen3_5` checkpoint form: the text model under `language_model`, forked for the engine.
/// Vision weights are dropped.
public final class HybridQwen35Model: Module, LLMModel, KVCacheDimensionProvider, HybridQwen35Forwarding {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: HybridQwen35TextModel

    public init(_ args: HybridQwen35Configuration) {
        let textModel = HybridQwen35TextModel(args.textConfig)
        self.vocabularySize = textModel.vocabularySize
        self.kvHeads = textModel.kvHeads
        _languageModel.wrappedValue = textModel
    }

    /// The text model this wrapper holds.
    public var textModel: HybridQwen35TextModel { languageModel }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func engineForward(
        _ inputs: MLXArray, cache: [KVCache], rows: LogitRows, capture: GDNCaptureSink?,
        wantHidden: Bool
    ) -> (logits: MLXArray?, hidden: MLXArray?) {
        languageModel.engineForward(inputs, cache: cache, rows: rows, capture: capture, wantHidden: wantHidden)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    /// Maps checkpoint keys to `language_model.…` exactly like the stock `Qwen35Model`, drops
    /// vision weights, then applies the text model's `sanitize`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("vision_tower") || key.hasPrefix("model.visual") {
                continue
            }

            var key = key
            if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if !key.hasPrefix("language_model.") {
                key = "language_model." + key
            }
            sanitized[key] = value
        }

        return languageModel.sanitize(weights: sanitized)
    }
}

extension HybridQwen35Model: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
