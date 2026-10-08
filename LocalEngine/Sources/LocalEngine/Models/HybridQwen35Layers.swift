// The engine's fork of the Qwen3.5 text model ("HybridQwen35"): the layers.
//
// Copied from mlx-swift-lm 3.31.4 (tag 3.31.4):
// - Libraries/MLXLLM/Models/Qwen35.swift (`Qwen35GatedDeltaNet`, `Qwen35Attention`,
//   `Qwen35DecoderLayer`)
// - Libraries/MLXLLM/Models/Qwen3Next.swift (`sigmoidMultiply`, `preciseSwiGLU`,
//   `Qwen3NextRMSNormGated`, `Qwen3NextMLP`)
// Changes: the types carry an `HQ` prefix; the gated-delta layer can record what it passes to
// `gatedDeltaUpdate` into a `GDNCaptureSink`; the decoder layer has no mixture-of-experts block
// (the configuration refuses one). Every module key and property name is unchanged, so
// `loadWeights` and quantize-by-path treat the fork exactly like the stock model.
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
import MLXLMCommon
import MLXNN

// MARK: - Helpers (Qwen3Next.swift)

func hqSigmoidMultiply(_ x: MLXArray, _ gate: MLXArray) -> MLXArray {
    x * sigmoid(gate)
}

private func hqPreciseSwiGLU(_ hiddenStates: MLXArray, gate: MLXArray, x: MLXArray) -> MLXArray {
    (silu(gate.asType(.float32)) * x.asType(.float32)).asType(hiddenStates.dtype)
}

// MARK: - Components (Qwen3Next.swift)

final class HQRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray? = nil) -> MLXArray {
        var x = MLXFast.rmsNorm(hiddenStates, weight: weight, eps: eps)
        if let gate {
            x = hqPreciseSwiGLU(hiddenStates, gate: gate, x: x)
        }
        return x
    }
}

final class HQMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - GatedDeltaNet (Qwen35.swift)

final class HQGatedDeltaNet: Module {
    let hiddenSize: Int
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int
    /// The decoder layer (and cache index) this block belongs to, for `GDNCapture.layer`.
    let layerIndex: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray

    @ModuleInfo(key: "norm") var norm: HQRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: HybridQwen35TextConfiguration, layerIndex: Int) {
        self.hiddenSize = args.hiddenSize
        self.numVHeads = args.linearNumValueHeads
        self.numKHeads = args.linearNumKeyHeads
        self.headKDim = args.linearKeyHeadDim
        self.headVDim = args.linearValueHeadDim
        self.keyDim = headKDim * numKHeads
        self.valueDim = headVDim * numVHeads
        self.convKernelSize = args.linearConvKernelDim
        self.convDim = keyDim * 2 + valueDim
        self.layerIndex = layerIndex

        precondition(
            numVHeads % numKHeads == 0,
            "num_v_heads (\(numVHeads)) must be divisible by num_k_heads (\(numKHeads))"
        )

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: convDim,
            bias: false
        )

        _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
        _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)

        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        let a = MLXRandom.uniform(low: 0, high: 16, [numVHeads])
        _aLog.wrappedValue = log(a)

        _norm.wrappedValue = HQRMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)

        super.init()
    }

    /// The stock computation. When `capture` is given, it also records the exact arrays passed to
    /// `gatedDeltaUpdate`, `concat(convState, qkv)`, and `cache[1]` from before the call. Recording
    /// only keeps references, so the computed values are the same either way.
    func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: MambaCache? = nil,
        capture: GDNCaptureSink? = nil
    ) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)

        var qkv = inProjQKV(inputs)
        let z = inProjZ(inputs).reshaped(B, S, numVHeads, headVDim)
        let b = inProjB(inputs)
        let a = inProjA(inputs)

        let convState: MLXArray
        if let cacheState = cache?[0] {
            convState = cacheState
        } else {
            convState = MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: inputs.dtype)
        }

        if let mask {
            qkv = MLX.where(mask[.ellipsis, .newAxis], qkv, 0)
        }

        let convInput = concatenated([convState, qkv], axis: 1)
        if let cache {
            cache[0] = contiguous(convInput[0..., (-(convKernelSize - 1))..., 0...])
        }

        let convOut = silu(conv1d(convInput))

        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = convSplit[0].reshaped(B, S, numKHeads, headKDim)
        let k = convSplit[1].reshaped(B, S, numKHeads, headKDim)
        let v = convSplit[2].reshaped(B, S, numVHeads, headVDim)

        var state = cache?[1]
        let initialState = state
        let dtype = q.dtype
        let invScale = pow(Float(headKDim), -0.5)
        let qNormed =
            MLXArray(pow(invScale, 2)).asType(dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        let kNormed =
            MLXArray(invScale).asType(dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

        if let capture {
            capture.record(
                GDNCapture(
                    layer: layerIndex, convInput: convInput, q: qNormed, k: kNormed, v: v, a: a, b: b,
                    aLog: aLog, dtBias: dtBias, mask: mask, initialState: initialState))
        }

        var out: MLXArray

        (out, state) = gatedDeltaUpdate(
            q: qNormed,
            k: kNormed,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: state,
            mask: mask
        )

        if let cache {
            cache[1] = state
            cache.advance(S)
        }

        out = norm(out, gate: z)
        return outProj(out.reshaped(B, S, -1))
    }
}

// MARK: - Attention (Qwen35.swift)

final class HQAttention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ args: HybridQwen35TextConfiguration) {
        let headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * headDim * 2, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(
            args.attentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeDims = Int(Float(headDim) * args.partialRotaryFactor)
        self.rope = initializeRope(
            dims: max(1, ropeDims),
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let qProjOutput = qProj(x)
        let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qSplit[0]
        let gate = qSplit[1].reshaped(B, L, -1)

        var keys = kProj(x)
        var values = vProj(x)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(hqSigmoidMultiply(output, gate))
    }
}

// MARK: - Decoder layer (Qwen35.swift)

final class HQDecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: HQAttention?
    @ModuleInfo(key: "linear_attn") var linearAttn: HQGatedDeltaNet?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    @ModuleInfo(key: "mlp") var mlp: HQMLP

    init(_ args: HybridQwen35TextConfiguration, layerIdx: Int) {
        self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

        if isLinear {
            _linearAttn.wrappedValue = HQGatedDeltaNet(args, layerIndex: layerIdx)
        } else {
            _selfAttn.wrappedValue = HQAttention(args)
        }

        // The configuration refuses `num_experts > 0`, so this is always the dense MLP.
        _mlp.wrappedValue = HQMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.intermediateSize
        )

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?,
        capture: GDNCaptureSink? = nil
    ) -> MLXArray {
        let r: MLXArray
        if isLinear {
            r = linearAttn!(
                inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache, capture: capture)
        } else {
            r = selfAttn!(inputLayerNorm(x), mask: attentionMask, cache: cache)
        }

        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}
