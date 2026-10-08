import MLX
import MLXLMCommon

/// What one gated-delta layer handed to `gatedDeltaUpdate` during a forward pass of `S` tokens,
/// plus the recurrent state it started from. These are references to arrays of the pass's graph,
/// not copies: forwards replace cache slots instead of mutating them (F6), so the arrays keep
/// their values.
///
/// With it, the layer's recurrent state can be rebuilt for any prefix `m ≤ S` of the pass's
/// tokens without running the model again (plan §4.3): the conv state is a slice of `convInput`,
/// and the gated-delta state is `gatedDeltaUpdate` over the first `m` steps from `initialState`.
/// The kernel takes the step count as an input array, so the replay runs the same compiled
/// kernel with the same per-step arithmetic, and the state is bitwise the one the pass reached
/// after `m` steps (F5).
public struct GDNCapture {
    /// The model layer (and cache index) this entry belongs to.
    public let layer: Int
    /// `concat(convState, qkv)`: `[1, K − 1 + S, convDim]`, where `K` is the conv kernel size.
    public let convInput: MLXArray
    /// Queries after RMS normalization and scaling: `[1, S, Hk, Dk]`.
    public let q: MLXArray
    /// Keys after RMS normalization and scaling: `[1, S, Hk, Dk]`.
    public let k: MLXArray
    /// Values: `[1, S, Hv, Dv]`.
    public let v: MLXArray
    /// The raw `in_proj_a` output: `[1, S, Hv]`.
    public let a: MLXArray
    /// The raw `in_proj_b` output: `[1, S, Hv]`.
    public let b: MLXArray
    /// The layer's `A_log` parameter: `[Hv]`.
    public let aLog: MLXArray
    /// The layer's `dt_bias` parameter: `[Hv]`.
    public let dtBias: MLXArray
    /// The SSM mask passed to `gatedDeltaUpdate`: `[1, S]`, or nil (always nil for the engine,
    /// which never prepares cache lengths).
    public let mask: MLXArray?
    /// `cache[1]` before the pass: `[1, Hv, Dv, Dk]` float32, or nil for an empty cache (the
    /// kernel then starts from zeros).
    public let initialState: MLXArray?

    public init(
        layer: Int, convInput: MLXArray, q: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray, b: MLXArray,
        aLog: MLXArray, dtBias: MLXArray, mask: MLXArray?, initialState: MLXArray?
    ) {
        self.layer = layer
        self.convInput = convInput
        self.q = q
        self.k = k
        self.v = v
        self.a = a
        self.b = b
        self.aLog = aLog
        self.dtBias = dtBias
        self.mask = mask
        self.initialState = initialState
    }

    /// The number of tokens the pass fed (`S`).
    public var steps: Int { q.dim(1) }

    /// The number of rows the conv state holds (`K − 1`).
    public var convStateRows: Int { convInput.dim(1) - q.dim(1) }

    /// The conv state after the first `m` of the pass's tokens: the last `K − 1` rows of
    /// `concat(convState, qkv[..<m])`, which are rows `m ..< m + K − 1` of `convInput`. Lazy.
    public func convState(keeping m: Int) -> MLXArray {
        precondition(m >= 0 && m <= steps, "Can't keep \(m) of \(steps) tokens.")
        return contiguous(convInput[0..., m ..< (m + convStateRows), 0...])
    }

    /// The gated-delta state after the first `m` of the pass's tokens, replayed from
    /// `initialState` on the captured inputs. Lazy. `m == 0` gives `initialState` back.
    public func recurrentState(keeping m: Int) -> MLXArray? {
        precondition(m >= 0 && m <= steps, "Can't keep \(m) of \(steps) tokens.")
        guard m > 0 else { return initialState }
        let (_, state) = gatedDeltaUpdate(
            q: q[0..., ..<m],
            k: k[0..., ..<m],
            v: v[0..., ..<m],
            a: a[0..., ..<m],
            b: b[0..., ..<m],
            aLog: aLog,
            dtBias: dtBias,
            state: initialState,
            mask: mask.map { $0[0..., ..<m] }
        )
        return state
    }
}

/// Collects one `GDNCapture` per gated-delta layer, in layer order, during one forward pass of
/// the `HybridQwen35` fork. `HybridTarget.commit` uses it to roll the recurrent layers back to the
/// accepted prefix of a verified block.
public final class GDNCaptureSink: RoundCapture {
    public internal(set) var entries: [GDNCapture] = []

    public init() {}

    func record(_ entry: GDNCapture) {
        entries.append(entry)
    }
}
