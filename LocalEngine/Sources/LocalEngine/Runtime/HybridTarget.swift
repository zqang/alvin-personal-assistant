import MLX
import MLXLMCommon

/// Why a target refused a cache handed to `adopt`.
public enum TargetCacheError: Error, Equatable {
    /// The cache's layers are not the classes, in the order, that the model's `newCache` makes
    /// (`found` is nil when some layer is neither `KVCacheSimple` nor `MambaCache`).
    case layoutMismatch(expected: CacheLayout, found: CacheLayout?)
}

/// The engine's target for models built from the `HybridQwen35` fork (Qwen3.5 text and its
/// `qwen3_5` wrapper): row-selective logits, and exact rollback of a verified block through
/// gated-delta capture and replay (plan §4.3).
///
/// All work stays lazy; the caller evaluates. Not thread-safe: use it from the engine queue only.
public final class HybridTarget: TargetModel {
    /// The fork model, as the engine's model protocol.
    public let fork: any HybridQwen35Forwarding
    public private(set) var cache: [KVCache]
    public let layout: CacheLayout
    public let vocabularySize: Int

    public var model: any LanguageModel { fork }
    /// `commit` can keep any prefix of a verified block: attention layers are trimmed and the
    /// recurrent layers are replayed from the round's capture.
    public var supportsRollback: Bool { true }
    /// `.last` slices the hidden states before the head and `.none` skips the head.
    public var supportsRowSelection: Bool { true }

    /// A target over `model` with a fresh, empty cache.
    public init(model: any HybridQwen35Forwarding) {
        let cache = model.newCache(parameters: nil)
        guard let layout = CacheLayout.detect(cache) else {
            preconditionFailure("The fork's newCache made layers other than KVCacheSimple and MambaCache.")
        }
        self.fork = model
        self.cache = cache
        self.layout = layout
        self.vocabularySize = model.vocabularySize
    }

    /// Feeds `tokens` (1-D int32 `[S]`, or `[1, S]`, with `S ≥ 1`) through the fork.
    ///
    /// - `logits`: `[S, V]` for `.all`, `[1, V]` for `.last`, nil for `.none`.
    /// - `hidden`: `[S, H]` post-final-norm states of every input row when `wantHidden`,
    ///   whatever `rows` is (a drafter that conditions on the target's hidden states needs every
    ///   fed position, even during a `.none` prefill).
    /// - `capture`: a `GDNCaptureSink` when `captureForRollback`, for `commit`.
    public func forward(_ tokens: MLXArray, rows: LogitRows, captureForRollback: Bool, wantHidden: Bool) -> ForwardResult {
        let inputs = tokens.ndim == 1 ? tokens[.newAxis] : tokens
        precondition(inputs.ndim == 2 && inputs.dim(0) == 1, "The engine feeds one sequence: got shape \(tokens.shape).")
        // An empty feed would launch zero-length gated-delta kernels, whose zero-size inputs
        // also change the kernel's generated source (see `commit`).
        precondition(inputs.dim(1) > 0, "Nothing to feed: got shape \(tokens.shape).")
        let sink = captureForRollback ? GDNCaptureSink() : nil
        let (logits, hidden) = fork.engineForward(inputs, cache: cache, rows: rows, capture: sink, wantHidden: wantHidden)
        return ForwardResult(
            logits: logits.map { $0.reshaped(-1, $0.dim(-1)) },
            hidden: hidden.map { $0.reshaped(-1, $0.dim(-1)) },
            capture: sink)
    }

    /// Keeps the first `keep` of the `verified` tokens that the last forward fed, and drops the
    /// rest:
    /// - every attention layer is trimmed by `verified − keep`;
    /// - every recurrent layer gets the conv state and gated-delta state it would hold after the
    ///   first `keep` tokens, replayed from the capture (bitwise the verify pass's state at that
    ///   step, F5).
    ///
    /// `keep == verified` changes nothing, and `keep == 0` puts back the pre-pass slots without
    /// launching a kernel. Rolling back needs the capture of that same forward. The results stay
    /// lazy. Evaluating them in one graph with forwards of other lengths is safe only for models
    /// with at least 8 value heads; the catalog's Qwen3.5 checkpoints have 16 or 32 (see
    /// `GDNCapture.recurrentState(keeping:)`).
    public func commit(_ capture: (any RoundCapture)?, keep: Int, of verified: Int) {
        precondition(keep >= 0 && keep <= verified, "Can't keep \(keep) of \(verified) tokens.")
        guard keep < verified else { return }

        EngineCacheOps.trimAttention(cache, layout: layout, by: verified - keep)
        guard layout.isHybrid else { return }

        guard let sink = capture as? GDNCaptureSink else {
            preconditionFailure("Rolling back a hybrid cache needs the GDNCaptureSink of the forward that fed the tokens.")
        }
        precondition(
            sink.entries.count == layout.recurrent.count,
            "The capture has \(sink.entries.count) recurrent layers; the cache has \(layout.recurrent.count).")

        for (index, entry) in zip(layout.recurrent, sink.entries) {
            precondition(entry.layer == index, "Capture entry for layer \(entry.layer) doesn't match cache layer \(index).")
            precondition(entry.steps == verified, "The capture fed \(entry.steps) tokens, not \(verified).")
            guard let layer = cache[index] as? ArraysCache else {
                preconditionFailure("Cache layer \(index) is not a recurrent cache.")
            }
            layer[0] = entry.convState(keeping: keep)
            layer[1] = entry.recurrentState(keeping: keep)
        }
    }

    public func resetCache() {
        cache = fork.newCache(parameters: nil)
    }

    /// Takes over `cache`, for example a persisted prefix. Its layers must be the classes the
    /// fork's `newCache` makes, in the same order.
    public func adopt(_ cache: [KVCache]) throws {
        let found = CacheLayout.detect(cache)
        guard found == layout, cache.count == self.cache.count else {
            throw TargetCacheError.layoutMismatch(expected: layout, found: found)
        }
        self.cache = cache
    }
}
