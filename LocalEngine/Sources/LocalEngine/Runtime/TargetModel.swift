import MLX
import MLXLMCommon

/// Which rows of a forward pass need logits.
public enum LogitRows: Equatable, Sendable {
    /// No logits: the pass only fills the cache (intermediate prefill chunks).
    case none
    /// Only the last row's logits, computed by slicing the hidden state before the head.
    case last
    /// One row of logits per input token (verification of drafted tokens).
    case all
}

/// What a forward pass recorded so that a later `commit` can roll the recurrent state back to
/// any prefix of the verified tokens (plan §4.3). Concrete types belong to the target.
public protocol RoundCapture: AnyObject {}

public struct ForwardResult {
    /// `[rows, V]`: `[1, V]` for `.last`, `[S, V]` for `.all`, nil for `.none`.
    public let logits: MLXArray?
    /// `[rows, H]` post-final-norm hidden states, when asked for and supported.
    public let hidden: MLXArray?
    /// Present when `captureForRollback` was asked for and the target supports rollback.
    public let capture: (any RoundCapture)?

    public init(logits: MLXArray?, hidden: MLXArray? = nil, capture: (any RoundCapture)? = nil) {
        self.logits = logits
        self.hidden = hidden
        self.capture = capture
    }
}

/// The model the engine decodes with, together with its cache.
///
/// All calls stay lazy: nothing here evaluates arrays unless documented. Positions are tracked
/// by the caller (the session's ledger), never read from `MambaCache.offset`, which stays 0.
public protocol TargetModel: AnyObject {
    var model: any LanguageModel { get }
    var cache: [KVCache] { get }
    var layout: CacheLayout { get }
    var vocabularySize: Int { get }
    /// Whether `commit` can keep fewer tokens than were verified.
    var supportsRollback: Bool { get }
    /// Whether `.last` and `.none` avoid computing the other rows' logits.
    var supportsRowSelection: Bool { get }

    /// Feeds `tokens` (1-D int32 `[S]`) through the model, appending them to the cache.
    func forward(_ tokens: MLXArray, rows: LogitRows, captureForRollback: Bool, wantHidden: Bool) -> ForwardResult
    /// Keeps the first `keep` of the `verified` tokens fed by the last forward and drops the rest
    /// from the cache. `keep == verified` changes nothing.
    func commit(_ capture: (any RoundCapture)?, keep: Int, of verified: Int)
    /// Replaces the cache with an empty one.
    func resetCache()
    /// Takes over `cache` (for example a persisted prefix); throws if its layout doesn't match.
    func adopt(_ cache: [KVCache]) throws
}
