import Foundation
import MLX
import MLXLMCommon

/// The engine's target for models that run on the stock mlx-swift-lm model code (Qwen3 and
/// every other family outside the `HybridQwen35` fork).
///
/// - `forward` calls `model(tokens[.newAxis], cache:)`. The stock code computes logits for every
///   row, so `.last` feeds all but the last `lastRowWindow` tokens cache-only first (their
///   logits are never evaluated, so the head never runs on them) and slices the last row of the
///   rest.
/// - `commit` trims every attention layer. Rolling back a recurrent layer needs the fork's
///   capture, so a hybrid model on the stock code (`supportsRollback == false`) never
///   speculates; it still reuses its session through checkpoints.
/// - A cache with layers other than `KVCacheSimple` / `MambaCache` (rotating, quantized) can't
///   be rewound exactly: `exactLayout` is false and the engine renders every reply into an empty
///   cache.
///
/// All work stays lazy. Not thread-safe: engine queue only.
public final class StockTarget: TargetModel {
    /// Rows whose logits `.last` computes before slicing.
    public static let lastRowWindow = 16

    public let model: any LanguageModel
    public private(set) var cache: [KVCache]
    public let layout: CacheLayout
    public let vocabularySize: Int
    /// Whether every cache layer is a `KVCacheSimple` or `MambaCache`.
    public let exactLayout: Bool

    /// Only a pure-attention cache can drop verified tokens by trimming.
    public var supportsRollback: Bool { exactLayout && !layout.isHybrid }
    /// The stock code computes the head for every row it is given.
    public var supportsRowSelection: Bool { false }

    /// A target over `model` with a fresh, empty cache. `vocabularySize` defaults to what
    /// `inferVocabularySize` finds.
    public init(model: any LanguageModel, vocabularySize: Int? = nil, directory: URL? = nil) {
        let cache = model.newCache(parameters: nil)
        let detected = CacheLayout.detect(cache)
        self.model = model
        self.cache = cache
        self.layout = detected ?? CacheLayout(attention: [], recurrent: [])
        self.exactLayout = detected != nil
        self.vocabularySize = vocabularySize ?? Self.inferVocabularySize(model: model, directory: directory) ?? 0
    }

    public func forward(_ tokens: MLXArray, rows: LogitRows, captureForRollback: Bool, wantHidden: Bool) -> ForwardResult {
        let flat = tokens.ndim == 2 ? tokens.reshaped([-1]) : tokens
        precondition(flat.ndim == 1, "The engine feeds one sequence: got shape \(tokens.shape).")
        let count = flat.dim(0)
        switch rows {
        case .none:
            _ = model(flat[.newAxis], cache: cache)
            return ForwardResult(logits: nil)
        case .all:
            let logits = model(flat[.newAxis], cache: cache)
            return ForwardResult(logits: logits.reshaped(-1, logits.dim(-1)))
        case .last:
            var tail = flat
            if count > Self.lastRowWindow {
                let split = count - Self.lastRowWindow
                _ = model(flat[..<split][.newAxis], cache: cache)
                tail = flat[split...]
            }
            let logits = model(tail[.newAxis], cache: cache)
            let last = logits.dim(1) - 1
            return ForwardResult(logits: logits[0..., last, 0...].reshaped(1, logits.dim(-1)))
        }
    }

    public func commit(_ capture: (any RoundCapture)?, keep: Int, of verified: Int) {
        precondition(keep >= 0 && keep <= verified, "Can't keep \(keep) of \(verified) tokens.")
        guard keep < verified else { return }
        precondition(supportsRollback, "This cache can't drop verified tokens.")
        EngineCacheOps.trimAttention(cache, layout: layout, by: verified - keep)
    }

    public func resetCache() {
        cache = model.newCache(parameters: nil)
    }

    /// Takes over `cache` (a persisted prefix); its layers must be the classes `newCache` makes.
    public func adopt(_ cache: [KVCache]) throws {
        let found = CacheLayout.detect(cache)
        guard exactLayout, found == layout, cache.count == self.cache.count else {
            throw TargetCacheError.layoutMismatch(expected: layout, found: found)
        }
        self.cache = cache
    }

    /// The vocabulary size from `config.json` (`vocab_size`, or `text_config.vocab_size`), else
    /// from the token embedding's first dimension.
    public static func inferVocabularySize(model: any LanguageModel, directory: URL?) -> Int? {
        if let directory,
           let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            if let size = json["vocab_size"] as? Int { return size }
            if let text = json["text_config"] as? [String: Any], let size = text["vocab_size"] as? Int { return size }
        }
        for (key, value) in model.parameters().flattened() where key.hasSuffix("embed_tokens.weight") {
            return value.dim(0)
        }
        return nil
    }
}
