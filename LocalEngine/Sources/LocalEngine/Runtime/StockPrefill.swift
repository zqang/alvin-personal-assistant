import Foundation
import MLX
import MLXLMCommon

/// Prefills a prompt for MLX's stock `TokenIterator` in chunks that can be stopped between them.
/// The iterator prefills its whole input in `LLMModel.prepare`, one call that can't be
/// interrupted, so a long prompt could keep the GPU busy after the app left the foreground. The
/// app's stock path feeds all but the prompt's last token here and starts the iterator from that
/// token, so the cache ends up holding the same tokens as after a one-shot prefill.
public enum StockPrefill {
    /// Feeds `tokens` (1-D) into a fresh cache from `model.newCache(parameters:)` in chunks of
    /// `parameters.prefillStepSize`, as `LLMModel.prepare` does, and returns the cache. Asks
    /// `isAllowed` before every chunk and throws `EngineError.leftForeground` once it refuses.
    /// Each chunk is evaluated before the next question, so nothing is left running on the GPU
    /// then.
    public static func run(
        _ tokens: MLXArray, model: any LanguageModel, parameters: GenerateParameters, isAllowed: () -> Bool
    ) throws -> [KVCache] {
        precondition(tokens.ndim == 1, "StockPrefill takes 1-D tokens, as a text processor makes them.")
        let cache = model.newCache(parameters: parameters)
        let step = max(1, parameters.prefillStepSize)
        let count = tokens.size
        var start = 0
        var state: LMOutput.State?
        while start < count {
            guard isAllowed() else { throw EngineError.leftForeground }
            let end = min(start + step, count)
            let chunk = LMInput.Text(tokens: tokens[start ..< end])
            let output = model(chunk[.newAxis], cache: cache.isEmpty ? nil : cache, state: state)
            state = output.state
            // Synchronous, unlike `prepare`'s asyncEval: a refusal then waits for one chunk at most.
            eval(cache)
            start = end
        }
        return cache
    }
}
