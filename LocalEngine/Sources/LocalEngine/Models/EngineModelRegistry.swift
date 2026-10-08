import Foundation
import MLXLMCommon

/// Which model code the engine loads, and which target it decodes with.
///
/// The registry is private to the engine: `ModelLoader` hands it to its own `LLMModelFactory`,
/// so the global `LLMTypeRegistry.shared` (which the stock app path uses) is never changed.
public enum EngineModelRegistry {
    /// Creators for the `HybridQwen35` fork: `qwen3_5` (the `language_model` wrapper) and
    /// `qwen3_5_text`, decoding `config.json` with `JSONDecoder.json5()`. A mixture-of-experts
    /// configuration throws `EngineModelError.unsupported("moe")`, so `ModelLoader` falls back to
    /// the stock factory for it.
    public static func makeTypeRegistry() -> ModelTypeRegistry<LanguageModel> {
        ModelTypeRegistry<LanguageModel>(creators: [
            "qwen3_5": { data in
                let configuration = try JSONDecoder.json5().decode(HybridQwen35Configuration.self, from: data)
                return HybridQwen35Model(configuration)
            },
            "qwen3_5_text": { data in
                let configuration = try JSONDecoder.json5().decode(HybridQwen35TextConfiguration.self, from: data)
                return HybridQwen35TextModel(configuration)
            },
        ])
    }

    /// Whether `model` was built from the fork.
    public static func isFork(_ model: any LanguageModel) -> Bool {
        model is any HybridQwen35Forwarding
    }

    /// The engine target for `loaded`: a `HybridTarget` (with a fresh cache) for fork models;
    /// nil otherwise.
    public static func makeTarget(for loaded: LoadedModel) -> (any TargetModel)? {
        guard let fork = loaded.model as? any HybridQwen35Forwarding else { return nil }
        return HybridTarget(model: fork)
    }
}
