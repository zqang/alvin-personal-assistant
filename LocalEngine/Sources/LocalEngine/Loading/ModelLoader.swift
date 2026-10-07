import Foundation
import MLXLLM
import MLXLMCommon

public enum EngineModelError: Error, Equatable {
    /// The engine's own model code can't run this checkpoint (for example a MoE variant); the
    /// loader then falls back to the stock model code.
    case unsupported(String)
}

/// A model loaded once and shared by the engine and the stock `ChatSession` fallback.
public struct LoadedModel: @unchecked Sendable {
    /// The catalog id, such as `mlx-community/Qwen3-0.6B-4bit`.
    public let id: String
    /// The snapshot directory the files were loaded from.
    public let directory: URL
    /// `model_type` from config.json.
    public let modelType: String
    public let model: any LanguageModel
    public let tokenizer: any MLXLMCommon.Tokenizer
    public let renderer: any ChatTemplateRendering
    /// EOS ids from config.json / generation_config.json, and the inferred tool-call format.
    public let configuration: ModelConfiguration
    /// The same model, for the `ChatSession` fallback (no second load).
    public let container: ModelContainer
    /// `eosTokenIds` ∪ the tokenizer's EOS token ∪ the ids of `<|im_end|>` and `<|endoftext|>`.
    public let stopTokenIDs: Set<Int>

    public init(
        id: String, directory: URL, modelType: String, model: any LanguageModel,
        tokenizer: any MLXLMCommon.Tokenizer, renderer: any ChatTemplateRendering,
        configuration: ModelConfiguration, container: ModelContainer, stopTokenIDs: Set<Int>
    ) {
        self.id = id
        self.directory = directory
        self.modelType = modelType
        self.model = model
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.configuration = configuration
        self.container = container
        self.stopTokenIDs = stopTokenIDs
    }
}

public enum ModelLoader {
    /// Loads the model in `directory` (already downloaded).
    ///
    /// When `typeRegistry` can create the checkpoint's `model_type`, it is used through a private
    /// `LLMModelFactory`, so the global `LLMTypeRegistry.shared` is never changed. If that model
    /// code throws `EngineModelError.unsupported`, the stock factory loads the model instead.
    /// No stop strings or extra EOS tokens are configured: a text-level stop filter would leave
    /// tokens in the cache that the reply never shows.
    public static func load(directory: URL, id: String, typeRegistry: ModelTypeRegistry<LanguageModel>? = nil) async throws -> LoadedModel {
        let configurationData = try Data(contentsOf: directory.appending(component: "config.json"))
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configurationData)
        let tokenizerLoader = TransformersTokenizerLoader()

        var context: ModelContext?
        if let typeRegistry, await typeRegistry.contains(base.modelType) {
            let factory = LLMModelFactory(typeRegistry: typeRegistry, modelRegistry: LLMRegistry.shared)
            do {
                context = try await factory.load(from: directory, using: tokenizerLoader)
            } catch is EngineModelError {
                context = nil
            }
        }
        if context == nil {
            context = try await LLMModelFactory.shared.load(from: directory, using: tokenizerLoader)
        }
        guard let context else {
            throw EngineModelError.unsupported(base.modelType)
        }
        return try makeLoadedModel(context: context, id: id, directory: directory, modelType: base.modelType)
    }

    /// Wraps an already loaded `ModelContext`. The tokenizer must also be a
    /// `ChatTemplateRendering` (the `TokenizerBridge` from `TransformersTokenizerLoader`, or a
    /// test tokenizer).
    public static func makeLoadedModel(context: ModelContext, id: String, directory: URL, modelType: String) throws -> LoadedModel {
        guard let renderer = context.tokenizer as? any ChatTemplateRendering else {
            throw EngineModelError.unsupported("a tokenizer without chat template rendering")
        }
        let model = context.model
        let tokenizer = context.tokenizer
        let configuration = context.configuration
        let stopTokenIDs = renderer.stopTokenIDs(eosTokenIds: configuration.eosTokenIds, eosToken: tokenizer.eosToken)
        return LoadedModel(
            id: id,
            directory: directory,
            modelType: modelType,
            model: model,
            tokenizer: tokenizer,
            renderer: renderer,
            configuration: configuration,
            container: ModelContainer(context: context),
            stopTokenIDs: stopTokenIDs
        )
    }
}
