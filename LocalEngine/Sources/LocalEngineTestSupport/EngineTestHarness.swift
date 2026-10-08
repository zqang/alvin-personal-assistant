import AssistantKit
import Foundation
import LocalEngine
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Builds engines around the tiny random models and the fake ChatML tokenizer, for tests that
/// run without model files.
public enum EngineTestHarness {
    /// Which tiny model: plain Qwen3 (`StockTarget`) or the Qwen3.5 hybrid on the engine's fork
    /// (`HybridTarget`).
    public enum Tiny: String, CaseIterable, Sendable {
        case qwen3
        case hybrid
    }

    /// The fake tokenizer's generation prompt with thinking off:
    /// `<|im_start|>assistant\n<think>\n\n</think>\n\n`.
    public static let generationPromptText = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    /// Greedy, no disk prefix, 64 tokens per reply.
    public static func testConfiguration(maxTokens: Int = 64) -> EngineConfiguration {
        var configuration = EngineConfiguration()
        configuration.temperature = 0
        configuration.maxTokens = maxTokens
        return configuration
    }

    /// A tiny float32 model with weights from `seed`; `sharpen` scales the tied embedding by 8
    /// so greedy margins are large (plan §6.4).
    public static func makeModel(_ tiny: Tiny, seed: UInt64 = 0, sharpen: Bool = true) throws -> any LanguageModel {
        let model: any LanguageModel
        switch tiny {
        case .qwen3:
            model = try TinyModels.makeStockQwen3(seed: seed)
        case .hybrid:
            model = try TinyForkModels.makeForkHybrid(seed: seed)
        }
        if sharpen {
            TinyModels.sharpen(model)
        }
        return model
    }

    /// A `LoadedModel` around `model` and `tokenizer`, with a `ModelContainer` for the
    /// `ChatSession` path. The directory needn't exist; its name is the prefix key's revision.
    public static func loadedModel(
        _ model: any LanguageModel, tokenizer: FakeChatMLTokenizer = FakeChatMLTokenizer(), id: String = "tiny",
        modelType: String, directory: URL? = nil, toolCallFormat: ToolCallFormat? = nil
    ) throws -> LoadedModel {
        let directory = directory ?? FileManager.default.temporaryDirectory.appendingPathComponent("local-engine-\(id)")
        let context = ModelContext(
            configuration: ModelConfiguration(directory: directory, toolCallFormat: toolCallFormat), model: model,
            processor: UnusedTestInputProcessor(), tokenizer: tokenizer)
        return try ModelLoader.makeLoadedModel(context: context, id: id, directory: directory, modelType: modelType)
    }

    /// An engine around a tiny model (plan WP20 instruction 10).
    public static func makeEngine(
        tiny: Tiny, tokenizer: FakeChatMLTokenizer = FakeChatMLTokenizer(), seed: UInt64 = 0,
        configuration: EngineConfiguration = testConfiguration()
    ) throws -> InferenceEngine {
        let model = try makeModel(tiny, seed: seed)
        let loaded = try loadedModel(model, tokenizer: tokenizer, id: "tiny-\(tiny.rawValue)-\(seed)", modelType: modelType(tiny))
        return InferenceEngine(loaded: loaded, configuration: configuration)
    }

    /// An engine whose replies follow `scripts` (one per reply, in order; see `ScriptedTarget`).
    public static func makeScriptedEngine(
        tiny: Tiny, scripts: [[Int]], tokenizer: FakeChatMLTokenizer = FakeChatMLTokenizer(), seed: UInt64 = 0,
        configuration: EngineConfiguration = testConfiguration()
    ) throws -> (engine: InferenceEngine, target: ScriptedTarget) {
        // The script decides the replies, so the weights needn't be sharpened.
        let model = try makeModel(tiny, seed: seed, sharpen: false)
        let loaded = try loadedModel(model, tokenizer: tokenizer, id: "tiny-scripted-\(tiny.rawValue)-\(seed)", modelType: modelType(tiny))
        let inner: any TargetModel = EngineModelRegistry.makeTarget(for: loaded) ?? StockTarget(model: model)
        let target = ScriptedTarget(
            inner: inner, generationPrompt: tokenizer.encodeRaw(generationPromptText), scripts: scripts,
            stopToken: FakeChatMLTokenizer.imEnd)
        return (InferenceEngine(loaded: loaded, target: target, configuration: configuration), target)
    }

    public static func modelType(_ tiny: Tiny) -> String {
        switch tiny {
        case .qwen3: return "qwen3"
        case .hybrid: return "qwen3_5_text"
        }
    }

    // MARK: Streams

    /// Every event of `stream`, in order.
    public static func collect(_ stream: AsyncThrowingStream<EngineEvent, Error>) async throws -> [EngineEvent] {
        var events: [EngineEvent] = []
        for try await event in stream {
            events.append(event)
        }
        return events
    }

    /// The visible text of `events`.
    public static func text(_ events: [EngineEvent]) -> String {
        events.compactMap { event -> String? in
            if case .text(let text) = event { return text }
            return nil
        }.joined()
    }

    /// The `.finished` event of `events`.
    public static func finish(_ events: [EngineEvent]) -> EngineFinish? {
        for case .finished(let finish) in events {
            return finish
        }
        return nil
    }

    /// The tool calls of `events`.
    public static func toolCalls(_ events: [EngineEvent]) -> [PendingToolCall] {
        events.flatMap { event -> [PendingToolCall] in
            if case .toolCalls(let calls) = event { return calls }
            return []
        }
    }

    /// A short name per event, for order checks: `responseStarted`, `prefillDone`, `firstToken`,
    /// `toolCallStarted(name)`, `text`, `toolCalls`, `finished(reason)`.
    public static func kinds(_ events: [EngineEvent]) -> [String] {
        events.map { event in
            switch event {
            case .progress(.responseStarted): return "responseStarted"
            case .progress(.prefillDone): return "prefillDone"
            case .progress(.firstToken): return "firstToken"
            case .progress(.toolCallStarted(let name)): return "toolCallStarted(\(name ?? "nil"))"
            case .text: return "text"
            case .toolCalls: return "toolCalls"
            case .finished(let finish): return "finished(\(finish.reason))"
            }
        }
    }
}

/// The engine renders prompts itself, so the tiny `LoadedModel`s never use their processor.
public struct UnusedTestInputProcessor: UserInputProcessor {
    public struct Unused: Error {}

    public init() {}

    public func prepare(input: UserInput) async throws -> LMInput {
        throw Unused()
    }
}

/// A target that runs a real (tiny) model, so its cache behaves exactly like the engine's, but
/// whose replies follow a script.
///
/// It records every token fed at each cache position. A logits row whose context ends with the
/// generation prompt starts a reply; that reply takes the next unused script (remembered by the
/// context, so re-feeding the same context replays the same script), and its rows k = 0, 1, …
/// get a one-hot logit row (100 on the scripted token) for script token k, then for the stop
/// token. Rows outside a scripted reply keep the model's logits. `LiveSession.assertConsistent`
/// compares the raw model, so it checks the real cache.
public final class ScriptedTarget: TargetModel {
    public let inner: any TargetModel
    public let generationPrompt: [Int]
    public private(set) var scripts: [[Int]]
    public let stopToken: Int
    /// The token at each cache position, as fed (-1 where unknown, after `adopt`).
    public private(set) var history: [Int] = []
    private var assignments: [String: Int] = [:]
    private var nextScript = 0

    public init(inner: any TargetModel, generationPrompt: [Int], scripts: [[Int]], stopToken: Int) {
        precondition(!inner.layout.attention.isEmpty, "ScriptedTarget reads positions from an attention layer.")
        self.inner = inner
        self.generationPrompt = generationPrompt
        self.scripts = scripts
        self.stopToken = stopToken
    }

    /// Adds scripts for later replies.
    public func append(scripts more: [[Int]]) {
        scripts += more
    }

    /// How many scripts replies have used.
    public var usedScripts: Int { nextScript }

    public var model: any LanguageModel { inner.model }
    public var cache: [KVCache] { inner.cache }
    public var layout: CacheLayout { inner.layout }
    public var vocabularySize: Int { inner.vocabularySize }
    public var supportsRollback: Bool { inner.supportsRollback }
    public var supportsRowSelection: Bool { inner.supportsRowSelection }

    private var position: Int {
        inner.cache[inner.layout.attention[0]].offset
    }

    public func forward(_ tokens: MLXArray, rows: LogitRows, captureForRollback: Bool, wantHidden: Bool) -> ForwardResult {
        let start = position
        let values = tokens.reshaped([-1]).asType(.int32).asArray(Int32.self).map { Int($0) }
        if history.count > start {
            history.removeLast(history.count - start)
        }
        while history.count < start {
            history.append(-1)
        }
        history += values

        let result = inner.forward(tokens, rows: rows, captureForRollback: captureForRollback, wantHidden: wantHidden)
        guard let logits = result.logits else { return result }

        // The position of the last context token of each logits row.
        let ends: [Int]
        switch rows {
        case .all: ends = Array(start ..< (start + values.count))
        case .last, .none: ends = [start + values.count - 1]
        }
        let vocabulary = logits.dim(-1)
        var scripted = [Float](repeating: 0, count: ends.count * vocabulary)
        var mask = [Bool](repeating: false, count: ends.count)
        for (row, end) in ends.enumerated() {
            guard let token = scriptedToken(after: end), token >= 0, token < vocabulary else { continue }
            mask[row] = true
            scripted[row * vocabulary + token] = 100
        }
        guard mask.contains(true) else { return result }
        let replaced = which(
            MLXArray(mask).reshaped(ends.count, 1), MLXArray(scripted, [ends.count, vocabulary]),
            logits.asType(.float32))
        return ForwardResult(logits: replaced, hidden: result.hidden, capture: result.capture)
    }

    /// The scripted token at position `end + 1`, if `end` is inside a scripted reply.
    private func scriptedToken(after end: Int) -> Int? {
        let length = generationPrompt.count
        guard length > 0, end < history.count else { return nil }
        var promptEnd = end
        while promptEnd >= length - 1 {
            if Array(history[(promptEnd - length + 1) ... promptEnd]) == generationPrompt {
                break
            }
            promptEnd -= 1
        }
        guard promptEnd >= length - 1 else { return nil }
        let key = PrefixKey.tokenHash(history[0 ... promptEnd])
        let index: Int
        if let assigned = assignments[key] {
            index = assigned
        } else {
            guard nextScript < scripts.count else { return nil }
            index = nextScript
            assignments[key] = index
            nextScript += 1
        }
        let step = end - promptEnd
        let script = scripts[index]
        return step < script.count ? script[step] : stopToken
    }

    public func commit(_ capture: (any RoundCapture)?, keep: Int, of verified: Int) {
        inner.commit(capture, keep: keep, of: verified)
    }

    public func resetCache() {
        inner.resetCache()
        history = []
    }

    public func adopt(_ cache: [KVCache]) throws {
        try inner.adopt(cache)
        history = []
    }
}
