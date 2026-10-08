import AssistantKit
import Foundation

/// How the engine asks the app whether it may use the GPU (plan §4.11).
///
/// iOS terminates an app that submits GPU work in the background, so the app guards every
/// engine job: `beginGPU` before a job (false refuses it), `endGPU` after it, and `isAllowed`
/// before every decode step and prefill chunk (false stops the job at the next safe point).
public struct EngineHooks: Sendable {
    public var beginGPU: @Sendable () -> Bool
    public var endGPU: @Sendable () -> Void
    public var isAllowed: @Sendable () -> Bool

    public init(
        beginGPU: @escaping @Sendable () -> Bool,
        endGPU: @escaping @Sendable () -> Void,
        isAllowed: @escaping @Sendable () -> Bool
    ) {
        self.beginGPU = beginGPU
        self.endGPU = endGPU
        self.isAllowed = isAllowed
    }

    /// Always allowed: for tests and tools that run in the foreground.
    public static let alwaysAllowed = EngineHooks(beginGPU: { true }, endGPU: {}, isAllowed: { true })
}

/// Everything about how the engine samples, reuses its cache and extends itself. Change it
/// with `InferenceEngine.updateConfiguration(_:)`.
public struct EngineConfiguration: @unchecked Sendable {
    /// Sampling: the app's 0.7 / 0.8 / 20. A temperature of 0 decodes greedily.
    public var temperature: Float = 0.7
    public var topP: Float = 0.8
    public var topK = 20
    /// The default reply length limit, in tokens; `EngineRequest.maxTokens` overrides it.
    public var maxTokens = 1024
    /// Chat-template variables. Part of the prefix key.
    public var chatContext: [String: Bool] = ["enable_thinking": false]
    /// Most tokens one prefill forward feeds.
    public var prefillChunk = 512
    /// When the session planner rebuilds the cache instead of extending it.
    public var limits = SessionPlanner.Limits()
    /// Memory the rewind checkpoints of a hybrid model may hold; over it, `replyStart` is
    /// dropped first, then `lastUserStart` (`systemEnd` is always kept).
    public var checkpointBudgetBytes = CheckpointPolicy.defaultBudgetBytes
    /// Where the system prefix is saved between launches (`<prefixKey>.safetensors`); nil turns
    /// the disk prefix off.
    public var prefixCacheDirectory: URL? = nil
    /// Makes the token generator of each reply; nil uses the plain pipelined `DecodeLoop`.
    public var generatorFactory: (any GeneratorFactory)? = nil
    /// Engine extensions (drafters, kernels). Prepared by `warmUp()`.
    public var extensions: [any EngineExtension] = []
    public var hooks: EngineHooks = .alwaysAllowed
    /// Tests: a seed makes sampling position-keyed, so the same position always draws with the
    /// same key (`MLXRandom.key(seed &+ position)`).
    public var seed: UInt64? = nil

    public init() {}
}

/// What the loaded engine is.
public struct EngineInfo: Sendable {
    public var modelID: String
    /// Whether some layers keep a recurrent state (rewinds use checkpoints).
    public var isHybrid: Bool
    /// Whether the model runs on the engine's `HybridQwen35` fork (row-selective logits and
    /// exact rollback) rather than the stock model code.
    public var forked: Bool
    public var vocabularySize: Int
    /// Bytes one rewind checkpoint holds (0 for pure-attention models, and until the first
    /// forward has shaped the recurrent state).
    public var bytesPerCheckpoint: Int
    public var stopTokenIDs: Set<Int>
    /// The tool-call format's raw value, such as `xml_function` or `json`.
    public var toolCallFormat: String

    /// Changes whenever the cache, ledger or prefix-file layout changes; part of every prefix
    /// key and stored self-test result.
    public static let formatVersion = 1

    public init(
        modelID: String, isHybrid: Bool, forked: Bool, vocabularySize: Int, bytesPerCheckpoint: Int,
        stopTokenIDs: Set<Int>, toolCallFormat: String
    ) {
        self.modelID = modelID
        self.isHybrid = isHybrid
        self.forked = forked
        self.vocabularySize = vocabularySize
        self.bytesPerCheckpoint = bytesPerCheckpoint
        self.stopTokenIDs = stopTokenIDs
        self.toolCallFormat = toolCallFormat
    }
}

public enum EngineError: Error, Equatable, LocalizedError {
    /// The app is leaving the foreground: the GPU may not be used.
    case leftForeground
    /// The tokenizer has no `<|im_start|>` / `<|im_end|>`, so token-exact reuse is impossible.
    /// The engine handles it itself (every reply renders the whole conversation).
    case notChatML
    /// The chat template couldn't render a request.
    case renderFailed(String)
    /// `continueReply(after:)` has nothing to continue: the reply that asked for the tools isn't
    /// the engine's latest work any more.
    case busy

    public var errorDescription: String? {
        switch self {
        case .leftForeground:
            return "The on-device model paused because the app left the foreground."
        case .notChatML:
            return "This model's tokenizer has no ChatML turn markers."
        case .renderFailed(let detail):
            return "The on-device model couldn't read the conversation: \(detail)"
        case .busy:
            return "The on-device model moved on before the tool results arrived."
        }
    }
}

/// One reply to generate: the conversation so far, ending with the user turn to answer.
public struct EngineRequest: Sendable {
    public var system: String
    /// Tools offered to the model, in the order they render.
    public var tools: [ToolDefinition]
    public var turns: [ChatTurn]
    /// Overrides `EngineConfiguration.maxTokens`.
    public var maxTokens: Int? = nil
    /// Decode greedily whatever the configured temperature (self-test, benchmarks).
    public var greedy = false

    public init(system: String, tools: [ToolDefinition] = [], turns: [ChatTurn], maxTokens: Int? = nil, greedy: Bool = false) {
        self.system = system
        self.tools = tools
        self.turns = turns
        self.maxTokens = maxTokens
        self.greedy = greedy
    }
}
