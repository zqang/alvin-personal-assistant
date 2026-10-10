import AssistantKit
import Foundation
import MLX
import MLXLMCommon

// The seams speculative decoding (WP30), extra drafters (WP42) and kernels (WP41) plug into.
// The engine builds one `GeneratorContext` per reply (and per tool-round continuation) and asks
// the configured `GeneratorFactory` for the `TokenGenerator` that decodes it.

/// Guesses the tokens the target will produce next.
public protocol Drafter: AnyObject {
    var source: DraftSource { get }
    /// Cost of drafting one token, relative to one single-row target forward.
    var costPerToken: Double { get }
    /// Whether `observe` needs the target's post-final-norm hidden states.
    var wantsHidden: Bool { get }
    /// Called once the reply's prompt is in the cache, with the whole ledger.
    func reset(ledger: [Int], request: EngineRequest)
    /// Up to `maxTokens` guesses for the tokens that follow `context` (the ledger so far).
    func propose(context: ArraySlice<Int>, maxTokens: Int) -> DraftProposal?
    /// What the target did with a round: the tokens it kept and the tokens it emitted.
    func observe(_ round: RoundObservation)
}

public struct RoundObservation {
    /// Tokens the round left in the cache.
    public let committed: [Int]
    /// Tokens the round emitted (visible order; stop tokens excluded).
    public let emitted: [Int]
    /// `[rows, H]` hidden states of the round's forward, when a drafter wants them.
    public let hidden: MLXArray?

    public init(committed: [Int], emitted: [Int], hidden: MLXArray?) {
        self.committed = committed
        self.emitted = emitted
        self.hidden = hidden
    }
}

/// Decodes one reply, step by step, on the engine queue.
public protocol TokenGenerator: AnyObject {
    /// One step or round. Tokens in `emitted` are visible text order; stop tokens excluded;
    /// `finished` set when done.
    func step() throws -> (emitted: [Int], finished: EngineFinish.Reason?)
    /// Feed emitted-but-unfed tokens so the cache holds everything emitted (skip if the GPU is
    /// no longer allowed).
    func flush() throws
    var speculation: SpeculationStats? { get }
}

public protocol GeneratorFactory: Sendable {
    func makeGenerator(_ context: GeneratorContext) -> TokenGenerator
}

/// Collects the sampled tokens' top-1 probabilities for `TokenConfidence` telemetry.
public final class ConfidenceRecorder {
    public private(set) var samples: [Float] = []

    public init() {}

    public func record(_ top1: Float) {
        samples.append(top1)
    }

    public func record<S: Sequence>(contentsOf top1: S) where S.Element == Float {
        samples.append(contentsOf: top1)
    }

    public var summary: TokenConfidence? {
        TokenConfidence(top1: samples)
    }
}

/// Everything a generator needs for one reply. Built by the engine per reply.
public struct GeneratorContext {
    /// The live session; its ledger ends with the generation prompt. Feed through it so the
    /// ledger stays exact (invariant L1).
    public let session: LiveSession
    public let sampler: FastSampler
    /// `[1, V]` logits predicting the first token (position `session.ledger.count`).
    public let firstLogits: MLXArray
    public let stopTokens: Set<Int>
    /// Most tokens to emit.
    public let maxTokens: Int
    public let request: EngineRequest
    /// Drafters from the engine's extensions, already `reset` to the ledger.
    public let drafters: [any Drafter]
    /// Whether the text so far is inside a tool call (for tool-call drafting).
    public let insideToolCall: () -> Bool
    /// False once the stream was terminated or the GPU stopped being allowed: stop with
    /// `.cancelled` at the next step.
    public let isAllowed: () -> Bool
    public let renderer: any ChatTemplateRendering
    public let toolCallFormat: ToolCallFormat
    /// Record each sampled token's top-1 probability here.
    public let confidence: ConfidenceRecorder

    public init(
        session: LiveSession, sampler: FastSampler, firstLogits: MLXArray, stopTokens: Set<Int>, maxTokens: Int,
        request: EngineRequest, drafters: [any Drafter], insideToolCall: @escaping () -> Bool,
        isAllowed: @escaping () -> Bool, renderer: any ChatTemplateRendering, toolCallFormat: ToolCallFormat,
        confidence: ConfidenceRecorder = ConfidenceRecorder()
    ) {
        self.session = session
        self.sampler = sampler
        self.firstLogits = firstLogits
        self.stopTokens = stopTokens
        self.maxTokens = maxTokens
        self.request = request
        self.drafters = drafters
        self.insideToolCall = insideToolCall
        self.isAllowed = isAllowed
        self.renderer = renderer
        self.toolCallFormat = toolCallFormat
        self.confidence = confidence
    }
}

/// The default generator factory: plain pipelined decoding.
public struct PlainGeneratorFactory: GeneratorFactory {
    public init() {}

    public func makeGenerator(_ context: GeneratorContext) -> TokenGenerator {
        DecodeLoop(context)
    }
}
