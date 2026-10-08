import AssistantKit
import Foundation

/// What a reply stream from `InferenceEngine` carries, in order:
///
/// 1. `.progress(.responseStarted)`;
/// 2. `.progress(.prefillDone(...))` once the prompt is in the cache;
/// 3. `.progress(.firstToken)` once, when the first token has been decoded (visible or not);
/// 4. `.text` chunks of visible answer text, with `.progress(.toolCallStarted(...))` when the
///    model begins a tool call (first without, then with the tool's name);
/// 5. `.toolCalls(...)` when the reply ended by asking for tools, then
/// 6. `.finished(...)`, the last event of a stream that doesn't throw.
public enum EngineEvent: Sendable {
    case progress(ReplyProgress)
    case text(String)
    /// The calls to run; answer them with `InferenceEngine.continueReply(after:)`. A call the
    /// model wrote that didn't parse has no `input` and keeps its text in `rawInput`.
    case toolCalls([PendingToolCall])
    case finished(EngineFinish)
}

/// How a reply ended, and what it cost.
public struct EngineFinish: Sendable {
    public enum Reason: Sendable, Equatable {
        /// The model ended its turn (the stop token is in the cache).
        case stop
        /// The reply reached its token limit.
        case length
        /// The stream was terminated or the GPU stopped being allowed. The cache still holds
        /// exactly what was decoded.
        case cancelled
        /// The model asked for tools; `.toolCalls` came just before.
        case toolCalls
    }

    public var reason: Reason
    public var stats: LocalGenerationStats

    public init(reason: Reason, stats: LocalGenerationStats) {
        self.reason = reason
        self.stats = stats
    }
}
