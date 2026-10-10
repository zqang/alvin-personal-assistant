import Foundation

/// Latency marks a provider reports while it works. Consumers may ignore them.
public enum ReplyProgress: Equatable, Sendable {
    /// The service or engine began its response.
    case responseStarted
    /// The on-device engine finished reading the prompt; `reusedTokens` came from its cache.
    case prefillDone(prefilledTokens: Int, reusedTokens: Int)
    /// The model produced its first token, visible or not.
    case firstToken
    /// The model began a tool call. `name` is nil until the tool's name has been read.
    case toolCallStarted(name: String?)
}

/// Everything a reply stream can carry. `ReplyEvent` stays as it is, so existing consumers that
/// switch over it exhaustively keep compiling; the richer events travel only through this type.
public enum AssistantEvent: Equatable, Sendable {
    /// `.text`, `.activity` and `.finished`, exactly as `ChatProvider` streams them.
    case reply(ReplyEvent)
    /// Speech that is not part of the answer, e.g. "Let me check." Never stored.
    case cue(SpokenCue)
    /// A finished round of client tools. Consumers persist it with the reply.
    case toolRound(ToolRound)
    /// The route the orchestrator picked. Its first event, and again after a fallback.
    case routed(RouteDecision)
    /// Latency marks; consumers may ignore them.
    case progress(ReplyProgress)
}

/// A provider that streams `AssistantEvent`s. It is also a `ChatProvider`: the default
/// `streamReply` drops every event that isn't a `.reply`.
public protocol AssistantProvider: ChatProvider {
    func streamEvents(system: String, turns: [ChatTurn]) -> AsyncThrowingStream<AssistantEvent, Error>
}

extension AssistantProvider {
    public func streamReply(system: String, turns: [ChatTurn]) -> AsyncThrowingStream<ReplyEvent, Error> {
        AssistantEvents.replies(streamEvents(system: system, turns: turns))
    }
}

/// Presents a plain `ChatProvider` as an `AssistantProvider` whose events are all `.reply`.
public struct LegacyProviderAdapter: AssistantProvider {
    public let base: any ChatProvider

    public init(_ base: any ChatProvider) {
        self.base = base
    }

    public func streamEvents(system: String, turns: [ChatTurn]) -> AsyncThrowingStream<AssistantEvent, Error> {
        AssistantEvents.wrap(base.streamReply(system: system, turns: turns))
    }

    public func streamReply(system: String, turns: [ChatTurn]) -> AsyncThrowingStream<ReplyEvent, Error> {
        base.streamReply(system: system, turns: turns)
    }
}

/// Converts between `ReplyEvent` and `AssistantEvent` streams.
///
/// Each conversion relays through its own task. Ending or cancelling the returned stream cancels
/// that task, which in turn ends the source stream, so cancellation reaches the producer.
public enum AssistantEvents {
    /// Every reply event, wrapped as `.reply`.
    public static func wrap(_ replies: AsyncThrowingStream<ReplyEvent, Error>) -> AsyncThrowingStream<AssistantEvent, Error> {
        relay(replies) { AssistantEvent.reply($0) }
    }

    /// Only the `.reply` events, unwrapped, in order.
    public static func replies(_ events: AsyncThrowingStream<AssistantEvent, Error>) -> AsyncThrowingStream<ReplyEvent, Error> {
        relay(events) { event -> ReplyEvent? in
            if case .reply(let reply) = event { return reply }
            return nil
        }
    }

    static func relay<Input: Sendable, Output: Sendable>(
        _ source: AsyncThrowingStream<Input, Error>,
        _ transform: @escaping @Sendable (Input) -> Output?
    ) -> AsyncThrowingStream<Output, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await element in source {
                        if let output = transform(element) { continuation.yield(output) }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

/// Thrown by an on-device reply that asks to hand the request to the cloud assistant before it
/// has shown anything. The orchestrator catches it and falls back to the cloud.
public struct ReplyHandoff: Error, Equatable, Sendable, LocalizedError {
    public var reason: String

    public init(reason: String) {
        self.reason = reason
    }

    public var errorDescription: String? {
        "This request needs the online assistant, which isn't available right now."
    }
}
