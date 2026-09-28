import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Streams Claude replies from the Messages API, continuing server-tool turns that pause.
public struct ClaudeProvider: ChatProvider {
    public let configuration: ClaudeConfiguration
    private let transport: HTTPStreamingTransport
    private let maxContinuations = 4

    public init(configuration: ClaudeConfiguration, transport: HTTPStreamingTransport) {
        self.configuration = configuration
        self.transport = transport
    }

    public func streamReply(system: String, turns: [ChatTurn]) -> AsyncThrowingStream<ReplyEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await run(system: system, turns: turns) { continuation.yield($0) }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    func run(system: String, turns: [ChatTurn], emit: (ReplyEvent) -> Void) async throws {
        guard !configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AssistantError.missingAPIKey(service: "Anthropic")
        }
        var messages = ClaudeRequest.messages(from: turns)
        var continuations = 0
        var activityShown = false

        while true {
            let body = ClaudeRequest.body(configuration: configuration, system: system, messages: messages)
            let request = try ClaudeRequest.urlRequest(configuration: configuration, body: body)
            let lines: AsyncThrowingStream<String, Error>
            do {
                lines = try await RetryPolicy.open(request, with: transport)
            } catch let error as HTTPStatusError {
                throw AssistantError.from(error, service: "Anthropic")
            }

            var parser = SSEParser()
            var decoder = ClaudeStreamDecoder()
            decoder.showingActivity = activityShown
            for try await line in lines {
                if let event = parser.consume(line: line) {
                    try Self.decode(event, with: &decoder, emit: emit)
                }
            }
            if let event = parser.finish() {
                try Self.decode(event, with: &decoder, emit: emit)
            }
            try Task.checkCancellation()

            guard let stopReason = decoder.stopReason else {
                throw AssistantError.invalidResponse("The reply ended before it finished.")
            }
            if stopReason == "pause_turn", continuations < maxContinuations {
                // The server-side tool loop hit its iteration limit; sending the turn back resumes it.
                continuations += 1
                activityShown = decoder.showingActivity
                let assistant: [String: JSONValue] = [
                    "role": .string("assistant"),
                    "content": .array(decoder.continuationBlocks()),
                ]
                messages.append(.object(assistant))
                continue
            }
            emit(.finished(Self.stop(for: stopReason, details: decoder.stopDetails)))
            return
        }
    }

    static func stop(for reason: String, details: JSONValue?) -> ReplyStop {
        switch reason {
        case "end_turn", "stop_sequence":
            return .completed
        case "max_tokens", "model_context_window_exceeded":
            return .truncated
        case "refusal":
            return .refused(category: details?["category"]?.stringValue)
        default:
            return .other(reason)
        }
    }

    private static func decode(
        _ event: SSEParser.Event,
        with decoder: inout ClaudeStreamDecoder,
        emit: (ReplyEvent) -> Void
    ) throws {
        guard let json = try? JSONValue.parse(event.data) else { return }
        for output in try decoder.handle(json) {
            emit(output)
        }
    }
}
