import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Streams Claude replies from the Messages API. It continues server-tool turns that pause and
/// runs client tool calls through `tools`, looping until the model answers.
///
/// Within one reply the request only ever grows: each continuation appends the assistant content
/// verbatim (thinking signatures included) and, after client tools, one user message with all
/// their results. `system` and `tools` stay byte-identical, so every request reads the previous
/// one's cache.
public struct ClaudeProvider: AssistantProvider {
    /// Client tool rounds one reply may run; the response after the last one ends the reply.
    static let maxToolRounds = 6

    public let configuration: ClaudeConfiguration
    private let transport: HTTPStreamingTransport
    private let tools: any ToolExecutor
    private let options: ClaudeTurnOptions
    private let toolContext: @Sendable () -> ToolContext
    private let maxContinuations = 4

    public init(
        configuration: ClaudeConfiguration,
        transport: HTTPStreamingTransport,
        tools: any ToolExecutor = NoToolExecutor(),
        options: ClaudeTurnOptions = .init(),
        toolContext: @escaping @Sendable () -> ToolContext = { ToolContext() }
    ) {
        self.configuration = configuration
        self.transport = transport
        self.tools = tools
        self.options = options
        self.toolContext = toolContext
    }

    /// `GET {baseURL}/v1/models?limit=1` with the API key and version headers. Sending it through
    /// the app's transport opens the connection before the user's turn needs it.
    /// Throws `AssistantError.missingAPIKey` when there is no key.
    public static func prewarmRequest(configuration: ClaudeConfiguration) throws -> URLRequest {
        try ClaudeRequest.prewarmRequest(configuration: configuration)
    }

    public func streamEvents(system: String, turns: [ChatTurn]) -> AsyncThrowingStream<AssistantEvent, Error> {
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

    func run(system: String, turns: [ChatTurn], emit: (AssistantEvent) -> Void) async throws {
        guard !configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AssistantError.missingAPIKey(service: "Anthropic")
        }
        let capabilities = ClaudeModelCatalog.capabilities(for: configuration.model)
        let options = ClaudeRequest.resolved(self.options, for: configuration)
        // Read once, so every request of this reply declares the same tools.
        let definitions = tools.definitions
        var presentations: [String: ToolPresentation] = [:]
        for definition in definitions {
            presentations[definition.name] = tools.presentation(for: definition.name)
        }

        var messages = ClaudeRequest.messages(from: turns, options: options, capabilities: capabilities)
        var continuations = 0
        var rounds = 0
        var activityShown = false
        var responseStarted = false

        while true {
            let body = ClaudeRequest.body(
                configuration: configuration,
                system: system,
                messages: messages,
                clientTools: definitions,
                options: options
            )
            let request = try ClaudeRequest.urlRequest(configuration: configuration, body: body)
            let lines: AsyncThrowingStream<String, Error>
            do {
                lines = try await RetryPolicy.open(request, with: transport)
            } catch let error as HTTPStatusError {
                throw AssistantError.from(error, service: "Anthropic")
            }

            var parser = SSEParser()
            var decoder = ClaudeStreamDecoder(clientTools: presentations)
            decoder.showingActivity = activityShown
            decoder.responseStartReported = responseStarted
            for try await line in lines {
                if let event = parser.consume(line: line) {
                    try Self.decode(event, with: &decoder, emit: emit)
                }
            }
            if let event = parser.finish() {
                try Self.decode(event, with: &decoder, emit: emit)
            }
            try Task.checkCancellation()
            activityShown = decoder.showingActivity
            responseStarted = decoder.responseStartReported

            guard let stopReason = decoder.stopReason else {
                throw AssistantError.invalidResponse("The reply ended before it finished.")
            }
            switch stopReason {
            case "pause_turn" where continuations < maxContinuations:
                // The server-side tool loop hit its iteration limit; sending the turn back resumes it.
                continuations += 1
                messages.append(Self.assistantMessage(decoder.continuationBlocks()))
                continue
            case "tool_use":
                let calls = decoder.clientToolCalls()
                guard !calls.isEmpty else {
                    emit(.reply(.finished(.other("tool_use"))))
                    return
                }
                guard rounds < Self.maxToolRounds else {
                    emit(.reply(.finished(.other("tool_limit"))))
                    return
                }
                messages.append(Self.assistantMessage(decoder.continuationBlocks()))
                let round = await runRound(calls)
                rounds += 1
                emit(.toolRound(round))
                try Task.checkCancellation()
                // All results go back in one message, in call order, so the model keeps calling
                // tools in parallel.
                let results = round.calls.map(ClaudeRequest.toolResultBlock)
                messages.append(.object(ClaudeRequest.message(role: "user", content: results)))
                continue
            default:
                // `max_tokens` and `refusal` land here too: a tool call they cut off never runs.
                emit(.reply(.finished(Self.stop(for: stopReason, details: decoder.stopDetails))))
                return
            }
        }
    }

    /// Runs one round. Calls whose input didn't parse get `{"INVALID_JSON": raw}` without running;
    /// the rest go to `tools`. The records follow `calls`' order, one per call.
    private func runRound(_ calls: [PendingToolCall]) async -> ToolRound {
        let runnable = calls.filter { $0.input != nil }
        var executed: [ToolCallRecord] = []
        if !runnable.isEmpty {
            executed = await tools.run(runnable, context: toolContext()).calls
        }
        var records: [ToolCallRecord] = []
        for call in calls {
            guard call.input != nil else {
                let content: JSONValue = ["INVALID_JSON": .string(call.rawInput ?? "")]
                records.append(ToolCallRecord(call: call, output: ToolOutput(content: content, isError: true)))
                continue
            }
            if let index = executed.firstIndex(where: { $0.id == call.id }) {
                records.append(executed.remove(at: index))
            } else {
                // Every tool_use needs a result, or the next request is rejected.
                records.append(ToolCallRecord(call: call, output: .error("The tool returned no result.")))
            }
        }
        return ToolRound(calls: records)
    }

    static func assistantMessage(_ content: [JSONValue]) -> JSONValue {
        .object(ClaudeRequest.message(role: "assistant", content: content))
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
        emit: (AssistantEvent) -> Void
    ) throws {
        guard let json = try? JSONValue.parse(event.data) else { return }
        for output in try decoder.handle(json) {
            emit(output)
        }
    }
}
