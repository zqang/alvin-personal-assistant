import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// A service that speaks the OpenAI Chat Completions protocol.
public struct CompatibleServicePreset: Identifiable, Hashable, Sendable {
    public var id: String { name }
    public let name: String
    public let baseURL: String
    public let modelHint: String
}

public enum CompatibleServices {
    public static let presets: [CompatibleServicePreset] = [
        CompatibleServicePreset(
            name: "Doubao (Volcengine Ark)",
            baseURL: "https://ark.cn-beijing.volces.com/api/v3",
            modelHint: "Model or endpoint ID from the Ark console"
        ),
        CompatibleServicePreset(name: "DeepSeek", baseURL: "https://api.deepseek.com/v1", modelHint: "e.g. deepseek-chat"),
        CompatibleServicePreset(name: "OpenAI", baseURL: "https://api.openai.com/v1", modelHint: "e.g. gpt-5-mini"),
    ]

    /// A readable name for error messages: the matching preset, else the host.
    public static func serviceName(forBaseURL baseURL: String) -> String {
        if let preset = presets.first(where: { $0.baseURL == baseURL }) { return preset.name }
        return URL(string: baseURL)?.host ?? "The model service"
    }
}

public struct OpenAICompatibleConfiguration: Equatable, Sendable {
    public var serviceName: String
    public var baseURL: URL
    public var apiKey: String
    public var model: String
    /// Older turns are dropped past this; these models have smaller context windows than Claude.
    public var maxHistoryTurns: Int

    public init(serviceName: String, baseURL: URL, apiKey: String, model: String, maxHistoryTurns: Int = 60) {
        self.serviceName = serviceName
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.model = model
        self.maxHistoryTurns = maxHistoryTurns
    }
}

/// Streams replies from any OpenAI-compatible `/chat/completions` endpoint.
public struct OpenAICompatibleProvider: ChatProvider {
    public let configuration: OpenAICompatibleConfiguration
    private let transport: HTTPStreamingTransport

    public init(configuration: OpenAICompatibleConfiguration, transport: HTTPStreamingTransport) {
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
        let service = configuration.serviceName
        guard !configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AssistantError.missingAPIKey(service: service)
        }
        guard !configuration.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AssistantError.missingConfiguration("Enter a model ID for \(service) in Settings.")
        }
        let request = try Self.urlRequest(configuration: configuration, system: system, turns: turns)
        let lines: AsyncThrowingStream<String, Error>
        do {
            lines = try await RetryPolicy.open(request, with: transport)
        } catch let error as HTTPStatusError {
            throw AssistantError.from(error, service: service)
        }

        var parser = SSEParser()
        var finishReason: String?
        reading: for try await line in lines {
            guard let event = parser.consume(line: line) else { continue }
            let payload = event.data.trimmingCharacters(in: .whitespaces)
            if payload == "[DONE]" { break reading }
            guard let json = try? JSONValue.parse(payload) else { continue }
            if let error = json["error"], !error.isNull {
                throw AssistantError.stream(
                    type: error["type"]?.stringValue ?? "error",
                    message: error["message"]?.stringValue ?? "Unknown error"
                )
            }
            guard let choice = json["choices"]?.arrayValue?.first else { continue }
            if let text = choice["delta"]?["content"]?.stringValue, !text.isEmpty {
                emit(.text(text))
            }
            if let reason = choice["finish_reason"]?.stringValue {
                finishReason = reason
            }
        }
        try Task.checkCancellation()

        switch finishReason {
        case "length":
            emit(.finished(.truncated))
        case "content_filter":
            emit(.finished(.refused(category: "content_filter")))
        default:
            emit(.finished(.completed))
        }
    }

    static func body(model: String, system: String, turns: [ChatTurn], maxHistoryTurns: Int) -> JSONValue {
        var recent = Array(turns.suffix(max(maxHistoryTurns, 1)))
        while recent.first?.role == .assistant { recent.removeFirst() }

        // These services get no tools, so an assistant turn's tool rounds are shown by their
        // summaries when the turn has no text of its own.
        var entries: [(role: ChatRole, content: String)] = []
        for turn in recent {
            var content = turn.text
            switch turn.role {
            case .user:
                if let context = turn.context, !context.isEmpty {
                    content = context + "\n\n" + content
                }
            case .assistant:
                if content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    content = turn.toolRounds.compactMap(\.summaryLine).joined(separator: " ")
                }
                if content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { continue }
            }
            if turn.role == .user, let last = entries.last, last.role == .user {
                // A skipped assistant turn leaves two user turns in a row; some services reject
                // that, so they become one, the way PromptBuilder folds user messages.
                entries[entries.count - 1].content = last.content + "\n\n" + content
            } else {
                entries.append((role: turn.role, content: content))
            }
        }

        var messages: [JSONValue] = [message(role: "system", content: system)]
        for entry in entries {
            messages.append(message(role: entry.role.rawValue, content: entry.content))
        }
        let body: [String: JSONValue] = [
            "model": .string(model),
            "stream": .bool(true),
            "messages": .array(messages),
        ]
        return .object(body)
    }

    static func urlRequest(configuration: OpenAICompatibleConfiguration, system: String, turns: [ChatTurn]) throws -> URLRequest {
        var request = URLRequest(url: configuration.baseURL.appendingPathComponent("chat/completions"))
        request.httpMethod = "POST"
        request.timeoutInterval = 120
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.setValue("Bearer \(configuration.apiKey)", forHTTPHeaderField: "authorization")
        let payload = Self.body(
            model: configuration.model,
            system: system,
            turns: turns,
            maxHistoryTurns: configuration.maxHistoryTurns
        )
        request.httpBody = try payload.serialized()
        return request
    }

    private static func message(role: String, content: String) -> JSONValue {
        let object: [String: JSONValue] = ["role": .string(role), "content": .string(content)]
        return .object(object)
    }
}
