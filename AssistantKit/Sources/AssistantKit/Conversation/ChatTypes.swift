import Foundation

public enum ChatRole: String, Codable, Sendable {
    case user
    case assistant
}

/// One turn of conversation history as sent to a model.
public struct ChatTurn: Equatable, Sendable {
    public var role: ChatRole
    public var text: String
    /// Per-turn metadata (local time, spoken or typed) sent ahead of a user turn's text.
    public var context: String?

    public init(role: ChatRole, text: String, context: String? = nil) {
        self.role = role
        self.text = text
        self.context = context
    }
}

/// Streamed output of a model reply.
public enum ReplyEvent: Equatable, Sendable {
    /// Visible answer text.
    case text(String)
    /// What the model is doing while it isn't writing, e.g. "Searching the web"; `nil` clears it.
    case activity(String?)
    /// The reply ended. Always the last event of a stream that doesn't throw.
    case finished(ReplyStop)
}

public enum ReplyStop: Equatable, Sendable {
    case completed
    /// Cut off by the output token limit.
    case truncated
    /// Declined by the model or its safety classifiers; any partial text should be discarded.
    case refused(category: String?)
    case other(String)
}

/// A chat model that streams replies.
public protocol ChatProvider: Sendable {
    func streamReply(system: String, turns: [ChatTurn]) -> AsyncThrowingStream<ReplyEvent, Error>
}

public enum AssistantError: Error, Equatable, Sendable, LocalizedError {
    case missingAPIKey(service: String)
    case missingConfiguration(String)
    case api(service: String, status: Int, type: String?, message: String)
    case stream(type: String, message: String)
    case invalidResponse(String)

    public var errorDescription: String? {
        switch self {
        case .missingAPIKey(let service):
            return "Add your \(service) API key in Settings."
        case .missingConfiguration(let detail):
            return detail
        case .api(let service, let status, _, let message):
            switch status {
            case 401: return "\(service) rejected the API key. Check it in Settings."
            case 429: return "\(service) rate limit reached. Try again in a moment."
            case 529: return "\(service) is overloaded right now. Try again shortly."
            default: return "\(service) error \(status): \(message)"
            }
        case .stream(_, let message):
            return "The reply was interrupted: \(message)"
        case .invalidResponse(let detail):
            return "Unexpected response: \(detail)"
        }
    }
}

extension AssistantError {
    /// Maps a failed HTTP response to an error, reading the `{"error": {"type", "message"}}` body
    /// both the Anthropic and OpenAI-style APIs return.
    static func from(_ error: HTTPStatusError, service: String) -> AssistantError {
        let json = try? JSONValue.parse(error.body)
        let type = json?["error"]?["type"]?.stringValue
        let message = json?["error"]?["message"]?.stringValue
            ?? String(error.body.trimmingCharacters(in: .whitespacesAndNewlines).prefix(300))
        return .api(service: service, status: error.statusCode, type: type, message: message)
    }
}
