import AssistantKit
import Foundation

/// Builds the configured model provider and streams replies for a conversation.
@MainActor
enum ReplyService {
    private static let transport = URLSessionStreamingTransport()

    static func stream(for conversation: Conversation, store: SettingsStore) -> AsyncThrowingStream<ReplyEvent, Error> {
        let settings = store.settings
        let turns = PromptBuilder.turns(from: conversation.orderedMessages.map(\.stored))
        let system = PromptBuilder.systemPrompt(userName: settings.userName, customInstructions: settings.customInstructions)
        do {
            return try provider(settings: settings, store: store).streamReply(system: system, turns: turns)
        } catch {
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: error)
            }
        }
    }

    /// What the user still needs to set up before the assistant can answer, if anything.
    static func missingSetup(settings: AssistantSettings, store: SettingsStore) -> String? {
        switch settings.provider {
        case .anthropic:
            return store.secret(.anthropic).isEmpty ? "Add your Anthropic API key in Settings to start." : nil
        case .openAICompatible:
            if store.secret(.compatible).isEmpty { return "Add an API key for your model service in Settings." }
            if settings.compatibleModel.trimmed.isEmpty { return "Enter a model ID in Settings." }
            return nil
        }
    }

    private static func provider(settings: AssistantSettings, store: SettingsStore) throws -> ChatProvider {
        switch settings.provider {
        case .anthropic:
            let configuration = ClaudeConfiguration(
                apiKey: store.secret(.anthropic),
                model: settings.claudeModel.trimmed,
                effort: settings.effort,
                webSearchEnabled: settings.webSearchEnabled,
                timeZoneIdentifier: TimeZone.current.identifier
            )
            return ClaudeProvider(configuration: configuration, transport: transport)
        case .openAICompatible:
            let base = settings.compatibleBaseURL.trimmed
            guard let url = URL(string: base), url.scheme == "https" || url.scheme == "http" else {
                throw AssistantError.missingConfiguration("Enter a valid base URL for your model service in Settings.")
            }
            let configuration = OpenAICompatibleConfiguration(
                serviceName: CompatibleServices.serviceName(forBaseURL: base),
                baseURL: url,
                apiKey: store.secret(.compatible),
                model: settings.compatibleModel.trimmed
            )
            return OpenAICompatibleProvider(configuration: configuration, transport: transport)
        }
    }
}

/// How a finished or stopped reply is recorded.
@MainActor
enum ReplyOutcome {
    static func apply(_ stop: ReplyStop, to reply: ChatMessage) {
        switch stop {
        case .refused:
            // A declined reply's partial text shouldn't be kept or sent back to the model.
            reply.text = ""
            reply.status = .refused
            reply.errorText = "The model declined to answer this. Try rephrasing."
        case .completed, .truncated, .other:
            reply.status = .complete
        }
    }

    /// Keeps a stopped reply's text, or removes the reply if nothing arrived. Returns false if removed.
    @discardableResult
    static func keepStopped(_ reply: ChatMessage, text: String? = nil, in conversation: Conversation) -> Bool {
        let kept = (text ?? reply.text).trimmed
        guard !kept.isEmpty else {
            conversation.remove(reply)
            return false
        }
        reply.text = kept
        reply.status = .interrupted
        return true
    }
}
