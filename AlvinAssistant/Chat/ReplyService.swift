import AssistantKit
import Foundation

/// Builds the configured model provider and streams replies for a conversation.
///
/// `stream(for:store:)` and `missingSetup(settings:store:)` are today's single-provider behaviour,
/// which `LegacyReplyPipeline` keeps. `OrchestratedReplyPipeline` builds its engines from the same
/// configuration helpers and sends through the same transport.
@MainActor
enum ReplyService {
    /// The transport every cloud request uses, so a connection prewarm (`ConnectionPrewarmer`)
    /// warms the very connection the next reply reads.
    static let transport = URLSessionStreamingTransport()

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
        case .onDevice:
            return LocalModelCatalog.option(for: settings.localModelID) == nil ? "Choose an on-device model in Settings." : nil
        }
    }

    /// The Claude configuration of `settings`, with the stored Anthropic key.
    static func claudeConfiguration(settings: AssistantSettings, store: SettingsStore) -> ClaudeConfiguration {
        ClaudeConfiguration(
            apiKey: store.secret(.anthropic),
            model: settings.claudeModel.trimmed,
            effort: settings.effort,
            webSearchEnabled: settings.webSearchEnabled,
            timeZoneIdentifier: TimeZone.current.identifier
        )
    }

    /// The OpenAI-compatible service of `settings`. Throws when its base URL isn't a web address.
    static func compatibleProvider(settings: AssistantSettings, store: SettingsStore) throws -> OpenAICompatibleProvider {
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

    private static func provider(settings: AssistantSettings, store: SettingsStore) throws -> ChatProvider {
        switch settings.provider {
        case .anthropic:
            return ClaudeProvider(configuration: claudeConfiguration(settings: settings, store: store), transport: transport)
        case .openAICompatible:
            return try compatibleProvider(settings: settings, store: store)
        case .onDevice:
            return LocalProvider(settings: settings)
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

    /// Keeps a stopped reply's text, or removes the reply if nothing arrived. A reply that ran
    /// tool rounds is kept even without text: its actions happened, so the history must show
    /// them. Returns false if removed.
    @discardableResult
    static func keepStopped(_ reply: ChatMessage, text: String? = nil, in conversation: Conversation) -> Bool {
        let kept = (text ?? reply.text).trimmed
        guard !kept.isEmpty || !reply.toolRounds.isEmpty else {
            conversation.remove(reply)
            return false
        }
        reply.text = kept
        reply.status = .interrupted
        return true
    }
}
