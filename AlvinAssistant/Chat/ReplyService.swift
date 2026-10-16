import AssistantKit
import Foundation

/// The configured model services: what is still missing to reach them, and how to build them.
/// `OrchestratedReplyPipeline` builds its engines from these and sends through `transport`.
@MainActor
enum ReplyService {
    /// The transport every cloud request uses, so a connection prewarm (`ConnectionPrewarmer`)
    /// warms the very connection the next reply reads.
    static let transport = URLSessionStreamingTransport()

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
}
