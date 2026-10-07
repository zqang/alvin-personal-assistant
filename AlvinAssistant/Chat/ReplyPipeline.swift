import AssistantKit
import Foundation

/// One reply to produce for a conversation.
@MainActor
struct ReplyRequest {
    let conversation: Conversation
    /// A user turn that isn't stored in `conversation` yet: the transcript an early voice reply
    /// starts from before the turn is final. Only pipelines that support early start read it.
    var pendingUser: StoredMessage? = nil
    /// Side-effect tools wait on this before acting. Nil means they act at once.
    var commitGate: CommitGate? = nil
    var inputIsVoice: Bool
    /// The user asked for deep mode for this reply.
    var deep: Bool = false
}

/// How the chat and voice screens get replies: the seam between the UI and the providers,
/// routing and tools behind it.
@MainActor
protocol ReplyPipeline: AnyObject {
    /// Whether `stream` honours `pendingUser` and `commitGate`, so a voice reply may start early.
    var supportsEarlyStart: Bool { get }
    func stream(_ request: ReplyRequest) -> AsyncThrowingStream<AssistantEvent, Error>
    /// What the user still needs to set up before the assistant can answer, if anything.
    func missingSetup() -> String?
    /// Gets whatever the next reply will use ready (a connection, a model) without starting one.
    func prewarm(inputIsVoice: Bool)
    /// The name to show for the model that answers under `decision`, or the configured one.
    func displayName(for decision: RouteDecision?) -> String
}

@MainActor
enum ReplyPipelines {
    /// Builds the pipeline the chat and voice screens use. Setup at launch may replace it.
    static var make: @MainActor (SettingsStore) -> any ReplyPipeline = { LegacyReplyPipeline(store: $0) }
}

/// Today's behaviour: the provider picked in Settings answers every reply, with no routing,
/// no client tools and no early start. All its events are `.reply`.
@MainActor
final class LegacyReplyPipeline: ReplyPipeline {
    private let store: SettingsStore

    init(store: SettingsStore) {
        self.store = store
    }

    var supportsEarlyStart: Bool { false }

    func stream(_ request: ReplyRequest) -> AsyncThrowingStream<AssistantEvent, Error> {
        AssistantEvents.wrap(ReplyService.stream(for: request.conversation, store: store))
    }

    func missingSetup() -> String? {
        ReplyService.missingSetup(settings: store.settings, store: store)
    }

    func prewarm(inputIsVoice: Bool) {
        if store.settings.provider == .onDevice {
            LocalModelHost.shared.prepare(store.settings)
        }
    }

    func displayName(for decision: RouteDecision?) -> String {
        let settings = store.settings
        switch settings.provider {
        case .anthropic: return ClaudeModelCatalog.displayName(for: settings.claudeModel)
        case .openAICompatible: return settings.compatibleModel
        case .onDevice: return LocalModelCatalog.option(for: settings.localModelID)?.displayName ?? "On-device model"
        }
    }
}
