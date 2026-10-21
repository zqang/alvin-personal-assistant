import AssistantKit
import Foundation

/// One reply to produce for a conversation.
public struct ReplyRequest: Sendable {
    /// The conversation's messages, oldest first.
    public let messages: [StoredMessage]
    /// A user turn that isn't in `messages` yet: the transcript an early voice reply
    /// starts from before the turn is final. Only pipelines that support early start read it.
    public var pendingUser: StoredMessage?
    /// Side-effect tools wait on this before acting. Nil means they act at once.
    public var commitGate: CommitGate?
    public var inputIsVoice: Bool
    /// The user asked for deep mode for this reply.
    public var deep: Bool

    public init(messages: [StoredMessage], pendingUser: StoredMessage? = nil, commitGate: CommitGate? = nil, inputIsVoice: Bool, deep: Bool = false) {
        self.messages = messages
        self.pendingUser = pendingUser
        self.commitGate = commitGate
        self.inputIsVoice = inputIsVoice
        self.deep = deep
    }
}

/// How the chat and voice screens get replies: the seam between the UI and the providers,
/// routing and tools behind it.
@MainActor
public protocol ReplyPipeline: AnyObject {
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
