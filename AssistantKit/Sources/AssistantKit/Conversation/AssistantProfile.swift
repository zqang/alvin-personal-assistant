import Foundation

/// What an app tells AssistantKit and LocalEngine about its own domain: who the assistant is,
/// which words ask for one of the app's actions, what a reply claiming such an action sounds like,
/// and which tool calls to stop. Every default is domain-neutral, so a new app starts from a
/// working assistant and adds its own words; the packages themselves know nothing about any app.
///
/// An app builds one value in its own code, usually `extension AssistantProfile { static let app
/// = … }`, and passes that same value wherever a profile is asked for:
/// `PromptBuilder.systemPrompt` and `localSystemPrompt`, `RouteDecider(classifier:
/// profile.intents)`, `profile.handoffTool`, and LocalEngine's `LocalToolLoop`. Claude's prompt
/// cache and the on-device prefix prewarmed ahead of a reply only hit when every prompt comes from
/// the same profile.
///
/// Facts about particular tools are kept here by tool name (`contextLookups`, `callGuard`) rather
/// than on each `AssistantTool`: the on-device loop only sees a `ToolExecutor`, the router reads
/// the user's words even when the app's tools are switched off, and the benchmark runs stand-ins.
public struct AssistantProfile: Sendable {
    /// The system prompt's first sentence, given the user's name (trimmed; "" when there is none):
    /// who the assistant is. The shared rules for spoken and typed replies follow it.
    public var persona: @Sendable (_ userName: String) -> String
    /// The on-device model's line about what its tools are for.
    public var localToolHint: String
    /// Added to `localToolHint`, after a space, when the on-device model may call
    /// `handoff_to_cloud`.
    public var localHandoffHint: String
    /// `handoff_to_cloud`'s description: when the on-device model should pass a request on. It
    /// works best when it names requests that look like the app's tools but that they can't do.
    public var handoffDescription: String
    /// Routing hints. Its action phrases also switch on the on-device claim guard.
    public var intents: IntentClassifier
    /// What a reply that says an action was done sounds like.
    public var claims: SuccessClaim
    /// Tools that only gather context for an action, such as reading the time to place "next
    /// Monday": the claim guard stays on after them.
    public var contextLookups: Set<String>
    /// Checks each on-device tool call against the user's words before it runs.
    public var callGuard: ToolCallGuard

    public init(
        persona: @escaping @Sendable (_ userName: String) -> String = { name in
            "You are a warm, capable assistant\(name.isEmpty ? "" : " for \(name)")."
        },
        localToolHint: String = "You run on the user's device. Use your tools when a request needs them.",
        localHandoffHint: String = "If a request needs the internet or deep expertise, call \(HandoffTool.name) before saying anything.",
        handoffDescription: String = HandoffTool.neutralDescription,
        intents: IntentClassifier = IntentClassifier(),
        claims: SuccessClaim = SuccessClaim(),
        contextLookups: Set<String> = [],
        callGuard: ToolCallGuard = .none
    ) {
        self.persona = persona
        self.localToolHint = localToolHint
        self.localHandoffHint = localHandoffHint
        self.handoffDescription = handoffDescription
        self.intents = intents
        self.claims = claims
        self.contextLookups = contextLookups
        self.callGuard = callGuard
    }

    /// `handoff_to_cloud` with this profile's description.
    public var handoffTool: any AssistantTool {
        HandoffTool.tool(description: handoffDescription)
    }
}

/// Checks an on-device tool call against the user's words before it runs: a small model sometimes
/// reaches for the wrong tool. A call the guard objects to never runs (see LocalEngine's
/// `LocalToolLoop`).
public struct ToolCallGuard: Sendable {
    /// Why `call` doesn't fit `userText` (`userText(of:)` of the request), or nil when it may run.
    public let check: @Sendable (_ call: PendingToolCall, _ userText: String) -> String?

    public init(_ check: @escaping @Sendable (_ call: PendingToolCall, _ userText: String) -> String?) {
        self.check = check
    }

    /// Lets every call run.
    public static let none = ToolCallGuard { _, _ in nil }

    /// The user's words a call should fit: the newest user turn's text and, when the reply before
    /// it asked the user something ("What should I call it?"), the user turn that reply answered,
    /// so an amount given there still counts.
    public static func userText(of turns: [ChatTurn]) -> String {
        guard let last = turns.lastIndex(where: { $0.role == .user }) else { return "" }
        var texts = [turns[last].text]
        if last >= 2, turns[last - 1].role == .assistant, turns[last - 2].role == .user,
           turns[last - 1].text.contains(where: { $0 == "?" || $0 == "？" }) {
            texts.insert(turns[last - 2].text, at: 0)
        }
        return texts.joined(separator: "\n")
    }
}
