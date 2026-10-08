import Foundation

/// Per-reply adjustments to a Claude request, on top of `ClaudeConfiguration`.
///
/// The instruction, data and per-message effort are rendered around the last user turn, so the
/// cached prefix (tools, system prompt, earlier turns) stays the same as for a plain reply. Only a
/// top-level effort change (on models without per-message effort) invalidates the cached turns.
public struct ClaudeTurnOptions: Equatable, Sendable {
    /// An operator brief for this reply. Sent as a system message after the last user turn on models
    /// that accept one, otherwise as an `<instructions>` block at the end of the last user turn.
    public var instruction: String?
    /// Data for this reply (e.g. analyst notes), appended to the last user turn as a text block
    /// before any instruction block.
    public var userData: String?
    /// Effort for this reply when it differs from the configuration's. Sent as a per-message effort
    /// change where the model accepts one (which keeps the conversation cache), otherwise as the
    /// top-level effort.
    public var effort: String?
    /// Output token cap for this reply; nil keeps the configuration's.
    public var maxTokens: Int?
    /// Streams client tool input as it is generated (`eager_input_streaming`). Applies only when the
    /// request goes straight to api.anthropic.com; the input is validated client-side either way.
    public var eagerToolInput = true

    public init(
        instruction: String? = nil,
        userData: String? = nil,
        effort: String? = nil,
        maxTokens: Int? = nil,
        eagerToolInput: Bool = true
    ) {
        self.instruction = instruction
        self.userData = userData
        self.effort = effort
        self.maxTokens = maxTokens
        self.eagerToolInput = eagerToolInput
    }
}
