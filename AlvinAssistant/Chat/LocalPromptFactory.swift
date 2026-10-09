import AssistantKit
import Foundation

/// The system prompt and tools of on-device replies. Replies (`OrchestratedReplyPipeline`) and the
/// prewarm (`LocalModelHost.systemProvider`) both take them from here, so the system prefix the
/// engine prepares ahead of time is exactly the one replies use.
@MainActor
enum LocalPromptFactory {
    /// - `system`: the base system prompt, as Claude gets it. The on-device reply adds
    ///   `PromptBuilder.localSystemPrompt`'s lines itself.
    /// - `tools`: the on-device subset of the device tools (`DeviceTools.localRegistry`), plus
    ///   `handoff_to_cloud` when `handoff` is true. With the device tools turned off, only
    ///   `handoff_to_cloud` (when `handoff` is true) or nothing.
    /// - `handoff`: whether the on-device model may hand a request to Claude (see
    ///   `handoffAvailable(settings:store:)`).
    static func make(settings: AssistantSettings, store: SettingsStore) -> (system: String, tools: ToolRegistry, handoff: Bool) {
        let system = PromptBuilder.systemPrompt(userName: settings.userName, customInstructions: settings.customInstructions)
        let handoff = handoffAvailable(settings: settings, store: store)
        let tools: ToolRegistry
        if settings.deviceToolsEnabled {
            tools = DeviceTools.localRegistry(settings: settings, handoff: handoff)
        } else {
            // The system prompt tells the model to call handoff_to_cloud whenever a handoff is
            // available, so the tool stays defined without the device tools.
            tools = handoff ? ToolRegistry([HandoffTool.tool]) : .empty
        }
        return (system, tools, handoff)
    }

    /// Whether a handoff can reach Claude: an Anthropic key is present and automatic routing is on
    /// (with Claude as the provider), so the orchestrator can pass the request on. With a single
    /// provider there is nothing to hand over to: "On this iPhone" answers everything itself, and
    /// with Claude the on-device model never answers.
    static func handoffAvailable(settings: AssistantSettings, store: SettingsStore) -> Bool {
        !store.secret(.anthropic).isEmpty && settings.provider == .anthropic && settings.routingMode == .automatic
    }
}
