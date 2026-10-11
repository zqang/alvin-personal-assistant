import Foundation

/// `handoff_to_cloud`: lets the on-device model pass a request it can't answer well to the cloud
/// assistant. It is offered only to the local model, and only when a cloud key exists.
///
/// The local tool loop intercepts the call before it would run (it throws `ReplyHandoff`, or
/// answers it itself when offline), so the tool's own `run` only reports that it was meant to be
/// intercepted. Its input schema matches `handoff_to_cloud` in `scripts/lab_prompts.json`, which
/// the lab measures models against. Its description comes from the app (`AssistantProfile
/// .handoffDescription`), which can name the requests its own tools can't do.
public enum HandoffTool {
    public static let name = "handoff_to_cloud"

    /// A description that names no app's tools, for `AssistantProfile`'s default.
    public static let neutralDescription = "Hand this request to the more capable cloud assistant, which sees the whole conversation. Call this first, instead of answering or calling another tool, when the request needs the internet, asks about something none of your other tools can read or do, or needs long, careful reasoning. Never call another tool as a stand-in for a request it doesn't fit."

    public static func definition(description: String) -> ToolDefinition {
        ToolDefinition(
            name: name,
            description: description,
            inputSchema: [
                "type": "object",
                "properties": [
                    "reason": [
                        "type": "string",
                        "description": "One short sentence on why the cloud assistant is needed.",
                    ],
                ],
                "required": ["reason"],
                "additionalProperties": false,
            ]
        )
    }

    /// The tool with `description` (usually `AssistantProfile.handoffTool`).
    public static func tool(description: String) -> any AssistantTool {
        Handoff(definition: definition(description: description))
    }

    /// The result `run` gives when nothing intercepted the call.
    public static let interceptedMessage = "intercepted"

    private struct Handoff: AssistantTool {
        let definition: ToolDefinition
        // It changes nothing itself, so it never waits for the commit gate.
        var effect: ToolEffect { .readOnly }
        // No cue: whoever performs the handoff announces it with `.handingOff`.
        var presentation: ToolPresentation { ToolPresentation(activity: "Checking online", cue: nil) }

        func run(_ input: [String: JSONValue], context: ToolContext) async throws -> ToolOutput {
            .error(HandoffTool.interceptedMessage)
        }
    }
}
