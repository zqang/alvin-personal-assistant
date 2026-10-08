import Foundation

/// `handoff_to_cloud`: lets the on-device model pass a request it can't answer well to the cloud
/// assistant. It is offered only to the local model, and only when a cloud key exists.
///
/// The local tool loop intercepts the call before it would run (it throws `ReplyHandoff`, or
/// answers it itself when offline), so the tool's own `run` only reports that it was meant to be
/// intercepted. Its definition matches `handoff_to_cloud` in `scripts/lab_prompts.json`, which the
/// lab measures models against.
public enum HandoffTool {
    public static let name = "handoff_to_cloud"

    public static let definition = ToolDefinition(
        name: name,
        description: "Hand this request to the more capable cloud assistant, which sees the whole conversation. Call this when the request needs current information from the web, long or careful reasoning, or anything you cannot do well yourself.",
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

    public static let tool: any AssistantTool = Handoff()

    /// The result `run` gives when nothing intercepted the call.
    public static let interceptedMessage = "intercepted"

    private struct Handoff: AssistantTool {
        var definition: ToolDefinition { HandoffTool.definition }
        // It changes nothing itself, so it never waits for the commit gate.
        var effect: ToolEffect { .readOnly }
        // No cue: whoever performs the handoff announces it with `.handingOff`.
        var presentation: ToolPresentation { ToolPresentation(activity: "Checking online", cue: nil) }

        func run(_ input: [String: JSONValue], context: ToolContext) async throws -> ToolOutput {
            .error(HandoffTool.interceptedMessage)
        }
    }
}
