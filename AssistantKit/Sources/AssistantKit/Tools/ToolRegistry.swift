import Foundation

/// The client tools offered to a model, by name.
///
/// Every tool the registry knows keeps its definition in `definitions`, even when it may not run
/// here (turned off by the user, blocked by `readOnly()`, or a stub from `covering(_:)`). A call to
/// such a tool gets an error result instead. The tool list, and with it the cached prompt prefix,
/// therefore only changes when the set of tools does.
public struct ToolRegistry: Sendable {
    /// The result of calling a tool the user turned off.
    public static let disabledMessage = "The user turned this off in Settings."
    /// The result of calling a tool that may not run in this context.
    public static let unavailableMessage = "Not available here."

    public static let empty = ToolRegistry([])

    private struct Entry: Sendable {
        var tool: any AssistantTool
        /// When set, calls get this error instead of running the tool.
        var blocked: String?
    }

    private var entries: [String: Entry]
    private let disabled: Set<String>

    /// - Parameters:
    ///   - tools: the tools; a later tool replaces an earlier one with the same name.
    ///   - disabled: names of tools the user turned off. They keep their definitions, and calls
    ///     to them give `disabledMessage`.
    public init(_ tools: [any AssistantTool], disabled: Set<String> = []) {
        var entries: [String: Entry] = [:]
        for tool in tools {
            entries[tool.definition.name] = Entry(tool: tool)
        }
        self.entries = entries
        self.disabled = disabled
    }

    private init(entries: [String: Entry], disabled: Set<String>) {
        self.entries = entries
        self.disabled = disabled
    }

    /// Every tool's definition, sorted by name, so equal registries give identical requests.
    public var definitions: [ToolDefinition] {
        entries.keys.sorted().compactMap { entries[$0]?.tool.definition }
    }

    public var isEmpty: Bool {
        entries.isEmpty
    }

    /// The tool for `name` as calls see it: nil if there is none; a tool whose `run` returns the
    /// error result if it is turned off or blocked here.
    public func tool(named name: String) -> (any AssistantTool)? {
        guard let entry = entries[name] else { return nil }
        switch resolve(name) {
        case .runnable(let tool):
            return tool
        case .unavailable(let message):
            return UnavailableTool(base: entry.tool, message: message)
        case .unknown:
            return nil
        }
    }

    /// How the app shows a call to `name`: the tool's own presentation, or `.generic` for a name
    /// the registry doesn't know.
    public func presentation(for name: String) -> ToolPresentation {
        entries[name]?.tool.presentation ?? .generic
    }

    /// This registry with every side-effect tool blocked: each keeps its definition, but calls get
    /// `unavailableMessage` without running. For work that must not change anything, such as a
    /// deep-mode researcher.
    public func readOnly() -> ToolRegistry {
        var blocked = entries
        for (name, entry) in entries where entry.tool.effect == .sideEffect && entry.blocked == nil {
            blocked[name]?.blocked = Self.unavailableMessage
        }
        return ToolRegistry(entries: blocked, disabled: disabled)
    }

    /// Only the tools named in `names`.
    public func subset(_ names: Set<String>) -> ToolRegistry {
        ToolRegistry(entries: entries.filter { names.contains($0.key) }, disabled: disabled)
    }

    /// This registry plus `tools`, which replace any tools with the same names. Names the user
    /// turned off stay off.
    public func adding(_ tools: [any AssistantTool]) -> ToolRegistry {
        var combined = entries
        for tool in tools {
            combined[tool.definition.name] = Entry(tool: tool)
        }
        return ToolRegistry(entries: combined, disabled: disabled)
    }

    /// This registry plus a stub for every tool that `history` called but this registry lacks.
    ///
    /// The model sees those calls in its history; a stub keeps their name defined (APIs reject
    /// history that uses undefined tools) and answers any new call with `unavailableMessage`.
    /// Calls are looked up, and stubbed, under the name Claude sees (`ClaudeRequest.claudeToolName`):
    /// the on-device model may have written a name the API rejects, or a server tool's name.
    public func covering(_ history: [ChatTurn]) -> ToolRegistry {
        var combined = entries
        for turn in history {
            for round in turn.toolRounds {
                for call in round.calls {
                    let name = ClaudeRequest.claudeToolName(call.name)
                    if combined[name] == nil {
                        combined[name] = Entry(tool: StubTool(name: name), blocked: Self.unavailableMessage)
                    }
                }
            }
        }
        return ToolRegistry(entries: combined, disabled: disabled)
    }

    // MARK: - Resolution

    enum Resolution {
        case unknown
        /// Defined, but calls get this error instead of running.
        case unavailable(String)
        case runnable(any AssistantTool)
    }

    /// What a call to `name` does. Turned off by the user wins over blocked.
    func resolve(_ name: String) -> Resolution {
        guard let entry = entries[name] else { return .unknown }
        if disabled.contains(name) { return .unavailable(Self.disabledMessage) }
        if let message = entry.blocked { return .unavailable(message) }
        return .runnable(entry.tool)
    }
}

/// Stands in for a tool that may not run here: same definition and presentation, no side
/// effects, and an error result.
private struct UnavailableTool: AssistantTool {
    let base: any AssistantTool
    let message: String

    var definition: ToolDefinition { base.definition }
    var effect: ToolEffect { .readOnly }
    var presentation: ToolPresentation { base.presentation }

    func run(_ input: [String: JSONValue], context: ToolContext) async throws -> ToolOutput {
        .error(message)
    }
}

/// The definition `covering(_:)` gives a tool that only appears in history.
private struct StubTool: AssistantTool {
    let name: String

    var definition: ToolDefinition {
        ToolDefinition(
            name: name,
            description: "Not available in this conversation any more. Don't call it.",
            inputSchema: [
                "type": "object",
                "properties": .object([:]),
                "required": .array([]),
                "additionalProperties": false,
            ]
        )
    }

    var effect: ToolEffect { .readOnly }
    var presentation: ToolPresentation { ToolPresentation(activity: ToolPresentation.generic.activity, cue: nil) }

    func run(_ input: [String: JSONValue], context: ToolContext) async throws -> ToolOutput {
        .error(ToolRegistry.unavailableMessage)
    }
}
