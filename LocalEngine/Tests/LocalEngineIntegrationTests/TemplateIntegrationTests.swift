import AssistantKit
import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLXLMCommon
import XCTest

/// `TurnRenderer` with the real tokenizers and chat templates: the system prefix plus the first
/// turns equal the full render, and the continuation and tool-round deltas equal the canonical
/// render's suffix (plan §4.5), with and without tools. The rendered pieces go into the report.
final class TemplateIntegrationTests: XCTestCase {
    private static let context: [String: Bool] = ["enable_thinking": false]
    private static let system = "You are Alvin, a personal voice assistant. Answer briefly."

    /// Tools shaped like the app's device tools (plan §5.3).
    private static let tools = [
        ToolDefinition(
            name: "create_reminder", description: "Adds a reminder to the user's Reminders.",
            inputSchema: [
                "type": "object",
                "properties": [
                    "title": ["type": "string", "description": "What to be reminded of."],
                    "due": ["type": "string", "description": "ISO 8601 local date-time."],
                ],
                "required": ["title"],
            ]),
        ToolDefinition(
            name: "get_current_time", description: "The current local date and time.",
            inputSchema: ["type": "object", "properties": [:], "required": []]),
        ToolDefinition(
            name: "set_timer", description: "Starts a countdown timer.",
            inputSchema: [
                "type": "object",
                "properties": ["seconds": ["type": "integer", "minimum": 1]],
                "required": ["seconds"],
            ]),
    ]

    private static let round = ToolRound(calls: [
        ToolCallRecord(
            id: "call_1", name: "create_reminder", input: ["title": "Call mum", "due": "2026-10-08T17:00"],
            result: "{\"ok\":true}", summary: "Reminder: Call mum"),
    ])

    private static let conversation: [ChatTurn] = [
        ChatTurn(role: .user, text: "Hi there.", context: "[spoken, 09:00]"),
        ChatTurn(role: .assistant, text: "Hello! How can I help?"),
        ChatTurn(role: .user, text: "Remind me to call mum at five."),
        ChatTurn(role: .assistant, text: "Done, I'll remind you at five.", toolRounds: [round]),
        ChatTurn(role: .user, text: "Thanks!"),
    ]

    override func setUpWithError() throws {
        try IntegrationEnvironment.requireEnabled()
        try MetalAvailability.require()
    }

    func testTemplatesOfQwen3_0_6B() async throws {
        try await checkTemplates("mlx-community/Qwen3-0.6B-4bit")
    }

    func testTemplatesOfQwen35_0_8B() async throws {
        try await checkTemplates("mlx-community/Qwen3.5-0.8B-MLX-4bit")
    }

    func testTemplatesOfWoof() async throws {
        guard IntegrationEnvironment.woof else {
            throw XCTSkip("Woof runs only with LOCAL_ENGINE_WOOF=1 ([ci woof]).")
        }
        try await checkTemplates(IntegrationEnvironment.woofRepo)
    }

    private func checkTemplates(_ repo: String) async throws {
        let directory = try await IntegrationEnvironment.snapshot(repo)
        let loaded = try await ModelLoader.load(directory: directory, id: repo, typeRegistry: EngineModelRegistry.makeTypeRegistry())
        let templates = loaded.renderer
        let renderer = try TurnRenderer(renderer: templates, chatContext: Self.context)
        var lines = ["", "### Turn rendering, \(repo)", ""]

        for tools in [[], Self.tools] {
            let label = tools.isEmpty ? "no tools" : "\(tools.count) tools"
            // System prefix + first turns = full render.
            let prefix = try XCTUnwrap(renderer.systemPrefix(system: Self.system, tools: tools), "\(repo) \(label): no system prefix")
            let full = try renderer.fullRender(system: Self.system, tools: tools, turns: Self.conversation)
            let rest = try renderer.firstTurns(system: Self.system, tools: tools, turns: Self.conversation, after: prefix)
            XCTAssertEqual(prefix + rest, full, "\(repo) \(label)")

            // The continuation after the last reply = the canonical suffix.
            let delta = try renderer.continuation(system: Self.system, tools: tools, turns: Array(Self.conversation.suffix(1)))
            XCTAssertEqual(delta.first, renderer.turnEnd, "\(repo) \(label)")
            XCTAssertEqual(Array(full.suffix(delta.count)), delta, "\(repo) \(label): continuation")

            // Two new turns, one of them a reply with a tool round.
            let three = try renderer.continuation(system: Self.system, tools: tools, turns: Array(Self.conversation.suffix(3)))
            XCTAssertEqual(Array(full.suffix(three.count)), three, "\(repo) \(label): continuation with a tool round")

            // The tool-round delta = the suffix of a render that ends with that round.
            let toolTurns = Array(Self.conversation.prefix(3)) + [ChatTurn(role: .assistant, text: "", toolRounds: [Self.round])]
            let toolFull = try renderer.fullRender(system: Self.system, tools: tools, turns: toolTurns)
            let toolDelta = try renderer.toolRoundContinuation(system: Self.system, tools: tools, round: Self.round)
            XCTAssertEqual(Array(toolFull.suffix(toolDelta.count)), toolDelta, "\(repo) \(label): tool round")

            // The newest user turn's start is where the planner's rewinds land.
            let userStart = TurnDelta.lastUserTurnStart(in: full[...], turnStart: renderer.turnStart)
            XCTAssertNotNil(userStart, "\(repo) \(label)")

            lines += [
                "**\(label)**: system prefix \(prefix.count) tokens, full render \(full.count), continuation \(delta.count), tool round \(toolDelta.count)",
                "",
                "System prefix (tail): `\(Self.visible(templates.decodeRaw(Array(prefix.suffix(40)))))`",
                "",
                "Continuation: `\(Self.visible(templates.decodeRaw(delta)))`",
                "",
                "Tool round: `\(Self.visible(templates.decodeRaw(toolDelta)))`",
                "",
            ]
        }
        let generationPrompt = try renderer.fullRender(system: Self.system, tools: [], turns: [ChatTurn(role: .user, text: "Hi")])
        lines.append("Generation prompt ends: `\(Self.visible(templates.decodeRaw(Array(generationPrompt.suffix(8)))))`")
        lines.append("Stop tokens: \(loaded.stopTokenIDs.sorted()); tool-call format: \(loaded.configuration.toolCallFormat?.rawValue ?? "json")")
        EngineReport.append(lines.joined(separator: "\n"))
    }

    private static func visible(_ text: String) -> String {
        text.replacingOccurrences(of: "\n", with: "\\n").replacingOccurrences(of: "`", with: "'")
    }
}
