import AssistantKit
import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLXLMCommon
import XCTest

/// `TurnRenderer` with the fake ChatML tokenizer: every piece equals that segment of a full
/// render (plan §4.5). No GPU needed.
final class TurnRendererTests: XCTestCase {
    private let tokenizer = FakeChatMLTokenizer()
    private let system = "You are Alvin. Answer briefly."

    private var renderer: TurnRenderer {
        try! TurnRenderer(renderer: tokenizer, chatContext: ["enable_thinking": false])
    }

    private let tools = [
        ToolDefinition(
            name: "get_time", description: "The current local time.",
            inputSchema: ["type": "object", "properties": [:], "required": []]),
        ToolDefinition(
            name: "set_timer", description: "Starts a timer.",
            inputSchema: ["type": "object", "properties": ["seconds": ["type": "integer"]], "required": ["seconds"]]),
    ]

    private var conversation: [ChatTurn] {
        [
            ChatTurn(role: .user, text: "Hi there.", context: "[time 09:00]"),
            ChatTurn(role: .assistant, text: "Hello! How can I help?"),
            ChatTurn(role: .user, text: "What's 2 + 2?"),
            ChatTurn(role: .assistant, text: "4."),
            ChatTurn(role: .user, text: "Thanks."),
        ]
    }

    private var round: ToolRound {
        ToolRound(calls: [
            ToolCallRecord(id: "call_1", name: "set_timer", input: ["seconds": 600], result: "{\"ok\":true}", summary: "Timer 10 min"),
        ])
    }

    func testSystemPrefixPlusFirstTurnsEqualsTheFullRender() throws {
        for tools in [[], self.tools] {
            let prefix = try XCTUnwrap(renderer.systemPrefix(system: system, tools: tools))
            XCTAssertEqual(prefix.first, FakeChatMLTokenizer.imStart)
            XCTAssertEqual(Array(prefix.suffix(2)), [FakeChatMLTokenizer.imEnd, FakeChatMLTokenizer.newline])
            let full = try renderer.fullRender(system: system, tools: tools, turns: conversation)
            let rest = try renderer.firstTurns(system: system, tools: tools, turns: conversation, after: prefix)
            XCTAssertEqual(prefix + rest, full)
            // With nothing in the ledger, the first turns are the whole render.
            XCTAssertEqual(try renderer.firstTurns(system: system, tools: tools, turns: conversation, after: []), full)
            XCTAssertTrue(tokenizer.decodeRaw(full).hasSuffix(EngineTestHarness.generationPromptText))
        }
    }

    func testFirstTurnsRefuseAForeignPrefix() throws {
        let other = try XCTUnwrap(renderer.systemPrefix(system: "Another system prompt.", tools: []))
        XCTAssertThrowsError(try renderer.firstTurns(system: system, tools: [], turns: conversation, after: other)) { error in
            guard case EngineError.renderFailed = error else { return XCTFail("unexpected \(error)") }
        }
    }

    /// The continuation after a cached reply equals the canonical render's suffix from the
    /// reply's turn end on.
    func testContinuationEqualsTheCanonicalSuffix() throws {
        for tools in [[], self.tools] {
            let turns = conversation
            let full = try renderer.fullRender(system: system, tools: tools, turns: turns)
            let delta = try renderer.continuation(system: system, tools: tools, turns: Array(turns.suffix(1)))
            XCTAssertEqual(delta.first, FakeChatMLTokenizer.imEnd)
            XCTAssertEqual(Array(full.suffix(delta.count)), delta)
            XCTAssertTrue(tokenizer.decodeRaw(delta).hasPrefix("<|im_end|>\n<|im_start|>user\nThanks.<|im_end|>\n"))

            // Two new turns (an intervening reply from elsewhere) also give the canonical suffix.
            let two = try renderer.continuation(system: system, tools: tools, turns: Array(turns.suffix(3)))
            XCTAssertEqual(Array(full.suffix(two.count)), two)
            XCTAssertTrue(tokenizer.decodeRaw(two).hasPrefix("<|im_end|>\n<|im_start|>user\nWhat's 2 + 2?"))

            // As generated, the cache holds the reply then its stop token; the delta overlaps that
            // stop token by one.
            XCTAssertEqual(TurnDelta.overlap(ledgerTail: [52, FakeChatMLTokenizer.imEnd][...], delta: delta), 1)
        }
    }

    /// The tool-round continuation: turn end, the results as a user turn, the generation prompt.
    func testToolRoundContinuation() throws {
        let delta = try renderer.toolRoundContinuation(system: system, tools: tools, round: round)
        let text = tokenizer.decodeRaw(delta)
        XCTAssertEqual(
            text,
            "<|im_end|>\n<|im_start|>user\n<tool_response>\n{\"ok\":true}\n</tool_response><|im_end|>\n"
                + EngineTestHarness.generationPromptText)

        // It is the suffix of a full render of the turn with that round.
        let turns = [
            ChatTurn(role: .user, text: "Set a timer for ten minutes."),
            ChatTurn(role: .assistant, text: "", toolRounds: [round]),
        ]
        let messages = TurnRenderer.raw(system: system, TurnRenderer.messages(turns))
        let full = try tokenizer.renderTokens(
            messages: messages, tools: JSONBridge.templateTools(tools), context: ["enable_thinking": false], addGenerationPrompt: true)
        XCTAssertEqual(Array(full.suffix(delta.count)), delta)
    }

    /// A replaced user turn after a rewind to its `<|im_start|>`: the ledger ends with
    /// `<|im_end|>\n`, which the delta's first two tokens overlap.
    func testOverlapAfterRewindingToAUserTurn() throws {
        let delta = try renderer.continuation(system: system, tools: [], turns: [ChatTurn(role: .user, text: "Edited.")])
        let tail = [70, FakeChatMLTokenizer.imEnd, FakeChatMLTokenizer.newline]
        XCTAssertEqual(TurnDelta.overlap(ledgerTail: tail[...], delta: delta), 2)
        XCTAssertEqual(delta[2], FakeChatMLTokenizer.imStart)
        // A cancelled reply ends with content: nothing overlaps.
        XCTAssertEqual(TurnDelta.overlap(ledgerTail: [70, 71][...], delta: delta), 0)
    }

    func testAssistantTurnsWithRoundsRenderCallsThenResultsThenText() throws {
        let turn = ChatTurn(role: .assistant, text: "Done, timer set.", toolRounds: [round, round])
        let messages = TurnRenderer.raw(system: system, TurnRenderer.messages([ChatTurn(role: .user, text: "Hi"), turn]))
        let roles = messages.map { $0["role"] as? String ?? "?" }
        XCTAssertEqual(roles, ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"])
        XCTAssertNotNil(messages[2]["tool_calls"])
        XCTAssertEqual(messages[3]["tool_call_id"] as? String, "call_1")
        XCTAssertEqual(messages[6]["content"] as? String, "Done, timer set.")

        // A user turn's context goes ahead of its text.
        let user = TurnRenderer.messages([ChatTurn(role: .user, text: "Hi", context: "[voice]")])
        XCTAssertEqual(user.first?.content, "[voice]\n\nHi")
    }

    func testAssistantTextHasNoTurnEnd() {
        let tokens = renderer.assistantText("Sure, here")
        XCTAssertEqual(tokenizer.decodeRaw(tokens), "Sure, here")
        XCTAssertFalse(tokens.contains(FakeChatMLTokenizer.imEnd))
    }

    func testTokenizersWithoutChatMLAreRefused() {
        XCTAssertThrowsError(try TurnRenderer(renderer: NoChatML(base: tokenizer), chatContext: [:])) { error in
            XCTAssertEqual(error as? EngineError, .notChatML)
        }
    }

    func testTemplateToolsUseTheFunctionFormat() throws {
        let rendered = JSONBridge.templateTools(tools)
        XCTAssertEqual(rendered.count, 2)
        XCTAssertEqual(rendered[0]["type"] as? String, "function")
        let function = try XCTUnwrap(rendered[1]["function"] as? [String: any Sendable])
        XCTAssertEqual(function["name"] as? String, "set_timer")
        let parameters = try XCTUnwrap(function["parameters"] as? [String: any Sendable])
        XCTAssertEqual((parameters["required"] as? [any Sendable])?.first as? String, "seconds")
        XCTAssertNil(JSONBridge.templateToolsOrNil([]))
    }

    func testToolCallArgumentsRoundTrip() {
        let input: AssistantKit.JSONValue = ["title": "Call mum", "minutes": 30, "flags": [true, .null], "note": .null]
        let arguments = JSONBridge.arguments(input)
        XCTAssertEqual(arguments["title"], .string("Call mum"))
        XCTAssertEqual(arguments["minutes"], .int(30))
        XCTAssertNil(arguments["note"], "null members are dropped for templates")
        XCTAssertEqual(arguments["flags"], .array([.bool(true), .string("null")]))
        XCTAssertEqual(JSONBridge.input(["seconds": .int(600)]), ["seconds": 600])
        XCTAssertTrue(JSONBridge.arguments(.string("not an object")).isEmpty)
    }
}

/// The fake tokenizer without ChatML markers.
private struct NoChatML: ChatTemplateRendering {
    let base: FakeChatMLTokenizer

    func renderTokens(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                      context: [String: any Sendable]?, addGenerationPrompt: Bool) throws -> [Int] {
        try base.renderTokens(messages: messages, tools: tools, context: context, addGenerationPrompt: addGenerationPrompt)
    }

    func encodeRaw(_ text: String) -> [Int] { base.encodeRaw(text) }
    func decodeRaw(_ tokens: [Int]) -> String { base.decodeRaw(tokens) }
    func tokenID(_ token: String) -> Int? {
        token == "<|im_start|>" || token == "<|im_end|>" ? nil : base.tokenID(token)
    }
}
