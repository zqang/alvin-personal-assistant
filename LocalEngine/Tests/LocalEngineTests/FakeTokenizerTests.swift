import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLXLMCommon
import XCTest

/// The fake tokenizer stands in for Qwen's in engine tests, so its encoding and template must
/// behave exactly as documented.
final class FakeTokenizerTests: XCTestCase {
    private let tokenizer = FakeChatMLTokenizer()
    private let context = FakeChatMLTokenizer.chatContext

    private let system: [String: any Sendable] = ["role": "system", "content": "Be brief."]
    private let user: [String: any Sendable] = ["role": "user", "content": "Hi"]
    private let reply: [String: any Sendable] = ["role": "assistant", "content": "Hello!"]
    private let followUp: [String: any Sendable] = ["role": "user", "content": "Bye"]

    // MARK: Vocabulary

    func testEncodesASCIINewlinesAndAddedTokens() {
        XCTAssertEqual(tokenizer.encode(text: "Hi\n", addSpecialTokens: true), [72, 105, 10])
        XCTAssertEqual(tokenizer.encodeRaw("<|im_start|>user\nA<|im_end|>"), [1, 117, 115, 101, 114, 10, 65, 2])
        XCTAssertEqual(
            tokenizer.encodeRaw("<think></think><tool_call></tool_call><tool_response></tool_response><|endoftext|><unk>"),
            [4, 5, 6, 7, 8, 9, 3, 0])
        // Characters outside the table are unknown; a lone "<" is just a character.
        XCTAssertEqual(tokenizer.encodeRaw("é\t<x"), [0, 0, 60, 120])
        XCTAssertEqual(tokenizer.encodeRaw("<|im_end"), [60, 124, 105, 109, 95, 101, 110, 100])
        let ids = tokenizer.encodeRaw("新加坡 <|im_start|>~\n")
        XCTAssertTrue(ids.allSatisfy { (0 ..< FakeChatMLTokenizer.vocabularySize).contains($0) })
        XCTAssertEqual(FakeChatMLTokenizer.vocabularySize, TinyModels.vocabularySize)
    }

    func testDecodeRoundTripsAndSkipsOnlySpecialTokens() {
        let text = "<|im_start|>assistant\n<think>\n\n</think>\n\nHello!<|im_end|><|endoftext|>"
        let ids = tokenizer.encodeRaw(text)
        XCTAssertEqual(tokenizer.decodeRaw(ids), text)
        XCTAssertEqual(tokenizer.decode(tokenIds: ids, skipSpecialTokens: true), "assistant\n<think>\n\n</think>\n\nHello!")
        XCTAssertEqual(tokenizer.decode(tokenIds: [0, 72], skipSpecialTokens: false), "<unk>H")
        XCTAssertEqual(tokenizer.decode(tokenIds: [0, 72], skipSpecialTokens: true), "H")
        // Unused ids decode to nothing.
        XCTAssertEqual(tokenizer.decode(tokenIds: [72, 11, 31, 127, 105], skipSpecialTokens: false), "Hi")
    }

    func testTokenLookups() {
        XCTAssertEqual(tokenizer.tokenID("<|im_start|>"), FakeChatMLTokenizer.imStart)
        XCTAssertEqual(tokenizer.tokenID("<|im_end|>"), FakeChatMLTokenizer.imEnd)
        XCTAssertEqual(tokenizer.tokenID("</tool_call>"), FakeChatMLTokenizer.toolCallEnd)
        XCTAssertEqual(tokenizer.tokenID("\n"), FakeChatMLTokenizer.newline)
        XCTAssertEqual(tokenizer.convertTokenToId("A"), 65)
        XCTAssertNil(tokenizer.tokenID("<|missing|>"))
        XCTAssertNil(tokenizer.tokenID("ab"))
        XCTAssertNil(tokenizer.tokenID("é"))
        XCTAssertEqual(tokenizer.convertIdToToken(1), "<|im_start|>")
        XCTAssertEqual(tokenizer.convertIdToToken(126), "~")
        XCTAssertNil(tokenizer.convertIdToToken(20))
        XCTAssertNil(tokenizer.convertIdToToken(128))
        XCTAssertNil(tokenizer.bosToken)
        XCTAssertEqual(tokenizer.eosTokenId, FakeChatMLTokenizer.imEnd)
        XCTAssertEqual(tokenizer.unknownTokenId, FakeChatMLTokenizer.unknown)
        XCTAssertTrue(tokenizer.isChatML)
    }

    func testStopTokensAddChatMLEndsToTheConfiguredOnes() {
        XCTAssertEqual(tokenizer.stopTokenIDs(eosTokenIds: [99], eosToken: tokenizer.eosToken), [99, 2, 3])
        XCTAssertEqual(tokenizer.stopTokenIDs(eosTokenIds: [], eosToken: nil), [2, 3])
    }

    func testStreamingDetokenizerReassemblesText() {
        let text = "Line one.\nLine two, then <think>x</think> done."
        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        var streamed = ""
        for id in tokenizer.encodeRaw(text) {
            detokenizer.append(token: id)
            if let piece = detokenizer.next() {
                streamed += piece
            }
        }
        XCTAssertEqual(streamed, text)
    }

    // MARK: Template

    func testRendersSystemUserAndGenerationPrompt() throws {
        let text = try tokenizer.renderText(messages: [system, user], tools: nil, context: context, addGenerationPrompt: true)
        XCTAssertEqual(text, "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        XCTAssertEqual(try tokenizer.renderTokens(messages: [system, user], tools: nil, context: context, addGenerationPrompt: true), tokenizer.encodeRaw(text))
        // MLXLMCommon's entry point always adds the generation prompt.
        XCTAssertEqual(try tokenizer.applyChatTemplate(messages: [system, user], tools: nil, additionalContext: context), tokenizer.encodeRaw(text))

        let thinking = try tokenizer.renderText(messages: [system, user], tools: nil, context: nil, addGenerationPrompt: true)
        XCTAssertTrue(thinking.hasSuffix("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"))
        let noPrompt = try tokenizer.renderText(messages: [system, user], tools: nil, context: context, addGenerationPrompt: false)
        XCTAssertTrue(noPrompt.hasSuffix("<|im_start|>user\nHi<|im_end|>\n"))
        XCTAssertEqual(try tokenizer.renderText(messages: [system], tools: nil, context: context, addGenerationPrompt: false), "<|im_start|>system\nBe brief.<|im_end|>\n")
    }

    func testToolsGoInsideTheSystemBlock() throws {
        let tools: [[String: any Sendable]] = [[
            "type": "function",
            "function": [
                "name": "get_time",
                "description": "The current time",
                "parameters": ["type": "object", "properties": [String: any Sendable]()] as [String: any Sendable],
            ] as [String: any Sendable],
        ]]
        let text = try tokenizer.renderText(messages: [system, user], tools: tools, context: context, addGenerationPrompt: false)
        XCTAssertTrue(text.hasPrefix("<|im_start|>system\nBe brief.\n\n# Tools\n\n"))
        XCTAssertTrue(text.contains(
            "<tools>\n{\"function\": {\"description\": \"The current time\", \"name\": \"get_time\", \"parameters\": {\"properties\": {}, \"type\": \"object\"}}, \"type\": \"function\"}\n</tools>"),
            text)
        XCTAssertTrue(text.hasSuffix("</tool_call><|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"))
        XCTAssertEqual(text.components(separatedBy: "<|im_start|>").count - 1, 2)

        // Without a system message the tools still get their own system block.
        let bare = try tokenizer.renderText(messages: [user], tools: tools, context: context, addGenerationPrompt: false)
        XCTAssertTrue(bare.hasPrefix("<|im_start|>system\n# Tools\n\n"))
    }

    func testHistoryRepliesDropThinkBlocksAndToolRoundsRenderLikeQwen3() throws {
        let call = ToolCall(
            function: ToolCall.Function(name: "get_time", arguments: ["zone": "Asia/Singapore"] as [String: any Sendable]),
            id: "call_1")
        let messages = DefaultMessageGenerator().generate(messages: [
            .system("S"),
            .user("What time is it?"),
            .assistant("", toolCalls: [call]),
            .tool("{\"time\": \"16:05\"}", id: "call_1"),
            .tool("second", id: "call_2"),
            .assistant("<think>\nhmm\n</think>\n\nIt is 16:05."),
            .user("Thanks"),
        ])
        let text = try tokenizer.renderText(messages: messages, tools: nil, context: context, addGenerationPrompt: true)
        let expected = "<|im_start|>system\nS<|im_end|>\n"
            + "<|im_start|>user\nWhat time is it?<|im_end|>\n"
            + "<|im_start|>assistant\n<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"zone\": \"Asia/Singapore\"}}\n</tool_call><|im_end|>\n"
            + "<|im_start|>user\n<tool_response>\n{\"time\": \"16:05\"}\n</tool_response>\n<tool_response>\nsecond\n</tool_response><|im_end|>\n"
            + "<|im_start|>assistant\nIt is 16:05.<|im_end|>\n"
            + "<|im_start|>user\nThanks<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        XCTAssertEqual(text, expected)

        // Text before the calls is separated from the first call by a newline.
        let spoken = DefaultMessageGenerator().generate(messages: [.user("Q"), .assistant("Checking.", toolCalls: [call, call])])
        let both = try tokenizer.renderText(messages: spoken, tools: nil, context: context, addGenerationPrompt: false)
        XCTAssertTrue(both.contains("<|im_start|>assistant\nChecking.\n<tool_call>\n"), both)
        XCTAssertTrue(both.contains("</tool_call>\n<tool_call>\n"), both)
    }

    /// The properties session reuse relies on (plan §4.5): renders are prefix-stable, the system
    /// prefix is everything before the last `<|im_start|>`, and the sentinel delta equals the
    /// canonical suffix.
    func testRendersArePrefixStable() throws {
        let short = try tokenizer.renderTokens(messages: [system, user], tools: nil, context: context, addGenerationPrompt: false)
        let long = try tokenizer.renderTokens(messages: [system, user, reply, followUp], tools: nil, context: context, addGenerationPrompt: true)
        XCTAssertEqual(Array(long.prefix(short.count)), short)

        let lastStart = try XCTUnwrap(short.lastIndex(of: FakeChatMLTokenizer.imStart))
        let systemOnly = try tokenizer.renderTokens(messages: [system], tools: nil, context: context, addGenerationPrompt: false)
        XCTAssertEqual(Array(short[..<lastStart]), systemOnly)

        // Sentinel render: the text from the first <|im_end|> after the cached reply.
        let sentinelReply: [String: any Sendable] = ["role": "assistant", "content": "SA"]
        let sentinel = try tokenizer.renderText(messages: [system, user, sentinelReply, followUp], tools: nil, context: context, addGenerationPrompt: true)
        let replyStart = try XCTUnwrap(sentinel.range(of: "<|im_start|>assistant\nSA"))
        let end = try XCTUnwrap(sentinel.range(of: "<|im_end|>", range: replyStart.upperBound ..< sentinel.endIndex))
        let delta = tokenizer.encodeRaw(String(sentinel[end.lowerBound...]))
        let canonical = try tokenizer.renderTokens(messages: [system, user, reply, followUp], tools: nil, context: context, addGenerationPrompt: true)
        XCTAssertEqual(Array(canonical.suffix(delta.count)), delta)
    }

    func testTemplateErrors() {
        let bare = FakeChatMLTokenizer(templateAvailable: false)
        XCTAssertThrowsError(try bare.renderTokens(messages: [user], tools: nil, context: nil, addGenerationPrompt: true)) { error in
            XCTAssertEqual(error as? MLXLMCommon.TokenizerError, .missingChatTemplate)
        }
        XCTAssertThrowsError(try bare.applyChatTemplate(messages: [user], tools: nil, additionalContext: nil)) { error in
            XCTAssertEqual(error as? MLXLMCommon.TokenizerError, .missingChatTemplate)
        }
        let robot: [String: any Sendable] = ["role": "robot", "content": "beep"]
        XCTAssertThrowsError(try tokenizer.renderTokens(messages: [robot], tools: nil, context: nil, addGenerationPrompt: true)) { error in
            XCTAssertEqual(error as? FakeChatMLTokenizer.TemplateError, .unknownRole("robot"))
        }
    }

    func testJSONIsCompactSortedAndEscaped() {
        let value: [String: any Sendable] = ["b": [1, 2.5, true] as [any Sendable], "a": "say \"hi\"\n", "c": NSNull()]
        XCTAssertEqual(FakeChatMLTokenizer.json(value), "{\"a\": \"say \\\"hi\\\"\\n\", \"b\": [1, 2.5, true], \"c\": null}")
    }

    // MARK: As a loaded model

    func testLoadedModelWithFakeTokenizerStopsOnChatMLEnds() throws {
        try MetalAvailability.require()
        let model = try TinyModels.makeStockQwen3(seed: 1)
        let configuration = ModelConfiguration(id: "test/tiny-qwen3", eosTokenIds: [FakeChatMLTokenizer.endOfText])
        let context = ModelContext(
            configuration: configuration, model: model, processor: StandInUserInputProcessor(),
            tokenizer: FakeChatMLTokenizer())
        let loaded = try ModelLoader.makeLoadedModel(
            context: context, id: "test/tiny-qwen3", directory: FileManager.default.temporaryDirectory, modelType: "qwen3")
        XCTAssertEqual(loaded.stopTokenIDs, [FakeChatMLTokenizer.imEnd, FakeChatMLTokenizer.endOfText])
        XCTAssertTrue(loaded.renderer is FakeChatMLTokenizer)
        XCTAssertEqual(loaded.modelType, "qwen3")
        XCTAssertEqual(loaded.id, "test/tiny-qwen3")
    }
}
