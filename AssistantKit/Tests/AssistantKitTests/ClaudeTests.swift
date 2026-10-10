import XCTest
@testable import AssistantKit

final class ClaudeRequestTests: XCTestCase {
    func testOpus5RequestUsesLowEffortCachingSearchAndFallbacks() throws {
        let configuration = ClaudeConfiguration(
            apiKey: "sk-test",
            model: "claude-opus-5",
            effort: "low",
            webSearchEnabled: true,
            timeZoneIdentifier: "Asia/Singapore"
        )
        let turns = [ChatTurn(role: .user, text: "Hi", context: "<context>time</context>")]
        let body = ClaudeRequest.body(configuration: configuration, system: "SYS", messages: ClaudeRequest.messages(from: turns))
        let request = try ClaudeRequest.urlRequest(configuration: configuration, body: body)

        XCTAssertEqual(request.url?.absoluteString, "https://api.anthropic.com/v1/messages")
        XCTAssertEqual(request.httpMethod, "POST")
        XCTAssertEqual(request.value(forHTTPHeaderField: "x-api-key"), "sk-test")
        XCTAssertEqual(request.value(forHTTPHeaderField: "anthropic-version"), "2023-06-01")
        XCTAssertEqual(request.value(forHTTPHeaderField: "anthropic-beta"), "server-side-fallback-2026-07-01")

        let sent = try JSONValue.parse(XCTUnwrap(request.httpBody))
        XCTAssertEqual(sent["model"]?.stringValue, "claude-opus-5")
        XCTAssertEqual(sent["stream"]?.boolValue, true)
        XCTAssertEqual(sent["max_tokens"]?.intValue, 16_000)
        XCTAssertEqual(sent["fallbacks"]?.stringValue, "default")
        XCTAssertEqual(sent["output_config"]?["effort"]?.stringValue, "low")
        XCTAssertNil(sent["thinking"])
        XCTAssertEqual(sent["cache_control"]?["type"]?.stringValue, "ephemeral")
        let system = try XCTUnwrap(sent["system"]?.arrayValue?.first)
        XCTAssertEqual(system["text"]?.stringValue, "SYS")
        XCTAssertEqual(system["cache_control"]?["type"]?.stringValue, "ephemeral")

        let tool = try XCTUnwrap(sent["tools"]?.arrayValue?.first)
        XCTAssertEqual(tool["type"]?.stringValue, "web_search_20260209")
        XCTAssertEqual(tool["name"]?.stringValue, "web_search")
        XCTAssertEqual(tool["user_location"]?["type"]?.stringValue, "approximate")
        XCTAssertEqual(tool["user_location"]?["timezone"]?.stringValue, "Asia/Singapore")

        let message = try XCTUnwrap(sent["messages"]?.arrayValue?.first)
        XCTAssertEqual(message["role"]?.stringValue, "user")
        let texts = message["content"]?.arrayValue?.compactMap { $0["text"]?.stringValue }
        XCTAssertEqual(texts, ["<context>time</context>", "Hi"])
    }

    func testHaikuRequestOmitsEffortAndFallbacks() throws {
        let configuration = ClaudeConfiguration(apiKey: "k", model: "claude-haiku-4-5", timeZoneIdentifier: "UTC")
        let body = ClaudeRequest.body(configuration: configuration, system: "S", messages: [])
        XCTAssertNil(body["output_config"])
        XCTAssertNil(body["fallbacks"])
        let tool = try XCTUnwrap(body["tools"]?.arrayValue?.first)
        XCTAssertEqual(tool["type"]?.stringValue, "web_search_20250305")
        XCTAssertNil(tool["user_location"])
        let request = try ClaudeRequest.urlRequest(configuration: configuration, body: body)
        XCTAssertNil(request.value(forHTTPHeaderField: "anthropic-beta"))
    }

    func testSonnetGetsEffortButNoFallbacks() {
        let configuration = ClaudeConfiguration(apiKey: "k", model: "claude-sonnet-5", webSearchEnabled: false)
        let body = ClaudeRequest.body(configuration: configuration, system: "S", messages: [])
        XCTAssertEqual(body["output_config"]?["effort"]?.stringValue, "low")
        XCTAssertNil(body["fallbacks"])
        XCTAssertNil(body["tools"])
    }

    func testModelCapabilities() {
        XCTAssertTrue(ClaudeModelCatalog.capabilities(for: "claude-opus-5").supportsDefaultFallbacks)
        XCTAssertTrue(ClaudeModelCatalog.capabilities(for: "claude-fable-5-1").supportsDefaultFallbacks)
        XCTAssertTrue(ClaudeModelCatalog.capabilities(for: "claude-opus-5-5").supportsDefaultFallbacks)
        XCTAssertTrue(ClaudeModelCatalog.capabilities(for: "claude-sonnet-5-5").supportsDefaultFallbacks)
        XCTAssertFalse(ClaudeModelCatalog.capabilities(for: "claude-sonnet-5").supportsDefaultFallbacks)
        XCTAssertFalse(ClaudeModelCatalog.capabilities(for: "claude-haiku-4-5").supportsEffort)
        XCTAssertEqual(ClaudeModelCatalog.capabilities(for: "claude-fable-5-1").webSearchToolType, "web_search_20260209")
        XCTAssertEqual(ClaudeModelCatalog.capabilities(for: "claude-sonnet-5-5").webSearchToolType, "web_search_20260209")
        XCTAssertEqual(ClaudeModelCatalog.displayName(for: "claude-opus-5"), "Claude Opus 5")
        XCTAssertEqual(ClaudeModelCatalog.displayName(for: "custom-model"), "custom-model")
    }

    func testOpus55IsTheDefaultAndSonnet55IsListed() {
        XCTAssertEqual(ClaudeModelCatalog.defaultModelID, "claude-opus-5-5")
        XCTAssertEqual(ClaudeModelCatalog.displayName(for: "claude-opus-5-5"), "Claude Opus 5.5")
        XCTAssertEqual(ClaudeModelCatalog.displayName(for: "claude-sonnet-5-5"), "Claude Sonnet 5.5")
        XCTAssertEqual(ClaudeConfiguration(apiKey: "k").model, "claude-opus-5-5")
        XCTAssertEqual(Set(ClaudeModelCatalog.options.map(\.id)).count, ClaudeModelCatalog.options.count)
    }

    func testMidConversationSystemAndPerMessageEffortSupport() {
        let midConversation = ["claude-opus-5", "claude-opus-5-5", "claude-opus-4-8", "claude-fable-5", "claude-fable-5-1", "claude-sonnet-5-5"]
        for model in midConversation {
            XCTAssertTrue(ClaudeModelCatalog.capabilities(for: model).supportsMidConversationSystem, model)
        }
        for model in ["claude-sonnet-5", "claude-haiku-4-5", "custom-model"] {
            XCTAssertFalse(ClaudeModelCatalog.capabilities(for: model).supportsMidConversationSystem, model)
        }
        let perMessage = ["claude-opus-5", "claude-opus-5-5", "claude-fable-5-1", "claude-sonnet-5-5"]
        for model in perMessage {
            XCTAssertTrue(ClaudeModelCatalog.capabilities(for: model).supportsPerMessageEffort, model)
        }
        for model in ["claude-opus-4-8", "claude-fable-5", "claude-sonnet-5", "claude-haiku-4-5"] {
            XCTAssertFalse(ClaudeModelCatalog.capabilities(for: model).supportsPerMessageEffort, model)
        }
    }
}

final class ClaudeStreamDecoderTests: XCTestCase {
    func testStreamsTextAndSearchActivity() throws {
        let events = [
            #"{"type":"message_start","message":{"id":"msg_1","model":"claude-opus-5","content":[]}}"#,
            #"{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}}"#,
            #"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\": \"weather "}}"#,
            #"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"Singapore\"}"}}"#,
            #"{"type":"content_block_stop","index":0}"#,
            #"{"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[]}}"#,
            #"{"type":"content_block_stop","index":1}"#,
            #"{"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}"#,
            #"{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"It's sunny"}}"#,
            #"{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":" today."}}"#,
            #"{"type":"content_block_stop","index":2}"#,
            #"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}"#,
            #"{"type":"message_stop"}"#,
        ]
        var decoder = ClaudeStreamDecoder()
        var output: [AssistantEvent] = []
        for event in events {
            output += try decoder.handle(JSONValue.parse(event))
        }
        XCTAssertEqual(output, [
            .progress(.responseStarted),
            .cue(.lookingUp),
            .reply(.activity("Searching the web")),
            .reply(.activity("Searching the web for “weather Singapore”")),
            .reply(.activity(nil)),
            .reply(.text("It's sunny")),
            .reply(.text(" today.")),
        ])
        XCTAssertEqual(decoder.stopReason, "end_turn")
        XCTAssertEqual(decoder.model, "claude-opus-5")
    }

    func testContinuationBlocksRebuildTheAssistantTurn() throws {
        let events = [
            #"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
            #"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig123"}}"#,
            #"{"type":"content_block_stop","index":0}"#,
            #"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            #"{"type":"content_block_stop","index":1}"#,
            #"{"type":"content_block_start","index":2,"content_block":{"type":"server_tool_use","id":"srvtoolu_9","name":"web_search","input":{}}}"#,
            #"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"news\"}"}}"#,
            #"{"type":"content_block_stop","index":2}"#,
            #"{"type":"message_delta","delta":{"stop_reason":"pause_turn"}}"#,
        ]
        var decoder = ClaudeStreamDecoder()
        var output: [AssistantEvent] = []
        for event in events {
            output += try decoder.handle(JSONValue.parse(event))
        }
        XCTAssertEqual(output, [
            .cue(.lookingUp),
            .reply(.activity("Searching the web")),
            .reply(.activity("Searching the web for “news”")),
        ])
        let blocks = decoder.continuationBlocks()
        XCTAssertEqual(blocks.count, 2, "the empty text block is dropped")
        XCTAssertEqual(blocks[0]["type"]?.stringValue, "thinking")
        XCTAssertEqual(blocks[0]["signature"]?.stringValue, "sig123")
        XCTAssertEqual(blocks[0]["thinking"]?.stringValue, "")
        XCTAssertEqual(blocks[1]["type"]?.stringValue, "server_tool_use")
        XCTAssertEqual(blocks[1]["input"]?["query"]?.stringValue, "news")
        XCTAssertEqual(decoder.stopReason, "pause_turn")
    }

    func testContinuationDropsTheDeclinedModelsReasoningBeforeAFallback() throws {
        let events = [
            #"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":"a"}}"#,
            #"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":"Partial "}}"#,
            #"{"type":"content_block_start","index":2,"content_block":{"type":"fallback","from":{"model":"claude-opus-5"},"to":{"model":"claude-opus-4-8"}}}"#,
            #"{"type":"content_block_start","index":3,"content_block":{"type":"text","text":"answer."}}"#,
        ]
        var decoder = ClaudeStreamDecoder()
        var output: [AssistantEvent] = []
        for event in events {
            output += try decoder.handle(JSONValue.parse(event))
        }
        XCTAssertEqual(output, [.reply(.text("Partial ")), .reply(.text("answer."))])
        XCTAssertEqual(decoder.continuationBlocks().compactMap { $0["type"]?.stringValue }, ["text", "fallback", "text"])
    }

    func testCitationsAreKeptForContinuation() throws {
        let events = [
            #"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"","citations":[]}}"#,
            #"{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","url":"https://a.example","cited_text":"x"}}}"#,
            #"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Cited."}}"#,
        ]
        var decoder = ClaudeStreamDecoder()
        for event in events {
            _ = try decoder.handle(JSONValue.parse(event))
        }
        let block = try XCTUnwrap(decoder.continuationBlocks().first)
        XCTAssertEqual(block["text"]?.stringValue, "Cited.")
        XCTAssertEqual(block["citations"]?.arrayValue?.count, 1)
    }

    func testClientToolUseEmitsProgressCueAndActivity() throws {
        let presentations = ["list_events": ToolPresentation(activity: "Checking your calendar", cue: .checking)]
        var decoder = ClaudeStreamDecoder(clientTools: presentations)
        var output: [AssistantEvent] = []
        let events = MockTransport.toolUse(index: 0, id: "toolu_1", name: "list_events", json: #"{"start":"a","end":"b"}"#)
            + MockTransport.toolUse(index: 1, id: "toolu_2", name: "mystery", json: "{}")
            + ClaudeSSE.text(index: 2, "Done")
        for event in events {
            output += try decoder.handle(JSONValue.parse(event))
        }
        XCTAssertEqual(output, [
            .progress(.toolCallStarted(name: "list_events")),
            .cue(.checking),
            .reply(.activity("Checking your calendar")),
            .progress(.toolCallStarted(name: "mystery")),
            .cue(.working),
            .reply(.activity("Working on it")),
            .reply(.activity(nil)),
            .reply(.text("Done")),
        ])
        XCTAssertEqual(decoder.clientToolCalls(), [
            PendingToolCall(id: "toolu_1", name: "list_events", input: ["start": "a", "end": "b"]),
            PendingToolCall(id: "toolu_2", name: "mystery", input: .object([:])),
        ])
    }

    func testUnparseableClientToolInputIsKeptRawAndSentBackEmpty() throws {
        var decoder = ClaudeStreamDecoder()
        let events = MockTransport.toolUse(index: 0, id: "toolu_1", name: "create_reminder", json: #"{"title": "Call"#)
            + MockTransport.toolUse(index: 1, id: "toolu_2", name: "create_reminder", json: #"["not an object"]"#)
            + Array(MockTransport.toolUse(index: 2, id: "toolu_3", name: "create_reminder", json: #"{"title":"x"}"#).prefix(2))
        for event in events {
            _ = try decoder.handle(JSONValue.parse(event))
        }
        XCTAssertEqual(decoder.clientToolCalls(), [
            PendingToolCall(id: "toolu_1", name: "create_reminder", input: nil, rawInput: #"{"title": "Call"#),
            PendingToolCall(id: "toolu_2", name: "create_reminder", input: nil, rawInput: #"["not an object"]"#),
            PendingToolCall(id: "toolu_3", name: "create_reminder", input: nil, rawInput: #"{"titl"#),
        ])
        let inputs = decoder.continuationBlocks().compactMap { $0["input"] }
        XCTAssertEqual(inputs, [.object([:]), .object([:]), .object([:])])
    }

    func testClientToolCallsBeforeAFallbackAreDropped() throws {
        let events = MockTransport.toolUse(index: 0, id: "toolu_old", name: "list_events", json: "{}") + [
            #"{"type":"content_block_start","index":1,"content_block":{"type":"fallback","from":{"model":"claude-opus-5-5"},"to":{"model":"claude-opus-5"}}}"#,
        ] + MockTransport.toolUse(index: 2, id: "toolu_new", name: "list_events", json: "{}")
        var decoder = ClaudeStreamDecoder()
        for event in events {
            _ = try decoder.handle(JSONValue.parse(event))
        }
        XCTAssertEqual(decoder.clientToolCalls().map(\.id), ["toolu_new"])
        XCTAssertEqual(decoder.continuationBlocks().compactMap { $0["type"]?.stringValue }, ["fallback", "tool_use"])
    }

    func testResponseStartIsReportedOnce() throws {
        var decoder = ClaudeStreamDecoder()
        let start = try JSONValue.parse(ClaudeSSE.messageStart())
        XCTAssertEqual(try decoder.handle(start), [.progress(.responseStarted)])
        XCTAssertEqual(try decoder.handle(start), [])

        var resumed = ClaudeStreamDecoder()
        resumed.responseStartReported = true
        XCTAssertEqual(try resumed.handle(start), [])
        XCTAssertEqual(resumed.model, "claude-opus-5-5")
    }

    func testErrorEventThrows() {
        var decoder = ClaudeStreamDecoder()
        let event = #"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#
        XCTAssertThrowsError(try decoder.handle(JSONValue.parse(event))) { error in
            XCTAssertEqual(error as? AssistantError, .stream(type: "overloaded_error", message: "Overloaded"))
        }
    }
}

final class ClaudeProviderTests: XCTestCase {
    private let hello = [ChatTurn(role: .user, text: "Hi")]

    func testStreamsAReply() async throws {
        let transport = MockTransport([.lines(MockTransport.sse([
            #"{"type":"message_start","message":{"model":"claude-opus-5"}}"#,
            #"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            #"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello!"}}"#,
            #"{"type":"content_block_stop","index":0}"#,
            #"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
            #"{"type":"message_stop"}"#,
        ]))])
        let provider = ClaudeProvider(configuration: ClaudeConfiguration(apiKey: "k"), transport: transport)
        let events = try await collect(provider.streamReply(system: "S", turns: hello))
        XCTAssertEqual(events, [.text("Hello!"), .finished(.completed)])
        XCTAssertEqual(transport.requests.count, 1)
    }

    func testResumesAPausedServerToolTurn() async throws {
        let first = MockTransport.sse([
            #"{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}}"#,
            #"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"q\"}"}}"#,
            #"{"type":"content_block_stop","index":0}"#,
            #"{"type":"message_delta","delta":{"stop_reason":"pause_turn"}}"#,
        ])
        let second = MockTransport.sse([
            #"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"Done."}}"#,
            #"{"type":"content_block_stop","index":0}"#,
            #"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
        ])
        let transport = MockTransport([.lines(first), .lines(second)])
        let provider = ClaudeProvider(configuration: ClaudeConfiguration(apiKey: "k"), transport: transport)
        let events = try await collect(provider.streamReply(system: "S", turns: hello))
        XCTAssertEqual(events, [
            .activity("Searching the web"),
            .activity("Searching the web for “q”"),
            .activity(nil),
            .text("Done."),
            .finished(.completed),
        ])

        XCTAssertEqual(transport.requests.count, 2)
        let resumed = try JSONValue.parse(XCTUnwrap(transport.requests[1].httpBody))
        let messages = try XCTUnwrap(resumed["messages"]?.arrayValue)
        XCTAssertEqual(messages.count, 2)
        XCTAssertEqual(messages[1]["role"]?.stringValue, "assistant")
        XCTAssertEqual(messages[1]["content"]?.arrayValue?.first?["input"]?["query"]?.stringValue, "q")
    }

    func testRetriesWhenOverloaded() async throws {
        let overloaded = HTTPStatusError(
            statusCode: 529,
            body: #"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
            retryAfter: 0
        )
        let transport = MockTransport([
            .failure(overloaded),
            .lines(MockTransport.sse([
                #"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"OK"}}"#,
                #"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
            ])),
        ])
        let provider = ClaudeProvider(configuration: ClaudeConfiguration(apiKey: "k"), transport: transport)
        let events = try await collect(provider.streamReply(system: "S", turns: hello))
        XCTAssertEqual(events, [.text("OK"), .finished(.completed)])
        XCTAssertEqual(transport.requests.count, 2)
    }

    func testMapsAuthenticationErrorsWithoutRetrying() async {
        let transport = MockTransport([.failure(HTTPStatusError(
            statusCode: 401,
            body: #"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#
        ))])
        let provider = ClaudeProvider(configuration: ClaudeConfiguration(apiKey: "bad"), transport: transport)
        do {
            _ = try await collect(provider.streamReply(system: "S", turns: hello))
            XCTFail("Expected an error")
        } catch {
            XCTAssertEqual(
                error as? AssistantError,
                .api(service: "Anthropic", status: 401, type: "authentication_error", message: "invalid x-api-key")
            )
        }
        XCTAssertEqual(transport.requests.count, 1)
    }

    func testRequiresAnAPIKey() async {
        let provider = ClaudeProvider(configuration: ClaudeConfiguration(apiKey: "  "), transport: MockTransport([]))
        do {
            _ = try await collect(provider.streamReply(system: "S", turns: hello))
            XCTFail("Expected an error")
        } catch {
            XCTAssertEqual(error as? AssistantError, .missingAPIKey(service: "Anthropic"))
        }
    }

    func testReportsRefusals() async throws {
        let transport = MockTransport([.lines(MockTransport.sse([
            #"{"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"type":"refusal","category":"cyber"}}}"#,
        ]))])
        let provider = ClaudeProvider(configuration: ClaudeConfiguration(apiKey: "k"), transport: transport)
        let events = try await collect(provider.streamReply(system: "S", turns: hello))
        XCTAssertEqual(events, [.finished(.refused(category: "cyber"))])
    }

    func testStreamEndingEarlyIsAnError() async {
        let transport = MockTransport([.lines(MockTransport.sse([
            #"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"Half"}}"#,
        ]))])
        let provider = ClaudeProvider(configuration: ClaudeConfiguration(apiKey: "k"), transport: transport)
        var received: [ReplyEvent] = []
        do {
            for try await event in provider.streamReply(system: "S", turns: hello) {
                received.append(event)
            }
            XCTFail("Expected an error")
        } catch {
            XCTAssertEqual(received, [.text("Half")])
            XCTAssertNotNil(error as? AssistantError)
        }
    }
}
