import XCTest
@testable import AssistantKit

final class OpenAICompatibleTests: XCTestCase {
    private func configuration(model: String = "doubao-test") -> OpenAICompatibleConfiguration {
        OpenAICompatibleConfiguration(
            serviceName: "Doubao",
            baseURL: URL(string: "https://ark.cn-beijing.volces.com/api/v3")!,
            apiKey: "k",
            model: model
        )
    }

    func testStreamsChatCompletionDeltas() async throws {
        let transport = MockTransport([.lines([
            #"data: {"choices":[{"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#, "",
            #"data: {"choices":[{"delta":{"content":"你好"},"finish_reason":null}]}"#, "",
            #"data: {"choices":[{"delta":{"reasoning_content":"thinking"},"finish_reason":null}]}"#, "",
            #"data: {"choices":[{"delta":{"content":"！"},"finish_reason":"stop"}]}"#, "",
            "data: [DONE]", "",
        ])])
        let provider = OpenAICompatibleProvider(configuration: configuration(), transport: transport)
        let turns = [ChatTurn(role: .user, text: "Hi", context: "<context>c</context>")]
        let events = try await collect(provider.streamReply(system: "SYS", turns: turns))
        XCTAssertEqual(events, [.text("你好"), .text("！"), .finished(.completed)])

        let request = try XCTUnwrap(transport.requests.first)
        XCTAssertEqual(request.url?.absoluteString, "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
        XCTAssertEqual(request.value(forHTTPHeaderField: "authorization"), "Bearer k")
        let body = try JSONValue.parse(XCTUnwrap(request.httpBody))
        XCTAssertEqual(body["model"]?.stringValue, "doubao-test")
        XCTAssertEqual(body["stream"]?.boolValue, true)
        let messages = try XCTUnwrap(body["messages"]?.arrayValue)
        XCTAssertEqual(messages.first?["role"]?.stringValue, "system")
        XCTAssertEqual(messages.first?["content"]?.stringValue, "SYS")
        XCTAssertEqual(messages.last?["content"]?.stringValue, "<context>c</context>\n\nHi")
    }

    func testLengthFinishIsTruncated() async throws {
        let transport = MockTransport([.lines([
            #"data: {"choices":[{"delta":{"content":"Long"},"finish_reason":"length"}]}"#,
            "data: [DONE]",
        ])])
        let provider = OpenAICompatibleProvider(configuration: configuration(), transport: transport)
        let events = try await collect(provider.streamReply(system: "S", turns: [ChatTurn(role: .user, text: "Hi")]))
        XCTAssertEqual(events, [.text("Long"), .finished(.truncated)])
    }

    func testMapsErrorResponses() async {
        let transport = MockTransport([.failure(HTTPStatusError(
            statusCode: 400,
            body: #"{"error":{"message":"The model does not exist","type":"invalid_request_error"}}"#
        ))])
        let provider = OpenAICompatibleProvider(configuration: configuration(), transport: transport)
        do {
            _ = try await collect(provider.streamReply(system: "S", turns: [ChatTurn(role: .user, text: "Hi")]))
            XCTFail("Expected an error")
        } catch {
            XCTAssertEqual(
                error as? AssistantError,
                .api(service: "Doubao", status: 400, type: "invalid_request_error", message: "The model does not exist")
            )
        }
    }

    func testRequiresAModel() async {
        let provider = OpenAICompatibleProvider(configuration: configuration(model: " "), transport: MockTransport([]))
        do {
            _ = try await collect(provider.streamReply(system: "S", turns: [ChatTurn(role: .user, text: "Hi")]))
            XCTFail("Expected an error")
        } catch {
            XCTAssertEqual(error as? AssistantError, .missingConfiguration("Enter a model ID for Doubao in Settings."))
        }
    }

    func testHistoryIsTrimmedToStartWithTheUser() throws {
        let turns = [
            ChatTurn(role: .user, text: "one"),
            ChatTurn(role: .assistant, text: "two"),
            ChatTurn(role: .user, text: "three"),
        ]
        let body = OpenAICompatibleProvider.body(model: "m", system: "S", turns: turns, maxHistoryTurns: 2)
        let contents = body["messages"]?.arrayValue?.compactMap { $0["content"]?.stringValue }
        XCTAssertEqual(contents, ["S", "three"])
    }

    func testAssistantTurnsWithoutTextShowTheirToolSummaries() {
        func record(_ id: String, _ name: String, summary: String? = nil) -> ToolCallRecord {
            ToolCallRecord(id: id, name: name, input: [:], result: "{}", summary: summary)
        }
        let reminder = ToolRound(calls: [record("a", "create_reminder", summary: "Reminder: Call mum · today 17:00")])
        let timers = ToolRound(calls: [
            record("b", "set_timer", summary: "Timer: 10 min"),
            record("c", "list_events"),
            record("d", "set_timer", summary: "Timer: 1 h"),
        ])
        let silent = ToolRound(calls: [record("e", "list_events")])
        let turns = [
            ChatTurn(role: .user, text: "one"),
            ChatTurn(role: .assistant, text: "", toolRounds: [reminder, silent, timers]),
            ChatTurn(role: .user, text: "two", context: "<context>c2</context>"),
            // No text and no summaries: skipped, and the user turns around it become one.
            ChatTurn(role: .assistant, text: " \n", toolRounds: [silent]),
            ChatTurn(role: .user, text: "three", context: "<context>c3</context>"),
            // Text wins over summaries.
            ChatTurn(role: .assistant, text: "Done.", toolRounds: [reminder]),
            ChatTurn(role: .user, text: "four"),
        ]

        let body = OpenAICompatibleProvider.body(model: "m", system: "S", turns: turns, maxHistoryTurns: 60)
        let messages = body["messages"]?.arrayValue ?? []

        XCTAssertEqual(messages.compactMap { $0["role"]?.stringValue }, ["system", "user", "assistant", "user", "assistant", "user"])
        XCTAssertEqual(messages.compactMap { $0["content"]?.stringValue }, [
            "S",
            "one",
            "Reminder: Call mum · today 17:00 Timer: 10 min; Timer: 1 h",
            "<context>c2</context>\n\ntwo\n\n<context>c3</context>\n\nthree",
            "Done.",
            "four",
        ])
    }

    func testTurnsWithoutToolRoundsAreUnchanged() {
        let turns = [
            ChatTurn(role: .user, text: "Hi", context: "<context>c</context>"),
            ChatTurn(role: .assistant, text: "Hello!"),
            ChatTurn(role: .user, text: "Bye"),
        ]
        let body = OpenAICompatibleProvider.body(model: "m", system: "S", turns: turns, maxHistoryTurns: 60)
        XCTAssertEqual(body["messages"], [
            ["role": "system", "content": "S"],
            ["role": "user", "content": "<context>c</context>\n\nHi"],
            ["role": "assistant", "content": "Hello!"],
            ["role": "user", "content": "Bye"],
        ])
    }

    func testServiceNames() {
        XCTAssertEqual(CompatibleServices.serviceName(forBaseURL: "https://api.deepseek.com/v1"), "DeepSeek")
        XCTAssertEqual(CompatibleServices.serviceName(forBaseURL: "https://llm.example.com/v1"), "llm.example.com")
    }

    func testSpeechRequestAsksForRawPCM() throws {
        let request = try OpenAISpeech.request(for: "Hello", configuration: OpenAISpeechConfiguration(apiKey: "k", voice: "nova"))
        XCTAssertEqual(request.url?.absoluteString, "https://api.openai.com/v1/audio/speech")
        XCTAssertEqual(request.value(forHTTPHeaderField: "authorization"), "Bearer k")
        let body = try JSONValue.parse(XCTUnwrap(request.httpBody))
        XCTAssertEqual(body["response_format"]?.stringValue, "pcm")
        XCTAssertEqual(body["voice"]?.stringValue, "nova")
        XCTAssertEqual(body["input"]?.stringValue, "Hello")
        XCTAssertNotNil(body["instructions"])
    }

    func testPCM16DecoderHandlesSamplesSplitAcrossChunks() {
        var decoder = PCM16Decoder()
        XCTAssertEqual(decoder.decode(Data([0x00, 0x40, 0x00])), [0.5])
        XCTAssertEqual(decoder.decode(Data([0xC0])), [-0.5])
        XCTAssertEqual(decoder.decode(Data()), [])
    }
}
