import XCTest
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
@testable import AssistantKit

final class ClaudeAgentLoopTests: XCTestCase {
    private let question = [ChatTurn(role: .user, text: "What's on today?")]
    private let checking = ToolPresentation(activity: "Checking your calendar", cue: .checking)

    // MARK: - The loop

    func testToolUseSendsTheAssistantContentBackVerbatimWithOneResultsMessage() async throws {
        let calendar = FakeTool(
            name: "list_events",
            output: .ok(["events": 2], summary: "2 events"),
            schema: stringSchema(["start", "end"]),
            presentation: checking
        )
        let timer = FakeTool(name: "set_timer", effect: .sideEffect)
        let executor = FakeToolExecutor([timer, calendar])
        let arguments = #"{"end":"2026-10-07T23:59","start":"2026-10-07T00:00"}"#
        let transport = MockTransport([
            ClaudeSSE.response(
                [ClaudeSSE.messageStart()],
                ClaudeSSE.thinking(index: 0, signature: "sig-1"),
                MockTransport.toolUse(index: 1, id: "toolu_1", name: "list_events", json: arguments),
                [ClaudeSSE.stop("tool_use")]
            ),
            ClaudeSSE.response(
                [ClaudeSSE.messageStart()],
                ClaudeSSE.text(index: 0, "You have 2 events."),
                [ClaudeSSE.stop("end_turn")]
            ),
        ])
        let provider = ClaudeProvider(configuration: claudeConfiguration(), transport: transport, tools: executor)

        let events = try await collectEvents(provider.streamEvents(system: "S", turns: question))

        let expectedInput: JSONValue = ["start": "2026-10-07T00:00", "end": "2026-10-07T23:59"]
        let round = ToolRound(calls: [
            ToolCallRecord(id: "toolu_1", name: "list_events", input: expectedInput, result: #"{"events":2}"#, summary: "2 events"),
        ])
        XCTAssertEqual(events, [
            .progress(.responseStarted),
            .progress(.toolCallStarted(name: "list_events")),
            .cue(.checking),
            .reply(.activity("Checking your calendar")),
            .toolRound(round),
            .reply(.activity(nil)),
            .reply(.text("You have 2 events.")),
            .reply(.finished(.completed)),
        ])
        XCTAssertEqual(executor.rounds, [[PendingToolCall(id: "toolu_1", name: "list_events", input: expectedInput)]])
        XCTAssertEqual(calendar.inputs, [["start": "2026-10-07T00:00", "end": "2026-10-07T23:59"]])

        XCTAssertEqual(transport.requests.count, 2)
        let first = transport.requests[0].jsonBody
        let second = transport.requests[1].jsonBody
        XCTAssertEqual(first["tools"], second["tools"], "tools stay byte-identical within a reply")
        XCTAssertEqual(first["system"], second["system"])
        XCTAssertNil(second["tool_choice"])

        let messages = try XCTUnwrap(second["messages"]?.arrayValue)
        XCTAssertEqual(messages.count, 3)
        XCTAssertEqual(messages[0], first["messages"]?.arrayValue?.first, "the history is resent unchanged")
        let assistant: JSONValue = [
            "role": "assistant",
            "content": [
                ["type": "thinking", "thinking": "", "signature": "sig-1"],
                ["type": "tool_use", "id": "toolu_1", "name": "list_events", "input": expectedInput],
            ],
        ]
        XCTAssertEqual(messages[1], assistant)
        let results: JSONValue = [
            "role": "user",
            "content": [["type": "tool_result", "tool_use_id": "toolu_1", "content": #"{"events":2}"#]],
        ]
        XCTAssertEqual(messages[2], results, "is_error is sent only when true")
    }

    func testParallelCallsAnswerInOneMessageInCallOrder() async throws {
        let slow = FakeTool(name: "list_events", output: .ok(["events": 0]), delay: 0.1)
        let fast = FakeTool(name: "list_reminders", output: .ok(["reminders": 1]))
        let executor = FakeToolExecutor([slow, fast], concurrent: true)
        let transport = MockTransport([
            ClaudeSSE.response(
                MockTransport.toolUse(index: 0, id: "toolu_a", name: "list_events", json: "{}"),
                MockTransport.toolUse(index: 1, id: "toolu_b", name: "list_reminders", json: #"{"scope":"today"}"#),
                [ClaudeSSE.stop("tool_use")]
            ),
            ClaudeSSE.response(ClaudeSSE.text(index: 0, "Nothing on."), [ClaudeSSE.stop("end_turn")]),
        ])
        let provider = ClaudeProvider(configuration: claudeConfiguration(), transport: transport, tools: executor)

        let events = try await collectEvents(provider.streamEvents(system: "S", turns: question))

        XCTAssertEqual(executor.rounds.map(\.count), [2], "both calls run in one round")
        XCTAssertEqual(executor.log.entries.last, .end("toolu_a"), "the first call finishes last")
        let rounds = events.compactMap { event -> ToolRound? in
            if case .toolRound(let round) = event { return round }
            return nil
        }
        XCTAssertEqual(rounds.count, 1)
        XCTAssertEqual(rounds.first?.calls.map(\.id), ["toolu_a", "toolu_b"])

        let messages = transport.requests[1].sentMessages
        XCTAssertEqual(messages.count, 3)
        XCTAssertEqual(messages[1].blockTypes, ["tool_use", "tool_use"])
        XCTAssertEqual(messages[2].blockTypes, ["tool_result", "tool_result"])
        XCTAssertEqual(messages[2]["content"]?.arrayValue?.compactMap { $0["tool_use_id"]?.stringValue }, ["toolu_a", "toolu_b"])
        XCTAssertEqual(messages[2]["content"]?.arrayValue?.compactMap { $0["content"]?.stringValue }, [#"{"events":0}"#, #"{"reminders":1}"#])
    }

    func testServerSearchAndAClientToolInOneResponse() async throws {
        let reminder = FakeTool(
            name: "create_reminder",
            effect: .sideEffect,
            output: .ok(["id": "r1"], summary: "Reminder: Buy umbrella"),
            presentation: ToolPresentation(activity: "Adding a reminder")
        )
        let executor = FakeToolExecutor([reminder])
        let transport = MockTransport([
            ClaudeSSE.response(
                ClaudeSSE.webSearch(index: 0, id: "srvtoolu_1", query: "rain tomorrow"),
                ClaudeSSE.text(index: 2, "Rain is likely."),
                MockTransport.toolUse(index: 3, id: "toolu_r", name: "create_reminder", json: #"{"title":"Buy umbrella"}"#),
                [ClaudeSSE.stop("tool_use")]
            ),
            ClaudeSSE.response(ClaudeSSE.text(index: 0, " Reminder added."), [ClaudeSSE.stop("end_turn")]),
        ])
        let configuration = claudeConfiguration(webSearch: true)
        let provider = ClaudeProvider(configuration: configuration, transport: transport, tools: executor)

        let events = try await collectEvents(provider.streamEvents(system: "S", turns: question))

        let round = ToolRound(calls: [
            ToolCallRecord(id: "toolu_r", name: "create_reminder", input: ["title": "Buy umbrella"], result: #"{"id":"r1"}"#, summary: "Reminder: Buy umbrella"),
        ])
        XCTAssertEqual(events, [
            .cue(.lookingUp),
            .reply(.activity("Searching the web")),
            .reply(.activity("Searching the web for “rain tomorrow”")),
            .reply(.activity(nil)),
            .reply(.text("Rain is likely.")),
            .progress(.toolCallStarted(name: "create_reminder")),
            .reply(.activity("Adding a reminder")),
            .toolRound(round),
            .reply(.activity(nil)),
            .reply(.text(" Reminder added.")),
            .reply(.finished(.completed)),
        ])
        XCTAssertEqual(reminder.runCount, 1)
        let messages = transport.requests[1].sentMessages
        XCTAssertEqual(messages.count, 3)
        XCTAssertEqual(messages[1].blockTypes, ["server_tool_use", "web_search_tool_result", "text", "tool_use"])
        XCTAssertEqual(messages[2].blockTypes, ["tool_result"], "server tools are answered by the server")
        let tools = transport.requests[0].jsonBody["tools"]?.arrayValue ?? []
        XCTAssertEqual(tools.compactMap { $0["name"]?.stringValue }, ["web_search", "create_reminder"])
    }

    func testPauseTurnThenToolUse() async throws {
        let calendar = FakeTool(name: "list_events")
        let executor = FakeToolExecutor([calendar])
        let transport = MockTransport([
            ClaudeSSE.response(ClaudeSSE.webSearch(index: 0, id: "srvtoolu_1", query: "q"), [ClaudeSSE.stop("pause_turn")]),
            ClaudeSSE.response(
                MockTransport.toolUse(index: 0, id: "toolu_1", name: "list_events", json: "{}"),
                [ClaudeSSE.stop("tool_use")]
            ),
            ClaudeSSE.response(ClaudeSSE.text(index: 0, "Done."), [ClaudeSSE.stop("end_turn")]),
        ])
        let provider = ClaudeProvider(configuration: claudeConfiguration(webSearch: true), transport: transport, tools: executor)

        let events = try await collectEvents(provider.streamEvents(system: "S", turns: question))

        XCTAssertEqual(events.last, .reply(.finished(.completed)))
        XCTAssertEqual(calendar.runCount, 1)
        XCTAssertEqual(transport.requests.count, 3)
        XCTAssertEqual(transport.requests[1].sentMessages.map(\.blockTypes), [["text"], ["server_tool_use", "web_search_tool_result"]])
        XCTAssertEqual(transport.requests[2].sentMessages.map(\.blockTypes), [
            ["text"],
            ["server_tool_use", "web_search_tool_result"],
            ["tool_use"],
            ["tool_result"],
        ])
    }

    func testMaxTokensWithAHalfBuiltToolCallNeverRunsIt() async throws {
        let reminder = FakeTool(name: "create_reminder", effect: .sideEffect)
        let executor = FakeToolExecutor([reminder])
        let cutOff = Array(MockTransport.toolUse(index: 0, id: "toolu_1", name: "create_reminder", json: #"{"title":"Call mum"}"#).prefix(2))
        let transport = MockTransport([ClaudeSSE.response(cutOff, [ClaudeSSE.stop("max_tokens")])])
        let provider = ClaudeProvider(configuration: claudeConfiguration(), transport: transport, tools: executor)

        let events = try await collectEvents(provider.streamEvents(system: "S", turns: question))

        XCTAssertEqual(events.last, .reply(.finished(.truncated)))
        XCTAssertFalse(events.contains { if case .toolRound = $0 { return true } else { return false } })
        XCTAssertEqual(reminder.runCount, 0)
        XCTAssertTrue(executor.rounds.isEmpty)
        XCTAssertEqual(transport.requests.count, 1)
    }

    func testRefusalNeverRunsTools() async throws {
        let reminder = FakeTool(name: "create_reminder", effect: .sideEffect)
        let executor = FakeToolExecutor([reminder])
        let transport = MockTransport([ClaudeSSE.response(
            MockTransport.toolUse(index: 0, id: "toolu_1", name: "create_reminder", json: #"{"title":"x"}"#),
            [ClaudeSSE.stop("refusal", details: ["type": "refusal", "category": "cyber"])]
        )])
        let provider = ClaudeProvider(configuration: claudeConfiguration(), transport: transport, tools: executor)

        let events = try await collectEvents(provider.streamEvents(system: "S", turns: question))

        XCTAssertEqual(events.last, .reply(.finished(.refused(category: "cyber"))))
        XCTAssertEqual(reminder.runCount, 0)
        XCTAssertTrue(executor.rounds.isEmpty)
        XCTAssertEqual(transport.requests.count, 1)
    }

    func testToolLimitEndsTheReplyAfterSixRounds() async throws {
        let calendar = FakeTool(name: "list_events")
        let executor = FakeToolExecutor([calendar])
        let responses = (1...8).map { round in
            ClaudeSSE.response(
                MockTransport.toolUse(index: 0, id: "toolu_\(round)", name: "list_events", json: "{}"),
                [ClaudeSSE.stop("tool_use")]
            )
        }
        let transport = MockTransport(responses)
        let provider = ClaudeProvider(configuration: claudeConfiguration(), transport: transport, tools: executor)

        let events = try await collectEvents(provider.streamEvents(system: "S", turns: question))

        XCTAssertEqual(ClaudeProvider.maxToolRounds, 6)
        XCTAssertEqual(calendar.runCount, 6)
        XCTAssertEqual(transport.requests.count, 7)
        XCTAssertEqual(events.filter { if case .toolRound = $0 { return true } else { return false } }.count, 6)
        XCTAssertEqual(events.last, .reply(.finished(.other("tool_limit"))))
        // The unanswered seventh call is not sent back.
        XCTAssertEqual(transport.requests[6].sentMessages.count, 13)
    }

    func testToolUseWithoutClientCallsEnds() async throws {
        let transport = MockTransport([ClaudeSSE.response(ClaudeSSE.text(index: 0, "Hmm."), [ClaudeSSE.stop("tool_use")])])
        let provider = ClaudeProvider(configuration: claudeConfiguration(), transport: transport, tools: FakeToolExecutor([]))

        let events = try await collectEvents(provider.streamEvents(system: "S", turns: question))

        XCTAssertEqual(events, [.reply(.text("Hmm.")), .reply(.finished(.other("tool_use")))])
        XCTAssertEqual(transport.requests.count, 1)
    }

    func testInvalidToolInputIsAnsweredWithoutRunning() async throws {
        let reminder = FakeTool(name: "create_reminder", effect: .sideEffect, schema: stringSchema(["title"]))
        let calendar = FakeTool(name: "list_events")
        let executor = FakeToolExecutor([reminder, calendar])
        let broken = #"{"title": "Call mum"#
        let transport = MockTransport([
            ClaudeSSE.response(
                MockTransport.toolUse(index: 0, id: "toolu_bad", name: "create_reminder", json: broken),
                MockTransport.toolUse(index: 1, id: "toolu_ok", name: "list_events", json: "{}"),
                [ClaudeSSE.stop("tool_use")]
            ),
            ClaudeSSE.response(ClaudeSSE.text(index: 0, "Let me try again."), [ClaudeSSE.stop("end_turn")]),
        ])
        let provider = ClaudeProvider(configuration: claudeConfiguration(), transport: transport, tools: executor)

        let events = try await collectEvents(provider.streamEvents(system: "S", turns: question))

        XCTAssertEqual(reminder.runCount, 0)
        XCTAssertEqual(calendar.runCount, 1)
        XCTAssertEqual(executor.rounds, [[PendingToolCall(id: "toolu_ok", name: "list_events", input: .object([:]))]])
        let invalid = ToolCallRecord(
            id: "toolu_bad",
            name: "create_reminder",
            input: .object([:]),
            result: JSONValue.object(["INVALID_JSON": .string(broken)]).serializedText,
            isError: true
        )
        let round = events.compactMap { event -> ToolRound? in
            if case .toolRound(let round) = event { return round }
            return nil
        }.first
        XCTAssertEqual(round?.calls.first, invalid)
        XCTAssertEqual(round?.calls.map(\.id), ["toolu_bad", "toolu_ok"])

        let messages = transport.requests[1].sentMessages
        XCTAssertEqual(messages[1]["content"]?.arrayValue?.first?["input"], .object([:]), "a bad input goes back empty")
        let result = try XCTUnwrap(messages[2]["content"]?.arrayValue?.first)
        XCTAssertEqual(result["is_error"], true)
        let content = try XCTUnwrap(result["content"]?.stringValue)
        XCTAssertEqual(try JSONValue.parse(content), ["INVALID_JSON": .string(broken)])
    }

    func testCancellationDuringAToolRunStopsTheLoop() async throws {
        let slow = FakeTool(name: "list_events", delay: 30)
        let executor = FakeToolExecutor([slow])
        let transport = MockTransport([
            ClaudeSSE.response(
                MockTransport.toolUse(index: 0, id: "toolu_1", name: "list_events", json: "{}"),
                [ClaudeSSE.stop("tool_use")]
            ),
            ClaudeSSE.response(ClaudeSSE.text(index: 0, "Too late."), [ClaudeSSE.stop("end_turn")]),
        ])
        let provider = ClaudeProvider(configuration: claudeConfiguration(), transport: transport, tools: executor)
        let stream = provider.streamEvents(system: "S", turns: question)
        let consumer = Task { try await collectEvents(stream) }

        let started = await waitUntil { slow.runCount == 1 }
        XCTAssertTrue(started)
        consumer.cancel()
        let ended = await waitUntil { slow.log.entries.count == 2 }
        XCTAssertTrue(ended, "the running tool is cancelled")
        _ = try? await consumer.value
        try await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertEqual(transport.requests.count, 1, "no request follows a cancelled round")
    }

    func testToolContextReachesTheExecutor() async throws {
        let gate = CommitGate()
        let executor = FakeToolExecutor([FakeTool(name: "list_events")])
        let transport = MockTransport([
            ClaudeSSE.response(MockTransport.toolUse(index: 0, id: "toolu_1", name: "list_events", json: "{}"), [ClaudeSSE.stop("tool_use")]),
            ClaudeSSE.response(ClaudeSSE.text(index: 0, "OK"), [ClaudeSSE.stop("end_turn")]),
        ])
        let provider = ClaudeProvider(
            configuration: claudeConfiguration(),
            transport: transport,
            tools: executor,
            toolContext: { ToolContext(commitGate: gate) }
        )

        _ = try await collectEvents(provider.streamEvents(system: "S", turns: question))

        XCTAssertEqual(executor.contexts.count, 1)
        XCTAssertTrue(executor.contexts.first?.commitGate === gate)
    }

    func testStreamReplyKeepsOnlyReplyEvents() async throws {
        let executor = FakeToolExecutor([FakeTool(name: "list_events", presentation: checking)])
        let transport = MockTransport([
            ClaudeSSE.response(
                [ClaudeSSE.messageStart()],
                MockTransport.toolUse(index: 0, id: "toolu_1", name: "list_events", json: "{}"),
                [ClaudeSSE.stop("tool_use")]
            ),
            ClaudeSSE.response(ClaudeSSE.text(index: 0, "Free all day."), [ClaudeSSE.stop("end_turn")]),
        ])
        let provider = ClaudeProvider(configuration: claudeConfiguration(), transport: transport, tools: executor)

        let replies = try await collect(provider.streamReply(system: "S", turns: question))

        XCTAssertEqual(replies, [
            .activity("Checking your calendar"),
            .activity(nil),
            .text("Free all day."),
            .finished(.completed),
        ])
    }

    // MARK: - Rendering history

    private func historyWithRounds() -> [ChatTurn] {
        let local = ToolCallRecord(id: "call_0", name: "set_timer", input: ["seconds": 300], result: #"{"id":"t1"}"#, summary: "Timer: 5 min")
        let failed = ToolCallRecord(id: "toolu_9", name: "list_events", input: .object([:]), result: #"{"error":"Calendar access is off."}"#, isError: true)
        return [
            ChatTurn(role: .user, text: "Set a timer", context: "<context>t1</context>"),
            ChatTurn(role: .assistant, text: "Timer set.", toolRounds: [ToolRound(calls: [local])]),
            ChatTurn(role: .user, text: "And my events?"),
            ChatTurn(role: .assistant, text: "", toolRounds: [ToolRound(calls: [failed])]),
            ChatTurn(role: .user, text: "Try again"),
        ]
    }

    func testRendersRoundsAsToolUseAndResultMessages() {
        let messages = ClaudeRequest.messages(from: historyWithRounds())

        XCTAssertEqual(messages.map { $0["role"]?.stringValue ?? "" }, ["user", "assistant", "user", "assistant", "user", "assistant", "user"])
        XCTAssertEqual(messages.map(\.blockTypes), [
            ["text", "text"],
            ["tool_use"],
            ["tool_result"],
            ["text"],
            ["text"],
            ["tool_use"],
            ["tool_result", "text"],
        ])
        let toolUse: JSONValue = ["type": "tool_use", "id": "toolu_call_0", "name": "set_timer", "input": ["seconds": 300]]
        XCTAssertEqual(messages[1]["content"]?.arrayValue, [toolUse])
        let result: JSONValue = ["type": "tool_result", "tool_use_id": "toolu_call_0", "content": #"{"id":"t1"}"#]
        XCTAssertEqual(messages[2]["content"]?.arrayValue, [result])
        let failed = messages[6]["content"]?.arrayValue?.first
        XCTAssertEqual(failed?["is_error"], true)
        XCTAssertEqual(messages[6]["content"]?.arrayValue?.last?["text"]?.stringValue, "Try again", "the next user turn joins the results")
    }

    func testRenderingTheSameRoundsTwiceIsByteIdentical() throws {
        let configuration = claudeConfiguration()
        let tools = [ToolDefinition(name: "set_timer", description: "Timer", inputSchema: stringSchema(["seconds"]))]
        let render = { () throws -> Data in
            let messages = ClaudeRequest.messages(from: self.historyWithRounds())
            let body = ClaudeRequest.body(configuration: configuration, system: "S", messages: messages, clientTools: tools)
            return try XCTUnwrap(ClaudeRequest.urlRequest(configuration: configuration, body: body).httpBody)
        }
        XCTAssertEqual(try render(), try render())
    }

    func testToolIDsAreRemappedConsistently() {
        XCTAssertEqual(ClaudeRequest.claudeToolID("toolu_01AbC"), "toolu_01AbC")
        XCTAssertEqual(ClaudeRequest.claudeToolID("call_0"), "toolu_call_0")
        XCTAssertEqual(ClaudeRequest.claudeToolID("call 1:a.b/é-x"), "toolu_call_1_a_b__-x")

        let record = ToolCallRecord(id: "call 1:a", name: "get_current_time", input: .object([:]), result: "{}")
        let turns = [
            ChatTurn(role: .user, text: "Time?"),
            ChatTurn(role: .assistant, text: "Noon.", toolRounds: [ToolRound(calls: [record])]),
            ChatTurn(role: .user, text: "Thanks"),
        ]
        let messages = ClaudeRequest.messages(from: turns)
        XCTAssertEqual(messages[1]["content"]?.arrayValue?.first?["id"]?.stringValue, "toolu_call_1_a")
        XCTAssertEqual(messages[2]["content"]?.arrayValue?.first?["tool_use_id"]?.stringValue, "toolu_call_1_a")
    }

    // MARK: - Options

    private let followUp = [
        ChatTurn(role: .user, text: "Plan my week"),
        ChatTurn(role: .assistant, text: "Sure."),
        ChatTurn(role: .user, text: "Think harder"),
    ]

    private func singleRequest(
        configuration: ClaudeConfiguration,
        options: ClaudeTurnOptions,
        turns: [ChatTurn]? = nil,
        tools: [FakeTool] = []
    ) async throws -> URLRequest {
        let transport = MockTransport([ClaudeSSE.response(ClaudeSSE.text(index: 0, "OK"), [ClaudeSSE.stop("end_turn")])])
        let provider = ClaudeProvider(configuration: configuration, transport: transport, tools: FakeToolExecutor(tools), options: options)
        _ = try await collectEvents(provider.streamEvents(system: "S", turns: turns ?? followUp))
        return try XCTUnwrap(transport.requests.first)
    }

    func testEffortOverrideOnOpus55IsAPerMessageChange() async throws {
        let request = try await singleRequest(
            configuration: claudeConfiguration(model: "claude-opus-5-5", effort: "low"),
            options: ClaudeTurnOptions(effort: "high")
        )

        let body = request.jsonBody
        XCTAssertEqual(body["output_config"]?["effort"]?.stringValue, "low", "the cached top-level effort is unchanged")
        let messages = request.sentMessages
        XCTAssertEqual(messages.map { $0["role"]?.stringValue ?? "" }, ["user", "assistant", "system", "user"])
        let change: JSONValue = ["role": "system", "content": .array([]), "output_config": ["effort": "high"]]
        XCTAssertEqual(messages[2], change)
        XCTAssertEqual(
            request.value(forHTTPHeaderField: "anthropic-beta"),
            "server-side-fallback-2026-07-01,mid-conversation-output-config-2026-07-01"
        )
    }

    func testEffortOverrideOnSonnet5SetsTheTopLevelEffort() async throws {
        let request = try await singleRequest(
            configuration: claudeConfiguration(model: "claude-sonnet-5", effort: "low"),
            options: ClaudeTurnOptions(effort: "high")
        )

        XCTAssertEqual(request.jsonBody["output_config"]?["effort"]?.stringValue, "high")
        XCTAssertEqual(request.sentMessages.map { $0["role"]?.stringValue ?? "" }, ["user", "assistant", "user"])
        XCTAssertNil(request.value(forHTTPHeaderField: "anthropic-beta"))
    }

    func testEffortEqualToTheConfigurationChangesNothing() async throws {
        let request = try await singleRequest(
            configuration: claudeConfiguration(model: "claude-opus-5-5", effort: "high"),
            options: ClaudeTurnOptions(effort: "high")
        )

        XCTAssertEqual(request.jsonBody["output_config"]?["effort"]?.stringValue, "high")
        XCTAssertEqual(request.sentMessages.count, 3)
        XCTAssertEqual(request.value(forHTTPHeaderField: "anthropic-beta"), "server-side-fallback-2026-07-01")
    }

    func testEffortChangeSkipsAResultsMessageThatTheUserTurnJoined() {
        let record = ToolCallRecord(id: "toolu_1", name: "list_events", input: .object([:]), result: "{}")
        let turns = [
            ChatTurn(role: .user, text: "Events?"),
            ChatTurn(role: .assistant, text: "", toolRounds: [ToolRound(calls: [record])]),
            ChatTurn(role: .user, text: "Think harder"),
        ]
        let capabilities = ClaudeModelCatalog.capabilities(for: "claude-opus-5-5")
        let messages = ClaudeRequest.messages(from: turns, options: ClaudeTurnOptions(effort: "high"), capabilities: capabilities)

        XCTAssertEqual(messages.map { $0["role"]?.stringValue ?? "" }, ["user", "system", "assistant", "user"])
        XCTAssertEqual(messages[3].blockTypes, ["tool_result", "text"], "results still directly follow their tool_use")
    }

    func testInstructionIsATrailingSystemMessageWhereSupported() async throws {
        let calendar = FakeTool(name: "list_events")
        let transport = MockTransport([
            ClaudeSSE.response(MockTransport.toolUse(index: 0, id: "toolu_1", name: "list_events", json: "{}"), [ClaudeSSE.stop("tool_use")]),
            ClaudeSSE.response(ClaudeSSE.text(index: 0, "Merged."), [ClaudeSSE.stop("end_turn")]),
        ])
        let options = ClaudeTurnOptions(instruction: "Merge the notes.", userData: "<analyst_notes>n</analyst_notes>")
        let provider = ClaudeProvider(configuration: claudeConfiguration(), transport: transport, tools: FakeToolExecutor([calendar]), options: options)

        _ = try await collectEvents(provider.streamEvents(system: "S", turns: followUp))

        let first = transport.requests[0].sentMessages
        XCTAssertEqual(first.map { $0["role"]?.stringValue ?? "" }, ["user", "assistant", "user", "system"])
        XCTAssertEqual(first[3], ["role": "system", "content": "Merge the notes."])
        XCTAssertEqual(first[2]["content"]?.arrayValue?.compactMap { $0["text"]?.stringValue }, ["Think harder", "<analyst_notes>n</analyst_notes>"])
        let second = transport.requests[1].sentMessages
        XCTAssertEqual(Array(second.prefix(4)), first, "the loop only appends")
        XCTAssertEqual(second.map { $0["role"]?.stringValue ?? "" }, ["user", "assistant", "user", "system", "assistant", "user"])
    }

    func testInstructionIsAnInstructionsBlockOnSonnet5() async throws {
        let request = try await singleRequest(
            configuration: claudeConfiguration(model: "claude-sonnet-5"),
            options: ClaudeTurnOptions(instruction: "Merge the notes.", userData: "<analyst_notes>n</analyst_notes>")
        )

        let messages = request.sentMessages
        XCTAssertEqual(messages.map { $0["role"]?.stringValue ?? "" }, ["user", "assistant", "user"])
        XCTAssertEqual(messages[2]["content"]?.arrayValue?.compactMap { $0["text"]?.stringValue }, [
            "Think harder",
            "<analyst_notes>n</analyst_notes>",
            "<instructions>\nMerge the notes.\n</instructions>",
        ])
    }

    func testMaxTokensOverride() async throws {
        let request = try await singleRequest(configuration: claudeConfiguration(), options: ClaudeTurnOptions(maxTokens: 4_000))
        XCTAssertEqual(request.jsonBody["max_tokens"]?.intValue, 4_000)
    }

    func testClientToolsAreStrictSortedAndEagerOnlyForTheDirectAPI() async throws {
        let tools = [FakeTool(name: "set_timer", schema: stringSchema(["label"])), FakeTool(name: "list_events")]

        let direct = try await singleRequest(configuration: claudeConfiguration(webSearch: true), options: .init(), tools: tools)
        let declared = try XCTUnwrap(direct.jsonBody["tools"]?.arrayValue)
        XCTAssertEqual(declared.compactMap { $0["name"]?.stringValue }, ["web_search", "list_events", "set_timer"])
        XCTAssertNil(declared[0]["strict"], "the server tool keeps its own shape")
        let timer: JSONValue = [
            "name": "set_timer",
            "description": "Fake tool set_timer.",
            "input_schema": stringSchema(["label"]),
            "strict": true,
            "eager_input_streaming": true,
        ]
        XCTAssertEqual(declared[2], timer)
        XCTAssertNil(direct.jsonBody["tool_choice"])

        let proxied = try await singleRequest(
            configuration: claudeConfiguration(baseURL: "https://proxy.example.com"),
            options: .init(),
            tools: tools
        )
        let proxiedTools = try XCTUnwrap(proxied.jsonBody["tools"]?.arrayValue)
        XCTAssertEqual(proxiedTools.count, 2)
        XCTAssertTrue(proxiedTools.allSatisfy { $0["eager_input_streaming"] == nil && $0["strict"] == true })

        let off = try await singleRequest(configuration: claudeConfiguration(), options: ClaudeTurnOptions(eagerToolInput: false), tools: tools)
        XCTAssertTrue(off.jsonBody["tools"]?.arrayValue?.allSatisfy { $0["eager_input_streaming"] == nil } ?? false)
    }

    func testNoToolsMeansNoToolsField() async throws {
        let request = try await singleRequest(configuration: claudeConfiguration(), options: .init())
        XCTAssertNil(request.jsonBody["tools"])
    }

    // MARK: - Prewarm

    func testPrewarmRequestShape() throws {
        let request = try ClaudeProvider.prewarmRequest(configuration: claudeConfiguration())
        XCTAssertEqual(request.httpMethod, "GET")
        XCTAssertEqual(request.url?.absoluteString, "https://api.anthropic.com/v1/models?limit=1")
        XCTAssertEqual(request.value(forHTTPHeaderField: "x-api-key"), "sk-test")
        XCTAssertEqual(request.value(forHTTPHeaderField: "anthropic-version"), "2023-06-01")
        XCTAssertNil(request.value(forHTTPHeaderField: "anthropic-beta"))
        XCTAssertNil(request.httpBody)

        let proxied = try ClaudeProvider.prewarmRequest(configuration: claudeConfiguration(baseURL: "https://proxy.example.com/anthropic"))
        XCTAssertEqual(proxied.url?.absoluteString, "https://proxy.example.com/anthropic/v1/models?limit=1")

        XCTAssertThrowsError(try ClaudeProvider.prewarmRequest(configuration: ClaudeConfiguration(apiKey: " "))) { error in
            XCTAssertEqual(error as? AssistantError, .missingAPIKey(service: "Anthropic"))
        }
    }
}
