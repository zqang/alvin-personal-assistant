import XCTest
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
@testable import AssistantKit

final class DeliberationTests: XCTestCase {
    private let question = [ChatTurn(role: .user, text: "Should I move to Singapore for the job?")]
    private let opening: [AssistantEvent] = [.cue(.deepThinking), .reply(.activity("Thinking it through"))]
    private let thinking = AssistantEvent.reply(.activity("Thinking it through"))

    /// The events of a streamed answer `text` after "Thinking it through" was shown.
    private func answer(_ text: String) -> [AssistantEvent] {
        [.progress(.responseStarted), .reply(.activity(nil)), .reply(.text(text)), .reply(.finished(.completed))]
    }

    private func toolRounds(_ events: [AssistantEvent]) -> [ToolRound] {
        events.compactMap { event -> ToolRound? in
            if case .toolRound(let round) = event { return round }
            return nil
        }
    }

    private func workerBodies(_ transport: ScriptedTransport) -> [JSONValue] {
        transport.bodies.filter { body in
            if case .worker = DeepRequests.kind(of: body) { return true }
            return false
        }
    }

    // MARK: - Configuration

    func testPresets() {
        let voice = DeliberationConfiguration.voice()
        XCTAssertEqual(voice.deadline, .seconds(25))
        XCTAssertEqual(voice.mergerMaxTokens, 4_000)
        let typed = DeliberationConfiguration.typed()
        XCTAssertEqual(typed.deadline, .seconds(60))
        XCTAssertEqual(typed.mergerMaxTokens, 16_000)
        for configuration in [voice, typed] {
            XCTAssertEqual(configuration.strategy, .single)
            XCTAssertEqual(configuration.singleEffort, "high")
            XCTAssertEqual(configuration.workers, [.researcher, .reasoner, .critic])
            XCTAssertNil(configuration.workerModel)
            XCTAssertEqual(configuration.workerMaxTokens, 12_000)
            XCTAssertEqual(configuration.mergerInstruction, DeliberationConfiguration.defaultMergerInstruction)
        }
    }

    func testBriefs() {
        let briefs: [WorkerBrief] = [.researcher, .reasoner, .critic]
        XCTAssertEqual(briefs.map(\.id), ["researcher", "reasoner", "critic"])
        XCTAssertEqual(briefs.map(\.effort), ["low", "high", "medium"])
        XCTAssertEqual(briefs.map(\.usesTools), [true, false, false])
        XCTAssertEqual(briefs.map(\.forwardsActivity), [true, false, false])
        XCTAssertTrue(WorkerBrief.researcher.instruction.contains("Use web search"))
        XCTAssertTrue(WorkerBrief.reasoner.instruction.contains("Don't search the web."))
        XCTAssertTrue(WorkerBrief.critic.instruction.contains("Don't search the web."))
        XCTAssertTrue(WorkerBrief.reasoner.instruction.contains("the 3–5 considerations"))
        XCTAssertTrue(DeliberationConfiguration.defaultMergerInstruction.contains("Never mention analysts, notes or a deliberation."))
        // Conclusions only: nothing asks for the reasoning itself to be written out.
        for text in briefs.map(\.instruction) + [DeliberationConfiguration.defaultMergerInstruction] {
            XCTAssertFalse(text.lowercased().contains("step by step"))
            XCTAssertFalse(text.lowercased().contains("chain of thought"))
        }
    }

    func testAnalystNotesBlock() {
        XCTAssertEqual(
            AnalystNotes.block([(id: "researcher", text: "Rents rose."), (id: "critic", text: "Ask first.")]),
            "<analyst_notes>\n<research>\nRents rose.\n</research>\n<critique>\nAsk first.\n</critique>\n</analyst_notes>"
        )
        XCTAssertEqual(AnalystNotes.tag(for: "reasoner"), "reasoning")
        XCTAssertEqual(AnalystNotes.tag(for: "Fact-checker"), "fact_checker")
        XCTAssertEqual(AnalystNotes.tag(for: "2nd opinion"), "notes_2nd_opinion")
        XCTAssertEqual(AnalystNotes.tag(for: ""), "notes")
    }

    // MARK: - Single

    func testSingleCuesBeforeAnyNetworkEvent() async throws {
        // The service fails at once: the cue and the activity still come first.
        let failing = MockTransport([.failure(HTTPStatusError(statusCode: 400, body: #"{"error":{"type":"invalid_request_error","message":"bad"}}"#))])
        var events: [AssistantEvent] = []
        do {
            for try await event in makeDeliberation(deepConfiguration(.single), transport: failing).streamEvents(system: "S", turns: question) {
                events.append(event)
            }
            XCTFail("the request's error should end the reply")
        } catch {
            XCTAssertEqual(error as? AssistantError, .api(service: "Anthropic", status: 400, type: "invalid_request_error", message: "bad"))
        }
        XCTAssertEqual(events, opening)

        // The network never answers: the cue and the activity arrive anyway.
        let hanging = HangingTransport()
        let recorder = DeepEventRecorder()
        let consumer = recorder.consume(makeDeliberation(deepConfiguration(.single), transport: hanging).streamEvents(system: "S", turns: question))
        let cued = await waitUntil { recorder.events.count == 2 && hanging.requests.count == 1 }
        XCTAssertTrue(cued)
        XCTAssertEqual(recorder.events, opening)
        consumer.cancel()
        _ = await consumer.value
        let ended = await waitUntil { hanging.terminationCount == 1 }
        XCTAssertTrue(ended, "cancelling the reply cancels its request")
    }

    func testSinglePassesTheAnswerThrough() async throws {
        let transport = MockTransport([deepText("Stay in Berlin.")])
        let events = try await collectEvents(makeDeliberation(deepConfiguration(.single), transport: transport).streamEvents(system: "S", turns: question))
        XCTAssertEqual(events, opening + answer("Stay in Berlin."))
        XCTAssertEqual(transport.requests.count, 1)

        let replies = try await collect(
            makeDeliberation(deepConfiguration(.single), transport: MockTransport([deepText("Stay.")])).streamReply(system: "S", turns: question)
        )
        XCTAssertEqual(replies, [.activity("Thinking it through"), .activity(nil), .text("Stay."), .finished(.completed)])
    }

    func testSingleIsOneHighEffortCallWithTheFullTools() async throws {
        let reminder = FakeTool(name: "create_reminder", effect: .sideEffect, output: .ok(["id": "r1"], summary: "Reminder: Visit the flat"))
        let calendar = FakeTool(name: "list_events")
        let transport = deepTransport { call, body in
            guard call == .plain else { return .failure(URLError(.badURL)) }
            return DeepRequests.answersToolResults(body) ? deepText("Done. Here's my advice.") : deepToolCalls([("toolu_1", "create_reminder")])
        }
        let deliberation = makeDeliberation(deepConfiguration(.single), transport: transport, tools: ToolRegistry([reminder, calendar]))

        let events = try await collectEvents(deliberation.streamEvents(system: "S", turns: question))

        XCTAssertEqual(Array(events.prefix(2)), opening)
        XCTAssertEqual(events.last, .reply(.finished(.completed)))
        XCTAssertEqual(reminder.runCount, 1, "the single call may act")
        XCTAssertEqual(toolRounds(events).map { $0.calls.map(\.name) }, [["create_reminder"]])
        XCTAssertEqual(transport.requests.count, 2)
        let body = transport.bodies[0]
        XCTAssertEqual(DeepRequests.effortMessage(of: body), "high")
        XCTAssertEqual(
            transport.requests[0].value(forHTTPHeaderField: "anthropic-beta"),
            "server-side-fallback-2026-07-01,mid-conversation-output-config-2026-07-01"
        )
        XCTAssertNil(DeepRequests.instruction(of: body))
        XCTAssertNil(DeepRequests.analystNotes(of: body))
        XCTAssertEqual(body["max_tokens"], 16_000)
        XCTAssertEqual(body["tools"]?.arrayValue?.compactMap { $0["name"]?.stringValue }, ["create_reminder", "list_events"])
    }

    func testSingleSaysStillThinkingAtHalfTheDeadlineBeforeAnyText() async {
        let clock = ManualClock()
        let transport = ScriptedTransport { _, _ in .hang(MockTransport.sse([ClaudeSSE.messageStart()])) }
        let recorder = DeepEventRecorder()
        let deliberation = makeDeliberation(deepConfiguration(.single, voice: true), transport: transport, clock: clock)
        let consumer = recorder.consume(deliberation.streamEvents(system: "S", turns: question))
        let waiting = await waitUntil { recorder.events.count == 3 && clock.sleeperCount == 1 }
        XCTAssertTrue(waiting)
        XCTAssertEqual(recorder.events, opening + [.progress(.responseStarted)])

        clock.advance(to: 12.4)
        try? await Task.sleep(nanoseconds: 20_000_000)
        XCTAssertEqual(recorder.events.count, 3)
        clock.advance(to: 12.5)
        let cued = await waitUntil { recorder.events.count == 4 }
        XCTAssertTrue(cued)
        XCTAssertEqual(recorder.events.last, .cue(.stillThinking))

        clock.advance(to: 100)
        try? await Task.sleep(nanoseconds: 20_000_000)
        XCTAssertEqual(recorder.events.count, 4, "the cue plays once, and the single call has no deadline")
        consumer.cancel()
        _ = await consumer.value
        let cancelled = await waitUntil { transport.cancellationCount == 1 }
        XCTAssertTrue(cancelled)
    }

    func testNoStillThinkingOnceTextHasArrived() async {
        let clock = ManualClock()
        let transport = ScriptedTransport { _, _ in
            .hang(MockTransport.sse([ClaudeSSE.messageStart()] + ClaudeSSE.text(index: 0, "Short answer:")))
        }
        let recorder = DeepEventRecorder()
        let deliberation = makeDeliberation(deepConfiguration(.single, voice: true), transport: transport, clock: clock)
        let consumer = recorder.consume(deliberation.streamEvents(system: "S", turns: question))
        let answering = await waitUntil { recorder.events.contains(.reply(.text("Short answer:"))) && clock.sleeperCount == 1 }
        XCTAssertTrue(answering)

        clock.advance(to: 30)
        try? await Task.sleep(nanoseconds: 30_000_000)
        XCTAssertEqual(clock.sleeperCount, 0)
        XCTAssertEqual(recorder.events, opening + [.progress(.responseStarted), .reply(.activity(nil)), .reply(.text("Short answer:"))])
        consumer.cancel()
        _ = await consumer.value
    }

    func testNoDeadlineMeansNoTimers() async {
        let clock = ManualClock()
        var configuration = deepConfiguration(.single)
        configuration.deadline = .zero
        let transport = HangingTransport()
        let recorder = DeepEventRecorder()
        let consumer = recorder.consume(makeDeliberation(configuration, transport: transport, clock: clock).streamEvents(system: "S", turns: question))
        let started = await waitUntil { transport.requests.count == 1 }
        XCTAssertTrue(started)
        try? await Task.sleep(nanoseconds: 30_000_000)
        XCTAssertEqual(clock.sleeperCount, 0)
        consumer.cancel()
        _ = await consumer.value
    }

    // MARK: - Parallel

    func testParallelRunsEachWorkerOnceThenStreamsTheMerger() async throws {
        let transport = deepTransport { call, _ in
            switch call {
            case .worker("researcher"): return deepText("Rents rose 4% (URA, Sept 2026).")
            case .worker("reasoner"): return deepText("Move only for a 20% raise.")
            case .worker("critic"): return deepText("Ask whether the family can move.")
            case .merger: return deepText("Move only if the raise covers the rent.")
            default: return .failure(URLError(.badURL))
            }
        }
        let deliberation = makeDeliberation(deepConfiguration(.parallel), transport: transport)

        let events = try await collectEvents(deliberation.streamEvents(system: "S", turns: question))

        XCTAssertEqual(events, opening + answer("Move only if the raise covers the rent."), "worker events stay inside")
        let kinds = transport.bodies.map { DeepRequests.kind(of: $0) }
        XCTAssertEqual(kinds.count, 4)
        XCTAssertEqual(Set(kinds.prefix(3)), [.worker("researcher"), .worker("reasoner"), .worker("critic")])
        XCTAssertEqual(kinds.last, .merger)

        var efforts: [String: String] = [:]
        for body in workerBodies(transport) {
            guard case .worker(let id) = DeepRequests.kind(of: body) else { continue }
            efforts[id] = DeepRequests.effortMessage(of: body) ?? "none"
            XCTAssertEqual(body["max_tokens"], 12_000)
            XCTAssertEqual(body["model"], "claude-opus-5-5")
            XCTAssertNil(DeepRequests.analystNotes(of: body))
        }
        XCTAssertEqual(efforts, ["researcher": "none", "reasoner": "high", "critic": "medium"], "low is the configuration's effort already")

        let merger = try XCTUnwrap(transport.bodies.last)
        XCTAssertEqual(merger["max_tokens"], 16_000)
        XCTAssertEqual(merger["model"], "claude-opus-5-5")
        XCTAssertNil(DeepRequests.effortMessage(of: merger), "the merger keeps the user's effort")
        XCTAssertEqual(DeepRequests.analystNotes(of: merger), """
        <analyst_notes>
        <research>
        Rents rose 4% (URA, Sept 2026).
        </research>
        <reasoning>
        Move only for a 20% raise.
        </reasoning>
        <critique>
        Ask whether the family can move.
        </critique>
        </analyst_notes>
        """)
    }

    func testEveryRequestSharesThePrefixThroughTheHistory() async throws {
        let reminder = FakeTool(name: "create_reminder", effect: .sideEffect)
        let calendar = FakeTool(name: "list_events")
        let history = [
            ChatTurn(role: .user, text: "Remind me to call mum at five.", context: "Typed at 16:00."),
            ChatTurn(role: .assistant, text: "Done, and a timer too.", toolRounds: [ToolRound(calls: [
                ToolCallRecord(id: "toolu_a", name: "create_reminder", input: ["title": "Call mum"], result: #"{"id":"r1"}"#),
                ToolCallRecord(id: "call_b", name: "set_timer", input: ["seconds": 60], result: #"{"ok":true}"#),
            ])]),
            ChatTurn(role: .user, text: "Should I move to Singapore for the job?"),
        ]
        let transport = deepTransport { call, _ in call == .merger ? deepText("Answer.") : deepText("Notes.") }
        let deliberation = makeDeliberation(deepConfiguration(.parallel), transport: transport, tools: ToolRegistry([reminder, calendar]))

        _ = try await collectEvents(deliberation.streamEvents(system: "S", turns: history))

        XCTAssertEqual(transport.bodies.count, 4)
        // user; assistant tool_use; user tool_result; assistant text. The last user turn follows.
        let historyCount = 4
        func prefix(_ body: JSONValue) -> String {
            var shared: [String: JSONValue] = [:]
            for key in ["model", "system", "tools", "cache_control", "fallbacks", "output_config", "stream"] {
                shared[key] = body[key] ?? .null
            }
            shared["history"] = .array(Array((body["messages"]?.arrayValue ?? []).prefix(historyCount)))
            return JSONValue.object(shared).serializedText
        }
        let workers = workerBodies(transport)
        XCTAssertEqual(workers.count, 3)
        XCTAssertEqual(Set(workers.map(prefix)).count, 1, "workers send the same bytes through the history")
        XCTAssertEqual(Set(transport.bodies.map(prefix)).count, 1, "and so does the merger")
        XCTAssertEqual(Set(workers.compactMap { $0["max_tokens"]?.intValue }), [12_000])

        let messages = transport.bodies[0]["messages"]?.arrayValue ?? []
        XCTAssertEqual(messages.prefix(historyCount).map { $0["role"]?.stringValue }, ["user", "assistant", "user", "assistant"])
        XCTAssertEqual(messages[1].blockTypes, ["tool_use", "tool_use"])
        XCTAssertEqual(
            transport.bodies[0]["tools"]?.arrayValue?.compactMap { $0["name"]?.stringValue },
            ["create_reminder", "list_events", "set_timer"],
            "a tool only the history used is stubbed"
        )
        for body in workers {
            let tail = (body["messages"]?.arrayValue ?? []).dropFirst(historyCount)
            XCTAssertEqual(tail.last?["role"], "system", "the brief comes last")
            XCTAssertTrue(tail.contains { $0["role"]?.stringValue == "user" && $0["content"]?.arrayValue?.first?["text"] == "Should I move to Singapore for the job?" })
        }
    }

    func testBriefsAreTrailingSystemMessagesOnOpus55() async throws {
        let transport = deepTransport { call, _ in call == .merger ? deepText("Answer.") : deepText("Notes.") }
        _ = try await collectEvents(makeDeliberation(deepConfiguration(.parallel), transport: transport).streamEvents(system: "S", turns: question))

        var briefs: [String] = []
        for body in transport.bodies {
            let last = try XCTUnwrap(body["messages"]?.arrayValue?.last)
            XCTAssertEqual(last["role"], "system")
            briefs.append(try XCTUnwrap(last["content"]?.stringValue))
            XCTAssertFalse(body.serializedText.contains("<instructions>"))
        }
        let expected = [WorkerBrief.researcher, .reasoner, .critic].map(\.instruction) + [DeliberationConfiguration.defaultMergerInstruction]
        XCTAssertEqual(Set(briefs), Set(expected))
        XCTAssertEqual(briefs.last, DeliberationConfiguration.defaultMergerInstruction)

        // The notes end the last user message; the brief follows it.
        let merger = try XCTUnwrap(transport.bodies.last?["messages"]?.arrayValue)
        let user = merger[merger.count - 2]
        XCTAssertEqual(user["role"], "user")
        XCTAssertEqual(user["content"]?.arrayValue?.last?["text"]?.stringValue?.hasPrefix("<analyst_notes>"), true)
    }

    func testBriefsAreInstructionsBlocksOnSonnet5() async throws {
        let transport = deepTransport { call, _ in call == .merger ? deepText("Answer.") : deepText("Notes.") }
        let claude = claudeConfiguration(model: "claude-sonnet-5")
        _ = try await collectEvents(makeDeliberation(deepConfiguration(.parallel), claude: claude, transport: transport).streamEvents(system: "S", turns: question))

        XCTAssertEqual(transport.bodies.count, 4)
        var efforts: [DeepCall: String] = [:]
        for body in transport.bodies {
            let messages = body["messages"]?.arrayValue ?? []
            XCTAssertFalse(messages.contains { $0["role"]?.stringValue == "system" })
            let text = try XCTUnwrap(messages.last?["content"]?.arrayValue?.last?["text"]?.stringValue)
            XCTAssertTrue(text.hasPrefix("<instructions>\n") && text.hasSuffix("\n</instructions>"))
            efforts[DeepRequests.kind(of: body)] = body["output_config"]?["effort"]?.stringValue
        }
        XCTAssertEqual(efforts, [.worker("researcher"): "low", .worker("reasoner"): "high", .worker("critic"): "medium", .merger: "low"])

        let blocks = try XCTUnwrap(transport.bodies.last?["messages"]?.arrayValue?.last?["content"]?.arrayValue)
        XCTAssertEqual(blocks.count, 3)
        XCTAssertEqual(blocks[0]["text"], "Should I move to Singapore for the job?")
        XCTAssertEqual(blocks[1]["text"]?.stringValue?.hasPrefix("<analyst_notes>\n<research>\nNotes.\n</research>"), true)
        XCTAssertEqual(blocks[2]["text"]?.stringValue, "<instructions>\n\(DeliberationConfiguration.defaultMergerInstruction)\n</instructions>")
    }

    func testWorkerModelAppliesOnlyToTheWorkers() async throws {
        var configuration = deepConfiguration(.parallel)
        configuration.workerModel = "claude-sonnet-5-5"
        let transport = deepTransport { call, _ in call == .merger ? deepText("Answer.") : deepText("Notes.") }
        _ = try await collectEvents(makeDeliberation(configuration, transport: transport).streamEvents(system: "S", turns: question))
        XCTAssertEqual(workerBodies(transport).compactMap { $0["model"]?.stringValue }, Array(repeating: "claude-sonnet-5-5", count: 3))
        XCTAssertEqual(transport.bodies.last?["model"], "claude-opus-5-5")

        configuration.workerModel = "  "
        let blank = deepTransport { call, _ in call == .merger ? deepText("Answer.") : deepText("Notes.") }
        _ = try await collectEvents(makeDeliberation(configuration, transport: blank).streamEvents(system: "S", turns: question))
        XCTAssertEqual(Set(blank.bodies.compactMap { $0["model"]?.stringValue }), ["claude-opus-5-5"], "a blank model is the conversation's")
    }

    func testARefusedOrFailedWorkerIsDroppedAndATruncatedOneKeepsItsText() async throws {
        let transport = deepTransport { call, _ in
            switch call {
            case .worker("researcher"): return deepText("Rents rose 4%", stop: "max_tokens")
            case .worker("reasoner"): return deepText("I won't", stop: "refusal", details: ["category": "cyber"])
            case .worker("critic"): return .failure(HTTPStatusError(statusCode: 400, body: "{}"))
            case .merger: return deepText("Answer.")
            default: return .failure(URLError(.badURL))
            }
        }
        let events = try await collectEvents(makeDeliberation(deepConfiguration(.parallel), transport: transport).streamEvents(system: "S", turns: question))

        XCTAssertEqual(events, opening + answer("Answer."))
        let merger = try XCTUnwrap(transport.bodies.last)
        XCTAssertEqual(DeepRequests.kind(of: merger), .merger)
        XCTAssertEqual(DeepRequests.analystNotes(of: merger), "<analyst_notes>\n<research>\nRents rose 4%\n</research>\n</analyst_notes>")
    }

    func testWithoutNotesAPlainCallAnswers() async throws {
        let transport = deepTransport { call, _ in
            switch call {
            case .worker("researcher"): return deepText("No.", stop: "refusal")
            case .worker("reasoner"): return .failure(HTTPStatusError(statusCode: 400, body: "{}"))
            case .worker("critic"): return deepText("   ")
            case .plain: return deepText("Here's a quick answer.")
            default: return .failure(URLError(.badURL))
            }
        }
        let deliberation = makeDeliberation(deepConfiguration(.parallel, voice: true), transport: transport)

        let events = try await collectEvents(deliberation.streamEvents(system: "S", turns: question))

        XCTAssertEqual(events, opening + answer("Here's a quick answer."))
        XCTAssertEqual(transport.bodies.count, 4)
        let plain = try XCTUnwrap(transport.bodies.last)
        XCTAssertEqual(DeepRequests.kind(of: plain), .plain)
        XCTAssertNil(DeepRequests.analystNotes(of: plain))
        XCTAssertNil(DeepRequests.effortMessage(of: plain))
        XCTAssertEqual(plain["max_tokens"], 16_000, "the conversation's own limit, not the merger's")
        XCTAssertEqual(plain["messages"]?.arrayValue?.count, 1)
    }

    func testNoWorkersMeansAPlainAnswer() async throws {
        var configuration = deepConfiguration(.parallel)
        configuration.workers = []
        let transport = deepTransport { call, _ in call == .plain ? deepText("Answer.") : .failure(URLError(.badURL)) }
        let events = try await collectEvents(makeDeliberation(configuration, transport: transport).streamEvents(system: "S", turns: question))
        XCTAssertEqual(events, opening + answer("Answer."))
        XCTAssertEqual(transport.requests.count, 1)
    }

    func testSideEffectToolsNeverRunInWorkersButDoInTheMerger() async throws {
        let reminder = FakeTool(name: "create_reminder", effect: .sideEffect, output: .ok(["id": "r1"], summary: "Reminder: Call the agent"))
        let calendar = FakeTool(name: "list_events", output: .ok(["events": 0]))
        let transport = deepTransport { call, body in
            let followUp = DeepRequests.answersToolResults(body)
            switch call {
            case .worker("researcher"):
                return followUp ? deepText("No events this week.") : deepToolCalls([("toolu_r1", "create_reminder"), ("toolu_r2", "list_events")])
            case .worker("reasoner"):
                return followUp ? deepText("Decide after a visit.") : deepToolCalls([("toolu_q1", "list_events")])
            case .worker("critic"):
                return deepText("Ask about the lease.")
            case .merger:
                return followUp ? deepText("I've added a reminder.") : deepToolCalls([("toolu_m1", "create_reminder")])
            default:
                return .failure(URLError(.badURL))
            }
        }
        let deliberation = makeDeliberation(deepConfiguration(.parallel), transport: transport, tools: ToolRegistry([reminder, calendar]))

        let events = try await collectEvents(deliberation.streamEvents(system: "S", turns: question))

        XCTAssertEqual(reminder.runCount, 1, "only the merger acts")
        XCTAssertEqual(calendar.runCount, 1, "only the researcher runs read-only tools")
        XCTAssertEqual(toolRounds(events), [ToolRound(calls: [
            ToolCallRecord(id: "toolu_m1", name: "create_reminder", input: .object([:]), result: #"{"id":"r1"}"#, summary: "Reminder: Call the agent"),
        ])], "only the merger's round reaches the reply")
        XCTAssertEqual(events.last, .reply(.finished(.completed)))
        XCTAssertEqual(transport.requests.count, 7)

        let blocked: JSONValue = .string(#"{"error":"Not available here."}"#)
        let researcher = try XCTUnwrap(transport.bodies.first { DeepRequests.kind(of: $0) == .worker("researcher") && DeepRequests.answersToolResults($0) })
        let researched = DeepRequests.toolResults(researcher)
        XCTAssertEqual(researched.map { $0["tool_use_id"] }, ["toolu_r1", "toolu_r2"])
        XCTAssertEqual(researched[0]["content"], blocked)
        XCTAssertEqual(researched[0]["is_error"], true)
        XCTAssertEqual(researched[1]["content"], #"{"events":0}"#)
        XCTAssertNil(researched[1]["is_error"])

        let reasoner = try XCTUnwrap(transport.bodies.first { DeepRequests.kind(of: $0) == .worker("reasoner") && DeepRequests.answersToolResults($0) })
        let reasoned = DeepRequests.toolResults(reasoner)
        XCTAssertEqual(reasoned.map { $0["content"] }, [blocked], "a worker without tools runs none")
        XCTAssertEqual(reasoned.first?["is_error"], true)

        let merger = try XCTUnwrap(transport.bodies.last)
        XCTAssertEqual(DeepRequests.kind(of: merger), .merger)
        XCTAssertEqual(DeepRequests.toolResults(merger).first?["content"], #"{"id":"r1"}"#)
    }

    func testOnlyTheResearchersActivityIsShownAtMostEveryThreeSeconds() async throws {
        let transport = deepTransport { call, _ in
            switch call {
            case .worker("researcher"):
                return ClaudeSSE.response(
                    [ClaudeSSE.messageStart()],
                    ClaudeSSE.webSearch(index: 0, id: "srvtoolu_1", query: "Singapore rent 2026"),
                    ClaudeSSE.webSearch(index: 2, id: "srvtoolu_2", query: "Singapore tax"),
                    ClaudeSSE.text(index: 4, "Rents rose 4%."),
                    [ClaudeSSE.stop("end_turn")]
                )
            case .worker("critic"):
                return ClaudeSSE.response(
                    [ClaudeSSE.messageStart()],
                    ClaudeSSE.webSearch(index: 0, id: "srvtoolu_3", query: "critic search"),
                    ClaudeSSE.text(index: 2, "Ask first."),
                    [ClaudeSSE.stop("end_turn")]
                )
            case .worker: return deepText("Notes.")
            case .merger: return deepText("Answer.")
            default: return .failure(URLError(.badURL))
            }
        }
        let claude = claudeConfiguration(webSearch: true)

        // A clock that stands still: the second search falls inside the 3 s window.
        let still = try await collectEvents(
            makeDeliberation(deepConfiguration(.parallel), claude: claude, transport: transport, uptime: { 100 }).streamEvents(system: "S", turns: question)
        )
        XCTAssertEqual(still, opening + [
            .reply(.activity("Searching the web")),
            .reply(.activity("Searching the web for “Singapore rent 2026”")),
            thinking,
        ] + answer("Answer."), "no cues and no other worker's lines; the query extends its line at once")

        // A clock that moves 5 s per reading: each search shows.
        let ticking = SteppingUptime(step: 5)
        let moving = try await collectEvents(
            makeDeliberation(deepConfiguration(.parallel), claude: claude, transport: transport, uptime: { ticking.next() }).streamEvents(system: "S", turns: question)
        )
        XCTAssertEqual(moving, opening + [
            .reply(.activity("Searching the web")),
            .reply(.activity("Searching the web for “Singapore rent 2026”")),
            .reply(.activity("Searching the web")),
            .reply(.activity("Searching the web for “Singapore tax”")),
            thinking,
        ] + answer("Answer."))
    }

    // MARK: - Deadline

    func testAHangingWorkerIsCutAtTheDeadlineAndKeepsWhatItWrote() async throws {
        let clock = ManualClock()
        let ends = WorkerEndLog()
        // The critic shows its activity here, so the test can see it has written before it hangs.
        var critic = WorkerBrief.critic
        critic.forwardsActivity = true
        var configuration = deepConfiguration(.parallel, voice: true)
        configuration.workers = [.researcher, .reasoner, critic]
        let searchStart = Array(ClaudeSSE.webSearch(index: 1, id: "srvtoolu_c", query: "visa").prefix(1))
        let transport = deepTransport { call, _ in
            switch call {
            case .worker("researcher"): return deepText("Rents rose 4%.")
            case .worker("reasoner"): return deepText("Move for a 20% raise.")
            case .worker("critic"):
                return .hang(MockTransport.sse([ClaudeSSE.messageStart()] + ClaudeSSE.text(index: 0, "Ask about family.") + searchStart))
            case .merger: return deepText("Probably move.")
            default: return .failure(URLError(.badURL))
            }
        }
        let recorder = DeepEventRecorder()
        let deliberation = makeDeliberation(configuration, transport: transport, clock: clock, workerEnds: ends)
        let consumer = recorder.consume(deliberation.streamEvents(system: "S", turns: question))

        let working = await waitUntil {
            Set(ends.ids) == ["researcher", "reasoner"] && clock.sleeperCount == 2
                && recorder.events.contains(.reply(.activity("Searching the web")))
        }
        XCTAssertTrue(working)
        XCTAssertEqual(transport.requests.count, 3)

        clock.advance(to: 12.5)
        let cued = await waitUntil { recorder.events.contains(.cue(.stillThinking)) }
        XCTAssertTrue(cued)
        try? await Task.sleep(nanoseconds: 20_000_000)
        XCTAssertEqual(transport.requests.count, 3, "the merger waits for the deadline")
        XCTAssertEqual(transport.cancellationCount, 0)

        clock.advance(to: 25)
        let error = await consumer.value
        XCTAssertNil(error)
        XCTAssertEqual(recorder.events, opening + [.reply(.activity("Searching the web")), .cue(.stillThinking), thinking] + answer("Probably move."))
        let cancelled = await waitUntil { transport.cancellationCount == 1 }
        XCTAssertTrue(cancelled, "the hanging worker's request is cancelled")
        XCTAssertEqual(ends.ids.last, "critic")
        let merger = try XCTUnwrap(transport.bodies.last)
        XCTAssertEqual(DeepRequests.kind(of: merger), .merger)
        XCTAssertEqual(
            DeepRequests.analystNotes(of: merger),
            "<analyst_notes>\n<research>\nRents rose 4%.\n</research>\n<reasoning>\nMove for a 20% raise.\n</reasoning>\n<critique>\nAsk about family.\n</critique>\n</analyst_notes>"
        )
    }

    func testASilentWorkerIsCutAtTheDeadlineAndTheMergerStillRuns() async throws {
        let clock = ManualClock()
        let ends = WorkerEndLog()
        let transport = deepTransport { call, _ in
            switch call {
            case .worker("critic"): return .hang([])
            case .merger: return deepText("Answer.")
            default: return deepText("Notes.")
            }
        }
        let recorder = DeepEventRecorder()
        let consumer = recorder.consume(
            makeDeliberation(deepConfiguration(.parallel, voice: true), transport: transport, clock: clock, workerEnds: ends)
                .streamEvents(system: "S", turns: question)
        )
        let waiting = await waitUntil { ends.ids.count == 2 && clock.sleeperCount == 2 }
        XCTAssertTrue(waiting)

        clock.advance(to: 12.5)
        let cued = await waitUntil { recorder.events.count == 3 }
        XCTAssertTrue(cued)
        clock.advance(to: 25)
        let error = await consumer.value
        XCTAssertNil(error)
        XCTAssertEqual(recorder.events, opening + [.cue(.stillThinking)] + answer("Answer."))
        let cancelled = await waitUntil { transport.cancellationCount == 1 }
        XCTAssertTrue(cancelled)
        let notes = try XCTUnwrap(DeepRequests.analystNotes(of: transport.bodies.last ?? .null))
        XCTAssertEqual(notes, "<analyst_notes>\n<research>\nNotes.\n</research>\n<reasoning>\nNotes.\n</reasoning>\n</analyst_notes>")
    }

    func testWhenNoWorkerReachesTheServerByTheDeadlineTheReplyTimesOut() async {
        let clock = ManualClock()
        let transport = HangingTransport()
        let recorder = DeepEventRecorder()
        let consumer = recorder.consume(
            makeDeliberation(deepConfiguration(.parallel, voice: true), transport: transport, clock: clock).streamEvents(system: "S", turns: question)
        )
        let waiting = await waitUntil { transport.requests.count == 3 && clock.sleeperCount == 2 }
        XCTAssertTrue(waiting)

        clock.advance(to: 12.5)
        let cued = await waitUntil { recorder.events.count == 3 }
        XCTAssertTrue(cued)
        clock.advance(to: 25)
        let error = await consumer.value

        XCTAssertEqual((error as? URLError)?.code, .timedOut)
        XCTAssertEqual(error?.isConnectivityFailure, true, "so the orchestrator can fall back to the on-device model")
        XCTAssertEqual(recorder.events, opening + [.cue(.stillThinking)])
        let cancelled = await waitUntil { transport.terminationCount == 3 }
        XCTAssertTrue(cancelled)
        XCTAssertEqual(transport.requests.count, 3, "no plain call into a dead network")
    }

    func testAWorkerThatReachedTheServerMakesTheDeadlineGiveAPlainAnswer() async throws {
        let clock = ManualClock()
        let ends = WorkerEndLog()
        let transport = deepTransport { call, _ in
            switch call {
            case .worker("researcher"): return .failure(HTTPStatusError(statusCode: 400, body: "{}"))
            case .plain: return deepText("Quick answer.")
            default: return .hang([])
            }
        }
        let recorder = DeepEventRecorder()
        let consumer = recorder.consume(
            makeDeliberation(deepConfiguration(.parallel, voice: true), transport: transport, clock: clock, workerEnds: ends)
                .streamEvents(system: "S", turns: question)
        )
        let waiting = await waitUntil { ends.ids == ["researcher"] && transport.requests.count == 3 && clock.sleeperCount == 2 }
        XCTAssertTrue(waiting)

        clock.advance(to: 12.5)
        let cued = await waitUntil { recorder.events.count == 3 }
        XCTAssertTrue(cued)
        clock.advance(to: 25)
        let error = await consumer.value
        XCTAssertNil(error)
        XCTAssertEqual(recorder.events, opening + [.cue(.stillThinking)] + answer("Quick answer."))
        XCTAssertEqual(transport.bodies.last.map { DeepRequests.kind(of: $0) }, .plain)
    }

    // MARK: - Cancellation

    func testCancellingTheReplyCancelsEveryWorkerRequest() async {
        let transport = HangingTransport()
        let recorder = DeepEventRecorder()
        let consumer = recorder.consume(makeDeliberation(deepConfiguration(.parallel), transport: transport).streamEvents(system: "S", turns: question))
        let started = await waitUntil { transport.requests.count == 3 }
        XCTAssertTrue(started)
        XCTAssertEqual(transport.terminationCount, 0)

        consumer.cancel()
        _ = await consumer.value
        let ended = await waitUntil { transport.terminationCount == 3 }
        XCTAssertTrue(ended, "every worker's request is cancelled")
        try? await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertEqual(transport.requests.count, 3, "no merger or plain call follows a cancelled reply")
    }

    func testCancellingTheReplyCancelsTheMerger() async {
        let transport = deepTransport { call, _ in
            call == .merger ? .hang(MockTransport.sse([ClaudeSSE.messageStart()])) : deepText("Notes.")
        }
        let recorder = DeepEventRecorder()
        let consumer = recorder.consume(makeDeliberation(deepConfiguration(.parallel), transport: transport).streamEvents(system: "S", turns: question))
        let merging = await waitUntil { recorder.events.contains(.progress(.responseStarted)) }
        XCTAssertTrue(merging)
        XCTAssertEqual(transport.requests.count, 4)

        consumer.cancel()
        _ = await consumer.value
        let ended = await waitUntil { transport.cancellationCount == 1 }
        XCTAssertTrue(ended, "the merger's request is cancelled")
    }
}
