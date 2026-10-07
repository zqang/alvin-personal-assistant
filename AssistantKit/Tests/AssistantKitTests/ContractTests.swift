import XCTest
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
@testable import AssistantKit

final class ContractEventTests: XCTestCase {
    private let hello = [ChatTurn(role: .user, text: "Hi")]
    private let mixed: [AssistantEvent] = [
        .routed(RouteDecision(engine: .cloud, reason: .cloudDefault, fallback: .local)),
        .progress(.responseStarted),
        .cue(.checking),
        .reply(.activity("Checking your calendar")),
        .toolRound(ToolRound(calls: [ToolCallRecord(id: "t1", name: "list_events", input: [:], result: "{}")])),
        .reply(.activity(nil)),
        .progress(.firstToken),
        .reply(.text("You're ")),
        .reply(.text("free.")),
        .reply(.finished(.completed)),
    ]

    func testWrapPreservesOrder() async throws {
        let replies: [ReplyEvent] = [.activity("Searching the web"), .activity(nil), .text("A"), .text("B"), .finished(.truncated)]
        let source = AsyncThrowingStream<ReplyEvent, Error> { continuation in
            for reply in replies { continuation.yield(reply) }
            continuation.finish()
        }
        let events = try await collectEvents(AssistantEvents.wrap(source))
        XCTAssertEqual(events, replies.map { .reply($0) })
    }

    func testRepliesKeepOnlyReplyEventsInOrder() async throws {
        let replies = try await collect(AssistantEvents.replies(ScriptedProvider(mixed).streamEvents(system: "S", turns: hello)))
        XCTAssertEqual(replies, [.activity("Checking your calendar"), .activity(nil), .text("You're "), .text("free."), .finished(.completed)])
    }

    func testDefaultStreamReplyDropsNonReplyEvents() async throws {
        let provider = ScriptedProvider(mixed)
        let replies = try await collect(provider.streamReply(system: "S", turns: hello))
        XCTAssertEqual(replies.count, 5)
        XCTAssertEqual(replies.first, .activity("Checking your calendar"))
        XCTAssertEqual(provider.callCount, 1)
        XCTAssertEqual(provider.receivedSystems, ["S"])
        XCTAssertEqual(provider.receivedTurns, [hello])
    }

    func testAdaptersRethrowAfterTheEventsBeforeTheError() async {
        let provider = ScriptedProvider(mixed, failAfter: 4, error: URLError(.networkConnectionLost))
        var received: [ReplyEvent] = []
        do {
            for try await reply in provider.streamReply(system: "S", turns: hello) { received.append(reply) }
            XCTFail("Expected an error")
        } catch {
            XCTAssertEqual((error as? URLError)?.code, .networkConnectionLost)
        }
        XCTAssertEqual(received, [.activity("Checking your calendar")])

        let failing = AsyncThrowingStream<ReplyEvent, Error> { continuation in
            continuation.yield(.text("Half"))
            continuation.finish(throwing: AssistantError.stream(type: "overloaded_error", message: "Overloaded"))
        }
        var wrapped: [AssistantEvent] = []
        do {
            for try await event in AssistantEvents.wrap(failing) { wrapped.append(event) }
            XCTFail("Expected an error")
        } catch {
            XCTAssertEqual(error as? AssistantError, .stream(type: "overloaded_error", message: "Overloaded"))
        }
        XCTAssertEqual(wrapped, [.reply(.text("Half"))])
    }

    func testLegacyAdapterWrapsEveryReplyEvent() async throws {
        let transport = MockTransport([
            .lines(MockTransport.sse([
                #"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
                #"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello!"}}"#,
                #"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
            ])),
            .lines(MockTransport.sse([
                #"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"Again."}}"#,
                #"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
            ])),
        ])
        let adapter = LegacyProviderAdapter(ClaudeProvider(configuration: ClaudeConfiguration(apiKey: "k"), transport: transport))
        let events = try await collectEvents(adapter.streamEvents(system: "S", turns: hello))
        XCTAssertEqual(events, [.reply(.text("Hello!")), .reply(.finished(.completed))])
        let replies = try await collect(adapter.streamReply(system: "S", turns: hello))
        XCTAssertEqual(replies, [.text("Again."), .finished(.completed)])
        XCTAssertEqual(transport.requests.count, 2)
    }

    func testCancellingTheLegacyAdapterCancelsTheRequest() async throws {
        let transport = HangingTransport()
        let adapter = LegacyProviderAdapter(ClaudeProvider(configuration: ClaudeConfiguration(apiKey: "k"), transport: transport))
        let consumer = Task { try await collectEvents(adapter.streamEvents(system: "S", turns: hello)) }
        let requested = await waitUntil { transport.requests.count == 1 }
        XCTAssertTrue(requested)
        XCTAssertEqual(transport.terminationCount, 0)
        consumer.cancel()
        let terminated = await waitUntil { transport.terminationCount == 1 }
        XCTAssertTrue(terminated, "cancelling the outer stream must end the hanging request")
        _ = try? await consumer.value
    }

    func testCancellingConvertedStreamsCancelsTheProvider() async throws {
        let first = ScriptedProvider([.cue(.working), .reply(.text("Hi"))], hangs: true)
        let replies = AssistantEvents.replies(first.streamEvents(system: "S", turns: hello))
        let consumer = Task { () -> [ReplyEvent] in
            var received: [ReplyEvent] = []
            for try await reply in replies { received.append(reply) }
            return received
        }
        try await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertEqual(first.terminationCount, 0)
        consumer.cancel()
        let ended = await waitUntil { first.terminationCount == 1 }
        XCTAssertTrue(ended)
        let received = try await consumer.value
        XCTAssertEqual(received, [.text("Hi")])

        let second = ScriptedProvider([], hangs: true)
        let wrapped = AssistantEvents.wrap(second.streamReply(system: "S", turns: hello))
        let wrapper = Task { try await collectEvents(wrapped) }
        try await Task.sleep(nanoseconds: 20_000_000)
        wrapper.cancel()
        let wrappedEnded = await waitUntil { second.terminationCount == 1 }
        XCTAssertTrue(wrappedEnded, "cancellation reaches the provider through two relays")
        _ = try? await wrapper.value
    }

    func testReplyHandoffCarriesItsReason() {
        let handoff = ReplyHandoff(reason: "needs the news")
        XCTAssertEqual(handoff, ReplyHandoff(reason: "needs the news"))
        XCTAssertNotEqual(handoff, ReplyHandoff(reason: "other"))
        XCTAssertNotNil(handoff.errorDescription)
    }
}

final class ContractToolTypesTests: XCTestCase {
    func testToolRoundCodableRoundTrip() throws {
        let round = ToolRound(calls: [
            ToolCallRecord(
                id: "toolu_01",
                name: "create_reminder",
                input: ["title": "Call mum", "due": "2026-10-07T17:00", "nested": ["count": 3, "ratio": 2.5, "on": true, "none": .null]],
                result: #"{"id":"r1","ok":true}"#,
                summary: "Reminder: Call mum · today 17:00"
            ),
            ToolCallRecord(id: "call_2", name: "set_timer", input: ["seconds": 90], result: #"{"error":"Denied"}"#, isError: true),
        ])
        let data = try JSONEncoder().encode(round)
        let decoded = try JSONDecoder().decode(ToolRound.self, from: data)
        XCTAssertEqual(decoded, round)
        XCTAssertEqual(try JSONDecoder().decode([ToolRound].self, from: JSONEncoder().encode([round, round])), [round, round])

        let encoded = try JSONValue.parse(data)
        XCTAssertNil(encoded["calls"]?.arrayValue?[1]["summary"], "a missing summary isn't written")
    }

    func testToolCallRecordDecodingIsTolerant() throws {
        let json = #"{"calls":[{"id":"a","name":"get_current_time"}]}"#
        let round = try JSONDecoder().decode(ToolRound.self, from: Data(json.utf8))
        XCTAssertEqual(round.calls, [ToolCallRecord(id: "a", name: "get_current_time", input: [:], result: "")])
        XCTAssertEqual(try JSONDecoder().decode(ToolRound.self, from: Data("{}".utf8)), ToolRound(calls: []))
        XCTAssertThrowsError(try JSONDecoder().decode(ToolRound.self, from: Data(#"{"calls":[{"name":"x"}]}"#.utf8)))
    }

    func testSummaryLine() {
        func record(_ summary: String?) -> ToolCallRecord {
            ToolCallRecord(id: "i", name: "n", input: [:], result: "{}", summary: summary)
        }
        XCTAssertNil(ToolRound(calls: []).summaryLine)
        XCTAssertNil(ToolRound(calls: [record(nil), record("  ")]).summaryLine)
        XCTAssertEqual(ToolRound(calls: [record("Timer: 5 minutes")]).summaryLine, "Timer: 5 minutes")
        XCTAssertEqual(ToolRound(calls: [record("A"), record(nil), record(" B ")]).summaryLine, "A; B")
    }

    func testToolOutputHelpers() {
        let error = ToolOutput.error("Unknown tool")
        XCTAssertEqual(error.content, ["error": "Unknown tool"])
        XCTAssertTrue(error.isError)
        XCTAssertNil(error.summary)
        XCTAssertEqual(error.serializedContent, #"{"error":"Unknown tool"}"#)

        let ok = ToolOutput.ok(["b": 1, "a": "x/y"], summary: "Done")
        XCTAssertFalse(ok.isError)
        XCTAssertEqual(ok.summary, "Done")
        XCTAssertEqual(ok.serializedContent, #"{"a":"x/y","b":1}"#, "sorted keys, slashes unescaped")

        let call = PendingToolCall(id: "c1", name: "list_events", input: ["start": "2026-10-07T09:00"])
        XCTAssertEqual(
            ToolCallRecord(call: call, output: ok),
            ToolCallRecord(id: "c1", name: "list_events", input: ["start": "2026-10-07T09:00"], result: #"{"a":"x/y","b":1}"#, summary: "Done")
        )
        XCTAssertEqual(ToolCallRecord(call: PendingToolCall(id: "c2", name: "x", rawInput: "{oops"), output: error).input, [:])

        XCTAssertEqual(ToolPresentation.generic, ToolPresentation(activity: "Working on it", cue: .working))
        XCTAssertNil(ToolPresentation(activity: "Adding a reminder").cue)
    }

    func testToolContextDefaults() {
        let before = Date()
        let context = ToolContext()
        XCTAssertGreaterThanOrEqual(context.now, before)
        XCTAssertEqual(context.timeZone, TimeZone.current)
        XCTAssertNil(context.commitGate)
        let gate = CommitGate()
        XCTAssertTrue(ToolContext(commitGate: gate).commitGate === gate)
    }

    func testNoToolExecutorRejectsEveryCallInOrder() async {
        let executor = NoToolExecutor()
        XCTAssertTrue(executor.definitions.isEmpty)
        XCTAssertEqual(executor.presentation(for: "anything"), .generic)
        let round = await executor.run([
            PendingToolCall(id: "a", name: "create_reminder", input: ["title": "x"]),
            PendingToolCall(id: "b", name: "set_timer", rawInput: "{bad"),
        ], context: ToolContext())
        XCTAssertEqual(round.calls.map(\.id), ["a", "b"])
        XCTAssertEqual(round.calls.map(\.result), [#"{"error":"Unknown tool"}"#, #"{"error":"Unknown tool"}"#])
        XCTAssertEqual(round.calls.map(\.isError), [true, true])
        XCTAssertEqual(round.calls.map(\.input), [["title": "x"], [:]])
        XCTAssertNil(round.summaryLine)
    }

    func testRouteDecisionDefaults() {
        let decision = RouteDecision(engine: .local, reason: .offline)
        XCTAssertEqual(decision.mode, .standard)
        XCTAssertNil(decision.fallback)
        XCTAssertEqual(RouteReason(rawValue: "networkFallback"), .networkFallback)
        XCTAssertEqual(ReplyEngine.cloud.rawValue, "cloud")
        XCTAssertEqual(ReplyMode.deep.rawValue, "deep")
    }
}

final class ContractCommitGateTests: XCTestCase {
    func testOpenBeforeWaitReturnsAtOnce() async throws {
        let gate = CommitGate()
        XCTAssertFalse(gate.isOpen)
        gate.open()
        XCTAssertTrue(gate.isOpen)
        XCTAssertFalse(gate.isCancelled)
        try await gate.wait()
        try await gate.wait()
    }

    func testOpenAfterWaitReleasesEveryWaiter() async throws {
        let gate = CommitGate()
        let released = Counter()
        let waiters = (0..<3).map { _ in
            Task {
                try await gate.wait()
                released.increment()
            }
        }
        try await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertEqual(released.value, 0, "nobody passes before open")
        gate.open()
        for waiter in waiters { try await waiter.value }
        XCTAssertEqual(released.value, 3)
    }

    func testCancelThrowsForCurrentAndLaterWaiters() async throws {
        let gate = CommitGate()
        let waiter = Task { try await gate.wait() }
        try await Task.sleep(nanoseconds: 20_000_000)
        gate.cancel()
        await assertCancelled(waiter)
        XCTAssertTrue(gate.isCancelled)
        do {
            try await gate.wait()
            XCTFail("Expected CancellationError")
        } catch {
            XCTAssertTrue(error is CancellationError)
        }
    }

    func testTheFirstOutcomeWins() async throws {
        let opened = CommitGate()
        opened.open()
        opened.cancel()
        XCTAssertTrue(opened.isOpen)
        XCTAssertFalse(opened.isCancelled)
        try await opened.wait()

        let cancelled = CommitGate()
        cancelled.cancel()
        cancelled.open()
        XCTAssertFalse(cancelled.isOpen)
        do {
            try await cancelled.wait()
            XCTFail("Expected CancellationError")
        } catch {
            XCTAssertTrue(error is CancellationError)
        }
    }

    func testCancellingTheWaitingTaskLeavesTheGatePending() async throws {
        let gate = CommitGate()
        let waiter = Task { try await gate.wait() }
        try await Task.sleep(nanoseconds: 20_000_000)
        waiter.cancel()
        await assertCancelled(waiter)
        XCTAssertFalse(gate.isOpen)
        XCTAssertFalse(gate.isCancelled)

        let alreadyCancelled = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            try await gate.wait()
        }
        await assertCancelled(alreadyCancelled)

        gate.open()
        try await gate.wait()
    }

    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0

        var value: Int {
            lock.lock()
            defer { lock.unlock() }
            return count
        }

        func increment() {
            lock.lock()
            count += 1
            lock.unlock()
        }
    }

    private func assertCancelled(_ task: Task<Void, Error>, file: StaticString = #filePath, line: UInt = #line) async {
        do {
            try await task.value
            XCTFail("Expected CancellationError", file: file, line: line)
        } catch {
            XCTAssertTrue(error is CancellationError, "got \(error)", file: file, line: line)
        }
    }
}

final class ContractSpokenCueTests: XCTestCase {
    func testPhrases() {
        let english: [SpokenCue: String] = [
            .lookingUp: "Let me look that up.",
            .checking: "Let me check.",
            .working: "One moment.",
            .deepThinking: "Let me think that through properly.",
            .stillThinking: "Still thinking, almost there.",
            .handingOff: "Let me check online.",
            .loadingModel: "One moment, getting ready.",
        ]
        let chinese: [SpokenCue: String] = [
            .lookingUp: "我查一下。",
            .checking: "我看看。",
            .working: "稍等。",
            .deepThinking: "让我好好想想。",
            .stillThinking: "还在想，马上就好。",
            .handingOff: "我上网查一下。",
            .loadingModel: "稍等，正在准备。",
        ]
        XCTAssertEqual(english.count, 7)
        for (cue, phrase) in english {
            XCTAssertEqual(SpokenCue.phrase(cue, localeIdentifier: "en-US"), phrase)
            XCTAssertEqual(SpokenCue.phrase(cue, localeIdentifier: "fr-FR"), phrase, "unsupported languages use English")
            XCTAssertEqual(SpokenCue.phrase(cue, localeIdentifier: ""), phrase)
        }
        for (cue, phrase) in chinese {
            XCTAssertEqual(SpokenCue.phrase(cue, localeIdentifier: "zh-CN"), phrase)
            XCTAssertEqual(SpokenCue.phrase(cue, localeIdentifier: "zh-Hant-TW"), phrase)
            XCTAssertEqual(SpokenCue.phrase(cue, localeIdentifier: "zh_HK"), phrase)
        }
    }

    func testRawValuesAreStable() throws {
        XCTAssertEqual(SpokenCue.handingOff.rawValue, "handingOff")
        let data = try JSONEncoder().encode([SpokenCue.deepThinking])
        XCTAssertEqual(String(decoding: data, as: UTF8.self), #"["deepThinking"]"#)
    }
}

final class ContractSameTurnTests: XCTestCase {
    private let round = ToolRound(calls: [ToolCallRecord(id: "t1", name: "set_timer", input: ["seconds": 60], result: #"{"id":"x"}"#)])

    func testSameTurnComparesToolRounds() {
        let plain = ChatTurn(role: .assistant, text: "Done.")
        let withRound = ChatTurn(role: .assistant, text: "Done.", toolRounds: [round])
        XCTAssertTrue(LocalSessionPlan.sameTurn(plain, ChatTurn(role: .assistant, text: " Done.\n")))
        XCTAssertTrue(LocalSessionPlan.sameTurn(withRound, ChatTurn(role: .assistant, text: "Done.", toolRounds: [round])))
        XCTAssertFalse(LocalSessionPlan.sameTurn(plain, withRound))

        var other = round
        other.calls[0].result = #"{"id":"y"}"#
        XCTAssertFalse(LocalSessionPlan.sameTurn(withRound, ChatTurn(role: .assistant, text: "Done.", toolRounds: [other])))
        XCTAssertFalse(LocalSessionPlan.sameTurn(withRound, ChatTurn(role: .assistant, text: "Done.", toolRounds: [round, round])))
    }

    func testChangedRoundsRebuildTheSession() throws {
        let user = ChatTurn(role: .user, text: "Set a timer", context: "<context>c</context>")
        let reply = ChatTurn(role: .assistant, text: "Done.", toolRounds: [round])
        let next = ChatTurn(role: .user, text: "Thanks")
        let appended = try XCTUnwrap(LocalSessionPlan.make(cachedSystem: "S", cachedTurns: [user, reply], system: "S", turns: [user, reply, next]))
        XCTAssertEqual(appended.action, .append)

        let edited = ChatTurn(role: .assistant, text: "Done.")
        let rebuilt = try XCTUnwrap(LocalSessionPlan.make(cachedSystem: "S", cachedTurns: [user, reply], system: "S", turns: [user, edited, next]))
        XCTAssertEqual(rebuilt.action, .rebuild(history: [user, edited]))
    }

    func testChatTurnDefaultsToNoRounds() {
        XCTAssertEqual(ChatTurn(role: .user, text: "x").toolRounds, [])
        XCTAssertNotEqual(ChatTurn(role: .assistant, text: "x"), ChatTurn(role: .assistant, text: "x", toolRounds: [round]))
    }
}

/// Checks the shared test doubles behave as other test files rely on.
final class ContractTestSupportTests: XCTestCase {
    private final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var recorded: [String] = []

        var values: [String] {
            lock.lock()
            defer { lock.unlock() }
            return recorded
        }

        func append(_ value: String) {
            lock.lock()
            recorded.append(value)
            lock.unlock()
        }
    }

    func testToolUseEventsStreamTheInputInTwoParts() throws {
        let json = #"{"title":"Call \"mum\"","due":"2026-10-07T17:00"}"#
        let events = try MockTransport.toolUse(index: 2, id: "toolu_9", name: "create_reminder", json: json).map { try JSONValue.parse($0) }
        XCTAssertEqual(events.map { $0["type"]?.stringValue }, ["content_block_start", "content_block_delta", "content_block_delta", "content_block_stop"])
        XCTAssertEqual(events.map { $0["index"]?.intValue }, [2, 2, 2, 2])
        XCTAssertEqual(events[0]["content_block"], ["type": "tool_use", "id": "toolu_9", "name": "create_reminder", "input": .object([:])])
        let parts = events[1...2].compactMap { $0["delta"]?["partial_json"]?.stringValue }
        XCTAssertEqual(parts.count, 2)
        XCTAssertFalse(parts[0].isEmpty)
        XCTAssertEqual(parts.joined(), json)
        XCTAssertEqual(MockTransport.sse(MockTransport.toolUse(index: 0, id: "a", name: "b", json: "{}")).count, 12)
    }

    func testScriptedTransportAnswersPerRequest() async throws {
        let transport = ScriptedTransport { _, body in
            body["model"]?.stringValue == "hang" ? .hang(["data: first"]) : .lines(["data: \(body["model"]?.stringValue ?? "none")"])
        }
        func request(_ model: String) -> URLRequest {
            var request = URLRequest(url: URL(string: "https://example.com")!)
            request.httpBody = Data(#"{"model":"\#(model)"}"#.utf8)
            return request
        }
        var lines: [String] = []
        for try await line in try await transport.lines(for: request("m1")) { lines.append(line) }
        XCTAssertEqual(lines, ["data: m1"])
        XCTAssertEqual(transport.bodies, [["model": "m1"]])
        XCTAssertEqual(transport.cancellationCount, 0)

        let hanging = try await transport.lines(for: request("hang"))
        let reader = Task { () -> [String] in
            var received: [String] = []
            for try await line in hanging { received.append(line) }
            return received
        }
        try await Task.sleep(nanoseconds: 20_000_000)
        reader.cancel()
        let cancelled = await waitUntil { transport.cancellationCount == 1 }
        XCTAssertTrue(cancelled)
        let received = try await reader.value
        XCTAssertEqual(received, ["data: first"])
    }

    func testScriptedProviderDelaysAndFails() async throws {
        let provider = ScriptedProvider([.cue(.working), .reply(.text("a"))], failAfter: 2, delay: 0.01)
        var received: [AssistantEvent] = []
        do {
            for try await event in provider.streamEvents(system: "S", turns: []) { received.append(event) }
            XCTFail("Expected an error")
        } catch {
            XCTAssertEqual((error as? URLError)?.code, .notConnectedToInternet)
        }
        XCTAssertEqual(received, [.cue(.working), .reply(.text("a"))])
        let ended = await waitUntil { provider.terminationCount == 1 }
        XCTAssertTrue(ended)

        let immediate = ScriptedProvider([.cue(.working)], failAfter: 0)
        do {
            _ = try await collectEvents(immediate.streamEvents(system: "S", turns: []))
            XCTFail("Expected an error")
        } catch {}
    }

    func testFakeToolsRecordRunsAndConcurrency() async {
        let log = ToolActivityLog()
        let slow = FakeTool(name: "list_events", output: .ok(["events": []], summary: "Calendar"), delay: 0.05, log: log)
        let fast = FakeTool(name: "get_current_time", log: log)
        let broken = FakeTool(name: "create_reminder", effect: .sideEffect, error: URLError(.cancelled), log: log)
        let calls = [
            PendingToolCall(id: "1", name: "list_events", input: ["start": "a", "end": "b"]),
            PendingToolCall(id: "2", name: "get_current_time", input: [:]),
            PendingToolCall(id: "3", name: "create_reminder", input: ["title": "x"]),
            PendingToolCall(id: "4", name: "missing", input: [:]),
            PendingToolCall(id: "5", name: "get_current_time", rawInput: "{bad"),
        ]

        let serial = FakeToolExecutor([slow, fast, broken])
        XCTAssertEqual(serial.definitions.map(\.name), ["create_reminder", "get_current_time", "list_events"])
        let round = await serial.run(calls, context: ToolContext())
        XCTAssertEqual(round.calls.map(\.id), ["1", "2", "3", "4", "5"])
        XCTAssertEqual(round.calls.map(\.isError), [false, false, true, true, true])
        XCTAssertEqual(round.calls[3].result, #"{"error":"Unknown tool"}"#)
        XCTAssertEqual(round.calls[4].result, #"{"INVALID_JSON":"{bad"}"#)
        XCTAssertEqual(round.summaryLine, "Calendar")
        XCTAssertEqual(serial.log.maxConcurrency, 1)
        XCTAssertEqual(serial.rounds, [calls])
        XCTAssertEqual(log.entries, [.start("list_events"), .end("list_events"), .start("get_current_time"), .end("get_current_time"), .start("create_reminder"), .end("create_reminder")])
        XCTAssertEqual(slow.inputs, [["start": "a", "end": "b"]])
        XCTAssertEqual(fast.runCount, 1)

        let parallel = FakeToolExecutor([slow, FakeTool(name: "get_current_time", delay: 0.05)], concurrent: true)
        let both = await parallel.run(Array(calls[0...1]), context: ToolContext())
        XCTAssertEqual(both.calls.map(\.id), ["1", "2"])
        XCTAssertEqual(parallel.log.maxConcurrency, 2)
    }

    func testManualClockWakesSleepersInOrder() async throws {
        let clock = ManualClock(now: 10)
        let order = Recorder()
        let early = Task { try await clock.sleep(until: 10.5); order.append("early") }
        let late = Task { try await clock.sleep(for: 2); order.append("late") }
        let doomed = Task { try await clock.sleep(until: 100) }
        let waiting = await waitUntil { clock.sleeperCount == 3 }
        XCTAssertTrue(waiting)

        clock.advance(by: 0.4)
        XCTAssertEqual(clock.now, 10.4, accuracy: 1e-9)
        XCTAssertEqual(clock.sleeperCount, 3)
        clock.advance(by: 0.2)
        try await early.value
        XCTAssertEqual(order.values, ["early"])
        clock.advance(to: 5)
        XCTAssertEqual(clock.now, 10.6, accuracy: 1e-9, "never moves backwards")
        clock.advance(to: 12)
        try await late.value
        XCTAssertEqual(order.values, ["early", "late"])

        doomed.cancel()
        do {
            try await doomed.value
            XCTFail("Expected CancellationError")
        } catch {
            XCTAssertTrue(error is CancellationError)
        }
        XCTAssertEqual(clock.sleeperCount, 0)
        try await clock.sleep(until: 1)
    }
}
