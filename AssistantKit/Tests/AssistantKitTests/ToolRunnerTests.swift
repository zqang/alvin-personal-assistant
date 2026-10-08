import XCTest
@testable import AssistantKit

private let timerSchema: JSONValue = [
    "type": "object",
    "properties": ["seconds": ["type": "integer"], "label": ["type": "string"]],
    "required": ["seconds"],
    "additionalProperties": false,
]

private func call(_ id: String, _ name: String, _ input: JSONValue? = .object([:]), raw: String? = nil) -> PendingToolCall {
    PendingToolCall(id: id, name: name, input: input, rawInput: raw)
}

private let sideEffectTimedOut = #"{"error":"The tool didn't finish in time; the action may or may not have happened. Don't repeat it; tell the user it may not have gone through."}"#

/// A tool that ignores cancellation: it finishes only after `seconds`, whatever happens.
private struct StubbornTool: AssistantTool {
    var name = "stubborn"
    var effect: ToolEffect = .readOnly
    let seconds: Double
    /// Records `start(name)` and `end(name)` around each run.
    var log = ToolActivityLog()
    var definition: ToolDefinition { ToolDefinition(name: name, description: "Ignores cancellation.", inputSchema: FakeTool.emptySchema) }
    var presentation: ToolPresentation { .generic }

    func run(_ input: [String: JSONValue], context: ToolContext) async throws -> ToolOutput {
        log.record(.start(name))
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            DispatchQueue.global().asyncAfter(deadline: .now() + seconds) { continuation.resume() }
        }
        log.record(.end(name))
        return .ok(["late": true])
    }
}

final class ToolRunnerTests: XCTestCase {
    // MARK: - Order and concurrency

    func testRecordsKeepTheCallOrder() async {
        let slow = FakeTool(name: "slow", output: .ok(["n": 1]), delay: 0.15)
        let fast = FakeTool(name: "fast", output: .ok(["n": 2]))
        let write = FakeTool(name: "write", effect: .sideEffect, output: .ok(["n": 3], summary: "Wrote it"))
        let runner = ToolRunner(registry: ToolRegistry([slow, fast, write]))
        let calls = [call("c1", "slow"), call("c2", "write"), call("c3", "fast"), call("c4", "missing"), call("c5", "slow")]

        let round = await runner.run(calls, context: ToolContext())

        XCTAssertEqual(round.calls.map(\.id), ["c1", "c2", "c3", "c4", "c5"])
        XCTAssertEqual(round.calls.map(\.result), [#"{"n":1}"#, #"{"n":3}"#, #"{"n":2}"#, #"{"error":"Unknown tool"}"#, #"{"n":1}"#])
        XCTAssertEqual(round.calls.map(\.isError), [false, false, false, true, false])
        XCTAssertEqual(round.calls[1].summary, "Wrote it")
        XCTAssertEqual(round.summaryLine, "Wrote it")
        XCTAssertEqual(round.calls[0].input, .object([:]))
        XCTAssertEqual(slow.runCount, 2)
    }

    func testReadOnlyCallsOverlapAndSideEffectsRunOneAtATimeInOrder() async throws {
        let log = ToolActivityLog()
        let readA = FakeTool(name: "read_a", delay: 0.3, log: log)
        let readB = FakeTool(name: "read_b", delay: 0.3, log: log)
        let writeA = FakeTool(name: "write_a", effect: .sideEffect, delay: 0.2, log: log)
        let writeB = FakeTool(name: "write_b", effect: .sideEffect, delay: 0.2, log: log)
        let runner = ToolRunner(registry: ToolRegistry([readA, readB, writeA, writeB]))
        // The model asked for write_b before write_a.
        let calls = [call("1", "write_b"), call("2", "read_a"), call("3", "write_a"), call("4", "read_b")]

        let round = await runner.run(calls, context: ToolContext())
        XCTAssertEqual(round.calls.map(\.isError), [false, false, false, false])

        let entries = log.entries
        let times = log.times
        func position(_ entry: ToolActivityLog.Entry) throws -> Int { try XCTUnwrap(entries.firstIndex(of: entry), "\(entry)") }

        // Both reads start before either ends.
        let firstReadEnd = min(try position(.end("read_a")), try position(.end("read_b")))
        XCTAssertLessThan(try position(.start("read_a")), firstReadEnd)
        XCTAssertLessThan(try position(.start("read_b")), firstReadEnd)
        // The writes never overlap and keep the model's order.
        XCTAssertLessThan(try position(.end("write_b")), try position(.start("write_a")))
        XCTAssertGreaterThanOrEqual(times[try position(.start("write_a"))], times[try position(.end("write_b"))])
        // Reads don't wait for writes: a read starts before the first write ends.
        XCTAssertLessThan(min(try position(.start("read_a")), try position(.start("read_b"))), try position(.end("write_b")))
    }

    // MARK: - Failures that never run the tool

    func testUnknownToolGivesAnErrorRecord() async {
        let round = await ToolRunner(registry: .empty).run([call("c1", "nope", ["a": 1])], context: ToolContext())
        XCTAssertEqual(round.calls, [ToolCallRecord(id: "c1", name: "nope", input: ["a": 1], result: #"{"error":"Unknown tool"}"#, isError: true)])
    }

    func testDisabledToolKeepsItsDefinitionButNeverRuns() async {
        let reminder = FakeTool(name: "create_reminder", effect: .sideEffect)
        let time = FakeTool(name: "get_current_time")
        let registry = ToolRegistry([reminder, time], disabled: ["create_reminder"])
        let runner = ToolRunner(registry: registry)
        XCTAssertEqual(runner.definitions.map(\.name), ["create_reminder", "get_current_time"])

        let round = await runner.run([call("c1", "create_reminder"), call("c2", "get_current_time")], context: ToolContext())

        XCTAssertEqual(round.calls[0].result, #"{"error":"The user turned this off in Settings."}"#)
        XCTAssertTrue(round.calls[0].isError)
        XCTAssertFalse(round.calls[1].isError)
        XCTAssertEqual(reminder.runCount, 0)
        XCTAssertEqual(time.runCount, 1)
    }

    func testInvalidJSONIsReportedWithoutRunning() async {
        let timer = FakeTool(name: "set_timer", effect: .sideEffect, schema: timerSchema)
        let runner = ToolRunner(registry: ToolRegistry([timer]))
        let round = await runner.run([call("c1", "set_timer", nil, raw: #"{"seconds": 6"#)], context: ToolContext())
        XCTAssertEqual(round.calls, [
            ToolCallRecord(id: "c1", name: "set_timer", input: .object([:]), result: #"{"INVALID_JSON":"{\"seconds\": 6"}"#, isError: true),
        ])
        XCTAssertEqual(timer.runCount, 0)
    }

    func testInvalidInputIsReportedWithoutRunning() async {
        let timer = FakeTool(name: "set_timer", effect: .sideEffect, schema: timerSchema)
        let runner = ToolRunner(registry: ToolRegistry([timer]))
        let round = await runner.run([call("c1", "set_timer", ["label": "Tea"]), call("c2", "set_timer", ["seconds": "60"])], context: ToolContext())
        XCTAssertEqual(round.calls[0].result, #"{"error":"Missing required field \"seconds\"."}"#)
        XCTAssertEqual(round.calls[1].result, #"{"error":"Field \"seconds\" must be an integer."}"#)
        XCTAssertEqual(round.calls.map(\.isError), [true, true])
        XCTAssertEqual(timer.runCount, 0)
    }

    func testLenientInputIsCoercedBeforeTheToolRuns() async {
        let timer = FakeTool(name: "set_timer", effect: .sideEffect, schema: timerSchema)
        let runner = ToolRunner(registry: ToolRegistry([timer]), lenientInput: true)
        let round = await runner.run([call("c1", "set_timer", ["seconds": "60", "label": .null])], context: ToolContext())
        XCTAssertFalse(round.calls[0].isError)
        XCTAssertEqual(timer.inputs, [["seconds": 60]])
        // The record keeps the input as the model wrote it.
        XCTAssertEqual(round.calls[0].input, ["seconds": "60", "label": .null])
    }

    func testThrownErrorsBecomeErrorRecords() async {
        let failing = FakeTool(name: "list_events", error: AssistantError.missingConfiguration("Allow calendar access in Settings."))
        let round = await ToolRunner(registry: ToolRegistry([failing])).run([call("c1", "list_events")], context: ToolContext())
        XCTAssertEqual(round.calls[0].result, #"{"error":"Allow calendar access in Settings."}"#)
        XCTAssertTrue(round.calls[0].isError)
    }

    // MARK: - Timeout and truncation

    func testSlowToolTimesOut() async {
        let slow = FakeTool(name: "slow", delay: 5)
        let runner = ToolRunner(registry: ToolRegistry([slow]), timeout: .milliseconds(100))
        let started = Date()
        let round = await runner.run([call("c1", "slow")], context: ToolContext())
        XCTAssertLessThan(Date().timeIntervalSince(started), 2)
        XCTAssertTrue(round.calls[0].isError)
        XCTAssertEqual(round.calls[0].result, #"{"error":"The tool didn't finish in time."}"#)
    }

    func testTimeoutHoldsEvenWhenTheToolIgnoresCancellation() async {
        let write = FakeTool(name: "create_reminder", effect: .sideEffect)
        let runner = ToolRunner(registry: ToolRegistry([StubbornTool(seconds: 3), write]), timeout: .milliseconds(100))
        let started = Date()
        let round = await runner.run([call("c1", "stubborn")], context: ToolContext())
        XCTAssertLessThan(Date().timeIntervalSince(started), 2)
        XCTAssertTrue(round.calls[0].isError)
        XCTAssertEqual(round.calls[0].result, #"{"error":"The tool didn't finish in time."}"#)

        // A read-only tool that is still running doesn't hold side effects back.
        let next = await runner.run([call("c2", "create_reminder")], context: ToolContext())
        XCTAssertLessThan(Date().timeIntervalSince(started), 2)
        XCTAssertFalse(next.calls[0].isError)
        XCTAssertEqual(write.runCount, 1)
    }

    func testSideEffectTimeoutSaysTheActionMayHaveHappened() async {
        let slow = FakeTool(name: "slow_write", effect: .sideEffect, delay: 5)
        let runner = ToolRunner(registry: ToolRegistry([slow]), timeout: .milliseconds(50))
        let round = await runner.run([call("c1", "slow_write")], context: ToolContext())
        XCTAssertEqual(round.calls[0].result, sideEffectTimedOut)
    }

    func testATimedOutActionHoldsBackTheNextOneUntilItEnds() async {
        // The first action outlives its 1 s limit by half a second and ignores cancellation.
        let log = ToolActivityLog()
        let stuck = StubbornTool(name: "slow_write", effect: .sideEffect, seconds: 1.5, log: log)
        let next = FakeTool(name: "create_reminder", effect: .sideEffect, log: log)
        let runner = ToolRunner(registry: ToolRegistry([stuck, next]), timeout: .seconds(1))

        let round = await runner.run([call("c1", "slow_write"), call("c2", "create_reminder")], context: ToolContext())

        XCTAssertEqual(round.calls[0].result, sideEffectTimedOut)
        XCTAssertFalse(round.calls[1].isError)
        XCTAssertEqual(log.entries, [.start("slow_write"), .end("slow_write"), .start("create_reminder"), .end("create_reminder")])
    }

    func testATimedOutActionHoldsBackLaterRoundsToo() async {
        let log = ToolActivityLog()
        let stuck = StubbornTool(name: "slow_write", effect: .sideEffect, seconds: 1.5, log: log)
        let next = FakeTool(name: "create_reminder", effect: .sideEffect, log: log)
        let runner = ToolRunner(registry: ToolRegistry([stuck, next]), timeout: .seconds(1))

        let first = await runner.run([call("c1", "slow_write")], context: ToolContext())
        XCTAssertEqual(first.calls[0].result, sideEffectTimedOut)
        // A copy of the runner knows about the action that is still going.
        let copy = runner
        let second = await copy.run([call("c2", "create_reminder")], context: ToolContext())

        XCTAssertFalse(second.calls[0].isError)
        XCTAssertEqual(log.entries, [.start("slow_write"), .end("slow_write"), .start("create_reminder"), .end("create_reminder")])
    }

    func testAnActionStillRunningPastTheLimitStopsTheNextOne() async {
        let log = ToolActivityLog()
        let stuck = StubbornTool(name: "slow_write", effect: .sideEffect, seconds: 2, log: log)
        let next = FakeTool(name: "create_reminder", effect: .sideEffect, log: log)
        let read = FakeTool(name: "list_reminders", log: log)
        let runner = ToolRunner(registry: ToolRegistry([stuck, next, read]), timeout: .milliseconds(200))

        let started = Date()
        let round = await runner.run([call("c1", "slow_write"), call("c2", "create_reminder"), call("c3", "list_reminders")], context: ToolContext())

        XCTAssertLessThan(Date().timeIntervalSince(started), 1.5)
        XCTAssertEqual(round.calls.map(\.result), [
            sideEffectTimedOut,
            #"{"error":"An earlier action is still in progress, so this one wasn't started."}"#,
            #"{"ok":true}"#,
        ])
        XCTAssertEqual(next.runCount, 0)
        XCTAssertEqual(read.runCount, 1, "reads don't wait for actions")

        // Once the stuck action has ended, actions run again.
        let ended = await waitUntil(timeout: 5) { log.entries.contains(.end("slow_write")) }
        XCTAssertTrue(ended)
        let again = await runner.run([call("c4", "create_reminder")], context: ToolContext())
        XCTAssertFalse(again.calls[0].isError)
        XCTAssertEqual(next.runCount, 1)
    }

    func testCancellingTheRoundWhileAnEarlierActionRunsStopsTheNextOne() async throws {
        let stuck = StubbornTool(name: "slow_write", effect: .sideEffect, seconds: 2)
        let next = FakeTool(name: "create_reminder", effect: .sideEffect)
        let runner = ToolRunner(registry: ToolRegistry([stuck, next]), timeout: .milliseconds(500))
        let first = await runner.run([call("c1", "slow_write")], context: ToolContext())
        XCTAssertEqual(first.calls[0].result, sideEffectTimedOut)

        let running = Task { await runner.run([call("c2", "create_reminder")], context: ToolContext()) }
        try await Task.sleep(nanoseconds: 100_000_000)
        running.cancel()
        let round = await running.value

        XCTAssertEqual(round.calls.map(\.result), [#"{"error":"The request was cancelled before the action ran."}"#])
        XCTAssertEqual(next.runCount, 0)
    }

    func testLongResultsAreCutDeterministically() async throws {
        let text = String(repeating: "abcdefghij", count: 30)
        let long = FakeTool(name: "long", output: .ok(["text": .string(text)]))
        let short = FakeTool(name: "short", output: .ok(["text": "hi"]))
        let runner = ToolRunner(registry: ToolRegistry([long, short]), maxResultCharacters: 40)
        let full = String(decoding: try JSONValue.object(["text": .string(text)]).serialized(), as: UTF8.self)

        let first = await runner.run([call("c1", "long"), call("c2", "short")], context: ToolContext())
        let second = await runner.run([call("c1", "long")], context: ToolContext())

        XCTAssertEqual(first.calls[0].result, String(full.prefix(40)) + "…(truncated)")
        XCTAssertEqual(first.calls[0].result, second.calls[0].result)
        XCTAssertEqual(first.calls[1].result, #"{"text":"hi"}"#)
        XCTAssertEqual(ToolRunner.truncate("12345", to: 5), "12345")
        XCTAssertEqual(ToolRunner.truncate("123456", to: 5), "12345…(truncated)")
        XCTAssertEqual(ToolRunner.truncate("你好世界！", to: 2), "你好…(truncated)")
    }

    func testDefaultLimitIs2000Characters() async {
        let long = FakeTool(name: "long", output: .ok(["text": .string(String(repeating: "x", count: 5_000))]))
        let round = await ToolRunner(registry: ToolRegistry([long])).run([call("c1", "long")], context: ToolContext())
        XCTAssertEqual(round.calls[0].result.count, 2_000 + "…(truncated)".count)
    }

    // MARK: - Read-only registries

    func testReadOnlyRegistryBlocksSideEffects() async {
        let write = FakeTool(name: "create_event", effect: .sideEffect)
        let read = FakeTool(name: "list_events")
        let registry = ToolRegistry([write, read]).readOnly()
        let runner = ToolRunner(registry: registry)
        XCTAssertEqual(runner.definitions.map(\.name), ["create_event", "list_events"])

        let gate = CommitGate()  // never opened: a blocked call must not wait for it
        let round = await runner.run([call("c1", "create_event"), call("c2", "list_events")], context: ToolContext(commitGate: gate))

        XCTAssertEqual(round.calls[0].result, #"{"error":"Not available here."}"#)
        XCTAssertTrue(round.calls[0].isError)
        XCTAssertFalse(round.calls[1].isError)
        XCTAssertEqual(write.runCount, 0)
        XCTAssertEqual(read.runCount, 1)
    }

    // MARK: - Commit gate

    func testSideEffectsWaitForTheCommitGate() async throws {
        let write = FakeTool(name: "create_reminder", effect: .sideEffect)
        let read = FakeTool(name: "list_reminders")
        let runner = ToolRunner(registry: ToolRegistry([write, read]))
        let gate = CommitGate()

        let running = Task { await runner.run([call("c1", "create_reminder"), call("c2", "list_reminders")], context: ToolContext(commitGate: gate)) }
        let waiting = await waitUntil { gate.waiterCount == 1 && read.runCount == 1 }
        XCTAssertTrue(waiting)
        try await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertEqual(write.runCount, 0, "no side effect before the gate opens")

        gate.open()
        let round = await running.value
        XCTAssertEqual(write.runCount, 1)
        XCTAssertEqual(round.calls.map(\.isError), [false, false])
    }

    func testCancelledGateStopsEverySideEffect() async {
        let first = FakeTool(name: "create_reminder", effect: .sideEffect)
        let second = FakeTool(name: "set_timer", effect: .sideEffect)
        let read = FakeTool(name: "get_current_time")
        let runner = ToolRunner(registry: ToolRegistry([first, second, read]))
        let gate = CommitGate()

        let running = Task {
            await runner.run([call("c1", "create_reminder"), call("c2", "get_current_time"), call("c3", "set_timer")], context: ToolContext(commitGate: gate))
        }
        _ = await waitUntil { gate.waiterCount == 1 }
        gate.cancel()
        let round = await running.value

        let cancelled = #"{"error":"The request was cancelled before the action ran."}"#
        XCTAssertEqual(round.calls.map(\.result), [cancelled, #"{"ok":true}"#, cancelled])
        XCTAssertEqual(round.calls.map(\.isError), [true, false, true])
        XCTAssertEqual(first.runCount + second.runCount, 0)

        // A gate cancelled up front stops side effects just the same.
        let again = await runner.run([call("c4", "set_timer")], context: ToolContext(commitGate: gate))
        XCTAssertEqual(again.calls.map(\.result), [cancelled])
        XCTAssertEqual(second.runCount, 0)
    }

    func testCancellingTheRoundStopsWaitingSideEffects() async {
        let write = FakeTool(name: "create_reminder", effect: .sideEffect)
        let runner = ToolRunner(registry: ToolRegistry([write]))
        let gate = CommitGate()

        let running = Task { await runner.run([call("c1", "create_reminder")], context: ToolContext(commitGate: gate)) }
        _ = await waitUntil { gate.waiterCount == 1 }
        running.cancel()
        let round = await running.value
        gate.open()

        XCTAssertEqual(round.calls.map(\.result), [#"{"error":"The request was cancelled before the action ran."}"#])
        XCTAssertEqual(write.runCount, 0)
    }

    func testNoGateMeansSideEffectsRunAtOnce() async {
        let write = FakeTool(name: "create_reminder", effect: .sideEffect)
        let round = await ToolRunner(registry: ToolRegistry([write])).run([call("c1", "create_reminder")], context: ToolContext())
        XCTAssertFalse(round.calls[0].isError)
        XCTAssertEqual(write.runCount, 1)
    }

    func testEmptyRound() async {
        let round = await ToolRunner(registry: .empty).run([], context: ToolContext())
        XCTAssertEqual(round, ToolRound(calls: []))
    }
}

final class ToolRegistryTests: XCTestCase {
    func testDefinitionsAreSortedByNameAndLaterToolsReplaceEarlierOnes() {
        let registry = ToolRegistry([
            FakeTool(name: "set_timer"),
            FakeTool(name: "create_reminder", description: "old"),
            FakeTool(name: "list_events"),
            FakeTool(name: "create_reminder", description: "new"),
        ])
        XCTAssertEqual(registry.definitions.map(\.name), ["create_reminder", "list_events", "set_timer"])
        XCTAssertEqual(registry.definitions.first?.description, "new")
        XCTAssertFalse(registry.isEmpty)
        XCTAssertTrue(ToolRegistry.empty.isEmpty)
        XCTAssertEqual(ToolRegistry.empty.definitions, [])
    }

    func testPresentation() {
        let checking = ToolPresentation(activity: "Checking your calendar", cue: .checking)
        let registry = ToolRegistry([FakeTool(name: "list_events", presentation: checking)])
        XCTAssertEqual(registry.presentation(for: "list_events"), checking)
        XCTAssertEqual(registry.presentation(for: "unknown"), .generic)
        XCTAssertEqual(ToolRunner(registry: registry).presentation(for: "list_events"), checking)
    }

    func testToolNamedReflectsWhetherItMayRun() async throws {
        let write = FakeTool(name: "create_event", effect: .sideEffect)
        let read = FakeTool(name: "list_events")
        let off = FakeTool(name: "set_timer", effect: .sideEffect)
        let registry = ToolRegistry([write, read, off], disabled: ["set_timer"])
        XCTAssertNil(registry.tool(named: "nope"))

        let runnable = try XCTUnwrap(registry.tool(named: "create_event"))
        XCTAssertEqual(runnable.effect, .sideEffect)
        _ = try await runnable.run([:], context: ToolContext())
        XCTAssertEqual(write.runCount, 1)

        let disabled = try XCTUnwrap(registry.tool(named: "set_timer"))
        XCTAssertEqual(disabled.definition, off.definition)
        let disabledOutput = try await disabled.run([:], context: ToolContext())
        XCTAssertEqual(disabledOutput, .error("The user turned this off in Settings."))
        XCTAssertEqual(off.runCount, 0)

        let blocked = try XCTUnwrap(registry.readOnly().tool(named: "create_event"))
        XCTAssertEqual(blocked.effect, .readOnly)
        let blockedOutput = try await blocked.run([:], context: ToolContext())
        XCTAssertEqual(blockedOutput, .error("Not available here."))
        XCTAssertEqual(write.runCount, 1)
        XCTAssertEqual(registry.readOnly().tool(named: "list_events")?.effect, .readOnly)
    }

    func testSubsetAndAdding() {
        let registry = ToolRegistry(
            [FakeTool(name: "create_reminder"), FakeTool(name: "list_timers"), FakeTool(name: "set_timer")],
            disabled: ["set_timer"]
        )
        let local = registry.subset(["create_reminder", "set_timer", "not_there"])
        XCTAssertEqual(local.definitions.map(\.name), ["create_reminder", "set_timer"])
        if case .unavailable(let message) = local.resolve("set_timer") {
            XCTAssertEqual(message, ToolRegistry.disabledMessage)
        } else {
            XCTFail("set_timer should stay turned off")
        }

        let withHandoff = local.adding([HandoffTool.tool, FakeTool(name: "create_reminder", description: "replaced")])
        XCTAssertEqual(withHandoff.definitions.map(\.name), ["create_reminder", "handoff_to_cloud", "set_timer"])
        XCTAssertEqual(withHandoff.definitions.first?.description, "replaced")
        XCTAssertEqual(local.definitions.count, 2, "the original is unchanged")
    }

    func testCoveringStubsToolsThatOnlyHistoryKnows() async {
        let history = [
            ChatTurn(role: .user, text: "Remind me"),
            ChatTurn(role: .assistant, text: "", toolRounds: [
                ToolRound(calls: [
                    ToolCallRecord(id: "a", name: "create_reminder", input: ["title": "x"], result: "{}"),
                    ToolCallRecord(id: "b", name: "handoff_to_cloud", input: ["reason": "web"], result: "{}"),
                ]),
            ]),
            ChatTurn(role: .assistant, text: "Done", toolRounds: [
                ToolRound(calls: [ToolCallRecord(id: "c", name: "old_tool", input: [:], result: "{}")]),
            ]),
        ]
        let reminder = FakeTool(name: "create_reminder", effect: .sideEffect)
        let registry = ToolRegistry([reminder]).covering(history)

        XCTAssertEqual(registry.definitions.map(\.name), ["create_reminder", "handoff_to_cloud", "old_tool"])
        XCTAssertEqual(registry.definitions[0], reminder.definition, "a tool the registry has is not stubbed")
        let stub = registry.definitions[1]
        XCTAssertEqual(stub.inputSchema["additionalProperties"], false)
        XCTAssertEqual(stub.inputSchema["required"], .array([]))
        XCTAssertEqual(stub.inputSchema["properties"], .object([:]))
        XCTAssertEqual(registry.presentation(for: "old_tool").cue, nil)

        // Covering is deterministic, so requests built from it are byte-identical.
        XCTAssertEqual(ToolRegistry([reminder]).covering(history).definitions, registry.definitions)
        XCTAssertEqual(ToolRegistry([reminder]).covering([]).definitions, [reminder.definition])

        let round = await ToolRunner(registry: registry).run([call("d", "old_tool", ["x": 1]), call("e", "create_reminder")], context: ToolContext())
        XCTAssertEqual(round.calls[0].result, #"{"error":"Not available here."}"#)
        XCTAssertTrue(round.calls[0].isError)
        XCTAssertFalse(round.calls[1].isError)
    }

    func testHandoffTool() async throws {
        XCTAssertEqual(HandoffTool.name, "handoff_to_cloud")
        XCTAssertEqual(HandoffTool.definition.name, HandoffTool.name)
        XCTAssertEqual(HandoffTool.tool.definition, HandoffTool.definition)
        XCTAssertEqual(HandoffTool.definition.inputSchema["required"], ["reason"])
        XCTAssertEqual(HandoffTool.definition.inputSchema["additionalProperties"], false)
        XCTAssertEqual(HandoffTool.definition.inputSchema["properties"]?["reason"]?["type"], "string")
        XCTAssertEqual(HandoffTool.tool.effect, .readOnly)
        XCTAssertNil(HandoffTool.tool.presentation.cue)

        let output = try await HandoffTool.tool.run(["reason": "needs the web"], context: ToolContext())
        XCTAssertEqual(output, .error("intercepted"))

        let round = await ToolRunner(registry: ToolRegistry([HandoffTool.tool])).run(
            [call("h", "handoff_to_cloud", ["reason": "news"]), call("i", "handoff_to_cloud", [:])],
            context: ToolContext()
        )
        XCTAssertEqual(round.calls.map(\.result), [#"{"error":"intercepted"}"#, #"{"error":"Missing required field \"reason\"."}"#])
    }

    func testHandoffSchemaMatchesTheLabDefinition() throws {
        // scripts/lab_prompts.json holds the schema the lab measures models with; keep the input
        // schema in sync. The description was made more explicit after the lab ran, so it isn't
        // compared here.
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("scripts/lab_prompts.json")
        guard let data = try? Data(contentsOf: url) else {
            throw XCTSkip("scripts/lab_prompts.json is not reachable from \(url.path)")
        }
        let lab = try JSONValue.parse(data)
        let entry = try XCTUnwrap(lab["tool_schemas"]?.arrayValue?.first { $0["name"] == "handoff_to_cloud" })
        XCTAssertEqual(entry["input_schema"], HandoffTool.definition.inputSchema)
    }
}
