import AssistantKit
import Foundation
import LocalEngine
import LocalEngineTestSupport
import XCTest

/// `LocalToolLoop` on the scripted tiny engines (plan WP31): the cue and activity come before a
/// round; the results are appended to the cache, then the final text; the ledger stays exact
/// across rounds; `handoff_to_cloud` is intercepted (early abort, offline, after text, not
/// available); the plausibility guard; `maxRounds`; cancellation.
final class LocalToolLoopTests: XCTestCase {
    private let tokenizer = FakeChatMLTokenizer()
    private let system = "You are Alvin."

    override func setUpWithError() throws {
        try MetalAvailability.require()
    }

    // MARK: Cues, activity and the round

    func testCueAndActivityComeBeforeTheRoundThenTheAnswer() async throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let (engine, _) = try scriptedEngine(
                tiny: tiny, scripts: scripts([call("list_reminders", "{\"scope\": \"today\"}"), "You have one: call mum."]))
            let log = LoopToolLog()
            let fixture = makeLoop(engine, log: log)
            let outcome = await collect(fixture.loop.run(request("What's on my list today?", fixture.runner)))
            XCTAssertNil(outcome.error, "\(tiny)")
            XCTAssertEqual(
                kinds(outcome.events),
                ["cue(checking)", "activity(Checking your reminders)", "toolRound", "activity(nil)", "text", "finished(completed)"],
                "\(tiny)")
            XCTAssertEqual(text(outcome.events), "You have one: call mum.", "\(tiny)")
            XCTAssertEqual(outcome.events.first, .progress(.responseStarted), "\(tiny): progress marks pass through")
            XCTAssertTrue(outcome.events.contains(.progress(.toolCallStarted(name: "list_reminders"))), "\(tiny)")

            let round = try XCTUnwrap(rounds(outcome.events).first, "\(tiny)")
            XCTAssertEqual(round.calls.map(\.name), ["list_reminders"], "\(tiny)")
            XCTAssertEqual(round.calls.first?.input, ["scope": "today"], "\(tiny)")
            XCTAssertEqual(round.calls.first?.isError, false, "\(tiny)")
            XCTAssertEqual(round.calls.first?.summary, "1 reminder", "\(tiny)")
            XCTAssertEqual(log.runs.map(\.name), ["list_reminders"], "\(tiny)")
        }
    }

    /// The round's results go into the cache right after the call, then the final answer; the
    /// second generation reuses everything before them. Lenient input turns `"600"` into 600.
    func testResultsAreAppendedThenTheFinalText() async throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let callText = call("set_timer", "{\"seconds\": \"600\", \"label\": \"Pasta\"}")
            let (engine, _) = try scriptedEngine(
                tiny: tiny, scripts: scripts([callText, "Your pasta timer is running."]))
            let log = LoopToolLog()
            let reports = LoopReports()
            let fixture = makeLoop(engine, log: log)
            let outcome = await collect(
                fixture.loop.run(request("Set a timer for ten minutes for the pasta.", fixture.runner), report: reports.add))
            XCTAssertNil(outcome.error, "\(tiny)")
            // A side-effect tool has no cue.
            XCTAssertEqual(
                kinds(outcome.events), ["activity(Starting a timer)", "toolRound", "activity(nil)", "text", "finished(completed)"], "\(tiny)")
            XCTAssertEqual(log.runs.first?.input["seconds"], 600, "\(tiny): lenient input")
            let record = try XCTUnwrap(rounds(outcome.events).first?.calls.first, "\(tiny)")
            XCTAssertEqual(record.input, ["seconds": "600", "label": "Pasta"], "\(tiny): the record keeps what the model wrote")
            XCTAssertFalse(record.isError, "\(tiny)")

            let (ledger, replyStart) = try await engine.withSession { ($0.ledger, $0.snapshot?.turns.first?.replyStart) }
            let start = try XCTUnwrap(replyStart, "\(tiny)")
            let expected = callText + "<|im_end|>\n<|im_start|>user\n<tool_response>\n" + record.result
                + "\n</tool_response><|im_end|>\n" + EngineTestHarness.generationPromptText + "Your pasta timer is running.<|im_end|>"
            XCTAssertEqual(tokenizer.decodeRaw(Array(ledger[start...])), expected, "\(tiny)")

            let report = try XCTUnwrap(reports.all.first, "\(tiny)")
            XCTAssertEqual(reports.all.count, 1, "\(tiny)")
            XCTAssertEqual(report.rounds, 1, "\(tiny)")
            XCTAssertEqual(report.generations.count, 2, "\(tiny)")
            XCTAssertEqual(report.generations.first?.planReason, "newSession", "\(tiny)")
            XCTAssertEqual(report.generations.last?.planReason, "toolRound", "\(tiny)")
            XCTAssertEqual(report.generations.last?.reusedTokens, start + tokenizer.encodeRaw(callText).count + 1, "\(tiny)")
            XCTAssertNotNil(report.timeToFirstText, "\(tiny)")
            XCTAssertNil(report.handoffReason, "\(tiny)")
            XCTAssertEqual(report.stats.generatedTokens, report.generations.map(\.generatedTokens).reduce(0, +), "\(tiny)")
            try await assertConsistent(engine, "\(tiny)")
        }
    }

    /// Two rounds and the answer: the cache is exact between the rounds (checked from inside
    /// the tools) and after; the stored reply then extends the cache on the next request.
    func testLedgerStaysConsistentAcrossRounds() async throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let (engine, _) = try scriptedEngine(
                tiny: tiny,
                scripts: scripts([
                    call("get_current_time", "{}"), call("set_timer", "{\"seconds\": 600}"), "Timer set; it is noon.", "You're welcome.",
                ]))
            let checks = LoopChecks()
            let log = LoopToolLog()
            let fixture = makeLoop(engine, log: log, onRun: {
                let state = try? await engine.withSession { ($0.assertConsistent(), $0.pendingCount) }
                checks.add(state.map { $0.0.isConsistent() && $0.1 == 0 } ?? false)
            })
            let userTurn = ChatTurn(role: .user, text: "What time is it? Then set a 10 minute timer.")
            let outcome = await collect(fixture.loop.run(EngineRequest(system: system, tools: fixture.runner.definitions, turns: [userTurn])))
            XCTAssertNil(outcome.error, "\(tiny)")
            XCTAssertEqual(log.runs.map(\.name), ["get_current_time", "set_timer"], "\(tiny)")
            XCTAssertEqual(checks.all, [true, true], "\(tiny): exact before each round's tool ran")
            XCTAssertEqual(
                kinds(outcome.events),
                [
                    "cue(checking)", "activity(Checking the time)", "toolRound", "activity(Starting a timer)", "toolRound",
                    "activity(nil)", "text", "finished(completed)",
                ],
                "\(tiny)")
            try await assertConsistent(engine, "\(tiny) after the reply")

            // The reply as the app stores it: its text and both rounds.
            let reply = ChatTurn(role: .assistant, text: text(outcome.events), toolRounds: rounds(outcome.events))
            let turns = [userTurn, reply, ChatTurn(role: .user, text: "Thanks.")]
            let before = try await engine.withSession { $0.ledger.count }
            let reports = LoopReports()
            let next = await collect(
                fixture.loop.run(EngineRequest(system: system, tools: fixture.runner.definitions, turns: turns), report: reports.add))
            XCTAssertNil(next.error, "\(tiny)")
            XCTAssertEqual(text(next.events), "You're welcome.", "\(tiny)")
            XCTAssertEqual(reports.all.first?.generations.first?.planReason, "append", "\(tiny)")
            XCTAssertEqual(reports.all.first?.generations.first?.reusedTokens, before, "\(tiny)")
            try await assertConsistent(engine, "\(tiny) after the next turn")
        }
    }

    // MARK: Handoff

    /// Online, with nothing shown: generation stops as soon as the model has named
    /// `handoff_to_cloud`, before it writes its (long) reason, and the stream throws. The cache
    /// stays exact, and later requests (a retry, or the conversation after the cloud answered)
    /// rewind past the aborted reply.
    func testEarlyHandoffAbortsAndLaterRequestsStayConsistent() async throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let reason = String(repeating: "this needs live weather data from the web ", count: 16)
            let script = tokenizer.encodeRaw(call("handoff_to_cloud", "{\"reason\": \"\(reason)\"}"))
            let (engine, _) = try scriptedEngine(
                tiny: tiny, scripts: [script] + scripts(["Glad to help."]),
                configuration: EngineTestHarness.testConfiguration(maxTokens: 2_048))
            let log = LoopToolLog()
            let reports = LoopReports()
            let fixture = makeLoop(engine, log: log)
            let userTurn = ChatTurn(role: .user, text: "Will it rain in Singapore tomorrow?")
            let first = EngineRequest(system: system, tools: fixture.runner.definitions, turns: [userTurn])
            let outcome = await collect(fixture.loop.run(first, report: reports.add))

            XCTAssertEqual(outcome.error as? ReplyHandoff, ReplyHandoff(reason: LocalToolLoop.earlyHandoffReason), "\(tiny): \(String(describing: outcome.error))")
            XCTAssertEqual(kinds(outcome.events), [], "\(tiny): nothing but progress marks")
            XCTAssertEqual(outcome.events.last, .progress(.toolCallStarted(name: HandoffTool.name)), "\(tiny)")
            XCTAssertTrue(log.runs.isEmpty, "\(tiny)")
            XCTAssertEqual(reports.all.first?.handoffReason, LocalToolLoop.earlyHandoffReason, "\(tiny)")
            XCTAssertEqual(reports.all.first?.rounds, 0, "\(tiny)")

            // The engine stopped before decoding the reason.
            let (fed, replyStart) = try await engine.withSession { ($0.ledger.count, $0.snapshot?.turns.first?.replyStart) }
            let start = try XCTUnwrap(replyStart, "\(tiny)")
            XCTAssertLessThan(fed - start, script.count, "\(tiny): the reason wasn't decoded")
            try await assertConsistent(engine, "\(tiny) after the abort")

            // Retried locally (say the cloud failed): the user turn is replaced.
            let retried = try await EngineTestHarness.collect(engine.reply(first))
            XCTAssertEqual(EngineTestHarness.finish(retried)?.stats.planReason, "replaceLastUserTurn", "\(tiny)")
            try await assertConsistent(engine, "\(tiny) after the retry")

            // The conversation goes on after the cloud's answer: the aborted reply diverges.
            let after = EngineRequest(
                system: system, tools: fixture.runner.definitions,
                turns: [userTurn, ChatTurn(role: .assistant, text: "Light showers are likely."), ChatTurn(role: .user, text: "Thanks!")])
            let next = try await EngineTestHarness.collect(engine.reply(after))
            XCTAssertEqual(EngineTestHarness.finish(next)?.stats.planReason, "diverged", "\(tiny)")
            XCTAssertEqual(EngineTestHarness.text(next), "Glad to help.", "\(tiny)")
            try await assertConsistent(engine, "\(tiny) after the next turn")
        }
    }

    /// A parsed handoff call that reaches the round while a handoff is possible (here: the
    /// connection came back after the name was read) throws with the model's reason.
    func testParsedHandoffCallThrowsWithTheModelsReason() async throws {
        let (engine, _) = try scriptedEngine(
            tiny: .qwen3, scripts: scripts([call("handoff_to_cloud", "{\"reason\": \"needs the web\"}")]))
        let online = LoopSequence([false, true])
        let fixture = makeLoop(engine, log: LoopToolLog(), isOnline: { online.next() })
        let outcome = await collect(fixture.loop.run(request("What's the news?", fixture.runner)))
        XCTAssertEqual(outcome.error as? ReplyHandoff, ReplyHandoff(reason: "needs the web"))
        XCTAssertEqual(kinds(outcome.events), [])
        try await assertConsistent(engine, "handoff at the round")
    }

    func testOfflineHandoffIsAnsweredWithAToolResult() async throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let (engine, _) = try scriptedEngine(
                tiny: tiny,
                scripts: scripts([call("handoff_to_cloud", "{\"reason\": \"needs the web\"}"), "I can't check the weather offline."]))
            let log = LoopToolLog()
            let fixture = makeLoop(engine, log: log, isOnline: { false })
            let outcome = await collect(fixture.loop.run(request("Will it rain tomorrow?", fixture.runner)))
            XCTAssertNil(outcome.error, "\(tiny)")
            // Nothing runs, so there is no cue or activity.
            XCTAssertEqual(kinds(outcome.events), ["toolRound", "text", "finished(completed)"], "\(tiny)")
            XCTAssertEqual(text(outcome.events), "I can't check the weather offline.", "\(tiny)")
            let record = try XCTUnwrap(rounds(outcome.events).first?.calls.first, "\(tiny)")
            XCTAssertEqual(record.name, HandoffTool.name, "\(tiny)")
            XCTAssertTrue(record.isError, "\(tiny)")
            XCTAssertEqual(record.result, ToolOutput.error(LocalToolLoop.offlineMessage).serializedContent, "\(tiny)")
            XCTAssertTrue(log.runs.isEmpty, "\(tiny): the handoff tool itself never runs")

            let ledger = try await engine.withSession { $0.ledger }
            XCTAssertTrue(tokenizer.decodeRaw(ledger).contains("<tool_response>\n" + record.result + "\n</tool_response>"), "\(tiny)")
            try await assertConsistent(engine, "\(tiny)")
        }
    }

    func testHandoffAfterTextIsAnsweredNotThrown() async throws {
        let (engine, _) = try scriptedEngine(
            tiny: .hybrid,
            scripts: scripts(["Let me see. " + call("handoff_to_cloud", "{\"reason\": \"weather\"}"), "It should stay dry."]))
        let fixture = makeLoop(engine, log: LoopToolLog())
        let outcome = await collect(fixture.loop.run(request("Will it rain tomorrow?", fixture.runner)))
        XCTAssertNil(outcome.error)
        XCTAssertEqual(kinds(outcome.events), ["text", "toolRound", "text", "finished(completed)"])
        XCTAssertEqual(text(outcome.events), "Let me see. It should stay dry.")
        let record = try XCTUnwrap(rounds(outcome.events).first?.calls.first)
        XCTAssertEqual(record.result, ToolOutput.error(LocalToolLoop.lateHandoffMessage).serializedContent)
        try await assertConsistent(engine, "late handoff")
    }

    func testHandoffWithoutACloudAssistantIsAnswered() async throws {
        let (engine, _) = try scriptedEngine(
            tiny: .qwen3, scripts: scripts([call("handoff_to_cloud", "{\"reason\": \"news\"}"), "I can't look that up here."]))
        let fixture = makeLoop(engine, log: LoopToolLog(), handoffAvailable: false)
        let outcome = await collect(fixture.loop.run(request("Any news today?", fixture.runner)))
        XCTAssertNil(outcome.error)
        XCTAssertEqual(text(outcome.events), "I can't look that up here.")
        let record = try XCTUnwrap(rounds(outcome.events).first?.calls.first)
        XCTAssertEqual(record.result, ToolOutput.error(LocalToolLoop.handoffUnavailableMessage).serializedContent)
    }

    // MARK: Guard

    /// Asked about a running timer, the model starts a one-second timer (as Woof did in the lab).
    /// Online, the request goes to the cloud instead; offline, the call gets an error result.
    /// Neither runs the timer.
    func testGuardHandsOffOrRefusesAnImplausibleTimer() async throws {
        let implausible = call("set_timer", "{\"seconds\": \"1\"}")
        let question = "How long is left on my timer?"

        let (online, _) = try scriptedEngine(tiny: .qwen3, scripts: scripts([implausible]))
        let onlineLog = LoopToolLog()
        let handingOff = makeLoop(online, log: onlineLog)
        let handedOff = await collect(handingOff.loop.run(request(question, handingOff.runner)))
        let handoff = try XCTUnwrap(handedOff.error as? ReplyHandoff, "\(String(describing: handedOff.error))")
        XCTAssertTrue(handoff.reason.hasPrefix("set_timer"), handoff.reason)
        XCTAssertEqual(kinds(handedOff.events), [])
        XCTAssertTrue(onlineLog.runs.isEmpty)
        try await assertConsistent(online, "guard handoff")

        let (offline, _) = try scriptedEngine(
            tiny: .hybrid, scripts: scripts([implausible, "How long should the timer run?"]))
        let offlineLog = LoopToolLog()
        let refusing = makeLoop(offline, log: offlineLog, isOnline: { false })
        let refused = await collect(refusing.loop.run(request(question, refusing.runner)))
        XCTAssertNil(refused.error)
        XCTAssertEqual(kinds(refused.events), ["toolRound", "text", "finished(completed)"])
        XCTAssertEqual(text(refused.events), "How long should the timer run?")
        let record = try XCTUnwrap(rounds(refused.events).first?.calls.first)
        XCTAssertEqual(record.name, "set_timer")
        XCTAssertTrue(record.isError)
        XCTAssertTrue(record.result.contains("already running"), record.result)
        XCTAssertTrue(offlineLog.runs.isEmpty)
        try await assertConsistent(offline, "guard refusal")

        // With the guard off, the call runs.
        let (unguarded, _) = try scriptedEngine(tiny: .qwen3, scripts: scripts([implausible, "Done."]))
        let unguardedLog = LoopToolLog()
        let permissive = makeLoop(unguarded, log: unguardedLog, callGuard: .none)
        let ran = await collect(permissive.loop.run(request(question, permissive.runner)))
        XCTAssertNil(ran.error)
        XCTAssertEqual(unguardedLog.runs.map(\.name), ["set_timer"])
    }

    /// A duration typed in words of a language the guard can't read still sets the timer, also
    /// offline and also when the amount came in the answer to the model's question.
    func testGuardLetsATimerInAnotherLanguageRun() async throws {
        let timer = call("set_timer", "{\"seconds\": \"600\", \"label\": \"pasta\"}")
        let conversations: [[ChatTurn]] = [
            [ChatTurn(role: .user, text: "Pon un temporizador de diez minutos para la pasta.")],
            [
                ChatTurn(role: .user, text: "Pon un temporizador para la pasta."),
                ChatTurn(role: .assistant, text: "¿Cuánto tiempo?"),
                ChatTurn(role: .user, text: "Diez minutos."),
            ],
            [ChatTurn(role: .user, text: "Tolong set timer sepuluh minit untuk pasta.")],
        ]
        for turns in conversations {
            let (engine, _) = try scriptedEngine(tiny: .hybrid, scripts: scripts([timer, "Listo."]))
            let log = LoopToolLog()
            let fixture = makeLoop(engine, log: log, isOnline: { false })
            let outcome = await collect(fixture.loop.run(EngineRequest(system: system, tools: fixture.runner.definitions, turns: turns)))
            let label = turns.last?.text ?? ""
            XCTAssertNil(outcome.error, label)
            XCTAssertEqual(log.runs.map(\.name), ["set_timer"], label)
            XCTAssertEqual(log.runs.first?.input["seconds"], 600, label)
            XCTAssertEqual(rounds(outcome.events).first?.calls.first?.isError, false, label)
            XCTAssertEqual(text(outcome.events), "Listo.", label)
            try await assertConsistent(engine, label)
        }
    }

    // MARK: Limits and endings

    func testMaxRoundsEndsWithToolLimit() async throws {
        let calls = Array(repeating: call("get_current_time", "{}"), count: 3)
        let (engine, _) = try scriptedEngine(tiny: .hybrid, scripts: scripts(calls + ["Never shown."]))
        let log = LoopToolLog()
        let reports = LoopReports()
        let fixture = makeLoop(engine, log: log, maxRounds: 2)
        let outcome = await collect(fixture.loop.run(request("What time is it?", fixture.runner), report: reports.add))
        XCTAssertNil(outcome.error)
        XCTAssertEqual(
            kinds(outcome.events),
            [
                "cue(checking)", "activity(Checking the time)", "toolRound", "cue(checking)", "activity(Checking the time)", "toolRound",
                "finished(\(ReplyStop.other(LocalToolLoop.toolLimitReason)))",
            ])
        XCTAssertEqual(log.runs.count, 2)
        XCTAssertEqual(reports.all.first?.rounds, 2)
        XCTAssertEqual(reports.all.first?.generations.count, 3)
        XCTAssertNil(reports.all.first?.timeToFirstText)
        try await assertConsistent(engine, "tool limit")
    }

    func testLengthLimitIsTruncated() async throws {
        let (engine, _) = try scriptedEngine(
            tiny: .qwen3, scripts: scripts(["This answer goes on for far too long."]),
            configuration: EngineTestHarness.testConfiguration(maxTokens: 8))
        let fixture = makeLoop(engine, log: LoopToolLog())
        let outcome = await collect(fixture.loop.run(request("Talk.", fixture.runner)))
        XCTAssertNil(outcome.error)
        XCTAssertEqual(kinds(outcome.events), ["text", "finished(truncated)"])
        XCTAssertEqual(text(outcome.events), "This ans")
    }

    /// The engine stopping a reply (the GPU stopped being allowed) ends the loop with
    /// `CancellationError`; the cache stays exact.
    func testEngineCancellationThrowsCancellationError() async throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let budget = LoopBudget(calls: 12)
            var configuration = EngineTestHarness.testConfiguration()
            configuration.hooks = EngineHooks(beginGPU: { true }, endGPU: {}, isAllowed: { budget.take() })
            let (engine, _) = try scriptedEngine(
                tiny: tiny, scripts: scripts([String(repeating: "more words ", count: 5)]), configuration: configuration)
            let fixture = makeLoop(engine, log: LoopToolLog())
            let outcome = await collect(fixture.loop.run(request("Keep going.", fixture.runner)))
            XCTAssertTrue(outcome.error is CancellationError, "\(tiny): \(String(describing: outcome.error))")
            XCTAssertFalse(kinds(outcome.events).contains { $0.hasPrefix("finished") }, "\(tiny)")
            budget.reset(calls: 1_000_000)
            try await assertConsistent(engine, "\(tiny) cancelled by the engine")
        }
    }

    /// Ending the loop's stream mid-reply stops the engine (the termination reaches it through
    /// the loop's task and the engine's stream) well before the reply's end; its cache stays exact.
    func testTerminatingTheStreamStopsTheEngine() async throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let script = tokenizer.encodeRaw(String(repeating: "word ", count: 80))
            let (engine, _) = try scriptedEngine(
                tiny: tiny, scripts: [script], configuration: EngineTestHarness.testConfiguration(maxTokens: 512))
            let fixture = makeLoop(engine, log: LoopToolLog())
            var texts = 0
            for try await event in fixture.loop.run(request("Say many words.", fixture.runner)) {
                if case .reply(.text) = event {
                    texts += 1
                    break
                }
            }
            XCTAssertEqual(texts, 1, "\(tiny)")
            await engine.waitUntilIdle()

            let (fed, replyStart) = try await engine.withSession { ($0.ledger.count, $0.snapshot?.turns.first?.replyStart) }
            let start = try XCTUnwrap(replyStart, "\(tiny)")
            XCTAssertLessThan(fed - start, script.count, "\(tiny): the engine stopped before the end of the reply")
            try await assertConsistent(engine, "\(tiny) terminated")
        }
    }

    // MARK: Fixtures

    private struct Fixture {
        let loop: LocalToolLoop
        let runner: ToolRunner
    }

    private func makeLoop(
        _ engine: InferenceEngine, log: LoopToolLog, handoffAvailable: Bool = true,
        isOnline: @escaping @Sendable () -> Bool = { true }, maxRounds: Int = 3,
        callGuard: LocalToolLoop.CallGuard = .standard, onRun: (@Sendable () async -> Void)? = nil
    ) -> Fixture {
        let runner = ToolRunner(registry: ToolRegistry(LoopTools.make(log: log, onRun: onRun) + [HandoffTool.tool]), lenientInput: true)
        let loop = LocalToolLoop(
            engine: engine, executor: runner, handoffAvailable: handoffAvailable, isOnline: isOnline,
            toolContext: { ToolContext() }, maxRounds: maxRounds, callGuard: callGuard)
        return Fixture(loop: loop, runner: runner)
    }

    /// A scripted engine whose replies may be long enough for a whole tool call (the fake
    /// tokenizer spends one token per character).
    private func scriptedEngine(
        tiny: EngineTestHarness.Tiny, scripts: [[Int]], configuration: EngineConfiguration = EngineTestHarness.testConfiguration(maxTokens: 256)
    ) throws -> (engine: InferenceEngine, target: ScriptedTarget) {
        try EngineTestHarness.makeScriptedEngine(tiny: tiny, scripts: scripts, configuration: configuration)
    }

    private func request(_ text: String, _ runner: ToolRunner) -> EngineRequest {
        EngineRequest(system: system, tools: runner.definitions, turns: [ChatTurn(role: .user, text: text)])
    }

    /// A tool call in the fake tokenizer's (JSON) format.
    private func call(_ name: String, _ arguments: String) -> String {
        "<tool_call>\n{\"name\": \"\(name)\", \"arguments\": \(arguments)}\n</tool_call>"
    }

    private func scripts(_ texts: [String]) -> [[Int]] {
        texts.map { tokenizer.encodeRaw($0) }
    }

    private struct Outcome {
        var events: [AssistantEvent]
        var error: Error?
    }

    private func collect(_ stream: AsyncThrowingStream<AssistantEvent, Error>) async -> Outcome {
        var events: [AssistantEvent] = []
        do {
            for try await event in stream {
                events.append(event)
            }
            return Outcome(events: events, error: nil)
        } catch {
            return Outcome(events: events, error: error)
        }
    }

    /// The events other than progress marks, as short names, consecutive texts merged:
    /// `text`, `activity(…)`, `cue(…)`, `toolRound`, `finished(…)`.
    private func kinds(_ events: [AssistantEvent]) -> [String] {
        var kinds: [String] = []
        for event in events {
            let kind: String
            switch event {
            case .progress:
                continue
            case .reply(.text):
                kind = "text"
            case .reply(.activity(let activity)):
                kind = "activity(\(activity ?? "nil"))"
            case .reply(.finished(let stop)):
                kind = "finished(\(stop))"
            case .cue(let cue):
                kind = "cue(\(cue.rawValue))"
            case .toolRound:
                kind = "toolRound"
            case .routed:
                kind = "routed"
            }
            if kind == "text", kinds.last == "text" { continue }
            kinds.append(kind)
        }
        return kinds
    }

    private func text(_ events: [AssistantEvent]) -> String {
        events.compactMap { event -> String? in
            if case .reply(.text(let text)) = event { return text }
            return nil
        }.joined()
    }

    private func rounds(_ events: [AssistantEvent]) -> [ToolRound] {
        events.compactMap { event -> ToolRound? in
            if case .toolRound(let round) = event { return round }
            return nil
        }
    }

    private func assertConsistent(_ engine: InferenceEngine, _ label: String, file: StaticString = #filePath, line: UInt = #line) async throws {
        let (report, pending) = try await engine.withSession { ($0.assertConsistent(), $0.pendingCount) }
        XCTAssertEqual(pending, 0, label, file: file, line: line)
        XCTAssertTrue(report.isConsistent(), "\(label): \(report)", file: file, line: line)
    }
}

/// The plausibility guard and the statistics of a whole reply; no model needed.
final class LocalToolLoopLogicTests: XCTestCase {
    func testStatesAnAmount() {
        let amounts = [
            "Set a timer for 10 minutes.", "ten minute timer please", "Timer for half an hour", "an hour", "Give me a minute timer",
            "a couple of minutes", "timer for ３ minutes", "帮我设一个二十分钟的计时器。", "计时半小时", "五分钟后提醒我",
            "一个半小时后叫我", "倒计时一百二十秒", "计时十多分钟", "设一刻钟的计时器",
        ]
        for text in amounts {
            XCTAssertTrue(LocalToolLoop.CallGuard.statesAnAmount(text), text)
        }
        let questions = [
            "How long is left on my timer?", "我的计时器还剩多少时间？", "Set the timer", "what's left on the pasta timer", "Set a timer", "",
            "帮我看一下计时器", "设一个计时器", "计时器还剩几分钟？",
        ]
        for text in questions {
            XCTAssertFalse(LocalToolLoop.CallGuard.statesAnAmount(text), text)
        }
    }

    /// The guard reads the newest user turn, plus the one before it when the reply in between
    /// asked the user something.
    func testGuardUserText() {
        let asked = [
            ChatTurn(role: .user, text: "Set a timer for ten minutes."),
            ChatTurn(role: .assistant, text: "Sure. What should I call it?"),
            ChatTurn(role: .user, text: "Pasta."),
        ]
        XCTAssertEqual(LocalToolLoop.CallGuard.userText(of: asked), "Set a timer for ten minutes.\nPasta.")
        XCTAssertTrue(LocalToolLoop.CallGuard.statesAnAmount(LocalToolLoop.CallGuard.userText(of: asked)))

        let answered = [
            ChatTurn(role: .user, text: "Set a timer for ten minutes."),
            ChatTurn(role: .assistant, text: "Done, ten minutes."),
            ChatTurn(role: .user, text: "How long is left on it?"),
        ]
        XCTAssertEqual(LocalToolLoop.CallGuard.userText(of: answered), "How long is left on it?")
        XCTAssertEqual(LocalToolLoop.CallGuard.userText(of: [ChatTurn(role: .user, text: "Hi")]), "Hi")
        XCTAssertEqual(LocalToolLoop.CallGuard.userText(of: []), "")
    }

    func testStandardGuardChecksOnlyTimers() {
        let timer = PendingToolCall(id: "call_1", name: "set_timer", input: ["seconds": 1])
        let reminder = PendingToolCall(id: "call_2", name: "create_reminder", input: ["title": "Call mum"])
        let question = "How long is left on my timer?"
        XCTAssertNotNil(LocalToolLoop.CallGuard.standard.check(timer, question))
        XCTAssertNil(LocalToolLoop.CallGuard.standard.check(timer, "Set a timer for five minutes."))
        XCTAssertNil(LocalToolLoop.CallGuard.standard.check(reminder, question))
        XCTAssertNil(LocalToolLoop.CallGuard.none.check(timer, question))
    }

    /// The standard guard objects to a timer only when the user's words ask about a running
    /// timer or the duration is too short to go unsaid; an amount it can't read lets the call run.
    func testStandardGuardObjectsOnlyToTheMeasuredFailure() {
        func timer(_ seconds: JSONValue) -> PendingToolCall {
            PendingToolCall(id: "call_1", name: "set_timer", input: ["seconds": seconds])
        }
        let guardCheck = LocalToolLoop.CallGuard.standard.check

        // Asked about a running timer: refused whatever the duration (the lab's case: "1").
        for question in ["How long is left on my timer?", "我的计时器还剩多少时间？", "计时器还剩几分钟？", "How much time on the pasta timer?"] {
            for seconds: JSONValue in [1, "1", 600, "600"] {
                let objection = guardCheck(timer(seconds), question)
                XCTAssertTrue(objection?.contains("already running") == true, "\(question) \(seconds): \(String(describing: objection))")
            }
        }

        // A tiny duration nobody stated, in any language.
        for text in ["¿Cuánto le queda a mi temporizador?", "Wie lange läuft mein Timer noch?", "Set a timer"] {
            for seconds: JSONValue in [1, "1", "4", 0, -3, "2.5"] {
                let objection = guardCheck(timer(seconds), text)
                XCTAssertTrue(objection?.contains("didn't say how long") == true, "\(text) \(seconds): \(String(describing: objection))")
            }
        }

        // Amounts in words of other languages, typed: the timer runs.
        let typed: [(String, JSONValue)] = [
            ("Pon un temporizador de diez minutos", 600), ("¿Puedes poner un temporizador de diez minutos?", "600"),
            ("Tolong set timer sepuluh minit", "600"), ("Stell einen Timer auf zwanzig Minuten", 1_200),
            ("Mets un minuteur de trente secondes", 30), ("タイマーを五分にセットして", 300), ("타이머 십 분 맞춰 줘", 600),
        ]
        for (text, seconds) in typed {
            XCTAssertNil(guardCheck(timer(seconds), text), text)
        }

        // An amount the guard reads lets even a one-second timer run.
        XCTAssertNil(guardCheck(timer(1), "Set a timer for 1 second."))
        XCTAssertNil(guardCheck(timer("1"), "Set a timer for one second."))
        // Without an amount or a question, a plausible duration runs: the guard can't tell an
        // invented duration from one stated in a language it doesn't read.
        XCTAssertNil(guardCheck(timer(300), "Set a timer"))
        // Unreadable `seconds` is left to input validation.
        XCTAssertNil(guardCheck(timer("soon"), "Set a timer"))
        XCTAssertNil(guardCheck(PendingToolCall(id: "call_2", name: "set_timer", input: ["label": "Tea"]), "Set a timer"))
    }

    func testAsksAboutRunningTimer() {
        let questions = [
            "How long is left on my timer?", "how much time remaining", "What's left on the pasta timer", "How long until the timer ends?",
            "我的计时器还剩多少时间？", "计时器还剩几分钟？", "还有多久？", "計時器還有幾分鐘", "倒计时还有几个小时", "计时器剩下的时间",
        ]
        for text in questions {
            XCTAssertTrue(LocalToolLoop.CallGuard.asksAboutRunningTimer(text), text)
        }
        let requests = [
            "Set a timer for ten minutes", "Set a timer", "Timer for half an hour", "", "设一个计时器", "帮我设一个二十分钟的计时器",
            "Pon un temporizador de diez minutos", "Show me how to cook rice", "几乎完成了",
        ]
        for text in requests {
            XCTAssertFalse(LocalToolLoop.CallGuard.asksAboutRunningTimer(text), text)
        }
    }

    func testCombinedStatistics() {
        let first = LocalGenerationStats(
            timeToFirstText: 0.4, promptTokens: 100, promptTime: 0.2, generatedTokens: 20, generateTime: 0.5, reusedSession: true,
            draftTokens: 8, acceptedDraftTokens: 6, peakMemoryBytes: 1_000, engine: "alvin",
            phases: EnginePhaseTimes(plan: 0.001, prefill: 0.2, firstToken: 0.05, start: "warm"), prefilledTokens: 100,
            reusedTokens: 400, planReason: "append",
            speculation: SpeculationStats(rounds: 2, plainTokens: 4, drafted: ["lookup": 8], accepted: ["lookup": 6], tokensPerRound: [3: 2]),
            confidence: TokenConfidence(meanTop1: 0.9, p10Top1: 0.6, tokens: 20))
        let second = LocalGenerationStats(
            timeToFirstText: 0.1, promptTokens: 30, promptTime: 0.05, generatedTokens: 10, generateTime: 0.25, reusedSession: true,
            peakMemoryBytes: 3_000, engine: "alvin", prefilledTokens: 30, reusedTokens: 520, planReason: "toolRound",
            speculation: SpeculationStats(rounds: 1, plainTokens: 2, drafted: ["seeds": 4], accepted: ["seeds": 4], tokensPerRound: [3: 1, 5: 1]),
            confidence: TokenConfidence(meanTop1: 0.6, p10Top1: 0.3, tokens: 10))

        let combined = LocalToolLoop.combine([first, second], timeToFirstText: 1.2)
        XCTAssertEqual(combined.timeToFirstText, 1.2)
        XCTAssertEqual(combined.promptTokens, 130)
        XCTAssertEqual(combined.promptTime, 0.25, accuracy: 1e-9)
        XCTAssertEqual(combined.generatedTokens, 30)
        XCTAssertEqual(combined.generateTime, 0.75, accuracy: 1e-9)
        XCTAssertEqual(combined.draftTokens, 8)
        XCTAssertEqual(combined.acceptedDraftTokens, 6)
        XCTAssertEqual(combined.peakMemoryBytes, 3_000)
        XCTAssertEqual(combined.prefilledTokens, 130)
        XCTAssertEqual(combined.reusedTokens, 400, "the first generation's")
        XCTAssertEqual(combined.planReason, "append")
        XCTAssertEqual(combined.phases, first.phases)
        XCTAssertEqual(combined.engine, "alvin")
        XCTAssertTrue(combined.reusedSession)
        XCTAssertEqual(
            combined.speculation,
            SpeculationStats(
                rounds: 3, plainTokens: 6, drafted: ["lookup": 8, "seeds": 4], accepted: ["lookup": 6, "seeds": 4], tokensPerRound: [3: 3, 5: 1]))
        XCTAssertEqual(combined.confidence?.tokens, 30)
        XCTAssertEqual(combined.confidence?.meanTop1 ?? 0, 0.8, accuracy: 1e-9)
        XCTAssertEqual(combined.confidence?.p10Top1, 0.3)

        XCTAssertNil(LocalToolLoop.combine([first], timeToFirstText: nil).timeToFirstText)
        let empty = LocalToolLoop.combine([], timeToFirstText: nil)
        XCTAssertEqual(empty.generatedTokens, 0)
        XCTAssertEqual(empty.engine, "alvin")
    }
}

// MARK: - Test tools

/// The tools the loop tests offer: `get_current_time` and `list_reminders` (read-only, with the
/// `.checking` cue) and `set_timer` (a side effect, no cue), shaped like the app's (plan §5.3).
enum LoopTools {
    static func make(log: LoopToolLog, onRun: (@Sendable () async -> Void)?) -> [any AssistantTool] {
        [
            LoopTool(
                definition: ToolDefinition(
                    name: "get_current_time", description: "The current local date and time.",
                    inputSchema: ["type": "object", "properties": .object([:]), "required": .array([]), "additionalProperties": false]),
                effect: .readOnly, presentation: ToolPresentation(activity: "Checking the time", cue: .checking), log: log, onRun: onRun,
                output: { _ in .ok(["time": "12:00"], summary: "12:00") }),
            LoopTool(
                definition: ToolDefinition(
                    name: "list_reminders", description: "Reads the user's open reminders.",
                    inputSchema: [
                        "type": "object",
                        "properties": ["scope": ["type": "string", "enum": ["today", "upcoming", "overdue", "all"]]],
                        "required": ["scope"], "additionalProperties": false,
                    ]),
                effect: .readOnly, presentation: ToolPresentation(activity: "Checking your reminders", cue: .checking), log: log,
                onRun: onRun, output: { _ in .ok(["reminders": [["title": "Call mum"]]], summary: "1 reminder") }),
            LoopTool(
                definition: ToolDefinition(
                    name: "set_timer", description: "Starts a countdown timer.",
                    inputSchema: [
                        "type": "object",
                        "properties": ["seconds": ["type": "integer"], "label": ["type": "string"]],
                        "required": ["seconds"], "additionalProperties": false,
                    ]),
                effect: .sideEffect, presentation: ToolPresentation(activity: "Starting a timer", cue: nil), log: log, onRun: onRun,
                output: { input in .ok(["id": "timer-1", "seconds": input["seconds"] ?? .null], summary: "Timer") }),
        ]
    }
}

struct LoopTool: AssistantTool {
    let definition: ToolDefinition
    let effect: ToolEffect
    let presentation: ToolPresentation
    let log: LoopToolLog
    let onRun: (@Sendable () async -> Void)?
    let output: @Sendable ([String: JSONValue]) -> ToolOutput

    func run(_ input: [String: JSONValue], context: ToolContext) async throws -> ToolOutput {
        if let onRun {
            await onRun()
        }
        log.record(name: definition.name, input: input)
        return output(input)
    }
}

/// The tool runs, in the order they happened.
final class LoopToolLog: @unchecked Sendable {
    struct Run {
        let name: String
        let input: [String: JSONValue]
    }

    private let lock = NSLock()
    private var recorded: [Run] = []

    func record(name: String, input: [String: JSONValue]) {
        lock.lock()
        recorded.append(Run(name: name, input: input))
        lock.unlock()
    }

    var runs: [Run] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }
}

/// The reports a loop delivered.
final class LoopReports: @unchecked Sendable {
    private let lock = NSLock()
    private var reports: [LocalToolLoop.Report] = []

    var add: @Sendable (LocalToolLoop.Report) -> Void {
        { [self] report in
            lock.lock()
            reports.append(report)
            lock.unlock()
        }
    }

    var all: [LocalToolLoop.Report] {
        lock.lock()
        defer { lock.unlock() }
        return reports
    }
}

/// Results of checks made from inside tools.
final class LoopChecks: @unchecked Sendable {
    private let lock = NSLock()
    private var results: [Bool] = []

    func add(_ result: Bool) {
        lock.lock()
        results.append(result)
        lock.unlock()
    }

    var all: [Bool] {
        lock.lock()
        defer { lock.unlock() }
        return results
    }
}

/// Answers from a list in turn, repeating the last one.
final class LoopSequence: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Bool]

    init(_ values: [Bool]) {
        self.values = values
    }

    func next() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return values.count > 1 ? values.removeFirst() : (values.first ?? false)
    }
}

/// True for the first `calls` checks, then false.
final class LoopBudget: @unchecked Sendable {
    private let lock = NSLock()
    private var left: Int

    init(calls: Int) {
        left = calls
    }

    func take() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard left > 0 else { return false }
        left -= 1
        return true
    }

    func reset(calls: Int) {
        lock.lock()
        left = calls
        lock.unlock()
    }
}
