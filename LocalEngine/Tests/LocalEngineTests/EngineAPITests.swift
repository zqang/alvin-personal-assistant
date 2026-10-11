import AssistantKit
import Foundation
@testable import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLMCommon
import XCTest

/// The engine's public contract: event order, the GPU hooks, tool calls and their
/// continuation, idling and the engine's description of itself.
final class EngineAPITests: XCTestCase {
    private let tokenizer = FakeChatMLTokenizer()
    private let system = "You are Alvin."
    private let tools = [
        ToolDefinition(
            name: "set_timer", description: "Starts a timer.",
            inputSchema: ["type": "object", "properties": ["seconds": ["type": "integer"]], "required": ["seconds"]]),
    ]

    override func setUpWithError() throws {
        try MetalAvailability.require()
    }

    private func request(_ text: String, tools: [ToolDefinition] = []) -> EngineRequest {
        EngineRequest(system: system, tools: tools, turns: [ChatTurn(role: .user, text: text)])
    }

    func testEventOrderOfAPlainReply() async throws {
        let (engine, _) = try EngineTestHarness.makeScriptedEngine(tiny: .hybrid, scripts: [tokenizer.encodeRaw("Hi! Nice day.")])
        let events = try await EngineTestHarness.collect(engine.reply(request("Hello")))
        let kinds = EngineTestHarness.kinds(events)
        XCTAssertEqual(Array(kinds.prefix(3)), ["responseStarted", "prefillDone", "firstToken"])
        XCTAssertEqual(kinds.last, "finished(stop)")
        XCTAssertTrue(kinds.dropFirst(3).dropLast().allSatisfy { $0 == "text" }, "\(kinds)")
        XCTAssertEqual(EngineTestHarness.text(events), "Hi! Nice day.")

        let finish = try XCTUnwrap(EngineTestHarness.finish(events))
        let stats = finish.stats
        XCTAssertEqual(stats.engine, "alvin")
        XCTAssertEqual(stats.generatedTokens, "Hi! Nice day.".count)
        XCTAssertEqual(stats.planReason, "newSession")
        XCTAssertEqual(stats.reusedTokens, 0)
        XCTAssertEqual(stats.prefilledTokens, stats.promptTokens)
        XCTAssertGreaterThan(stats.prefilledTokens ?? 0, 0)
        XCTAssertNotNil(stats.timeToFirstText)
        XCTAssertNotNil(stats.phases)
        XCTAssertEqual(stats.confidence?.tokens, "Hi! Nice day.".count + 1, "every sampled token, the stop token included")
        XCTAssertNil(stats.speculation)

        if case .progress(.prefillDone(let prefilled, let reused)) = events[1] {
            XCTAssertEqual(prefilled, stats.prefilledTokens)
            XCTAssertEqual(reused, 0)
        } else {
            XCTFail("the second event is \(events[1])")
        }
    }

    func testToolCallsThenContinueReply() async throws {
        let call = "<tool_call>\n{\"name\": \"set_timer\", \"arguments\": {\"seconds\": 600}}\n</tool_call>"
        let (engine, _) = try EngineTestHarness.makeScriptedEngine(
            tiny: .qwen3, scripts: [tokenizer.encodeRaw("Sure. " + call), tokenizer.encodeRaw("Timer set.")])
        let first = try await EngineTestHarness.collect(engine.reply(request("Ten minute timer please", tools: tools)))
        let kinds = EngineTestHarness.kinds(first)
        XCTAssertEqual(Array(kinds.suffix(2)), ["toolCalls", "finished(toolCalls)"])
        let started = kinds.filter { $0.hasPrefix("toolCallStarted") }
        XCTAssertEqual(started, ["toolCallStarted(nil)", "toolCallStarted(set_timer)"])
        XCTAssertEqual(EngineTestHarness.text(first), "Sure. ", "the call itself isn't visible")
        let calls = EngineTestHarness.toolCalls(first)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.name, "set_timer")
        XCTAssertEqual(calls.first?.input, ["seconds": 600])
        XCTAssertTrue(calls.first?.id.hasPrefix("call_") ?? false)
        await assertBusy(engine.retryContinuation(toolChoice: .required), "no continuation to retry yet")

        let round = ToolRound(calls: calls.map { ToolCallRecord(call: $0, output: .ok(["ok": true], summary: "Timer 10 min")) })
        let second = try await EngineTestHarness.collect(engine.continueReply(after: round))
        let secondKinds = EngineTestHarness.kinds(second)
        XCTAssertEqual(Array(secondKinds.prefix(3)), ["responseStarted", "prefillDone", "firstToken"])
        XCTAssertEqual(secondKinds.last, "finished(stop)")
        XCTAssertEqual(EngineTestHarness.text(second), "Timer set.")
        XCTAssertEqual(EngineTestHarness.finish(second)?.stats.planReason, "toolRound")

        // The reply is recorded as one assistant turn: its text across rounds, and the round.
        let recorded = try await engine.withSession { $0.snapshot?.turns.last?.turn }
        XCTAssertEqual(recorded?.role, .assistant)
        XCTAssertEqual(recorded?.text, "Sure. Timer set.")
        XCTAssertEqual(recorded?.toolRounds, [round])

        // Nothing waits for results any more.
        await assertBusy(engine.continueReply(after: round), "continueReply without pending calls")

        // The continuation again: what it wrote leaves the cache and the reply first.
        let again = try await EngineTestHarness.collect(engine.retryContinuation(toolChoice: .native))
        XCTAssertEqual(EngineTestHarness.text(again), "Timer set.", "the same context replays its script")
        XCTAssertEqual(EngineTestHarness.finish(again)?.stats.planReason, "retry")
        XCTAssertEqual(EngineTestHarness.finish(again)?.stats.prefilledTokens, 1)
        let retried = try await engine.withSession { $0.snapshot?.turns.last?.turn }
        XCTAssertEqual(retried?.text, "Sure. Timer set.")
        XCTAssertEqual(retried?.toolRounds, [round])
        let report = try await engine.withSession { $0.assertConsistent() }
        XCTAssertTrue(report.isConsistent(), "\(report)")
    }

    private func assertBusy(
        _ events: AsyncThrowingStream<EngineEvent, Error>, _ label: String, file: StaticString = #filePath, line: UInt = #line
    ) async {
        do {
            _ = try await EngineTestHarness.collect(events)
            XCTFail("\(label) should throw", file: file, line: line)
        } catch {
            XCTAssertEqual(error as? EngineError, .busy, label, file: file, line: line)
        }
    }

    /// A tagged call that doesn't parse (here: malformed arguments) is never shown; it comes
    /// back as a call with its text in `rawInput`, so the tool runner can answer with an error.
    func testUnparsedToolCallIsReportedNotShown() async throws {
        let call = "<tool_call>\n{\"name\": \"set_timer\", \"arguments\": {\"seconds\": }}\n</tool_call>"
        let (engine, _) = try EngineTestHarness.makeScriptedEngine(tiny: .qwen3, scripts: [tokenizer.encodeRaw("Ok. " + call)])
        let events = try await EngineTestHarness.collect(engine.reply(request("Timer please", tools: tools)))
        XCTAssertEqual(EngineTestHarness.text(events), "Ok. ")
        XCTAssertEqual(Array(EngineTestHarness.kinds(events).suffix(2)), ["toolCalls", "finished(toolCalls)"])
        let calls = EngineTestHarness.toolCalls(events)
        XCTAssertEqual(calls.map(\.name), ["set_timer"])
        XCTAssertNil(calls.first?.input)
        XCTAssertEqual(calls.first?.rawInput, "{\"name\": \"set_timer\", \"arguments\": {\"seconds\": }}")
        XCTAssertTrue(calls.first?.id.hasPrefix("call_") ?? false)
    }

    /// A request that doesn't end with a user turn is refused before anything changes: the
    /// cached conversation and the reply waiting for tool results both survive it.
    func testInvalidRequestLeavesTheSessionAlone() async throws {
        let call = "<tool_call>\n{\"name\": \"set_timer\", \"arguments\": {\"seconds\": 60}}\n</tool_call>"
        let (engine, _) = try EngineTestHarness.makeScriptedEngine(
            tiny: .qwen3, scripts: [tokenizer.encodeRaw(call), tokenizer.encodeRaw("Timer set.")])
        let first = try await EngineTestHarness.collect(engine.reply(request("One minute timer", tools: tools)))
        XCTAssertEqual(EngineTestHarness.finish(first)?.reason, .toolCalls)
        let before = try await engine.withSession { ($0.ledger, $0.snapshot) }

        let invalid = EngineRequest(
            system: system, tools: tools,
            turns: [ChatTurn(role: .user, text: "One minute timer"), ChatTurn(role: .assistant, text: "On it.")])
        do {
            _ = try await EngineTestHarness.collect(engine.reply(invalid))
            XCTFail("a request ending with an assistant turn should throw")
        } catch {
            if case .renderFailed? = error as? EngineError {} else { XCTFail("unexpected \(error)") }
        }
        let after = try await engine.withSession { ($0.ledger, $0.snapshot) }
        XCTAssertEqual(after.0, before.0)
        XCTAssertEqual(after.1, before.1)

        let round = ToolRound(calls: EngineTestHarness.toolCalls(first).map { ToolCallRecord(call: $0, output: .ok(["ok": true], summary: "1 min")) })
        let second = try await EngineTestHarness.collect(engine.continueReply(after: round))
        XCTAssertEqual(EngineTestHarness.text(second), "Timer set.")
        XCTAssertEqual(EngineTestHarness.finish(second)?.stats.planReason, "toolRound")
    }

    /// Prewarming the prefix the live conversation already starts with keeps the conversation
    /// and its pending tool round, even after every checkpoint was dropped.
    func testPrewarmKeepsTheLiveConversationWithoutCheckpoints() async throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let call = "<tool_call>\n{\"name\": \"set_timer\", \"arguments\": {\"seconds\": 60}}\n</tool_call>"
            let (engine, _) = try EngineTestHarness.makeScriptedEngine(
                tiny: tiny, scripts: [tokenizer.encodeRaw(call), tokenizer.encodeRaw("Timer set.")])
            let first = try await EngineTestHarness.collect(engine.reply(request("One minute timer", tools: tools)))
            XCTAssertEqual(EngineTestHarness.finish(first)?.reason, .toolCalls, "\(tiny)")
            await engine.dropCheckpoints(keepSystem: false)
            let before = try await engine.withSession { ($0.ledger, $0.snapshot, $0.checkpoints.marks.isEmpty) }
            XCTAssertTrue(before.2, "\(tiny)")

            try await engine.prewarm(system: system, tools: tools)
            let after = try await engine.withSession { ($0.ledger, $0.snapshot) }
            XCTAssertEqual(after.0, before.0, "\(tiny)")
            XCTAssertEqual(after.1, before.1, "\(tiny)")

            let round = ToolRound(calls: EngineTestHarness.toolCalls(first).map { ToolCallRecord(call: $0, output: .ok(["ok": true], summary: "1 min")) })
            let second = try await EngineTestHarness.collect(engine.continueReply(after: round))
            XCTAssertEqual(EngineTestHarness.text(second), "Timer set.", "\(tiny)")
            XCTAssertEqual(EngineTestHarness.finish(second)?.stats.reusedTokens, before.0.count, "\(tiny)")
            let report = try await engine.withSession { $0.assertConsistent() }
            XCTAssertTrue(report.isConsistent(), "\(tiny): \(report)")
        }
    }

    func testDisallowedGPUCancels() async throws {
        let allowed = CallBudget(calls: 0)
        var configuration = EngineTestHarness.testConfiguration()
        configuration.hooks = EngineHooks(beginGPU: { true }, endGPU: {}, isAllowed: { allowed.take() })
        let (engine, _) = try EngineTestHarness.makeScriptedEngine(tiny: .hybrid, scripts: [tokenizer.encodeRaw("Never.")], configuration: configuration)
        let events = try await EngineTestHarness.collect(engine.reply(request("Hello")))
        XCTAssertEqual(EngineTestHarness.finish(events)?.reason, .cancelled)
        XCTAssertEqual(EngineTestHarness.text(events), "")
        let state = try await engine.withSession { ($0.pendingCount, $0.snapshot == nil) }
        XCTAssertEqual(state.0, 0)
        XCTAssertTrue(state.1, "an interrupted prefill leaves nothing to reuse")
    }

    func testRefusedGPUThrowsLeftForeground() async throws {
        let ends = CallBudget(calls: 0)
        var configuration = EngineTestHarness.testConfiguration()
        configuration.hooks = EngineHooks(beginGPU: { false }, endGPU: { _ = ends.take() }, isAllowed: { true })
        let engine = try EngineTestHarness.makeEngine(tiny: .qwen3, configuration: configuration)
        do {
            _ = try await EngineTestHarness.collect(engine.reply(request("Hello")))
            XCTFail("expected leftForeground")
        } catch {
            XCTAssertEqual(error as? EngineError, .leftForeground)
        }
        do {
            try await engine.warmUp()
            XCTFail("expected leftForeground")
        } catch {
            XCTAssertEqual(error as? EngineError, .leftForeground)
        }
    }

    func testGPUHooksBracketEveryJob() async throws {
        let counter = HookCounter()
        var configuration = EngineTestHarness.testConfiguration(maxTokens: 4)
        configuration.hooks = EngineHooks(
            beginGPU: { counter.begin() }, endGPU: { counter.end() }, isAllowed: { counter.check() })
        let engine = try EngineTestHarness.makeEngine(tiny: .hybrid, configuration: configuration)
        try await engine.warmUp()
        _ = try await EngineTestHarness.collect(engine.reply(request("One")))
        try await engine.prewarm(system: "Other system.", tools: [])
        XCTAssertEqual(counter.begins, 3)
        XCTAssertEqual(counter.ends, 3)
        XCTAssertFalse(counter.overlapped)
    }

    func testWaitUntilIdleWaitsForQueuedReplies() async throws {
        let engine = try EngineTestHarness.makeEngine(tiny: .qwen3, configuration: EngineTestHarness.testConfiguration(maxTokens: 16))
        let stream = engine.reply(request("Count"))
        let consumer = Task { try await EngineTestHarness.collect(stream) }
        await engine.waitUntilIdle()
        let finished = try await engine.withSession { $0.snapshot?.turns.last?.turn.role }
        XCTAssertEqual(finished, .assistant, "the reply was recorded before waitUntilIdle returned")
        let events = try await consumer.value
        XCTAssertNotNil(EngineTestHarness.finish(events))
    }

    func testInfoWarmUpAndSummary() async throws {
        let hybrid = try EngineTestHarness.makeEngine(tiny: .hybrid)
        XCTAssertTrue(hybrid.info.isHybrid)
        XCTAssertTrue(hybrid.info.forked)
        XCTAssertEqual(hybrid.info.vocabularySize, TinyModels.vocabularySize)
        XCTAssertEqual(hybrid.info.stopTokenIDs, [FakeChatMLTokenizer.imEnd, FakeChatMLTokenizer.endOfText])
        XCTAssertEqual(hybrid.info.toolCallFormat, "json")
        XCTAssertEqual(EngineInfo.formatVersion, 1)
        XCTAssertTrue(hybrid.target is HybridTarget)
        try await hybrid.warmUp()
        XCTAssertGreaterThan(hybrid.info.bytesPerCheckpoint, 0)
        let empty = try await hybrid.withSession { ($0.ledger.count, $0.target.cache.map(\.offset)) }
        XCTAssertEqual(empty.0, 0, "warm-up leaves the session as it found it")
        XCTAssertEqual(empty.1, [0, 0, 0, 0])

        let qwen3 = try EngineTestHarness.makeEngine(tiny: .qwen3)
        XCTAssertFalse(qwen3.info.isHybrid)
        XCTAssertFalse(qwen3.info.forked)
        XCTAssertEqual(qwen3.info.vocabularySize, TinyModels.vocabularySize)
        XCTAssertTrue(qwen3.target is StockTarget)
        try await qwen3.warmUp()
        XCTAssertEqual(qwen3.info.bytesPerCheckpoint, 0)

        _ = try await EngineTestHarness.collect(qwen3.reply(request("Hello")))
        let summary = await qwen3.sessionSummary()
        XCTAssertTrue(summary.hasPrefix("warm"), summary)
        XCTAssertTrue(summary.contains("systemEnd="), summary)
        await qwen3.invalidateSession()
        let cold = await qwen3.sessionSummary()
        XCTAssertTrue(cold.hasPrefix("cold"), cold)
    }

    func testDropCheckpointsKeepsSystemEndAndStaysExact() async throws {
        let (engine, _) = try EngineTestHarness.makeScriptedEngine(
            tiny: .hybrid, scripts: ["One.", "Two.", "Three."].map(tokenizer.encodeRaw))
        var turns = [ChatTurn(role: .user, text: "First?")]
        let first = try await EngineTestHarness.collect(engine.reply(EngineRequest(system: system, turns: turns)))
        turns += [ChatTurn(role: .assistant, text: EngineTestHarness.text(first)), ChatTurn(role: .user, text: "Second?")]
        let second = try await EngineTestHarness.collect(engine.reply(EngineRequest(system: system, turns: turns)))
        XCTAssertEqual(EngineHarnessCheck.reason(second), "append")
        await engine.dropCheckpoints(keepSystem: true)
        let marks = try await engine.withSession { $0.checkpoints.marks }
        XCTAssertEqual(Array(marks.keys), [.systemEnd])

        // Replacing the second user turn rewinds past the dropped checkpoint: the session
        // restores systemEnd and re-feeds the first exchange, exactly.
        turns[turns.count - 1] = ChatTurn(role: .user, text: "Second, edited?")
        let third = try await EngineTestHarness.collect(engine.reply(EngineRequest(system: system, turns: turns)))
        XCTAssertEqual(EngineHarnessCheck.reason(third), "replaceLastUserTurn")
        XCTAssertEqual(EngineTestHarness.text(third), "Three.")
        let report = try await engine.withSession { $0.assertConsistent() }
        XCTAssertTrue(report.isConsistent(), "\(report)")
    }

    /// A tokenizer without ChatML markers still answers, rendering everything each time.
    func testNoChatMLRendersEverythingEveryTime() async throws {
        let model = try EngineTestHarness.makeModel(.qwen3, seed: 5)
        let loaded = try EngineTestHarness.loadedModel(model, tokenizer: FakeChatMLTokenizer(), id: "plain", modelType: "qwen3")
        let plain = LoadedModel(
            id: loaded.id, directory: loaded.directory, modelType: loaded.modelType, model: loaded.model,
            tokenizer: loaded.tokenizer, renderer: WithoutTurnMarkers(base: FakeChatMLTokenizer()),
            configuration: loaded.configuration, container: loaded.container, stopTokenIDs: loaded.stopTokenIDs)
        let engine = InferenceEngine(loaded: plain, configuration: EngineTestHarness.testConfiguration(maxTokens: 6))
        for _ in 0 ..< 2 {
            let events = try await EngineTestHarness.collect(engine.reply(request("Hello")))
            let finish = try XCTUnwrap(EngineTestHarness.finish(events))
            XCTAssertEqual(finish.stats.planReason, "noReuse")
            XCTAssertEqual(finish.stats.reusedTokens, 0)
        }
        let noReuse = try await engine.withSession { $0.noReuse }
        XCTAssertTrue(noReuse)
    }

    // MARK: Tool choice and the call's opening

    /// `.required` opens the reply with a tool call (Woof's xml-function format here): the name
    /// is picked among the request's tools, the arguments decode freely, and the tool round's
    /// continuation decides for itself. The cache holds exactly the call as written.
    func testRequiredToolChoiceOpensACall() async throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let call = "<tool_call>\n<function=create_reminder>\n<parameter=title>\nparcel\n</parameter>\n</function>\n</tool_call>"
            let (engine, target) = try EngineTestHarness.makeScriptedEngine(
                tiny: tiny, scripts: [call, "Reminder set."].map(tokenizer.encodeRaw),
                configuration: EngineTestHarness.testConfiguration(maxTokens: 256), toolCallFormat: .xmlFunction)
            var required = request("Remind me about the parcel", tools: deviceTools)
            required.toolChoice = .required
            let first = try await EngineTestHarness.collect(engine.reply(required))
            let kinds = EngineTestHarness.kinds(first)
            XCTAssertEqual(
                Array(kinds.prefix(5)),
                ["responseStarted", "prefillDone", "firstToken", "toolCallStarted(nil)", "toolCallStarted(create_reminder)"], "\(tiny)")
            XCTAssertEqual(Array(kinds.suffix(2)), ["toolCalls", "finished(toolCalls)"], "\(tiny)")
            XCTAssertEqual(EngineTestHarness.text(first), "", "\(tiny)")
            let calls = EngineTestHarness.toolCalls(first)
            XCTAssertEqual(calls.map(\.name), ["create_reminder"], "\(tiny)")
            XCTAssertEqual(calls.first?.input, ["title": "parcel"], "\(tiny)")
            let stats = try XCTUnwrap(EngineTestHarness.finish(first)?.stats, "\(tiny)")
            XCTAssertTrue(stats.forcedToolCall, "\(tiny)")
            XCTAssertGreaterThan(stats.toolCallProbability ?? 0, 0.99, "\(tiny): the script's one-hot row")
            XCTAssertEqual(stats.generatedTokens, tokenizer.encodeRaw(call).count, "\(tiny): the opening counts as generated")

            let (ledger, replyStart) = try await engine.withSession { ($0.ledger, $0.snapshot?.turns.first?.replyStart) }
            let start = try XCTUnwrap(replyStart, "\(tiny)")
            XCTAssertEqual(tokenizer.decodeRaw(Array(ledger[start...])), call + "<|im_end|>", "\(tiny)")

            let round = ToolRound(calls: calls.map { ToolCallRecord(call: $0, output: .ok(["ok": true], summary: "Reminder")) })
            let second = try await EngineTestHarness.collect(engine.continueReply(after: round))
            XCTAssertEqual(EngineTestHarness.text(second), "Reminder set.", "\(tiny): the continuation isn't forced")
            let secondStats = try XCTUnwrap(EngineTestHarness.finish(second)?.stats, "\(tiny)")
            XCTAssertFalse(secondStats.forcedToolCall, "\(tiny)")
            XCTAssertNil(secondStats.toolCallProbability, "\(tiny)")
            XCTAssertEqual(target.usedScripts, 2, "\(tiny)")
            let report = try await engine.withSession { $0.assertConsistent() }
            XCTAssertTrue(report.isConsistent(), "\(tiny): \(report)")
        }
    }

    /// A model that would answer in text (a claim that it is done) still writes a call under
    /// `.required`, to one of the request's tools, and the claim is never shown.
    func testRequiredToolChoiceOverridesText() async throws {
        let (engine, _) = try EngineTestHarness.makeScriptedEngine(
            tiny: .qwen3, scripts: [tokenizer.encodeRaw("Done, your reminder is set.")], toolCallFormat: .xmlFunction)
        var required = request("Remind me about the parcel", tools: deviceTools)
        required.toolChoice = .required
        let events = try await EngineTestHarness.collect(engine.reply(required))
        XCTAssertEqual(EngineTestHarness.text(events), "")
        let named = EngineTestHarness.kinds(events).filter { $0.hasPrefix("toolCallStarted(") && $0 != "toolCallStarted(nil)" }
        XCTAssertEqual(named.count, 1, "\(named)")
        XCTAssertTrue(deviceTools.contains { "toolCallStarted(\($0.name))" == named.first }, "\(named)")
        let stats = try XCTUnwrap(EngineTestHarness.finish(events)?.stats)
        XCTAssertTrue(stats.forcedToolCall)
        XCTAssertLessThan(stats.toolCallProbability ?? 1, 0.01)
        let report = try await engine.withSession { $0.assertConsistent() }
        XCTAssertTrue(report.isConsistent(), "\(report)")
    }

    /// A call the model starts by itself (`.auto`) is also named among the request's tools: the
    /// undeclared `launch` it begins to write can't be.
    func testSampledCallIsNamedAmongTheTools() async throws {
        let (engine, _) = try EngineTestHarness.makeScriptedEngine(
            tiny: .qwen3, scripts: [tokenizer.encodeRaw("<tool_call>\n<function=launch>\n</function>\n</tool_call>")],
            toolCallFormat: .xmlFunction)
        let events = try await EngineTestHarness.collect(engine.reply(request("Launch it", tools: deviceTools)))
        let named = EngineTestHarness.kinds(events).filter { $0.hasPrefix("toolCallStarted(") && $0 != "toolCallStarted(nil)" }
        XCTAssertEqual(named.count, 1, "\(named)")
        XCTAssertTrue(deviceTools.contains { "toolCallStarted(\($0.name))" == named.first }, "\(named)")
        let stats = try XCTUnwrap(EngineTestHarness.finish(events)?.stats)
        XCTAssertFalse(stats.forcedToolCall)
        XCTAssertGreaterThan(stats.toolCallProbability ?? 0, 0.99)
        let report = try await engine.withSession { $0.assertConsistent() }
        XCTAssertTrue(report.isConsistent(), "\(report)")
    }

    /// `.native` leaves the call to the model, as before tool choice: nothing is forced or fed, so
    /// the undeclared `launch` it writes keeps its name and the generator samples every token.
    /// The probability of a call is still measured.
    func testNativeToolChoiceLeavesTheCallToTheModel() async throws {
        let script = "<tool_call>\n<function=launch>\n</function>\n</tool_call>"
        let (engine, _) = try EngineTestHarness.makeScriptedEngine(
            tiny: .hybrid, scripts: [tokenizer.encodeRaw(script)], toolCallFormat: .xmlFunction)
        var native = request("Launch it", tools: deviceTools)
        native.toolChoice = .native
        let events = try await EngineTestHarness.collect(engine.reply(native))
        let kinds = EngineTestHarness.kinds(events)
        XCTAssertEqual(kinds.filter { $0.hasPrefix("toolCallStarted(") && $0 != "toolCallStarted(nil)" }, ["toolCallStarted(launch)"], "\(kinds)")
        let stats = try XCTUnwrap(EngineTestHarness.finish(events)?.stats)
        XCTAssertFalse(stats.forcedToolCall)
        XCTAssertGreaterThan(stats.toolCallProbability ?? 0, 0.99)
        XCTAssertEqual(stats.confidence?.tokens, tokenizer.encodeRaw(script).count + 1, "every token sampled by the generator")
        let report = try await engine.withSession { $0.assertConsistent() }
        XCTAssertTrue(report.isConsistent(), "\(report)")
    }

    /// `.requiredIfLikely` forces no call the model didn't consider: a text reply streams as with
    /// `.auto`, its first token sampled by the generator as before.
    func testLikelyToolChoiceLeavesATextReplyAlone() async throws {
        let (engine, _) = try EngineTestHarness.makeScriptedEngine(
            tiny: .hybrid, scripts: [tokenizer.encodeRaw("It is noon.")], toolCallFormat: .xmlFunction)
        var likely = request("What time is it?", tools: deviceTools)
        likely.toolChoice = .requiredIfLikely(0.05)
        let events = try await EngineTestHarness.collect(engine.reply(likely))
        XCTAssertEqual(EngineTestHarness.text(events), "It is noon.")
        XCTAssertFalse(EngineTestHarness.kinds(events).contains { $0.hasPrefix("toolCallStarted") })
        let stats = try XCTUnwrap(EngineTestHarness.finish(events)?.stats)
        XCTAssertFalse(stats.forcedToolCall)
        XCTAssertLessThan(stats.toolCallProbability ?? 1, 0.05)
        XCTAssertEqual(stats.confidence?.tokens, "It is noon.".count + 1, "every token sampled by the generator")
    }

    /// A first token the engine samples itself (every token is kept, so the start token could
    /// be) and finds to be text reaches the generator as that token: the seeded reply is the one
    /// an engine that never opens calls (no one-token start tag in `.gemma`) samples.
    func testSampledTextFirstTokenIsTheGeneratorsToo() async throws {
        var configuration = EngineTestHarness.testConfiguration(maxTokens: 24)
        configuration.temperature = 1
        configuration.topP = 1
        configuration.topK = 0
        configuration.seed = 3
        for tiny in EngineTestHarness.Tiny.allCases {
            var replies: [[Int]] = []
            for format in [ToolCallFormat.json, .gemma] {
                // Unsharpened, so which key a position draws with shows in the reply.
                let model = try EngineTestHarness.makeModel(tiny, sharpen: false)
                let loaded = try EngineTestHarness.loadedModel(
                    model, tokenizer: tokenizer, id: "tiny-first-token-\(tiny.rawValue)-\(format.rawValue)",
                    modelType: EngineTestHarness.modelType(tiny), toolCallFormat: format)
                let engine = InferenceEngine(loaded: loaded, configuration: configuration)
                let events = try await EngineTestHarness.collect(engine.reply(request("Hello", tools: deviceTools)))
                let stats = try XCTUnwrap(EngineTestHarness.finish(events)?.stats, "\(tiny) \(format)")
                XCTAssertEqual(stats.toolCallProbability == nil, format == .gemma, "\(tiny) \(format)")
                let (ledger, replyStart) = try await engine.withSession { ($0.ledger, $0.snapshot?.turns.first?.replyStart) }
                let start = try XCTUnwrap(replyStart, "\(tiny) \(format)")
                replies.append(Array(ledger[start...]))
            }
            XCTAssertNotEqual(replies[0].first, FakeChatMLTokenizer.toolCallStart, "\(tiny): pick a seed whose first token is text")
            XCTAssertEqual(replies[0], replies[1], "\(tiny)")
        }
    }

    /// A consumer that stops at the call's start (as the loop's early handoff does) leaves the
    /// cache exact wherever the opening got to; the retry is planned as a replaced user turn and
    /// writes the call.
    func testStoppingAtTheCallsStartLeavesTheCacheExact() async throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let call = "<tool_call>\n<function=list_timers>\n<parameter=title>\nall\n</parameter>\n</function>\n</tool_call>"
            let (engine, _) = try EngineTestHarness.makeScriptedEngine(
                tiny: tiny, scripts: [tokenizer.encodeRaw(call)], configuration: EngineTestHarness.testConfiguration(maxTokens: 256),
                toolCallFormat: .xmlFunction)
            let timers = request("How long is left on my timer?", tools: deviceTools)
            for try await event in engine.reply(timers) {
                if case .progress(.toolCallStarted) = event { break }
            }
            await engine.waitUntilIdle()
            let stopped = try await engine.withSession { $0.assertConsistent() }
            XCTAssertTrue(stopped.isConsistent(), "\(tiny): \(stopped)")

            let retried = try await EngineTestHarness.collect(engine.reply(timers))
            XCTAssertEqual(EngineTestHarness.finish(retried)?.stats.planReason, "replaceLastUserTurn", "\(tiny)")
            XCTAssertEqual(EngineTestHarness.toolCalls(retried).map(\.name), ["list_timers"], "\(tiny)")
            let report = try await engine.withSession { $0.assertConsistent() }
            XCTAssertTrue(report.isConsistent(), "\(tiny): \(report)")
        }
    }
}

/// Tools whose names share prefixes, as the on-device set's do.
private var deviceTools: [ToolDefinition] {
    ["create_event", "create_reminder", "list_reminders", "list_timers", "set_timer"].map {
        ToolDefinition(
            name: $0, description: "A device tool.",
            inputSchema: ["type": "object", "properties": ["title": ["type": "string"]], "required": ["title"]])
    }
}

enum EngineHarnessCheck {
    static func reason(_ events: [EngineEvent]) -> String? {
        EngineTestHarness.finish(events)?.stats.planReason
    }
}

/// Counts GPU hook calls and notices a job beginning inside another.
final class HookCounter: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var begins = 0
    private(set) var ends = 0
    private(set) var overlapped = false
    private var open = 0

    func begin() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        begins += 1
        open += 1
        if open > 1 { overlapped = true }
        return true
    }

    func end() {
        lock.lock()
        defer { lock.unlock() }
        ends += 1
        open -= 1
    }

    func check() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return open == 1
    }
}

/// The fake tokenizer's templates without ChatML marker ids.
private struct WithoutTurnMarkers: ChatTemplateRendering {
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

/// `TextStreamer` on its own: visible text, thinking filtered, tool calls found as they start
/// and parsed at the end, cut-off calls never shown. No GPU needed.
final class TextStreamerTests: XCTestCase {
    private let tokenizer = FakeChatMLTokenizer()

    private struct Streamed {
        let outputs: [TextStreamer.Output]
        let text: String
        let toolCalls: [PendingToolCall]
        let streamer: TextStreamer
    }

    private func stream(_ text: String, format: ToolCallFormat = .json, tools: [[String: any Sendable]]? = nil) -> Streamed {
        let streamer = TextStreamer(tokenizer: tokenizer, renderer: tokenizer, format: format, tools: tools)
        var outputs: [TextStreamer.Output] = []
        for token in tokenizer.encodeRaw(text) {
            outputs += streamer.append([token])
        }
        let (tail, calls) = streamer.finish()
        let shown = outputs.compactMap { output -> String? in
            if case .text(let text) = output { return text }
            return nil
        }.joined() + tail
        return Streamed(outputs: outputs, text: shown, toolCalls: calls, streamer: streamer)
    }

    func testThinkingIsFilteredAndTextKept() {
        let result = stream("<think>\nweighing it\n</think>\n\nHello <b> world, 3 < 4.")
        XCTAssertEqual(result.text, "Hello <b> world, 3 < 4.")
        XCTAssertEqual(result.streamer.visibleText, "Hello <b> world, 3 < 4.")
        XCTAssertTrue(result.toolCalls.isEmpty)
        XCTAssertTrue(result.streamer.startedCalls.isEmpty)
    }

    func testJSONToolCallIsReportedAndParsed() {
        let tools = JSONBridge.templateTools([
            ToolDefinition(name: "set_timer", description: "Timer.", inputSchema: ["type": "object", "properties": ["seconds": ["type": "integer"]]]),
        ])
        let result = stream("Sure. <tool_call>\n{\"name\": \"set_timer\", \"arguments\": {\"seconds\": 5}}\n</tool_call>", tools: tools)
        XCTAssertEqual(result.text, "Sure. ")
        let progress = result.outputs.filter { output in
            if case .toolCallStarted = output { return true }
            return false
        }
        XCTAssertEqual(progress, [.toolCallStarted(nil), .toolCallStarted("set_timer")])
        XCTAssertEqual(result.toolCalls.map(\.name), ["set_timer"])
        XCTAssertEqual(result.toolCalls.first?.input, ["seconds": 5])
        XCTAssertNil(result.toolCalls.first?.rawInput)
        XCTAssertEqual(result.streamer.startedCalls, ["set_timer"])
        XCTAssertFalse(result.streamer.insideToolCall)
    }

    func testXMLFunctionCallNameAndArguments() {
        let tools = JSONBridge.templateTools([
            ToolDefinition(
                name: "create_reminder", description: "Reminder.",
                inputSchema: ["type": "object", "properties": ["title": ["type": "string"], "minutes": ["type": "integer"]]]),
        ])
        let call = "<tool_call>\n<function=create_reminder>\n<parameter=title>\nCall mum\n</parameter>\n"
            + "<parameter=minutes>\n30\n</parameter>\n</function>\n</tool_call>"
        let result = stream(call, format: .xmlFunction, tools: tools)
        XCTAssertEqual(result.text, "")
        XCTAssertTrue(result.outputs.contains(.toolCallStarted("create_reminder")))
        XCTAssertEqual(result.toolCalls.first?.name, "create_reminder")
        XCTAssertEqual(result.toolCalls.first?.input, ["title": "Call mum", "minutes": 30])
    }

    /// The processor hands a closed call it can't parse back as text; the streamer takes the
    /// markup out and reports the call with the model's text as `rawInput`.
    func testUnparsedCallIsReportedWithItsRawInput() {
        let tools = JSONBridge.templateTools([
            ToolDefinition(name: "set_timer", description: "Timer.", inputSchema: ["type": "object", "properties": ["seconds": ["type": "integer"]]]),
        ])
        let payload = "{\"name\": \"set_timer\", \"arguments\": {\"seconds\": }}"
        let result = stream("Sure. <tool_call>\n" + payload + "\n</tool_call>", tools: tools)
        XCTAssertEqual(result.text, "Sure. ")
        XCTAssertEqual(result.streamer.visibleText, "Sure. ")
        XCTAssertEqual(result.toolCalls.count, 1)
        XCTAssertEqual(result.toolCalls.first?.name, "set_timer")
        XCTAssertNil(result.toolCalls.first?.input)
        XCTAssertEqual(result.toolCalls.first?.rawInput, payload)
        XCTAssertTrue(result.toolCalls.first?.id.hasPrefix("call_") ?? false)
        XCTAssertEqual(result.streamer.startedCalls, ["set_timer"])
        XCTAssertFalse(result.streamer.insideToolCall)
    }

    /// The JSON parser refuses a tool the request didn't declare; the call still isn't shown,
    /// the text around it is, and calls keep the order they were written in.
    func testCallToAnUndeclaredToolIsReportedNotShown() {
        let tools = JSONBridge.templateTools([
            ToolDefinition(name: "get_time", description: "Time.", inputSchema: ["type": "object", "properties": [:]]),
        ])
        let result = stream(
            "One: <tool_call>\n{\"name\": \"launch\", \"arguments\": {}}\n</tool_call> then "
                + "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {}}\n</tool_call>", tools: tools)
        XCTAssertEqual(result.text, "One:  then ")
        XCTAssertEqual(result.toolCalls.map(\.name), ["launch", "get_time"])
        XCTAssertNil(result.toolCalls.first?.input)
        XCTAssertEqual(result.toolCalls.first?.rawInput, "{\"name\": \"launch\", \"arguments\": {}}")
        XCTAssertEqual(result.toolCalls.last?.input, [:])
        XCTAssertNil(result.toolCalls.last?.rawInput)
    }

    func testCutOffCallIsNeverShown() {
        let result = stream("<tool_call>\n{\"name\": \"set_ti")
        XCTAssertEqual(result.text, "")
        XCTAssertTrue(result.toolCalls.isEmpty)
        XCTAssertEqual(result.streamer.startedCalls, [nil], "the start was seen, the name never completed")
    }

    func testInsideToolCallFollowsTheTags() {
        let streamer = TextStreamer(tokenizer: tokenizer, renderer: tokenizer, format: .json, tools: nil)
        _ = streamer.append(tokenizer.encodeRaw("Ok "))
        XCTAssertFalse(streamer.insideToolCall)
        _ = streamer.append(tokenizer.encodeRaw("<tool_call>\n{\"name\""))
        XCTAssertTrue(streamer.insideToolCall)
        _ = streamer.append(tokenizer.encodeRaw(": \"x\", \"arguments\": {}}\n</tool_call>"))
        XCTAssertFalse(streamer.insideToolCall)
    }

    // MARK: A tool call's opening (the engine's selector)

    func testToolChoiceForcesByItsFloor() {
        XCTAssertFalse(ToolChoice.auto.forcesCall(probability: 1))
        XCTAssertFalse(ToolChoice.native.forcesCall(probability: 1))
        XCTAssertTrue(ToolChoice.required.forcesCall(probability: 0))
        XCTAssertTrue(ToolChoice.requiredIfLikely(0.05).forcesCall(probability: 0.05))
        XCTAssertFalse(ToolChoice.requiredIfLikely(0.05).forcesCall(probability: 0.049))
    }

    /// The openings stop right after the function's name in either format, and start with the
    /// start token; a format without skeletons opens with the start token alone.
    func testCallOpeningsEndAfterTheName() {
        let start = FakeChatMLTokenizer.toolCallStart
        let xml = InferenceEngine.callOpenings(tools: deviceTools, format: .xmlFunction, start: start, renderer: tokenizer)
        XCTAssertEqual(xml.map(tokenizer.decodeRaw), deviceTools.map { "<tool_call>\n<function=\($0.name)>\n" })
        XCTAssertTrue(xml.allSatisfy { $0.first == start })
        let json = InferenceEngine.callOpenings(tools: deviceTools, format: .json, start: start, renderer: tokenizer)
        XCTAssertEqual(json.map(tokenizer.decodeRaw), deviceTools.map { "<tool_call>\n{\"name\": \"\($0.name)\", \"arguments\":" })
        XCTAssertEqual(InferenceEngine.callOpenings(tools: deviceTools, format: .gemma, start: start, renderer: tokenizer), [[start]])
    }

    /// The selector takes what the remaining openings share and asks only where they branch,
    /// among the tokens that continue one of them; it ends with a whole opening.
    func testSelectorAsksOnlyWhereTheNamesBranch() {
        let openings = InferenceEngine.callOpenings(
            tools: deviceTools, format: .xmlFunction, start: FakeChatMLTokenizer.toolCallStart, renderer: tokenizer)
        let expected = [
            "create_event": ["cls", "er"], "create_reminder": ["cls", "er"], "list_reminders": ["cls", "rt"],
            "list_timers": ["cls", "rt"], "set_timer": ["cls"],
        ]
        for (target, tool) in zip(openings, deviceTools) {
            var asked: [String] = []
            let chosen = InferenceEngine.selectOpening(openings) { prefix, allowed in
                XCTAssertEqual(prefix, Array(target.prefix(prefix.count)), tool.name)
                asked.append(tokenizer.decodeRaw(allowed))
                return target[prefix.count]
            }
            XCTAssertEqual(chosen, target, tool.name)
            XCTAssertEqual(asked, expected[tool.name], tool.name)
        }
    }

    /// The engine streams the opening it feeds like emitted tokens: the call starts at the start
    /// token, its name is reported once the opening is in (before any argument), and the
    /// arguments the model writes after it complete a call that parses.
    func testStreamedOpeningThenArgumentsParse() {
        let openings = InferenceEngine.callOpenings(
            tools: deviceTools, format: .xmlFunction, start: FakeChatMLTokenizer.toolCallStart, renderer: tokenizer)
        let opening = InferenceEngine.selectOpening(openings) { _, allowed in allowed[allowed.count - 1] }
        let streamer = TextStreamer(tokenizer: tokenizer, renderer: tokenizer, format: .xmlFunction, tools: JSONBridge.templateTools(deviceTools))
        XCTAssertEqual(streamer.append(Array(opening.prefix(1))), [.toolCallStarted(nil)])
        XCTAssertEqual(streamer.append(Array(opening.dropFirst())), [.toolCallStarted("set_timer")])
        XCTAssertTrue(streamer.insideToolCall)
        XCTAssertEqual(streamer.append(tokenizer.encodeRaw("<parameter=title>\nTea\n</parameter>\n</function>\n</tool_call>")), [])
        let (tail, calls) = streamer.finish()
        XCTAssertEqual(tail, "")
        XCTAssertEqual(calls.map(\.name), ["set_timer"])
        XCTAssertEqual(calls.first?.input, ["title": "Tea"])
    }
}
