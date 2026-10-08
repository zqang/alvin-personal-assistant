import AssistantKit
import Foundation
import LocalEngine
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
        do {
            _ = try await EngineTestHarness.collect(engine.continueReply(after: round))
            XCTFail("continueReply without pending calls should throw")
        } catch {
            XCTAssertEqual(error as? EngineError, .busy)
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
}
