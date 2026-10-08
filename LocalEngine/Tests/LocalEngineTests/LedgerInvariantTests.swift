import AssistantKit
import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLMCommon
import XCTest

/// Invariant L1 (the cache holds exactly the ledger) after every way a reply can end: a stop,
/// a terminated stream, the length limit and a tool call with its continuation; on both tiny
/// models.
final class LedgerInvariantTests: XCTestCase {
    private let tokenizer = FakeChatMLTokenizer()
    private let system = "You are Alvin."

    override func setUpWithError() throws {
        try MetalAvailability.require()
    }

    private func assertConsistent(_ engine: InferenceEngine, _ label: String, file: StaticString = #filePath, line: UInt = #line) async throws {
        let (report, pending) = try await engine.withSession { ($0.assertConsistent(), $0.pendingCount) }
        XCTAssertEqual(pending, 0, label, file: file, line: line)
        XCTAssertTrue(report.isConsistent(), "\(label): \(report)", file: file, line: line)
    }

    func testAfterStop() async throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let (engine, _) = try EngineTestHarness.makeScriptedEngine(tiny: tiny, scripts: [tokenizer.encodeRaw("All done.")])
            let events = try await EngineTestHarness.collect(
                engine.reply(EngineRequest(system: system, turns: [ChatTurn(role: .user, text: "Finish up.")])))
            XCTAssertEqual(EngineTestHarness.finish(events)?.reason, .stop)
            let last = try await engine.withSession { $0.ledger.last }
            XCTAssertEqual(last, FakeChatMLTokenizer.imEnd, "\(tiny): the stop token is fed")
            try await assertConsistent(engine, "\(tiny) stop")
        }
    }

    func testAfterLength() async throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let reply = tokenizer.encodeRaw("This reply is far longer than the limit allows.")
            let (engine, _) = try EngineTestHarness.makeScriptedEngine(tiny: tiny, scripts: [reply])
            let events = try await EngineTestHarness.collect(
                engine.reply(EngineRequest(system: system, turns: [ChatTurn(role: .user, text: "Talk.")], maxTokens: 9)))
            let finish = try XCTUnwrap(EngineTestHarness.finish(events))
            XCTAssertEqual(finish.reason, .length)
            XCTAssertEqual(finish.stats.generatedTokens, 9)
            let tail = try await engine.withSession { Array($0.ledger.suffix(9)) }
            XCTAssertEqual(tail, Array(reply.prefix(9)), "\(tiny): the ledger ends with the last emitted token")
            try await assertConsistent(engine, "\(tiny) length")
        }
    }

    /// Terminating the stream mid-reply stops the engine at a step boundary; whatever it had
    /// decoded by then is exactly in the cache, and the next turn reuses it consistently.
    func testAfterTerminatingTheStream() async throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let long = tokenizer.encodeRaw(String(repeating: "word ", count: 12))
            let (engine, _) = try EngineTestHarness.makeScriptedEngine(
                tiny: tiny, scripts: [long, tokenizer.encodeRaw("Next.")])
            var seen = ""
            var turns = [ChatTurn(role: .user, text: "Say many words.")]
            for try await event in engine.reply(EngineRequest(system: system, turns: turns)) {
                if case .text(let text) = event {
                    seen += text
                    break
                }
            }
            await engine.waitUntilIdle()
            try await assertConsistent(engine, "\(tiny) terminated")

            turns += [ChatTurn(role: .assistant, text: seen), ChatTurn(role: .user, text: "Go on.")]
            let events = try await EngineTestHarness.collect(engine.reply(EngineRequest(system: system, turns: turns)))
            XCTAssertEqual(EngineTestHarness.text(events), "Next.")
            try await assertConsistent(engine, "\(tiny) after the terminated reply")
        }
    }

    /// The hooks can stop a reply mid-stream: `.cancelled`, nothing pending, an exact ledger.
    func testAfterCancellationByTheHooks() async throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let gate = CallBudget(calls: 10)
            var configuration = EngineTestHarness.testConfiguration()
            configuration.hooks = EngineHooks(beginGPU: { true }, endGPU: {}, isAllowed: { gate.take() })
            let script = tokenizer.encodeRaw(String(repeating: "more ", count: 10))
            let (engine, _) = try EngineTestHarness.makeScriptedEngine(tiny: tiny, scripts: [script], configuration: configuration)
            let events = try await EngineTestHarness.collect(
                engine.reply(EngineRequest(system: system, turns: [ChatTurn(role: .user, text: "Go.")])))
            let finish = try XCTUnwrap(EngineTestHarness.finish(events))
            XCTAssertEqual(finish.reason, .cancelled, "\(tiny)")
            XCTAssertGreaterThan(finish.stats.generatedTokens, 0, "\(tiny): the prefill took at most a few calls")
            XCTAssertLessThan(finish.stats.generatedTokens, script.count, "\(tiny)")
            gate.reset(calls: 1_000)
            let tail = try await engine.withSession { Array($0.ledger.suffix(finish.stats.generatedTokens)) }
            XCTAssertEqual(tail, Array(script.prefix(finish.stats.generatedTokens)), "\(tiny)")
            try await assertConsistent(engine, "\(tiny) cancelled")
        }
    }

    /// A prefill interrupted part-way leaves the cached conversation reusable: the retried
    /// request still appends, after rewinding the partly fed tokens.
    func testInterruptedPrefillKeepsTheSessionReusable() async throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let gate = CallBudget(calls: 1_000)
            var configuration = EngineTestHarness.testConfiguration()
            configuration.hooks = EngineHooks(beginGPU: { true }, endGPU: {}, isAllowed: { gate.take() })
            let (engine, _) = try EngineTestHarness.makeScriptedEngine(
                tiny: tiny, scripts: ["First.", "Second."].map(tokenizer.encodeRaw), configuration: configuration)
            var turns = [ChatTurn(role: .user, text: "One?")]
            let first = try await EngineTestHarness.collect(engine.reply(EngineRequest(system: system, turns: turns)))
            turns += [ChatTurn(role: .assistant, text: EngineTestHarness.text(first)), ChatTurn(role: .user, text: "Two?")]
            let cached = try await engine.withSession { $0.ledger.count }

            // One prefill chunk is allowed, then the GPU is refused.
            gate.reset(calls: 1)
            let interrupted = try await EngineTestHarness.collect(engine.reply(EngineRequest(system: system, turns: turns)))
            XCTAssertEqual(EngineTestHarness.finish(interrupted)?.reason, .cancelled, "\(tiny)")
            let fed = try await engine.withSession { $0.ledger.count }
            XCTAssertGreaterThan(fed, cached, "\(tiny): part of the new turn was fed")
            try await assertConsistent(engine, "\(tiny) interrupted")

            gate.reset(calls: 1_000)
            let retried = try await EngineTestHarness.collect(engine.reply(EngineRequest(system: system, turns: turns)))
            let finish = try XCTUnwrap(EngineTestHarness.finish(retried))
            XCTAssertEqual(finish.stats.planReason, "append", "\(tiny)")
            XCTAssertEqual(finish.stats.reusedTokens, cached, "\(tiny)")
            XCTAssertEqual(EngineTestHarness.text(retried), "Second.")
            try await assertConsistent(engine, "\(tiny) retried")
        }
    }

    /// A generator that throws with a token still pending (fed, not yet in the ledger) ends the
    /// reply with its error and leaves an empty, exact session: warm-up, the consistency check
    /// and the next reply all work.
    func testAfterAGeneratorThrowsWithATokenPending() async throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            var configuration = EngineTestHarness.testConfiguration()
            configuration.generatorFactory = PendingThenThrowFactory()
            let (engine, _) = try EngineTestHarness.makeScriptedEngine(
                tiny: tiny, scripts: [tokenizer.encodeRaw("Fine.")], configuration: configuration)
            let request = EngineRequest(system: system, turns: [ChatTurn(role: .user, text: "How are you?")])
            do {
                _ = try await EngineTestHarness.collect(engine.reply(request))
                XCTFail("\(tiny): the generator's error should end the stream")
            } catch {
                XCTAssertTrue(error is GeneratorFailure, "\(tiny): \(error)")
            }
            let state = try await engine.withSession { ($0.pendingCount, $0.ledger.count, $0.snapshot == nil) }
            XCTAssertEqual(state.0, 0, "\(tiny)")
            XCTAssertEqual(state.1, 0, "\(tiny): the session starts over")
            XCTAssertTrue(state.2, "\(tiny)")
            try await engine.warmUp()
            try await assertConsistent(engine, "\(tiny) after the failed generator")

            await engine.updateConfiguration { $0.generatorFactory = nil }
            let events = try await EngineTestHarness.collect(engine.reply(request))
            XCTAssertEqual(EngineTestHarness.text(events), "Fine.", "\(tiny)")
            XCTAssertEqual(EngineTestHarness.finish(events)?.stats.planReason, "newSession", "\(tiny)")
            try await assertConsistent(engine, "\(tiny) after the next reply")
        }
    }

    /// Scratch work and the consistency check on a cache that can't be rewound exactly
    /// (rotating layers) rebuild it from the ledger, so no probe token stays behind.
    func testScratchWorkOnACacheThatCantBeRewound() async throws {
        let model = try EngineTestHarness.makeModel(.qwen3)
        let session = LiveSession(target: RotatingCacheTarget(model: model))
        XCTAssertFalse(session.reusable)
        XCTAssertTrue(session.noReuse)
        let tokens = tokenizer.encodeRaw("Hello there.")
        session.feed(tokens, rows: .none)
        session.withScratch {
            session.feed([20, 21, 22, 23], rows: .all)
            session.feed([24], rows: .last)
        }
        XCTAssertEqual(session.ledger, tokens)
        XCTAssertEqual(session.target.cache.map(\.offset), Array(repeating: tokens.count, count: session.target.cache.count))
        let report = session.assertConsistent()
        XCTAssertTrue(report.offsetsMatch, "\(report)")
        XCTAssertTrue(report.isConsistent(), "\(report)")
        XCTAssertEqual(session.ledger, tokens)
        XCTAssertEqual(session.target.cache.map(\.offset), Array(repeating: tokens.count, count: session.target.cache.count))

        // The engine warms up on such a target without leaving its probe tokens in the cache,
        // and answers by rendering everything.
        let loaded = try EngineTestHarness.loadedModel(model, id: "tiny-rotating", modelType: "qwen3")
        let engine = InferenceEngine(
            loaded: loaded, target: RotatingCacheTarget(model: model), configuration: EngineTestHarness.testConfiguration(maxTokens: 6))
        try await engine.warmUp()
        let offsets = try await engine.withSession { $0.target.cache.map(\.offset) }
        XCTAssertTrue(offsets.allSatisfy { $0 == 0 }, "\(offsets)")
        let events = try await EngineTestHarness.collect(
            engine.reply(EngineRequest(system: system, turns: [ChatTurn(role: .user, text: "Hi.")])))
        XCTAssertEqual(EngineTestHarness.finish(events)?.stats.planReason, "noReuse")
        try await assertConsistent(engine, "rotating cache after a reply")
    }

    /// A scripted tool call, its results and the final answer.
    func testAfterAToolCallAndItsContinuation() async throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let call = "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {}}\n</tool_call>"
            let (engine, _) = try EngineTestHarness.makeScriptedEngine(
                tiny: tiny, scripts: [tokenizer.encodeRaw(call), tokenizer.encodeRaw("It is noon.")])
            let tools = [ToolDefinition(name: "get_time", description: "The current time.", inputSchema: ["type": "object", "properties": [:]])]
            let request = EngineRequest(system: system, tools: tools, turns: [ChatTurn(role: .user, text: "What time is it?")])
            let first = try await EngineTestHarness.collect(engine.reply(request))
            XCTAssertEqual(EngineTestHarness.finish(first)?.reason, .toolCalls, "\(tiny)")
            let calls = EngineTestHarness.toolCalls(first)
            XCTAssertEqual(calls.map(\.name), ["get_time"])
            try await assertConsistent(engine, "\(tiny) tool call")

            let round = ToolRound(calls: calls.map { ToolCallRecord(call: $0, output: .ok(["time": "12:00"], summary: "12:00")) })
            let second = try await EngineTestHarness.collect(engine.continueReply(after: round))
            XCTAssertEqual(EngineTestHarness.text(second), "It is noon.")
            XCTAssertEqual(EngineTestHarness.finish(second)?.reason, .stop)
            try await assertConsistent(engine, "\(tiny) after the tool round")

            // The stored turn (rounds plus the final text) extends the cache on the next request.
            let turns = request.turns + [
                ChatTurn(role: .assistant, text: "It is noon.", toolRounds: [round]),
                ChatTurn(role: .user, text: "Thanks."),
            ]
            let before = try await engine.withSession { $0.ledger.count }
            let third = try await EngineTestHarness.collect(engine.reply(EngineRequest(system: system, tools: tools, turns: turns)))
            let finish = try XCTUnwrap(EngineTestHarness.finish(third))
            XCTAssertEqual(finish.stats.planReason, "append", "\(tiny)")
            XCTAssertEqual(finish.stats.reusedTokens, before, "\(tiny)")
            try await assertConsistent(engine, "\(tiny) after the next turn")
        }
    }
}

struct GeneratorFailure: Error {}

/// Makes generators that feed one sampled token lazily and throw before it is resolved, as a
/// failing speculative generator could.
struct PendingThenThrowFactory: GeneratorFactory {
    func makeGenerator(_ context: GeneratorContext) -> TokenGenerator {
        PendingThenThrow(context)
    }
}

final class PendingThenThrow: TokenGenerator {
    private let context: GeneratorContext

    init(_ context: GeneratorContext) {
        self.context = context
    }

    func step() throws -> (emitted: [Int], finished: EngineFinish.Reason?) {
        let position = context.session.cachedCount
        let (token, _) = context.sampler.sample(context.firstLogits, positions: position ..< (position + 1))
        context.session.feed(token, count: 1, rows: .last)
        throw GeneratorFailure()
    }

    func flush() throws {}

    var speculation: SpeculationStats? { nil }
}

/// A target over a tiny Qwen3 whose layers cache in `RotatingKVCache`s, which can't be rewound
/// exactly (`CacheLayout.detect` refuses them).
final class RotatingCacheTarget: TargetModel {
    let model: any LanguageModel
    private(set) var cache: [KVCache]
    let layout = CacheLayout(attention: [], recurrent: [])
    var vocabularySize: Int { TinyModels.vocabularySize }
    var supportsRollback: Bool { false }
    var supportsRowSelection: Bool { false }

    init(model: any LanguageModel) {
        self.model = model
        self.cache = Self.emptyCache(model)
    }

    private static func emptyCache(_ model: any LanguageModel) -> [KVCache] {
        model.newCache(parameters: nil).map { _ in RotatingKVCache(maxSize: 512) }
    }

    func forward(_ tokens: MLXArray, rows: LogitRows, captureForRollback: Bool, wantHidden: Bool) -> ForwardResult {
        let logits = model(tokens.reshaped([1, -1]), cache: cache)
        switch rows {
        case .none:
            return ForwardResult(logits: nil)
        case .all:
            return ForwardResult(logits: logits.reshaped(-1, logits.dim(-1)))
        case .last:
            return ForwardResult(logits: logits[0..., logits.dim(1) - 1, 0...].reshaped(1, logits.dim(-1)))
        }
    }

    func commit(_ capture: (any RoundCapture)?, keep: Int, of verified: Int) {
        precondition(keep == verified, "A rotating cache can't drop verified tokens.")
    }

    func resetCache() {
        cache = Self.emptyCache(model)
    }

    func adopt(_ cache: [KVCache]) throws {
        throw TargetCacheError.layoutMismatch(expected: layout, found: CacheLayout.detect(cache))
    }
}

/// Allows a fixed number of calls, then refuses.
final class CallBudget: @unchecked Sendable {
    private let lock = NSLock()
    private var remaining: Int

    init(calls: Int) {
        remaining = calls
    }

    func take() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard remaining > 0 else { return false }
        remaining -= 1
        return true
    }

    func reset(calls: Int) {
        lock.lock()
        remaining = calls
        lock.unlock()
    }
}
