import AssistantKit
import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLMCommon
import XCTest

/// A scripted conversation through the engine on the tiny hybrid and the tiny Qwen3: every
/// planner rule runs, the plan reasons and reused counts are the expected ones, and after every
/// reply the cache equals a fresh prefill of the ledger (`assertConsistent`).
final class SessionRuntimeTests: XCTestCase {
    private let tokenizer = FakeChatMLTokenizer()
    private let system = "You are Alvin, a concise assistant."

    override func setUpWithError() throws {
        try MetalAvailability.require()
    }

    private struct Reply {
        let text: String
        let finish: EngineFinish
    }

    private func ask(_ engine: InferenceEngine, _ request: EngineRequest, file: StaticString = #filePath, line: UInt = #line) async throws -> Reply {
        let events = try await EngineTestHarness.collect(engine.reply(request))
        let finish = try XCTUnwrap(EngineTestHarness.finish(events), file: file, line: line)
        let report = try await engine.withSession { $0.assertConsistent() }
        XCTAssertTrue(report.isConsistent(), "after \(finish.stats.planReason ?? "?"): \(report)", file: file, line: line)
        return Reply(text: EngineTestHarness.text(events), finish: finish)
    }

    private func ledgerCount(_ engine: InferenceEngine) async throws -> Int {
        try await engine.withSession { $0.ledger.count }
    }

    private func systemEnd(_ engine: InferenceEngine) async throws -> Int {
        try await engine.withSession { $0.snapshot?.systemEnd ?? -1 }
    }

    func testScriptedConversationOnTheHybrid() async throws {
        try await runConversation(.hybrid)
    }

    func testScriptedConversationOnQwen3() async throws {
        try await runConversation(.qwen3)
    }

    private func runConversation(_ tiny: EngineTestHarness.Tiny) async throws {
        let replies = (1 ... 12).map { "Answer number \($0), all good." }
        let (engine, _) = try EngineTestHarness.makeScriptedEngine(tiny: tiny, scripts: replies.map(tokenizer.encodeRaw))
        var turns: [ChatTurn] = []
        var rows: [[String]] = []
        func note(_ step: String, _ reply: Reply) {
            let stats = reply.finish.stats
            rows.append([step, stats.planReason ?? "–", "\(stats.prefilledTokens ?? -1)", "\(stats.reusedTokens ?? -1)", stats.phases?.start ?? "–"])
        }

        // 1. A new session.
        turns.append(ChatTurn(role: .user, text: "Hello there.", context: "[09:00]"))
        var reply = try await ask(engine, EngineRequest(system: system, turns: turns))
        XCTAssertEqual(reply.text, replies[0])
        XCTAssertEqual(reply.finish.reason, .stop)
        XCTAssertEqual(reply.finish.stats.planReason, "newSession")
        XCTAssertEqual(reply.finish.stats.reusedTokens, 0)
        note("first turn", reply)
        turns.append(ChatTurn(role: .assistant, text: reply.text))

        // 2. Append: only the new turn is prefilled.
        var before = try await ledgerCount(engine)
        turns.append(ChatTurn(role: .user, text: "And then?"))
        reply = try await ask(engine, EngineRequest(system: system, turns: turns))
        XCTAssertEqual(reply.text, replies[1])
        XCTAssertEqual(reply.finish.stats.planReason, "append")
        XCTAssertEqual(reply.finish.stats.reusedTokens, before)
        XCTAssertTrue(reply.finish.stats.reusedSession)
        note("append", reply)
        turns.append(ChatTurn(role: .assistant, text: reply.text))

        // 3. An intervening turn answered elsewhere (two new turns before the new user turn).
        before = try await ledgerCount(engine)
        turns += [
            ChatTurn(role: .user, text: "What's the news?"),
            ChatTurn(role: .assistant, text: "Here's what the cloud said."),
            ChatTurn(role: .user, text: "Thanks, and locally?"),
        ]
        reply = try await ask(engine, EngineRequest(system: system, turns: turns))
        XCTAssertEqual(reply.text, replies[2])
        XCTAssertEqual(reply.finish.stats.planReason, "append")
        XCTAssertEqual(reply.finish.stats.reusedTokens, before)
        note("intervening turn", reply)
        turns.append(ChatTurn(role: .assistant, text: reply.text))

        // 4. A barge-in: the reply is cut short and only its spoken prefix is stored.
        turns.append(ChatTurn(role: .user, text: "Tell me a long story."))
        reply = try await ask(engine, EngineRequest(system: system, turns: turns, maxTokens: 6))
        XCTAssertEqual(reply.finish.reason, .length)
        XCTAssertEqual(reply.text, String(replies[3].prefix(6)))
        let spoken = String(reply.text.prefix(3))
        turns.append(ChatTurn(role: .assistant, text: spoken))
        before = try await ledgerCount(engine)
        turns.append(ChatTurn(role: .user, text: "Sorry, go on."))
        reply = try await ask(engine, EngineRequest(system: system, turns: turns))
        XCTAssertEqual(reply.text, replies[4])
        XCTAssertEqual(reply.finish.stats.planReason, "replaceLastReply")
        XCTAssertLessThan(reply.finish.stats.reusedTokens ?? .max, before, "rewound to the cut reply's start")
        XCTAssertGreaterThan(reply.finish.stats.reusedTokens ?? 0, 0)
        note("barge-in, spoken prefix stored", reply)
        turns.append(ChatTurn(role: .assistant, text: reply.text))

        // 5. The last user turn replaced (a tentative turn, then the committed one).
        let committed = Array(turns.dropLast(2)) + [ChatTurn(role: .user, text: "Sorry, please go on.")]
        reply = try await ask(engine, EngineRequest(system: system, turns: committed))
        XCTAssertEqual(reply.text, replies[5])
        XCTAssertEqual(reply.finish.stats.planReason, "replaceLastUserTurn")
        XCTAssertGreaterThan(reply.finish.stats.reusedTokens ?? 0, 0)
        note("replaced last user turn", reply)
        turns = committed + [ChatTurn(role: .assistant, text: reply.text)]

        // 6. Another conversation: back to the system prefix.
        let systemEnd = try await systemEnd(engine)
        XCTAssertGreaterThan(systemEnd, 0)
        reply = try await ask(engine, EngineRequest(system: system, turns: [ChatTurn(role: .user, text: "New chat, hi.")]))
        XCTAssertEqual(reply.text, replies[6])
        XCTAssertEqual(reply.finish.stats.planReason, "diverged")
        XCTAssertEqual(reply.finish.stats.reusedTokens, systemEnd)
        note("another conversation", reply)

        // 7. A different system prompt: nothing can be reused.
        reply = try await ask(engine, EngineRequest(system: "You are terse.", turns: [ChatTurn(role: .user, text: "Hi.")]))
        XCTAssertEqual(reply.text, replies[7])
        XCTAssertEqual(reply.finish.stats.planReason, "prefixChanged")
        XCTAssertEqual(reply.finish.stats.reusedTokens, 0)
        note("system change", reply)
        var short = [ChatTurn(role: .user, text: "Hi."), ChatTurn(role: .assistant, text: reply.text)]

        // 8. Over budget: rebuilt from the window on the system prefix.
        let count = try await ledgerCount(engine)
        await engine.updateConfiguration { $0.limits = SessionPlanner.Limits(maxTokens: count + 20, keepTurns: 12) }
        short.append(ChatTurn(role: .user, text: String(repeating: "A long question. ", count: 6)))
        let terseEnd = try await self.systemEnd(engine)
        reply = try await ask(engine, EngineRequest(system: "You are terse.", turns: short))
        XCTAssertEqual(reply.text, replies[8])
        XCTAssertEqual(reply.finish.stats.planReason, "overBudget")
        XCTAssertEqual(reply.finish.stats.reusedTokens, terseEnd)
        note("over budget", reply)

        EngineReport.appendTable(
            title: "Session reuse, scripted conversation (tiny \(tiny.rawValue))",
            header: ["Step", "Plan", "Prefilled", "Reused", "Start"], rows: rows)
    }

    /// Checkpoints follow the plan: `systemEnd`, the newest user turn's start and `replyStart`,
    /// within the budget.
    func testCheckpointsAfterAReply() async throws {
        let (engine, _) = try EngineTestHarness.makeScriptedEngine(tiny: .hybrid, scripts: [tokenizer.encodeRaw("Fine.")])
        _ = try await ask(engine, EngineRequest(system: system, turns: [ChatTurn(role: .user, text: "How are you?")]))
        let (marks, snapshot, ledger) = try await engine.withSession { ($0.checkpoints.marks, $0.snapshot, $0.ledger) }
        let snap = try XCTUnwrap(snapshot)
        XCTAssertEqual(marks[.systemEnd], snap.systemEnd)
        let user = try XCTUnwrap(snap.turns.first)
        XCTAssertEqual(marks[.lastUserStart], user.start)
        XCTAssertEqual(marks[.replyStart], user.replyStart)
        XCTAssertEqual(ledger[try XCTUnwrap(user.start)], FakeChatMLTokenizer.imStart)
        XCTAssertEqual(snap.turns.count, 2, "the user turn and the recorded reply")
        XCTAssertEqual(snap.turns.last?.turn.text, "Fine.")
        XCTAssertEqual(snap.tokenCount, ledger.count)
        XCTAssertEqual(ledger.last, FakeChatMLTokenizer.imEnd, "the stop token is in the cache")

        // A budget of zero keeps only systemEnd.
        await engine.updateConfiguration { $0.checkpointBudgetBytes = 0 }
        let kept = try await engine.withSession { $0.checkpoints.marks }
        XCTAssertEqual(Array(kept.keys), [.systemEnd])
    }
}
