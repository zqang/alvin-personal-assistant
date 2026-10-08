import AssistantKit
import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

/// Session reuse on real models: five scripted turns including a barge-in. Every continued turn
/// reuses cached tokens, and every reply passes the teacher-forced check against a fresh
/// rebuild of its context (plan §6.4: same greedy token at each position, up to near-ties).
final class SessionReuseIntegrationTests: XCTestCase {
    private static let system = """
        You are Alvin, a personal voice assistant. Answer in one or two short sentences. \
        Use plain words, no lists or markdown.
        """

    override func setUpWithError() throws {
        try IntegrationEnvironment.requireEnabled()
        try MetalAvailability.require()
    }

    func testSessionReuseOnQwen3_0_6B() async throws {
        try await runConversation("mlx-community/Qwen3-0.6B-4bit")
    }

    func testSessionReuseOnQwen35_0_8B() async throws {
        try await runConversation("mlx-community/Qwen3.5-0.8B-MLX-4bit")
    }

    func testSessionReuseOnWoof() async throws {
        guard IntegrationEnvironment.woof else {
            throw XCTSkip("Woof runs only with LOCAL_ENGINE_WOOF=1 ([ci woof]).")
        }
        try await runConversation(IntegrationEnvironment.woofRepo)
    }

    private struct Step {
        let user: String
        var maxTokens: Int? = nil
        /// Store only the first words of the reply (a barge-in).
        var spokenWords: Int? = nil
        let expectedPlan: String
    }

    private func runConversation(_ repo: String) async throws {
        let directory = try await IntegrationEnvironment.snapshot(repo)
        var configuration = EngineConfiguration()
        configuration.temperature = 0
        configuration.maxTokens = 48
        let engine = try await InferenceEngine.load(directory: directory, modelID: repo, configuration: configuration)
        try await engine.warmUp()

        let steps = [
            Step(user: "Hi! Who are you? Answer in one sentence.", expectedPlan: "newSession"),
            Step(user: "Name three fruits.", expectedPlan: "append"),
            Step(user: "Tell me a long story about a clever fox.", maxTokens: 14, spokenWords: 4, expectedPlan: "append"),
            Step(user: "Sorry, make it one short sentence instead.", expectedPlan: "replaceLastReply"),
            Step(user: "Thanks. What was the first thing I asked you?", expectedPlan: "append"),
        ]
        var turns: [ChatTurn] = []
        var rows: [[String]] = []
        var tally = NearTieTally(tolerance: NearTieTally.quantizedTolerance)
        for (index, step) in steps.enumerated() {
            turns.append(ChatTurn(role: .user, text: step.user))
            let request = EngineRequest(system: Self.system, turns: turns, maxTokens: step.maxTokens)
            let events = try await EngineTestHarness.collect(engine.reply(request))
            let finish = try XCTUnwrap(EngineTestHarness.finish(events), "turn \(index + 1)")
            let text = EngineTestHarness.text(events)
            let stats = finish.stats
            XCTAssertEqual(stats.planReason, step.expectedPlan, "\(repo) turn \(index + 1)")
            if index > 0 {
                XCTAssertGreaterThan(stats.reusedTokens ?? 0, 0, "\(repo) turn \(index + 1) reused nothing")
            }

            // The teacher-forced check: a fresh pass over the reply's context predicts the same
            // greedy tokens (up to near-ties).
            let (context, generated, report) = try await engine.withSession { session -> ([Int], [Int], LiveSession.ConsistencyReport) in
                let replyStart = session.snapshot?.turns.last(where: { $0.turn.role == .user })?.replyStart ?? session.ledger.count
                return (Array(session.ledger[..<replyStart]), Array(session.ledger[replyStart...]), session.assertConsistent())
            }
            XCTAssertTrue(report.agrees(margin: NearTieTally.quantizedTolerance), "\(repo) turn \(index + 1): \(report)")
            if !generated.isEmpty {
                let forced = TeacherForcing.run(model: engine.loaded.model, prompt: context, continuation: generated)
                tally.record(
                    "\(repo) turn \(index + 1)", reference: forced.map(\.argmax), candidate: generated,
                    referenceMargins: forced.map(\.margin))
            }

            var stored = text
            if let words = step.spokenWords {
                stored = text.split(separator: " ").prefix(words).joined(separator: " ")
            }
            turns.append(ChatTurn(role: .assistant, text: stored))
            let phases = stats.phases
            rows.append([
                "\(index + 1)", stats.planReason ?? "–", "\(stats.prefilledTokens ?? 0)/\(stats.reusedTokens ?? 0)",
                phases.map { String(format: "%.0f/%.0f/%.0f/%.0f", $0.render * 1000, $0.rewind * 1000, $0.prefill * 1000, $0.firstToken * 1000) } ?? "–",
                stats.timeToFirstText.map { String(format: "%.0f", $0 * 1000) } ?? "–",
                "\(stats.generatedTokens)",
                String(format: "%.1f", stats.tokensPerSecond ?? 0),
                text.prefix(50).replacingOccurrences(of: "\n", with: " ").replacingOccurrences(of: "|", with: "/"),
            ])
        }
        EngineReport.appendTable(
            title: "Session reuse, \(repo) (engine \(engine.info.forked ? "fork" : "stock"), greedy)",
            header: ["Turn", "Plan", "Prefilled/reused", "Render/rewind/prefill/first ms", "TTFT ms", "Tokens", "tok/s", "Reply"],
            rows: rows)
        EngineReport.append("- Teacher-forced check (\(repo)): \(tally.summary)")
        tally.assertPassed()
    }
}
