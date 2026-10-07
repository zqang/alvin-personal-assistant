import XCTest
@testable import AssistantKit

final class LocalSessionPlanTests: XCTestCase {
    private let system = "You are helpful."

    private func user(_ text: String, context: String? = "<context>t</context>") -> ChatTurn {
        ChatTurn(role: .user, text: text, context: context)
    }

    private func assistant(_ text: String) -> ChatTurn {
        ChatTurn(role: .assistant, text: text)
    }

    func testFirstRequestRebuildsWithoutHistory() throws {
        let plan = try XCTUnwrap(LocalSessionPlan.make(cachedSystem: nil, cachedTurns: nil, system: system, turns: [user("Hi")]))
        XCTAssertEqual(plan.action, .rebuild(history: []))
        XCTAssertEqual(plan.newTurn.text, "Hi")
    }

    func testRequestExtendingTheCachedSessionAppends() throws {
        let cached = [user("Hi"), assistant("Hello! How can I help?")]
        // Storage trims the reply the model generated.
        let turns = [user("Hi"), assistant("Hello! How can I help?\n"), user("Tell me a joke")]
        let plan = try XCTUnwrap(LocalSessionPlan.make(cachedSystem: system, cachedTurns: cached, system: system, turns: turns))
        XCTAssertEqual(plan.action, .append)
        XCTAssertEqual(plan.newTurn.text, "Tell me a joke")
    }

    func testInterruptedReplyRebuilds() throws {
        let cached = [user("Hi"), assistant("Hello! How can I help you today with anything?")]
        let turns = [user("Hi"), assistant("Hello! How can I"), user("Weather?")]
        let plan = try XCTUnwrap(LocalSessionPlan.make(cachedSystem: system, cachedTurns: cached, system: system, turns: turns))
        XCTAssertEqual(plan.action, .rebuild(history: [user("Hi"), assistant("Hello! How can I")]))
    }

    func testChangedSystemPromptRebuilds() throws {
        let cached = [user("Hi"), assistant("Hello")]
        let turns = cached + [user("Again")]
        let plan = try XCTUnwrap(LocalSessionPlan.make(cachedSystem: "Old", cachedTurns: cached, system: system, turns: turns))
        XCTAssertEqual(plan.action, .rebuild(history: cached))
    }

    func testLongHistoryRebuildsFromAWindowStartingWithTheUser() throws {
        var turns: [ChatTurn] = []
        for index in 0..<5 {
            turns += [user("q\(index)"), assistant("a\(index)")]
        }
        let cached = turns
        turns.append(user("q5"))
        let plan = try XCTUnwrap(LocalSessionPlan.make(cachedSystem: system, cachedTurns: cached, system: system, turns: turns, maxTurns: 10, keepTurns: 4))
        // Keeps up to three earlier turns, dropping the leading reply.
        XCTAssertEqual(plan.action, .rebuild(history: [user("q4"), assistant("a4")]))
        XCTAssertEqual(plan.newTurn.text, "q5")
    }

    func testRequestNotEndingWithUserIsRejected() {
        XCTAssertNil(LocalSessionPlan.make(cachedSystem: nil, cachedTurns: nil, system: system, turns: [user("Hi"), assistant("Hello")]))
        XCTAssertNil(LocalSessionPlan.make(cachedSystem: nil, cachedTurns: nil, system: system, turns: []))
    }

    func testContentPutsTheContextTagFirst() {
        XCTAssertEqual(LocalSessionPlan.content(of: user("Hi", context: "<context>x</context>")), "<context>x</context>\n\nHi")
        XCTAssertEqual(LocalSessionPlan.content(of: user("Hi", context: nil)), "Hi")
        XCTAssertEqual(LocalSessionPlan.content(of: assistant("Hello")), "Hello")
    }
}

final class ThinkingFilterTests: XCTestCase {
    private func run(_ chunks: [String]) -> String {
        var filter = ThinkingFilter()
        return chunks.map { filter.feed($0) }.joined() + filter.finish()
    }

    func testPassesPlainText() {
        XCTAssertEqual(run(["Hello ", "there."]), "Hello there.")
    }

    func testRemovesReasoningSplitAcrossChunks() {
        XCTAssertEqual(run(["<thi", "nk>\nplanning", " the answer</th", "ink>\n\nSure", ", here it is."]), "Sure, here it is.")
    }

    func testRemovesEmptyReasoningBlock() {
        XCTAssertEqual(run(["<think>\n\n</think>\n\n", "Hi!"]), "Hi!")
    }

    func testKeepsTextThatOnlyLooksLikeATagStart() {
        XCTAssertEqual(run(["Use a < b", " and <th", "ese."]), "Use a < b and <these.")
        XCTAssertEqual(run(["Ends with <"]), "Ends with <")
    }

    func testDropsUnterminatedReasoning() {
        XCTAssertEqual(run(["Hi. <think>never closed"]), "Hi. ")
    }

    func testPartialTagSuffixLength() {
        XCTAssertEqual(ThinkingFilter.partialTagSuffixLength(of: "abc<thi", tag: "<think>"), 4)
        XCTAssertEqual(ThinkingFilter.partialTagSuffixLength(of: "abc", tag: "<think>"), 0)
        XCTAssertEqual(ThinkingFilter.partialTagSuffixLength(of: "<", tag: "<think>"), 1)
    }
}

final class LocalBenchmarkTests: XCTestCase {
    func testStatsDerivedValues() {
        let stats = LocalGenerationStats(promptTokens: 200, promptTime: 0.5, generatedTokens: 60, generateTime: 2, draftTokens: 40, acceptedDraftTokens: 30)
        XCTAssertEqual(stats.tokensPerSecond, 30)
        XCTAssertEqual(stats.prefillTokensPerSecond, 400)
        XCTAssertEqual(stats.acceptanceRate, 0.75)
        XCTAssertNil(LocalGenerationStats().tokensPerSecond)
        XCTAssertNil(LocalGenerationStats(draftTokens: 0, acceptedDraftTokens: 0).acceptanceRate)
    }

    func testReportListsTurnsAndComparesCacheReuse() {
        let rows = [
            LocalBenchmark.Row(prompt: "Hi", reply: "Hello!\n", stats: LocalGenerationStats(timeToFirstText: 1.2, promptTokens: 300, promptTime: 1, generatedTokens: 20, generateTime: 1, reusedSession: false, peakMemoryBytes: 2_600_000_000)),
            LocalBenchmark.Row(prompt: "Why?", reply: "Because.", stats: LocalGenerationStats(timeToFirstText: 0.3, promptTokens: 30, promptTime: 0.1, generatedTokens: 25, generateTime: 1, reusedSession: true, draftTokens: 10, acceptedDraftTokens: 7)),
        ]
        let report = LocalBenchmark.report(model: "Woof", device: "iPhone", speculative: "off", loadTime: 4.25, rows: rows)
        XCTAssertTrue(report.contains("Load time: 4.2 s") || report.contains("Load time: 4.3 s"))
        XCTAssertTrue(report.contains("1 | 1.20 | 300 (300) | 20.0 | – | rebuilt | 2.60"))
        XCTAssertTrue(report.contains("2 | 0.30 | 30 (300) | 25.0 | 70% | reused | –"))
        XCTAssertTrue(report.contains("first text 0.30 s with the cache vs 1.20 s without"))
        XCTAssertTrue(report.contains("   → Hello!"))
    }

    func testMedian() {
        XCTAssertNil(LocalBenchmark.median([]))
        XCTAssertEqual(LocalBenchmark.median([3, 1, 2]), 2)
        XCTAssertEqual(LocalBenchmark.median([4, 1, 2, 3]), 2.5)
    }

    func testCatalogAndSettings() throws {
        XCTAssertNotNil(LocalModelCatalog.option(for: LocalModelCatalog.defaultModelID))
        XCTAssertFalse(LocalModelCatalog.woof4B.supportsSpeculativeDecoding)
        XCTAssertTrue(LocalModelCatalog.qwen3_4B.supportsSpeculativeDecoding)

        // Settings saved before on-device models existed still load with the defaults.
        let old = try JSONDecoder().decode(AssistantSettings.self, from: Data(#"{"provider":"anthropic"}"#.utf8))
        XCTAssertEqual(old.localModelID, LocalModelCatalog.defaultModelID)
        XCTAssertTrue(old.localSpeculativeDecoding)

        var settings = AssistantSettings()
        settings.provider = .onDevice
        settings.localModelID = LocalModelCatalog.qwen3_4B.id
        settings.localSpeculativeDecoding = false
        let roundTrip = try JSONDecoder().decode(AssistantSettings.self, from: JSONEncoder().encode(settings))
        XCTAssertEqual(roundTrip, settings)
    }
}
