import XCTest
@testable import AssistantKit

private typealias ToolExpectation = LocalBenchmark.ToolExpectation
private typealias ToolScore = LocalBenchmark.ToolScore

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

    func testOldInitializersStillCompile() {
        // The initializers as the app called them before the engine fields existed.
        let stats = LocalGenerationStats(
            timeToFirstText: 1, promptTokens: 2, promptTime: 3, generatedTokens: 4, generateTime: 5,
            reusedSession: true, draftTokens: 6, acceptedDraftTokens: 7, peakMemoryBytes: 8
        )
        XCTAssertNil(stats.engine)
        XCTAssertNil(stats.phases)
        XCTAssertNil(stats.prefilledTokens)
        XCTAssertNil(stats.reusedTokens)
        XCTAssertNil(stats.planReason)
        XCTAssertNil(stats.speculation)
        XCTAssertNil(stats.confidence)
        XCTAssertEqual(LocalGenerationStats(reusedSession: true).reusedSession, true)
        XCTAssertNil(LocalBenchmark.Row(prompt: "p", reply: "r", stats: stats).toolScore)
        XCTAssertFalse(LocalModelOption(id: "x", displayName: "X", approximateBytes: 1, note: "n").isHybrid)
        XCTAssertTrue(LocalBenchmark.report(model: "m", device: "d", speculative: "s", loadTime: nil, rows: []).hasPrefix("On-device benchmark"))
    }

    func testCatalogMarksHybridModels() {
        XCTAssertTrue(LocalModelCatalog.woof4B.isHybrid)
        XCTAssertTrue(LocalModelCatalog.woof2B.isHybrid)
        XCTAssertTrue(LocalModelCatalog.qwen35_2B.isHybrid)
        XCTAssertFalse(LocalModelCatalog.qwen3_4B.isHybrid)
        XCTAssertEqual(LocalModelCatalog.woof2B.id, "ConwayResearch/Underdog-Woof-2B-1.1")
        XCTAssertFalse(LocalModelCatalog.woof2B.supportsSpeculativeDecoding)
        XCTAssertEqual(LocalModelCatalog.option(for: "ConwayResearch/Underdog-Woof-2B-1.1"), LocalModelCatalog.woof2B)
        XCTAssertEqual(LocalModelCatalog.defaultModelID, LocalModelCatalog.woof4B.id)
        XCTAssertEqual(Set(LocalModelCatalog.options.map(\.id)).count, LocalModelCatalog.options.count)
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

final class LocalBenchmarkScenarioTests: XCTestCase {
    private typealias Scenario = LocalBenchmark.Scenario

    private func turns(_ scenario: Scenario) -> [Scenario.Turn] {
        scenario.actions.compactMap { action in
            if case .turn(let turn) = action { return turn }
            return nil
        }
    }

    func testScenarioList() {
        XCTAssertEqual(LocalBenchmark.scenarios.map(\.id), ["continued", "bargeIn", "coldPrefix", "longChat", "copyHeavy", "tools"])
        XCTAssertEqual(Set(LocalBenchmark.scenarios.map(\.title)).count, 6)
        XCTAssertEqual(LocalBenchmark.scenario(id: "tools"), LocalBenchmark.tools)
        XCTAssertNil(LocalBenchmark.scenario(id: "nope"))
    }

    func testContinuedPlaysTodaysPrompts() {
        XCTAssertEqual(LocalBenchmark.continued.steps, LocalBenchmark.prompts.map(Scenario.Step.user))
        XCTAssertEqual(LocalBenchmark.continued.context, LocalBenchmark.spokenContext)
        XCTAssertEqual(LocalBenchmark.continued.turnCount, 8)
    }

    func testBargeInCancelsTurnTwo() {
        let turns = turns(LocalBenchmark.bargeIn)
        XCTAssertGreaterThanOrEqual(turns.count, 4)
        XCTAssertEqual(turns[1].cancelAfterTokens, 10)
        XCTAssertEqual(turns[1].storedWords, 6)
        XCTAssertTrue(turns.enumerated().allSatisfy { $0.offset == 1 || ($0.element.cancelAfterTokens == nil && $0.element.storedWords == nil) })
        XCTAssertEqual(LocalBenchmark.spokenPrefix(of: "  Coffee  began in\nEthiopia, where legend says a goat herder", words: 6), "Coffee began in Ethiopia, where legend")
        XCTAssertEqual(LocalBenchmark.spokenPrefix(of: "Short one", words: 6), "Short one")
        XCTAssertEqual(LocalBenchmark.spokenPrefix(of: "Short one", words: 0), "")
    }

    func testColdPrefixReloadsWithoutThenWithTheSavedPrefix() {
        // The first run makes sure the prefix is on disk; then turn 1 of a new conversation right
        // after a reload, without and with it.
        let first = Scenario.Action.turn(Scenario.Turn(prompt: LocalBenchmark.prompts[0]))
        XCTAssertEqual(LocalBenchmark.coldPrefix.actions, [
            .reloadModel(usePrefixCache: true), first,
            .newConversation, .reloadModel(usePrefixCache: false), first,
            .newConversation, .reloadModel(usePrefixCache: true), first,
        ])
    }

    func testLongChatHasThirtyShortTurns() {
        let turns = turns(LocalBenchmark.longChat)
        XCTAssertEqual(turns.count, 30)
        XCTAssertTrue(turns.allSatisfy { $0.prompt.count <= 60 })
        XCTAssertFalse(LocalBenchmark.longChat.steps.contains(.newConversation))
    }

    func testCopyHeavyPromptsAreSeparateConversations() {
        let scenario = LocalBenchmark.copyHeavy
        XCTAssertEqual(scenario.turnCount, 6)
        XCTAssertEqual(scenario.steps.filter { $0 == .newConversation }.count, 5)
        XCTAssertEqual(scenario.context, LocalBenchmark.labTypedContext)
        XCTAssertNotEqual(scenario.steps.first, .newConversation)
    }

    func testToolPromptsCarryTheLabsExpectations() throws {
        let scenario = LocalBenchmark.tools
        let turns = turns(scenario)
        XCTAssertEqual(turns.count, 12)
        XCTAssertEqual(scenario.steps.filter { $0 == .newConversation }.count, 11)
        XCTAssertEqual(scenario.context, "<context>time: Wednesday 7 October 2026, 16:05 Asia/Singapore; input: spoken</context>")
        XCTAssertTrue(turns.allSatisfy { $0.expectation != nil })
        XCTAssertEqual(turns[0].prompt, "Remind me at 5 to call mum.")
        XCTAssertEqual(turns[0].expectation, ToolExpectation(name: "create_reminder", arguments: ["title": "call mum", "due": "2026-10-07T17:00"]))
        XCTAssertEqual(turns[11].expectation, ToolExpectation(name: "list_timers"))
        XCTAssertEqual(Set(turns.compactMap(\.expectation?.name)), ["create_reminder", "set_timer", "list_events", "create_event", "list_reminders", "list_timers"])
        XCTAssertEqual(turns.filter { $0.prompt.unicodeScalars.contains { $0.value > 0x2E80 } }.count, 6)
    }

    func testModifiersWithoutATurnAreIgnored() {
        let scenario = Scenario(id: "x", title: "X", steps: [.cancelAfter(tokens: 3), .newConversation, .storeSpokenPrefix(words: 2), .user("Hi"), .cancelAfter(tokens: 5)])
        XCTAssertEqual(scenario.actions, [.newConversation, .turn(Scenario.Turn(prompt: "Hi", cancelAfterTokens: 5))])
    }
}

final class ToolExpectationTests: XCTestCase {
    private func matches(_ expected: JSONValue, _ actual: JSONValue?) -> Bool {
        ToolExpectation.matches(expected, actual)
    }

    func testDates() {
        XCTAssertTrue(matches("2026-10-08", "2026-10-08"))
        XCTAssertTrue(matches("2026-10-08", "2026-10-08T09:30:00+08:00"))
        XCTAssertTrue(matches("2026-10-08", "tomorrow, 2026-10-8"))
        XCTAssertFalse(matches("2026-10-08", "2026-10-07"))
        XCTAssertFalse(matches("2026-10-08", "tomorrow"))

        XCTAssertTrue(matches("2026-10-07T17:00", "2026-10-07T17:00:00"))
        XCTAssertTrue(matches("2026-10-07T17:00", "2026-10-07 17:00"))
        XCTAssertTrue(matches("2026-10-07T17:00", "2026-10-07T17:00:59+08:00"))
        XCTAssertFalse(matches("2026-10-07T17:00", "2026-10-07"))
        XCTAssertFalse(matches("2026-10-07T17:00", "2026-10-07T05:00"))
        XCTAssertFalse(matches("2026-10-07T17:00", "2026-10-08T17:00"))
    }

    func testNumbers() {
        XCTAssertTrue(matches(600, 600))
        XCTAssertTrue(matches(600, "600"))
        XCTAssertTrue(matches(600, 600.0))
        XCTAssertTrue(matches(600, " 600 "))
        XCTAssertFalse(matches(600, "ten minutes"))
        XCTAssertFalse(matches(600, 60))
    }

    func testWords() {
        XCTAssertTrue(matches("call mum", "Call Mum!"))
        XCTAssertTrue(matches("call mum", "Remember to call your mum"))
        XCTAssertTrue(matches("快递", "去取快递"))
        XCTAssertTrue(matches("王经理", "和王经理开会"))
        XCTAssertTrue(matches("buy milk", "Buy milk."))
        XCTAssertFalse(matches("call mum", "Call dad"))
        XCTAssertTrue(matches("today", "today"))
        XCTAssertFalse(matches("overdue", "today"))
    }

    func testListsAndMissingValues() {
        XCTAssertTrue(matches(["today", "overdue"], "overdue"))
        XCTAssertFalse(matches(["today", "overdue"], "all"))
        XCTAssertFalse(matches("call mum", nil))
        XCTAssertFalse(matches("call mum", .null))
        XCTAssertTrue(matches(true, "True"))
        XCTAssertFalse(matches(true, false))
    }

    func testScore() {
        let expectation = ToolExpectation(name: "create_reminder", arguments: ["title": "call mum", "due": "2026-10-07T17:00"])
        XCTAssertEqual(
            expectation.score(name: "create_reminder", arguments: ["title": "Call mum", "due": "2026-10-07T17:00:00", "notes": "x"]),
            ToolScore(nameMatches: true, argumentsMatch: true, argumentResults: ["title": true, "due": true])
        )
        XCTAssertEqual(
            expectation.score(name: "create_reminder", arguments: ["title": "Call mum"]),
            ToolScore(nameMatches: true, argumentsMatch: false, argumentResults: ["title": true, "due": false])
        )
        let wrongTool = expectation.score(name: "create_event", arguments: ["title": "Call mum", "due": "2026-10-07T17:00"])
        XCTAssertFalse(wrongTool.nameMatches)
        XCTAssertFalse(wrongTool.argumentsMatch)
        XCTAssertEqual(expectation.score(name: nil, arguments: nil), ToolScore(nameMatches: false, argumentsMatch: false))
        XCTAssertEqual(ToolExpectation(name: "list_timers").score(name: "list_timers", arguments: [:]), ToolScore(nameMatches: true, argumentsMatch: true))
    }
}
