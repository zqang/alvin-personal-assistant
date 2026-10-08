import XCTest
@testable import AssistantKit

final class SessionPlannerTests: XCTestCase {
    private typealias F = SessionFixtures

    private func plan(
        _ live: SessionSnapshot?,
        _ turns: [ChatTurn],
        key: String = SessionFixtures.key,
        persisted: Bool = false,
        limits: SessionPlanner.Limits = .init()
    ) -> SessionPlan? {
        SessionPlanner.plan(live: live, prefixKey: key, hasPersistedPrefix: persisted, turns: turns, limits: limits)
    }

    // MARK: Rule 0

    func testRequestNotEndingInAUserTurnHasNoPlan() {
        XCTAssertNil(plan(nil, []))
        XCTAssertNil(plan(nil, [F.user("Hi"), F.assistant("Hello")]))
        XCTAssertNil(plan(F.snapshot([F.user("Hi"), F.assistant("Hello")]), [F.user("Hi"), F.assistant("Hello")]))
    }

    // MARK: Rules 1 and 2

    func testNewSessionStartsFromAnEmptyCache() {
        XCTAssertEqual(
            plan(nil, [F.user("Hi")]),
            SessionPlan(base: .empty, pieces: [.systemPrefix, .firstTurns([F.user("Hi")])], firstTurnIndex: 0, reason: .newSession)
        )
    }

    func testNewSessionLoadsThePersistedPrefix() {
        XCTAssertEqual(
            plan(nil, [F.user("Hi")], persisted: true),
            SessionPlan(base: .persistedPrefix, pieces: [.firstTurns([F.user("Hi")])], firstTurnIndex: 0, reason: .newSession)
        )
    }

    func testNewSessionStartsFromTheWindow() throws {
        let turns = F.conversation(31)
        let result = try XCTUnwrap(plan(nil, turns))
        // 31 − 12 = 19 is a reply, so the window moves forward to the user turn at 20.
        XCTAssertEqual(result.firstTurnIndex, 20)
        XCTAssertEqual(result.pieces, [.systemPrefix, .firstTurns(Array(turns[20...]))])
    }

    func testPrefixChangeRebuilds() {
        let cached = [F.user("Hi"), F.assistant("Hello")]
        let turns = cached + [F.user("Again")]
        let live = F.snapshot(cached, prefixKey: "old")
        XCTAssertEqual(
            plan(live, turns),
            SessionPlan(base: .empty, pieces: [.systemPrefix, .firstTurns(turns)], firstTurnIndex: 0, reason: .prefixChanged)
        )
        XCTAssertEqual(
            plan(live, turns, persisted: true),
            SessionPlan(base: .persistedPrefix, pieces: [.firstTurns(turns)], firstTurnIndex: 0, reason: .prefixChanged)
        )
    }

    // MARK: Rule 3

    func testAppendKeepsTheWholeCache() {
        let live = F.snapshot([F.user("Hi"), F.assistant("Hello! How can I help?")])
        // Storage trims the reply the model generated.
        let turns = [F.user("Hi"), F.assistant("Hello! How can I help?\n"), F.user("Tell me a joke")]
        XCTAssertEqual(
            plan(live, turns),
            SessionPlan(base: .keep(240), pieces: [.continuation([F.user("Tell me a joke")])], firstTurnIndex: 0, reason: .append)
        )
    }

    func testAppendWithAnInterveningCloudTurn() {
        let live = F.snapshot([F.user("Hi"), F.assistant("Hello")])
        // The user asked something Claude answered (with a tool round), then came back on device.
        let newTurns = [F.user("Search the news"), F.assistant("Here's the news.", rounds: [F.round("web_search")]), F.user("Thanks")]
        let result = plan(live, [F.user("Hi"), F.assistant("Hello")] + newTurns)
        XCTAssertEqual(result, SessionPlan(base: .keep(240), pieces: [.continuation(newTurns)], firstTurnIndex: 0, reason: .append))
    }

    func testAppendToAWindowedCache() {
        let turns = F.conversation(8)
        let live = F.snapshot(Array(turns[4...]), firstTurnIndex: 4, tokenCount: 900)
        let request = turns + [F.user("next")]
        XCTAssertEqual(
            plan(live, request),
            SessionPlan(base: .keep(900), pieces: [.continuation([F.user("next")])], firstTurnIndex: 4, reason: .append)
        )
    }

    func testAppendComparesToolRounds() {
        let reply = F.assistant("It's 4 pm.", rounds: [F.round()])
        let live = F.snapshot([F.user("Time?"), reply])
        XCTAssertEqual(plan(live, [F.user("Time?"), reply, F.user("Thanks")])?.reason, .append)

        // Same text, another round: not what the cache saw.
        let other = F.assistant("It's 4 pm.", rounds: [F.round(result: #"{"time":"16:06"}"#)])
        XCTAssertEqual(plan(live, [F.user("Time?"), other, F.user("Thanks")])?.reason, .diverged)
    }

    func testChangedContextIsNotTheSameTurn() {
        let live = F.snapshot([F.user("Hi"), F.assistant("Hello")])
        let turns = [F.user("Hi", context: "<context>input: typed</context>"), F.assistant("Hello"), F.user("Again")]
        XCTAssertEqual(plan(live, turns)?.reason, .diverged)
    }

    // MARK: Rule 4

    func testReplacedLastReplyRewindsToTheReplyStart() {
        let live = F.snapshot([F.user("Hi"), F.assistant("Hello! How can I help you today with anything?")], replyStart: 200)
        let turns = [F.user("Hi"), F.assistant("Hello! How can I "), F.user("Weather?")]
        XCTAssertEqual(
            plan(live, turns),
            SessionPlan(
                base: .keep(200),
                pieces: [.assistantText("Hello! How can I"), .continuation([F.user("Weather?")])],
                firstTurnIndex: 0,
                reason: .replaceLastReply
            )
        )
    }

    func testReplacedLastReplyWithMoreTurnsAfterIt() {
        let turns = F.conversation(6)
        let live = F.snapshot(Array(turns[2...]), firstTurnIndex: 2, userStart: 480, replyStart: 500, tokenCount: 600)
        var request = turns
        request[5] = F.assistant("a2 short")
        request += [F.user("q3"), F.assistant("from the cloud"), F.user("q4")]
        let result = plan(live, request)
        XCTAssertEqual(
            result,
            SessionPlan(
                base: .keep(500),
                pieces: [.assistantText("a2 short"), .continuation(Array(request[6...]))],
                firstTurnIndex: 2,
                reason: .replaceLastReply
            )
        )
    }

    func testReplacedReplyWithToolRoundsRebuilds() {
        let window = [F.user("Remind me"), F.assistant("Done, reminder set for five.", rounds: [F.round("create_reminder")])]
        let live = F.snapshot(window)
        // The interrupted reply keeps its round but loses text: rule 4 can't express rounds.
        let turns = [F.user("Remind me"), F.assistant("Done,", rounds: [F.round("create_reminder")]), F.user("Thanks")]
        XCTAssertEqual(
            plan(live, turns),
            SessionPlan(base: .keep(100), pieces: [.firstTurns(turns)], firstTurnIndex: 0, reason: .diverged)
        )
        // Nor a request whose replacement has rounds the cached reply didn't.
        let plainLive = F.snapshot([F.user("Remind me"), F.assistant("Sure.")])
        XCTAssertEqual(plan(plainLive, turns)?.reason, .diverged)
    }

    func testReplacedReplyWithoutAReplyMarkRebuilds() {
        var live = F.snapshot([F.user("Hi"), F.assistant("Hello there, friend")])
        live.turns[0].replyStart = nil
        XCTAssertEqual(plan(live, [F.user("Hi"), F.assistant("Hello"), F.user("Again")])?.reason, .diverged)
    }

    // MARK: Rule 5

    func testReplacedTentativeUserTurnRewindsToItsStart() {
        let expected = SessionPlan(base: .keep(180), pieces: [.continuation([F.user("What's the weather in Paris?")])], firstTurnIndex: 0, reason: .replaceLastUserTurn)
        let request = [F.user("Hi"), F.assistant("Hello"), F.user("What's the weather in Paris?")]

        // The early reply was cancelled before the engine recorded it…
        let bare = F.snapshot([F.user("Hi"), F.assistant("Hello"), F.user("What's the weather")], userStart: 180)
        XCTAssertEqual(plan(bare, request), expected)

        // …or after it recorded part of it.
        let partial = F.snapshot([F.user("Hi"), F.assistant("Hello"), F.user("What's the weather"), F.assistant("The weath")], userStart: 180)
        XCTAssertEqual(plan(partial, request), expected)
    }

    func testRetriedUserTurnIsFedAgain() {
        let cached = [F.user("Hi"), F.assistant("Hello"), F.user("Tell me a story"), F.assistant("Once")]
        let request = Array(cached.prefix(3))
        XCTAssertEqual(
            plan(F.snapshot(cached, userStart: 180), request),
            SessionPlan(base: .keep(180), pieces: [.continuation([F.user("Tell me a story")])], firstTurnIndex: 0, reason: .replaceLastUserTurn)
        )
    }

    func testReplacedFirstUserTurn() {
        let live = F.snapshot([F.user("Hi")], userStart: 100, replyStart: 120, tokenCount: 120)
        XCTAssertEqual(
            plan(live, [F.user("Hello")]),
            SessionPlan(base: .keep(100), pieces: [.continuation([F.user("Hello")])], firstTurnIndex: 0, reason: .replaceLastUserTurn)
        )
    }

    func testReplacedUserTurnFollowedByMoreTurnsRebuilds() {
        let live = F.snapshot([F.user("Hi"), F.assistant("Hello"), F.user("Weather")])
        let request = [F.user("Hi"), F.assistant("Hello"), F.user("Weather?"), F.assistant("Sunny"), F.user("Thanks")]
        XCTAssertEqual(plan(live, request)?.reason, .diverged)
    }

    // MARK: Rule 6

    func testAnotherConversationKeepsOnlyTheSystemPrefix() {
        let live = F.snapshot([F.user("Hi"), F.assistant("Hello")])
        let other = [F.user("Plan a trip"), F.assistant("Where to?"), F.user("Japan")]
        XCTAssertEqual(
            plan(live, other),
            SessionPlan(base: .keep(100), pieces: [.firstTurns(other)], firstTurnIndex: 0, reason: .diverged)
        )
    }

    func testShorterConversationThanTheCachedWindowDiverges() {
        let live = F.snapshot(Array(F.conversation(10)[6...]), firstTurnIndex: 6)
        XCTAssertEqual(
            plan(live, [F.user("New")]),
            SessionPlan(base: .keep(100), pieces: [.firstTurns([F.user("New")])], firstTurnIndex: 0, reason: .diverged)
        )
    }

    func testPrewarmedSessionFeedsTheFirstTurnsAfterTheSystemPrefix() {
        let live = SessionSnapshot(prefixKey: F.key, systemEnd: 100, firstTurnIndex: 0, turns: [], tokenCount: 100)
        XCTAssertEqual(
            plan(live, [F.user("Hi")]),
            SessionPlan(base: .keep(100), pieces: [.firstTurns([F.user("Hi")])], firstTurnIndex: 0, reason: .diverged)
        )
    }

    func testWithoutASystemPrefixDivergenceStartsOver() {
        let live = F.snapshot([F.user("Hi"), F.assistant("Hello"), F.user("More"), F.assistant("Sure")], systemEnd: 0)
        let other = [F.user("Something else"), F.assistant("Ok"), F.user("Again")]
        XCTAssertEqual(plan(live, other), SessionPlan(base: .keep(0), pieces: [.firstTurns(other)], firstTurnIndex: 0, reason: .diverged))

        // A replaced first turn can't be continued from an empty ledger either.
        let first = F.snapshot([F.user("Hi"), F.assistant("Hello")], systemEnd: 0, userStart: 0, replyStart: 20, tokenCount: 40)
        XCTAssertEqual(plan(first, [F.user("Hey")]), SessionPlan(base: .keep(0), pieces: [.firstTurns([F.user("Hey")])], firstTurnIndex: 0, reason: .diverged))
    }

    func testOneTurnRequestAfterAnotherConversationReplacesTheFirstTurn() {
        // Rule 5 applies: nothing before the newest user turn, which the request replaces. Its
        // start is the system prefix's end, so this feeds what rule 6 would.
        let live = F.snapshot([F.user("Hi"), F.assistant("Hello")], systemEnd: 100, userStart: 100, replyStart: 120, tokenCount: 140)
        XCTAssertEqual(
            plan(live, [F.user("Something else")]),
            SessionPlan(base: .keep(100), pieces: [.continuation([F.user("Something else")])], firstTurnIndex: 0, reason: .replaceLastUserTurn)
        )
    }

    // MARK: Rule 7

    func testOverBudgetRebuildsFromAWindowStartingOnAUserTurn() throws {
        let cached = F.conversation(20)
        let live = F.snapshot(cached, tokenCount: 6_000)
        let request = cached + [F.user(String(repeating: "word ", count: 120))]
        let result = try XCTUnwrap(plan(live, request))
        XCTAssertEqual(result.reason, .overBudget)
        XCTAssertEqual(result.base, .keep(100))
        // 21 − 12 = 9 is a reply; the window starts at the user turn after it.
        XCTAssertEqual(result.firstTurnIndex, 10)
        XCTAssertEqual(request[result.firstTurnIndex].role, .user)
        XCTAssertEqual(result.pieces, [.firstTurns(Array(request[10...]))])
    }

    func testBudgetBoundary() {
        let cached = [F.user("Hi", context: nil), F.assistant("Hello")]
        let next = F.user("abcdef", context: nil)  // 6 / 3 + 8 = 10 tokens
        let request = cached + [next]
        let limits = SessionPlanner.Limits(maxTokens: 250, keepTurns: 12)
        XCTAssertEqual(plan(F.snapshot(cached, tokenCount: 240), request, limits: limits)?.reason, .append)
        XCTAssertEqual(plan(F.snapshot(cached, tokenCount: 241), request, limits: limits)?.reason, .overBudget)
    }

    func testReplacingTheReplyOrUserTurnAlsoRespectsTheBudget() {
        let limits = SessionPlanner.Limits(maxTokens: 220, keepTurns: 12)
        let reply = F.snapshot([F.user("Hi"), F.assistant("Hello there")], replyStart: 200)
        XCTAssertEqual(plan(reply, [F.user("Hi"), F.assistant("Hello"), F.user("Again")])?.reason, .replaceLastReply)
        XCTAssertEqual(plan(reply, [F.user("Hi"), F.assistant("Hello"), F.user(String(repeating: "x", count: 60))], limits: limits)?.reason, .overBudget)

        let user = F.snapshot([F.user("Hi"), F.assistant("Hello"), F.user("Wea")], userStart: 180)
        XCTAssertEqual(plan(user, [F.user("Hi"), F.assistant("Hello"), F.user("Weather")], limits: limits)?.reason, .replaceLastUserTurn)
        XCTAssertEqual(plan(user, [F.user("Hi"), F.assistant("Hello"), F.user(String(repeating: "x", count: 120))], limits: limits)?.reason, .overBudget)
    }

    // MARK: Helpers

    func testWindowStart() {
        let turns = F.conversation(9)
        XCTAssertEqual(SessionPlanner.windowStart(turns, keepTurns: 12), 0)
        XCTAssertEqual(SessionPlanner.windowStart(turns, keepTurns: 4), 6)   // 5 is a reply
        XCTAssertEqual(SessionPlanner.windowStart(turns, keepTurns: 5), 4)
        XCTAssertEqual(SessionPlanner.windowStart(turns, keepTurns: 1), 8)
        XCTAssertEqual(SessionPlanner.windowStart([], keepTurns: 12), 0)
        XCTAssertEqual(SessionPlanner.windowStart([F.assistant("x")], keepTurns: 12), 1)
    }

    func testEstimatedTokens() {
        // "<c>\n\nhello" is 10 bytes, "你好" 6: 16 / 3 = 5, plus 8 + 8 per turn and 64 for the round.
        let turns = [F.user("hello", context: "<c>"), F.assistant("你好", rounds: [F.round()])]
        XCTAssertEqual(SessionPlanner.estimatedTokens(turns), 85)
        XCTAssertEqual(SessionPlanner.estimatedTokens(turns[1...]), 6 / 3 + 8 + 64)
        XCTAssertEqual(SessionPlanner.estimatedTokens([ChatTurn]()), 0)
    }

    func testDefaultLimits() {
        XCTAssertEqual(SessionPlanner.Limits(), SessionPlanner.Limits(maxTokens: 6_144, keepTurns: 12))
    }
}

final class SessionSnapshotTests: XCTestCase {
    private typealias F = SessionFixtures

    func testAfterPrefillHoldsTheWindowAndMarksTheNewestUserTurn() throws {
        let turns = F.conversation(5)
        let plan = SessionPlan(base: .keep(100), pieces: [.firstTurns(Array(turns[2...]))], firstTurnIndex: 2, reason: .diverged)
        let snapshot = SessionSnapshot.afterPrefill(prefixKey: "k", systemEnd: 100, plan: plan, turns: turns, userStart: 160, replyStart: 175)
        XCTAssertEqual(snapshot.firstTurnIndex, 2)
        XCTAssertEqual(snapshot.turns.map(\.turn), Array(turns[2...]))
        XCTAssertEqual(snapshot.turns.map(\.start), [nil, nil, 160])
        XCTAssertEqual(snapshot.turns.map(\.replyStart), [nil, nil, 175])
        XCTAssertEqual(snapshot.tokenCount, 175)
        XCTAssertEqual(snapshot.newestUserTurnIndex, 2)
    }

    func testRecordReplyAppendsThenReplacesTheReplyInProgress() {
        let turns = [F.user("Remind me at 5")]
        let plan = SessionPlan(base: .empty, pieces: [.systemPrefix, .firstTurns(turns)], firstTurnIndex: 0, reason: .newSession)
        var snapshot = SessionSnapshot.afterPrefill(prefixKey: "k", systemEnd: 100, plan: plan, turns: turns, userStart: 100, replyStart: 130)

        snapshot.recordReply(F.assistant("", rounds: [F.round("create_reminder")]), tokenCount: 190)
        XCTAssertEqual(snapshot.turns.count, 2)
        XCTAssertEqual(snapshot.tokenCount, 190)

        let reply = F.assistant("Done.", rounds: [F.round("create_reminder")])
        snapshot.recordReply(reply, tokenCount: 200)
        XCTAssertEqual(snapshot.turns.map(\.turn), turns + [reply])
        XCTAssertEqual(snapshot.tokenCount, 200)

        // The next request continues it.
        let next = SessionPlanner.plan(live: snapshot, prefixKey: "k", hasPersistedPrefix: false, turns: turns + [reply, F.user("Thanks")])
        XCTAssertEqual(next, SessionPlan(base: .keep(200), pieces: [.continuation([F.user("Thanks")])], firstTurnIndex: 0, reason: .append))
    }

    func testKeptTokens() {
        XCTAssertEqual(SessionPlan(base: .keep(42), pieces: [], firstTurnIndex: 0, reason: .append).keptTokens, 42)
        XCTAssertEqual(SessionPlan(base: .empty, pieces: [], firstTurnIndex: 0, reason: .newSession).keptTokens, 0)
        XCTAssertEqual(SessionPlan(base: .persistedPrefix, pieces: [], firstTurnIndex: 0, reason: .newSession).keptTokens, 0)
    }
}

final class CheckpointPolicyTests: XCTestCase {
    private let mib = 1 << 20
    private let marks: [CheckpointMark: Int] = [.systemEnd: 100, .lastUserStart: 180, .replyStart: 200]

    func testKeepsEveryMarkWithinTheBudget() {
        XCTAssertEqual(CheckpointPolicy.marksToKeep(marks, bytesPerMark: 49 * mib, budgetBytes: 160 * mib), marks)
        XCTAssertEqual(CheckpointPolicy.defaultBudgetBytes, 160 * mib)
        XCTAssertEqual(CheckpointPolicy.marksToKeep(marks, bytesPerMark: 49 * mib), marks)
    }

    func testDropsTheReplyStartFirst() {
        XCTAssertEqual(
            CheckpointPolicy.marksToKeep(marks, bytesPerMark: 49 * mib, budgetBytes: 100 * mib),
            [.systemEnd: 100, .lastUserStart: 180]
        )
    }

    func testThenTheLastUserStart() {
        XCTAssertEqual(CheckpointPolicy.marksToKeep(marks, bytesPerMark: 49 * mib, budgetBytes: 60 * mib), [.systemEnd: 100])
    }

    func testAlwaysKeepsTheSystemEnd() {
        XCTAssertEqual(CheckpointPolicy.marksToKeep(marks, bytesPerMark: 49 * mib, budgetBytes: 0), [.systemEnd: 100])
        XCTAssertEqual(CheckpointPolicy.marksToKeep([.replyStart: 200], bytesPerMark: 49 * mib, budgetBytes: 0), [:])
    }

    func testMarksAtOnePositionShareACheckpoint() {
        let shared: [CheckpointMark: Int] = [.systemEnd: 100, .lastUserStart: 100, .replyStart: 130]
        XCTAssertEqual(CheckpointPolicy.bytes(of: shared, bytesPerMark: 49 * mib), 98 * mib)
        XCTAssertEqual(CheckpointPolicy.marksToKeep(shared, bytesPerMark: 49 * mib, budgetBytes: 100 * mib), shared)
        XCTAssertEqual(CheckpointPolicy.marksToKeep(shared, bytesPerMark: 49 * mib, budgetBytes: 60 * mib), [.systemEnd: 100, .lastUserStart: 100])
    }

    func testTrimmableModelsKeepEverything() {
        XCTAssertEqual(CheckpointPolicy.marksToKeep(marks, bytesPerMark: 0, budgetBytes: 0), marks)
    }
}

final class PrefixKeyTests: XCTestCase {
    func testFNV1a64KnownValues() {
        XCTAssertEqual(PrefixKey.hex(PrefixKey.fnv1a64([])), "cbf29ce484222325")
        XCTAssertEqual(PrefixKey.hex(PrefixKey.fnv1a64(Array("a".utf8))), "af63dc4c8601ec8c")
        XCTAssertEqual(PrefixKey.hex(PrefixKey.fnv1a64(Array("foobar".utf8))), "85944171f73967e8")
        XCTAssertEqual(PrefixKey.hex(0xab), "00000000000000ab")
    }

    func testKeyIsStable() {
        // Reference values from an independent FNV-1a-64 implementation.
        XCTAssertEqual(SessionFixtures.key, "58bb0c8b3c2e646a")
        XCTAssertEqual(
            PrefixKey.make(
                modelID: "ConwayResearch/Underdog-Woof-4B-1.1",
                revision: "models--x--snapshot-abc",
                system: "系统提示 System",
                toolsJSON: #"[{"name":"t"}]"#,
                context: "enable_thinking=false",
                formatVersion: 1
            ),
            "f39f5800a5304535"
        )
    }

    func testEveryFieldChangesTheKey() {
        let base = ["m", "r", "s", "[]", "c"]
        func key(_ fields: [String], _ version: Int = 1) -> String {
            PrefixKey.make(modelID: fields[0], revision: fields[1], system: fields[2], toolsJSON: fields[3], context: fields[4], formatVersion: version)
        }
        var keys: Set<String> = [key(base), key(base, 2)]
        for index in base.indices {
            var changed = base
            changed[index] += "x"
            keys.insert(key(changed))
        }
        XCTAssertEqual(keys.count, 7)
        // The separator keeps fields apart.
        XCTAssertNotEqual(key(["ab", "", "s", "[]", "c"]), key(["a", "b", "s", "[]", "c"]))
        XCTAssertTrue(key(base).allSatisfy { "0123456789abcdef".contains($0) })
        XCTAssertEqual(key(base).count, 16)
    }

    func testTokenHash() {
        XCTAssertEqual(PrefixKey.tokenHash([1, 2, 3]), "da2bfb225e0d1f05")
        XCTAssertEqual(PrefixKey.tokenHash([151_644, 8_948, 198]), "12b970a10f6a1983")
        XCTAssertEqual(PrefixKey.tokenHash([1, 2, 3][1...]), PrefixKey.tokenHash([2, 3]))
        XCTAssertNotEqual(PrefixKey.tokenHash([1, 2, 3]), PrefixKey.tokenHash([3, 2, 1]))
        XCTAssertEqual(PrefixKey.tokenHash([Int]()), "cbf29ce484222325")
    }

    func testCanonicalToolsAndContext() {
        let tools = [
            ToolDefinition(name: "set_timer", description: "Starts a timer.", inputSchema: ["type": "object", "required": ["seconds"]]),
            ToolDefinition(name: "get_current_time", description: "The time.", inputSchema: ["type": "object"]),
        ]
        XCTAssertEqual(
            PrefixKey.canonicalToolsJSON(tools),
            #"[{"description":"Starts a timer.","input_schema":{"required":["seconds"],"type":"object"},"name":"set_timer"},{"description":"The time.","input_schema":{"type":"object"},"name":"get_current_time"}]"#
        )
        XCTAssertEqual(PrefixKey.canonicalToolsJSON([]), "[]")
        XCTAssertEqual(PrefixKey.canonicalContext(["enable_thinking": false, "a": true]), "a=true,enable_thinking=false")
        XCTAssertEqual(PrefixKey.canonicalContext([:]), "")
    }
}
