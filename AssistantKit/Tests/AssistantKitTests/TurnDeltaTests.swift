import XCTest
@testable import AssistantKit

final class TurnDeltaTests: XCTestCase {
    private typealias F = SessionFixtures
    private typealias T = FakeChatMLTemplate

    private let system = "You are a helpful assistant."
    private let tools: [JSONValue] = [
        ["type": "function", "function": ["name": "get_current_time", "description": "The time now.", "parameters": ["type": "object", "properties": [:]]]],
    ]

    private func template(tools: Bool = true) -> FakeChatMLTemplate {
        FakeChatMLTemplate(tools: tools ? self.tools : [])
    }

    private func render(_ turns: [ChatTurn], tools: Bool = true, generationPrompt: Bool = true) -> [Int] {
        template(tools: tools).renderTokens(T.messages(system: system, turns: turns), addGenerationPrompt: generationPrompt)
    }

    /// The sentinel delta for `newTurns`, as tokens.
    private func delta(_ newTurns: [ChatTurn], tools: Bool = true) throws -> [Int] {
        let fake = template(tools: tools)
        let text = fake.render(T.messages(system: system, turns: TurnDelta.sentinelTurns(followedBy: newTurns)), addGenerationPrompt: true)
        let cut = try XCTUnwrap(TurnDelta.continuation(in: text, after: TurnDelta.assistantSentinel, turnEnd: T.turnEnd))
        XCTAssertTrue(cut.hasPrefix(T.turnEnd))
        return fake.encode(cut)
    }

    // MARK: Continuation

    func testDeltaEqualsTheCanonicalSuffix() throws {
        let cached = [F.user("Hi"), F.assistant("Hello! How can I help?")]
        for newTurns in [
            [F.user("Tell me a joke")],
            [F.user("Search the news"), F.assistant("Here it is.", rounds: [F.round("web_search", result: #"{"results":[]}"#)]), F.user("Thanks")],
        ] {
            for tools in [true, false] {
                let delta = try delta(newTurns, tools: tools)
                let canonical = render(cached + newTurns, tools: tools)
                XCTAssertEqual(Array(canonical.suffix(delta.count)), delta)
                // The cut is right after the cached reply's text.
                let throughReply = render(cached, tools: tools, generationPrompt: false).dropLast(2)  // "<|im_end|>\n"
                XCTAssertEqual(Array(canonical.dropLast(delta.count)), Array(throughReply))
            }
        }
    }

    func testDeltaAppendsToTheAsGeneratedLedger() throws {
        let fake = template()
        let user = F.user("Hi")
        let reply = "Hello! How can I help?"
        // After turn 1 the cache holds its render, the reply as generated, and the stop token.
        let firstTurn = render([user])
        let ledger = firstTurn + fake.encode(reply) + [T.imEnd]

        let next = F.user("Tell me a joke")
        let delta = try delta([next])
        let overlap = TurnDelta.overlap(ledgerTail: ledger[...], delta: delta)
        XCTAssertEqual(overlap, 1)

        let expected = fake.encode(
            fake.render(T.messages(system: system, turns: [user]), addGenerationPrompt: true) + reply
                + "<|im_end|>\n<|im_start|>user\n" + LocalSessionPlan.content(of: next) + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        XCTAssertEqual(ledger + delta.dropFirst(overlap), expected)
    }

    func testToolRoundDelta() throws {
        let fake = template()
        let call = T.ToolCall(name: "get_current_time", arguments: [:])
        let result = #"{"time":"16:05"}"#
        // Sentinel render of a tool round: the placeholder reply carries the calls.
        let text = fake.render(
            [.system(system), .user(TurnDelta.userSentinel), .assistant(TurnDelta.assistantSentinel, toolCalls: [call]), .tool(result)],
            addGenerationPrompt: true
        )
        let cut = try XCTUnwrap(TurnDelta.continuation(in: text, after: TurnDelta.assistantSentinel, turnEnd: T.turnEnd))
        XCTAssertEqual(cut, "<|im_end|>\n<|im_start|>user\n<tool_response>\n\(result)\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

        // The real conversation: the model's reply was only the call.
        let user = F.user("What time is it?")
        let canonical = fake.renderTokens([.system(system), .user(LocalSessionPlan.content(of: user)), .assistant("", toolCalls: [call]), .tool(result)], addGenerationPrompt: true)
        let delta = fake.encode(cut)
        XCTAssertEqual(Array(canonical.suffix(delta.count)), delta)
        XCTAssertTrue(fake.decode(Array(canonical.dropLast(delta.count))).hasSuffix("</tool_call>"))

        // As generated: the call, then the stop token; the delta overlaps it.
        let generated = render([user]) + fake.encode("<tool_call>\n{\"name\": \"get_current_time\", \"arguments\": {}}\n</tool_call>") + [T.imEnd]
        XCTAssertEqual(TurnDelta.overlap(ledgerTail: generated[...], delta: delta), 1)
    }

    func testDuplicateOrMissingSentinelGivesNil() throws {
        let fake = template()
        // A user quoting the sentinel makes the cut ambiguous.
        let quoting = [F.user("Say \(TurnDelta.assistantSentinel) back to me")]
        let text = fake.render(T.messages(system: system, turns: TurnDelta.sentinelTurns(followedBy: quoting)), addGenerationPrompt: true)
        XCTAssertNil(TurnDelta.continuation(in: text, after: TurnDelta.assistantSentinel, turnEnd: T.turnEnd))

        XCTAssertNil(TurnDelta.continuation(in: "no sentinel <|im_end|>", after: TurnDelta.assistantSentinel, turnEnd: T.turnEnd))
        XCTAssertNil(TurnDelta.continuation(in: "<|im_end|>\(TurnDelta.assistantSentinel) and no turn end", after: TurnDelta.assistantSentinel, turnEnd: T.turnEnd))
        XCTAssertNil(TurnDelta.continuation(in: "x", after: "", turnEnd: T.turnEnd))
    }

    func testContinuationStartsAtTheFirstTurnEndAfterTheSentinel() {
        let text = "a<|im_end|>b\(TurnDelta.assistantSentinel)c<|im_end|>d<|im_end|>"
        XCTAssertEqual(TurnDelta.continuation(in: text, after: TurnDelta.assistantSentinel, turnEnd: T.turnEnd), "<|im_end|>d<|im_end|>")
    }

    func testSentinelTurns() {
        XCTAssertEqual(
            TurnDelta.sentinelTurns(followedBy: [F.user("x")]),
            [ChatTurn(role: .user, text: TurnDelta.userSentinel), ChatTurn(role: .assistant, text: TurnDelta.assistantSentinel), F.user("x")]
        )
    }

    // MARK: Overlap

    func testOverlapWithTheLedgerEnd() throws {
        let fake = template()
        let delta = try delta([F.user("Next")])
        XCTAssertEqual(Array(delta.prefix(3)), [T.imEnd, T.newline, T.imStart])

        let base = render([F.user("Hi")]) + fake.encode("Hello")
        // A normal stop fed `<|im_end|>`.
        XCTAssertEqual(TurnDelta.overlap(ledgerTail: (base + [T.imEnd])[...], delta: delta), 1)
        // A rewind to a user turn's start leaves `<|im_end|>\n`.
        XCTAssertEqual(TurnDelta.overlap(ledgerTail: (base + [T.imEnd, T.newline])[...], delta: delta), 2)
        // A cancelled reply ends with content.
        XCTAssertEqual(TurnDelta.overlap(ledgerTail: base[...], delta: delta), 0)
        XCTAssertEqual(TurnDelta.overlap(ledgerTail: (base + fake.encode("\n"))[...], delta: delta), 0)
    }

    func testOverlapLimits() {
        let delta = [2, 10, 1, 117, 115, 101]
        let ledger = [97, 2, 10, 1, 117, 115]
        XCTAssertEqual(TurnDelta.overlap(ledgerTail: ledger[...], delta: delta, maxOverlap: 6), 5)
        XCTAssertEqual(TurnDelta.overlap(ledgerTail: ledger[...], delta: delta), 0)   // 4 tokens can't reach the match
        XCTAssertEqual(TurnDelta.overlap(ledgerTail: ledger[1...], delta: Array(delta.prefix(2))), 0)
        XCTAssertEqual(TurnDelta.overlap(ledgerTail: [2, 10][...], delta: delta), 2)
        XCTAssertEqual(TurnDelta.overlap(ledgerTail: [][...], delta: delta), 0)
        XCTAssertEqual(TurnDelta.overlap(ledgerTail: ledger[...], delta: []), 0)
        XCTAssertEqual(TurnDelta.overlap(ledgerTail: ledger[...], delta: delta, maxOverlap: 0), 0)
    }

    func testReplacingTheUserTurnGivesTheSameLedgerAsNeverHavingSeenIt() throws {
        let fake = template()
        let firstTurn = render([F.user("Hi")])
        let afterReply = firstTurn + fake.encode("Hello") + [T.imEnd]

        func append(_ newTurns: [ChatTurn], to ledger: [Int]) throws -> [Int] {
            let delta = try delta(newTurns)
            return ledger + delta.dropFirst(TurnDelta.overlap(ledgerTail: ledger[...], delta: delta))
        }

        // The engine answered a tentative turn, then the final transcript replaced it.
        let tentative = try append([F.user("What's the wea")], to: afterReply) + fake.encode("It's")
        let start = try XCTUnwrap(TurnDelta.lastUserTurnStart(in: tentative[...], turnStart: T.imStart))
        XCTAssertTrue(fake.decode(Array(tentative[start...])).hasPrefix("<|im_start|>user\n"))
        let replaced = try append([F.user("What's the weather?")], to: Array(tentative[..<start]))

        XCTAssertEqual(replaced, try append([F.user("What's the weather?")], to: afterReply))
    }

    // MARK: Positions and prefixes

    func testLastUserTurnStart() {
        let fake = template()
        let tokens = render([F.user("Hi"), F.assistant("Hello"), F.user("Again")])
        let start = TurnDelta.lastUserTurnStart(in: tokens[...], turnStart: T.imStart)
        XCTAssertEqual(start.map { fake.decode(Array(tokens[$0...])) }, "<|im_start|>user\n" + LocalSessionPlan.content(of: F.user("Again")) + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        // Indices are absolute in a slice.
        XCTAssertEqual(TurnDelta.lastUserTurnStart(in: tokens[10...], turnStart: T.imStart), start)
        XCTAssertNil(TurnDelta.lastUserTurnStart(in: [1, 5, 6][...], turnStart: T.imStart))
        XCTAssertNil(TurnDelta.lastUserTurnStart(in: [5, 6][...], turnStart: T.imStart))
        XCTAssertEqual(TurnDelta.lastUserTurnStart(in: [1, 5, 1, 6][...], turnStart: T.imStart), 0)
    }

    func testSystemPrefixAndFirstTurns() throws {
        for tools in [true, false] {
            let fake = template(tools: tools)
            let probe = render([ChatTurn(role: .user, text: TurnDelta.userSentinel)], tools: tools, generationPrompt: false)
            let prefix = try XCTUnwrap(TurnDelta.systemPrefix(in: probe, turnStart: T.imStart))
            let prefixText = fake.decode(prefix)
            XCTAssertTrue(prefixText.hasPrefix("<|im_start|>system\n" + system))
            XCTAssertTrue(prefixText.hasSuffix("<|im_end|>\n"))
            XCTAssertFalse(prefixText.contains(TurnDelta.userSentinel))

            let window = [F.user("Hi"), F.assistant("Hello"), F.user("Again")]
            let full = render(window, tools: tools)
            let rest = try XCTUnwrap(TurnDelta.dropPrefix(prefix, from: full))
            XCTAssertEqual(prefix + rest, full)
            XCTAssertEqual(rest.first, T.imStart)
        }
    }

    func testSystemPrefixNeedsATurnStartAfterSomething() {
        XCTAssertNil(TurnDelta.systemPrefix(in: [1, 117, 2], turnStart: T.imStart))
        XCTAssertNil(TurnDelta.systemPrefix(in: [97, 98], turnStart: T.imStart))
        XCTAssertEqual(TurnDelta.systemPrefix(in: [1, 115, 2, 1, 117], turnStart: T.imStart), [1, 115, 2])
    }

    func testDropPrefix() {
        XCTAssertEqual(TurnDelta.dropPrefix([1, 2], from: [1, 2, 3]), [3])
        XCTAssertEqual(TurnDelta.dropPrefix([1, 2, 3][..<2], from: [1, 2]), [])
        XCTAssertEqual(TurnDelta.dropPrefix([Int](), from: [1]), [1])
        XCTAssertNil(TurnDelta.dropPrefix([1, 3], from: [1, 2, 3]))
        XCTAssertNil(TurnDelta.dropPrefix([1, 2, 3], from: [1, 2]))
    }

    // MARK: The fake template mirrors WP02's

    func testFakeTemplateRoundTrips() {
        let fake = template()
        let text = "<|im_start|>user\nHi <think>x</think>\n<|im_end|>"
        XCTAssertEqual(fake.decode(fake.encode(text)), text)
        XCTAssertEqual(fake.encode("<|im_end|>\n"), [T.imEnd, T.newline])
        XCTAssertEqual(fake.encode("é"), [T.unknown])
    }
}
