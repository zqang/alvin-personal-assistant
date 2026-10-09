import XCTest
@testable import AssistantKit

final class TurnReopenPolicyTests: XCTestCase {
    private let timer = ToolRound(calls: [
        ToolCallRecord(id: "toolu_1", name: "set_timer", input: ["seconds": 600], result: #"{"id":"t"}"#, summary: "Timer: 10 min"),
    ])

    /// Whether the turn may be taken back after each of `events`, in order.
    private func canReopen(after events: [AssistantEvent]) -> [Bool] {
        var policy = TurnReopenPolicy()
        return events.map { event in
            policy.received(event)
            return policy.canReopen
        }
    }

    func testANewReplyAllowsIt() {
        XCTAssertTrue(TurnReopenPolicy().canReopen)
    }

    func testAReplyWithoutToolCallsAllowsIt() {
        let events: [AssistantEvent] = [
            .routed(RouteDecision(engine: .cloud, reason: .cloudDefault, fallback: .local)),
            .progress(.responseStarted),
            .progress(.firstToken),
            .cue(.lookingUp),
            .reply(.activity("Searching the web")),
            .reply(.activity(nil)),
            .reply(.text("It's sunny.")),
            .reply(.finished(.completed)),
        ]
        XCTAssertEqual(canReopen(after: events), Array(repeating: true, count: events.count))
    }

    func testAToolCallEndsIt() {
        // Claude names a call as it starts it; the on-device model reports the call first, then its name.
        XCTAssertEqual(canReopen(after: [.progress(.toolCallStarted(name: "set_timer")), .reply(.text("Done."))]), [false, false])
        XCTAssertEqual(
            canReopen(after: [.progress(.toolCallStarted(name: nil)), .progress(.toolCallStarted(name: "create_reminder"))]),
            [true, false]
        )
    }

    func testAToolRoundEndsIt() {
        XCTAssertEqual(canReopen(after: [.toolRound(timer), .reply(.text("Timer set."))]), [false, false])
    }

    func testAHandoffAllowsItUntilTheCloudCallsATool() {
        let events: [AssistantEvent] = [
            .routed(RouteDecision(engine: .local, reason: .localFirst, fallback: .cloud)),
            .progress(.toolCallStarted(name: nil)),
            .progress(.toolCallStarted(name: HandoffTool.name)),
            .reply(.activity(nil)),
            .routed(RouteDecision(engine: .cloud, reason: .escalated)),
            .cue(.handingOff),
            .progress(.toolCallStarted(name: "set_timer")),
        ]
        XCTAssertEqual(canReopen(after: events), [true, true, true, true, true, true, false])
    }

    /// A turn that stands keeps the reply that acted on it, and the new words follow as the next
    /// turn, so the next request shows the model what it already did.
    func testATurnThatStandsKeepsItsActionInTheNextRequest() throws {
        let start = Date(timeIntervalSince1970: 0)
        let messages = [
            StoredMessage(role: .user, text: "Set a timer for ten minutes", createdAt: start, isVoice: true),
            StoredMessage(role: .assistant, text: "", createdAt: start, isVoice: true, status: .interrupted, toolRounds: [timer]),
            StoredMessage(role: .user, text: "for the pasta", createdAt: start.addingTimeInterval(2), isVoice: true),
        ]
        let turns = PromptBuilder.turns(from: messages, timeZone: try XCTUnwrap(TimeZone(identifier: "UTC")))
        XCTAssertEqual(turns.map(\.role), [.user, .assistant, .user])
        XCTAssertEqual(turns[0].text, "Set a timer for ten minutes")
        XCTAssertEqual(turns[1].toolRounds, [timer])
        XCTAssertEqual(turns[2].text, "for the pasta")
        XCTAssertEqual(turns[2].context?.hasSuffix("note: the user interrupted your previous reply</context>"), true)
    }
}
