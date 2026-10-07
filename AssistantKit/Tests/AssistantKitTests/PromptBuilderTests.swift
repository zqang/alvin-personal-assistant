import XCTest
@testable import AssistantKit

final class PromptBuilderTests: XCTestCase {
    func testTurnsSkipFailedRepliesAndMergeUserMessages() throws {
        let start = Date(timeIntervalSince1970: 0)
        let zone = try XCTUnwrap(TimeZone(identifier: "Asia/Tokyo"))
        let messages = [
            StoredMessage(role: .assistant, text: "Hi! How can I help?", createdAt: start, isVoice: false),
            StoredMessage(role: .user, text: "What's the time?", createdAt: start, isVoice: true),
            StoredMessage(role: .assistant, text: "", createdAt: start, isVoice: true, status: .failed),
            StoredMessage(role: .user, text: "Hello?", createdAt: start.addingTimeInterval(60), isVoice: true),
            StoredMessage(role: .assistant, text: "It's nine o'clock", createdAt: start, isVoice: true, status: .interrupted),
            StoredMessage(role: .user, text: "Thanks", createdAt: start.addingTimeInterval(120), isVoice: false),
            StoredMessage(role: .assistant, text: "Sorry, I can't", createdAt: start, isVoice: false, status: .refused),
        ]
        let turns = PromptBuilder.turns(from: messages, timeZone: zone)

        XCTAssertEqual(turns.map(\.role), [.user, .assistant, .user])
        XCTAssertEqual(turns[0].text, "What's the time?\n\nHello?")
        XCTAssertEqual(turns[0].context, "<context>time: Thursday 1 January 1970, 09:01 Asia/Tokyo; input: spoken</context>")
        XCTAssertEqual(turns[1].text, "It's nine o'clock")
        XCTAssertNil(turns[1].context)
        XCTAssertEqual(turns[2].text, "Thanks")
        XCTAssertEqual(
            turns[2].context,
            "<context>time: Thursday 1 January 1970, 09:02 Asia/Tokyo; input: typed; note: the user interrupted your previous reply</context>"
        )
    }

    func testTurnsAreStableAsTheConversationGrows() throws {
        let zone = try XCTUnwrap(TimeZone(identifier: "Europe/London"))
        var messages = [
            StoredMessage(role: .user, text: "One", createdAt: Date(timeIntervalSince1970: 1_000), isVoice: true),
            StoredMessage(role: .assistant, text: "Two", createdAt: Date(timeIntervalSince1970: 1_001), isVoice: true),
            StoredMessage(role: .user, text: "Three", createdAt: Date(timeIntervalSince1970: 1_002), isVoice: false),
        ]
        let before = PromptBuilder.turns(from: messages, timeZone: zone)
        messages.append(StoredMessage(role: .assistant, text: "Four", createdAt: Date(timeIntervalSince1970: 1_003), isVoice: false))
        messages.append(StoredMessage(role: .user, text: "Five", createdAt: Date(timeIntervalSince1970: 1_004), isVoice: false))
        let after = PromptBuilder.turns(from: messages, timeZone: zone)
        XCTAssertEqual(Array(after.prefix(before.count)), before, "earlier turns must not change, or the prompt cache misses")
    }

    func testSystemPromptPersonalization() {
        let prompt = PromptBuilder.systemPrompt(userName: "Alvin", customInstructions: " I live in Singapore. ")
        XCTAssertTrue(prompt.contains("personal assistant for Alvin"))
        XCTAssertTrue(prompt.hasSuffix("<user_instructions>\nI live in Singapore.\n</user_instructions>"))
        let plain = PromptBuilder.systemPrompt(userName: "", customInstructions: "")
        XCTAssertFalse(plain.contains("user_instructions"))
        XCTAssertTrue(plain.contains("personal assistant, running"))
    }

    private func round(_ id: String, summary: String? = nil) -> ToolRound {
        ToolRound(calls: [ToolCallRecord(id: id, name: "create_reminder", input: ["title": "x"], result: #"{"id":"r"}"#, summary: summary)])
    }

    func testFailedReplyWithRoundsIsKeptWithoutItsText() {
        let start = Date(timeIntervalSince1970: 0)
        let actions = round("toolu_1", summary: "Reminder: x")
        let messages = [
            StoredMessage(role: .user, text: "Remind me", createdAt: start, isVoice: false),
            StoredMessage(role: .assistant, text: "I've added", createdAt: start, isVoice: false, status: .failed, toolRounds: [actions]),
            StoredMessage(role: .user, text: "Did it work?", createdAt: start, isVoice: false),
        ]
        let turns = PromptBuilder.turns(from: messages, timeZone: TimeZone(identifier: "UTC")!)
        XCTAssertEqual(turns.map(\.role), [.user, .assistant, .user])
        XCTAssertEqual(turns[1].text, "", "a failed reply's text is never sent back")
        XCTAssertEqual(turns[1].toolRounds, [actions], "its actions happened, so the model must see them")
        XCTAssertEqual(turns[0].text, "Remind me", "the user turns are no longer merged")
        XCTAssertEqual(turns[2].text, "Did it work?")

        let refused = [
            StoredMessage(role: .user, text: "Q", createdAt: start, isVoice: false),
            StoredMessage(role: .assistant, text: "Partial", createdAt: start, isVoice: false, status: .refused, toolRounds: [actions]),
            StoredMessage(role: .assistant, text: "Streaming", createdAt: start, isVoice: false, status: .streaming),
            StoredMessage(role: .user, text: "Q2", createdAt: start, isVoice: false),
        ]
        let refusedTurns = PromptBuilder.turns(from: refused, timeZone: TimeZone(identifier: "UTC")!)
        XCTAssertEqual(refusedTurns.map(\.role), [.user, .assistant, .user])
        XCTAssertEqual(refusedTurns[1], ChatTurn(role: .assistant, text: "", toolRounds: [actions]))
    }

    func testAssistantMergeRules() {
        let start = Date(timeIntervalSince1970: 0)
        let first = round("toolu_1")
        let second = round("toolu_2")
        let messages = [
            StoredMessage(role: .user, text: "Do things", createdAt: start, isVoice: true),
            StoredMessage(role: .assistant, text: "  ", createdAt: start, isVoice: true),
            StoredMessage(role: .assistant, text: "Working.", createdAt: start, isVoice: true),
            StoredMessage(role: .assistant, text: "lost", createdAt: start, isVoice: true, status: .failed, toolRounds: [first]),
            StoredMessage(role: .assistant, text: "", createdAt: start, isVoice: true, status: .failed),
            StoredMessage(role: .assistant, text: "All done", createdAt: start, isVoice: true, status: .interrupted, toolRounds: [second]),
            StoredMessage(role: .user, text: "Thanks", createdAt: start.addingTimeInterval(60), isVoice: true),
        ]
        let turns = PromptBuilder.turns(from: messages, timeZone: TimeZone(identifier: "Asia/Tokyo")!)
        XCTAssertEqual(turns.count, 3)
        XCTAssertEqual(turns[1].role, .assistant)
        XCTAssertEqual(turns[1].text, "Working.\n\nAll done", "non-empty texts join with a blank line")
        XCTAssertEqual(turns[1].toolRounds, [first, second], "rounds concatenate in order")
        XCTAssertNil(turns[1].context)
        XCTAssertEqual(turns[2].context, "<context>time: Thursday 1 January 1970, 09:01 Asia/Tokyo; input: spoken; note: the user interrupted your previous reply</context>")

        let roundsOnly = [
            StoredMessage(role: .user, text: "A", createdAt: start, isVoice: false),
            StoredMessage(role: .assistant, text: "", createdAt: start, isVoice: false, toolRounds: [first]),
            StoredMessage(role: .assistant, text: "", createdAt: start, isVoice: false, status: .failed, toolRounds: [second]),
            StoredMessage(role: .user, text: "B", createdAt: start, isVoice: false),
        ]
        let merged = PromptBuilder.turns(from: roundsOnly, timeZone: TimeZone(identifier: "UTC")!)
        XCTAssertEqual(merged.map(\.text), ["A", "", "B"])
        XCTAssertEqual(merged[1].toolRounds, [first, second])

        let trailing = Array(roundsOnly.prefix(2))
        XCTAssertEqual(PromptBuilder.turns(from: trailing, timeZone: TimeZone(identifier: "UTC")!).map(\.role), [.user], "a trailing reply is still trimmed")
    }

    func testSystemPromptAsksForToolsBeforeSpeaking() {
        let prompt = PromptBuilder.systemPrompt(userName: "", customInstructions: "")
        XCTAssertTrue(prompt.hasSuffix(
            "Keep responses focused, brief, and concise to avoid overwhelming the person. Latency-sensitive: begin your visible answer immediately, unless you need a tool first; then call it before saying anything. Don't narrate tool use; the app plays a short cue. After an action, confirm the outcome in one short sentence."
        ))
        XCTAssertFalse(prompt.contains("Latency-sensitive; begin your visible answer immediately."))
        XCTAssertEqual(PromptBuilder.systemPrompt(userName: "", customInstructions: ""), prompt, "the prompt is static, so it caches")
    }

    func testLocalSystemPrompt() {
        let base = PromptBuilder.systemPrompt(userName: "Alvin", customInstructions: "")
        XCTAssertEqual(
            PromptBuilder.localSystemPrompt(base: base, handoffAvailable: false),
            base + "\n\nYou run on the user's iPhone. Use your tools for reminders, calendar, timers and the current time."
        )
        XCTAssertEqual(
            PromptBuilder.localSystemPrompt(base: "BASE", handoffAvailable: true),
            "BASE\n\nYou run on the user's iPhone. Use your tools for reminders, calendar, timers and the current time. If a request needs the internet (news, weather, prices, scores) or deep expertise, call handoff_to_cloud before saying anything."
        )
    }

    func testSettingsDecodeWithMissingAndUnknownValues() throws {
        let json = #"{"provider":"somethingNew","claudeModel":"claude-sonnet-5","speechRate":1.2}"#
        let settings = try JSONDecoder().decode(AssistantSettings.self, from: Data(json.utf8))
        XCTAssertEqual(settings.provider, .anthropic)
        XCTAssertEqual(settings.claudeModel, "claude-sonnet-5")
        XCTAssertEqual(settings.speechRate, 1.2)
        XCTAssertEqual(settings.voiceEngine, .apple)
        XCTAssertEqual(settings.effort, "low")

        let roundTrip = try JSONDecoder().decode(AssistantSettings.self, from: JSONEncoder().encode(settings))
        XCTAssertEqual(roundTrip, settings)
    }
}
