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
