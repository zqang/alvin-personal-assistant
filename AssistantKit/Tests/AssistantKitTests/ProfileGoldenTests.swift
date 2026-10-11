import AlvinProfile
import XCTest
@testable import AssistantKit

/// This app's prompt text and phrase tables (`AlvinProfile`), pinned byte for byte as they were
/// before the packages took a per-app profile (captured at 248efe3). Claude caches the system prompt, the iPhone prewarms the
/// on-device prefix, and the on-device tool benchmark (12/12) was measured on exactly these bytes,
/// so any drift is a regression even when the wording still reads the same.
final class ProfileGoldenTests: XCTestCase {
    private let profile = AlvinProfile.profile

    func testSystemPromptIsUnchanged() {
        XCTAssertEqual(PromptBuilder.systemPrompt(userName: " Alvin ", customInstructions: " I live in Singapore. ", profile: profile), Self.namedSystemPrompt)
        XCTAssertEqual(PromptBuilder.systemPrompt(userName: "", customInstructions: "", profile: profile), Self.anonymousSystemPrompt)
    }

    func testLocalSystemPromptIsUnchanged() {
        XCTAssertEqual(PromptBuilder.localSystemPrompt(base: "BASE", handoffAvailable: false, profile: profile), "BASE\n\n" + Self.localToolHint)
        XCTAssertEqual(PromptBuilder.localSystemPrompt(base: "BASE", handoffAvailable: true, profile: profile), "BASE\n\n" + Self.localToolHint + " " + Self.localHandoffHint)
    }

    func testHandoffDescriptionIsUnchanged() {
        XCTAssertEqual(profile.handoffTool.definition.description, Self.handoffDescription)
    }

    func testActionPhrasesAreUnchanged() {
        XCTAssertEqual(profile.intents.actionPhrases.sorted(), Self.actionPhrases)
        XCTAssertEqual(profile.intents.verbs.sorted(), Self.verbs)
    }

    /// The claim table holds 305 generated phrases, so it is pinned by count and an FNV-1a hash
    /// of the sorted phrases joined with "|".
    func testClaimPhrasesAreUnchanged() {
        let phrases = profile.claims.phrases.sorted()
        XCTAssertEqual(phrases.count, 305)
        XCTAssertEqual(Self.fnv1a(phrases.joined(separator: "|")), 0xa20d69c20f10cf61)
        XCTAssertEqual(profile.claims.allParticiples.sorted(), Self.participles)
    }

    /// A profile with the defaults belongs to no app: nothing is an action, nothing is a claim of
    /// one, every call may run, and the prompts name no device or domain.
    func testDefaultProfileHasNoDomain() {
        let neutral = AssistantProfile()
        XCTAssertFalse(neutral.intents.classify("Remind me at 5 to call mum.").contains(.deviceAction))
        XCTAssertFalse(neutral.intents.classify("帮我设一个二十分钟的计时器。").contains(.deviceAction))
        XCTAssertFalse(neutral.claims.claimsDone("Reminder set."))
        XCTAssertTrue(neutral.claims.claimsDone("I've added it."))
        XCTAssertNil(neutral.callGuard.check(PendingToolCall(id: "1", name: "set_timer", input: ["seconds": 1]), "How long is left?"))
        XCTAssertTrue(neutral.contextLookups.isEmpty)
        let system = PromptBuilder.systemPrompt(userName: "Sam", customInstructions: "", profile: neutral)
        XCTAssertTrue(system.hasPrefix("You are a warm, capable assistant for Sam. The user talks to you by voice or types.\n\n"))
        let local = PromptBuilder.localSystemPrompt(base: system, handoffAvailable: true, profile: neutral) + neutral.handoffDescription
        for word in ["iPhone", "personal", "reminder", "timer", "calendar", "weather"] {
            XCTAssertFalse(local.localizedCaseInsensitiveContains(word), word)
        }
    }

    /// An app's own phrases are matched like the shared ones: lowercased, with verb endings for its
    /// verbs.
    func testAppPhrasesAreNormalizedAndInflected() {
        let intents = IntentClassifier(actionPhrases: ["Publish", "My Post"], actionVerbs: ["publish"])
        XCTAssertTrue(intents.classify("publishing it now").contains(.deviceAction))
        XCTAssertTrue(intents.classify("Read me my posts").contains(.deviceAction))
        XCTAssertFalse(intents.classify("The publisher called").contains(.deviceAction))
        XCTAssertTrue(intents.classify("Write an in-depth post").contains(.explicitDepth))
        let noDepth = IntentClassifier(explicitDepthPhrases: [])
        XCTAssertFalse(noDepth.classify("Write an in-depth post").contains(.explicitDepth))
        let claims = SuccessClaim(nouns: ["Post"], participles: ["Published"])
        XCTAssertTrue(claims.claimsDone("Post published."))
        XCTAssertTrue(claims.claimsDone("Published your draft."))
    }

    static func fnv1a(_ text: String) -> UInt64 {
        text.utf8.reduce(0xcbf2_9ce4_8422_2325) { ($0 ^ UInt64($1)) &* 0x100_0000_01b3 }
    }

    static let namedSystemPrompt = """
        You are a warm, capable personal assistant for Alvin, running as an iPhone app. The user talks to you by voice or types.

        Each user message starts with a <context> tag giving the local time it was sent and whether it was spoken or typed. Use the time when it matters, and never mention the tag.

        When the input is spoken, your reply is read aloud by a text-to-speech voice:
        - Talk like a thoughtful person in conversation: natural, warm, and to the point. Usually one to three short sentences; go longer only when asked for detail, a story, or step-by-step help.
        - Write plain spoken sentences only: no Markdown, lists, headings, tables, code, emoji, or URLs. Say numbers, dates, units, and symbols the way you would read them aloud.
        - The words come from speech recognition and may contain errors. Infer the likely meaning, and ask a short clarifying question only when it really matters.
        - If the user interrupted your previous reply, respond to what they said rather than repeating yourself.

        When the input is typed, you may use light Markdown and go into more detail when it helps.

        Reply in the language the user is using. When you search the web, answer in your own words; don't read out links or source names unless asked.

        Keep responses focused, brief, and concise to avoid overwhelming the person. Latency-sensitive: begin your visible answer immediately, unless you need a tool first; then call it before saying anything. Don't narrate tool use; the app plays a short cue. After an action, confirm the outcome in one short sentence.

        <user_instructions>
        I live in Singapore.
        </user_instructions>
        """

    static let anonymousSystemPrompt = """
        You are a warm, capable personal assistant, running as an iPhone app. The user talks to you by voice or types.

        Each user message starts with a <context> tag giving the local time it was sent and whether it was spoken or typed. Use the time when it matters, and never mention the tag.

        When the input is spoken, your reply is read aloud by a text-to-speech voice:
        - Talk like a thoughtful person in conversation: natural, warm, and to the point. Usually one to three short sentences; go longer only when asked for detail, a story, or step-by-step help.
        - Write plain spoken sentences only: no Markdown, lists, headings, tables, code, emoji, or URLs. Say numbers, dates, units, and symbols the way you would read them aloud.
        - The words come from speech recognition and may contain errors. Infer the likely meaning, and ask a short clarifying question only when it really matters.
        - If the user interrupted your previous reply, respond to what they said rather than repeating yourself.

        When the input is typed, you may use light Markdown and go into more detail when it helps.

        Reply in the language the user is using. When you search the web, answer in your own words; don't read out links or source names unless asked.

        Keep responses focused, brief, and concise to avoid overwhelming the person. Latency-sensitive: begin your visible answer immediately, unless you need a tool first; then call it before saying anything. Don't narrate tool use; the app plays a short cue. After an action, confirm the outcome in one short sentence.
        """

    static let localToolHint = "You run on the user's iPhone. Use your tools for reminders, calendar, timers and the current time."

    static let localHandoffHint = "If a request needs the internet (news, weather, prices, scores) or deep expertise, call handoff_to_cloud before saying anything."

    static let handoffDescription = "Hand this request to the more capable cloud assistant, which sees the whole conversation. Call this first, instead of answering or calling another tool, when the request needs the internet (news, weather, prices, scores, opening hours), asks about something none of your other tools can read or do (for example cancelling a timer, when your tools can only start and list timers), or needs long, careful reasoning. Never call another tool as a stand-in for a request it doesn't fit."

    static let actionPhrases = ["alarm", "am i free", "appointment", "calendar", "meeting", "remind me", "reminder", "schedule", "timer", "to-do", "todo", "wake me", "what's on", "会议", "安排", "定时", "开会", "待办", "提醒", "日历", "日程", "计时", "闹钟"]

    static let verbs = ["compare", "google", "plan", "schedule", "score", "search"]

    static let participles = ["added", "booked", "created", "done", "saved", "scheduled", "set", "started"]
}
