import AssistantKit
import AVFoundation
import Foundation
import XCTest
@testable import VoiceKit

// A host of the smallest shape (Roam's): plain message and conversation types, a reply pipeline,
// and a host with no second transcription and no photo reader. An app shows voice mode with
// `VoiceModeView(session: VoiceSession(conversation: chat, pipeline: Pipeline(), host: Host()))`.

@MainActor
final class Message: VoiceMessage {
    let role: ChatRole
    var text: String
    var createdAt = Date()
    let isVoice: Bool
    var status: StoredMessage.Status
    var errorText: String?
    var toolRounds: [ToolRound] = []
    var photoData: Data?
    var photoNote: String?

    init(role: ChatRole, text: String, isVoice: Bool, status: StoredMessage.Status) {
        self.role = role
        self.text = text
        self.isVoice = isVoice
        self.status = status
    }

    var stored: StoredMessage {
        StoredMessage(role: role, text: text, createdAt: createdAt, isVoice: isVoice, status: status, toolRounds: toolRounds, photoNote: photoNote)
    }
}

@MainActor
final class Chat: VoiceConversation {
    var title = ""
    private(set) var messages: [Message] = []

    var voiceMessages: [any VoiceMessage] { messages }

    func appendMessage(role: ChatRole, text: String, isVoice: Bool, status: StoredMessage.Status) -> any VoiceMessage {
        let message = Message(role: role, text: text, isVoice: isVoice, status: status)
        messages.append(message)
        if title.isEmpty, role == .user { title = text }
        return message
    }

    func remove(_ message: any VoiceMessage) {
        messages.removeAll { $0 === message }
    }

    func userTurnRevised(from old: String, to new: String) {
        if title == old { title = new }
    }

    func save() {}
}

/// Answers every turn by repeating it.
@MainActor
final class Pipeline: ReplyPipeline {
    /// What `missingSetup()` reports.
    var problem: String?
    private(set) var prewarms = 0

    var supportsEarlyStart: Bool { false }

    func stream(_ request: ReplyRequest) -> AsyncThrowingStream<AssistantEvent, Error> {
        let heard = request.messages.last { $0.role == .user }?.text ?? ""
        return AsyncThrowingStream { continuation in
            continuation.yield(.reply(.text("You said: \(heard)")))
            continuation.yield(.reply(.finished(.completed)))
            continuation.finish()
        }
    }

    func missingSetup() -> String? { problem }

    func prewarm(inputIsVoice: Bool) { prewarms += 1 }

    func displayName(for decision: RouteDecision?) -> String { "Echo" }
}

@MainActor
final class Host: VoiceSessionHost {
    var settings = AssistantSettings()
}

@MainActor
final class VoiceKitTests: XCTestCase {
    func testAHostWithoutHooksGetsTheDefaults() async {
        let host: any VoiceSessionHost = Host()
        XCTAssertEqual(host.openAIKey, "")
        XCTAssertEqual(host.refineClipSamples, 0)
        XCTAssertNil(host.photoReader)
        let refined = await host.refineTurn([0, 0], locale: "en-US")
        XCTAssertNil(refined)
        host.voiceSessionWillStart(host.settings)
        host.voiceSessionDidStop()
    }

    func testStartWithMissingSetupFailsBeforeTouchingAudio() async {
        let audioSession = AVAudioSession.sharedInstance()
        let category = audioSession.category
        let chat = Chat()
        let pipeline = Pipeline()
        pipeline.problem = "Add your API key in Settings to start."
        let host = CountingHost()
        let session = VoiceSession(conversation: chat, pipeline: pipeline, host: host)

        await session.start()

        XCTAssertEqual(session.phase, .failed("Add your API key in Settings to start."))
        XCTAssertEqual(audioSession.category, category)
        XCTAssertEqual(pipeline.prewarms, 0)
        XCTAssertTrue(chat.messages.isEmpty)
        // The host heard of no session, so it hears of no stop either, even when the screen
        // goes away and stops the failed session again.
        session.stop()
        XCTAssertEqual(host.starts, 0)
        XCTAssertEqual(host.stops, 0)
    }

    func testReplyOutcomeRecordsHowAReplyEnded() {
        let chat = Chat()
        let refused = chat.appendMessage(role: .assistant, text: "Partial", isVoice: true, status: .streaming)
        ReplyOutcome.apply(.refused(category: nil), to: refused)
        XCTAssertEqual(refused.text, "")
        XCTAssertEqual(refused.status, .refused)
        XCTAssertEqual(refused.errorText, "The model declined to answer this. Try rephrasing.")

        let finished = chat.appendMessage(role: .assistant, text: "Done", isVoice: true, status: .streaming)
        ReplyOutcome.apply(.truncated, to: finished)
        XCTAssertEqual(finished.status, .complete)
        XCTAssertEqual(finished.text, "Done")

        // Stopped: what was heard stays, trimmed.
        let heard = chat.appendMessage(role: .assistant, text: "Half a sentence and more", isVoice: true, status: .streaming)
        XCTAssertTrue(ReplyOutcome.keepStopped(heard, text: " Half a sentence ", in: chat))
        XCTAssertEqual(heard.text, "Half a sentence")
        XCTAssertEqual(heard.status, .interrupted)

        // Stopped with no text but after its tools acted: it stays in the history.
        let acted = chat.appendMessage(role: .assistant, text: "", isVoice: true, status: .streaming)
        acted.toolRounds = [ToolRound(calls: [])]
        XCTAssertTrue(ReplyOutcome.keepStopped(acted, in: chat))
        XCTAssertEqual(acted.status, .interrupted)

        // Stopped before anything arrived: it goes.
        let empty = chat.appendMessage(role: .assistant, text: "  ", isVoice: true, status: .streaming)
        XCTAssertFalse(ReplyOutcome.keepStopped(empty, in: chat))
        XCTAssertFalse(chat.messages.contains { $0 === empty })
        XCTAssertEqual(chat.messages.count, 4)
    }

    /// The two spoken apologies are in the speech language, as the cues are.
    func testApologiesAreSpokenInTheSpeechLanguage() {
        XCTAssertEqual(VoiceSession.apology(refused: false, localeIdentifier: "en-US"), "Sorry, something went wrong.")
        XCTAssertEqual(VoiceSession.apology(refused: true, localeIdentifier: "en-GB"), "Sorry, I can't help with that one.")
        XCTAssertEqual(VoiceSession.apology(refused: false, localeIdentifier: "zh-CN"), "抱歉，出了点问题。")
        XCTAssertEqual(VoiceSession.apology(refused: true, localeIdentifier: "zh_TW"), "抱歉，这个我帮不上忙。")
    }

    /// The screen's text follows the speech language when the app declares it, whatever the
    /// iPhone's language: Roam's own setting wins, and an English-only app stays English.
    func testScreenTextFollowsTheSpeechLanguageTheAppDeclares() {
        let roam = ["en", "zh-Hans"]
        XCTAssertEqual(Bundle.voiceKitStrings(for: "zh-CN", appLanguages: roam).bundleURL.lastPathComponent, "zh-Hans.lproj")
        // English is the catalog's en.lproj too, never left to the iPhone's language.
        XCTAssertEqual(Bundle.voiceKitStrings(for: "en-US", appLanguages: roam).bundleURL.lastPathComponent, "en.lproj")
        XCTAssertEqual(Bundle.voiceKitStrings(for: "zh-CN", appLanguages: ["en"]).bundleURL.lastPathComponent, "en.lproj")
        XCTAssertEqual(String(localized: "Thinking…", bundle: Bundle.voiceKitStrings(for: "zh-CN", appLanguages: roam)), "正在思考…")
        XCTAssertEqual(String(localized: "Couldn't start the microphone: \("x")", bundle: Bundle.voiceKitStrings(for: "zh-CN", appLanguages: roam)), "无法启动麦克风：x")

        // A session points VoiceKit's strings at its host's speech language, in an app that
        // declares it (the test runner declares only English).
        let saved = (Bundle.voiceKit, Bundle.voiceKitAppLanguages)
        defer {
            Bundle.voiceKitCurrent.withLock { $0 = saved.0 }
            Bundle.voiceKitAppLanguages = saved.1
        }
        Bundle.voiceKitAppLanguages = roam
        let host = Host()
        host.settings.speechLocale = "zh-CN"
        _ = VoiceSession(conversation: Chat(), pipeline: Pipeline(), host: host)
        XCTAssertEqual(Bundle.voiceKit.bundleURL.lastPathComponent, "zh-Hans.lproj")
        host.settings.speechLocale = "en-US"
        _ = VoiceSession(conversation: Chat(), pipeline: Pipeline(), host: host)
        XCTAssertEqual(Bundle.voiceKit.bundleURL.lastPathComponent, "en.lproj")
    }

    func testScreenTextMatchesRegionalAppLocalizations() {
        let cases: [(app: [String], speech: String, expected: String)] = [
            (["en-US", "zh-Hans"], "en-US", "en.lproj"),
            (["en", "zh-CN"], "zh-CN", "zh-Hans.lproj"),
            (["en-GB", "zh-SG"], "en-US", "en.lproj"),
            (["en-GB", "zh-SG"], "zh-CN", "zh-Hans.lproj"),
            (["English", "zh-Hans"], "en-US", "en.lproj"),
        ]
        for item in cases {
            let bundle = Bundle.voiceKitStrings(for: item.speech, appLanguages: item.app)
            XCTAssertEqual(bundle.bundleURL.lastPathComponent, item.expected, "app=\(item.app), speech=\(item.speech)")
        }
    }

    func testRegionalEnglishOnlyAppDoesNotEnableChineseText() {
        let bundle = Bundle.voiceKitStrings(for: "zh-CN", appLanguages: ["en-US"])
        XCTAssertEqual(bundle.bundleURL.lastPathComponent, "en.lproj")
        XCTAssertEqual(String(localized: "Thinking…", bundle: bundle), "Thinking…")
    }

    func testTraditionalChineseDeclarationDoesNotEnableSimplifiedChineseText() {
        for chinese in ["zh-TW", "zh-Hant", "zh-HK"] {
            let bundle = Bundle.voiceKitStrings(for: "zh-CN", appLanguages: ["en-US", chinese])
            XCTAssertEqual(bundle.bundleURL.lastPathComponent, "en.lproj", chinese)
        }
    }

    /// Every string in the catalog has Simplified Chinese, and the built bundle serves it.
    func testEveryCatalogKeyHasChinese() throws {
        struct Catalog: Decodable {
            struct Entry: Decodable { var localizations: [String: Localization]? }
            struct Localization: Decodable { var stringUnit: Unit }
            struct Unit: Decodable { var state: String; var value: String }
            var strings: [String: Entry]
        }
        let source = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("../../Sources/VoiceKit/Localizable.xcstrings").standardized
        let catalog = try JSONDecoder().decode(Catalog.self, from: Data(contentsOf: source))
        let chinese = try XCTUnwrap(Bundle.module.path(forResource: "zh-Hans", ofType: "lproj").flatMap(Bundle.init(path:)))
        XCTAssertFalse(catalog.strings.isEmpty)
        for (key, entry) in catalog.strings {
            let unit = try XCTUnwrap(entry.localizations?["zh-Hans"]?.stringUnit, key)
            XCTAssertEqual(unit.state, "translated", key)
            XCTAssertEqual(unit.value.components(separatedBy: "%@").count, key.components(separatedBy: "%@").count, key)
            XCTAssertEqual(chinese.localizedString(forKey: key, value: nil, table: nil), unit.value, key)
        }
    }
}

/// Counts the lifecycle hooks.
@MainActor
private final class CountingHost: VoiceSessionHost {
    var settings = AssistantSettings()
    private(set) var starts = 0
    private(set) var stops = 0

    func voiceSessionWillStart(_ settings: AssistantSettings) { starts += 1 }
    func voiceSessionDidStop() { stops += 1 }
}
