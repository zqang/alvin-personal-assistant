import AssistantKit
import SwiftUI
import XCTest
@testable import VoiceKit

/// The character in the orb's place, and the voice settings around it.
@MainActor
final class AvatarTests: XCTestCase {
    func testStandardStyleKeepsTheOrbAndDark() {
        XCTAssertEqual(VoiceModeStyle.standard.avatar, .orb)
        XCTAssertEqual(VoiceModeStyle.standard.colorScheme, .dark)
    }

    func testSpeechServerSettingDecodesAndDefaultsEmpty() throws {
        XCTAssertEqual(AssistantSettings().openAISpeechBaseURL, "")
        let json = #"{"voiceEngine":"openAI","openAISpeechBaseURL":"http://100.90.33.96:3900/v1","openAIVoice":"alvin"}"#
        let settings = try JSONDecoder().decode(AssistantSettings.self, from: Data(json.utf8))
        XCTAssertEqual(settings.openAISpeechBaseURL, "http://100.90.33.96:3900/v1")
        XCTAssertEqual(settings.openAIVoice, "alvin")
    }

    func testPersonalVoicesAreKeptApartFromSystemVoices() {
        for voice in VoiceCatalog.voices(forLanguage: "en-US") {
            XCTAssertFalse(voice.voiceTraits.contains(.isPersonalVoice))
        }
        for voice in VoiceCatalog.personalVoices(forLanguage: "en-US") {
            XCTAssertTrue(voice.voiceTraits.contains(.isPersonalVoice))
        }
    }

    /// Renders every mood of the face at both sizes the screen uses, on the dark page and on paper. With
    /// `VOICEKIT_RENDER_DIR` set, the sheets are written there as PNGs for a design review.
    func testFaceRendersEveryMood() throws {
        let moods: [OrbMood] = [.idle, .listening, .thinking, .speaking, .error]
        for scheme in [ColorScheme.dark, .light] {
            let sheet = HStack(spacing: 24) {
                ForEach(moods, id: \.self) { mood in
                    VStack(spacing: 12) {
                        FaceView(level: 0.6, mood: mood).frame(width: 250, height: 250)
                        FaceView(level: 0.6, mood: mood).frame(width: 96, height: 96)
                    }
                }
            }
            .padding(32)
            .background(scheme == .dark ? Color(red: 0.11, green: 0.1, blue: 0.09) : Color(red: 0.96, green: 0.95, blue: 0.9))
            .environment(\.colorScheme, scheme)
            .environment(\.voiceModeStyle, VoiceModeStyle(
                background: VoiceModeStyle.standard.background, orb: VoiceModeStyle.standard.orb,
                accent: .white, onAccent: .black, ink: scheme == .dark ? .white : Color(red: 0.12, green: 0.16, blue: 0.27), stop: .red,
                avatar: .face(name: "Roamy"), colorScheme: nil))
            let renderer = ImageRenderer(content: sheet)
            renderer.scale = 2
            let image = try XCTUnwrap(renderer.uiImage, "the face must render in \(scheme)")
            XCTAssertGreaterThan(image.size.width, 1000)
            if let directory = ProcessInfo.processInfo.environment["VOICEKIT_RENDER_DIR"] {
                let url = URL(fileURLWithPath: directory).appendingPathComponent("face-\(scheme == .dark ? "dark" : "light").png")
                try XCTUnwrap(image.pngData()).write(to: url)
            }
        }
    }
}
