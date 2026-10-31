import AVFoundation
import SwiftUI

/// Picks the on-device voice for `AssistantSettings.appleVoiceIdentifier`: the user's own Personal Voices first
/// (iOS 17, recorded in Settings › Accessibility › Personal Voice), then the installed system voices for
/// `language`, best quality first. A host shows it in its settings; empty means the best installed voice.
public struct VoicePicker: View {
    @Binding var voiceIdentifier: String
    let language: String
    @State private var personalStatus = VoiceCatalog.personalVoiceStatus
    @State private var preview = AVSpeechSynthesizer()

    public init(voiceIdentifier: Binding<String>, language: String) {
        _voiceIdentifier = voiceIdentifier
        self.language = language
    }

    public var body: some View {
        Section {
            switch personalStatus {
            case .authorized:
                let voices = VoiceCatalog.personalVoices(forLanguage: language)
                if voices.isEmpty {
                    Text("Record one in Settings › Accessibility › Personal Voice, then allow apps to request it.", bundle: .voiceKit)
                        .font(.footnote).foregroundStyle(.secondary)
                } else {
                    ForEach(voices, id: \.identifier) { voice in
                        row(voice, label: voice.name, detail: String(localized: "Personal Voice", bundle: .voiceKit))
                    }
                }
            case .denied:
                Text("Personal Voice access was declined. Allow it in Settings › Accessibility › Personal Voice.", bundle: .voiceKit)
                    .font(.footnote).foregroundStyle(.secondary)
            case .unsupported:
                Text("Personal Voice is not available on this iPhone.", bundle: .voiceKit)
                    .font(.footnote).foregroundStyle(.secondary)
            default:
                Button {
                    Task {
                        _ = await VoiceCatalog.requestPersonalVoiceAccess()
                        personalStatus = VoiceCatalog.personalVoiceStatus
                    }
                } label: {
                    Label { Text("Allow Personal Voice", bundle: .voiceKit) } icon: { Image(systemName: "person.wave.2") }
                }
            }
        } header: {
            Text("Your voice", bundle: .voiceKit)
        }
        Section {
            row(nil, label: String(localized: "Automatic (best installed)", bundle: .voiceKit), detail: nil)
            ForEach(VoiceCatalog.voices(forLanguage: language), id: \.identifier) { voice in
                row(voice, label: voice.name, detail: VoiceCatalog.qualityLabel(voice))
            }
            Button {
                previewVoice()
            } label: {
                Label { Text("Preview voice", bundle: .voiceKit) } icon: { Image(systemName: "play.circle") }
            }
        } header: {
            Text("System voices", bundle: .voiceKit)
        }
    }

    private func row(_ voice: AVSpeechSynthesisVoice?, label: String, detail: String?) -> some View {
        let identifier = voice?.identifier ?? ""
        return Button {
            voiceIdentifier = identifier
        } label: {
            HStack {
                VStack(alignment: .leading) {
                    Text(label)
                    if let detail { Text(detail).font(.caption).foregroundStyle(.secondary) }
                }
                Spacer()
                if voiceIdentifier == identifier { Image(systemName: "checkmark").foregroundStyle(.tint) }
            }
        }
        .foregroundStyle(.primary)
        .accessibilityAddTraits(voiceIdentifier == identifier ? .isSelected : [])
    }

    /// Speaks one sentence in the chosen voice. Personal Voices only play this way, never rendered.
    private func previewVoice() {
        let utterance = AVSpeechUtterance(string: String(localized: "Hi! What can I help you with today?", bundle: .voiceKit))
        utterance.voice = voiceIdentifier.isEmpty ? VoiceCatalog.bestVoice(forLanguage: language) : AVSpeechSynthesisVoice(identifier: voiceIdentifier)
        preview.stopSpeaking(at: .immediate)
        preview.speak(utterance)
    }
}
