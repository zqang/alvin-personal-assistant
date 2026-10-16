import AssistantKit
import Foundation
import MiniCPM
import VoiceKit

// How this app hosts voice mode (VoiceKit): its stored conversations, and the settings store with
// the on-device models voice mode uses (Qwen3-ASR, the on-device model, MiniCPM-V).

extension ChatMessage: VoiceMessage {}

extension Conversation: VoiceConversation {
    var voiceMessages: [any VoiceMessage] { orderedMessages }

    func appendMessage(role: ChatRole, text: String, isVoice: Bool, status: StoredMessage.Status) -> any VoiceMessage {
        let message = ChatMessage(role: role, text: text, isVoice: isVoice, status: status)
        append(message)
        return message
    }

    func remove(_ message: any VoiceMessage) {
        if let message = message as? ChatMessage { remove(message) }
    }

    func userTurnRevised(from old: String, to new: String) {
        if title == Conversation.makeTitle(from: old) {
            title = Conversation.makeTitle(from: new)
        }
    }

    func save() {
        updatedAt = .now
        try? modelContext?.save()
    }
}

extension SettingsStore: VoiceSessionHost {
    var openAIKey: String { secret(.openAI) }

    func voiceSessionWillStart(_ settings: AssistantSettings) {
        // "Most accurate on device" loads its own model, through the pipeline's prewarm.
        if settings.provider == .onDevice, !settings.accurateOnDevice {
            LocalModelHost.shared.prepare(settings)
        }
        guard settings.usesQwenListening else { return }
        QwenListener.shared.isWanted = true
        if settings.provider == .onDevice || settings.accurateOnDevice {
            // One model loads at a time, and the reply model gets the memory first: the on-device
            // model, or MiniCPM-V 35B, which the session's first prewarm starts loading before
            // this task runs.
            Task {
                await LocalModelHost.shared.settle()
                await MiniCPMHost.shared.settle()
                if QwenListener.shared.isWanted { QwenListener.shared.load() }
            }
        } else {
            QwenListener.shared.load()
        }
    }

    func voiceSessionDidStop() {
        QwenListener.shared.isWanted = false
        QwenListener.shared.unload()
    }

    /// Qwen3-ASR reads each committed turn again, when it's on.
    var refineClipSamples: Int {
        settings.usesQwenListening ? QwenListener.maxSamples : 0
    }

    func refineTurn(_ samples: [Float], locale: String) async -> String? {
        await QwenListener.shared.transcribe(samples, locale: locale)
    }

    /// MiniCPM-V 1B reads photos once it's ready.
    var photoReader: ((Data) -> AsyncThrowingStream<String, Error>)? {
        guard MiniCPMHost.shared.status[.photoReader] == .ready else { return nil }
        return { MiniCPMHost.shared.readPhoto($0) }
    }
}
