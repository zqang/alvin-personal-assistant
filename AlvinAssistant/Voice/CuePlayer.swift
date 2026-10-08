import AssistantKit
import AVFoundation

/// Plays spoken cues ("Let me check.", "One moment.") in the session's voice.
///
/// Every phrase is rendered ahead of time when a voice session starts, so a cue sounds at once
/// instead of after a synthesis round trip. A cue that isn't rendered yet, or whose render no
/// longer matches the voice (the cloud voice fell back to the built-in one), is spoken live.
/// Renders are kept for the next session while the voice and language stay the same.
@MainActor
final class CuePlayer {
    /// Every cue, most common first, so those are ready soonest.
    static let cues: [SpokenCue] = [.working, .checking, .lookingUp, .handingOff, .loadingModel, .deepThinking, .stillThinking]

    private struct Renders {
        var voice: String
        /// Buffers by phrase.
        var phrases: [String: [AVAudioPCMBuffer]]
    }

    /// Renders of the most recent voice, shared by sessions.
    private static var renders = Renders(voice: "", phrases: [:])

    private let speaker: Speaker
    private let localeIdentifier: String
    private var rendering: Task<Void, Never>?

    init(speaker: Speaker, localeIdentifier: String) {
        self.speaker = speaker
        self.localeIdentifier = localeIdentifier
    }

    /// Starts rendering every phrase that isn't rendered for the current voice yet.
    func prerender() {
        guard rendering == nil else { return }
        let voice = speaker.voiceSignature
        if CuePlayer.renders.voice != voice {
            CuePlayer.renders = Renders(voice: voice, phrases: [:])
        }
        let phrases = CuePlayer.cues.map { phrase(for: $0) }
        rendering = Task { [weak self] in
            for phrase in phrases where CuePlayer.renders.phrases[phrase] == nil {
                guard let self, !Task.isCancelled else { return }
                let buffers = await self.speaker.prerender(phrase)
                guard !Task.isCancelled else { return }
                // Only keep it if the voice didn't change while it rendered.
                if let buffers, CuePlayer.renders.voice == voice, self.speaker.voiceSignature == voice {
                    CuePlayer.renders.phrases[phrase] = buffers
                }
            }
        }
    }

    func play(_ cue: SpokenCue) {
        let text = phrase(for: cue)
        if CuePlayer.renders.voice == speaker.voiceSignature, let buffers = CuePlayer.renders.phrases[text] {
            speaker.enqueueCue(buffers: buffers, text: text)
        } else {
            speaker.enqueue(text, isCue: true)
        }
    }

    /// Stops rendering. What was rendered stays for later sessions.
    func cancel() {
        rendering?.cancel()
        rendering = nil
    }

    private func phrase(for cue: SpokenCue) -> String {
        SpokenCue.phrase(cue, localeIdentifier: localeIdentifier)
    }
}
