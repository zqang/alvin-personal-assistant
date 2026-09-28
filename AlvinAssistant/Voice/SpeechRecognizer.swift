import AVFoundation
import Speech

/// Hands microphone buffers from the audio thread to the current recognition request.
final class RecognitionFeeder: @unchecked Sendable {
    private let lock = NSLock()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var muted = false

    func setRequest(_ request: SFSpeechAudioBufferRecognitionRequest?) {
        lock.lock()
        self.request = request
        lock.unlock()
    }

    func setMuted(_ muted: Bool) {
        lock.lock()
        self.muted = muted
        lock.unlock()
    }

    func append(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        let target = muted ? nil : request
        lock.unlock()
        target?.append(buffer)
    }
}

struct SpeechRecognitionUnavailable: LocalizedError {
    var errorDescription: String? { "Speech recognition isn't available right now." }
}

/// Live speech-to-text with Apple's recognizer (on-device when the language supports it),
/// one recognition request per user turn.
@MainActor
final class SpeechRecognizer {
    enum Event {
        case transcript(String, isFinal: Bool)
        /// The request ended without a final transcript (long silence, time limit, or an error).
        case ended(Error?)
    }

    let feeder = RecognitionFeeder()
    var onEvent: ((Event) -> Void)?

    private let recognizer: SFSpeechRecognizer?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var generation = 0

    init(localeIdentifier: String) {
        recognizer = SFSpeechRecognizer(locale: Locale(identifier: localeIdentifier))
    }

    var isAvailable: Bool { recognizer?.isAvailable ?? false }

    nonisolated static func requestAuthorization() async -> Bool {
        let status = await withCheckedContinuation { (continuation: CheckedContinuation<SFSpeechRecognizerAuthorizationStatus, Never>) in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
        return status == .authorized
    }

    /// Starts a fresh request; audio appended before this is discarded.
    func startTurn() {
        stopTurn()
        guard let recognizer, recognizer.isAvailable else {
            onEvent?(.ended(SpeechRecognitionUnavailable()))
            return
        }
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.addsPunctuation = true
        request.taskHint = .dictation
        if recognizer.supportsOnDeviceRecognition {
            request.requiresOnDeviceRecognition = true
        }
        self.request = request
        feeder.setRequest(request)
        task = recognizer.recognitionTask(with: request, resultHandler: Self.resultHandler(for: self, generation: generation))
    }

    func stopTurn() {
        generation += 1
        feeder.setRequest(nil)
        request?.endAudio()
        task?.cancel()
        request = nil
        task = nil
    }

    /// Built outside the main actor: the recognizer calls it on its own queue.
    nonisolated private static func resultHandler(
        for recognizer: SpeechRecognizer,
        generation: Int
    ) -> @Sendable (SFSpeechRecognitionResult?, Error?) -> Void {
        { [weak recognizer] result, error in
            let text = result?.bestTranscription.formattedString
            let isFinal = result?.isFinal ?? false
            Task { @MainActor [weak recognizer] in
                recognizer?.handle(text: text, isFinal: isFinal, error: error, generation: generation)
            }
        }
    }

    private func handle(text: String?, isFinal: Bool, error: Error?, generation: Int) {
        guard generation == self.generation else { return }
        if let text {
            onEvent?(.transcript(text, isFinal: isFinal))
        }
        guard isFinal || error != nil else { return }
        // This request is finished; ignore anything else it reports.
        self.generation += 1
        feeder.setRequest(nil)
        request = nil
        task = nil
        if !isFinal || text == nil {
            onEvent?(.ended(error))
        }
    }
}
