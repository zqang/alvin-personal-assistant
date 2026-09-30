import AssistantKit
import AVFoundation
import Speech

/// Hands microphone buffers from the audio thread to the current recognition request, and, when a
/// second pass may read it, keeps the request's latest 16 kHz samples.
final class RecognitionFeeder: @unchecked Sendable {
    private let lock = NSLock()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var muted = false
    private var window: SampleWindow?
    /// False once this request's audio holds the assistant's own voice.
    private var recording = false

    /// `recordingCapacity`: how many samples to keep, or nil to keep none.
    init(recordingCapacity: Int?) {
        window = recordingCapacity.map(SampleWindow.init(capacity:))
    }

    /// A new request starts a new recording; clearing it keeps the samples, because a retired
    /// request's final result still commits the turn.
    func setRequest(_ request: SFSpeechAudioBufferRecognitionRequest?) {
        lock.lock()
        self.request = request
        if request != nil {
            window?.reset()
            recording = true
        }
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
        // Copy now: the converter may hand back the tap's own buffer.
        if target != nil, recording, let channel = buffer.floatChannelData?[0] {
            window?.append(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
        }
        lock.unlock()
        target?.append(buffer)
    }

    /// Stops keeping this request's audio, e.g. when it holds the assistant's own voice.
    func dropSamples() {
        lock.lock()
        recording = false
        lock.unlock()
    }

    /// How many samples this request has recorded so far, including discarded ones.
    var sampleCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return window?.count ?? 0
    }

    /// This request's audio for a second pass over the turn whose first words arrived at sample
    /// `speechStart` (counted like `sampleCount`), up to now; see `FinalTranscript.clipStart`.
    /// Nothing if the audio isn't kept, was dropped, or has already been discarded.
    func turnClip(speechStart: Int, maxLength: Int) -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        guard recording, let end = window?.count else { return [] }
        let start = FinalTranscript.clipStart(speechStart: speechStart, end: end, maxLength: maxLength, leadIn: 5 * 16_000, minLeadIn: 24_000)
        return window?.samples(from: start) ?? []
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

    let feeder: RecognitionFeeder
    var onEvent: ((Event) -> Void)?

    private let recognizer: SFSpeechRecognizer?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var generation = 0

    /// `recordingCapacity`: samples of each request to keep for a second pass, or nil for none.
    init(localeIdentifier: String, recordingCapacity: Int? = nil) {
        recognizer = SFSpeechRecognizer(locale: Locale(identifier: localeIdentifier))
        feeder = RecognitionFeeder(recordingCapacity: recordingCapacity)
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
        let finished = isFinal || error != nil
        if finished {
            // Retire this request before reporting, because the handler may start the next one.
            self.generation += 1
            feeder.setRequest(nil)
            request = nil
            task = nil
        }
        let current = self.generation
        if let text {
            onEvent?(.transcript(text, isFinal: isFinal))
        }
        // Unless handling the transcript already started a new request, say this one ended.
        if finished, current == self.generation {
            onEvent?(.ended(error))
        }
    }
}
