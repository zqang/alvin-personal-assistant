import AssistantKit
import AVFoundation
import Foundation
import Observation
import UIKit

/// A hands-free voice conversation: listens, detects the end of the user's turn, streams the
/// reply, speaks it sentence by sentence, and stops talking when the user interrupts.
@MainActor
@Observable
final class VoiceSession: Identifiable {
    enum Phase: Equatable {
        case idle
        case starting
        case listening
        case thinking
        case speaking
        case failed(String)
    }

    let id = UUID()
    private(set) var phase: Phase = .idle
    /// What the user is saying (live) or just said.
    private(set) var userCaption = ""
    /// The sentence the assistant is speaking.
    private(set) var assistantCaption = ""
    /// What the model is doing while it isn't talking, e.g. searching the web.
    private(set) var activity: String?
    /// A non-fatal problem worth showing.
    private(set) var notice: String?
    /// Smoothed loudness (0...1) of whoever is talking, for the orb.
    private(set) var level: Float = 0
    private(set) var isMuted = false

    private let conversation: Conversation
    private let store: SettingsStore
    private let audio = AudioIO()
    private let bargeIn = BargeInDetector()
    @ObservationIgnored private var recognizer: SpeechRecognizer?
    @ObservationIgnored private var speaker: Speaker?
    @ObservationIgnored private var turn = TurnDetector()
    @ObservationIgnored private var chunker = SentenceChunker()
    @ObservationIgnored private var reply: ChatMessage?
    @ObservationIgnored private var lastUserMessage: ChatMessage?
    @ObservationIgnored private var replyTask: Task<Void, Never>?
    @ObservationIgnored private var replyStreamDone = false
    /// Earlier words of the current turn, when the user resumed talking after a pause.
    @ObservationIgnored private var carriedText = ""
    /// Where this recognition request's speech starts in its recorded samples, once words arrive.
    @ObservationIgnored private var speechStart: Int?
    @ObservationIgnored private var ticker: Task<Void, Never>?
    @ObservationIgnored private var observers: [NSObjectProtocol] = []
    @ObservationIgnored private var recognitionRestarts: [TimeInterval] = []
    @ObservationIgnored private var recognitionRetry: Task<Void, Never>?
    @ObservationIgnored private var unavailableEndings = 0
    @ObservationIgnored private var interruptions = InterruptionMonitor()

    init(conversation: Conversation, store: SettingsStore) {
        self.conversation = conversation
        self.store = store
    }

    var modelName: String {
        switch store.settings.provider {
        case .anthropic: return ClaudeModelCatalog.displayName(for: store.settings.claudeModel)
        case .openAICompatible: return store.settings.compatibleModel
        case .onDevice: return LocalModelCatalog.option(for: store.settings.localModelID)?.displayName ?? "On-device model"
        }
    }

    // MARK: - Lifecycle

    func start() async {
        guard phase == .idle else { return }
        phase = .starting
        notice = nil
        if let problem = ReplyService.missingSetup(settings: store.settings, store: store) {
            return fail(problem)
        }
        guard await AVAudioApplication.requestRecordPermission() else {
            return fail("Microphone access is off. Turn it on in Settings › Privacy & Security › Microphone.")
        }
        guard await SpeechRecognizer.requestAuthorization() else {
            return fail("Speech recognition is off. Turn it on in Settings › Privacy & Security › Speech Recognition.")
        }
        guard phase == .starting else { return }

        let settings = store.settings
        // Keep twice the longest clip Qwen3-ASR reads, so a full one always fits.
        let recognizer = SpeechRecognizer(localeIdentifier: settings.speechLocale, recordingCapacity: settings.usesQwenListening ? 2 * QwenListener.maxSamples : nil)
        guard recognizer.isAvailable else {
            return fail("Speech recognition for \(settings.speechLocale) isn't available right now. Check your connection or pick another language in Settings.")
        }
        recognizer.onEvent = { [weak self] event in self?.handle(event) }
        self.recognizer = recognizer
        if settings.provider == .onDevice {
            LocalModelHost.shared.prepare(settings)
        }
        if settings.usesQwenListening {
            QwenListener.shared.isWanted = true
            if settings.provider == .onDevice {
                // One model loads at a time, and the reply model gets the memory first.
                Task {
                    await LocalModelHost.shared.settle()
                    if QwenListener.shared.isWanted { QwenListener.shared.load() }
                }
            } else {
                QwenListener.shared.load()
            }
        }

        let speaker = Speaker(audio: audio, engine: speechEngine(for: settings), languageHint: settings.speechLocale)
        speaker.onChunkStarted = { [weak self] text in self?.assistantStartedSpeaking(text) }
        speaker.onDrained = { [weak self] in self?.speechDrained() }
        speaker.onFallback = { [weak self] message in
            self?.notice = "\(message) Using the built-in voice instead."
        }
        self.speaker = speaker

        let feeder = recognizer.feeder
        audio.onInput = { buffer in feeder.append(buffer) }
        audio.onInputLevel = { [weak self] level in
            Task { @MainActor [weak self] in self?.inputLevel(level) }
        }
        audio.onOutputLevel = { [weak self] level in
            Task { @MainActor [weak self] in self?.outputLevel(level) }
        }
        do {
            try audio.start(echoCancellation: settings.echoCancellation)
        } catch {
            return fail("Couldn't start the microphone: \(error.localizedDescription)")
        }

        observeAudioSession()
        UIApplication.shared.isIdleTimerDisabled = true
        turn = TurnDetector(silenceTimeout: settings.endOfTurnDelay)
        startTicker()
        listen()
    }

    /// Ends the conversation, keeping whatever the assistant already said.
    func stop() {
        guard phase != .idle else { return }
        ticker?.cancel()
        ticker = nil
        replyTask?.cancel()
        replyTask = nil
        recognitionRetry?.cancel()
        recognitionRetry = nil
        if let reply {
            keepSpokenPart(of: reply)
        }
        reply = nil
        speaker?.stop()
        recognizer?.stopTurn()
        QwenListener.shared.isWanted = false
        QwenListener.shared.unload()
        audio.stop()
        for observer in observers {
            NotificationCenter.default.removeObserver(observer)
        }
        observers.removeAll()
        UIApplication.shared.isIdleTimerDisabled = false
        save()
        phase = .idle
        level = 0
    }

    func restart() async {
        stop()
        await start()
    }

    // MARK: - Controls

    /// Tapping the orb sends what's been said so far, or cuts the assistant off.
    func tapOrb() {
        switch phase {
        case .listening:
            if !userCaption.trimmed.isEmpty { commitUserTurn() }
        case .thinking, .speaking:
            cancelReply()
            listen()
        default:
            break
        }
    }

    func toggleMute() {
        isMuted.toggle()
        recognizer?.feeder.setMuted(isMuted)
        if !isMuted, phase == .listening {
            // Start a fresh request so the muted gap doesn't confuse turn detection.
            listen(continuing: userCaption)
        }
    }

    // MARK: - Listening

    private func listen(continuing text: String = "") {
        phase = .listening
        carriedText = text.trimmed
        userCaption = carriedText
        assistantCaption = ""
        activity = nil
        turn.reset(transcript: carriedText, at: now)
        beginRecognition()
    }

    private func handle(_ event: SpeechRecognizer.Event) {
        switch event {
        case .transcript(let text, let isFinal):
            unavailableEndings = 0
            if speechStart == nil, !text.trimmed.isEmpty {
                speechStart = recognizer?.feeder.sampleCount
            }
            heard(text, isFinal: isFinal)
        case .ended(let error):
            if error is SpeechRecognitionUnavailable {
                unavailableEndings += 1
                if unavailableEndings >= 5 {
                    return fail("Speech recognition isn't available right now. Check your internet connection, or pick a language in Settings that your iPhone can recognize offline.")
                }
            }
            // The recognizer ended this request (silence limit or a hiccup). Keep the mic live.
            switch phase {
            case .listening:
                if isMuted || userCaption.trimmed.isEmpty {
                    restartRecognition()
                } else {
                    commitUserTurn()
                }
            case .thinking, .speaking:
                restartRecognition()
            default:
                break
            }
        }
    }

    private func heard(_ text: String, isFinal: Bool) {
        switch phase {
        case .listening:
            guard !isMuted else { return }
            let full = Self.join(carriedText, interruptions.userWords(in: text))
            userCaption = full
            turn.transcriptChanged(full, at: now)
            if isFinal { commitUserTurn() }
        case .thinking:
            // Nothing is playing yet, so new words mean the user wasn't finished.
            if bargeIn.isInterruption(heard: text, assistantSpeech: "") {
                reopenUserTurn(with: text)
            }
        case .speaking:
            guard store.settings.voiceInterruptions, let speaker else { return }
            if let words = interruptions.interruption(in: text, assistantSpeech: speaker.recentSpeech) {
                cancelReply()
                // This request's audio also holds the assistant's voice, so keep the live text.
                recognizer?.feeder.dropSamples()
                // The running request already holds the interruption, so keep it.
                phase = .listening
                carriedText = ""
                userCaption = words
                turn.reset(transcript: words, at: now)
            }
        default:
            break
        }
    }

    private func restartRecognition() {
        let time = now
        recognitionRestarts = recognitionRestarts.filter { time - $0 < 5 } + [time]
        guard recognitionRestarts.count > 3 else {
            beginRecognition()
            return
        }
        // Back off if the recognizer keeps ending straight away.
        recognitionRetry?.cancel()
        recognitionRetry = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            guard !Task.isCancelled, let self, self.isLive else { return }
            self.beginRecognition()
        }
    }

    /// Starts a fresh recognition request, forgetting what was judged to be echo in the last one.
    private func beginRecognition() {
        recognitionRetry?.cancel()
        recognitionRetry = nil
        interruptions.reset()
        speechStart = nil
        recognizer?.startTurn()
    }

    private var isLive: Bool {
        phase == .listening || phase == .thinking || phase == .speaking
    }

    private func commitUserTurn() {
        let text = userCaption.trimmed
        guard !text.isEmpty else {
            restartRecognition()
            return
        }
        let message = ChatMessage(role: .user, text: text, isVoice: true)
        conversation.append(message)
        lastUserMessage = message
        // Take the turn's audio before the next request clears it.
        var samples: [Float] = []
        if let speechStart, let feeder = recognizer?.feeder {
            samples = feeder.turnClip(speechStart: speechStart, maxLength: QwenListener.maxSamples)
        }
        startReply(refining: message, carried: carriedText, samples: samples)
        carriedText = ""
        // Keep listening while the assistant thinks and talks, so the user can cut in.
        beginRecognition()
    }

    /// The user kept talking before the reply started: take back their turn and keep listening.
    private func reopenUserTurn(with text: String) {
        guard let user = lastUserMessage else { return }
        cancelReply()
        let earlier = user.text
        conversation.remove(user)
        lastUserMessage = nil
        phase = .listening
        carriedText = earlier
        userCaption = Self.join(earlier, text)
        turn.reset(transcript: userCaption, at: now)
    }

    // MARK: - Replying

    /// Replies to `user`, first swapping in Qwen3-ASR's reading of the turn's audio when it's on.
    /// `carried` is earlier text of the turn that `samples` don't cover.
    private func startReply(refining user: ChatMessage, carried: String, samples: [Float]) {
        phase = .thinking
        activity = nil
        notice = nil
        assistantCaption = ""
        chunker.reset()
        replyStreamDone = false
        speaker?.beginReply()

        let reply = ChatMessage(role: .assistant, text: "", isVoice: true, status: .streaming)
        conversation.append(reply)
        self.reply = reply
        let locale = store.settings.speechLocale
        replyTask = Task { [weak self] in
            let live = user.text
            if !samples.isEmpty, let heard = await QwenListener.shared.transcribe(samples, locale: locale), !Task.isCancelled {
                user.text = FinalTranscript.pick(live: live, carried: carried, refined: heard)
                // ponytail: comparison line; delete after checking the results on the iPhone.
                print("[ASR] apple=\(live) qwen=\(user.text)")
            }
            guard let self, !Task.isCancelled, reply === self.reply else { return }
            if user.text != live {
                if self.conversation.title == Conversation.makeTitle(from: live) {
                    self.conversation.title = Conversation.makeTitle(from: user.text)
                }
                if self.phase == .thinking { self.userCaption = user.text }
            }
            // Built only now: it reads the conversation, including the refined text, right away.
            await self.consume(ReplyService.stream(for: self.conversation, store: self.store), into: reply)
        }
    }

    private func consume(_ stream: AsyncThrowingStream<ReplyEvent, Error>, into reply: ChatMessage) async {
        do {
            for try await event in stream {
                guard reply === self.reply else { return }
                switch event {
                case .text(let text):
                    reply.text += text
                    activity = nil
                    for chunk in chunker.append(text) {
                        speak(chunk)
                    }
                case .activity(let description):
                    activity = description
                case .finished(let stop):
                    ReplyOutcome.apply(stop, to: reply)
                    if case .refused = stop {
                        speaker?.stop()
                        speaker?.beginReply()
                        chunker.reset()
                        speak("Sorry, I can't help with that one.")
                    }
                }
            }
        } catch {
            guard reply === self.reply, !Task.isCancelled else { return }
            reply.status = .failed
            reply.errorText = error.localizedDescription
            notice = error.localizedDescription
            chunker.reset()
            speak("Sorry, something went wrong.")
        }
        guard reply === self.reply, !Task.isCancelled else { return }
        for chunk in chunker.flush() {
            speak(chunk)
        }
        if reply.status == .streaming {
            reply.status = .complete
        }
        replyStreamDone = true
        speaker?.finishInput()
    }

    private func speak(_ text: String) {
        let spoken = SpeechTextCleaner.clean(text)
        guard !spoken.isEmpty else { return }
        speaker?.enqueue(spoken)
    }

    private func assistantStartedSpeaking(_ text: String) {
        guard phase == .thinking || phase == .speaking else { return }
        if phase == .thinking {
            phase = .speaking
        }
        assistantCaption = text
    }

    private func speechDrained() {
        guard replyStreamDone, phase == .thinking || phase == .speaking else { return }
        reply = nil
        lastUserMessage = nil
        save()
        listen()
    }

    /// Stops the reply in progress, keeping the part that was already spoken.
    private func cancelReply() {
        replyTask?.cancel()
        replyTask = nil
        if let reply {
            keepSpokenPart(of: reply)
        }
        reply = nil
        speaker?.stop()
        assistantCaption = ""
        activity = nil
    }

    /// Records what the user actually heard of a reply that was cut short. A refused or failed
    /// reply keeps its status: only the spoken apology was cut off.
    private func keepSpokenPart(of reply: ChatMessage) {
        guard reply.status == .streaming || reply.status == .complete else { return }
        ReplyOutcome.keepStopped(reply, text: speaker?.spokenText ?? "", in: conversation)
    }

    private func speechEngine(for settings: AssistantSettings) -> Speaker.Engine {
        let rate = Float(min(max(settings.speechRate, 0.5), 1.6)) * AVSpeechUtteranceDefaultSpeechRate
        switch settings.voiceEngine {
        case .apple:
            return .apple(voiceIdentifier: settings.appleVoiceIdentifier, rate: rate)
        case .openAI:
            let key = store.secret(.openAI)
            guard !key.isEmpty else {
                notice = "Add an OpenAI API key in Settings to use OpenAI voices. Using the built-in voice."
                return .apple(voiceIdentifier: settings.appleVoiceIdentifier, rate: rate)
            }
            return .openAI(OpenAISpeechConfiguration(apiKey: key, model: settings.openAISpeechModel, voice: settings.openAIVoice))
        }
    }

    // MARK: - Levels and timing

    private var now: TimeInterval { ProcessInfo.processInfo.systemUptime }

    private func startTicker() {
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 100_000_000)
                guard let self else { return }
                self.tick()
            }
        }
    }

    private func tick() {
        if phase == .thinking || isMuted {
            level *= 0.85
        }
        guard phase == .listening, !isMuted else { return }
        if turn.shouldEndTurn(at: now) {
            commitUserTurn()
        }
    }

    private func inputLevel(_ decibels: Float) {
        guard !isMuted, phase != .idle else { return }
        turn.audioLevel(decibels, at: now)
        if phase == .listening { setLevel(decibels) }
    }

    private func outputLevel(_ decibels: Float) {
        if phase == .speaking { setLevel(decibels) }
    }

    private func setLevel(_ decibels: Float) {
        let normalized = min(max((decibels + 50) / 40, 0), 1)
        level += (normalized - level) * 0.35
    }

    // MARK: - Audio session events

    private func observeAudioSession() {
        let center = NotificationCenter.default
        observers.append(center.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            let type = notification.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt
            guard type == AVAudioSession.InterruptionType.began.rawValue else { return }
            Task { @MainActor [weak self] in self?.audioInterrupted() }
        })
        observers.append(center.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: audio.engine,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in self?.audioRouteChanged() }
        })
    }

    private func audioInterrupted() {
        guard phase != .idle else { return }
        fail("Voice chat stopped because another app or a call took over the audio.")
    }

    private func audioRouteChanged() {
        guard phase == .listening || phase == .thinking || phase == .speaking else { return }
        do {
            try audio.restart()
            // Queued speech was lost with the old graph, so hand the turn back to the user.
            if phase != .listening { cancelReply() }
            listen(continuing: phase == .listening ? userCaption : "")
        } catch {
            fail("The audio route changed and the microphone couldn't restart.")
        }
    }

    // MARK: - Helpers

    private func fail(_ message: String) {
        stop()
        phase = .failed(message)
    }

    private func save() {
        conversation.updatedAt = .now
        try? conversation.modelContext?.save()
    }

    private static func join(_ first: String, _ second: String) -> String {
        let first = first.trimmed
        let second = second.trimmed
        if first.isEmpty { return second }
        if second.isEmpty { return first }
        return first + " " + second
    }
}
