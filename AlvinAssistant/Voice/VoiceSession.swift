import AssistantKit
import AVFoundation
import Foundation
import Observation
import UIKit

/// A hands-free voice conversation: listens, detects the end of the user's turn, streams the
/// reply, speaks it sentence by sentence, and stops talking when the user interrupts.
///
/// Replies come from the app's `ReplyPipeline`. When the pipeline supports it, a reply may start
/// early, while the user's turn is still open: its events wait silently until the turn commits,
/// and are then either adopted (same request) or thrown away.
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
    /// The route of the latest reply, once the pipeline reported one.
    private(set) var lastDecision: RouteDecision?
    /// The next reply runs in deep mode.
    private(set) var deepNextTurn = false
    /// The reply in flight runs in deep mode, so a turn reopened before it speaks keeps asking for that.
    @ObservationIgnored private var replyDeep = false
    /// The latency trace of the last finished turn.
    private(set) var lastTrace: TurnLatencyTrace?

    private let conversation: Conversation
    private let store: SettingsStore
    private let pipeline: any ReplyPipeline
    private let audio = AudioIO()
    @ObservationIgnored private var recognizer: SpeechRecognizer?
    @ObservationIgnored private var speaker: Speaker?
    @ObservationIgnored private var cuePlayer: CuePlayer?
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
    /// Starts tentative replies; one per session, so its delay tuner learns across turns.
    @ObservationIgnored private var early = EarlyReplyCoordinator(enabled: false)
    /// The reply started early for the open turn, if any.
    @ObservationIgnored private var tentative: TentativeReply?
    /// A tentative reply was started during the open turn.
    @ObservationIgnored private var tentativeStartedThisTurn = false
    /// The pipeline was prewarmed for the open turn's first words.
    @ObservationIgnored private var prewarmedThisTurn = false
    @ObservationIgnored private var cuePolicy = CuePolicy(enabled: false, isVoice: true, fillerDelay: -1)
    @ObservationIgnored private var ducker = BargeInDucker()
    @ObservationIgnored private var duckingEnabled = false
    /// The latency trace of the turn in progress, from its commit to the end of its reply.
    @ObservationIgnored private var trace: TurnLatencyTrace?

    /// Playback volume while the user seems to talk over the assistant.
    private static let duckedGain: Float = 0.3
    private static let duckRamp: TimeInterval = 0.08

    init(conversation: Conversation, store: SettingsStore) {
        self.conversation = conversation
        self.store = store
        pipeline = ReplyPipelines.make(store)
    }

    var modelName: String {
        pipeline.displayName(for: lastDecision)
    }

    /// Whether "Think deeper" does anything: the pipeline routes deep replies and deep mode is on.
    var canGoDeep: Bool {
        !usesLegacyPipeline && store.settings.provider == .anthropic && store.settings.deepMode != .off
    }

    /// A one-line summary of the last turn's latency, for debug builds.
    var latencyReadout: String? {
        guard let trace = lastTrace else { return nil }
        var parts: [String] = []
        if let heard = trace.heardSomething { parts.append("first sound \(Self.seconds(heard))") }
        if let answer = trace.answerLatency { parts.append("answer \(Self.seconds(answer))") }
        if let commit = trace.interval(.speechEnded, .committed) { parts.append("commit \(Self.seconds(commit))") }
        if trace.adoptedEarlyStart == true { parts.append("early") }
        if let engine = trace.engine { parts.append(engine) }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    // MARK: - Lifecycle

    func start() async {
        guard phase == .idle else { return }
        phase = .starting
        notice = nil
        if let problem = pipeline.missingSetup() {
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
        speaker.onCueStarted = { [weak self] _ in self?.cueStarted() }
        speaker.onDrained = { [weak self] in self?.speechDrained() }
        speaker.onFallback = { [weak self] message in
            self?.notice = "\(message) Using the built-in voice instead."
        }
        self.speaker = speaker
        let cuePlayer = CuePlayer(speaker: speaker, localeIdentifier: settings.speechLocale)
        if cuesWanted {
            cuePlayer.prerender()
        }
        self.cuePlayer = cuePlayer

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
        // Ducking listens for the user over the assistant's voice, so it needs echo cancellation,
        // and it only makes sense when talking over the assistant can interrupt it. The legacy
        // pipeline keeps today's playback untouched.
        duckingEnabled = !usesLegacyPipeline && settings.bargeInDucking && settings.voiceInterruptions && settings.echoCancellation
        ducker.reset()
        startTicker()
        listen()
    }

    /// Ends the conversation, keeping whatever the assistant already said.
    func stop() {
        guard phase != .idle else { return }
        ticker?.cancel()
        ticker = nil
        discardTentative()
        early.reset()
        replyTask?.cancel()
        replyTask = nil
        recognitionRetry?.cancel()
        recognitionRetry = nil
        if let reply {
            keepSpokenPart(of: reply)
        }
        reply = nil
        endTrace(interrupted: true)
        speaker?.stop()
        cuePlayer?.cancel()
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

    /// Runs the next reply in deep mode (`wanted`), or takes that back.
    func requestDeepNextTurn(_ wanted: Bool = true) {
        deepNextTurn = wanted
        updateEarlyStart()
        // An early reply started without the new setting can't be adopted.
        if let tentative, tentative.deep != wanted {
            discardTentative()
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
        pipeline.prewarm(inputIsVoice: true)
        beginTurn(with: carriedText)
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
            noteTranscript(full)
            if isFinal { commitUserTurn() }
        case .thinking:
            // Nothing of the answer is playing yet, so new words mean the user wasn't finished.
            // Once a cue has played, though, the words may be its echo, misheard: talking over a
            // cue then follows the rule for talking over the answer.
            if speaker?.hasPlayedCue == true, !store.settings.voiceInterruptions { return }
            if let words = interruptions.interruption(in: text, assistantSpeech: speaker?.recentSpeech ?? "") {
                reopenUserTurn(with: words)
            }
        case .speaking:
            guard store.settings.voiceInterruptions, let speaker else { return }
            if let words = interruptions.interruption(in: text, assistantSpeech: speaker.recentSpeech) {
                ducker.interruptionConfirmed()
                cancelReply()
                // This request's audio also holds the assistant's voice, so keep the live text.
                recognizer?.feeder.dropSamples()
                // The running request already holds the interruption, so keep it.
                phase = .listening
                carriedText = ""
                userCaption = words
                turn.reset(transcript: words, at: now)
                beginTurn(with: words)
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
        let time = now
        var trace = TurnLatencyTrace()
        trace.mark(.speechEnded, at: min(turn.lastChange ?? time, time))
        trace.mark(.committed, at: time)
        trace.engine = store.settings.provider == .onDevice ? ReplyEngine.local.rawValue : ReplyEngine.cloud.rawValue
        if tentativeStartedThisTurn { trace.adoptedEarlyStart = false }
        self.trace = trace
        if store.settings.turnChime { audio.playChime() }

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
        // The turn wasn't over, so its trace isn't a turn of its own: drop it unlogged, and the
        // next commit starts the turn's trace afresh.
        trace = nil
        let deep = replyDeep
        cancelReply()
        // It's still the turn "Think deeper" was asked for.
        deepNextTurn = deepNextTurn || deep
        let earlier = user.text
        conversation.remove(user)
        lastUserMessage = nil
        phase = .listening
        carriedText = earlier
        userCaption = Self.join(earlier, text)
        turn.reset(transcript: userCaption, at: now)
        beginTurn(with: userCaption)
    }

    // MARK: - Early start

    /// Whether replies may start before the user's turn is committed.
    private var earlyStartAllowed: Bool {
        let settings = store.settings
        guard pipeline.supportsEarlyStart, settings.earlyReplyStart != .off else { return false }
        let localRoute = settings.provider == .onDevice || (settings.routingMode == .automatic && settings.preferOnDevice)
        // "On-device only" leaves cloud replies alone: an early start there may cost a request.
        if settings.earlyReplyStart == .onDeviceOnly, !localRoute { return false }
        // Qwen3-ASR and the reply model would compete for the GPU.
        if localRoute, settings.usesQwenListening { return false }
        return true
    }

    /// Lets the coordinator start tentative replies when allowed. Never for a deep reply: a
    /// discarded one would still spend deep-mode work and budget.
    private func updateEarlyStart() {
        early.enabled = earlyStartAllowed && !deepNextTurn
    }

    /// Starts following a new user turn whose transcript so far is `text`.
    private func beginTurn(with text: String) {
        discardTentative()
        early.reset()
        updateEarlyStart()
        tentativeStartedThisTurn = false
        prewarmedThisTurn = false
        noteTranscript(text)
    }

    /// The open turn's live transcript is now `text`.
    private func noteTranscript(_ text: String) {
        if !prewarmedThisTurn, !text.trimmed.isEmpty {
            prewarmedThisTurn = true
            pipeline.prewarm(inputIsVoice: true)
        }
        perform(early.transcriptChanged(text, at: now))
    }

    private func perform(_ actions: [EarlyReplyCoordinator.Action]) {
        for action in actions {
            switch action {
            case .startTentative(let text):
                startTentative(text)
            case .cancelTentative:
                discardTentative()
            case .adopt, .startFresh:
                // Only a commit returns these; `replyEvents(for:deep:)` handles them.
                break
            }
        }
    }

    /// Starts a reply to `text` that nobody hears until the turn commits to the same request.
    private func startTentative(_ text: String) {
        discardTentative()
        let pendingUser = StoredMessage(role: .user, text: text, createdAt: Date(), isVoice: true)
        let gate = CommitGate()
        let deep = deepNextTurn
        let tentative = TentativeReply(pendingUser: pendingUser, gate: gate, deep: deep, startedAt: now)
        tentative.start(pipeline.stream(ReplyRequest(conversation: conversation, pendingUser: pendingUser, commitGate: gate, inputIsVoice: true, deep: deep)))
        self.tentative = tentative
        tentativeStartedThisTurn = true
    }

    /// Cancels the tentative reply, if any. Its side effects never ran: they wait on its gate.
    private func discardTentative() {
        tentative?.cancel()
        tentative = nil
    }

    /// The events of the reply to the committed `user` turn: the tentative reply's when it asked
    /// for the same thing, else a fresh reply's.
    private func replyEvents(for user: ChatMessage, deep: Bool) -> AsyncThrowingStream<AssistantEvent, Error> {
        let time = now
        let adopt = early.commit(finalText: user.text, at: time).contains(.adopt)
        if adopt, let tentative, !tentative.failed, tentative.deep == deep {
            self.tentative = nil
            // Store the turn exactly as the reply saw it, so the next request's history matches
            // what the model (and its caches) already read. It differs from the final transcript
            // at most in case, punctuation and spacing.
            user.text = tentative.pendingUser.text
            user.createdAt = tentative.pendingUser.createdAt
            trace?.adoptedEarlyStart = true
            trace?.mark(.earlyStart, at: tentative.startedAt)
            trace?.mark(.requestStarted, at: tentative.startedAt)
            for reached in tentative.marks {
                trace?.mark(reached.mark, at: reached.time)
            }
            if tentative.hasText {
                // The answer is ready; a cue now would only delay it.
                cuePolicy = CuePolicy(enabled: false, isVoice: true, fillerDelay: -1)
            }
            tentative.gate.open()
            return tentative.events
        }
        discardTentative()
        trace?.mark(.requestStarted, at: time)
        return pipeline.stream(ReplyRequest(conversation: conversation, inputIsVoice: true, deep: deep))
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
        resetDucking()
        let deep = deepNextTurn
        deepNextTurn = false
        replyDeep = deep
        cuePolicy = CuePolicy(enabled: cuesEnabled, isVoice: true, fillerDelay: store.settings.fillerDelay, deep: deep)
        cuePolicy.replyStarted(at: now, expectedFirstText: LatencyLog.shared.expectedFirstText())

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
                self?.trace?.mark(.transcriptRefined, at: ProcessInfo.processInfo.systemUptime)
            }
            guard let self, !Task.isCancelled, reply === self.reply else { return }
            // Decided only now: after any refinement, and reading the conversation as it is.
            let events = self.replyEvents(for: user, deep: deep)
            if user.text != live {
                if self.conversation.title == Conversation.makeTitle(from: live) {
                    self.conversation.title = Conversation.makeTitle(from: user.text)
                }
                if self.phase == .thinking { self.userCaption = user.text }
            }
            await self.consume(events, into: reply)
        }
    }

    private func consume(_ stream: AsyncThrowingStream<AssistantEvent, Error>, into reply: ChatMessage) async {
        do {
            for try await event in stream {
                guard reply === self.reply else { return }
                if let cue = cuePolicy.received(event, at: now) {
                    cuePlayer?.play(cue)
                }
                switch event {
                case .reply(.text(let text)):
                    if !text.isEmpty { trace?.mark(.firstText, at: now) }
                    reply.text += text
                    activity = nil
                    for chunk in chunker.append(text) {
                        speak(chunk)
                    }
                case .reply(.activity(let description)):
                    activity = description
                case .reply(.finished(let stop)):
                    ReplyOutcome.apply(stop, to: reply)
                    if case .refused = stop {
                        speaker?.stop()
                        speaker?.beginReply()
                        chunker.reset()
                        speak("Sorry, I can't help with that one.")
                    }
                case .toolRound(let round):
                    reply.toolRounds.append(round)
                case .routed(let decision):
                    lastDecision = decision
                    trace?.engine = decision.engine.rawValue + (decision.mode == .deep ? " deep" : "")
                    trace?.mark(.routed, at: now)
                case .cue:
                    // Played above, when the cue policy allows it.
                    break
                case .progress(let progress):
                    switch progress {
                    case .responseStarted: trace?.mark(.responseStarted, at: now)
                    case .prefillDone: trace?.mark(.prefillDone, at: now)
                    case .firstToken: trace?.mark(.firstToken, at: now)
                    case .toolCallStarted: break
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
        trace?.mark(.finished, at: now)
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
        trace?.mark(.firstAudio, at: now)
        if phase == .thinking {
            phase = .speaking
        }
        assistantCaption = text
    }

    private func cueStarted() {
        guard phase == .thinking || phase == .speaking else { return }
        trace?.mark(.firstCue, at: now)
    }

    private func speechDrained() {
        guard replyStreamDone, phase == .thinking || phase == .speaking else { return }
        endTrace(interrupted: false)
        reply = nil
        lastUserMessage = nil
        save()
        listen()
    }

    /// Stops the reply in progress (and any tentative one), keeping the part that was already spoken.
    private func cancelReply() {
        discardTentative()
        replyTask?.cancel()
        replyTask = nil
        if let reply {
            keepSpokenPart(of: reply)
        }
        reply = nil
        speaker?.stop()
        resetDucking()
        endTrace(interrupted: true)
        assistantCaption = ""
        activity = nil
    }

    /// Records what the user actually heard of a reply that was cut short. A refused or failed
    /// reply keeps its status: only the spoken apology was cut off.
    private func keepSpokenPart(of reply: ChatMessage) {
        guard reply.status == .streaming || reply.status == .complete else { return }
        let spoken = speaker?.spokenText ?? ""
        if spoken.trimmed.isEmpty, !reply.toolRounds.isEmpty {
            // Its tools already acted, so the reply stays in the history even though none of it was heard.
            reply.text = ""
            reply.status = .interrupted
            return
        }
        ReplyOutcome.keepStopped(reply, text: spoken, in: conversation)
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

    // MARK: - Cues, ducking and traces

    /// The legacy pipeline answers exactly as before this pipeline seam existed: no cues, no
    /// early start, no deep mode.
    private var usesLegacyPipeline: Bool {
        pipeline is LegacyReplyPipeline
    }

    /// Cues play while the microphone listens for the user, so they need echo cancellation to
    /// keep the assistant's own voice from reading as the user's.
    private var cuesWanted: Bool {
        store.settings.spokenCues && store.settings.echoCancellation && !usesLegacyPipeline
    }

    private var cuesEnabled: Bool {
        cuesWanted && cuePlayer != nil
    }

    /// Back to full volume and a fresh ducker, for the next reply.
    private func resetDucking() {
        ducker.reset()
        audio.setPlaybackGain(1, ramp: 0)
    }

    /// Files the open turn's trace in the latency log.
    private func endTrace(interrupted: Bool) {
        guard var trace else { return }
        if interrupted { trace.mark(.interrupted, at: now) }
        self.trace = nil
        lastTrace = trace
        LatencyLog.shared.append(trace)
    }

    private static func seconds(_ value: TimeInterval) -> String {
        String(format: "%.2f s", value)
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
        if phase == .thinking, reply != nil, !replyStreamDone, let cue = cuePolicy.tick(at: now) {
            cuePlayer?.play(cue)
        }
        guard phase == .listening, !isMuted else { return }
        if turn.shouldEndTurn(at: now) {
            commitUserTurn()
            return
        }
        perform(early.tick(at: now))
    }

    private func inputLevel(_ decibels: Float) {
        if duckingEnabled {
            switch ducker.level(decibels, at: now, speaking: phase == .speaking && !isMuted) {
            case .duck: audio.setPlaybackGain(Self.duckedGain, ramp: Self.duckRamp)
            case .restore: audio.setPlaybackGain(1, ramp: Self.duckRamp)
            case .none: break
            }
        }
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

/// A reply started before the user's turn was committed. Its events wait, unheard, in `events`
/// until the turn commits: adopted, they are consumed like any reply's (the buffered ones first);
/// discarded, the reply is cancelled and its gate keeps its side effects from ever running.
@MainActor
private final class TentativeReply {
    /// The user turn the reply answers.
    let pendingUser: StoredMessage
    let gate: CommitGate
    let deep: Bool
    /// When the request was sent.
    let startedAt: TimeInterval
    /// Everything the reply produced, buffered until someone consumes it.
    let events: AsyncThrowingStream<AssistantEvent, Error>
    /// A latency mark reached while buffering.
    struct Reached {
        let mark: LatencyMark
        let time: TimeInterval
    }

    /// Latency marks reached while buffering, in order.
    private(set) var marks: [Reached] = []
    /// The reply produced visible text.
    private(set) var hasText = false
    /// The reply ended with an error.
    private(set) var failed = false

    private let continuation: AsyncThrowingStream<AssistantEvent, Error>.Continuation
    private var relay: Task<Void, Never>?

    init(pendingUser: StoredMessage, gate: CommitGate, deep: Bool, startedAt: TimeInterval) {
        self.pendingUser = pendingUser
        self.gate = gate
        self.deep = deep
        self.startedAt = startedAt
        let (stream, continuation) = AsyncThrowingStream<AssistantEvent, Error>.makeStream()
        events = stream
        self.continuation = continuation
    }

    /// Starts buffering `source`. Ending `events` (its consumer was cancelled) cancels `source`.
    func start(_ source: AsyncThrowingStream<AssistantEvent, Error>) {
        let continuation = continuation
        let task = Task { [weak self] in
            do {
                for try await event in source {
                    self?.note(event)
                    continuation.yield(event)
                }
                continuation.finish()
            } catch {
                if !(error is CancellationError) { self?.failed = true }
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }
        relay = task
    }

    /// Cancels the reply. Its gate is cancelled too, so a side effect waiting on it never runs.
    func cancel() {
        gate.cancel()
        relay?.cancel()
        continuation.finish()
    }

    private func note(_ event: AssistantEvent) {
        let mark: LatencyMark
        switch event {
        case .routed:
            mark = .routed
        case .progress(.responseStarted):
            mark = .responseStarted
        case .progress(.prefillDone):
            mark = .prefillDone
        case .progress(.firstToken):
            mark = .firstToken
        case .reply(.text(let text)) where !text.isEmpty && !hasText:
            hasText = true
            mark = .firstText
        default:
            return
        }
        marks.append(Reached(mark: mark, time: ProcessInfo.processInfo.systemUptime))
    }
}
