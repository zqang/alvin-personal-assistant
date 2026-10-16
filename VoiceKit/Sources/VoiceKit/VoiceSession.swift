import AssistantKit
import AVFoundation
import Foundation
import Observation
import os
import UIKit

/// A hands-free voice conversation: listens, detects the end of the user's turn, streams the
/// reply, speaks it sentence by sentence, and stops talking when the user interrupts.
///
/// Replies come from the app's `ReplyPipeline`. When the pipeline supports it, a reply may start
/// early, while the user's turn is still open: its events wait silently until the turn commits,
/// and are then either adopted (same request) or thrown away.
@MainActor
@Observable
public final class VoiceSession: Identifiable {
    enum Phase: Equatable {
        case idle
        case starting
        case listening
        case thinking
        case speaking
        case failed(String)
    }

    public let id = UUID()
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
    /// The photo for the user's next turn, taken with the camera and read while they talk.
    private(set) var pendingPhoto: PendingPhoto?
    /// The live camera preview is open.
    private(set) var cameraOpen = false

    private let conversation: any VoiceConversation
    private let host: any VoiceSessionHost
    private let pipeline: any ReplyPipeline
    private let audio = AudioIO()
    @ObservationIgnored private var recognizer: SpeechRecognizer?
    @ObservationIgnored private var speaker: Speaker?
    @ObservationIgnored private var cuePlayer: CuePlayer?
    @ObservationIgnored private var turn = TurnDetector()
    @ObservationIgnored private var chunker = SentenceChunker()
    @ObservationIgnored private var reply: (any VoiceMessage)?
    @ObservationIgnored private var lastUserMessage: (any VoiceMessage)?
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
    /// Whether the user's turn may still be taken back, given what the reply in flight has done.
    @ObservationIgnored private var reopenPolicy = TurnReopenPolicy()
    @ObservationIgnored private var ducker = BargeInDucker()
    @ObservationIgnored private var duckingEnabled = false
    /// The latency trace of the turn in progress, from its commit to the end of its reply.
    @ObservationIgnored private var trace: TurnLatencyTrace?
    /// The photo of the committed turn, so taking the turn back takes its photo back too.
    @ObservationIgnored private var lastUserPhoto: PendingPhoto?
    /// When the camera was last used (opened, a photo, a turn); it closes `cameraIdleLimit` after.
    @ObservationIgnored private var cameraUsedAt: TimeInterval = 0
    /// The host's `refineClipSamples`, read when the session starts.
    @ObservationIgnored private var refineClipSamples = 0
    /// `voiceSessionWillStart` ran, so stopping owes the host one `voiceSessionDidStop`.
    @ObservationIgnored private var hostStarted = false

    /// Playback volume while the user seems to talk over the assistant.
    private static let duckedGain: Float = 0.3
    private static let duckRamp: TimeInterval = 0.08
    private static let cameraIdleLimit: TimeInterval = 120
    /// The activity while a turn waits for its photo's read.
    static var readingPhoto: String { String(localized: "Reading the photo…", bundle: .voiceKit) }

    /// What voice mode says when a reply fails, or the model declines it, in the speech language
    /// the way `SpokenCue.phrase` picks it: Chinese for any "zh" locale, otherwise English.
    static func apology(refused: Bool, localeIdentifier: String) -> String {
        let chinese = localeIdentifier.lowercased().hasPrefix("zh")
        if refused { return chinese ? "抱歉，这个我帮不上忙。" : "Sorry, I can't help with that one." }
        return chinese ? "抱歉，出了点问题。" : "Sorry, something went wrong."
    }

    public init(conversation: any VoiceConversation, pipeline: any ReplyPipeline, host: any VoiceSessionHost) {
        self.conversation = conversation
        self.pipeline = pipeline
        self.host = host
        // Here as well as in `start()`, so the screen's first frame is already in this language.
        Bundle.useVoiceKitStrings(for: host.settings.speechLocale)
    }

    var modelName: String {
        pipeline.displayName(for: lastDecision)
    }

    /// Whether "Think deeper" does anything: Claude answers and deep mode is on.
    /// ponytail: tied to the app's `provider`, so a host that routes per request turns deep mode
    /// off instead; a `ReplyPipeline.supportsDeep` is the upgrade path once a second host needs it.
    var canGoDeep: Bool {
        host.settings.provider == .anthropic && host.settings.deepMode != .off
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
        Bundle.useVoiceKitStrings(for: host.settings.speechLocale)
        phase = .starting
        notice = nil
        if let problem = pipeline.missingSetup() {
            return fail(problem)
        }
        guard await AVAudioApplication.requestRecordPermission() else {
            return fail(String(localized: "Microphone access is off. Turn it on in Settings › Privacy & Security › Microphone.", bundle: .voiceKit))
        }
        guard await SpeechRecognizer.requestAuthorization() else {
            return fail(String(localized: "Speech recognition is off. Turn it on in Settings › Privacy & Security › Speech Recognition.", bundle: .voiceKit))
        }
        guard phase == .starting else { return }

        let settings = host.settings
        refineClipSamples = host.refineClipSamples
        // Keep twice the longest clip the host's second pass reads, so a full one always fits.
        let recognizer = SpeechRecognizer(localeIdentifier: settings.speechLocale, recordingCapacity: refineClipSamples > 0 ? 2 * refineClipSamples : nil)
        guard recognizer.isAvailable else {
            return fail(String(localized: "Speech recognition for \(settings.speechLocale) isn't available right now. Check your connection or pick another language in Settings.", bundle: .voiceKit))
        }
        recognizer.onEvent = { [weak self] event in self?.handle(event) }
        self.recognizer = recognizer
        host.voiceSessionWillStart(settings)
        hostStarted = true

        let speaker = Speaker(audio: audio, engine: speechEngine(for: settings), languageHint: settings.speechLocale)
        speaker.onChunkStarted = { [weak self] text in self?.assistantStartedSpeaking(text) }
        speaker.onCueStarted = { [weak self] _ in self?.cueStarted() }
        speaker.onDrained = { [weak self] in self?.speechDrained() }
        speaker.onFallback = { [weak self] message in
            self?.notice = String(localized: "\(message) Using the built-in voice instead.", bundle: .voiceKit)
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
            return fail(String(localized: "Couldn't start the microphone: \(error.localizedDescription)", bundle: .voiceKit))
        }

        observeAudioSession()
        UIApplication.shared.isIdleTimerDisabled = true
        turn = TurnDetector(silenceTimeout: settings.endOfTurnDelay)
        // Ducking listens for the user over the assistant's voice, so it needs echo cancellation,
        // and it only makes sense when talking over the assistant can interrupt it.
        duckingEnabled = settings.bargeInDucking && settings.voiceInterruptions && settings.echoCancellation
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
        cameraOpen = false
        pendingPhoto?.cancel()
        pendingPhoto = nil
        lastUserPhoto = nil
        if hostStarted {
            hostStarted = false
            host.voiceSessionDidStop()
        }
        audio.stop()
        for observer in observers {
            NotificationCenter.default.removeObserver(observer)
        }
        observers.removeAll()
        UIApplication.shared.isIdleTimerDisabled = false
        conversation.save()
        phase = .idle
        level = 0
    }

    func restart() async {
        stop()
        await start()
    }

    // MARK: - Controls

    /// Tapping the orb sends what's been said so far (with no words but a photo, it asks what's in
    /// the photo), or cuts the assistant off.
    func tapOrb() {
        switch phase {
        case .listening:
            if !userCaption.trimmed.isEmpty {
                commitUserTurn()
            } else if pendingPhoto != nil {
                commitUserTurn(asking: PendingPhoto.defaultQuestion(localeIdentifier: host.settings.speechLocale))
            }
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

    // MARK: - Photos

    /// Photos can be attached while the host can read them.
    var canAttachPhoto: Bool {
        host.photoReader != nil
    }

    /// The host's photo reader; without one, a read fails, so the photo gets the unreadable note.
    private var readPhoto: (Data) -> AsyncThrowingStream<String, Error> {
        host.photoReader ?? { _ in AsyncThrowingStream { $0.finish(throwing: CocoaError(.featureUnsupported)) } }
    }

    func toggleCamera() {
        cameraOpen.toggle()
        cameraUsedAt = now
    }

    /// Attaches a photo the camera took to the user's next turn, replacing any other, and starts
    /// reading it.
    func attachPhoto(_ data: Data) async {
        // A large photo takes a moment to decode, so off the main actor.
        guard let photo = await Task.detached(priority: .userInitiated, operation: { PendingPhoto(imageData: data) }).value, isLive else { return }
        pendingPhoto?.cancel()
        photo.startReading(readPhoto)
        pendingPhoto = photo
        cameraUsedAt = now
        // A reply started early didn't see the photo, and none starts while a photo is pending.
        discardTentative()
        updateEarlyStart()
    }

    func removePhoto() {
        pendingPhoto?.cancel()
        pendingPhoto = nil
        updateEarlyStart()
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
                    return fail(String(localized: "Speech recognition isn't available right now. Check your internet connection, or pick a language in Settings that your iPhone can recognize offline.", bundle: .voiceKit))
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
            if speaker?.hasPlayedCue == true, !host.settings.voiceInterruptions { return }
            if let words = interruptions.interruption(in: text, assistantSpeech: speaker?.recentSpeech ?? "") {
                if reopenPolicy.canReopen {
                    reopenUserTurn(with: words)
                } else {
                    // The reply may already have acted on the turn, so the turn stands and the
                    // words start the next one.
                    interruptReply(with: words)
                }
            }
        case .speaking:
            guard host.settings.voiceInterruptions, let speaker else { return }
            if let words = interruptions.interruption(in: text, assistantSpeech: speaker.recentSpeech) {
                ducker.interruptionConfirmed()
                // This request's audio also holds the assistant's voice, so keep the live text.
                recognizer?.feeder.dropSamples()
                interruptReply(with: words)
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

    /// Commits the user's turn: what they said, or `question` in its place.
    private func commitUserTurn(asking question: String? = nil) {
        let text = question ?? userCaption.trimmed
        guard !text.isEmpty else {
            restartRecognition()
            return
        }
        if question != nil { userCaption = text }
        let time = now
        var trace = TurnLatencyTrace()
        trace.mark(.speechEnded, at: min(turn.lastChange ?? time, time))
        trace.mark(.committed, at: time)
        trace.engine = host.settings.provider == .onDevice ? ReplyEngine.local.rawValue : ReplyEngine.cloud.rawValue
        if tentativeStartedThisTurn { trace.adoptedEarlyStart = false }
        self.trace = trace
        if host.settings.turnChime { audio.playChime() }

        let message = conversation.appendMessage(role: .user, text: text, isVoice: true, status: .complete)
        let photo = pendingPhoto
        message.photoData = photo?.jpeg
        pendingPhoto = nil
        lastUserPhoto = photo
        cameraUsedAt = now
        lastUserMessage = message
        // Take the turn's audio before the next request clears it.
        var samples: [Float] = []
        if let speechStart, let feeder = recognizer?.feeder {
            samples = feeder.turnClip(speechStart: speechStart, maxLength: refineClipSamples)
        }
        startReply(refining: message, carried: carriedText, samples: samples, photo: photo)
        carriedText = ""
        // Keep listening while the assistant thinks and talks, so the user can cut in.
        beginRecognition()
    }

    /// The user kept talking before the reply started: take back their turn and keep listening.
    /// Only while `reopenPolicy` allows it.
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
        // Its photo comes back too, unless a newer one was taken; a read the cancel cut short
        // starts again.
        pendingPhoto = pendingPhoto ?? lastUserPhoto
        pendingPhoto?.startReading(readPhoto)
        lastUserPhoto = nil
        phase = .listening
        carriedText = earlier
        userCaption = Self.join(earlier, text)
        turn.reset(transcript: userCaption, at: now)
        beginTurn(with: userCaption)
    }

    /// The user cut in: stop the reply and start their next turn with `words`. Their committed
    /// turn stays, and so does what the reply said or did before it stopped.
    private func interruptReply(with words: String) {
        cancelReply()
        // The running request already holds the interruption, so keep it.
        phase = .listening
        carriedText = ""
        userCaption = words
        turn.reset(transcript: words, at: now)
        beginTurn(with: words)
    }

    // MARK: - Early start

    /// Whether replies may start before the user's turn is committed.
    private var earlyStartAllowed: Bool {
        let settings = host.settings
        guard pipeline.supportsEarlyStart, settings.earlyReplyStart != .off else { return false }
        let localRoute = settings.provider == .onDevice || (settings.routingMode == .automatic && settings.preferOnDevice)
        // "On-device only" leaves cloud replies alone: an early start there may cost a request.
        // Automatic routing picks the route per request, so the pipeline also fails an early
        // reply it doesn't route on device; the reply then starts once the turn commits.
        if settings.earlyReplyStart == .onDeviceOnly, !localRoute { return false }
        // A second transcription and the local reply model would compete for the GPU.
        if localRoute, refineClipSamples > 0 { return false }
        return true
    }

    /// Lets the coordinator start tentative replies when allowed. Never for a deep reply: a
    /// discarded one would still spend deep-mode work and budget. Never while a photo is pending:
    /// the turn's request waits for its note.
    private func updateEarlyStart() {
        early.enabled = earlyStartAllowed && !deepNextTurn && pendingPhoto == nil
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
        tentative.start(pipeline.stream(ReplyRequest(messages: conversation.voiceMessages.map { $0.stored }, pendingUser: pendingUser, commitGate: gate, inputIsVoice: true, deep: deep)))
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
    private func replyEvents(for user: any VoiceMessage, deep: Bool) -> AsyncThrowingStream<AssistantEvent, Error> {
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
        return pipeline.stream(ReplyRequest(messages: conversation.voiceMessages.map { $0.stored }, inputIsVoice: true, deep: deep))
    }

    // MARK: - Replying

    /// Replies to `user`, first waiting for the note of its `photo` (and of any photo whose read was
    /// cut short), then swapping in the host's second transcription of the turn's audio, if any.
    /// `carried` is earlier text of the turn that `samples` don't cover.
    private func startReply(refining user: any VoiceMessage, carried: String, samples: [Float], photo: PendingPhoto?) {
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
        cuePolicy = CuePolicy(enabled: cuesEnabled, isVoice: true, fillerDelay: host.settings.fillerDelay, deep: deep)
        cuePolicy.replyStarted(at: now, expectedFirstText: LatencyLog.shared.expectedFirstText())
        reopenPolicy = TurnReopenPolicy()

        let reply = conversation.appendMessage(role: .assistant, text: "", isVoice: true, status: .streaming)
        self.reply = reply
        let locale = host.settings.speechLocale
        let messages = conversation.voiceMessages
        let host = host
        let readPhoto = readPhoto
        replyTask = Task { [weak self] in
            // The photo was being read since it was taken; the note goes in with the turn.
            var photoFailure: Error?
            do {
                try await PendingPhoto.noteUnread(messages, attached: photo.map { (user, $0) }, read: readPhoto) {
                    self?.waitForPhoto()
                }
            } catch {
                photoFailure = error
            }
            // The photo is read: "Reading the photo…" would no longer be true.
            if photoFailure == nil, let self, reply === self.reply, self.activity == Self.readingPhoto {
                self.activity = nil
            }
            let live = user.text
            if !samples.isEmpty, let heard = await host.refineTurn(samples, locale: locale), !Task.isCancelled {
                user.text = FinalTranscript.pick(live: live, carried: carried, refined: heard)
                #if DEBUG
                // ponytail: comparison line; delete after checking the results on the iPhone.
                print("[ASR] apple=\(live) refined=\(user.text)")
                #endif
                self?.trace?.mark(.transcriptRefined, at: ProcessInfo.processInfo.systemUptime)
            }
            guard let self, !Task.isCancelled, reply === self.reply else { return }
            // Decided only now: after any refinement, and reading the conversation as it is.
            let events: AsyncThrowingStream<AssistantEvent, Error>
            if let photoFailure {
                events = AsyncThrowingStream { $0.finish(throwing: photoFailure) }
            } else {
                events = self.replyEvents(for: user, deep: deep)
            }
            if user.text != live {
                self.conversation.userTurnRevised(from: live, to: user.text)
                if self.phase == .thinking { self.userCaption = user.text }
            }
            await self.consume(events, into: reply)
        }
    }

    /// The turn waits for its photo's read: say so, with the "Let me check" cue when cues play.
    private func waitForPhoto() {
        activity = Self.readingPhoto
        if let cue = cuePolicy.received(.cue(.checking), at: now) {
            cuePlayer?.play(cue)
        }
    }

    private func consume(_ stream: AsyncThrowingStream<AssistantEvent, Error>, into reply: any VoiceMessage) async {
        do {
            for try await event in stream {
                guard reply === self.reply else { return }
                reopenPolicy.received(event)
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
                        speak(Self.apology(refused: true, localeIdentifier: host.settings.speechLocale))
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
            speak(Self.apology(refused: false, localeIdentifier: host.settings.speechLocale))
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
        lastUserPhoto = nil
        conversation.save()
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
    private func keepSpokenPart(of reply: any VoiceMessage) {
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
            let key = host.openAIKey
            guard !key.isEmpty else {
                notice = String(localized: "Add an OpenAI API key in Settings to use OpenAI voices. Using the built-in voice.", bundle: .voiceKit)
                return .apple(voiceIdentifier: settings.appleVoiceIdentifier, rate: rate)
            }
            return .openAI(OpenAISpeechConfiguration(apiKey: key, model: settings.openAISpeechModel, voice: settings.openAIVoice))
        }
    }

    // MARK: - Cues, ducking and traces

    /// Cues play while the microphone listens for the user, so they need echo cancellation to
    /// keep the assistant's own voice from reading as the user's.
    private var cuesWanted: Bool {
        host.settings.spokenCues && host.settings.echoCancellation
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
        if cameraOpen, now - cameraUsedAt > Self.cameraIdleLimit {
            cameraOpen = false
        }
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
        fail(String(localized: "Voice chat stopped because another app or a call took over the audio.", bundle: .voiceKit))
    }

    private func audioRouteChanged() {
        guard phase == .listening || phase == .thinking || phase == .speaking else { return }
        do {
            try audio.restart()
            // Queued speech was lost with the old graph, so hand the turn back to the user.
            if phase != .listening { cancelReply() }
            listen(continuing: phase == .listening ? userCaption : "")
        } catch {
            fail(String(localized: "The audio route changed and the microphone couldn't restart.", bundle: .voiceKit))
        }
    }

    // MARK: - Helpers

    private func fail(_ message: String) {
        stop()
        phase = .failed(message)
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

extension String {
    var trimmed: String { trimmingCharacters(in: .whitespacesAndNewlines) }
}

extension Bundle {
    /// VoiceKit's strings, in the language of the latest voice session's `settings.speechLocale`
    /// when the app declares that language: an app's own language setting wins over the iPhone's,
    /// and an English-only app stays English. Until a session sets it, iOS picks (the app's language).
    /// ponytail: one language for the whole process, the latest session's; a bundle per session is
    /// the upgrade path if sessions in two languages ever run at once.
    static var voiceKit: Bundle { voiceKitCurrent.withLock { $0 } }

    static let voiceKitCurrent = OSAllocatedUnfairLock(initialState: Bundle.module)

    /// The languages the app declares (tests stand in another app's).
    static var voiceKitAppLanguages = Bundle.main.localizations

    static func useVoiceKitStrings(for localeIdentifier: String) {
        let strings = voiceKitStrings(for: localeIdentifier)
        voiceKitCurrent.withLock { $0 = strings }
    }

    /// The catalog's localization for `localeIdentifier`, from among the languages the app
    /// declares; when the app lacks that language, the app's own language, as iOS would pick it.
    static func voiceKitStrings(for localeIdentifier: String, appLanguages: [String] = voiceKitAppLanguages) -> Bundle {
        let declared = Bundle.module.localizations.filter(appLanguages.contains)
        let language = Bundle.preferredLocalizations(from: declared, forPreferences: [localeIdentifier] + Locale.preferredLanguages).first
        return language.flatMap { Bundle.module.path(forResource: $0, ofType: "lproj") }.flatMap(Bundle.init(path:)) ?? .module
    }
}
