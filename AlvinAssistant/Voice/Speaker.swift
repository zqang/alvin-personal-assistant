import AssistantKit
import AVFoundation
import NaturalLanguage

/// Installed system voices, best quality first.
enum VoiceCatalog {
    static func voices(forLanguage language: String) -> [AVSpeechSynthesisVoice] {
        let base = baseLanguage(language)
        return AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.language == language || baseLanguage($0.language) == base }
            .filter { !$0.voiceTraits.contains(.isNoveltyVoice) && !$0.voiceTraits.contains(.isPersonalVoice) }
            .sorted { lhs, rhs in
                let lhsExact = lhs.language == language
                let rhsExact = rhs.language == language
                if lhsExact != rhsExact { return lhsExact }
                if rank(lhs) != rank(rhs) { return rank(lhs) > rank(rhs) }
                return lhs.name < rhs.name
            }
    }

    static func bestVoice(forLanguage language: String) -> AVSpeechSynthesisVoice? {
        voices(forLanguage: language).first ?? AVSpeechSynthesisVoice(language: language)
    }

    static func qualityLabel(_ voice: AVSpeechSynthesisVoice) -> String {
        switch voice.quality {
        case .premium: return "Premium"
        case .enhanced: return "Enhanced"
        default: return "Standard"
        }
    }

    static func baseLanguage(_ identifier: String) -> String {
        String(identifier.split(whereSeparator: { $0 == "-" || $0 == "_" }).first ?? Substring(identifier)).lowercased()
    }

    private static func rank(_ voice: AVSpeechSynthesisVoice) -> Int {
        switch voice.quality {
        case .premium: return 3
        case .enhanced: return 2
        default: return 1
        }
    }
}

struct SpeechSynthesisError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

/// Speaks reply chunks in order. Synthesis of the next chunk overlaps playback of the current
/// one, and all audio goes through `AudioIO`'s player so echo cancellation can hear it.
@MainActor
final class Speaker {
    enum Engine {
        case apple(voiceIdentifier: String, rate: Float)
        case openAI(OpenAISpeechConfiguration)
    }

    /// A chunk started playing (the first one means the assistant started talking).
    var onChunkStarted: ((String) -> Void)?
    /// Everything queued since `beginReply()` has played and `finishInput()` was called.
    var onDrained: (() -> Void)?
    /// The cloud voice failed and speech switched to the on-device voice.
    var onFallback: ((String) -> Void)?

    /// Chunks that have started playing since `beginReply()`: what the user actually heard.
    private(set) var spokenText = ""

    /// Speech that may be coming out of the speaker right now, for telling echo from interruptions.
    var recentSpeech: String {
        ([spokenText] + playback.map(\.text)).joined(separator: " ")
    }

    private struct PlaybackItem {
        let text: String
        var produced = 0
        var outstanding = 0
        var synthesized = false
        var announced = false
    }

    private let audio: AudioIO
    private var engine: Engine
    private let languageHint: String
    private let synthesizer = AVSpeechSynthesizer()
    private var queue: [String] = []
    private var playback: [PlaybackItem] = []
    private var worker: Task<Void, Never>?
    private var generation = 0
    private var inputFinished = false
    private var drainReported = false
    private var voiceCache: [String: AVSpeechSynthesisVoice] = [:]

    init(audio: AudioIO, engine: Engine, languageHint: String) {
        self.audio = audio
        self.engine = engine
        self.languageHint = languageHint
    }

    func beginReply() {
        inputFinished = false
        drainReported = false
        spokenText = ""
    }

    func enqueue(_ text: String) {
        queue.append(text)
        startWorkerIfNeeded()
    }

    func finishInput() {
        inputFinished = true
        checkDrained()
    }

    /// Stops speaking immediately and drops everything queued.
    func stop() {
        generation += 1
        worker?.cancel()
        worker = nil
        queue.removeAll()
        playback.removeAll()
        synthesizer.stopSpeaking(at: .immediate)
        audio.stopPlayback()
        inputFinished = false
        drainReported = false
    }

    // MARK: - Queue

    private func startWorkerIfNeeded() {
        guard worker == nil else { return }
        let current = generation
        worker = Task { [weak self] in
            while let self, current == self.generation, let text = self.nextChunk() {
                await self.synthesize(text, generation: current)
            }
            self?.workerFinished(generation: current)
        }
    }

    private func nextChunk() -> String? {
        queue.isEmpty ? nil : queue.removeFirst()
    }

    private func workerFinished(generation: Int) {
        guard generation == self.generation else { return }
        worker = nil
        if queue.isEmpty {
            checkDrained()
        } else {
            startWorkerIfNeeded()
        }
    }

    private func synthesize(_ text: String, generation: Int) async {
        playback.append(PlaybackItem(text: text))
        switch engine {
        case .apple(let voiceIdentifier, let rate):
            await synthesizeWithApple(text, voiceIdentifier: voiceIdentifier, rate: rate, generation: generation)
        case .openAI(let configuration):
            do {
                try await synthesizeWithOpenAI(text, configuration: configuration, generation: generation)
            } catch {
                guard generation == self.generation, !Task.isCancelled else { return }
                // Keep talking with the on-device voice rather than going silent.
                engine = .apple(voiceIdentifier: "", rate: AVSpeechUtteranceDefaultSpeechRate)
                onFallback?(error.localizedDescription)
                if playback.last?.produced == 0 {
                    await synthesizeWithApple(text, voiceIdentifier: "", rate: AVSpeechUtteranceDefaultSpeechRate, generation: generation)
                }
            }
        }
        guard generation == self.generation, !playback.isEmpty else { return }
        playback[playback.count - 1].synthesized = true
        advancePlayback()
    }

    // MARK: - Playback tracking

    private func schedule(_ buffer: AVAudioPCMBuffer, generation: Int) {
        guard generation == self.generation, !playback.isEmpty else { return }
        playback[playback.count - 1].produced += 1
        playback[playback.count - 1].outstanding += 1
        audio.schedule(buffer) { [weak self] in
            Task { @MainActor in
                self?.bufferPlayed(generation: generation)
            }
        }
        advancePlayback()
    }

    private func bufferPlayed(generation: Int) {
        guard generation == self.generation, let first = playback.first, first.outstanding > 0 else { return }
        playback[0].outstanding -= 1
        advancePlayback()
    }

    private func advancePlayback() {
        while let first = playback.first, first.synthesized, first.outstanding == 0 {
            playback.removeFirst()
        }
        if let first = playback.first, first.produced > 0, !first.announced {
            playback[0].announced = true
            spokenText = Self.join(spokenText, first.text)
            onChunkStarted?(first.text)
        }
        checkDrained()
    }

    /// Joins sentences with a space, except around Chinese/Japanese text, which doesn't use one.
    private static func join(_ first: String, _ second: String) -> String {
        guard !first.isEmpty else { return second }
        let noSpace = "。！？；，、：".contains(first.last ?? " ")
            || (second.unicodeScalars.first.map { $0.value >= 0x3000 && $0.value <= 0x9FFF } ?? false)
        return noSpace ? first + second : first + " " + second
    }

    private func checkDrained() {
        guard inputFinished, !drainReported, queue.isEmpty, worker == nil, playback.isEmpty else { return }
        drainReported = true
        onDrained?()
    }

    // MARK: - On-device voice

    private func synthesizeWithApple(_ text: String, voiceIdentifier: String, rate: Float, generation: Int) async {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = voice(for: text, preferredIdentifier: voiceIdentifier)
        utterance.rate = rate
        let converter = PCMConverter(to: audio.playbackFormat)
        for await buffer in Self.renderedBuffers(of: utterance, with: synthesizer) {
            guard generation == self.generation else { return }
            if let converted = converter.convert(buffer) {
                schedule(converted, generation: generation)
            }
        }
    }

    /// Renders an utterance to PCM instead of playing it, so it can go through the engine.
    nonisolated private static func renderedBuffers(
        of utterance: AVSpeechUtterance,
        with synthesizer: AVSpeechSynthesizer
    ) -> AsyncStream<AVAudioPCMBuffer> {
        AsyncStream { continuation in
            // A zero-length buffer marks the end; the timeout covers a synthesizer that never sends it.
            let timeout = Task {
                try? await Task.sleep(nanoseconds: 15_000_000_000)
                continuation.finish()
            }
            continuation.onTermination = { _ in timeout.cancel() }
            synthesizer.write(utterance) { buffer in
                guard let pcm = buffer as? AVAudioPCMBuffer, pcm.frameLength > 0 else {
                    continuation.finish()
                    return
                }
                continuation.yield(pcm)
            }
        }
    }

    /// The chosen voice, or when a chunk is clearly in another language (an English name in a
    /// Chinese reply, say), the best voice for that language.
    private func voice(for text: String, preferredIdentifier: String) -> AVSpeechSynthesisVoice? {
        let chosen = preferredIdentifier.isEmpty ? nil : AVSpeechSynthesisVoice(identifier: preferredIdentifier)
        let expected = chosen?.language ?? languageHint
        if let detected = Self.dominantLanguage(of: text),
           VoiceCatalog.baseLanguage(detected) != VoiceCatalog.baseLanguage(expected) {
            return cachedVoice(forLanguage: Self.voiceLanguage(for: detected))
        }
        return chosen ?? cachedVoice(forLanguage: languageHint)
    }

    private func cachedVoice(forLanguage language: String) -> AVSpeechSynthesisVoice? {
        if let voice = voiceCache[language] { return voice }
        let voice = VoiceCatalog.bestVoice(forLanguage: language)
        voiceCache[language] = voice
        return voice
    }

    private static func dominantLanguage(of text: String) -> String? {
        guard text.count >= 6 else { return nil }
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)
        guard let best = recognizer.languageHypotheses(withMaximum: 1).max(by: { $0.value < $1.value }),
              best.value >= 0.75 else { return nil }
        return best.key.rawValue
    }

    /// Maps a detected language ("zh-Hans", "en") to a voice language ("zh-CN", "en-US").
    private static func voiceLanguage(for detected: String) -> String {
        switch detected {
        case "zh-Hans": return "zh-CN"
        case "zh-Hant": return "zh-TW"
        case "en": return "en-US"
        default: return detected
        }
    }

    // MARK: - OpenAI voice

    private func synthesizeWithOpenAI(_ text: String, configuration: OpenAISpeechConfiguration, generation: Int) async throws {
        let request = try OpenAISpeech.request(for: text, configuration: configuration)
        for try await samples in Self.pcmChunks(for: request) {
            guard generation == self.generation else { return }
            scheduleSamples(samples, generation: generation)
        }
    }

    private func scheduleSamples(_ samples: [Float], generation: Int) {
        guard !samples.isEmpty,
              let buffer = AVAudioPCMBuffer(pcmFormat: audio.playbackFormat, frameCapacity: AVAudioFrameCount(samples.count)),
              let channel = buffer.floatChannelData?[0] else { return }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { source in
            if let base = source.baseAddress {
                channel.update(from: base, count: samples.count)
            }
        }
        schedule(buffer, generation: generation)
    }

    /// Downloads speech as ~200 ms chunks of 24 kHz samples, off the main actor.
    nonisolated private static func pcmChunks(for request: URLRequest) -> AsyncThrowingStream<[Float], Error> {
        AsyncThrowingStream { continuation in
            let download = Task.detached {
                do {
                    let (bytes, response) = try await URLSession.shared.bytes(for: request)
                    let status = (response as? HTTPURLResponse)?.statusCode ?? 0
                    guard (200..<300).contains(status) else {
                        var body = Data()
                        for try await byte in bytes {
                            body.append(byte)
                            if body.count >= 4096 { break }
                        }
                        let message = (try? JSONValue.parse(body))?["error"]?["message"]?.stringValue ?? "HTTP \(status)"
                        throw SpeechSynthesisError(message: "OpenAI voice failed: \(message)")
                    }
                    let chunkBytes = 9_600
                    var decoder = PCM16Decoder()
                    var pending = Data()
                    pending.reserveCapacity(chunkBytes)
                    for try await byte in bytes {
                        pending.append(byte)
                        if pending.count >= chunkBytes {
                            continuation.yield(decoder.decode(pending))
                            pending.removeAll(keepingCapacity: true)
                        }
                    }
                    if !pending.isEmpty {
                        continuation.yield(decoder.decode(pending))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in download.cancel() }
        }
    }
}
