import AVFoundation

enum AudioError: LocalizedError {
    case microphoneUnavailable

    var errorDescription: String? {
        "The microphone isn't available."
    }
}

/// Owns the audio session and engine: microphone input with Apple's voice processing (echo
/// cancellation, noise suppression) and one player node for all assistant speech. Playing speech
/// through the same engine is what lets the echo canceller remove it from the microphone signal.
///
/// Not main-actor isolated: taps and completion handlers run on audio threads.
final class AudioIO: @unchecked Sendable {
    let engine = AVAudioEngine()
    let player = AVAudioPlayerNode()
    /// Every speech buffer is converted to this format before it is scheduled.
    let playbackFormat = AVAudioFormat(standardFormatWithSampleRate: 24_000, channels: 1)!
    /// Microphone audio is delivered in this format.
    let recognitionFormat = AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1)!

    /// Set these before `start`. They are called on the audio thread.
    var onInput: (@Sendable (AVAudioPCMBuffer) -> Void)?
    var onInputLevel: (@Sendable (Float) -> Void)?
    var onOutputLevel: (@Sendable (Float) -> Void)?

    private var voiceProcessing = true
    private var started = false
    /// Serializes playback gain changes; `gainGeneration` is touched only on it.
    private let gainQueue = DispatchQueue(label: "AudioIO.playbackGain")
    private var gainGeneration = 0
    private static let gainStep: TimeInterval = 0.01
    /// A short tone in `playbackFormat`, made on first use.
    private lazy var chime: AVAudioPCMBuffer? = AudioIO.makeChime(format: playbackFormat)

    init() {
        // Attached up front so stopping playback is safe even if the engine never started.
        engine.attach(player)
    }

    func start(echoCancellation: Bool) throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(
            .playAndRecord,
            mode: echoCancellation ? .voiceChat : .default,
            options: [.defaultToSpeaker, .allowBluetooth]
        )
        try session.setActive(true)
        started = true
        voiceProcessing = echoCancellation
        setPlaybackGain(1, ramp: 0)
        try configureEngine()
    }

    /// Rebuilds the graph after the route or hardware format changed; the engine stops itself then.
    func restart() throws {
        teardownEngine()
        try configureEngine()
    }

    func stop() {
        guard started else { return }
        started = false
        teardownEngine()
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    func schedule(_ buffer: AVAudioPCMBuffer, completion: @escaping @Sendable () -> Void) {
        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { _ in
            completion()
        }
        if engine.isRunning, !player.isPlaying {
            player.play()
        }
    }

    func stopPlayback() {
        player.stop()
    }

    /// Moves the volume of all assistant speech to `gain` (0...1) in 10 ms steps over `ramp`
    /// seconds, so a duck doesn't click. A newer call takes over from a ramp still running.
    func setPlaybackGain(_ gain: Float, ramp: TimeInterval) {
        let target = min(max(gain, 0), 1)
        let steps = max(Int((max(ramp, 0) / Self.gainStep).rounded()), 1)
        gainQueue.async { [weak self] in
            guard let self else { return }
            self.gainGeneration += 1
            let generation = self.gainGeneration
            let start = self.player.volume
            for step in 1...steps {
                let fraction = Float(step) / Float(steps)
                self.gainQueue.asyncAfter(deadline: .now() + Self.gainStep * Double(step - 1)) { [weak self] in
                    guard let self, self.gainGeneration == generation else { return }
                    self.player.volume = start + (target - start) * fraction
                }
            }
        }
    }

    /// Plays a short tone through the speech player, e.g. when the user's turn ends.
    func playChime() {
        guard engine.isRunning, let chime else { return }
        player.scheduleBuffer(chime, completionHandler: nil)
        if !player.isPlaying {
            player.play()
        }
    }

    /// An 80 ms sine tone with 10 ms fades, so it starts and ends without a click.
    private static func makeChime(format: AVAudioFormat) -> AVAudioPCMBuffer? {
        let frames = Int(format.sampleRate * 0.08)
        guard frames > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)),
              let channel = buffer.floatChannelData?[0] else { return nil }
        buffer.frameLength = AVAudioFrameCount(frames)
        let fade = max(format.sampleRate * 0.01, 1)
        let frequency = 880.0
        for frame in 0..<frames {
            let envelope = min(1, Double(frame) / fade, Double(frames - 1 - frame) / fade)
            channel[frame] = Float(0.2 * envelope * sin(2 * Double.pi * frequency * Double(frame) / format.sampleRate))
        }
        return buffer
    }

    private func configureEngine() throws {
        let input = engine.inputNode
        if input.isVoiceProcessingEnabled != voiceProcessing {
            do {
                try input.setVoiceProcessingEnabled(voiceProcessing)
            } catch {
                // Some routes and the Simulator can't do voice processing; plain audio still works.
                voiceProcessing = input.isVoiceProcessingEnabled
            }
        }
        engine.connect(player, to: engine.mainMixerNode, format: playbackFormat)

        let inputFormat = input.outputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0 else {
            throw AudioError.microphoneUnavailable
        }
        let converter = PCMConverter(to: recognitionFormat)
        input.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { [weak self] buffer, _ in
            guard let self, let converted = converter.convert(buffer) else { return }
            self.onInputLevel?(AudioIO.level(of: converted))
            self.onInput?(converted)
        }
        engine.mainMixerNode.installTap(onBus: 0, bufferSize: 1024, format: nil) { [weak self] buffer, _ in
            self?.onOutputLevel?(AudioIO.level(of: buffer))
        }
        engine.prepare()
        try engine.start()
    }

    private func teardownEngine() {
        engine.inputNode.removeTap(onBus: 0)
        engine.mainMixerNode.removeTap(onBus: 0)
        player.stop()
        engine.stop()
    }

    /// RMS level of the first channel in dBFS (-100 for silence or non-float audio).
    static func level(of buffer: AVAudioPCMBuffer) -> Float {
        guard let channels = buffer.floatChannelData, buffer.frameLength > 0 else { return -100 }
        let samples = UnsafeBufferPointer(start: channels[0], count: Int(buffer.frameLength))
        var sum: Float = 0
        for sample in samples {
            sum += sample * sample
        }
        let rms = (sum / Float(buffer.frameLength)).squareRoot()
        return 20 * log10(max(rms, 0.000_01))
    }
}

/// Converts PCM buffers to one output format (sample rate, channels, sample type), keeping the
/// converter's state between calls so streamed audio resamples without clicks.
/// Use each instance from one thread at a time.
final class PCMConverter: @unchecked Sendable {
    let outputFormat: AVAudioFormat
    private var converter: AVAudioConverter?

    init(to outputFormat: AVAudioFormat) {
        self.outputFormat = outputFormat
    }

    func convert(_ buffer: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        if buffer.format == outputFormat { return buffer }
        if converter == nil || converter?.inputFormat != buffer.format {
            converter = AVAudioConverter(from: buffer.format, to: outputFormat)
        }
        guard let converter else { return nil }

        let ratio = outputFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount((Double(buffer.frameLength) * ratio).rounded(.up)) + 64
        guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else { return nil }

        let feed = OneShotFeed(buffer)
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, inputStatus in
            guard let next = feed.take() else {
                inputStatus.pointee = .noDataNow
                return nil
            }
            inputStatus.pointee = .haveData
            return next
        }
        guard status != .error, error == nil, output.frameLength > 0 else { return nil }
        return output
    }
}

/// Hands the converter its input buffer exactly once per `convert` call.
private final class OneShotFeed: @unchecked Sendable {
    private var buffer: AVAudioPCMBuffer?

    init(_ buffer: AVAudioPCMBuffer) {
        self.buffer = buffer
    }

    func take() -> AVAudioPCMBuffer? {
        defer { buffer = nil }
        return buffer
    }
}
