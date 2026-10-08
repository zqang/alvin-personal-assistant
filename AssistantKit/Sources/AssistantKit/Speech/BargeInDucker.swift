import Foundation

/// Lowers the assistant's playback as soon as the user seems to talk over it, before the
/// recognizer has words to confirm an interruption, so the user isn't shouting over the reply.
///
/// Feed it the echo-cancelled microphone level. While the assistant is speaking:
/// - `minimumVoice` of unbroken voice at least `marginDB` over the noise floor (and above the
///   VAD's absolute threshold) gives `.duck`;
/// - once ducked, `restoreAfter` without voice gives `.restore`, unless an interruption was
///   confirmed. Restoring on quiet rather than on a timer keeps playback down while the user is
///   still talking, instead of pumping up and down.
/// When the assistant stops speaking while ducked, it gives `.restore` straight away.
/// After `interruptionConfirmed()` it gives nothing until `reset()`; the reply is being stopped,
/// and the caller restores full volume for the next one.
public struct BargeInDucker: Sendable {
    public enum Output: Equatable, Sendable {
        case none
        /// Lower playback.
        case duck
        /// Back to full volume.
        case restore
    }

    /// dB above the noise floor that counts as the user's voice.
    public var marginDB: Float = 18
    /// Seconds of unbroken voice before ducking.
    public var minimumVoice: TimeInterval = 0.15
    /// Seconds without voice after which ducked playback is restored.
    public var restoreAfter: TimeInterval = 0.6

    private var vad = EnergyVAD()
    private var voiceSince: TimeInterval?
    private var lastVoice: TimeInterval?
    private var ducked = false
    private var confirmed = false

    /// Absorbs rounding in caller-supplied times.
    private static let tolerance: TimeInterval = 1e-9

    public init() {}

    /// Whether playback is currently ducked.
    public var isDucked: Bool { ducked }

    /// The current noise floor estimate, in dBFS.
    public var noiseFloor: Float { vad.noiseFloor }

    /// A microphone level (dBFS) at `time`; `speaking` is whether the assistant's voice is playing.
    public mutating func level(_ dB: Float, at time: TimeInterval, speaking: Bool) -> Output {
        _ = vad.isVoice(level: dB)   // keeps the noise floor current
        guard !confirmed else { return .none }
        guard speaking else {
            voiceSince = nil
            lastVoice = nil
            guard ducked else { return .none }
            ducked = false
            return .restore
        }
        if dB >= max(vad.noiseFloor + marginDB, vad.absoluteThreshold) {
            lastVoice = time
            let since = voiceSince ?? time
            voiceSince = since
            guard !ducked, time - since + Self.tolerance >= minimumVoice else { return .none }
            ducked = true
            return .duck
        }
        voiceSince = nil
        guard ducked, let lastVoice, time - lastVoice + Self.tolerance >= restoreAfter else { return .none }
        ducked = false
        self.lastVoice = nil
        return .restore
    }

    /// The recognizer confirmed a real interruption: keep playback ducked; the reply is stopping.
    public mutating func interruptionConfirmed() {
        confirmed = true
    }

    /// Starts afresh for the next reply. Keeps the noise floor, which belongs to the room.
    public mutating func reset() {
        voiceSince = nil
        lastVoice = nil
        ducked = false
        confirmed = false
    }
}
