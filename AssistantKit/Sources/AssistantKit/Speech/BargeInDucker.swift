import Foundation

/// Lowers the assistant's playback as soon as the user seems to talk over it, before the
/// recognizer has words to confirm an interruption, so the user isn't shouting over the reply.
///
/// Feed it the echo-cancelled microphone level. While the assistant is speaking:
/// - `minimumVoice` of unbroken voice at least `marginDB` over the noise floor (and above the
///   VAD's absolute threshold) gives `.duck`;
/// - `restoreAfter` after the duck gives `.restore`, unless an interruption was confirmed by
///   then: voice the recognizer doesn't confirm in time was not an interruption.
/// The run of voice that ducked can't duck again once restored; only a fresh run, after a frame
/// below the threshold, can. So a television or a long backchannel lowers playback once instead
/// of pumping it up and down.
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
    /// Seconds after ducking at which playback is restored, unless an interruption was confirmed.
    public var restoreAfter: TimeInterval = 0.6

    private var vad = EnergyVAD()
    /// Start of the current unbroken run of voice.
    private var voiceSince: TimeInterval?
    /// The current run of voice already ducked and was restored, so it can't duck again.
    private var runSpent = false
    private var duckedAt: TimeInterval?
    private var confirmed = false

    /// Absorbs rounding in caller-supplied times.
    private static let tolerance: TimeInterval = 1e-9

    public init() {}

    /// Whether playback is currently ducked.
    public var isDucked: Bool { duckedAt != nil }

    /// The current noise floor estimate, in dBFS.
    public var noiseFloor: Float { vad.noiseFloor }

    /// A microphone level (dBFS) at `time`; `speaking` is whether the assistant's voice is playing.
    public mutating func level(_ dB: Float, at time: TimeInterval, speaking: Bool) -> Output {
        _ = vad.isVoice(level: dB)   // keeps the noise floor current
        guard !confirmed else { return .none }
        guard speaking else {
            voiceSince = nil
            runSpent = false
            guard duckedAt != nil else { return .none }
            duckedAt = nil
            return .restore
        }
        let isVoice = dB >= max(vad.noiseFloor + marginDB, vad.absoluteThreshold)
        if !isVoice {
            voiceSince = nil
            runSpent = false
        }
        if let duckedAt {
            guard time - duckedAt + Self.tolerance >= restoreAfter else { return .none }
            self.duckedAt = nil
            runSpent = isVoice
            return .restore
        }
        guard isVoice, !runSpent else { return .none }
        let since = voiceSince ?? time
        voiceSince = since
        guard time - since + Self.tolerance >= minimumVoice else { return .none }
        duckedAt = time
        return .duck
    }

    /// The recognizer confirmed a real interruption: keep playback ducked; the reply is stopping.
    public mutating func interruptionConfirmed() {
        confirmed = true
    }

    /// Starts afresh for the next reply. Keeps the noise floor, which belongs to the room.
    public mutating func reset() {
        voiceSince = nil
        runSpent = false
        duckedAt = nil
        confirmed = false
    }
}
