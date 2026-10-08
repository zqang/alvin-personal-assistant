import Foundation

/// Decides which spoken cues a voice reply plays, so the user hears at most one short phrase
/// before the answer instead of a stream of status chatter.
///
/// - Nothing is ever played for typed input, or when cues are turned off.
/// - At most one cue before the answer: a `.cue` event from the reply, or the `.working` filler.
///   In deep mode `.stillThinking` may follow as a second one.
/// - The filler plays `fillerDelay` after the reply started, or after 0.6 s when the reply is
///   expected to take more than 2.5 s to its first text.
/// - The first `.reply(.text)` (or the end of the reply) closes the policy: no cue after any text.
///
/// Create one per reply. Times are seconds on the caller's clock.
public struct CuePolicy: Sendable {
    /// Expected seconds to the first text above which the filler plays early.
    static let slowReplyThreshold: TimeInterval = 2.5
    /// When the filler plays for a reply expected to be slow.
    static let earlyFillerDelay: TimeInterval = 0.6

    private let isActive: Bool
    private let fillerDelay: TimeInterval
    private var deep: Bool
    private var closed = false
    private var cuePlayed = false
    private var stillThinkingPlayed = false
    private var fillerDue: TimeInterval?

    /// - Parameters:
    ///   - enabled: the spoken-cues setting.
    ///   - fillerDelay: seconds after the reply starts before the `.working` filler; negative or
    ///     infinite turns the filler off.
    ///   - deep: the reply runs in deep mode. A `.routed` deep decision also turns this on.
    public init(enabled: Bool, isVoice: Bool, fillerDelay: TimeInterval, deep: Bool = false) {
        isActive = enabled && isVoice
        self.fillerDelay = fillerDelay
        self.deep = deep
    }

    /// The reply started at `time`. Arms the filler. `expectedFirstText` is the expected time from
    /// now to the first text, when known.
    public mutating func replyStarted(at time: TimeInterval, expectedFirstText: TimeInterval?) {
        guard fillerDelay.isFinite, fillerDelay >= 0 else {
            fillerDue = nil
            return
        }
        var delay = fillerDelay
        if let expectedFirstText, expectedFirstText > Self.slowReplyThreshold {
            delay = min(delay, Self.earlyFillerDelay)
        }
        fillerDue = time + delay
    }

    /// The cue to play for `event`, if any.
    public mutating func received(_ event: AssistantEvent, at time: TimeInterval) -> SpokenCue? {
        switch event {
        case .reply(.text), .reply(.finished):
            closed = true
            fillerDue = nil
            return nil
        case .cue(let cue):
            return allow(cue)
        case .routed(let decision):
            if decision.mode == .deep { deep = true }
            return nil
        case .reply(.activity), .toolRound, .progress:
            return nil
        }
    }

    /// The filler, once it is due and nothing else has been played.
    public mutating func tick(at time: TimeInterval) -> SpokenCue? {
        // A small tolerance absorbs rounding in caller-supplied times.
        guard let due = fillerDue, time + 1e-9 >= due else { return nil }
        fillerDue = nil
        return allow(.working)
    }

    private mutating func allow(_ cue: SpokenCue) -> SpokenCue? {
        guard isActive, !closed else { return nil }
        if cue == .stillThinking, deep {
            guard !stillThinkingPlayed else { return nil }
            stillThinkingPlayed = true
            return cue
        }
        guard !cuePlayed else { return nil }
        cuePlayed = true
        fillerDue = nil
        return cue
    }
}
