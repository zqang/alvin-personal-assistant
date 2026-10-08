import Foundation

/// When a tentative reply may start while the user's voice turn is still open.
public struct EarlyReplyPolicy: Equatable, Sendable {
    /// Seconds the transcript must stay unchanged before a tentative reply starts.
    public var delay: TimeInterval = 0.35
    /// The same wait when the transcript ends a sentence ("?", "。"), which rarely continues.
    public var afterSentenceEnd: TimeInterval = 0.25
    /// A tentative reply needs at least this many words...
    public var minimumWords = 2
    /// ...or at least this many Chinese, Japanese or Korean characters.
    public var minimumCJKCharacters = 3
    /// The tuner keeps the delay between these bounds.
    public var minDelay: TimeInterval = 0.25
    public var maxDelay: TimeInterval = 0.6

    public init() {}
}

/// Compares the text of voice turns.
public enum TurnText {
    /// `s` in Unicode NFKC form, lowercased, without punctuation or whitespace, so a refined
    /// transcript that only adds punctuation or capitals still matches the live one.
    public static func normalized(_ s: String) -> String {
        var kept = String.UnicodeScalarView()
        for scalar in s.precomposedStringWithCompatibilityMapping.lowercased().unicodeScalars
        where !ignored.contains(scalar) {
            kept.append(scalar)
        }
        return String(kept)
    }

    /// Whether two transcripts ask for the same thing: equal once normalized.
    public static func sameRequest(_ a: String, _ b: String) -> Bool {
        normalized(a) == normalized(b)
    }

    private static let ignored = CharacterSet.punctuationCharacters.union(.whitespacesAndNewlines)
}

/// Starts a reply before the user's voice turn is committed, so the answer is ready sooner.
///
/// Feed it the live transcript and a regular tick. Once the transcript has been stable for the
/// current delay (shorter after a sentence end) and is long enough, it asks for a tentative
/// reply. The caller streams that reply silently, buffering its events, until the turn commits:
/// - the committed text is the same request (`TurnText.sameRequest`): `adopt`, so open the
///   reply's commit gate and replay what it buffered;
/// - otherwise: cancel it and start a fresh reply to the committed text.
/// A transcript that changes into a different request cancels the tentative reply at once.
///
/// A tuner moves the delay by 0.05 s after each tentative reply is settled: up when more than
/// 30% of the last 20 were discarded, down when fewer than 10% were, within the policy's bounds.
/// It waits for 5 settled replies before its first move. Replies dropped by `reset()` don't count.
public struct EarlyReplyCoordinator: Sendable {
    public enum Action: Equatable, Sendable {
        /// Start a reply to `text` now. Nothing of it may be shown or spoken until `adopt`.
        case startTentative(text: String)
        /// Cancel the tentative reply.
        case cancelTentative
        /// The committed turn is the tentative request: open its gate and replay its buffered events.
        case adopt
        /// Start the reply to the committed `text` the usual way.
        case startFresh(text: String)
    }

    public let policy: EarlyReplyPolicy
    /// Whether new tentative replies may start. One already running is still settled by
    /// `commit` or a transcript change after this turns false.
    public var enabled: Bool
    /// The request of the tentative reply that is running, if any.
    public private(set) var tentativeText: String?
    /// The tuned wait for a stable transcript; `policy.delay` until the tuner moves it.
    public private(set) var currentDelay: TimeInterval

    private var transcript = ""
    private var lastChange: TimeInterval?
    /// Normalized requests that already got a tentative reply this turn.
    private var started: Set<String> = []
    /// Settled tentative replies, oldest first: true when discarded.
    private var outcomes: [Bool] = []

    static let tunerWindow = 20
    static let tunerMinimumSamples = 5
    static let tunerStep: TimeInterval = 0.05
    /// Absorbs rounding in caller-supplied times.
    private static let tolerance: TimeInterval = 1e-9
    private static let sentenceEnds: Set<Character> = [".", "?", "!", "。", "？", "！"]

    public init(policy: EarlyReplyPolicy = .init(), enabled: Bool) {
        self.policy = policy
        self.enabled = enabled
        currentDelay = min(max(policy.delay, policy.minDelay), policy.maxDelay)
    }

    /// The live transcript of the open turn changed. A different request cancels the tentative reply.
    public mutating func transcriptChanged(_ text: String, at time: TimeInterval) -> [Action] {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard text != transcript else { return [] }
        transcript = text
        lastChange = time
        guard let tentative = tentativeText, !TurnText.sameRequest(tentative, text) else { return [] }
        tentativeText = nil
        record(discarded: true)
        return [.cancelTentative]
    }

    /// Starts a tentative reply once the transcript has been stable long enough. At most once
    /// per request in a turn, so a recognizer that flips back and forth doesn't restart it.
    public mutating func tick(at time: TimeInterval) -> [Action] {
        guard enabled, tentativeText == nil, let changed = lastChange, isLongEnough(transcript) else { return [] }
        guard time - changed + Self.tolerance >= wait(for: transcript) else { return [] }
        let request = TurnText.normalized(transcript)
        guard !request.isEmpty, !started.contains(request) else { return [] }
        started.insert(request)
        tentativeText = transcript
        return [.startTentative(text: transcript)]
    }

    /// The turn ended with `finalText` (after any refinement): adopt the tentative reply if it
    /// answers the same request, otherwise replace it. Clears the turn for the next one.
    public mutating func commit(finalText: String, at time: TimeInterval) -> [Action] {
        let text = finalText.trimmingCharacters(in: .whitespacesAndNewlines)
        let tentative = tentativeText
        reset()
        guard let tentative else { return [.startFresh(text: text)] }
        if TurnText.sameRequest(tentative, text) {
            record(discarded: false)
            return [.adopt]
        }
        record(discarded: true)
        return [.cancelTentative, .startFresh(text: text)]
    }

    /// Forgets the open turn, e.g. after a barge-in or stop. The caller cancels any tentative
    /// reply itself. The tuner keeps its history.
    public mutating func reset() {
        transcript = ""
        lastChange = nil
        tentativeText = nil
        started = []
    }

    /// How long `text` must be stable. The tuner moves the sentence-end wait by as much as it
    /// moved the delay.
    func wait(for text: String) -> TimeInterval {
        guard let last = text.last, Self.sentenceEnds.contains(last) else { return currentDelay }
        return min(max(policy.afterSentenceEnd + currentDelay - policy.delay, 0), currentDelay)
    }

    private func isLongEnough(_ text: String) -> Bool {
        let tokens = SpeechTokenizer.tokens(text)
        let cjk = tokens.filter(SpeechTokenizer.isCJKToken).count
        return tokens.count - cjk >= policy.minimumWords || cjk >= policy.minimumCJKCharacters
    }

    private mutating func record(discarded: Bool) {
        outcomes.append(discarded)
        if outcomes.count > Self.tunerWindow { outcomes.removeFirst(outcomes.count - Self.tunerWindow) }
        guard outcomes.count >= Self.tunerMinimumSamples else { return }
        let rate = Double(outcomes.filter { $0 }.count) / Double(outcomes.count)
        var delay = currentDelay
        if rate > 0.3 {
            delay += Self.tunerStep
        } else if rate < 0.1 {
            delay -= Self.tunerStep
        }
        // Whole milliseconds, so repeated steps don't drift.
        currentDelay = (min(max(delay, policy.minDelay), policy.maxDelay) * 1000).rounded() / 1000
    }
}
