import Foundation

/// Voice activity from microphone level, measured against an adaptive noise floor so a fan
/// or street noise doesn't count as the user still talking.
public struct EnergyVAD: Sendable {
    /// dB above the noise floor that counts as voice.
    public var margin: Float = 12
    /// Levels below this (dBFS) are never voice.
    public var absoluteThreshold: Float = -55
    public private(set) var noiseFloor: Float = -60

    public init() {}

    /// Updates the noise floor with a level (dBFS) and reports whether it sounds like voice.
    public mutating func isVoice(level: Float) -> Bool {
        let level = max(level, -100)
        // Fall quickly when it gets quieter; rise slowly so speech itself doesn't raise the floor.
        let rate: Float = level < noiseFloor ? 0.2 : 0.002
        noiseFloor = min(max(noiseFloor + (level - noiseFloor) * rate, -90), -25)
        return level > max(noiseFloor + margin, absoluteThreshold)
    }
}

/// Decides when the user has finished speaking, from how long the live transcript has been
/// stable and whether the microphone still hears voice.
public struct TurnDetector: Sendable {
    /// Seconds of silence after the last recognized word that end a turn.
    public var silenceTimeout: TimeInterval
    /// A stable transcript ends the turn after this much longer, even if background noise keeps the VAD active.
    public var noiseGrace: TimeInterval = 1.5

    public private(set) var transcript = ""
    private var lastTranscriptChange: TimeInterval?
    /// When the transcript last changed: the best estimate of when the user stopped speaking.
    public var lastChange: TimeInterval? { lastTranscriptChange }
    private var lastVoice: TimeInterval?
    private var vad = EnergyVAD()

    public init(silenceTimeout: TimeInterval = 0.9) {
        self.silenceTimeout = silenceTimeout
    }

    /// Starts a new turn, optionally continuing from text already heard.
    public mutating func reset(transcript: String = "", at time: TimeInterval? = nil) {
        self.transcript = transcript
        lastTranscriptChange = transcript.isEmpty ? nil : time
        lastVoice = nil
    }

    public mutating func transcriptChanged(_ text: String, at time: TimeInterval) {
        guard text != transcript else { return }
        transcript = text
        lastTranscriptChange = time
    }

    public mutating func audioLevel(_ level: Float, at time: TimeInterval) {
        if vad.isVoice(level: level) { lastVoice = time }
    }

    public func shouldEndTurn(at time: TimeInterval) -> Bool {
        let text = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, let changed = lastTranscriptChange else { return false }
        let timeout = Self.timeout(for: text, base: silenceTimeout)
        let stableFor = time - changed
        guard stableFor >= timeout else { return false }
        if stableFor >= timeout + noiseGrace { return true }
        // Recent voice without new words usually means the recognizer is about to catch up.
        if let voice = lastVoice, time - voice < timeout * 0.6 { return false }
        return true
    }

    /// Waits less after a finished sentence and longer after a word that promises more
    /// ("and", "because", "然后"), so pauses mid-thought don't cut the user off.
    static func timeout(for text: String, base: TimeInterval) -> TimeInterval {
        guard let last = text.last else { return base }
        if ".?!。？！".contains(last) { return base * 0.8 }
        let tokens = SpeechTokenizer.tokens(text)
        if let word = tokens.last, continuationWords.contains(word) { return base * 1.6 }
        if tokens.count >= 2, continuationWords.contains(tokens.suffix(2).joined()) { return base * 1.6 }
        return base
    }

    private static let continuationWords: Set<String> = [
        "and", "but", "or", "so", "because", "if", "then", "like", "the", "a", "an", "to", "with", "of",
        "for", "um", "uh", "erm", "hmm", "maybe", "actually", "basically",
        "和", "跟", "的", "然后", "就是", "那个", "这个", "所以", "但是", "因为", "而且", "还有", "嗯", "呃",
    ]
}

/// Tells a real interruption apart from the assistant's own voice leaking back into the microphone.
public struct BargeInDetector: Sendable {
    public enum Verdict: Equatable, Sendable {
        /// Too few words to judge yet.
        case tooShort
        /// Mostly the assistant's own words.
        case echo
        case interruption
    }

    public var minimumWords = 2
    public var minimumCJKCharacters = 3
    /// Largest share of the heard phrase that may also occur in the assistant's recent speech
    /// before it is treated as echo.
    public var maximumEchoOverlap = 0.6

    public init() {}

    public func isInterruption(heard: String, assistantSpeech: String) -> Bool {
        evaluate(heard: heard, assistantSpeech: assistantSpeech) == .interruption
    }

    public func evaluate(heard: String, assistantSpeech: String) -> Verdict {
        let tokens = SpeechTokenizer.tokens(heard)
        guard !tokens.isEmpty else { return .tooShort }
        let cjkCount = tokens.filter(SpeechTokenizer.isCJKToken).count
        let wordCount = tokens.count - cjkCount
        let isCommand = Self.stopCommands.contains(tokens.joined(separator: " "))
            || Self.stopCommands.contains(tokens.joined())
        guard isCommand
            || wordCount >= minimumWords
            || cjkCount >= minimumCJKCharacters
            || (wordCount >= 1 && cjkCount >= 2)
        else { return .tooShort }

        let heardUnits = SpeechTokenizer.matchingUnits(heard)
        let spokenUnits = Set(SpeechTokenizer.matchingUnits(assistantSpeech))
        guard !heardUnits.isEmpty, !spokenUnits.isEmpty else { return .interruption }
        let echoed = heardUnits.filter { spokenUnits.contains($0) }.count
        return Double(echoed) / Double(heardUnits.count) < maximumEchoOverlap ? .interruption : .echo
    }

    private static let stopCommands: Set<String> = [
        "stop", "wait", "pause", "hold on", "hang on", "okay stop", "ok stop", "shh",
        "停", "停下", "等等", "等一下", "暂停", "别说了",
    ]
}

/// Watches one recognition request's growing transcript while the assistant talks. Words judged to
/// be echo are set aside, so a real interruption later in the reply is judged on its own words.
public struct InterruptionMonitor: Sendable {
    public var detector = BargeInDetector()
    private var echoTokenCount = 0
    private var echoText = ""

    public init() {}

    /// Call whenever a new recognition request starts.
    public mutating func reset() {
        echoTokenCount = 0
        echoText = ""
    }

    /// The user's words if `transcript` now contains an interruption, else nil.
    public mutating func interruption(in transcript: String, assistantSpeech: String) -> String? {
        let tokens = SpeechTokenizer.tokens(transcript)
        let fresh = tokens.dropFirst(min(echoTokenCount, tokens.count)).joined(separator: " ")
        switch detector.evaluate(heard: fresh, assistantSpeech: assistantSpeech) {
        case .interruption:
            return userWords(in: transcript)
        case .echo:
            echoTokenCount = tokens.count
            echoText = transcript
            return nil
        case .tooShort:
            return nil
        }
    }

    /// The transcript without the leading words that were judged to be echo.
    public func userWords(in transcript: String) -> String {
        guard !echoText.isEmpty, transcript.hasPrefix(echoText) else { return transcript }
        return String(transcript.dropFirst(echoText.count)).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
