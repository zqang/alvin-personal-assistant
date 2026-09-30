import Foundation

/// Chooses the text of a finished voice turn: a second, more accurate on-device pass over the
/// turn's audio wins over the live caption, unless it looks broken.
public enum FinalTranscript {
    /// The language name Qwen3-ASR expects for a speech locale, or nil when it shouldn't be used.
    public static func qwenLanguage(forLocale identifier: String) -> String? {
        if identifier.hasPrefix("en") { return "English" }
        if identifier == "zh-CN" { return "Chinese" }
        return nil
    }

    /// `live` is the whole turn; `refined` covers only the part after `carried`, earlier words the
    /// second pass didn't hear.
    public static func pick(live: String, carried: String = "", refined: String?) -> String {
        guard !carried.isEmpty else { return pickSegment(live: live, refined: refined) }
        guard live.hasPrefix(carried) else { return live }
        let segment = pickSegment(live: String(live.dropFirst(carried.count)), refined: refined)
        return segment.isEmpty ? carried : carried + " " + segment
    }

    /// Where the second pass should start reading the request's audio: `leadIn` before the first
    /// recognized words (they arrive late), less if that would exceed `maxLength`, but never less
    /// than `minLeadIn`, the least the words may lag the speech. Longer speech gets a clip longer
    /// than `maxLength`, which is rejected rather than cut.
    public static func clipStart(speechStart: Int, end: Int, maxLength: Int, leadIn: Int, minLeadIn: Int) -> Int {
        min(max(speechStart - minLeadIn, 0), max(speechStart - leadIn, end - maxLength, 0))
    }

    /// The part of a 16 kHz clip from its first to its last voiced 30 ms frame, plus `margin`
    /// samples on each side, or nil if nothing sounds like voice. ASR models invent words in silence.
    public static func voicedRange(of samples: [Float], margin: Int = 4_800) -> Range<Int>? {
        let frame = 480
        var vad = EnergyVAD()
        var first: Int?
        var last = 0
        for start in stride(from: 0, to: samples.count, by: frame) {
            let end = min(start + frame, samples.count)
            var energy: Float = 0
            for index in start..<end { energy += samples[index] * samples[index] }
            let level = 10 * log10(max(energy / Float(end - start), 1e-10))
            if vad.isVoice(level: level) {
                if first == nil { first = start }
                last = end
            }
        }
        guard let first else { return nil }
        return max(first - margin, 0)..<min(last + margin, samples.count)
    }

    private static func pickSegment(live: String, refined: String?) -> String {
        let live = live.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let refined = refined?.trimmingCharacters(in: .whitespacesAndNewlines), !refined.isEmpty else { return live }
        let heard = SpeechTokenizer.tokens(live).count
        let second = SpeechTokenizer.tokens(refined).count
        // ponytail: length-ratio sanity check only; aligning the two transcripts would catch subtler hallucinations.
        return second <= 2 * heard + 4 && heard <= 2 * second + 4 ? refined : live
    }
}

/// The latest audio of one recognition request, for the second pass. Positions count every sample
/// appended since `reset()`, including ones discarded to make room. Not thread-safe.
public struct SampleWindow: Sendable {
    public let capacity: Int
    private var samples: [Float] = []
    private var dropped = 0

    public init(capacity: Int) {
        self.capacity = capacity
        samples.reserveCapacity(capacity)
    }

    /// Samples appended since `reset()`.
    public var count: Int { dropped + samples.count }

    public mutating func reset() {
        samples.removeAll(keepingCapacity: true)
        dropped = 0
    }

    /// Appends, discarding the older half when full. The capacity is reserved and `samples(from:)`
    /// never shares the storage, so this doesn't allocate.
    public mutating func append(_ newSamples: UnsafeBufferPointer<Float>) {
        if samples.count + newSamples.count > capacity {
            let half = samples.count / 2
            samples.removeFirst(half)
            dropped += half
        }
        samples.append(contentsOf: newSamples)
    }

    /// A copy of the samples from position `start` on, or nothing if they were discarded.
    public func samples(from start: Int) -> [Float] {
        guard start >= dropped, start < count else { return [] }
        return samples[(start - dropped)...].withUnsafeBufferPointer { Array($0) }
    }
}
