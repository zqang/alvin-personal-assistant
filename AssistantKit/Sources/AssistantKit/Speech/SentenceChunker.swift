import Foundation

/// Splits streamed reply text into speakable chunks, so speech can start after the first
/// sentence instead of waiting for the whole reply.
public struct SentenceChunker: Sendable {
    /// The first chunk may end at a comma once it is this long, to start speaking sooner.
    public var firstChunkSoftLimit = 40
    /// Later chunks may end at a comma once they are this long.
    public var softLimit = 140
    /// A chunk is cut at its last space once it reaches this length, even without punctuation.
    public var hardLimit = 260

    private var buffer = ""
    private var emittedCount = 0

    public init() {}

    /// Adds streamed text and returns the chunks it completed.
    public mutating func append(_ text: String) -> [String] {
        buffer += text
        return drain(flushing: false)
    }

    /// Returns everything left, including an unfinished last sentence.
    public mutating func flush() -> [String] {
        drain(flushing: true)
    }

    public mutating func reset() {
        buffer = ""
        emittedCount = 0
    }

    private mutating func drain(flushing: Bool) -> [String] {
        var chunks: [String] = []
        while let end = boundary(), end > buffer.startIndex {
            let chunk = buffer[..<end].trimmingCharacters(in: .whitespacesAndNewlines)
            buffer = String(buffer[end...])
            if !chunk.isEmpty {
                chunks.append(chunk)
                emittedCount += 1
            }
        }
        if flushing {
            let rest = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
            buffer = ""
            if !rest.isEmpty {
                chunks.append(rest)
                emittedCount += 1
            }
        }
        return chunks
    }

    /// End index of the first complete chunk in the buffer, if there is one yet.
    private func boundary() -> String.Index? {
        let characters = Array(buffer)
        let commaLimit = emittedCount == 0 ? firstChunkSoftLimit : softLimit

        for (offset, character) in characters.enumerated() {
            if character.isNewline || Self.fullWidthStops.contains(character) {
                return index(atOffset: Self.skipClosers(characters, from: offset + 1))
            }
            if Self.sentenceStops.contains(character) {
                let end = Self.skipClosers(characters, from: offset + 1)
                // Whether the sentence ends here depends on the next character, which hasn't arrived.
                guard end < characters.count else { return nil }
                let next = characters[end]
                let endsSentence: Bool
                if character == "." {
                    endsSentence = next.isWhitespace && !Self.isAbbreviationOrNumber(characters, periodAt: offset)
                } else {
                    endsSentence = next.isWhitespace || SpeechTokenizer.isCJK(next)
                }
                if endsSentence { return index(atOffset: end) }
            } else if offset + 1 >= commaLimit, Self.isSoftStop(characters, at: offset) {
                return index(atOffset: offset + 1)
            }
        }

        if characters.count > hardLimit {
            if let space = characters[..<hardLimit].lastIndex(where: { $0.isWhitespace }), space > hardLimit / 2 {
                return index(atOffset: space + 1)
            }
            return index(atOffset: hardLimit)
        }
        return nil
    }

    private func index(atOffset offset: Int) -> String.Index {
        buffer.index(buffer.startIndex, offsetBy: min(offset, buffer.count))
    }

    private static let sentenceStops: Set<Character> = [".", "!", "?", "…"]
    private static let fullWidthStops: Set<Character> = ["。", "！", "？", "；"]
    private static let fullWidthSoftStops: Set<Character> = ["，", "、", "："]
    private static let asciiSoftStops: Set<Character> = [",", ";", ":"]
    private static let closers: Set<Character> = ["\"", "'", "”", "’", "»", "」", "』", ")", "]", "）", "】"]
    private static let abbreviations: Set<String> = [
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "e.g", "i.e", "a.m", "p.m",
        "u.s", "u.k", "approx", "fig", "inc", "ltd", "mt",
    ]

    private static func skipClosers(_ characters: [Character], from start: Int) -> Int {
        var end = start
        while end < characters.count, closers.contains(characters[end]) { end += 1 }
        return end
    }

    private static func isSoftStop(_ characters: [Character], at offset: Int) -> Bool {
        let character = characters[offset]
        if fullWidthSoftStops.contains(character) { return true }
        guard asciiSoftStops.contains(character), offset + 1 < characters.count else { return false }
        return characters[offset + 1].isWhitespace
    }

    /// True when the period after this word doesn't end a sentence: "Dr.", "e.g.", initials, list numbers.
    private static func isAbbreviationOrNumber(_ characters: [Character], periodAt offset: Int) -> Bool {
        var start = offset
        while start > 0, !characters[start - 1].isWhitespace { start -= 1 }
        let word = String(characters[start..<offset])
            .lowercased()
            .trimmingCharacters(in: CharacterSet(charactersIn: "\"'“‘(["))
        guard !word.isEmpty else { return false }
        if word.count == 1, word.first?.isLetter == true { return true }
        if abbreviations.contains(word) { return true }
        let startsLine = start == 0 || characters[start - 1].isNewline
        return startsLine && word.count <= 2 && word.allSatisfy(\.isNumber)
    }
}
