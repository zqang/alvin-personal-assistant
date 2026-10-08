import Foundation

/// Cross-session drafting: past replies and tool results, kept locally. When the current context
/// ends like something in the corpus, it proposes what usually followed there.
///
/// Matching is the longest context suffix (at least 2 tokens, at most `maxMatch`) that occurs in a
/// document with a following token. The continuation is then built token by token: the most
/// frequent next token among the matching occurrences (ties go to the most recent occurrence),
/// keeping only the occurrences that agree with it.
public struct SuffixCorpus: Sendable {
    /// Longest context suffix compared.
    public static let maxMatch = 8

    /// Most tokens kept; adding past it evicts the oldest documents.
    public let capacityTokens: Int
    private var matcher: ContinuationMatcher

    public init(capacityTokens: Int = 65_536) {
        self.capacityTokens = max(1, capacityTokens)
        matcher = ContinuationMatcher(keyLength: 2, maxMatch: Self.maxMatch)
    }

    /// Tokens currently held.
    public var tokenCount: Int { matcher.tokenCount }
    /// Documents currently held, oldest first.
    public var documents: [[Int]] { matcher.documents }

    /// Adds a document, then evicts the oldest documents until the corpus fits its capacity. A
    /// document longer than the capacity keeps only its last `capacityTokens` tokens. Documents
    /// too short to ever produce a proposal (under 3 tokens) are ignored.
    public mutating func add(document: [Int]) {
        let kept = document.count > capacityTokens ? Array(document.suffix(capacityTokens)) : document
        guard kept.count >= 3 else { return }
        matcher.add(kept)
        while matcher.tokenCount > capacityTokens { matcher.evictOldest() }
    }

    /// The continuation of the longest context match of at least 2 tokens, at most `maxTokens`
    /// long. Tokens in `excluded` are never drafted: they do not count as candidate next tokens,
    /// so the continuation ends where only excluded tokens or document ends follow.
    public func propose(context: ArraySlice<Int>, maxTokens: Int, excluded: Set<Int> = []) -> DraftProposal? {
        matcher.propose(context: context, maxTokens: maxTokens, minMatch: 2, anchors: [], excluded: excluded)
            .map { DraftProposal(tokens: $0.tokens, source: .corpus, matchLength: $0.matchLength) }
    }

    // MARK: - Persistence

    private static let magic: [UInt8] = Array("ASC1".utf8)
    private static let formatVersion: UInt64 = 1

    /// A compact binary form (varint-encoded) of the capacity and the documents.
    public func encoded() -> Data {
        var bytes = Self.magic
        CorpusVarint.append(Self.formatVersion, to: &bytes)
        CorpusVarint.append(UInt64(capacityTokens), to: &bytes)
        let documents = matcher.documents
        CorpusVarint.append(UInt64(documents.count), to: &bytes)
        for document in documents {
            CorpusVarint.append(UInt64(document.count), to: &bytes)
            for token in document { CorpusVarint.append(CorpusVarint.zigzag(token), to: &bytes) }
        }
        return Data(bytes)
    }

    /// Restores a corpus written by `encoded()`. Nil for data in any other form.
    public init?(data: Data) {
        let bytes = [UInt8](data)
        guard bytes.count >= Self.magic.count, Array(bytes[0..<Self.magic.count]) == Self.magic else { return nil }
        var reader = CorpusVarint.Reader(bytes: bytes, offset: Self.magic.count)
        guard reader.next() == Self.formatVersion,
              let capacity = reader.next(), capacity >= 1, capacity <= UInt64(Int.max),
              let documentCount = reader.next(), documentCount <= UInt64(bytes.count)
        else { return nil }
        self.init(capacityTokens: Int(capacity))
        for _ in 0..<Int(documentCount) {
            // Each token takes at least one byte, which bounds the length before allocating.
            guard let length = reader.next(), length <= UInt64(reader.remaining) else { return nil }
            var document: [Int] = []
            document.reserveCapacity(Int(length))
            for _ in 0..<Int(length) {
                guard let value = reader.next() else { return nil }
                document.append(CorpusVarint.unzigzag(value))
            }
            add(document: document)
        }
        guard reader.remaining == 0 else { return nil }
    }
}

/// Unsigned LEB128 varints with zigzag for signed values.
enum CorpusVarint {
    static func append(_ value: UInt64, to bytes: inout [UInt8]) {
        var value = value
        while value >= 0x80 {
            bytes.append(UInt8(truncatingIfNeeded: value) | 0x80)
            value >>= 7
        }
        bytes.append(UInt8(value))
    }

    static func zigzag(_ value: Int) -> UInt64 {
        let v = Int64(value)
        return UInt64(bitPattern: (v << 1) ^ (v >> 63))
    }

    static func unzigzag(_ value: UInt64) -> Int {
        Int(Int64(bitPattern: value >> 1) ^ -Int64(bitPattern: value & 1))
    }

    struct Reader {
        let bytes: [UInt8]
        var offset: Int

        var remaining: Int { bytes.count - offset }

        /// The next varint, or nil if the bytes end inside it or it overflows 64 bits.
        mutating func next() -> UInt64? {
            var result: UInt64 = 0
            var shift: UInt64 = 0
            while offset < bytes.count {
                let byte = bytes[offset]
                offset += 1
                let payload = UInt64(byte & 0x7F)
                if shift == 63, payload > 1 { return nil }
                result |= payload << shift
                if byte & 0x80 == 0 { return result }
                shift += 7
                if shift > 63 { return nil }
            }
            return nil
        }
    }
}

/// The matcher behind `SuffixCorpus` and `ExemplarSeeds`: documents indexed by the `keyLength`
/// tokens before every position that has a continuation. A lookup extends each hit backwards to
/// measure its match length, which also rules out hash collisions.
struct ContinuationMatcher: Sendable {
    struct Match: Equatable, Sendable {
        var tokens: [Int]
        var matchLength: Int
    }

    private struct Entry: Sendable {
        /// Serial number of the document.
        var serial: Int
        /// Position in the document right after the key tokens; always has a token.
        var position: Int
    }

    private struct Candidate {
        var serial: Int
        var position: Int
    }

    let keyLength: Int
    let maxMatch: Int
    /// Live documents, oldest first; `documents[i]` has serial `firstSerial + i`.
    private(set) var documents: [[Int]] = []
    private(set) var tokenCount = 0
    private var firstSerial = 0
    private var index: [UInt64: [Entry]] = [:]
    private var liveEntries = 0
    private var staleEntries = 0

    init(keyLength: Int, maxMatch: Int) {
        self.keyLength = max(1, keyLength)
        self.maxMatch = max(self.keyLength, maxMatch)
    }

    mutating func add(_ document: [Int]) {
        guard !document.isEmpty else { return }
        let serial = firstSerial + documents.count
        documents.append(document)
        tokenCount += document.count
        liveEntries += indexDocument(document, serial: serial)
    }

    mutating func evictOldest() {
        guard let oldest = documents.first else { return }
        documents.removeFirst()
        firstSerial += 1
        tokenCount -= oldest.count
        let entries = max(0, oldest.count - keyLength)
        liveEntries -= entries
        staleEntries += entries
        // Entries of evicted documents are skipped by lookups; drop them once they dominate.
        if staleEntries > max(1_024, liveEntries) { rebuildIndex() }
    }

    /// The continuation of the longest match of at least `minMatch` tokens; a single-token match
    /// is also allowed when that token is in `anchors`.
    func propose(context: ArraySlice<Int>, maxTokens: Int, minMatch: Int, anchors: Set<Int>, excluded: Set<Int>) -> Match? {
        guard maxTokens > 0, context.count >= keyLength, let last = context.last else { return nil }
        let required = anchors.contains(last) ? 1 : max(minMatch, 1)
        var hash = DraftTokenHash.seed
        var i = context.endIndex
        for _ in 0..<keyLength {
            i -= 1
            hash = DraftTokenHash.mix(hash, context[i])
        }
        guard let bucket = index[hash] else { return nil }

        var best = 0
        var candidates: [Candidate] = []
        for entry in bucket where entry.serial >= firstSerial {
            let document = documents[entry.serial - firstSerial]
            let next = document[entry.position]
            if excluded.contains(next) { continue }
            var length = 0
            while length < maxMatch, length < context.count, entry.position - 1 - length >= 0,
                  document[entry.position - 1 - length] == context[context.endIndex - 1 - length]
            {
                length += 1
            }
            guard length >= keyLength, length >= required, length >= best else { continue }
            if length > best {
                best = length
                candidates.removeAll(keepingCapacity: true)
            }
            candidates.append(Candidate(serial: entry.serial, position: entry.position))
        }
        guard best > 0 else { return nil }

        var tokens: [Int] = []
        while tokens.count < maxTokens {
            // token -> (count, most recent occurrence)
            var tally: [Int: (count: Int, serial: Int, position: Int)] = [:]
            for candidate in candidates {
                let document = documents[candidate.serial - firstSerial]
                guard candidate.position < document.count else { continue }
                let token = document[candidate.position]
                if excluded.contains(token) { continue }
                if var current = tally[token] {
                    current.count += 1
                    if (candidate.serial, candidate.position) > (current.serial, current.position) {
                        current.serial = candidate.serial
                        current.position = candidate.position
                    }
                    tally[token] = current
                } else {
                    tally[token] = (1, candidate.serial, candidate.position)
                }
            }
            guard let winner = tally.max(by: { a, b in
                (a.value.count, a.value.serial, a.value.position) < (b.value.count, b.value.serial, b.value.position)
            }) else { break }
            tokens.append(winner.key)
            candidates = candidates.compactMap { candidate in
                let document = documents[candidate.serial - firstSerial]
                guard candidate.position < document.count, document[candidate.position] == winner.key else { return nil }
                return Candidate(serial: candidate.serial, position: candidate.position + 1)
            }
        }
        return tokens.isEmpty ? nil : Match(tokens: tokens, matchLength: best)
    }

    /// Indexes every position of `document` that follows `keyLength` tokens and has a token.
    private mutating func indexDocument(_ document: [Int], serial: Int) -> Int {
        guard document.count > keyLength else { return 0 }
        for position in keyLength..<document.count {
            var hash = DraftTokenHash.seed
            for i in stride(from: position - 1, through: position - keyLength, by: -1) {
                hash = DraftTokenHash.mix(hash, document[i])
            }
            index[hash, default: []].append(Entry(serial: serial, position: position))
        }
        return document.count - keyLength
    }

    private mutating func rebuildIndex() {
        index.removeAll(keepingCapacity: true)
        liveEntries = 0
        staleEntries = 0
        for (offset, document) in documents.enumerated() {
            liveEntries += indexDocument(document, serial: firstSerial + offset)
        }
    }
}
