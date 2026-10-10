import XCTest
@testable import AssistantKit

/// A seeded SplitMix64 generator, so the randomized drafting tests are reproducible.
struct DraftingTestRNG: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// A uniform double in [0, 1).
    mutating func unit() -> Double {
        Double(next() >> 11) * 0x1p-53
    }

    /// An index drawn from the categorical distribution `probabilities` (which sum to 1).
    mutating func categorical(_ probabilities: [Double]) -> Int {
        let u = unit()
        var cumulative = 0.0
        for (index, p) in probabilities.enumerated() {
            cumulative += p
            if u < cumulative { return index }
        }
        return probabilities.count - 1
    }

    mutating func tokens(count: Int, alphabet: Int) -> [Int] {
        (0..<count).map { _ in Int.random(in: 0..<alphabet, using: &self) }
    }
}

/// Brute-force references for the drafting indexes, written as directly as possible from their
/// definitions (and from scripts/drafter_lab.py, which mirrors them).
enum DraftingReference {
    /// The longest suffix n-gram (maxN down to minN) with an earlier occurrence; the most recent
    /// occurrence wins; the continuation stops at an excluded token or the end.
    static func promptLookup(_ ledger: [Int], minN: Int, maxN: Int, maxTokens: Int, excluded: Set<Int>) -> DraftProposal? {
        guard maxTokens > 0 else { return nil }
        let m = ledger.count
        for n in stride(from: maxN, through: minN, by: -1) where m >= n + 1 {
            let suffix = ledger[(m - n)..<m]
            for start in stride(from: m - n - 1, through: 0, by: -1) where ledger[start..<(start + n)].elementsEqual(suffix) {
                var tokens: [Int] = []
                var j = start + n
                while tokens.count < maxTokens, j < m, !excluded.contains(ledger[j]) {
                    tokens.append(ledger[j])
                    j += 1
                }
                return tokens.isEmpty ? nil : DraftProposal(tokens: tokens, source: .promptLookup, matchLength: n)
            }
        }
        return nil
    }

    /// The lab's `SequenceMatcher.propose`: for n from min(maxMatch, context length) down to
    /// minMatch (1 only for an anchor), every occurrence of the context's last n tokens; the
    /// continuation repeatedly takes the most frequent next token (ties to the most recent
    /// occurrence); the first n with a non-empty continuation wins.
    static func sequenceMatch(
        documents: [[Int]], context: [Int], maxTokens: Int, minMatch: Int, maxMatch: Int,
        anchors: Set<Int>, excluded: Set<Int>
    ) -> (tokens: [Int], matchLength: Int)? {
        guard maxTokens > 0, !context.isEmpty else { return nil }
        let lowest = anchors.contains(context[context.count - 1]) ? 1 : minMatch
        for n in stride(from: min(maxMatch, context.count), through: max(lowest, 1), by: -1) {
            let suffix = context[(context.count - n)...]
            var candidates: [(doc: Int, position: Int)] = []
            for (d, document) in documents.enumerated() where document.count >= n {
                for start in 0...(document.count - n) where document[start..<(start + n)].elementsEqual(suffix) {
                    candidates.append((d, start + n))
                }
            }
            var tokens: [Int] = []
            while tokens.count < maxTokens {
                var tally: [Int: (count: Int, doc: Int, position: Int)] = [:]
                for (d, p) in candidates where p < documents[d].count && !excluded.contains(documents[d][p]) {
                    let token = documents[d][p]
                    let current = tally[token] ?? (0, -1, -1)
                    let recent = (d, p) > (current.doc, current.position) ? (d, p) : (current.doc, current.position)
                    tally[token] = (current.count + 1, recent.0, recent.1)
                }
                guard let winner = tally.max(by: { ($0.value.count, $0.value.doc, $0.value.position) < ($1.value.count, $1.value.doc, $1.value.position) }) else { break }
                tokens.append(winner.key)
                candidates = candidates
                    .filter { $0.position < documents[$0.doc].count && documents[$0.doc][$0.position] == winner.key }
                    .map { ($0.doc, $0.position + 1) }
            }
            if !tokens.isEmpty { return (tokens, n) }
        }
        return nil
    }
}

final class NGramIndexTests: XCTestCase {
    func testMatchesBruteForceIncrementallyAndInBatch() {
        var rng = DraftingTestRNG(seed: 0xD1A7_2026)
        var compared = 0
        var proposals = 0
        for _ in 0..<1_000 {
            let alphabet = Int.random(in: 2...6, using: &rng)
            let length = Int.random(in: 0...40, using: &rng)
            let sequence = rng.tokens(count: length, alphabet: alphabet)
            let minN = Bool.random(using: &rng) ? 2 : Int.random(in: 1...3, using: &rng)
            let maxN = Bool.random(using: &rng) ? max(minN, 4) : Int.random(in: minN...(minN + 3), using: &rng)
            let excluded = Set(rng.tokens(count: Int.random(in: 0...2, using: &rng), alphabet: alphabet))
            let maxTokens = Int.random(in: 1...6, using: &rng)

            var index = NGramIndex(minN: minN, maxN: maxN, excluded: excluded)
            XCTAssertNil(index.propose(maxTokens: maxTokens))
            for (offset, token) in sequence.enumerated() {
                index.append(contentsOf: [token])
                let prefix = Array(sequence[...offset])
                let expected = DraftingReference.promptLookup(prefix, minN: minN, maxN: maxN, maxTokens: maxTokens, excluded: excluded)
                XCTAssertEqual(index.propose(maxTokens: maxTokens), expected, "prefix \(prefix) n \(minN)...\(maxN) excluded \(excluded)")
                XCTAssertEqual(index.count, prefix.count)
                compared += 1
                if expected != nil { proposals += 1 }
            }
            // Batch indexing of a random prefix proposes the same as the incremental index did.
            let cut = Int.random(in: 0...length, using: &rng)
            var batch = NGramIndex(minN: minN, maxN: maxN, excluded: excluded)
            batch.append(contentsOf: sequence[..<cut])
            XCTAssertEqual(
                batch.propose(maxTokens: maxTokens),
                DraftingReference.promptLookup(Array(sequence[..<cut]), minN: minN, maxN: maxN, maxTokens: maxTokens, excluded: excluded)
            )
            XCTAssertEqual(batch.tokens, Array(sequence[..<cut]))
        }
        // The random cases must exercise real proposals, not just nil.
        XCTAssertGreaterThan(proposals, compared / 4)
    }

    func testTruncateRestoresEarlierStateExactly() {
        var rng = DraftingTestRNG(seed: 42)
        for _ in 0..<300 {
            let alphabet = Int.random(in: 2...5, using: &rng)
            let excluded: Set<Int> = Bool.random(using: &rng) ? [0] : []
            var sequence = rng.tokens(count: Int.random(in: 0...40, using: &rng), alphabet: alphabet)
            var index = NGramIndex(excluded: excluded)
            index.append(contentsOf: sequence)
            for _ in 0..<4 {
                let keep = Int.random(in: 0...(sequence.count + 2), using: &rng)
                index.truncate(to: keep)
                sequence = Array(sequence.prefix(keep))
                XCTAssertEqual(index.tokens, sequence)
                XCTAssertEqual(index.propose(maxTokens: 4), DraftingReference.promptLookup(sequence, minN: 2, maxN: 4, maxTokens: 4, excluded: excluded))
                for token in rng.tokens(count: Int.random(in: 0...10, using: &rng), alphabet: alphabet) {
                    index.append(contentsOf: [token])
                    sequence.append(token)
                    XCTAssertEqual(index.propose(maxTokens: 4), DraftingReference.promptLookup(sequence, minN: 2, maxN: 4, maxTokens: 4, excluded: excluded))
                }
            }
            var fresh = NGramIndex(excluded: excluded)
            fresh.append(contentsOf: sequence)
            XCTAssertEqual(index.propose(maxTokens: 6), fresh.propose(maxTokens: 6))
        }
    }

    func testMostRecentOccurrenceWins() {
        var index = NGramIndex(excluded: [])
        index.append(contentsOf: [1, 2, 3, 1, 2, 4, 1, 2])
        XCTAssertEqual(index.propose(maxTokens: 2), DraftProposal(tokens: [4, 1], source: .promptLookup, matchLength: 2))
        // A longer match beats a more recent shorter one.
        var longer = NGramIndex(excluded: [])
        longer.append(contentsOf: [7, 1, 2, 5, 9, 1, 2, 6, 7, 1, 2])
        XCTAssertEqual(longer.propose(maxTokens: 1), DraftProposal(tokens: [5], source: .promptLookup, matchLength: 3))
    }

    func testContinuationStopsAtExcludedTokensAndTheEnd() {
        var index = NGramIndex(excluded: [9])
        index.append(contentsOf: [1, 2, 3, 9, 4, 1, 2])
        XCTAssertEqual(index.propose(maxTokens: 4), DraftProposal(tokens: [3], source: .promptLookup, matchLength: 2))

        // The winning occurrence is followed by an excluded token: no proposal, even though an
        // older occurrence has a usable continuation.
        var blocked = NGramIndex(excluded: [9])
        blocked.append(contentsOf: [1, 2, 3, 1, 2, 9, 1, 2])
        XCTAssertNil(blocked.propose(maxTokens: 4))

        // The continuation may run into the suffix itself, up to the current end.
        var overlapping = NGramIndex(excluded: [])
        overlapping.append(contentsOf: [5, 5, 5])
        XCTAssertEqual(overlapping.propose(maxTokens: 4), DraftProposal(tokens: [5], source: .promptLookup, matchLength: 2))
    }

    func testEdgeCases() {
        var index = NGramIndex(excluded: [])
        XCTAssertEqual(index.count, 0)
        XCTAssertNil(index.propose(maxTokens: 4))
        index.append(contentsOf: [1, 2, 1, 2])
        XCTAssertNil(index.propose(maxTokens: 0))
        XCTAssertEqual(index.propose(maxTokens: 8), DraftProposal(tokens: [1, 2], source: .promptLookup, matchLength: 2))
        index.truncate(to: 10)
        XCTAssertEqual(index.count, 4)
        index.truncate(to: -1)
        XCTAssertEqual(index.count, 0)
        XCTAssertNil(index.propose(maxTokens: 4))
        // minN below 1 and maxN below minN are clamped.
        let clamped = NGramIndex(minN: 0, maxN: -3, excluded: [])
        XCTAssertEqual(clamped.minN, 1)
        XCTAssertEqual(clamped.maxN, 1)
    }

    func testSyncFollowsTheLedger() {
        var index = NGramIndex(excluded: [])
        index.sync(to: [1, 2, 3, 4, 1, 2])
        XCTAssertEqual(index.propose(maxTokens: 2), DraftProposal(tokens: [3, 4], source: .promptLookup, matchLength: 2))
        // A rewind and a different continuation.
        index.sync(to: [1, 2, 3, 4, 7, 7, 1, 2])
        XCTAssertEqual(index.tokens, [1, 2, 3, 4, 7, 7, 1, 2])
        var fresh = NGramIndex(excluded: [])
        fresh.append(contentsOf: [1, 2, 3, 4, 7, 7, 1, 2])
        XCTAssertEqual(index.propose(maxTokens: 3), fresh.propose(maxTokens: 3))
        index.sync(to: [])
        XCTAssertEqual(index.count, 0)
    }
}

final class SuffixCorpusTests: XCTestCase {
    func testMostFrequentContinuationWithRecencyTieBreak() throws {
        var corpus = SuffixCorpus()
        corpus.add(document: [1, 2, 3, 9])
        corpus.add(document: [5, 1, 2, 3, 8])
        corpus.add(document: [1, 2, 4, 7])
        // 3 follows "1 2" twice, 4 once; after "1 2 3", 9 and 8 tie and the newer document wins.
        XCTAssertEqual(corpus.propose(context: [0, 1, 2], maxTokens: 4), DraftProposal(tokens: [3, 8], source: .corpus, matchLength: 2))

        var tie = SuffixCorpus()
        tie.add(document: [1, 2, 3])
        tie.add(document: [1, 2, 4])
        XCTAssertEqual(tie.propose(context: [1, 2], maxTokens: 4)?.tokens, [4])
    }

    func testLongestContextMatchWins() {
        var corpus = SuffixCorpus()
        corpus.add(document: [7, 1, 2, 3])
        corpus.add(document: [8, 1, 2, 4])
        corpus.add(document: [6, 1, 2, 5])
        XCTAssertEqual(corpus.propose(context: [8, 1, 2], maxTokens: 4), DraftProposal(tokens: [4], source: .corpus, matchLength: 3))
        // Match lengths are capped at SuffixCorpus.maxMatch.
        var long = SuffixCorpus()
        long.add(document: Array(1...20))
        XCTAssertEqual(long.propose(context: ArraySlice(Array(1...12)), maxTokens: 3), DraftProposal(tokens: [13, 14, 15], source: .corpus, matchLength: 8))
    }

    func testNeedsTwoMatchingTokensAndRespectsLimits() {
        var corpus = SuffixCorpus()
        corpus.add(document: [1, 2, 3, 4])
        XCTAssertNil(corpus.propose(context: [2], maxTokens: 4))
        XCTAssertNil(corpus.propose(context: [9, 2], maxTokens: 4))
        XCTAssertNil(corpus.propose(context: [1, 2], maxTokens: 0))
        XCTAssertNil(corpus.propose(context: [3, 4], maxTokens: 4), "no token follows the end of a document")
        XCTAssertEqual(corpus.propose(context: [1, 2], maxTokens: 1)?.tokens, [3])
        XCTAssertEqual(corpus.propose(context: [1, 2], maxTokens: 4, excluded: [4])?.tokens, [3])
        XCTAssertNil(corpus.propose(context: [1, 2], maxTokens: 4, excluded: [3]))
        // Matches never span documents.
        corpus.add(document: [5, 6, 7])
        XCTAssertNil(corpus.propose(context: [4, 5], maxTokens: 4))
        // Documents too short to propose anything are ignored.
        corpus.add(document: [8, 9])
        XCTAssertEqual(corpus.documents, [[1, 2, 3, 4], [5, 6, 7]])
    }

    func testEvictsOldestDocumentsPastCapacity() {
        var corpus = SuffixCorpus(capacityTokens: 10)
        corpus.add(document: [1, 2, 3, 4])
        corpus.add(document: [5, 6, 7, 8])
        XCTAssertEqual(corpus.tokenCount, 8)
        corpus.add(document: [9, 10, 11, 12])
        XCTAssertEqual(corpus.documents, [[5, 6, 7, 8], [9, 10, 11, 12]])
        XCTAssertEqual(corpus.tokenCount, 8)
        XCTAssertNil(corpus.propose(context: [1, 2], maxTokens: 4))
        XCTAssertEqual(corpus.propose(context: [5, 6], maxTokens: 4)?.tokens, [7, 8])
        // A document longer than the capacity keeps its tail and evicts everything else.
        corpus.add(document: Array(100..<115))
        XCTAssertEqual(corpus.documents, [Array(105..<115)])
        XCTAssertEqual(corpus.tokenCount, 10)
    }

    func testManyEvictionsMatchAFreshCorpus() {
        var rng = DraftingTestRNG(seed: 7)
        var corpus = SuffixCorpus(capacityTokens: 300)
        for _ in 0..<400 {
            corpus.add(document: rng.tokens(count: Int.random(in: 3...20, using: &rng), alphabet: 6))
        }
        XCTAssertLessThanOrEqual(corpus.tokenCount, 300)
        var fresh = SuffixCorpus(capacityTokens: 300)
        for document in corpus.documents { fresh.add(document: document) }
        XCTAssertEqual(fresh.documents, corpus.documents)
        for _ in 0..<300 {
            let context = rng.tokens(count: Int.random(in: 1...10, using: &rng), alphabet: 6)
            XCTAssertEqual(corpus.propose(context: ArraySlice(context), maxTokens: 5), fresh.propose(context: ArraySlice(context), maxTokens: 5))
        }
    }

    func testMatchesTheReferenceMatcher() {
        var rng = DraftingTestRNG(seed: 99)
        var proposals = 0
        for _ in 0..<400 {
            var corpus = SuffixCorpus()
            let alphabet = Int.random(in: 2...5, using: &rng)
            for _ in 0..<Int.random(in: 0...6, using: &rng) {
                corpus.add(document: rng.tokens(count: Int.random(in: 0...15, using: &rng), alphabet: alphabet))
            }
            let excluded: Set<Int> = Bool.random(using: &rng) ? [] : [Int.random(in: 0..<alphabet, using: &rng)]
            for _ in 0..<10 {
                let context = rng.tokens(count: Int.random(in: 0...12, using: &rng), alphabet: alphabet)
                let maxTokens = Int.random(in: 1...6, using: &rng)
                let expected = DraftingReference.sequenceMatch(
                    documents: corpus.documents, context: context, maxTokens: maxTokens, minMatch: 2,
                    maxMatch: SuffixCorpus.maxMatch, anchors: [], excluded: excluded
                ).map { DraftProposal(tokens: $0.tokens, source: .corpus, matchLength: $0.matchLength) }
                XCTAssertEqual(corpus.propose(context: ArraySlice(context), maxTokens: maxTokens, excluded: excluded), expected)
                if expected != nil { proposals += 1 }
            }
        }
        XCTAssertGreaterThan(proposals, 500)
    }

    func testEncodingRoundTrips() throws {
        var corpus = SuffixCorpus(capacityTokens: 1_000)
        corpus.add(document: [151_644, 77_091, 198, 151_645, 0, -3])
        corpus.add(document: [1, 2, 3, Int(Int32.max), Int.max, Int.min])
        corpus.add(document: [4, 5, 6, 7])
        let data = corpus.encoded()
        let decoded = try XCTUnwrap(SuffixCorpus(data: data))
        XCTAssertEqual(decoded.capacityTokens, 1_000)
        XCTAssertEqual(decoded.documents, corpus.documents)
        XCTAssertEqual(decoded.tokenCount, corpus.tokenCount)
        XCTAssertEqual(decoded.propose(context: [4, 5], maxTokens: 4), corpus.propose(context: [4, 5], maxTokens: 4))
        XCTAssertEqual(decoded.encoded(), data)

        let empty = try XCTUnwrap(SuffixCorpus(data: SuffixCorpus().encoded()))
        XCTAssertEqual(empty.capacityTokens, 65_536)
        XCTAssertEqual(empty.tokenCount, 0)
    }

    func testRejectsMalformedData() {
        let data = {
            var corpus = SuffixCorpus(capacityTokens: 50)
            corpus.add(document: [1, 2, 3, 300, 70_000])
            return corpus.encoded()
        }()
        XCTAssertNil(SuffixCorpus(data: Data()))
        XCTAssertNil(SuffixCorpus(data: Data("XXXX".utf8) + data.dropFirst(4)))
        for length in 0..<data.count {
            XCTAssertNil(SuffixCorpus(data: data.prefix(length)), "truncated to \(length) bytes")
        }
        XCTAssertNil(SuffixCorpus(data: data + Data([0])), "trailing bytes")
        // A document count or length larger than the data is rejected before allocating.
        XCTAssertNil(SuffixCorpus(data: Data("ASC1".utf8) + Data([1, 10, 0xFF, 0xFF, 0xFF, 0x7F])))
        XCTAssertNil(SuffixCorpus(data: Data("ASC1".utf8) + Data([1, 10, 1, 0xFF, 0xFF, 0x7F, 1, 2])))
        // Capacity zero, a future version, and a varint past 64 bits.
        XCTAssertNil(SuffixCorpus(data: Data("ASC1".utf8) + Data([1, 0, 0])))
        XCTAssertNil(SuffixCorpus(data: Data("ASC1".utf8) + Data([2, 10, 0])))
        XCTAssertNil(SuffixCorpus(data: Data("ASC1".utf8) + Data(repeating: 0xFF, count: 10) + Data([1])))
    }
}

final class ExemplarSeedsTests: XCTestCase {
    // <tool_call>=100, "\n"=1, "<function="=2, names 3 and 4, ">\n"=5, "<parameter="=6, params 7 and 8.
    private let weather = [100, 1, 2, 3, 5, 6, 7, 5]
    private let timer = [100, 1, 2, 4, 5, 6, 8, 5]

    func testAnchorTokenStartsASkeleton() {
        let seeds = ExemplarSeeds(sequences: [weather, timer])
        XCTAssertEqual(seeds.anchors, [100])
        // After <tool_call> both skeletons agree on "\n<function=", then the newer one wins the tie.
        XCTAssertEqual(seeds.propose(context: [42, 100], maxTokens: 8), DraftProposal(tokens: [1, 2, 4, 5, 6, 8, 5], source: .exemplar, matchLength: 1))
        XCTAssertEqual(seeds.propose(context: [100], maxTokens: 2)?.tokens, [1, 2])
        // Once the name is out, the matching skeleton continues.
        XCTAssertEqual(seeds.propose(context: [100, 1, 2, 3], maxTokens: 3), DraftProposal(tokens: [5, 6, 7], source: .exemplar, matchLength: 4))
    }

    func testOtherTokensNeedTwoMatchingTokens() {
        let seeds = ExemplarSeeds(sequences: [weather, timer])
        XCTAssertNil(seeds.propose(context: [9, 2], maxTokens: 4), "a lone non-anchor token does not match")
        XCTAssertEqual(seeds.propose(context: [9, 5, 6], maxTokens: 2), DraftProposal(tokens: [8, 5], source: .exemplar, matchLength: 2))
        XCTAssertNil(seeds.propose(context: [9, 5], maxTokens: 0))
        XCTAssertNil(seeds.propose(context: [], maxTokens: 4))
        XCTAssertNil(ExemplarSeeds(sequences: []).propose(context: [100], maxTokens: 4))

        let strict = ExemplarSeeds(sequences: [weather], anchors: [])
        XCTAssertNil(strict.propose(context: [100], maxTokens: 4))
        XCTAssertEqual(strict.propose(context: [100, 1], maxTokens: 2)?.tokens, [2, 3])
    }

    func testMatchesTheReferenceMatcher() {
        var rng = DraftingTestRNG(seed: 2_026)
        for _ in 0..<300 {
            let alphabet = Int.random(in: 2...6, using: &rng)
            let sequences = (0..<Int.random(in: 0...5, using: &rng)).map { _ in
                rng.tokens(count: Int.random(in: 0...10, using: &rng), alphabet: alphabet)
            }
            let anchors: Set<Int> = Bool.random(using: &rng) ? Set(sequences.compactMap(\.first)) : [0]
            let seeds = ExemplarSeeds(sequences: sequences, anchors: anchors)
            for _ in 0..<10 {
                let context = rng.tokens(count: Int.random(in: 0...10, using: &rng), alphabet: alphabet)
                let maxTokens = Int.random(in: 1...6, using: &rng)
                let expected = DraftingReference.sequenceMatch(
                    documents: sequences.filter { $0.count >= 2 }, context: context, maxTokens: maxTokens, minMatch: 2,
                    maxMatch: SuffixCorpus.maxMatch, anchors: anchors, excluded: []
                ).map { DraftProposal(tokens: $0.tokens, source: .exemplar, matchLength: $0.matchLength) }
                XCTAssertEqual(seeds.propose(context: ArraySlice(context), maxTokens: maxTokens), expected)
            }
        }
    }
}

final class CostCurveTests: XCTestCase {
    func testStockCurveInterpolatesAndExtrapolates() {
        let curve = CostCurve.stockMLXDefault
        XCTAssertEqual(curve.relative(1), 1)
        XCTAssertEqual(curve.relative(0), 1)
        XCTAssertEqual(curve.relative(4), 1.63, accuracy: 1e-12)
        XCTAssertEqual(curve.relative(7), 2.685, accuracy: 1e-12)
        XCTAssertEqual(curve.relative(9), 3.4, accuracy: 1e-12)
        XCTAssertEqual(curve.relative(10), 3.73, accuracy: 1e-12)
        XCTAssertEqual(curve.relative(12), 4.39, accuracy: 1e-12)
    }

    func testMeasuredSecondsAreRelativeToOneRow() {
        let curve = CostCurve(seconds: [1: 0.020, 2: 0.021, 4: 0.030])
        XCTAssertEqual(curve.relative(1), 1)
        XCTAssertEqual(curve.relative(2), 1.05, accuracy: 1e-12)
        XCTAssertEqual(curve.relative(3), 1.275, accuracy: 1e-12)
        XCTAssertEqual(curve.relative(6), 1.95, accuracy: 1e-12)

        // Without a 1-row point the first point is the base.
        let noOne = CostCurve(seconds: [2: 2.0, 4: 3.0])
        XCTAssertEqual(noOne.relative(1), 1)
        XCTAssertEqual(noOne.relative(3), 1.25, accuracy: 1e-12)
    }

    func testDegenerateCurvesStayConservative() {
        XCTAssertEqual(CostCurve(seconds: [:]).relative(4), 4)
        XCTAssertEqual(CostCurve(seconds: [1: 0.02]).relative(5), 5)
        XCTAssertEqual(CostCurve(seconds: [1: 0.02, 2: .nan, 3: -1, 0: 0.001]).relative(3), 3)
        // Noise never makes more rows cheaper than one.
        XCTAssertEqual(CostCurve(seconds: [1: 1.0, 2: 0.9, 3: 0.8]).relative(2), 1)
        XCTAssertEqual(CostCurve(seconds: [1: 1.0, 2: 0.9, 3: 0.8]).relative(9), 1)
    }

    func testFromSamplesTakesMedians() {
        let curve = CostCurve.fromSamples([1: [0.03, 0.01, 0.02], 2: [4, 1, 3, 2], 3: [], 4: [.nan, 5]])
        XCTAssertEqual(curve.seconds, [1: 0.02, 2: 2.5, 4: 5])
    }

    func testCodableRoundTrip() throws {
        let data = try JSONEncoder().encode(CostCurve.stockMLXDefault)
        XCTAssertEqual(try JSONDecoder().decode(CostCurve.self, from: data), .stockMLXDefault)
    }
}
