import AssistantKit
import Foundation
import MLX
import MLXLMCommon

/// The model's live cache and the exact tokens in it (plan §4.4).
///
/// - **L1.** The cache holds exactly `ledger` followed by `pendingCount` tokens that were fed
///   lazily (their values still on the GPU). `pendingCount` is non-zero only inside a pipelined
///   decode step; `resolvePending` moves them into the ledger once synced. Every forward
///   appends; every `commit` or rewind truncates.
/// - **L2.** A checkpoint is the recurrent state at a position (references, not copies).
///   Restoring position q trims the attention layers and sets the recurrent slots back, and
///   drops every checkpoint above q.
/// - **L3** (overlap) is applied by `SessionRuntime` before appending a continuation.
///
/// Not thread-safe: use it from the engine queue only.
public final class LiveSession {
    public let target: any TargetModel
    /// The exact tokens in the cache (synced ones).
    public private(set) var ledger: [Int] = []
    /// Tokens fed lazily whose values aren't in the ledger yet.
    public private(set) var pendingCount = 0
    public let checkpoints = CheckpointStore()
    /// The conversation the cache holds, for the session planner; nil when it holds none that a
    /// request could reuse.
    public internal(set) var snapshot: SessionSnapshot?
    /// When set, nothing is reused: every reply renders the whole conversation into an empty
    /// cache (a tokenizer without ChatML markers, or a cache layout that can't be rewound).
    public internal(set) var noReuse: Bool
    /// Most tokens one cache-only forward feeds while re-feeding after a rewind.
    public var prefillChunk = 512

    /// Whether every cache layer is a `KVCacheSimple` or `MambaCache`, so the cache can be
    /// trimmed, snapshotted and restored exactly.
    public let reusable: Bool

    public init(target: any TargetModel, noReuse: Bool = false) {
        self.target = target
        self.reusable = CacheLayout.detect(target.cache) != nil
        self.noReuse = noReuse || !reusable
    }

    public var layout: CacheLayout { target.layout }

    /// The prefix key of the conversation in the cache.
    public var prefixKey: String? { snapshot?.prefixKey }

    /// Tokens in the cache, synced or not.
    public var cachedCount: Int { ledger.count + pendingCount }

    // MARK: Feeding

    /// Feeds tokens whose values are known: they go into the cache and the ledger.
    @discardableResult
    public func feed(_ tokens: [Int], rows: LogitRows, capture: Bool = false, hidden: Bool = false) -> ForwardResult {
        precondition(!tokens.isEmpty, "Nothing to feed.")
        precondition(pendingCount == 0, "Resolve the pending tokens before feeding known ones.")
        let result = target.forward(Self.array(tokens), rows: rows, captureForRollback: capture, wantHidden: hidden)
        ledger.append(contentsOf: tokens)
        return result
    }

    /// Feeds `count` tokens that may still be lazy (sampled on the GPU): the cache gets them
    /// now, the ledger once `resolvePending` is given their values.
    @discardableResult
    public func feed(_ tokens: MLXArray, count: Int, rows: LogitRows, capture: Bool = false, hidden: Bool = false) -> ForwardResult {
        precondition(count > 0 && tokens.size == count, "Feeding \(tokens.size) tokens as \(count).")
        let result = target.forward(tokens.reshaped([count]), rows: rows, captureForRollback: capture, wantHidden: hidden)
        pendingCount += count
        return result
    }

    /// Moves the oldest pending tokens into the ledger, now that their values are known.
    public func resolvePending(_ tokens: [Int]) {
        precondition(tokens.count <= pendingCount, "Resolving \(tokens.count) of \(pendingCount) pending tokens.")
        ledger.append(contentsOf: tokens)
        pendingCount -= tokens.count
    }

    /// Feeds known tokens without computing logits, in chunks, leaving the cache lazy (each
    /// chunk is evaluated asynchronously).
    public func feedCacheOnly(_ tokens: [Int]) {
        _ = feedCacheOnly(tokens, isAllowed: { true })
    }

    /// `feedCacheOnly`, checking `isAllowed` before every chunk. False once it is refused: the
    /// chunks fed so far are in the cache and the ledger.
    private func feedCacheOnly(_ tokens: [Int], isAllowed: () -> Bool) -> Bool {
        var start = 0
        let chunk = max(1, prefillChunk)
        while start < tokens.count {
            guard isAllowed() else { return false }
            let end = min(start + chunk, tokens.count)
            feed(Array(tokens[start ..< end]), rows: .none)
            asyncEval(target.cache)
            start = end
        }
        return true
    }

    // MARK: Rolling back

    /// Keeps the first `keep` of the `verified` tokens the last forward fed (a speculative
    /// round), and drops the rest from the cache and the ledger. The tokens must have been fed
    /// as known tokens.
    public func commit(_ capture: (any RoundCapture)?, keep: Int, of verified: Int) {
        precondition(pendingCount == 0, "Resolve the pending tokens before committing.")
        precondition(keep >= 0 && keep <= verified && verified <= ledger.count, "Can't keep \(keep) of \(verified) tokens.")
        target.commit(capture, keep: keep, of: verified)
        if keep < verified {
            ledger.removeLast(verified - keep)
            checkpoints.drop(above: ledger.count)
        }
    }

    /// Moves the cache back to `position` (≤ the ledger's length): pure-attention caches trim;
    /// hybrid caches restore the deepest checkpoint ≤ `position` (or start empty) and re-feed
    /// the ledger from there, cache-only. Checkpoints above the restored point are dropped.
    public func rewind(to position: Int) {
        _ = rewind(to: position, refeedingWhile: { true })
    }

    /// `rewind(to:)`, checking `isAllowed` before every chunk it re-feeds (plan §4.11). Throws
    /// `PrefillInterrupted` once it is refused: the ledger then ends between the restored
    /// checkpoint and `position`, and holds exactly what the cache holds.
    func rewind(to position: Int, isAllowed: () -> Bool) throws {
        guard rewind(to: position, refeedingWhile: isAllowed) else { throw PrefillInterrupted() }
    }

    /// The rewind; false if a chunk of its re-feed was refused.
    private func rewind(to position: Int, refeedingWhile isAllowed: () -> Bool) -> Bool {
        precondition(pendingCount == 0, "Resolve the pending tokens before rewinding.")
        precondition(position >= 0 && position <= ledger.count, "Can't rewind to \(position) of \(ledger.count).")
        guard position < ledger.count else { return true }
        precondition(reusable, "This cache can't be rewound exactly.")

        if !layout.isHybrid {
            EngineCacheOps.trimAttention(target.cache, layout: layout, by: ledger.count - position)
            ledger.removeLast(ledger.count - position)
            checkpoints.drop(above: position)
            return true
        }

        let mark: Int
        if let snapshot = checkpoints.deepest(atOrBelow: position) {
            EngineCacheOps.restore(target.cache, layout: layout, to: snapshot, currentPosition: ledger.count)
            mark = snapshot.position
        } else {
            target.resetCache()
            mark = 0
        }
        let refeed = Array(ledger[mark ..< position])
        ledger.removeLast(ledger.count - mark)
        checkpoints.drop(above: mark)
        return feedCacheOnly(refeed, isAllowed: isAllowed)
    }

    /// Records a checkpoint at the end of the ledger.
    public func checkpoint(_ label: CheckpointMark) {
        precondition(pendingCount == 0, "Resolve the pending tokens before a checkpoint.")
        checkpoints.record(label, EngineCacheOps.snapshot(target.cache, layout: layout, position: ledger.count))
    }

    /// Holds a checkpoint at the end of the ledger under no mark until `releaseResumePoint()`
    /// (or a rewind below it, or `reset`): the start of a prefill. A prefill interrupted part-way
    /// leaves the cache past it, and the next plan rewinds to it; restoring it spares re-feeding
    /// from the checkpoint before (for an appended turn, the whole previous reply). Reuses a
    /// checkpoint already there. Pure-attention caches trim instead, so they hold none.
    public func holdResumePoint() {
        precondition(pendingCount == 0, "Resolve the pending tokens before a checkpoint.")
        guard layout.isHybrid else { return }
        let position = ledger.count
        let existing = checkpoints.deepest(atOrBelow: position).flatMap { $0.position == position ? $0 : nil }
        checkpoints.holdResume(existing ?? EngineCacheOps.snapshot(target.cache, layout: layout, position: position))
    }

    /// Lets go of the checkpoint `holdResumePoint()` held (its prefill completed).
    public func releaseResumePoint() {
        checkpoints.releaseResume()
    }

    /// Bytes one checkpoint of the current cache would hold.
    public var bytesPerCheckpoint: Int {
        EngineCacheOps.recurrentBytes(target.cache, layout: layout)
    }

    // MARK: Replacing the cache

    /// Starts over with an empty cache, ledger, checkpoints and snapshot.
    public func reset() {
        target.resetCache()
        ledger = []
        pendingCount = 0
        checkpoints.removeAll()
        snapshot = nil
    }

    /// Takes over `cache`, which holds exactly `tokens` (a persisted system prefix). Checkpoints
    /// and the snapshot are cleared.
    public func adopt(_ cache: [KVCache], holding tokens: [Int]) throws {
        try target.adopt(cache)
        ledger = tokens
        pendingCount = 0
        checkpoints.removeAll()
        snapshot = nil
    }

    // MARK: Scratch work

    /// Runs `body`, which may feed (and commit) tokens after the current end, then puts the
    /// cache, ledger and checkpoints back exactly as they were. For warm-up, cost probes and
    /// checks that must not disturb the session.
    ///
    /// A cache that can't be rewound exactly (`reusable == false`: rotating or quantized
    /// layers) is rebuilt from the ledger instead, which costs a prefill of the whole ledger
    /// (nothing when it is empty, as at warm-up). Probes that run often should check
    /// `reusable` first and skip such sessions.
    public func withScratch<R>(_ body: () throws -> R) rethrows -> R {
        precondition(pendingCount == 0, "Resolve the pending tokens before scratch work.")
        let position = ledger.count
        let saved = reusable ? EngineCacheOps.snapshot(target.cache, layout: layout, position: position) : nil
        let savedLedger = ledger
        defer {
            let current = ledger.count + pendingCount
            precondition(current >= position, "Scratch work rewound below its start.")
            if let saved {
                EngineCacheOps.restore(target.cache, layout: layout, to: saved, currentPosition: current)
                ledger = savedLedger
                pendingCount = 0
                checkpoints.drop(above: position)
            } else {
                rebuild(holding: savedLedger)
            }
        }
        return try body()
    }

    /// Replaces the cache with a fresh one holding `tokens` (fed again, cache-only): how a cache
    /// that can't be rewound exactly gets back to an earlier state. Checkpoints are dropped;
    /// the snapshot, which describes the same tokens, stays.
    private func rebuild(holding tokens: [Int]) {
        target.resetCache()
        ledger = []
        pendingCount = 0
        checkpoints.removeAll()
        if !tokens.isEmpty {
            feedCacheOnly(tokens)
        }
    }

    // MARK: Consistency

    /// What `assertConsistent` found.
    public struct ConsistencyReport: Sendable, CustomStringConvertible {
        /// Tokens in the ledger.
        public let ledgerCount: Int
        /// Whether every layer that counts its tokens (all but the recurrent ones) holds as many
        /// as the ledger.
        public let offsetsMatch: Bool
        /// `max |live − fresh|` of the next-step logits (float32).
        public let maxAbsDifference: Float
        /// Argmax of the live and the fresh next-step logits.
        public let liveArgmax: Int
        public let freshArgmax: Int
        /// The fresh run's top-1 minus top-2 logit, for the near-tie rule (§6.4).
        public let freshMargin: Float
        /// `max |fresh|`, the logits' scale.
        public let freshScale: Float

        /// allClose with `rtol = atol = tolerance`, relative to the logits' scale (float32 tiny
        /// models: 1e-4).
        public func isConsistent(tolerance: Float = 1e-4) -> Bool {
            offsetsMatch && maxAbsDifference <= tolerance * max(1, freshScale)
        }

        /// Same argmax, or a near-tie below `margin` (real quantized models: 0.25).
        public func agrees(margin: Float) -> Bool {
            offsetsMatch && (liveArgmax == freshArgmax || freshMargin < margin)
        }

        public var description: String {
            "ledger \(ledgerCount), offsets \(offsetsMatch ? "ok" : "MISMATCH"), max |Δ| \(maxAbsDifference), argmax \(liveArgmax)/\(freshArgmax), margin \(freshMargin)"
        }
    }

    /// Debug check of L1: feeds `probe` after the live cache (then restores it, or rebuilds it
    /// from the ledger when it can't be rewound exactly) and after a fresh cache rebuilt from
    /// the ledger, through the raw model, and compares the logits. `probe` defaults to the
    /// ledger's first token.
    public func assertConsistent(probe: Int? = nil) -> ConsistencyReport {
        precondition(pendingCount == 0, "Resolve the pending tokens before checking.")
        let count = ledger.count
        let offsetsMatch = target.cache.allSatisfy { $0 is ArraysCache || $0.offset == count }
        let token = probe ?? ledger.first ?? 0
        let model = target.model

        let saved = reusable ? EngineCacheOps.snapshot(target.cache, layout: layout, position: count) : nil
        let live = Self.lastRow(model(Self.array([token])[.newAxis], cache: target.cache))
        eval(live)
        if let saved {
            EngineCacheOps.restore(target.cache, layout: layout, to: saved, currentPosition: count + 1)
        } else {
            rebuild(holding: ledger)
        }

        let fresh = model.newCache(parameters: nil)
        let tokens = ledger + [token]
        var start = 0
        var last: MLXArray?
        let chunk = max(1, prefillChunk)
        while start < tokens.count {
            let end = min(start + chunk, tokens.count)
            let logits = model(Self.array(Array(tokens[start ..< end]))[.newAxis], cache: fresh)
            if end == tokens.count {
                last = Self.lastRow(logits)
            } else {
                eval(fresh)
            }
            start = end
        }
        let expected = last!
        let difference = MLX.abs(live - expected).max()
        let scale = MLX.abs(expected).max()
        let liveBest = live.argMax()
        let freshBest = expected.argMax()
        let topTwo = MLX.sorted(MLX.top(expected, k: 2, axis: -1), axis: -1)
        let margin = topTwo[1] - topTwo[0]
        eval(difference, scale, liveBest, freshBest, margin)
        return ConsistencyReport(
            ledgerCount: count, offsetsMatch: offsetsMatch, maxAbsDifference: difference.item(Float.self),
            liveArgmax: liveBest.item(Int.self), freshArgmax: freshBest.item(Int.self), freshMargin: margin.item(Float.self),
            freshScale: scale.item(Float.self))
    }

    static func array(_ tokens: [Int]) -> MLXArray {
        MLXArray(tokens.map { Int32($0) })
    }

    /// The last row of `[1, S, V]` logits, as float32 `[V]`.
    static func lastRow(_ logits: MLXArray) -> MLXArray {
        logits[0, logits.dim(1) - 1].asType(.float32)
    }
}
