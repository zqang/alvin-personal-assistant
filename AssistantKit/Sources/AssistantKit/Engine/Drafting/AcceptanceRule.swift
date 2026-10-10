import Foundation

/// The outcome of one verified speculative round.
public struct RoundResult: Equatable, Sendable {
    /// Tokens to show, in order. Stop tokens are never included.
    public var emitted: [Int]
    /// Leading drafts that matched the target's samples (an accepted stop draft included).
    public var acceptedDrafts: Int
    /// Input rows to keep in the cache, counting row 0 (`y`). Every emitted token but the last is
    /// among them; when the last one is not (it was sampled rather than drafted, or the remaining
    /// cap cut the round), it is the next round's `y`.
    public var keep: Int
    /// The reply ended at a stop token in this round.
    public var stopped: Bool

    public init(emitted: [Int], acceptedDrafts: Int, keep: Int, stopped: Bool) {
        self.emitted = emitted
        self.acceptedDrafts = acceptedDrafts
        self.keep = keep
        self.stopped = stopped
    }

    /// The reply ended at a stop token that was itself an accepted draft (as opposed to a stop the
    /// target sampled in place of a draft). `acceptedDrafts` counts the stop draft; the drafts after
    /// it never counted toward the reply, and whether they matched is not reported, so a policy must
    /// not score the next depth as a rejection (`DraftPolicy.record(_:proposed:round:)` does not).
    ///
    /// Derived from `AcceptanceRule.resolve`'s output: only then are fewer tokens emitted than
    /// drafts accepted, because the stop draft is accepted but not shown.
    public var stoppedOnDraft: Bool {
        stopped && emitted.count < acceptedDrafts
    }
}

/// Sample-match verification. The target samples one token at every input row exactly as plain
/// decoding would; a draft is accepted only when it equals the target's own sample at that row.
/// With drafts that do not depend on the target's random draws, the emitted tokens therefore have
/// exactly the plain-decoding distribution at any temperature (exact rejection sampling for
/// deterministic drafts).
public enum AcceptanceRule {
    /// input = [y] + drafts; sampled[i] = the target's sample at input row i. Drafts must be ≤ remaining − 1.
    ///
    /// - n = the number of leading drafts with `sampled[i] == drafts[i]`.
    /// - If an accepted `drafts[i]` is a stop token: emitted = `drafts[0..<i]`, keep = i + 2 (the
    ///   stop token stays in the cache), stopped.
    /// - Otherwise emitted = `drafts[0..<n] + [sampled[n]]` and keep = n + 1. A stop token at
    ///   `sampled[n]` is left out of `emitted` and ends the reply.
    /// - `emitted` is capped at `remaining`. When the cap cuts it, the reply ends by length rather
    ///   than at any stop token beyond the cap, and keep becomes `max(1, remaining)` so the cache
    ///   holds `y` and every kept emitted token but the last.
    ///
    /// Drafts beyond `sampled.count − 1` cannot be verified and are ignored.
    public static func resolve(drafts: [Int], sampled: [Int], stopTokens: Set<Int>, remaining: Int) -> RoundResult {
        precondition(!sampled.isEmpty, "AcceptanceRule: a round samples at least one row")
        let verifiable = min(drafts.count, sampled.count - 1)
        var n = 0
        while n < verifiable, sampled[n] == drafts[n] { n += 1 }

        var result: RoundResult
        if let i = (0..<n).first(where: { stopTokens.contains(drafts[$0]) }) {
            result = RoundResult(emitted: Array(drafts[0..<i]), acceptedDrafts: i + 1, keep: i + 2, stopped: true)
        } else {
            let next = sampled[n]
            let stops = stopTokens.contains(next)
            var emitted = Array(drafts[0..<n])
            if !stops { emitted.append(next) }
            result = RoundResult(emitted: emitted, acceptedDrafts: n, keep: n + 1, stopped: stops)
        }

        let cap = max(0, remaining)
        if result.emitted.count > cap {
            result.emitted.removeLast(result.emitted.count - cap)
            result.keep = max(1, cap)
            result.stopped = false
        }
        return result
    }
}
