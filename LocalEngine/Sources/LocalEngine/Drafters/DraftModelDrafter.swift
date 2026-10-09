import AssistantKit
import Foundation
import MLX
import MLXLMCommon

/// Drafting with a small pure-attention model that shares the target's tokenizer, such as
/// Qwen3-0.6B for Qwen3-4B (plan §4.7).
///
/// - The draft model has its own `KVCacheSimple` cache, kept across rounds and replies.
///   `propose` brings it to the context by trimming to the longest common prefix and feeding the
///   rest. That covers both upstream rules: after a rejection the cache is trimmed back to the
///   accepted tokens (`keep − 1` drafts), and after a full acceptance the last draft and the bonus
///   token are fed together.
/// - Drafting is greedy and pipelined: each drafted token is fed while still lazy, and the host
///   syncs once for the whole draft. Greedy drafts don't depend on the target's random draws, so
///   sample-match verification stays exact.
/// - `costPerToken` starts at a prior and is measured at the first round: the drafting time per
///   token relative to the target's single-row time (the round's verify time divided by
///   c(K + 1)).
/// - The draft must match the target's vocabulary size, stop tokens and tokenization
///   (`compatibilityProblem`).
///
/// An MLX error while drafting starts the draft cache over and skips the proposal; the target's
/// reply is unaffected. Not thread-safe: engine queue only.
public final class DraftModelDrafter: Drafter {
    public enum Problem: Error, Equatable, CustomStringConvertible {
        /// The draft model's cache has recurrent, rotating or quantized layers.
        case notPureAttention
        case vocabularyMismatch(target: Int, draft: Int)
        case stopTokensMismatch(target: Set<Int>, draft: Set<Int>)
        case tokenizerMismatch

        public var description: String {
            switch self {
            case .notPureAttention: return "the draft model's cache isn't pure attention"
            case .vocabularyMismatch(let target, let draft): return "vocabulary sizes differ (target \(target), draft \(draft))"
            case .stopTokensMismatch(let target, let draft): return "stop tokens differ (target \(target.sorted()), draft \(draft.sorted()))"
            case .tokenizerMismatch: return "the tokenizers encode differently"
            }
        }
    }

    public var source: DraftSource { .draftModel }
    /// Drafting cost per token relative to one single-row target forward.
    public private(set) var costPerToken: Double
    public var wantsHidden: Bool { false }

    public let draft: LoadedModel
    /// The draft model with its cache.
    public let target: StockTarget
    /// The draft model's vocabulary size.
    public let vocabularySize: Int
    /// The tokens in the draft cache, in order.
    public private(set) var fed: [Int] = []
    /// Whether `costPerToken` has been measured.
    public private(set) var calibrated = false
    /// Most tokens one draft-model forward feeds while catching up with the context.
    public var prefillChunk = 512

    /// The timing of the last measured draft (sync excluded), until calibrated.
    private var measuredDraft: (seconds: Double, count: Int)?

    /// Throws `Problem.notPureAttention` for a draft model whose cache can't be trimmed exactly.
    public init(draft: LoadedModel, defaultCostPerToken: Double = 0.25) throws {
        let target = StockTarget(model: draft.model, directory: draft.directory)
        guard target.exactLayout, !target.layout.isHybrid, !target.layout.attention.isEmpty else {
            throw Problem.notPureAttention
        }
        self.draft = draft
        self.target = target
        self.vocabularySize = target.vocabularySize
        self.costPerToken = defaultCostPerToken
    }

    /// Why `draft` can't draft for a target with this vocabulary size, stop tokens and
    /// tokenizer; nil when it can.
    public static func compatibilityProblem(
        draft: LoadedModel, draftVocabularySize: Int, targetVocabularySize: Int, stopTokens: Set<Int>,
        renderer: any ChatTemplateRendering
    ) -> Problem? {
        guard draftVocabularySize == targetVocabularySize else {
            return .vocabularyMismatch(target: targetVocabularySize, draft: draftVocabularySize)
        }
        guard draft.stopTokenIDs == stopTokens else {
            return .stopTokensMismatch(target: stopTokens, draft: draft.stopTokenIDs)
        }
        let sample = "Hello, world! 3.14 <|im_end|>\n<tool_call>\n{\"name\": \"x\"} Grüße, 你好."
        guard draft.renderer.encodeRaw(sample) == renderer.encodeRaw(sample) else {
            return .tokenizerMismatch
        }
        return nil
    }

    /// Why this drafter can't serve a target with this vocabulary size, stop tokens and
    /// tokenizer; nil when it can.
    public func compatibilityProblem(targetVocabularySize: Int, stopTokens: Set<Int>, renderer: any ChatTemplateRendering) -> Problem? {
        Self.compatibilityProblem(
            draft: draft, draftVocabularySize: vocabularySize, targetVocabularySize: targetVocabularySize,
            stopTokens: stopTokens, renderer: renderer)
    }

    /// Nothing to do: `propose` catches the draft cache up with whatever context it is given.
    public func reset(ledger: [Int], request: EngineRequest) {}

    public func propose(context: ArraySlice<Int>, maxTokens: Int) -> DraftProposal? {
        guard maxTokens > 0, !context.isEmpty else { return nil }
        do {
            let tokens = try withError { try draftTokens(after: context, count: maxTokens) }
            return tokens.isEmpty ? nil : DraftProposal(tokens: tokens, source: .draftModel, matchLength: 0)
        } catch {
            startOver()
            return nil
        }
    }

    public func observe(_ round: RoundObservation) {}

    /// Sets `costPerToken` from the first round that verified this drafter's proposal (and had
    /// a clean drafting measurement): `verifySeconds` for `rows` rows is c(rows) target rows, by
    /// `curve`, and each measured drafting forward is one drafted token.
    func calibrate(verifySeconds: Double, rows: Int, curve: CostCurve) {
        guard !calibrated, let measured = measuredDraft, measured.count > 0, verifySeconds > 0 else { return }
        let targetRow = verifySeconds / max(curve.relative(rows), 1)
        let perToken = measured.seconds / Double(measured.count)
        guard targetRow.isFinite, perToken.isFinite, targetRow > 0 else { return }
        costPerToken = min(max(perToken / targetRow, 0.01), 4)
        calibrated = true
        measuredDraft = nil
    }

    /// Empties the draft cache.
    public func startOver() {
        target.resetCache()
        fed = []
    }

    // MARK: Drafting

    /// Catches the cache up with `context`, then drafts `count` tokens greedily.
    private func draftTokens(after context: ArraySlice<Int>, count: Int) throws -> [Int] {
        // The longest common prefix, leaving at least the context's last token to feed (its
        // logits predict the first draft).
        var common = 0
        let limit = min(fed.count, context.count - 1)
        while common < limit && fed[common] == context[context.startIndex + common] {
            common += 1
        }
        if common < fed.count {
            EngineCacheOps.trimAttention(target.cache, layout: target.layout, by: fed.count - common)
            fed.removeLast(fed.count - common)
        }

        // Until calibrated, time the drafting forwards. A short catch-up (the round's bonus, or
        // the last draft and the bonus) is one single-token-like forward and counts as the first
        // draft's; a long one (a new prompt) is finished before the timing starts.
        let catchUp = context.count - common
        let timesCatchUp = !calibrated && catchUp <= 2
        let measuring = timesCatchUp || (!calibrated && count >= 2)
        var started = ProcessInfo.processInfo.systemUptime

        // Feed the rest of the context; only the last chunk computes logits (its last row).
        var logits: MLXArray?
        var start = common
        let chunk = max(1, prefillChunk)
        while start < context.count {
            let end = min(start + chunk, context.count)
            let piece = Array(context[(context.startIndex + start) ..< (context.startIndex + end)])
            if end == context.count {
                logits = target.forward(LiveSession.array(piece), rows: .last, captureForRollback: false, wantHidden: false).logits
            } else {
                _ = target.forward(LiveSession.array(piece), rows: .none, captureForRollback: false, wantHidden: false)
                asyncEval(target.cache)
            }
            fed += piece
            start = end
        }
        guard let first = logits else { return [] }
        if measuring && !timesCatchUp {
            try checkedEval(first)
            started = ProcessInfo.processInfo.systemUptime
        }

        var drafted: [MLXArray] = []
        var current = argMax(first, axis: -1).asType(.int32).reshaped([1])
        asyncEval(current)
        drafted.append(current)
        while drafted.count < count {
            guard let next = target.forward(current, rows: .last, captureForRollback: false, wantHidden: false).logits else { break }
            current = argMax(next, axis: -1).asType(.int32).reshaped([1])
            asyncEval(current)
            drafted.append(current)
        }
        let all = concatenated(drafted, axis: 0)
        try checkedEval(all)
        let tokens = all.asArray(Int32.self).map { Int($0) }
        // Every draft but the last was fed.
        fed += tokens.dropLast()
        if measuring {
            // One forward per draft (the catch-up's made the first), or per draft after the
            // first when the catch-up wasn't timed.
            let forwards = timesCatchUp ? tokens.count : tokens.count - 1
            if forwards > 0 {
                measuredDraft = (ProcessInfo.processInfo.systemUptime - started, forwards)
            }
        }
        return tokens
    }
}
