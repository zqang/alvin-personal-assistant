import AssistantKit
import Foundation
import MLX
import MLXLMCommon

/// The on-device check that decides whether the app may use the engine for this model and build
/// (plan §4.10, WP30 instruction 5). Greedy, at most 48 tokens per prompt, a few seconds on a
/// phone. Three checks:
///
/// 1. **Session reuse** (`sessionReuse`): two turns; the reused turn 2's first-token logits must
///    match a fresh rebuild of the same tokens: the same argmax, or a near-tie (the fresh top-1
///    minus top-2 margin below 0.25).
/// 2. **Speculation** (`speculation`): a copy-heavy prompt, 32 tokens decoded plainly and with
///    prompt lookup forced to 4 drafts per round. The tokens must be equal up to the first
///    near-tie of the plain run.
/// 3. **Rollback** (`rollback`): after a forced all-wrong round of 4 drafts, the cache must hold
///    exactly the kept tokens and the next logits must equal those of the same tokens fed without
///    the round (same argmax or a near-tie, and close in value).
///
/// The session is invalidated afterwards. A model without exact rollback (a hybrid on the stock
/// model code) never speculates, so checks 2 and 3 pass as not applicable; a tokenizer without
/// ChatML markers never reuses, so check 1 does.
public enum EngineSelfTest {
    public struct Result: Codable, Equatable, Sendable {
        public var passed: Bool
        /// Check name → passed: `sessionReuse`, `speculation`, `rollback`.
        public var checks: [String: Bool]
        /// One line per check (and the error that stopped the run, if any).
        public var detail: String
        public var seconds: Double
        /// `EngineInfo.formatVersion` when the test ran.
        public var formatVersion: Int

        public init(passed: Bool, checks: [String: Bool], detail: String, seconds: Double, formatVersion: Int) {
            self.passed = passed
            self.checks = checks
            self.detail = detail
            self.seconds = seconds
            self.formatVersion = formatVersion
        }
    }

    public static let checkNames = ["sessionReuse", "speculation", "rollback"]

    /// Fault injection for tests: the rollback check's round leaves the rejected rows' effect
    /// in the cache, so check 3 must fail.
    static var debugBreakRollback = false

    /// The near-tie margin for real quantized models (plan §6.4).
    static let nearTieMargin: Float = 0.25
    /// The largest difference the rollback check allows, relative to the reference: for the next
    /// logits `max |Δ|` over their scale, for each recurrent slot `‖Δ‖ / ‖reference‖`. Well above
    /// batched-versus-single-row rounding, well below a state that wasn't rolled back.
    static let rollbackTolerance: Float = 0.05
    static let maxTokens = 48
    static let speculationTokens = 32
    static let forcedDrafts = 4

    static let system = "You are a helpful assistant. Answer in one or two short sentences."
    static let firstQuestion = "Name three colours of the rainbow."
    static let secondQuestion = "Now name three animals that live in the sea."
    static let copyRequest = """
        Repeat this text exactly, then stop: "The quick brown fox jumps over the lazy dog. \
        The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."
        """

    public static func run(engine: InferenceEngine) async -> Result {
        let started = Date()
        var checks: [String: Bool] = [:]
        var lines: [String] = []
        await engine.invalidateSession()
        do {
            let (reused, reuseLine) = try await checkSessionReuse(engine)
            checks["sessionReuse"] = reused
            lines.append("sessionReuse: " + reuseLine)

            let (speculated, speculationLine, rolledBack, rollbackLine) = try await engine.onQueue {
                try onGPU(engine) { try checkSpeculationAndRollback(engine) }
            }
            checks["speculation"] = speculated
            lines.append("speculation: " + speculationLine)
            checks["rollback"] = rolledBack
            lines.append("rollback: " + rollbackLine)
        } catch {
            lines.append("stopped: \(error)")
        }
        await engine.invalidateSession()
        let passed = checkNames.allSatisfy { checks[$0] == true }
        return Result(
            passed: passed, checks: checks, detail: lines.joined(separator: "\n"),
            seconds: Date().timeIntervalSince(started), formatVersion: EngineInfo.formatVersion)
    }

    // MARK: Check 1

    private static func checkSessionReuse(_ engine: InferenceEngine) async throws -> (Bool, String) {
        let first = EngineRequest(system: system, turns: [ChatTurn(role: .user, text: firstQuestion)], maxTokens: maxTokens, greedy: true)
        var text = ""
        for try await event in engine.reply(first) {
            if case .text(let chunk) = event { text += chunk }
        }
        let second = EngineRequest(
            system: system,
            turns: [ChatTurn(role: .user, text: firstQuestion), ChatTurn(role: .assistant, text: text), ChatTurn(role: .user, text: secondQuestion)],
            maxTokens: maxTokens, greedy: true)

        return try await engine.onQueue {
            try onGPU(engine) {
                let hooks = engine.configuration.hooks
                let prepared = try withError { try engine.runtime.prepare(second, isAllowed: hooks.isAllowed) }
                let session = engine.session
                if session.noReuse || prepared.reason == "noReuse" {
                    engine.runtime.invalidate()
                    return (true, "not applicable (this tokenizer can't reuse the session)")
                }
                let live = prepared.firstLogits.reshaped(-1).asType(.float32)
                let fresh = try withError { try freshNextLogits(model: session.target.model, tokens: session.ledger, chunk: session.prefillChunk) }
                let comparison = compare(live, fresh)
                engine.runtime.invalidate()
                let ok = prepared.reusedTokens > 0 && comparison.agrees
                let line = "\(ok ? "ok" : "FAILED") (plan \(prepared.reason), reused \(prepared.reusedTokens) tokens, \(comparison))"
                return (ok, line)
            }
        }
    }

    // MARK: Checks 2 and 3

    /// Checks 2 and 3 on a fresh session holding the copy prompt. Engine queue only, under the GPU
    /// hooks.
    static func checkSpeculationAndRollback(_ engine: InferenceEngine) throws -> (Bool, String, Bool, String) {
        let hooks = engine.configuration.hooks
        let session = engine.session
        let request = EngineRequest(system: system, turns: [ChatTurn(role: .user, text: copyRequest)], maxTokens: speculationTokens, greedy: true)
        engine.runtime.invalidate()
        defer { engine.runtime.invalidate() }
        let prepared = try withError { try engine.runtime.prepare(request, isAllowed: hooks.isAllowed) }
        guard session.target.supportsRollback else {
            let line = "not applicable (this model can't roll a round back, so it never speculates)"
            return (true, line, true, line)
        }
        let stops = engine.loaded.stopTokenIDs

        // Plain greedy decoding, with the margin of every choice.
        let plain = try withError {
            try session.withScratch { () throws -> (tokens: [Int], margins: [Float]) in
                var logits = prepared.firstLogits
                var tokens: [Int] = []
                var margins: [Float] = []
                for _ in 0 ..< speculationTokens {
                    guard hooks.isAllowed() else { throw EngineError.leftForeground }
                    let (token, margin) = greedyChoice(logits)
                    margins.append(margin)
                    if stops.contains(token) { break }
                    tokens.append(token)
                    guard let next = session.feed([token], rows: .last).logits else { break }
                    logits = next
                }
                return (tokens, margins)
            }
        }

        // The same with prompt lookup forced to 4 drafts per round.
        let speculative = try withError {
            try session.withScratch { () throws -> (tokens: [Int], rounds: [SpeculativeLoop.Round]) in
                let lookup = PromptLookupDrafter(renderer: engine.loaded.renderer)
                lookup.reset(ledger: session.ledger, request: request)
                let context = GeneratorContext(
                    session: session, sampler: FastSampler(temperature: 0), firstLogits: prepared.firstLogits, stopTokens: stops,
                    maxTokens: speculationTokens, request: request, drafters: [], insideToolCall: { false },
                    isAllowed: hooks.isAllowed, renderer: engine.loaded.renderer, toolCallFormat: engine.toolCallFormat)
                let loop = SpeculativeLoop(
                    context, drafters: [lookup], policy: SharedDraftPolicy(curve: .stockMLXDefault),
                    options: SpeculativeLoop.Options(forcedDraftLength: forcedDrafts))
                var tokens: [Int] = []
                while true {
                    let (emitted, finished) = try loop.step()
                    tokens += emitted
                    if finished == .cancelled { throw EngineError.leftForeground }
                    if finished != nil { break }
                }
                try loop.flush()
                return (tokens, loop.rounds)
            }
        }
        let (agreed, comparedText) = compareSequences(plain: plain.tokens, margins: plain.margins, speculative: speculative.tokens)
        let accepted = speculative.rounds.reduce(0) { $0 + $1.accepted }
        let drafted = speculative.rounds.reduce(0) { $0 + $1.proposed }
        let speculationLine = "\(agreed ? "ok" : "FAILED") (\(comparedText); \(speculative.rounds.count) rounds, \(accepted)/\(drafted) drafts accepted)"

        let (rolledBack, rollbackLine) = try withError { try checkRollback(session, firstLogits: prepared.firstLogits, isAllowed: hooks.isAllowed) }
        return (agreed, speculationLine, rolledBack, rollbackLine)
    }

    /// From the prepared prompt: `y` (the greedy first token), then a round of `[y]` + 4 wrong
    /// drafts that keeps only `y`, then the greedy next token `z`. The cache must hold exactly
    /// the ledger, the recurrent state must be the one `[y]` alone leaves, and the logits after
    /// `z` must equal those of `[y, z]` fed without the round.
    private static func checkRollback(_ session: LiveSession, firstLogits: MLXArray, isAllowed: () -> Bool) throws -> (Bool, String) {
        guard isAllowed() else { throw EngineError.leftForeground }
        let layout = session.layout
        let y = greedyChoice(firstLogits).token
        let reference = session.withScratch { () -> (z: Int, logits: MLXArray, state: [(MLXArray?, MLXArray?)]) in
            let afterY = session.feed([y], rows: .last).logits!
            let state = recurrentSlots(session)
            let z = greedyChoice(afterY).token
            let afterZ = session.feed([z], rows: .last).logits!.reshaped(-1).asType(.float32)
            eval(afterZ)
            return (z, afterZ, state)
        }
        let vocabulary = session.target.vocabularySize > 0 ? session.target.vocabularySize : reference.logits.dim(0)
        // Four drafts, none of them the token the target picks after `y`.
        var wrong: [Int] = []
        var candidate = reference.z
        while wrong.count < forcedDrafts {
            candidate = (candidate + 1) % max(vocabulary, 2)
            if candidate != reference.z { wrong.append(candidate) }
        }

        let rolled = session.withScratch { () -> (offsetsMatch: Bool, rowZero: Int, stateDifference: Float, logits: MLXArray) in
            let result = session.feed([y] + wrong, rows: .all, capture: true)
            let rowZero = greedyChoice(result.logits![0 ..< 1]).token
            let unrolled = debugBreakRollback ? recurrentSlots(session) : []
            session.commit(result.capture, keep: 1, of: wrong.count + 1)
            if debugBreakRollback {
                // The broken rollback: recurrent layers keep the state after every row, and
                // pure-attention models keep the rejected rows in the cache.
                if layout.isHybrid {
                    for (index, slots) in zip(layout.recurrent, unrolled) {
                        let layer = session.target.cache[index] as! ArraysCache
                        layer[0] = slots.0
                        layer[1] = slots.1
                    }
                } else {
                    _ = session.target.forward(LiveSession.array(wrong), rows: .none, captureForRollback: false, wantHidden: false)
                }
            }
            let count = session.ledger.count
            let offsetsMatch = layout.attention.allSatisfy { session.target.cache[$0].offset == count }
            let stateDifference = zip(recurrentSlots(session), reference.state).map { live, expected in
                max(relativeDifference(live.0, expected.0), relativeDifference(live.1, expected.1))
            }.max() ?? 0
            let afterZ = session.feed([reference.z], rows: .last).logits!.reshaped(-1).asType(.float32)
            eval(afterZ)
            return (offsetsMatch, rowZero, stateDifference, afterZ)
        }
        if debugBreakRollback && !layout.isHybrid {
            // The injected fault left rows the scratch restore doesn't know about.
            session.reset()
        }

        let comparison = compare(rolled.logits, reference.logits)
        let close = comparison.maxDifference <= rollbackTolerance * max(1, comparison.scale)
        let stateClose = rolled.stateDifference <= rollbackTolerance
        let ok = rolled.offsetsMatch && comparison.agrees && close && stateClose
        var line = "\(ok ? "ok" : "FAILED") (\(comparison)"
        if layout.isHybrid {
            line += ", recurrent state off by \(String(format: "%.2g", rolled.stateDifference))"
        }
        if !rolled.offsetsMatch { line += ", the cache holds rows the ledger doesn't" }
        if rolled.rowZero != reference.z { line += ", the round's row 0 picked \(rolled.rowZero) instead of \(reference.z)" }
        return (ok, line + ")")
    }

    /// The recurrent layers' slots (conv state, gated-delta state), in layout order.
    private static func recurrentSlots(_ session: LiveSession) -> [(MLXArray?, MLXArray?)] {
        session.layout.recurrent.map { index in
            let layer = session.target.cache[index] as! ArraysCache
            return (layer[0], layer[1])
        }
    }

    /// `‖a − b‖ / ‖b‖` (Frobenius, float32); 0 when both are nil, 1 when only one is.
    static func relativeDifference(_ a: MLXArray?, _ b: MLXArray?) -> Float {
        guard let a, let b else { return a == nil && b == nil ? 0 : 1 }
        guard a.shape == b.shape else { return 1 }
        let x = a.asType(.float32), y = b.asType(.float32)
        let difference = MLX.sqrt(MLX.sum(MLX.square(x - y)))
        let norm = MLX.sqrt(MLX.sum(MLX.square(y)))
        eval(difference, norm)
        let denominator = max(norm.item(Float.self), 1e-12)
        return difference.item(Float.self) / denominator
    }

    // MARK: Helpers

    /// Runs `body` between the engine's `beginGPU` and `endGPU`. Engine queue only.
    private static func onGPU<T>(_ engine: InferenceEngine, _ body: () throws -> T) throws -> T {
        let hooks = engine.configuration.hooks
        guard hooks.beginGPU() else { throw EngineError.leftForeground }
        defer {
            hooks.endGPU()
            Memory.clearCache()
        }
        return try body()
    }

    /// The greedy token of `[1, V]` (or `[V]`) logits and its top-1 minus top-2 margin.
    static func greedyChoice(_ logits: MLXArray) -> (token: Int, margin: Float) {
        let row = logits.reshaped(-1).asType(.float32)
        let best = argMax(row, axis: -1)
        let topTwo = MLX.sorted(MLX.top(row, k: 2, axis: -1), axis: -1)
        let margin = topTwo[1] - topTwo[0]
        eval(best, margin)
        return (best.item(Int.self), margin.item(Float.self))
    }

    /// The logits after `tokens` from a fresh cache: every token but the last is fed cache-only
    /// (the head never runs on them), the last alone. `[V]` float32.
    static func freshNextLogits(model: any LanguageModel, tokens: [Int], chunk: Int) throws -> MLXArray {
        precondition(!tokens.isEmpty, "Nothing to feed.")
        let cache = model.newCache(parameters: nil)
        let body = tokens.dropLast()
        var start = body.startIndex
        let size = max(1, chunk)
        while start < body.endIndex {
            let end = min(start + size, body.endIndex)
            _ = model(LiveSession.array(Array(body[start ..< end]))[.newAxis], cache: cache)
            try checkedEval(cache)
            start = end
        }
        let logits = model(LiveSession.array([tokens[tokens.count - 1]])[.newAxis], cache: cache)
        let last = logits[0, logits.dim(1) - 1].asType(.float32)
        try checkedEval(last)
        return last
    }

    struct Comparison: CustomStringConvertible {
        let candidateArgmax: Int
        let referenceArgmax: Int
        /// Reference top-1 minus top-2.
        let referenceMargin: Float
        let maxDifference: Float
        /// `max |reference|`.
        let scale: Float

        var agrees: Bool {
            candidateArgmax == referenceArgmax || referenceMargin < nearTieMargin
        }

        var description: String {
            "argmax \(candidateArgmax)/\(referenceArgmax), margin \(String(format: "%.3f", referenceMargin)), max |Δ| \(String(format: "%.4g", maxDifference)) of \(String(format: "%.3g", scale))"
        }
    }

    /// `candidate` versus `reference`, both `[V]` float32.
    static func compare(_ candidate: MLXArray, _ reference: MLXArray) -> Comparison {
        let difference = MLX.abs(candidate - reference).max()
        let scale = MLX.abs(reference).max()
        let candidateBest = argMax(candidate, axis: -1)
        let (referenceBest, margin) = greedyChoice(reference)
        eval(difference, scale, candidateBest)
        return Comparison(
            candidateArgmax: candidateBest.item(Int.self), referenceArgmax: referenceBest, referenceMargin: margin,
            maxDifference: difference.item(Float.self), scale: scale.item(Float.self))
    }

    /// Token equality up to the first near-tie of the plain run (plan §6.4).
    static func compareSequences(plain: [Int], margins: [Float], speculative: [Int]) -> (Bool, String) {
        var position = 0
        while position < plain.count && position < speculative.count && plain[position] == speculative[position] {
            position += 1
        }
        if position == plain.count && position == speculative.count {
            return (true, "\(plain.count) tokens identical")
        }
        let margin = position < margins.count ? margins[position] : nil
        if let margin, margin < nearTieMargin {
            return (true, "identical up to a near-tie at token \(position) (margin \(String(format: "%.3f", margin)))")
        }
        let marginText = margin.map { String(format: "%.3f", $0) } ?? "unknown"
        return (false, "token \(position) differs: plain \(position < plain.count ? String(plain[position]) : "end"), speculative \(position < speculative.count ? String(speculative[position]) : "end"), margin \(marginText)")
    }
}
