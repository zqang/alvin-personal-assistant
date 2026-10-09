import AssistantKit
import Foundation
import MLX
import MLXLMCommon

/// Measures the target's cost curve c(S) on this device (plan §4.7, WP30 instruction 4): how
/// long one forward over S rows takes, for the widths a speculative round verifies.
///
/// It works on a scratch extension of the live session: the session is snapshotted, 32 tokens
/// of the system prefix are fed after its end (so attention runs over a realistic context),
/// then for each width S the forward of S tokens plus the evaluation of its logits is timed,
/// 5 times for S = 1 and 3 times for the others (after one untimed warm-up each), and each
/// forward is rolled back before the next. The medians form the curve. Afterwards the session is
/// restored exactly: the same ledger, checkpoints and cache, so the next logits are unchanged.
///
/// About a second on a phone. Runs on the engine queue between other jobs, under the engine's
/// GPU hooks; throws `EngineError.leftForeground` when the GPU is refused or stops being
/// allowed (the session is restored either way).
public enum CostProbe {
    /// Tokens fed after the session's end before measuring.
    public static let contextTokens = 32

    public static func measure(engine: InferenceEngine, widths: [Int] = [1, 2, 3, 4, 6, 8]) async throws -> CostCurve {
        try await engine.onQueue {
            let hooks = engine.configuration.hooks
            guard hooks.beginGPU() else { throw EngineError.leftForeground }
            defer {
                hooks.endGPU()
                Memory.clearCache()
            }
            return try withError {
                let filler = fillerTokens(session: engine.session, renderer: engine.loaded.renderer)
                return try measure(session: engine.session, filler: filler, widths: widths, isAllowed: hooks.isAllowed)
            }
        }
    }

    /// Times forwards of each width on a scratch extension of `session` (see the type's
    /// documentation). Engine queue only.
    static func measure(session: LiveSession, filler: [Int], widths: [Int], isAllowed: () -> Bool) throws -> CostCurve {
        precondition(!filler.isEmpty, "The probe needs tokens to feed.")
        let sizes = Array(Set(widths.filter { $0 >= 1 })).sorted()
        return try session.withScratch {
            guard isAllowed() else { throw EngineError.leftForeground }
            session.feedCacheOnly(filler)
            eval(session.target.cache)

            let rollback = session.target.supportsRollback
            var samples: [Int: [Double]] = [:]
            for width in sizes {
                let tokens = (0 ..< width).map { filler[$0 % filler.count] }
                let repeats = width == 1 ? 5 : 3
                for run in 0 ... repeats {
                    guard isAllowed() else { throw EngineError.leftForeground }
                    let started = ProcessInfo.processInfo.systemUptime
                    let result = session.feed(tokens, rows: .all, capture: rollback)
                    if let logits = result.logits {
                        try checkedEval(logits)
                    } else {
                        try checkedEval(session.target.cache)
                    }
                    let seconds = ProcessInfo.processInfo.systemUptime - started
                    // Run 0 warms up (kernel selection and compilation for this width).
                    if run > 0 {
                        samples[width, default: []].append(seconds)
                    }
                    if rollback {
                        // Back to the same context for the next forward; settle it outside the
                        // timing.
                        session.commit(result.capture, keep: 0, of: width)
                        eval(session.target.cache)
                    }
                }
            }
            return CostCurve.fromSamples(samples)
        }
    }

    /// `contextTokens` tokens from the start of the session (its system prefix), repeated if the
    /// session is shorter; a fixed text when it is empty.
    static func fillerTokens(session: LiveSession, renderer: any ChatTemplateRendering) -> [Int] {
        var source = Array(session.ledger.prefix(contextTokens))
        if source.isEmpty {
            source = renderer.encodeRaw("You are a helpful assistant. Answer briefly and clearly, in plain words.")
        }
        if source.isEmpty {
            source = [0]
        }
        return (0 ..< contextTokens).map { source[$0 % source.count] }
    }
}
