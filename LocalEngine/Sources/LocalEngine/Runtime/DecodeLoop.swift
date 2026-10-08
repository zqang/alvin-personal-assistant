import AssistantKit
import Foundation
import MLX
import MLXLMCommon

/// Plain pipelined decoding, the engine's default generator (plan §4.6).
///
/// Each step:
/// 1. feeds the still-lazy token y_k through `forward([y_k], .last)`;
/// 2. samples y_{k+1} lazily and starts evaluating it (`asyncEval`), so the GPU computes the
///    next step while the host handles this one;
/// 3. syncs y_k, appends it to the ledger and emits it;
/// 4. stops on a stop token (already fed, as `TokenIterator` leaves it, F4), at `maxTokens`, or
///    when `isAllowed()` turns false (checked before feeding, so the ledger ends with the last
///    emitted token and nothing is pending).
///
/// Every emitted token is in the cache when `step` returns, so `flush` has nothing to do.
public final class DecodeLoop: TokenGenerator {
    private let context: GeneratorContext
    /// The next token to feed, sampled but maybe not evaluated yet (`[1]`).
    private var next: MLXArray
    private var nextTop1: MLXArray
    private var emitted = 0
    private var finished: EngineFinish.Reason?

    public init(_ context: GeneratorContext) {
        self.context = context
        let position = context.session.cachedCount
        let (token, top1) = context.sampler.sample(context.firstLogits, positions: position ..< (position + 1))
        next = token
        nextTop1 = top1
        asyncEval(token, top1)
        if context.maxTokens <= 0 {
            finished = .length
        }
    }

    public func step() throws -> (emitted: [Int], finished: EngineFinish.Reason?) {
        if let finished {
            return ([], finished)
        }
        guard context.isAllowed() else {
            finished = .cancelled
            return ([], .cancelled)
        }

        let session = context.session
        let position = session.cachedCount
        let token = next
        let top1 = nextTop1
        let result = session.feed(token, count: 1, rows: .last)
        guard let logits = result.logits else {
            preconditionFailure("A `.last` forward returned no logits.")
        }
        let (following, followingTop1) = context.sampler.sample(logits, positions: (position + 1) ..< (position + 2))
        asyncEval(following, followingTop1)

        let value = token.item(Int.self)
        session.resolvePending([value])
        context.confidence.record(top1.item(Float.self))
        next = following
        nextTop1 = followingTop1

        if context.stopTokens.contains(value) {
            finished = .stop
            return ([], .stop)
        }
        emitted += 1
        if emitted >= context.maxTokens {
            finished = .length
            return ([value], .length)
        }
        return ([value], nil)
    }

    public func flush() throws {}

    public var speculation: SpeculationStats? { nil }
}
