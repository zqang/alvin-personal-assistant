import Foundation
import MLX
import MLXLMCommon

/// Temperature / top-p / top-k sampling that only sorts the top k entries (plan §4.6).
///
/// The stock `TopPSampler` sorts the whole vocabulary for every token (F2). Its filter chain
/// keeps the tokens of the nucleus computed on the full distribution (a token is kept when the
/// probability of the tokens ranked above it is below `topP`), intersected with the top k. That
/// set is a prefix of the descending order, so it can be found among the top k alone:
///
/// 1. float32 log-softmax over the vocabulary;
/// 2. `argPartition` for the top k, then sort those k descending;
/// 3. keep the entries whose *exclusive* cumulative probability is below `topP`;
/// 4. sample `categorical(kept / T)` and map back to vocabulary ids.
///
/// The kept set (and so the distribution) is the stock one. A temperature of 0 is `argMax`.
/// With a seed, row i draws with `MLXRandom.key(seed &+ position_i)`, so a position always
/// samples the same way (tests: speculative and plain decoding then agree token for token).
public struct FastSampler: @unchecked Sendable {
    public let temperature: Float
    public let topP: Float
    public let topK: Int
    public let seed: UInt64?
    private let randomState: MLXRandom.RandomState

    public init(temperature: Float, topP: Float = 1, topK: Int = 0, seed: UInt64? = nil) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.seed = seed
        self.randomState = MLXRandom.RandomState()
    }

    /// The configured sampler, or a greedy one.
    public init(configuration: EngineConfiguration, greedy: Bool) {
        self.init(
            temperature: greedy ? 0 : configuration.temperature, topP: configuration.topP, topK: configuration.topK,
            seed: configuration.seed)
    }

    public var isGreedy: Bool { temperature <= 0 }

    /// Samples one token per row of `logits` (`[S, V]`, or `[V]`).
    ///
    /// - Parameter positions: the ledger positions the sampled tokens will take (row i →
    ///   `positions.lowerBound + i`); used for position-keyed sampling when a seed is set.
    /// - Returns: `tokens` int32 `[S]` and `top1` float32 `[S]`, the probability of each row's
    ///   most likely token (confidence telemetry). Both lazy.
    public func sample(_ logits: MLXArray, positions: Range<Int>?) -> (tokens: MLXArray, top1: MLXArray) {
        let rows = logits.ndim == 1 ? logits.reshaped(1, -1) : logits.reshaped(-1, logits.dim(-1))
        let values = rows.asType(.float32)
        let logprobs = values - logSumExp(values, axis: -1, keepDims: true)

        if isGreedy {
            let tokens = argMax(logprobs, axis: -1).asType(.int32)
            let top1 = exp(logprobs.max(axis: -1))
            return (tokens, top1)
        }

        let (sortedIndices, sortedValues, kept) = candidates(logprobs: logprobs)
        let scaled = kept * (1 / temperature)

        let choice: MLXArray
        if let seed, let positions {
            let count = scaled.dim(0)
            precondition(positions.count >= count, "\(count) rows need \(count) positions, got \(positions.count).")
            let draws = (0 ..< count).map { row in
                MLXRandom.categorical(
                    scaled[row ..< (row + 1)], key: MLXRandom.key(seed &+ UInt64(positions.lowerBound + row)))
            }
            choice = count == 1 ? draws[0] : concatenated(draws, axis: 0)
        } else {
            choice = MLXRandom.categorical(scaled, key: randomState)
        }
        let tokens = takeAlong(sortedIndices, choice.asType(.int32).reshaped(-1, 1), axis: -1).reshaped(-1).asType(.int32)
        let top1 = exp(sortedValues[0..., 0])
        return (tokens, top1)
    }

    /// Each row's top k: vocabulary ids sorted by probability (descending), their
    /// log-probabilities, and the log-probabilities with `-inf` where top-p drops the entry
    /// (`[S, k]` each, lazy). `logits` is `[S, V]` or `[V]`.
    public func candidates(_ logits: MLXArray) -> (indices: MLXArray, logprobs: MLXArray, kept: MLXArray) {
        let rows = logits.ndim == 1 ? logits.reshaped(1, -1) : logits.reshaped(-1, logits.dim(-1))
        let values = rows.asType(.float32)
        return candidates(logprobs: values - logSumExp(values, axis: -1, keepDims: true))
    }

    private func candidates(logprobs: MLXArray) -> (indices: MLXArray, logprobs: MLXArray, kept: MLXArray) {
        let vocabularySize = logprobs.dim(-1)
        let k = topK > 0 ? min(topK, vocabularySize) : vocabularySize
        // The top k, unordered, then sorted descending.
        let unordered: MLXArray
        if k < vocabularySize {
            unordered = argPartition(-logprobs, kth: k - 1, axis: -1)[0..., ..<k]
        } else {
            unordered = argSort(-logprobs, axis: -1)
        }
        let unorderedValues = takeAlong(logprobs, unordered, axis: -1)
        let order = argSort(-unorderedValues, axis: -1)
        let indices = takeAlong(unordered, order, axis: -1)
        let sorted = takeAlong(unorderedValues, order, axis: -1)

        var kept = sorted
        if topP > 0 && topP < 1 {
            // Keep an entry while the probability ranked above it is below top-p.
            let above = cumsum(exp(sorted), axis: -1, inclusive: false)
            kept = which(above .< MLXArray(topP), sorted, MLXArray(-Float.infinity))
        }
        return (indices, sorted, kept)
    }

    /// The tokens a row may be sampled from and their probabilities after temperature, as the
    /// sampler computes them (for tests and diagnostics). Host values, sorted by probability.
    public func keptDistribution(_ logits: MLXArray) -> [(token: Int, probability: Double)] {
        let row = logits.reshaped(1, -1).asType(.float32)
        let logprobs = row - logSumExp(row, axis: -1, keepDims: true)
        if isGreedy {
            return [(argMax(logprobs, axis: -1).item(Int.self), 1)]
        }
        let (order, _, keptValues) = candidates(logprobs: logprobs)
        let values = keptValues.reshaped(-1).asArray(Float.self)
        let indices = order.reshaped(-1).asType(.int32).asArray(Int32.self)
        var kept: [(Int, Double)] = []
        for (index, value) in zip(indices, values) where value.isFinite {
            kept.append((Int(index), Double(value) / Double(temperature)))
        }
        let peak = kept.map(\.1).max() ?? 0
        let total = kept.reduce(0.0) { $0 + exp($1.1 - peak) }
        return kept.map { ($0.0, exp($0.1 - peak) / total) }
    }
}
