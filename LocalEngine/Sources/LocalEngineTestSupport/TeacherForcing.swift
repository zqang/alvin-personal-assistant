import Foundation
import MLX
import MLXLMCommon
#if canImport(XCTest)
import XCTest
#endif

/// Per-position greedy facts about a token sequence, from one teacher-forced pass: which token
/// the model ranks first and by how much (plan §6.4).
public enum TeacherForcing {
    public struct Step: Equatable, Sendable {
        /// The highest-ranked token at this position.
        public let argmax: Int
        /// Top-1 minus top-2 logit: how far this position is from a tie.
        public let margin: Float
    }

    /// Feeds `prompt + continuation` through `model` on a fresh cache and returns, for each
    /// continuation token, the step computed from the logits that predict it (row
    /// `prompt.count - 1 + i`). `prompt` must not be empty.
    public static func run(model: any LanguageModel, prompt: [Int], continuation: [Int], chunk: Int = 512) -> [Step] {
        precondition(!prompt.isEmpty, "Teacher forcing needs at least one prompt token.")
        let tokens = prompt + continuation
        let cache = model.newCache(parameters: nil)
        let first = prompt.count - 1
        let last = tokens.count - 1  // the last row predicts past the continuation; not needed
        var steps: [Step] = []
        var start = 0
        while start < last {
            let end = min(start + chunk, last)
            let rows = logits(model, Array(tokens[start ..< end]), cache: cache)
            // Row r of this chunk predicts tokens[start + r + 1].
            let lower = max(first, start)
            if lower < end {
                steps += self.steps(logits: rows[(lower - start) ..< (end - start)])
            }
            start = end
        }
        return steps
    }

    /// Argmax and top-1 minus top-2 margin for each row of `rows` (`[N, V]`).
    public static func steps(logits rows: MLXArray) -> [Step] {
        let values = rows.asType(.float32)
        let best = values.argMax(axis: -1)
        let topTwo = MLX.sorted(MLX.top(values, k: 2, axis: -1), axis: -1)
        let margin = topTwo[0..., 1] - topTwo[0..., 0]
        eval(best, margin)
        return zip(best.asArray(Int32.self), margin.asArray(Float.self)).map { Step(argmax: Int($0), margin: $1) }
    }

    private static func logits(_ model: any LanguageModel, _ tokens: [Int], cache: [KVCache]) -> MLXArray {
        let input = MLXArray(tokens.map { Int32($0) })[.newAxis]
        return model(input, cache: cache)[0]
    }
}

/// Token equality with the near-tie exemption of plan §6.4: at the first mismatch, the
/// reference run's top-1 minus top-2 margin decides. Below the tolerance the prompt counts as a
/// near-tie and comparison stops there; otherwise it fails. Too many near-ties also fail.
public struct NearTieTally: Sendable {
    /// Tiny float32 models.
    public static let float32Tolerance: Float = 1e-3
    /// Real 4-bit models. A reused cache and a fresh prefill reach the same position through
    /// differently shaped matmuls, so fp16 rounding can flip a close top-2: on the CI GPU, Qwen3-0.6B
    /// matched 56 positions and then flipped one with a 0.25 margin (Local engine run 37810247919).
    /// A cache bug shows up as early, repeated mismatches, which `maxNearTieRate` still fails.
    public static let quantizedTolerance: Float = 0.5
    /// The largest allowed share of near-ties among compared positions.
    public static let maxNearTieRate = 0.02

    public enum Outcome: Equatable, Sendable {
        case identical
        case nearTie(position: Int, margin: Float)
        case mismatch(position: Int, expected: Int?, actual: Int?, margin: Float?)
    }

    public let tolerance: Float
    public private(set) var compared = 0
    public private(set) var nearTies = 0
    public private(set) var prompts = 0
    public private(set) var failures: [String] = []

    public init(tolerance: Float) {
        self.tolerance = tolerance
    }

    /// Compares `candidate` with `reference` (the plain run) and records the result.
    /// `referenceMargins[i]` is the reference run's margin when choosing `reference[i]`
    /// (`TeacherForcing.run(...).map(\.margin)`).
    @discardableResult
    public mutating func record(_ label: String, reference: [Int], candidate: [Int], referenceMargins: [Float]) -> Outcome {
        let (outcome, count) = Self.compare(reference: reference, candidate: candidate, referenceMargins: referenceMargins, tolerance: tolerance)
        prompts += 1
        compared += count
        switch outcome {
        case .identical:
            break
        case .nearTie:
            nearTies += 1
        case .mismatch(let position, let expected, let actual, let margin):
            let marginText = margin.map { String($0) } ?? "unknown"
            let expectedText = expected.map { String($0) } ?? "end"
            let actualText = actual.map { String($0) } ?? "end"
            failures.append("\(label): position \(position) expected \(expectedText) got \(actualText), margin \(marginText) ≥ \(tolerance)")
        }
        return outcome
    }

    /// Near-ties as a share of compared positions.
    public var nearTieRate: Double {
        compared == 0 ? 0 : Double(nearTies) / Double(compared)
    }

    public var passed: Bool {
        failures.isEmpty && nearTieRate <= Self.maxNearTieRate
    }

    public var summary: String {
        var text = "\(prompts) prompts, \(compared) positions compared, \(nearTies) near-ties (\(String(format: "%.2f", nearTieRate * 100))%)"
        if !failures.isEmpty {
            text += "; failures: " + failures.joined(separator: "; ")
        } else if nearTieRate > Self.maxNearTieRate {
            text += "; too many near-ties (limit \(Self.maxNearTieRate * 100)%)"
        }
        return text
    }

    /// The outcome of one comparison and how many positions it compared (up to and including
    /// the first difference).
    public static func compare(reference: [Int], candidate: [Int], referenceMargins: [Float], tolerance: Float) -> (Outcome, Int) {
        let common = min(reference.count, candidate.count)
        var position = 0
        while position < common && reference[position] == candidate[position] {
            position += 1
        }
        if position == common && reference.count == candidate.count {
            return (.identical, common)
        }
        let margin = position < referenceMargins.count ? referenceMargins[position] : nil
        let expected = position < reference.count ? reference[position] : nil
        let actual = position < candidate.count ? candidate[position] : nil
        if let margin, margin < tolerance {
            return (.nearTie(position: position, margin: margin), position + 1)
        }
        return (.mismatch(position: position, expected: expected, actual: actual, margin: margin), position + 1)
    }

    #if canImport(XCTest)
    /// Fails the calling test unless the tally passed.
    public func assertPassed(file: StaticString = #filePath, line: UInt = #line) {
        if !passed {
            XCTFail("Near-tie comparison failed: \(summary)", file: file, line: line)
        }
    }
    #endif
}

/// Numeric comparisons for the "exact" and "allClose" checks of plan §6.4.
public enum LogitCheck {
    /// `max |a - b|` in float32. Shapes must match.
    public static func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        precondition(a.shape == b.shape, "Shapes differ: \(a.shape) vs \(b.shape)")
        return MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// Same shape, same dtype and bitwise-equal values (`max |Δ| == 0`).
    public static func isExactlyEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.shape == b.shape && a.dtype == b.dtype && maxAbsDifference(a, b) == 0
    }

    /// `|a - b| <= atol + rtol * |b|` everywhere.
    public static func isClose(_ a: MLXArray, _ b: MLXArray, rtol: Double = 1e-4, atol: Double = 1e-4) -> Bool {
        a.shape == b.shape && a.asType(.float32).allClose(b.asType(.float32), rtol: rtol, atol: atol).item(Bool.self)
    }
}
