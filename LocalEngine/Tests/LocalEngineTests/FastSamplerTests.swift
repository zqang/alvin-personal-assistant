import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLMCommon
import XCTest

/// `FastSampler`: the kept set equals the stock filter chain's, greedy is argmax, samples follow
/// the kept distribution, and seeded sampling is keyed by position.
final class FastSamplerTests: XCTestCase {
    private static let temperature: Float = 0.7
    private static let topP: Float = 0.8
    private static let topK = 20

    override func setUpWithError() throws {
        try MetalAvailability.require()
    }

    private var sampler: FastSampler {
        FastSampler(temperature: Self.temperature, topP: Self.topP, topK: Self.topK)
    }

    /// Random rows whose spread varies, so nuclei range from one token to more than k.
    private static func randomRows(_ count: Int, vocabulary: Int, seed: UInt64) -> MLXArray {
        let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 2)
        let scale = MLXRandom.uniform(Float(0.5) ..< Float(6), [count, 1], key: keys[0])
        return MLXRandom.normal([count, vocabulary], key: keys[1]) * scale
    }

    /// The stock chain, as `TopPSampler` applies it (F2): log-softmax over the vocabulary, then
    /// top-p by `argSort` / `cumsum` (`cumulative > 1 − p` in ascending order), then a top-k mask.
    private static func stockKept(_ logits: MLXArray) -> MLXArray {
        let values = logits.asType(.float32)
        let logprobs = values - logSumExp(values, axis: -1, keepDims: true)
        let negativeInfinity = MLXArray(-Float.infinity)
        let sortedIndices = argSort(logprobs, axis: -1)
        let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
        let cumulative = cumsum(exp(sortedLogprobs), axis: -1)
        let filtered = which(cumulative .> MLXArray(1 - topP), sortedLogprobs, negativeInfinity)
        var kept = putAlong(logprobs, sortedIndices, values: filtered, axis: -1)
        let vocabulary = kept.dim(-1)
        if topK < vocabulary {
            let masked = argPartition(-kept, kth: topK - 1, axis: -1)[0..., topK...]
            kept = putAlong(kept, masked, values: negativeInfinity, axis: -1)
        }
        return kept
    }

    func testKeptSetMatchesTheStockFilterChain() throws {
        var borderline = 0
        var rowsChecked = 0
        for (vocabulary, seed) in [(128, UInt64(1)), (4096, UInt64(2))] {
            let rows = Self.randomRows(1_000, vocabulary: vocabulary, seed: seed)
            let (indices, _, kept) = sampler.candidates(rows)
            let stock = Self.stockKept(rows)
            let probabilities = softmax(rows.asType(.float32), axis: -1)
            eval(indices, kept, stock, probabilities)

            let k = Self.topK
            let fastIndices = indices.asType(.int32).asArray(Int32.self)
            let fastKept = kept.asArray(Float.self)
            let stockValues = stock.asArray(Float.self)
            let rowProbabilities = probabilities.asArray(Float.self)
            for row in 0 ..< 1_000 {
                rowsChecked += 1
                var fast = Set<Int>()
                for column in 0 ..< k where fastKept[row * k + column].isFinite {
                    fast.insert(Int(fastIndices[row * k + column]))
                }
                var reference = Set<Int>()
                for token in 0 ..< vocabulary where stockValues[row * vocabulary + token].isFinite {
                    reference.insert(token)
                }
                if fast == reference { continue }
                // Allowed only at the top-p boundary: the probability ranked above each differing
                // token is within float32 rounding of p.
                let p = (0 ..< vocabulary).map { Double(rowProbabilities[row * vocabulary + $0]) }
                for token in fast.symmetricDifference(reference) {
                    let above = p.enumerated().filter { $0.element > p[token] }.reduce(0.0) { $0 + $1.element }
                    XCTAssertEqual(above, Double(Self.topP), accuracy: 1e-4, "V \(vocabulary) row \(row): token \(token) differs")
                }
                borderline += 1
            }
        }
        XCTAssertLessThanOrEqual(borderline, rowsChecked / 100, "\(borderline) borderline rows")
        EngineReport.append("- FastSampler kept set vs stock chain: \(rowsChecked) rows, \(borderline) borderline")
    }

    func testGreedyIsArgmax() throws {
        let rows = Self.randomRows(256, vocabulary: 4096, seed: 3)
        let greedy = FastSampler(temperature: 0, topP: Self.topP, topK: Self.topK)
        let (tokens, top1) = greedy.sample(rows, positions: 0 ..< 256)
        let expected = argMax(rows, axis: -1).asType(.int32)
        let expectedTop1 = softmax(rows.asType(.float32), axis: -1).max(axis: -1)
        eval(tokens, top1, expected, expectedTop1)
        XCTAssertEqual(tokens.asArray(Int32.self), expected.asArray(Int32.self))
        XCTAssertTrue(LogitCheck.isClose(top1, expectedTop1, rtol: 1e-5, atol: 1e-6))

        // A request can ask for greedy decoding whatever the temperature.
        var configuration = EngineConfiguration()
        configuration.temperature = 0.7
        XCTAssertTrue(FastSampler(configuration: configuration, greedy: true).isGreedy)
        XCTAssertFalse(FastSampler(configuration: configuration, greedy: false).isGreedy)
    }

    /// 50k samples of one row follow the kept distribution (chi-square, p > 0.001), and never
    /// leave the kept set.
    func testSamplesFollowTheKeptDistribution() throws {
        let row = Self.randomRows(1, vocabulary: 128, seed: 11) * 0.6
        let distribution = sampler.keptDistribution(row)
        XCTAssertGreaterThan(distribution.count, 2, "pick a row with a wider nucleus")
        let samples = 50_000
        let tiled = MLX.broadcast(row, to: [samples, 128])
        let (tokens, _) = sampler.sample(tiled, positions: nil)
        eval(tokens)

        var counts: [Int: Int] = [:]
        for token in tokens.asArray(Int32.self) {
            counts[Int(token), default: 0] += 1
        }
        let allowed = Set(distribution.map(\.token))
        XCTAssertTrue(Set(counts.keys).isSubset(of: allowed), "sampled outside the kept set: \(Set(counts.keys).subtracting(allowed))")

        let (statistic, degrees) = Self.chiSquare(
            expected: distribution.map { ($0.token, $0.probability * Double(samples)) }, counts: counts)
        let pValue = Self.chiSquarePValue(statistic, degreesOfFreedom: degrees)
        XCTAssertGreaterThan(pValue, 0.001, "chi-square \(statistic) on \(degrees) degrees of freedom")
        EngineReport.append("- FastSampler chi-square: \(String(format: "%.2f", statistic)) on \(degrees) df, p = \(String(format: "%.3f", pValue))")
    }

    func testSeededSamplingIsKeyedByPosition() throws {
        let rows = Self.randomRows(3, vocabulary: 128, seed: 21) * 0.5
        let seeded = FastSampler(temperature: Self.temperature, topP: Self.topP, topK: Self.topK, seed: 7)
        let (first, _) = seeded.sample(rows, positions: 10 ..< 13)
        let (second, _) = seeded.sample(rows, positions: 10 ..< 13)
        let oneByOne = (0 ..< 3).map { row in
            seeded.sample(rows[row ..< (row + 1)], positions: (10 + row) ..< (11 + row)).tokens
        }
        eval(first, second)
        eval(oneByOne)
        XCTAssertEqual(first.asArray(Int32.self), second.asArray(Int32.self))
        XCTAssertEqual(first.asArray(Int32.self), oneByOne.map { $0.item(Int32.self) })

        // Other positions draw with other keys.
        let wide = MLX.broadcast(rows[0 ..< 1], to: [64, 128])
        let (atZero, _) = seeded.sample(wide, positions: 0 ..< 64)
        let (atHundred, _) = seeded.sample(wide, positions: 100 ..< 164)
        eval(atZero, atHundred)
        XCTAssertNotEqual(atZero.asArray(Int32.self), atHundred.asArray(Int32.self))
    }

    // MARK: Statistics

    /// Pearson's statistic, merging bins expected below 5 into one.
    private static func chiSquare(expected: [(Int, Double)], counts: [Int: Int]) -> (statistic: Double, degrees: Int) {
        var bins: [(expected: Double, observed: Double)] = []
        var small = (expected: 0.0, observed: 0.0)
        for (token, value) in expected {
            let observed = Double(counts[token] ?? 0)
            if value < 5 {
                small.expected += value
                small.observed += observed
            } else {
                bins.append((value, observed))
            }
        }
        if small.expected > 0 { bins.append(small) }
        let statistic = bins.reduce(0.0) { $0 + pow($1.observed - $1.expected, 2) / $1.expected }
        return (statistic, max(1, bins.count - 1))
    }

    /// P(X ≥ statistic) for a chi-square distribution: the regularized upper incomplete gamma
    /// Q(k/2, x/2) (series below a + 1, continued fraction above).
    static func chiSquarePValue(_ statistic: Double, degreesOfFreedom: Int) -> Double {
        let a = Double(degreesOfFreedom) / 2
        let x = statistic / 2
        guard x > 0 else { return 1 }
        let logGammaA = logGamma(a)
        if x < a + 1 {
            var term = 1 / a
            var sum = term
            var n = a
            for _ in 0 ..< 500 {
                n += 1
                term *= x / n
                sum += term
                if abs(term) < abs(sum) * 1e-15 { break }
            }
            return 1 - sum * exp(-x + a * log(x) - logGammaA)
        }
        var b = x + 1 - a
        var c = 1 / 1e-300
        var d = 1 / b
        var h = d
        for i in 1 ..< 500 {
            let an = -Double(i) * (Double(i) - a)
            b += 2
            d = an * d + b
            if abs(d) < 1e-300 { d = 1e-300 }
            c = b + an / c
            if abs(c) < 1e-300 { c = 1e-300 }
            d = 1 / d
            let delta = d * c
            h *= delta
            if abs(delta - 1) < 1e-15 { break }
        }
        return exp(-x + a * log(x) - logGammaA) * h
    }

    /// ln Γ(x) for x > 0 (Lanczos, g = 7, nine coefficients; about 15 significant digits).
    static func logGamma(_ x: Double) -> Double {
        let coefficients = [
            0.999_999_999_999_809_93, 676.520_368_121_885_1, -1_259.139_216_722_402_8, 771.323_428_777_653_13,
            -176.615_029_162_140_59, 12.507_343_278_686_905, -0.138_571_095_265_720_12, 9.984_369_578_019_571_6e-6,
            1.505_632_735_149_311_6e-7,
        ]
        if x < 0.5 {
            return log(Double.pi / abs(sin(Double.pi * x))) - logGamma(1 - x)
        }
        let z = x - 1
        var sum = coefficients[0]
        for index in 1 ..< coefficients.count {
            sum += coefficients[index] / (z + Double(index))
        }
        let t = z + 7.5
        return 0.5 * log(2 * Double.pi) + (z + 0.5) * log(t) - t + log(sum)
    }

    func testChiSquarePValueHelper() {
        XCTAssertEqual(Self.logGamma(1), 0, accuracy: 1e-12)
        XCTAssertEqual(Self.logGamma(5), log(24), accuracy: 1e-12)
        XCTAssertEqual(Self.logGamma(0.5), 0.5 * log(Double.pi), accuracy: 1e-12)
        // Known values: P(χ²₁ ≥ 3.841) ≈ 0.05, P(χ²₁₀ ≥ 18.307) ≈ 0.05, P(χ²₄ ≥ 0) = 1.
        XCTAssertEqual(Self.chiSquarePValue(3.841, degreesOfFreedom: 1), 0.05, accuracy: 1e-3)
        XCTAssertEqual(Self.chiSquarePValue(18.307, degreesOfFreedom: 10), 0.05, accuracy: 1e-3)
        XCTAssertEqual(Self.chiSquarePValue(0, degreesOfFreedom: 4), 1, accuracy: 1e-12)
        XCTAssertEqual(Self.chiSquarePValue(29.588, degreesOfFreedom: 10), 0.001, accuracy: 1e-4)
    }
}
