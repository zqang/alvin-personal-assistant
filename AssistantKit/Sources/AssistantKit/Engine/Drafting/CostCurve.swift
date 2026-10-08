import Foundation

/// What one target forward costs as a function of how many rows it processes. Speculation pays
/// only where verifying S = K + 1 rows costs much less than S single-row steps.
public struct CostCurve: Codable, Equatable, Sendable {
    /// Forward time (or any proportional cost) by row count.
    public var seconds: [Int: Double]

    public init(seconds: [Int: Double]) {
        self.seconds = seconds
    }

    /// Cost of a forward over `rows` rows relative to one row, so `relative(1) == 1`.
    ///
    /// Linear interpolation between measured row counts; beyond the last point it extrapolates from
    /// the last segment. Costs are relative to the first point (1 row when it was measured), and
    /// row counts at or below it cost 1. No row count costs less than 1, whatever measurement noise
    /// says. Points with non-positive or non-finite values, or fewer than 1 row, are ignored. With
    /// fewer than two usable points the cost is taken as linear (`rows`), which never favours
    /// speculation.
    public func relative(_ rows: Int) -> Double {
        let points = seconds
            .filter { $0.key >= 1 && $0.value.isFinite && $0.value > 0 }
            .sorted { $0.key < $1.key }
        guard points.count >= 2 else { return Double(max(rows, 1)) }
        let base = points[0].value
        let x = Double(rows)
        if rows <= points[0].key { return 1 }
        for (lower, upper) in zip(points, points.dropFirst()) where rows <= upper.key {
            let t = (x - Double(lower.key)) / Double(upper.key - lower.key)
            return max(1, (lower.value + (upper.value - lower.value) * t) / base)
        }
        let lower = points[points.count - 2], upper = points[points.count - 1]
        let slope = (upper.value - lower.value) / Double(upper.key - lower.key)
        return max(1, (upper.value + slope * (x - Double(upper.key))) / base)
    }

    /// The curve of the median of each row count's samples. Row counts without finite samples are
    /// left out.
    public static func fromSamples(_ samples: [Int: [Double]]) -> CostCurve {
        var seconds: [Int: Double] = [:]
        for (rows, values) in samples {
            let sorted = values.filter(\.isFinite).sorted()
            guard !sorted.isEmpty else { continue }
            let middle = sorted.count / 2
            seconds[rows] = sorted.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
        }
        return CostCurve(seconds: seconds)
    }

    /// Stock MLX kernels on M-series GPUs (plan §4.7), until the on-device cost probe replaces it.
    public static let stockMLXDefault = CostCurve(seconds: [
        1: 1.0, 2: 1.05, 3: 1.35, 4: 1.63, 5: 1.95, 6: 2.3, 8: 3.07, 9: 3.4,
    ])
}
