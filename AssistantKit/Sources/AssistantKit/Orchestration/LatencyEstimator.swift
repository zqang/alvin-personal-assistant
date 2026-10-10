import Foundation

/// Learns how long each engine takes to show its first text, per engine and mode.
///
/// Each series keeps an exponentially weighted moving average (α = 0.3) and its last 20 samples for
/// a p90. A series answers from its first sample, and it keeps its history however long the gaps
/// between turns are, so an estimator persisted between launches helps from the first turn of a
/// session (for example, to play the filler cue early when the reply will be slow).
///
/// Each series also remembers when its newest sample arrived. Routing rule 8 ("the cloud is slow")
/// trusts only a recent cloud estimate, through `expectedFirstText(engine:mode:recordedWithin:now:)`
/// (see `RouteSignals.setExpectations(from:now:)`): while that rule keeps simple requests on device
/// the cloud gets few new samples, so an old "slow" verdict has to lapse there rather than hold.
public struct LatencyEstimator: Codable, Equatable, Sendable {
    /// Weight of the newest sample in the moving average.
    public static let smoothing = 0.3
    /// Samples kept for the p90.
    public static let windowSize = 20

    struct Series: Codable, Equatable, Sendable {
        /// The moving average, in seconds.
        var average: Double
        /// The newest samples, oldest first, at most `windowSize`.
        var recent: [Double]
        /// When the newest sample was recorded.
        var updated: Date
    }

    /// Keyed by `"engine.mode"`, e.g. `"cloud.standard"`.
    private(set) var series: [String: Series]

    public init() {
        series = [:]
    }

    /// Adds one measured time to first text, in seconds. Negative or non-finite values are ignored.
    public mutating func record(engine: ReplyEngine, mode: ReplyMode, firstText: TimeInterval, at now: Date = Date()) {
        guard firstText.isFinite, firstText >= 0 else { return }
        let key = Self.key(engine, mode)
        guard var current = series[key] else {
            series[key] = Series(average: firstText, recent: [firstText], updated: now)
            return
        }
        current.average = Self.smoothing * firstText + (1 - Self.smoothing) * current.average
        current.recent.append(firstText)
        if current.recent.count > Self.windowSize {
            current.recent.removeFirst(current.recent.count - Self.windowSize)
        }
        current.updated = max(current.updated, now)
        series[key] = current
    }

    /// The typical time to first text (the moving average), or nil before the first sample.
    public func expectedFirstText(engine: ReplyEngine, mode: ReplyMode) -> TimeInterval? {
        series[Self.key(engine, mode)]?.average
    }

    /// The moving average, but only while the newest sample is at most `maximumAge` seconds old.
    public func expectedFirstText(engine: ReplyEngine, mode: ReplyMode, recordedWithin maximumAge: TimeInterval, now: Date = Date()) -> TimeInterval? {
        guard let current = series[Self.key(engine, mode)], now.timeIntervalSince(current.updated) <= maximumAge else { return nil }
        return current.average
    }

    /// The 90th percentile (nearest rank) of the last 20 samples, or nil before the first sample.
    public func p90FirstText(engine: ReplyEngine, mode: ReplyMode) -> TimeInterval? {
        guard let recent = series[Self.key(engine, mode)]?.recent, !recent.isEmpty else { return nil }
        let sorted = recent.sorted()
        let rank = Int((0.9 * Double(sorted.count)).rounded(.up))
        return sorted[min(max(rank, 1), sorted.count) - 1]
    }

    /// Samples in the p90 window, at most `windowSize`.
    public func sampleCount(engine: ReplyEngine, mode: ReplyMode) -> Int {
        series[Self.key(engine, mode)]?.recent.count ?? 0
    }

    /// When the newest sample of a series was recorded, or nil before the first sample.
    public func lastRecorded(engine: ReplyEngine, mode: ReplyMode) -> Date? {
        series[Self.key(engine, mode)]?.updated
    }

    private static func key(_ engine: ReplyEngine, _ mode: ReplyMode) -> String {
        "\(engine.rawValue).\(mode.rawValue)"
    }

    private enum CodingKeys: String, CodingKey {
        case series
    }

    /// Missing or malformed data decodes as an empty estimator.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let decoded = (try? container.decodeIfPresent([String: Series].self, forKey: .series)) ?? [:]
        series = decoded.filter { !$0.value.recent.isEmpty && $0.value.average.isFinite }.mapValues { stored in
            var trimmed = stored
            trimmed.recent = Array(stored.recent.suffix(Self.windowSize))
            return trimmed
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(series, forKey: .series)
    }
}
