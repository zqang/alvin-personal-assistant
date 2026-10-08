import Foundation

/// Learns how long each engine takes to show its first text, per engine and mode.
///
/// Each series keeps an exponentially weighted moving average (α = 0.3) and its last 20 samples for
/// a p90. A series answers only once it has `minimumSamples` samples, and only while its newest
/// sample is at most `maximumAge` old. A sample that arrives after a longer gap starts the series
/// afresh. The age limit matters for routing: while the cloud looks slow, simple requests go on
/// device and the cloud gets no new samples, so an old verdict must expire rather than hold forever.
public struct LatencyEstimator: Codable, Equatable, Sendable {
    /// Weight of the newest sample in the moving average.
    public static let smoothing = 0.3
    /// Samples kept for the p90.
    public static let windowSize = 20
    /// Samples a series needs before it gives estimates.
    public static let minimumSamples = 3
    /// A series whose newest sample is older than this gives no estimates (15 minutes).
    public static let maximumAge: TimeInterval = 15 * 60

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
        guard var current = series[key], !Self.isStale(current, now: now) else {
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

    /// The typical time to first text (the moving average), or nil without enough recent samples.
    public func expectedFirstText(engine: ReplyEngine, mode: ReplyMode, now: Date = Date()) -> TimeInterval? {
        usable(engine, mode, now: now)?.average
    }

    /// The 90th percentile (nearest rank) of the last 20 samples, or nil without enough recent samples.
    public func p90FirstText(engine: ReplyEngine, mode: ReplyMode, now: Date = Date()) -> TimeInterval? {
        guard let current = usable(engine, mode, now: now) else { return nil }
        let sorted = current.recent.sorted()
        let rank = Int((0.9 * Double(sorted.count)).rounded(.up))
        return sorted[min(max(rank, 1), sorted.count) - 1]
    }

    /// Samples in the window of a series that isn't stale; 0 for a stale or empty one.
    public func sampleCount(engine: ReplyEngine, mode: ReplyMode, now: Date = Date()) -> Int {
        guard let current = series[Self.key(engine, mode)], !Self.isStale(current, now: now) else { return 0 }
        return current.recent.count
    }

    private func usable(_ engine: ReplyEngine, _ mode: ReplyMode, now: Date) -> Series? {
        guard let current = series[Self.key(engine, mode)], !Self.isStale(current, now: now),
              current.recent.count >= Self.minimumSamples else { return nil }
        return current
    }

    private static func key(_ engine: ReplyEngine, _ mode: ReplyMode) -> String {
        "\(engine.rawValue).\(mode.rawValue)"
    }

    private static func isStale(_ series: Series, now: Date) -> Bool {
        now.timeIntervalSince(series.updated) > maximumAge
    }

    private enum CodingKeys: String, CodingKey {
        case series
    }

    /// Missing or malformed data decodes as an empty estimator.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let decoded = (try? container.decodeIfPresent([String: Series].self, forKey: .series)) ?? [:]
        series = decoded.filter { !$0.value.recent.isEmpty && $0.value.average.isFinite }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(series, forKey: .series)
    }
}
