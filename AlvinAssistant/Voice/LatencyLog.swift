import AssistantKit
import Foundation
import Observation

/// The latency traces of the most recent voice turns, oldest first, for the latency report.
/// Kept across launches, so a report can cover several voice sessions.
@MainActor
@Observable
final class LatencyLog {
    static let shared = LatencyLog()
    /// Traces kept; older ones are dropped.
    static let capacity = 50

    private(set) var traces: [TurnLatencyTrace]

    private let defaults: UserDefaults
    private static let storageKey = "assistant.latencyTraces"
    /// Turns `expectedFirstText()` looks back over.
    private static let expectationWindow = 10

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        if let data = defaults.data(forKey: Self.storageKey),
           let saved = try? JSONDecoder().decode([TurnLatencyTrace].self, from: data) {
            traces = Array(saved.suffix(Self.capacity))
        } else {
            traces = []
        }
    }

    func append(_ trace: TurnLatencyTrace) {
        traces.append(trace)
        if traces.count > Self.capacity {
            traces.removeFirst(traces.count - Self.capacity)
        }
        save()
    }

    func clear() {
        traces.removeAll()
        save()
    }

    /// p50 and p90 of the recorded turns, per engine.
    func summary() -> String {
        LatencyReport.summary(traces)
    }

    /// The median time from sending a request to its first text over the last few turns, or nil
    /// before any turn got that far. Lets the filler cue play early when replies have been slow.
    func expectedFirstText() -> TimeInterval? {
        let recent = traces.suffix(Self.expectationWindow).compactMap { $0.interval(.requestStarted, .firstText) }
        guard !recent.isEmpty else { return nil }
        return recent.sorted()[recent.count / 2]
    }

    private func save() {
        if let data = try? JSONEncoder().encode(traces) {
            defaults.set(data, forKey: Self.storageKey)
        }
    }
}
