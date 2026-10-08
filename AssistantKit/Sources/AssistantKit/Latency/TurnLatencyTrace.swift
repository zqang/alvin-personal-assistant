import Foundation

/// Moments in one voice turn, from the end of the user's speech to the end of the reply.
public enum LatencyMark: String, Codable, CaseIterable, Sendable {
    /// The last change of the user's transcript.
    case speechEnded
    /// A tentative reply started before the turn was committed.
    case earlyStart
    /// The turn was committed.
    case committed
    /// The second, more accurate transcription finished.
    case transcriptRefined
    /// The reply request was sent.
    case requestStarted
    /// The orchestrator picked an engine.
    case routed
    case responseStarted
    /// The on-device engine finished reading the prompt.
    case prefillDone
    case firstToken
    case firstText
    case firstCue
    case firstAudio
    case finished
    case interrupted
}

/// Encodes `marks` as a JSON object keyed by mark name rather than as a flat array.
extension LatencyMark: CodingKeyRepresentable {}

/// When each moment of one voice turn happened, in seconds on one monotonic clock.
public struct TurnLatencyTrace: Codable, Equatable, Sendable {
    public var id: UUID
    /// What answered, e.g. "cloud" or "local"; reports group by it.
    public var engine: String?
    /// Whether the reply was a tentative one that was adopted; nil when none was started.
    public var adoptedEarlyStart: Bool?
    /// The first time each mark was reached. Later marks of the same kind are ignored.
    public private(set) var marks: [LatencyMark: TimeInterval] = [:]

    public init() {
        id = UUID()
    }

    /// Records `m` at `t` unless it was already recorded.
    public mutating func mark(_ m: LatencyMark, at t: TimeInterval) {
        if marks[m] == nil { marks[m] = t }
    }

    /// Seconds from `a` to `b`, or nil unless both were recorded.
    public func interval(_ a: LatencyMark, _ b: LatencyMark) -> TimeInterval? {
        guard let start = marks[a], let end = marks[b] else { return nil }
        return end - start
    }

    /// From the end of speech to the first sound the user heard: a cue or the answer.
    public var heardSomething: TimeInterval? {
        guard let start = marks[.speechEnded] else { return nil }
        guard let heard = [marks[.firstCue], marks[.firstAudio]].compactMap({ $0 }).min() else { return nil }
        return heard - start
    }

    /// From the end of speech to the first audio of the answer.
    public var answerLatency: TimeInterval? {
        interval(.speechEnded, .firstAudio)
    }
}

/// A plain-text latency summary of recent voice turns, for a debug screen.
public enum LatencyReport {
    /// p50 and p90 (nearest rank) of each measure, per engine in name order. A measure is left
    /// out for an engine with no turn that recorded it.
    public static func summary(_ traces: [TurnLatencyTrace]) -> String {
        guard !traces.isEmpty else { return "No voice turns recorded yet." }
        var lines = ["\(count(traces.count, "turn")), p50 / p90 in seconds"]
        let groups = Dictionary(grouping: traces) { $0.engine ?? "unknown" }
        for engine in groups.keys.sorted() {
            let group = groups[engine] ?? []
            lines.append("")
            lines.append("\(engine): \(count(group.count, "turn"))")
            for measure in measures {
                let values = group.compactMap(measure.value)
                guard !values.isEmpty else { continue }
                lines.append("  \(measure.name): \(format(percentile(values, 50))) / \(format(percentile(values, 90)))  (n=\(values.count))")
            }
            let early = group.compactMap(\.adoptedEarlyStart)
            if !early.isEmpty {
                lines.append("  early start adopted: \(early.filter { $0 }.count) of \(early.count)")
            }
            let interrupted = group.filter { $0.marks[.interrupted] != nil }.count
            if interrupted > 0 {
                lines.append("  interrupted: \(interrupted)")
            }
        }
        return lines.joined(separator: "\n")
    }

    /// The nearest-rank percentile of `values`, which must not be empty.
    static func percentile(_ values: [TimeInterval], _ percent: Int) -> TimeInterval {
        let sorted = values.sorted()
        let rank = (percent * sorted.count + 99) / 100
        return sorted[min(max(rank, 1), sorted.count) - 1]
    }

    private static let measures: [(name: String, value: @Sendable (TurnLatencyTrace) -> TimeInterval?)] = [
        ("speech end → first audio", { $0.answerLatency }),
        ("speech end → first sound", { $0.heardSomething }),
        ("speech end → commit", { $0.interval(.speechEnded, .committed) }),
        ("request → first text", { $0.interval(.requestStarted, .firstText) }),
    ]

    private static func format(_ seconds: TimeInterval) -> String {
        String(format: "%.2f", seconds)
    }

    private static func count(_ n: Int, _ noun: String) -> String {
        "\(n) \(noun)\(n == 1 ? "" : "s")"
    }
}
