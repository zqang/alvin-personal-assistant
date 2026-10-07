import Foundation

/// Measurements of one on-device reply.
public struct LocalGenerationStats: Equatable, Sendable {
    /// From the request to the first visible text, including any prefill.
    public var timeToFirstText: TimeInterval?
    public var promptTokens: Int
    public var promptTime: TimeInterval
    public var generatedTokens: Int
    public var generateTime: TimeInterval
    /// Whether the cached session was continued (only the new turn was prefilled).
    public var reusedSession: Bool
    public var draftTokens: Int?
    public var acceptedDraftTokens: Int?
    /// Peak MLX memory during the reply, in bytes.
    public var peakMemoryBytes: Int?

    public init(
        timeToFirstText: TimeInterval? = nil,
        promptTokens: Int = 0,
        promptTime: TimeInterval = 0,
        generatedTokens: Int = 0,
        generateTime: TimeInterval = 0,
        reusedSession: Bool = false,
        draftTokens: Int? = nil,
        acceptedDraftTokens: Int? = nil,
        peakMemoryBytes: Int? = nil
    ) {
        self.timeToFirstText = timeToFirstText
        self.promptTokens = promptTokens
        self.promptTime = promptTime
        self.generatedTokens = generatedTokens
        self.generateTime = generateTime
        self.reusedSession = reusedSession
        self.draftTokens = draftTokens
        self.acceptedDraftTokens = acceptedDraftTokens
        self.peakMemoryBytes = peakMemoryBytes
    }

    public var tokensPerSecond: Double? {
        generateTime > 0 ? Double(generatedTokens) / generateTime : nil
    }

    public var prefillTokensPerSecond: Double? {
        promptTime > 0 ? Double(promptTokens) / promptTime : nil
    }

    /// Share of drafted tokens the main model accepted, when speculative decoding ran.
    public var acceptanceRate: Double? {
        guard let draftTokens, let acceptedDraftTokens, draftTokens > 0 else { return nil }
        return Double(acceptedDraftTokens) / Double(draftTokens)
    }
}

/// The scripted conversation the on-device benchmark plays, and its report.
public enum LocalBenchmark {
    /// Voice-style turns in English and Chinese, played as one conversation so that later turns
    /// show the effect of reusing the cached session.
    public static let prompts = [
        "Hi! What can you help me with?",
        "Give me one tip for sleeping better.",
        "Why is the sky blue? Keep it short.",
        "And why are sunsets red then?",
        "用一句话介绍一下新加坡。",
        "我明天早上要早起，有什么建议吗？",
        "Write a two-line poem about coffee.",
        "Thanks, that's all for now.",
    ]

    public struct Row: Equatable, Sendable {
        public var prompt: String
        public var reply: String
        public var stats: LocalGenerationStats

        public init(prompt: String, reply: String, stats: LocalGenerationStats) {
            self.prompt = prompt
            self.reply = reply
            self.stats = stats
        }
    }

    /// A plain-text report to copy and share.
    public static func report(model: String, device: String, speculative: String, loadTime: TimeInterval?, rows: [Row]) -> String {
        var lines = [
            "On-device benchmark",
            "Model: \(model)",
            "Device: \(device)",
            "Speculative decoding: \(speculative)",
        ]
        if let loadTime { lines.append("Load time: \(format(loadTime, digits: 1)) s") }
        lines.append("")
        lines.append("turn | first text s | prefill tok (tok/s) | gen tok/s | draft accept | cache | peak GB")
        for (index, row) in rows.enumerated() {
            let stats = row.stats
            let columns = [
                "\(index + 1)",
                stats.timeToFirstText.map { format($0, digits: 2) } ?? "–",
                "\(stats.promptTokens) (\(stats.prefillTokensPerSecond.map { format($0, digits: 0) } ?? "–"))",
                stats.tokensPerSecond.map { format($0, digits: 1) } ?? "–",
                stats.acceptanceRate.map { "\(Int(($0 * 100).rounded()))%" } ?? "–",
                stats.reusedSession ? "reused" : "rebuilt",
                stats.peakMemoryBytes.map { format(Double($0) / 1e9, digits: 2) } ?? "–",
            ]
            lines.append(columns.joined(separator: " | "))
        }
        if let summary = summary(rows) {
            lines.append("")
            lines.append(summary)
        }
        lines.append("")
        for (index, row) in rows.enumerated() {
            lines.append("\(index + 1). \(row.prompt)")
            lines.append("   → \(row.reply.trimmed.replacingOccurrences(of: "\n", with: " "))")
        }
        return lines.joined(separator: "\n")
    }

    /// Medians across turns, which matter more than any single turn.
    static func summary(_ rows: [Row]) -> String? {
        let firstText = rows.compactMap(\.stats.timeToFirstText)
        let speed = rows.compactMap(\.stats.tokensPerSecond)
        guard !firstText.isEmpty || !speed.isEmpty else { return nil }
        var parts: [String] = []
        if let value = median(firstText) { parts.append("median first text \(format(value, digits: 2)) s") }
        if let value = median(speed) { parts.append("median \(format(value, digits: 1)) tok/s") }
        let reused = rows.filter(\.stats.reusedSession).compactMap(\.stats.timeToFirstText)
        let rebuilt = rows.filter { !$0.stats.reusedSession }.compactMap(\.stats.timeToFirstText)
        if let reused = median(reused), let rebuilt = median(rebuilt) {
            parts.append("first text \(format(reused, digits: 2)) s with the cache vs \(format(rebuilt, digits: 2)) s without")
        }
        return "Summary: " + parts.joined(separator: "; ")
    }

    static func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
    }

    static func format(_ value: Double, digits: Int) -> String {
        String(format: "%.\(digits)f", value)
    }
}
