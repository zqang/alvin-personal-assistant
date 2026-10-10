import XCTest
@testable import AssistantKit

private typealias ToolExpectation = LocalBenchmark.ToolExpectation
private typealias ToolScore = LocalBenchmark.ToolScore

final class EngineTelemetryTests: XCTestCase {
    func testPhaseTimes() throws {
        let phases = EnginePhaseTimes(plan: 0.001, render: 0.004, rewind: 0.002, prefill: 0.020, firstToken: 0.008, start: "warm")
        XCTAssertEqual(phases.total, 0.035, accuracy: 1e-12)
        XCTAssertEqual(EnginePhaseTimes().total, 0)
        XCTAssertEqual(EnginePhaseTimes.start(for: .keep(10)), "warm")
        XCTAssertEqual(EnginePhaseTimes.start(for: .persistedPrefix), "disk")
        XCTAssertEqual(EnginePhaseTimes.start(for: .empty), "cold")
        XCTAssertEqual(try JSONDecoder().decode(EnginePhaseTimes.self, from: JSONEncoder().encode(phases)), phases)
    }

    func testSpeculationStats() throws {
        var stats = SpeculationStats()
        XCTAssertNil(stats.acceptanceRate)
        XCTAssertNil(stats.meanTokensPerRound)

        stats.recordRound(source: "promptLookup", drafted: 4, accepted: 4, emitted: 5)
        stats.recordRound(source: "promptLookup", drafted: 4, accepted: 1, emitted: 2)
        stats.recordRound(source: "exemplar", drafted: 3, accepted: 0, emitted: 1)
        stats.recordPlain(7)
        stats.recordPlain()

        XCTAssertEqual(stats.rounds, 3)
        XCTAssertEqual(stats.plainTokens, 8)
        XCTAssertEqual(stats.drafted, ["promptLookup": 8, "exemplar": 3])
        XCTAssertEqual(stats.accepted, ["promptLookup": 5, "exemplar": 0])
        XCTAssertEqual(stats.tokensPerRound, [5: 1, 2: 1, 1: 1])
        XCTAssertEqual(stats.totalDrafted, 11)
        XCTAssertEqual(stats.totalAccepted, 5)
        XCTAssertEqual(stats.roundTokens, 8)
        XCTAssertEqual(try XCTUnwrap(stats.acceptanceRate), 5.0 / 11.0, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(stats.meanTokensPerRound), 8.0 / 3.0, accuracy: 1e-12)
        XCTAssertEqual(try JSONDecoder().decode(SpeculationStats.self, from: JSONEncoder().encode(stats)), stats)
    }

    func testTokenConfidence() throws {
        let samples: [Float] = [0.9, 0.5, 1.0, 0.8, 0.95, 0.99, 0.7, 0.6, 0.85, 0.75]
        let confidence = try XCTUnwrap(TokenConfidence(top1: samples))
        XCTAssertEqual(confidence.tokens, 10)
        XCTAssertEqual(confidence.meanTop1, 0.804, accuracy: 1e-6)
        XCTAssertEqual(confidence.p10Top1, 0.5, accuracy: 1e-6)

        // Nearest rank: the 2nd smallest of 20, the only value of 1.
        let twenty = TokenConfidence(top1: (1 ... 20).map { Double($0) / 20 })
        XCTAssertEqual(twenty?.p10Top1, 0.1)
        XCTAssertEqual(TokenConfidence(top1: [0.25]), TokenConfidence(meanTop1: 0.25, p10Top1: 0.25, tokens: 1))
        XCTAssertNil(TokenConfidence(top1: [Double]()))
        XCTAssertEqual(TokenConfidence(top1: [Double.nan, 0.5])?.tokens, 1)
        XCTAssertEqual(try JSONDecoder().decode(TokenConfidence.self, from: JSONEncoder().encode(confidence)), confidence)
    }

    func testReportShowsTheEngineColumns() {
        var speculation = SpeculationStats()
        speculation.recordRound(source: "promptLookup", drafted: 4, accepted: 3, emitted: 4)
        speculation.recordRound(source: "promptLookup", drafted: 4, accepted: 0, emitted: 1)
        let engine = LocalGenerationStats(
            timeToFirstText: 0.12, promptTokens: 48, promptTime: 0.08, generatedTokens: 40, generateTime: 1,
            reusedSession: true, peakMemoryBytes: 2_600_000_000,
            engine: "alvin",
            phases: EnginePhaseTimes(prefill: 0.08, start: "warm"),
            prefilledTokens: 48, reusedTokens: 812, planReason: SessionPlan.Reason.append.rawValue,
            speculation: speculation,
            confidence: TokenConfidence(meanTop1: 0.9, p10Top1: 0.6, tokens: 40)
        )
        let stock = LocalGenerationStats(timeToFirstText: 0.9, promptTokens: 860, promptTime: 0.8, generatedTokens: 40, generateTime: 2, engine: "stock")
        let rows = [
            LocalBenchmark.Row(prompt: "Hi", reply: "Hello", stats: engine),
            LocalBenchmark.Row(prompt: "Hi", reply: "Hello", stats: stock),
            LocalBenchmark.Row(prompt: "Old", reply: "Row", stats: LocalGenerationStats()),
        ]
        let report = LocalBenchmark.report(model: "Woof", device: "iPhone17,1", speculative: "automatic", loadTime: nil, rows: rows, scenario: "Continued chat")
        XCTAssertTrue(report.contains("Scenario: Continued chat"))
        XCTAssertTrue(report.contains("turn | first text s | prefill tok (tok/s) | gen tok/s | draft accept | cache | peak GB | engine | plan | prefilled/reused | tok/round | acc%"))
        XCTAssertTrue(report.contains("1 | 0.12 | 48 (600) | 40.0 | – | reused | 2.60 | alvin | append | 48/812 | 2.50 | 38%"), report)
        XCTAssertTrue(report.contains("2 | 0.90 | 860 (1075) | 20.0 | – | rebuilt | – | stock | – | – | – | –"), report)
        XCTAssertTrue(report.contains("3 | – | 0 (–) | – | – | rebuilt | – | – | – | – | – | –"), report)
        XCTAssertFalse(report.contains("Tool calls:"))
    }

    func testReportSummarizesToolScores() {
        let expectation = ToolExpectation(name: "set_timer", arguments: ["seconds": 600])
        let rows = [
            LocalBenchmark.Row(prompt: "Timer", reply: "", stats: LocalGenerationStats(), toolScore: expectation.score(name: "set_timer", arguments: ["seconds": 600])),
            LocalBenchmark.Row(prompt: "Timer", reply: "", stats: LocalGenerationStats(), toolScore: expectation.score(name: "set_timer", arguments: ["seconds": 60])),
            LocalBenchmark.Row(prompt: "Timer", reply: "Sure", stats: LocalGenerationStats(), toolScore: expectation.score(name: nil, arguments: nil)),
            LocalBenchmark.Row(prompt: "Chat", reply: "Hi", stats: LocalGenerationStats()),
        ]
        let report = LocalBenchmark.report(model: "Woof", device: "Mac", speculative: "off", loadTime: nil, rows: rows)
        XCTAssertTrue(report.contains("\n\nTool calls: name 2/3, arguments 1/3\n"), report)
        XCTAssertFalse(report.contains("Scenario:"))
    }

    func testPrefilledReusedColumn() {
        XCTAssertEqual(LocalBenchmark.prefilledReused(LocalGenerationStats()), "–")
        XCTAssertEqual(LocalBenchmark.prefilledReused(LocalGenerationStats(prefilledTokens: 900)), "900/–")
        XCTAssertEqual(LocalBenchmark.prefilledReused(LocalGenerationStats(prefilledTokens: 12, reusedTokens: 0)), "12/0")
    }
}
