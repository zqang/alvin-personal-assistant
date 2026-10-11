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
            LocalBenchmark.Row(prompt: "Hi", reply: "Hello", stats: engine, thermal: .nominal),
            LocalBenchmark.Row(prompt: "Hi", reply: "Hello", stats: stock, thermal: .serious, lowPowerMode: true),
            LocalBenchmark.Row(prompt: "Old", reply: "Row", stats: LocalGenerationStats()),
        ]
        let report = LocalBenchmark.report(model: "Woof", device: "iPhone17,1", speculative: "automatic", loadTime: 1, rows: rows, scenario: "Continued chat")
        XCTAssertTrue(report.contains("Scenario: Continued chat"))
        XCTAssertTrue(report.contains("turn | first text s | prefill tok (tok/s) | gen tok/s | draft accept | cache | peak GB | engine | plan | prefilled/reused | tok/round | acc% | thermal\n"))
        XCTAssertTrue(report.contains("1 | 0.12 | 48 (600) | 40.0 | – | reused | 2.60 | alvin | append | 48/812 | 2.50 | 38% | nominal\n"), report)
        XCTAssertTrue(report.contains("2 | 0.90 | 860 (1075) | 20.0 | – | rebuilt | – | stock | – | – | – | – | serious\n"), report)
        XCTAssertTrue(report.contains("3 | – | 0 (–) | – | – | rebuilt | – | – | – | – | – | – | –\n"), report)
        XCTAssertTrue(report.contains("Load time: 1.0 s\nLow Power Mode: on\n"), report)
        XCTAssertFalse(report.contains("Tool calls:"))
        XCTAssertFalse(report.contains("   tool:"))
    }

    func testReportSummarizesToolScores() {
        let expectation = ToolExpectation(name: "set_timer", arguments: ["seconds": 600])
        func row(_ prompt: String, reply: String = "", _ calls: [ToolCallRecord], stats: LocalGenerationStats = LocalGenerationStats(), scored: Bool = true) -> LocalBenchmark.Row {
            LocalBenchmark.Row(prompt: prompt, reply: reply, stats: stats, toolScore: scored ? expectation.score(calls: calls) : nil, calls: calls)
        }
        let rows = [
            row("Timer", [call("set_timer", ["seconds": 600])], stats: LocalGenerationStats(toolCallProbability: 0.984)),
            row("Timer", [call("set_timer", ["seconds": 60])]),
            row("Timer", reply: "Sure", [], stats: LocalGenerationStats(toolCallProbability: 0.004, claimRetries: 1)),
            // The expected call after a lookup, in a later round, passes.
            row("Timer", [call("get_current_time", [:]), call("set_timer", ["seconds": "600"])], stats: LocalGenerationStats(forcedToolCall: true)),
            // A turn without an expectation shows its calls unscored; one without calls shows none.
            row("Read it back", reply: "Eggs", [call("list_reminders", ["scope": "all"])], stats: LocalGenerationStats(toolCallProbability: 0.6), scored: false),
            row("Chat", reply: "Hi", [], scored: false),
        ]
        let report = LocalBenchmark.report(model: "Woof", device: "Mac", speculative: "off", loadTime: nil, rows: rows)
        XCTAssertTrue(report.contains("\n\nTool calls: name 3/4, arguments 2/4\n"), report)
        XCTAssertFalse(report.contains("Scenario:"))
        XCTAssertFalse(report.contains("Low Power Mode"))
        // Each scored turn says what it called, under its reply.
        XCTAssertTrue(report.contains("1. Timer\n   → \n   tool: set_timer {seconds: 600} — name ✓, args ✓ · P(tool) 0.98\n"), report)
        XCTAssertTrue(report.contains("2. Timer\n   → \n   tool: set_timer {seconds: 60} — name ✓, args ✗ (seconds)\n"), report)
        XCTAssertTrue(report.contains("3. Timer\n   → Sure\n   tool: none — expected set_timer · P(tool) 0.00, retried\n"), report)
        XCTAssertTrue(report.contains("4. Timer\n   → \n   tool: get_current_time → set_timer {seconds: 600} — name ✓, args ✓ · forced\n"), report)
        XCTAssertTrue(report.contains("5. Read it back\n   → Eggs\n   tool: list_reminders {scope: all} · P(tool) 0.60\n"), report)
        XCTAssertTrue(report.hasSuffix("6. Chat\n   → Hi"), report)
    }

    func testToolLine() {
        let expectation = ToolExpectation(name: "create_event", arguments: ["title": "dentist", "start": "2026-10-09T15:00"])
        func line(_ expectation: ToolExpectation, _ calls: [ToolCallRecord], stats: LocalGenerationStats = LocalGenerationStats()) -> String {
            LocalBenchmark.toolLine(calls: calls, score: expectation.score(calls: calls), stats: stats)
        }
        let long = String(repeating: "x", count: 40)
        XCTAssertEqual(
            line(expectation, [call("create_event", ["title": .string(long), "start": "2026-10-10T15:00", "all_day": false])]),
            "create_event {all_day: false, start: 2026-10-10T15:00, title: \(String(repeating: "x", count: 32))…} — name ✓, args ✗ (start, title)"
        )
        XCTAssertEqual(
            line(expectation, [call("create_reminder", ["title": "dentist"])]),
            "create_reminder {title: dentist} — name ✗ (expected create_event)"
        )
        XCTAssertEqual(line(ToolExpectation(name: "list_timers"), [call("list_timers", [:])]), "list_timers — name ✓, args ✓")
        // The whole sequence, with the expected call scored wherever it is.
        XCTAssertEqual(
            line(
                ToolExpectation(name: "create_event", arguments: ["title": "王经理", "start": "2026-10-12T10:00"]),
                [call("get_current_time", [:]), call("create_event", ["start": "2026-10-12T10:00:00", "title": "和王经理开会"])],
                stats: LocalGenerationStats(toolCallProbability: 0.77, forcedToolCall: true)),
            "get_current_time → create_event {start: 2026-10-12T10:00:00, title: 和王经理开会} — name ✓, args ✓ · P(tool) 0.77, forced"
        )
        XCTAssertEqual(
            line(expectation, [call("get_current_time", [:]), call("list_events", ["start": "2026-10-09"])]),
            "get_current_time → list_events {start: 2026-10-09} — name ✗ (expected create_event)"
        )
        // What the engine measured: the model's own odds of a call, and whether forcing or the
        // claim retry made it.
        let reminder = ToolExpectation(name: "create_reminder", arguments: ["title": "parcel"])
        XCTAssertEqual(
            line(reminder, [call("create_reminder", ["title": "parcel"])], stats: LocalGenerationStats(toolCallProbability: 0.031, forcedToolCall: true, claimRetries: 1)),
            "create_reminder {title: parcel} — name ✓, args ✓ · P(tool) 0.03, forced, retried"
        )
        XCTAssertEqual(
            line(reminder, [call("create_event", [:])], stats: LocalGenerationStats(forcedToolCall: true)),
            "create_event — name ✗ (expected create_reminder) · forced"
        )
        // Unscored: the calls alone.
        XCTAssertEqual(
            LocalBenchmark.toolLine(calls: [call("get_current_time", [:]), call("set_timer", ["seconds": 60])], score: nil),
            "get_current_time → set_timer {seconds: 60}"
        )
    }

    private func call(_ name: String, _ input: JSONValue) -> ToolCallRecord {
        ToolCallRecord(id: "call_\(name)", name: name, input: input, result: "{\"ok\":true}")
    }

    func testPrefilledReusedColumn() {
        XCTAssertEqual(LocalBenchmark.prefilledReused(LocalGenerationStats()), "–")
        XCTAssertEqual(LocalBenchmark.prefilledReused(LocalGenerationStats(prefilledTokens: 900)), "900/–")
        XCTAssertEqual(LocalBenchmark.prefilledReused(LocalGenerationStats(prefilledTokens: 12, reusedTokens: 0)), "12/0")
    }
}
