import XCTest
@testable import AssistantKit

final class TurnLatencyTraceTests: XCTestCase {
    private func makeTrace(_ engine: String?, adopted: Bool? = nil, _ marks: [LatencyMark: TimeInterval]) -> TurnLatencyTrace {
        var trace = TurnLatencyTrace()
        trace.engine = engine
        trace.adoptedEarlyStart = adopted
        for (mark, time) in marks { trace.mark(mark, at: time) }
        return trace
    }

    func testFirstWriteWins() {
        var trace = TurnLatencyTrace()
        trace.mark(.firstAudio, at: 2)
        trace.mark(.firstAudio, at: 3)
        trace.mark(.speechEnded, at: 1)
        XCTAssertEqual(trace.marks, [.firstAudio: 2, .speechEnded: 1])
    }

    func testIntervals() {
        let trace = makeTrace("cloud", [.speechEnded: 1, .committed: 1.9, .firstText: 3.2])
        XCTAssertEqual(trace.interval(.speechEnded, .committed) ?? 0, 0.9, accuracy: 1e-9)
        XCTAssertEqual(trace.interval(.committed, .firstText) ?? 0, 1.3, accuracy: 1e-9)
        XCTAssertEqual(trace.interval(.firstText, .speechEnded) ?? 0, -2.2, accuracy: 1e-9)
        XCTAssertNil(trace.interval(.speechEnded, .firstAudio))
        XCTAssertNil(trace.interval(.earlyStart, .committed))
    }

    func testHeardSomethingIsTheFirstCueOrAudio() {
        XCTAssertEqual(makeTrace(nil, [.speechEnded: 1, .firstCue: 1.8, .firstAudio: 3]).heardSomething ?? 0, 0.8, accuracy: 1e-9)
        XCTAssertEqual(makeTrace(nil, [.speechEnded: 1, .firstCue: 4, .firstAudio: 3]).heardSomething ?? 0, 2, accuracy: 1e-9)
        XCTAssertEqual(makeTrace(nil, [.speechEnded: 1, .firstAudio: 2.5]).heardSomething ?? 0, 1.5, accuracy: 1e-9)
        XCTAssertEqual(makeTrace(nil, [.speechEnded: 1, .firstCue: 1.5]).heardSomething ?? 0, 0.5, accuracy: 1e-9)
        XCTAssertNil(makeTrace(nil, [.speechEnded: 1, .finished: 4]).heardSomething)
        XCTAssertNil(makeTrace(nil, [.firstCue: 1.5, .firstAudio: 2]).heardSomething)
    }

    func testAnswerLatencyIsSpeechEndToFirstAudio() {
        let trace = makeTrace(nil, [.speechEnded: 1, .firstCue: 1.5, .firstAudio: 2.75])
        XCTAssertEqual(trace.answerLatency ?? 0, 1.75, accuracy: 1e-9)
        XCTAssertNil(makeTrace(nil, [.speechEnded: 1, .firstCue: 1.5]).answerLatency)
        XCTAssertNil(makeTrace(nil, [.firstAudio: 1]).answerLatency)
    }

    func testCodableRoundTripKeysMarksByName() throws {
        let trace = makeTrace("local", adopted: true, [.speechEnded: 1, .firstAudio: 1.5])
        let data = try JSONEncoder().encode(trace)
        XCTAssertEqual(try JSONDecoder().decode(TurnLatencyTrace.self, from: data), trace)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let marks = try XCTUnwrap(object["marks"] as? [String: Any])
        XCTAssertEqual(Set(marks.keys), ["speechEnded", "firstAudio"])
        XCTAssertEqual(object["engine"] as? String, "local")
        XCTAssertEqual(object["adoptedEarlyStart"] as? Bool, true)

        let empty = TurnLatencyTrace()
        XCTAssertEqual(try JSONDecoder().decode(TurnLatencyTrace.self, from: JSONEncoder().encode(empty)), empty)
        XCTAssertNotEqual(TurnLatencyTrace().id, TurnLatencyTrace().id)
    }

    func testMarksCoverTheTurn() {
        XCTAssertEqual(LatencyMark.allCases.map(\.rawValue), [
            "speechEnded", "earlyStart", "committed", "transcriptRefined", "requestStarted", "routed", "responseStarted",
            "prefillDone", "firstToken", "firstText", "firstCue", "firstAudio", "finished", "interrupted",
        ])
    }

    func testPercentilesUseTheNearestRank() {
        let values = (1...10).map(Double.init).shuffled()
        XCTAssertEqual(LatencyReport.percentile(values, 50), 5)
        XCTAssertEqual(LatencyReport.percentile(values, 90), 9)
        XCTAssertEqual(LatencyReport.percentile(values, 100), 10)
        XCTAssertEqual(LatencyReport.percentile([3], 50), 3)
        XCTAssertEqual(LatencyReport.percentile([3], 90), 3)
        XCTAssertEqual(LatencyReport.percentile([5, 1, 3], 50), 3)
        XCTAssertEqual(LatencyReport.percentile([5, 1, 3], 90), 5)
        XCTAssertEqual(LatencyReport.percentile([2, 1], 50), 1)
    }

    func testReportSummarizesEachEngine() {
        let traces = [
            makeTrace("cloud", adopted: false, [.speechEnded: 0, .committed: 0.9, .requestStarted: 0.95, .firstText: 2.0, .firstAudio: 2.2]),
            makeTrace("local", adopted: true, [.speechEnded: 0, .requestStarted: 0.2, .committed: 0.5, .firstText: 0.6, .firstAudio: 0.8]),
            makeTrace("cloud", adopted: true, [.speechEnded: 10, .firstCue: 10.5, .committed: 10.9, .requestStarted: 10.9, .firstText: 12.0, .firstAudio: 13.0]),
            makeTrace(nil, [.speechEnded: 5, .interrupted: 5.5]),
            makeTrace("cloud", [.speechEnded: 20, .committed: 20.9, .firstAudio: 21.4, .interrupted: 22]),
        ]
        XCTAssertEqual(LatencyReport.summary(traces), """
        5 turns, p50 / p90 in seconds

        cloud: 3 turns
          speech end → first audio: 2.20 / 3.00  (n=3)
          speech end → first sound: 1.40 / 2.20  (n=3)
          speech end → commit: 0.90 / 0.90  (n=3)
          request → first text: 1.05 / 1.10  (n=2)
          early start adopted: 1 of 2
          interrupted: 1

        local: 1 turn
          speech end → first audio: 0.80 / 0.80  (n=1)
          speech end → first sound: 0.80 / 0.80  (n=1)
          speech end → commit: 0.50 / 0.50  (n=1)
          request → first text: 0.40 / 0.40  (n=1)
          early start adopted: 1 of 1

        unknown: 1 turn
          interrupted: 1
        """)
    }

    func testReportWithoutTurns() {
        XCTAssertEqual(LatencyReport.summary([]), "No voice turns recorded yet.")
        XCTAssertEqual(LatencyReport.summary([TurnLatencyTrace()]), "1 turn, p50 / p90 in seconds\n\nunknown: 1 turn")
    }
}
