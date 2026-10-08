import XCTest
@testable import AssistantKit

final class LatencyEstimatorTests: XCTestCase {
    private let start = Date(timeIntervalSince1970: 1_790_000_000)

    private func at(_ seconds: TimeInterval) -> Date {
        start.addingTimeInterval(seconds)
    }

    /// The moving average of equal samples equals them only up to rounding.
    private func assertNear(_ value: TimeInterval?, _ expected: TimeInterval, file: StaticString = #filePath, line: UInt = #line) {
        guard let value else { return XCTFail("No estimate", file: file, line: line) }
        XCTAssertEqual(value, expected, accuracy: 1e-9, file: file, line: line)
    }

    func testTheFirstSampleGivesAnEstimate() {
        var estimator = LatencyEstimator()
        XCTAssertNil(estimator.expectedFirstText(engine: .cloud, mode: .standard))
        XCTAssertNil(estimator.p90FirstText(engine: .cloud, mode: .standard))
        XCTAssertNil(estimator.lastRecorded(engine: .cloud, mode: .standard))
        XCTAssertEqual(estimator.sampleCount(engine: .cloud, mode: .standard), 0)

        estimator.record(engine: .cloud, mode: .standard, firstText: 2.5, at: start)
        XCTAssertEqual(estimator.expectedFirstText(engine: .cloud, mode: .standard), 2.5)
        XCTAssertEqual(estimator.p90FirstText(engine: .cloud, mode: .standard), 2.5)
        XCTAssertEqual(estimator.lastRecorded(engine: .cloud, mode: .standard), start)
        XCTAssertEqual(estimator.sampleCount(engine: .cloud, mode: .standard), 1)
    }

    func testMovingAverageUsesAlphaPointThree() throws {
        var estimator = LatencyEstimator()
        for (index, sample) in [1.0, 2.0, 4.0, 3.0].enumerated() {
            estimator.record(engine: .cloud, mode: .standard, firstText: sample, at: at(Double(index)))
        }
        // 1 → 0.3·2 + 0.7·1 = 1.3 → 0.3·4 + 0.7·1.3 = 2.11 → 0.3·3 + 0.7·2.11 = 2.377
        let expected = try XCTUnwrap(estimator.expectedFirstText(engine: .cloud, mode: .standard))
        XCTAssertEqual(expected, 2.377, accuracy: 1e-9)
        XCTAssertEqual(LatencyEstimator.smoothing, 0.3)
    }

    func testP90IsTheNearestRankOverTheLastTwentySamples() throws {
        var estimator = LatencyEstimator()
        for value in 1...25 {
            estimator.record(engine: .local, mode: .standard, firstText: Double(value), at: at(Double(value)))
        }
        XCTAssertEqual(estimator.sampleCount(engine: .local, mode: .standard), 20)
        // The window holds 6...25; the 18th smallest of 20 is 23.
        XCTAssertEqual(estimator.p90FirstText(engine: .local, mode: .standard), 23)

        var small = LatencyEstimator()
        for (index, value) in [0.5, 0.9, 0.7].enumerated() {
            small.record(engine: .local, mode: .standard, firstText: value, at: at(Double(index)))
        }
        XCTAssertEqual(small.p90FirstText(engine: .local, mode: .standard), 0.9)
    }

    func testSeriesAreSeparatePerEngineAndMode() {
        var estimator = LatencyEstimator()
        for index in 0..<3 {
            estimator.record(engine: .cloud, mode: .standard, firstText: 1.5, at: at(Double(index)))
            estimator.record(engine: .cloud, mode: .deep, firstText: 9, at: at(Double(index)))
            estimator.record(engine: .local, mode: .standard, firstText: 0.4, at: at(Double(index)))
        }
        assertNear(estimator.expectedFirstText(engine: .cloud, mode: .standard), 1.5)
        assertNear(estimator.expectedFirstText(engine: .cloud, mode: .deep), 9)
        assertNear(estimator.expectedFirstText(engine: .local, mode: .standard), 0.4)
        XCTAssertNil(estimator.expectedFirstText(engine: .local, mode: .deep))
    }

    func testHistorySurvivesLongGaps() {
        var estimator = LatencyEstimator()
        for index in 0..<3 {
            estimator.record(engine: .cloud, mode: .standard, firstText: 6, at: at(Double(index)))
        }
        // A day later the estimate is still there, and the next sample averages with the old ones.
        let nextDay = at(86_400)
        assertNear(estimator.expectedFirstText(engine: .cloud, mode: .standard), 6)
        estimator.record(engine: .cloud, mode: .standard, firstText: 1, at: nextDay)
        assertNear(estimator.expectedFirstText(engine: .cloud, mode: .standard), 0.3 * 1 + 0.7 * 6)
        XCTAssertEqual(estimator.sampleCount(engine: .cloud, mode: .standard), 4)
        XCTAssertEqual(estimator.p90FirstText(engine: .cloud, mode: .standard), 6)
        XCTAssertEqual(estimator.lastRecorded(engine: .cloud, mode: .standard), nextDay)
    }

    func testRecentEstimate() {
        var estimator = LatencyEstimator()
        estimator.record(engine: .cloud, mode: .standard, firstText: 5, at: at(0))
        estimator.record(engine: .cloud, mode: .standard, firstText: 5, at: at(10))
        XCTAssertEqual(estimator.expectedFirstText(engine: .cloud, mode: .standard, recordedWithin: 60, now: at(70)), 5)
        XCTAssertNil(estimator.expectedFirstText(engine: .cloud, mode: .standard, recordedWithin: 60, now: at(71)))
        XCTAssertNil(estimator.expectedFirstText(engine: .local, mode: .standard, recordedWithin: 60, now: at(10)))
        // Age is measured from the newest sample; a sample with an older timestamp doesn't move it back.
        estimator.record(engine: .cloud, mode: .standard, firstText: 5, at: at(5))
        XCTAssertEqual(estimator.lastRecorded(engine: .cloud, mode: .standard), at(10))
        // The plain estimate doesn't age.
        XCTAssertEqual(estimator.expectedFirstText(engine: .cloud, mode: .standard), 5)
    }

    func testInvalidSamplesAreIgnored() {
        var estimator = LatencyEstimator()
        for value in [-1, Double.nan, Double.infinity] {
            estimator.record(engine: .cloud, mode: .standard, firstText: value, at: start)
        }
        XCTAssertEqual(estimator, LatencyEstimator())
    }

    func testDefaultClockIsNow() {
        var estimator = LatencyEstimator()
        estimator.record(engine: .cloud, mode: .standard, firstText: 4)
        XCTAssertEqual(estimator.expectedFirstText(engine: .cloud, mode: .standard, recordedWithin: 60), 4)
    }

    func testCodableRoundTripAndTolerantDecoding() throws {
        var estimator = LatencyEstimator()
        for index in 0..<4 {
            estimator.record(engine: .cloud, mode: .standard, firstText: 1 + Double(index), at: at(Double(index)))
            estimator.record(engine: .local, mode: .standard, firstText: 0.5, at: at(Double(index)))
        }
        let data = try JSONEncoder().encode(estimator)
        let decoded = try JSONDecoder().decode(LatencyEstimator.self, from: data)
        XCTAssertEqual(decoded, estimator)
        XCTAssertEqual(decoded.p90FirstText(engine: .cloud, mode: .standard), 4)
        XCTAssertEqual(decoded.lastRecorded(engine: .cloud, mode: .standard), at(3))

        XCTAssertEqual(try JSONDecoder().decode(LatencyEstimator.self, from: Data("{}".utf8)), LatencyEstimator())
        let samples = (1...25).map(String.init).joined(separator: ",")
        let oversized = #"{"series":{"cloud.standard":{"average":2,"recent":["# + samples + #"],"updated":0}}}"#
        let trimmed = try JSONDecoder().decode(LatencyEstimator.self, from: Data(oversized.utf8))
        XCTAssertEqual(trimmed.sampleCount(engine: .cloud, mode: .standard), 20, "a stored window is cut to the newest 20")
        XCTAssertEqual(trimmed.p90FirstText(engine: .cloud, mode: .standard), 23)
        XCTAssertEqual(try JSONDecoder().decode(LatencyEstimator.self, from: Data(#"{"series":[1,2]}"#.utf8)), LatencyEstimator())
    }
}
