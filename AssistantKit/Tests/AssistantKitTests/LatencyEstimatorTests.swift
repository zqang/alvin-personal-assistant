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

    func testNoEstimateUntilEnoughSamples() {
        var estimator = LatencyEstimator()
        XCTAssertNil(estimator.expectedFirstText(engine: .cloud, mode: .standard, now: start))
        XCTAssertNil(estimator.p90FirstText(engine: .cloud, mode: .standard, now: start))

        estimator.record(engine: .cloud, mode: .standard, firstText: 2, at: start)
        estimator.record(engine: .cloud, mode: .standard, firstText: 2, at: at(1))
        XCTAssertEqual(estimator.sampleCount(engine: .cloud, mode: .standard, now: at(1)), 2)
        XCTAssertNil(estimator.expectedFirstText(engine: .cloud, mode: .standard, now: at(1)))

        estimator.record(engine: .cloud, mode: .standard, firstText: 2, at: at(2))
        assertNear(estimator.expectedFirstText(engine: .cloud, mode: .standard, now: at(2)), 2)
        XCTAssertEqual(LatencyEstimator.minimumSamples, 3)
    }

    func testMovingAverageUsesAlphaPointThree() throws {
        var estimator = LatencyEstimator()
        for (index, sample) in [1.0, 2.0, 4.0, 3.0].enumerated() {
            estimator.record(engine: .cloud, mode: .standard, firstText: sample, at: at(Double(index)))
        }
        // 1 → 0.3·2 + 0.7·1 = 1.3 → 0.3·4 + 0.7·1.3 = 2.11 → 0.3·3 + 0.7·2.11 = 2.377
        let expected = try XCTUnwrap(estimator.expectedFirstText(engine: .cloud, mode: .standard, now: at(3)))
        XCTAssertEqual(expected, 2.377, accuracy: 1e-9)
        XCTAssertEqual(LatencyEstimator.smoothing, 0.3)
    }

    func testP90IsTheNearestRankOverTheLastTwentySamples() throws {
        var estimator = LatencyEstimator()
        for value in 1...25 {
            estimator.record(engine: .local, mode: .standard, firstText: Double(value), at: at(Double(value)))
        }
        XCTAssertEqual(estimator.sampleCount(engine: .local, mode: .standard, now: at(25)), 20)
        // The window holds 6...25; the 18th smallest of 20 is 23.
        XCTAssertEqual(estimator.p90FirstText(engine: .local, mode: .standard, now: at(25)), 23)

        var small = LatencyEstimator()
        for (index, value) in [0.5, 0.9, 0.7].enumerated() {
            small.record(engine: .local, mode: .standard, firstText: value, at: at(Double(index)))
        }
        XCTAssertEqual(small.p90FirstText(engine: .local, mode: .standard, now: at(3)), 0.9)
    }

    func testSeriesAreSeparatePerEngineAndMode() {
        var estimator = LatencyEstimator()
        for index in 0..<3 {
            estimator.record(engine: .cloud, mode: .standard, firstText: 1.5, at: at(Double(index)))
            estimator.record(engine: .cloud, mode: .deep, firstText: 9, at: at(Double(index)))
            estimator.record(engine: .local, mode: .standard, firstText: 0.4, at: at(Double(index)))
        }
        assertNear(estimator.expectedFirstText(engine: .cloud, mode: .standard, now: at(3)), 1.5)
        assertNear(estimator.expectedFirstText(engine: .cloud, mode: .deep, now: at(3)), 9)
        assertNear(estimator.expectedFirstText(engine: .local, mode: .standard, now: at(3)), 0.4)
        XCTAssertNil(estimator.expectedFirstText(engine: .local, mode: .deep, now: at(3)))
    }

    func testEstimatesExpireAndAStaleSeriesStartsAfresh() {
        var estimator = LatencyEstimator()
        for index in 0..<3 {
            estimator.record(engine: .cloud, mode: .standard, firstText: 6, at: at(Double(index)))
        }
        let maxAge = LatencyEstimator.maximumAge
        XCTAssertEqual(maxAge, 900)
        assertNear(estimator.expectedFirstText(engine: .cloud, mode: .standard, now: at(2 + maxAge)), 6)
        XCTAssertNil(estimator.expectedFirstText(engine: .cloud, mode: .standard, now: at(3 + maxAge)))
        XCTAssertNil(estimator.p90FirstText(engine: .cloud, mode: .standard, now: at(3 + maxAge)))
        XCTAssertEqual(estimator.sampleCount(engine: .cloud, mode: .standard, now: at(3 + maxAge)), 0)

        // The next sample after the gap replaces the old series rather than averaging with it.
        for index in 0..<3 {
            estimator.record(engine: .cloud, mode: .standard, firstText: 1, at: at(2000 + Double(index)))
        }
        assertNear(estimator.expectedFirstText(engine: .cloud, mode: .standard, now: at(2003)), 1)
        XCTAssertEqual(estimator.sampleCount(engine: .cloud, mode: .standard, now: at(2003)), 3)
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
        for _ in 0..<3 {
            estimator.record(engine: .cloud, mode: .standard, firstText: 4)
        }
        assertNear(estimator.expectedFirstText(engine: .cloud, mode: .standard), 4)
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
        XCTAssertEqual(decoded.p90FirstText(engine: .cloud, mode: .standard, now: at(4)), 4)

        XCTAssertEqual(try JSONDecoder().decode(LatencyEstimator.self, from: Data("{}".utf8)), LatencyEstimator())
        XCTAssertEqual(try JSONDecoder().decode(LatencyEstimator.self, from: Data(#"{"series":[1,2]}"#.utf8)), LatencyEstimator())
    }
}
