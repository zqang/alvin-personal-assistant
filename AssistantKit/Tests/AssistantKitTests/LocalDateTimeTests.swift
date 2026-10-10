import XCTest
@testable import AssistantKit

final class LocalDateTimeTests: XCTestCase {
    private let singapore = TimeZone(identifier: "Asia/Singapore")!
    private let newYork = TimeZone(identifier: "America/New_York")!
    private let losAngeles = TimeZone(identifier: "America/Los_Angeles")!
    private let utc = TimeZone(identifier: "UTC")!

    /// 2026-10-07 09:00 UTC, which is 17:00 in Singapore.
    private let fivePMInSingapore = Date(timeIntervalSince1970: 1_791_363_600)

    func testParsesLocalTimeInTheGivenTimeZone() {
        XCTAssertEqual(LocalDateTime.parse("2026-10-07T17:00", timeZone: singapore), .init(date: fivePMInSingapore, hasTime: true))
        XCTAssertEqual(LocalDateTime.parse("2026-10-07T17:00:00", timeZone: singapore), .init(date: fivePMInSingapore, hasTime: true))
        XCTAssertEqual(
            LocalDateTime.parse("2026-10-07T17:00", timeZone: utc)?.date,
            Date(timeIntervalSince1970: 1_791_392_400)
        )
    }

    func testFollowsDaylightSavingTime() {
        // New York is UTC-4 in July and UTC-5 in January.
        XCTAssertEqual(LocalDateTime.parse("2026-07-01T12:00", timeZone: newYork)?.date, Date(timeIntervalSince1970: 1_782_921_600))
        XCTAssertEqual(LocalDateTime.parse("2026-01-15T12:00", timeZone: newYork)?.date, Date(timeIntervalSince1970: 1_768_496_400))
    }

    func testDateAloneIsTheStartOfThatLocalDay() {
        let parsed = LocalDateTime.parse("2026-10-07", timeZone: singapore)
        // Midnight in Singapore is 16:00 UTC the day before.
        XCTAssertEqual(parsed, .init(date: Date(timeIntervalSince1970: 1_791_302_400), hasTime: false))
        XCTAssertEqual(LocalDateTime.parse("2028-02-29", timeZone: utc)?.date, Date(timeIntervalSince1970: 1_835_395_200))
    }

    func testExplicitOffsetsOverrideTheTimeZone() {
        for text in ["2026-10-07T09:00Z", "2026-10-07T09:00:00Z", "2026-10-07t09:00z"] {
            XCTAssertEqual(LocalDateTime.parse(text, timeZone: newYork), .init(date: fivePMInSingapore, hasTime: true), text)
        }
        for text in ["2026-10-07T17:00+08:00", "2026-10-07T17:00:00+08:00", "2026-10-07T17:00+0800", "2026-10-07T17:00+08"] {
            XCTAssertEqual(LocalDateTime.parse(text, timeZone: newYork)?.date, fivePMInSingapore, text)
        }
        XCTAssertEqual(LocalDateTime.parse("2026-10-07T04:00-05:00", timeZone: singapore)?.date, fivePMInSingapore)
        XCTAssertEqual(LocalDateTime.parse("2026-10-07T03:30-05:30", timeZone: singapore)?.date, fivePMInSingapore)
    }

    func testAcceptsWhatModelsWriteInPractice() {
        XCTAssertEqual(LocalDateTime.parse("2026-10-07 17:00", timeZone: singapore)?.date, fivePMInSingapore)
        XCTAssertEqual(LocalDateTime.parse("  2026-10-07T17:00\n", timeZone: singapore)?.date, fivePMInSingapore)
        XCTAssertEqual(
            LocalDateTime.parse("2026-10-07T17:00:30.5", timeZone: singapore)?.date,
            fivePMInSingapore.addingTimeInterval(30.5)
        )
        XCTAssertEqual(
            LocalDateTime.parse("2026-10-07T09:00:00.000Z", timeZone: newYork)?.date,
            fivePMInSingapore
        )
    }

    func testRejectsMalformedAndImpossibleTimes() {
        let bad = [
            "", "tomorrow", "07/10/2026", "2026-1-7", "2026-10-7", "26-10-07",
            "2026-13-01", "2026-00-10", "2026-02-29", "2026-02-30", "2026-04-31", "2026-10-00",
            "2026-10-07T24:00", "2026-10-07T17:60", "2026-10-07T17:00:60", "2026-10-07T17", "2026-10-07T5:00",
            "2026-10-07T17:00 tomorrow", "2026-10-07T17:00+", "2026-10-07T17:00+19:00", "2026-10-07T17:00+08:61",
            "2026-10-07T17:00:00.", "2026-10-07Z", "2026-10-07X17:00",
        ]
        for text in bad {
            XCTAssertNil(LocalDateTime.parse(text, timeZone: singapore), text)
        }
    }

    func testFormatsWallClockTimeWithoutOffset() {
        XCTAssertEqual(LocalDateTime.format(fivePMInSingapore, timeZone: singapore), "2026-10-07T17:00")
        XCTAssertEqual(LocalDateTime.format(fivePMInSingapore, timeZone: singapore, includeTime: false), "2026-10-07")
        XCTAssertEqual(LocalDateTime.format(fivePMInSingapore, timeZone: losAngeles), "2026-10-07T02:00")
        XCTAssertEqual(LocalDateTime.format(fivePMInSingapore, timeZone: utc), "2026-10-07T09:00")
        XCTAssertEqual(LocalDateTime.format(fivePMInSingapore, timeZone: newYork, includeTime: false), "2026-10-07")
        // Seconds are dropped, not rounded.
        XCTAssertEqual(LocalDateTime.format(fivePMInSingapore.addingTimeInterval(59.9), timeZone: singapore), "2026-10-07T17:00")
        // Single-digit fields are zero-padded.
        let early = LocalDateTime.parse("2026-01-02T03:04", timeZone: newYork)!.date
        XCTAssertEqual(LocalDateTime.format(early, timeZone: newYork), "2026-01-02T03:04")
    }

    func testFormatThenParseRoundTrips() {
        for zone in [singapore, newYork, losAngeles, utc] {
            for text in ["2026-10-07T17:00", "2026-03-01T00:00", "2026-12-31T23:59", "2028-02-29T12:30"] {
                let parsed = LocalDateTime.parse(text, timeZone: zone)
                XCTAssertEqual(parsed.map { LocalDateTime.format($0.date, timeZone: zone) }, text, "\(text) in \(zone.identifier)")
            }
            let day = LocalDateTime.parse("2026-10-07", timeZone: zone)
            XCTAssertEqual(day.map { LocalDateTime.format($0.date, timeZone: zone, includeTime: false) }, "2026-10-07")
            XCTAssertEqual(day.map { LocalDateTime.format($0.date, timeZone: zone) }, "2026-10-07T00:00")
        }
    }
}
