import XCTest
@testable import AssistantKit

final class DeepBudgetTests: XCTestCase {
    private let singapore = TimeZone(identifier: "Asia/Singapore")!  // UTC+8, no daylight saving
    private let utc = TimeZone(identifier: "UTC")!
    private let newYork = TimeZone(identifier: "America/New_York")!

    /// A moment given as UTC calendar fields.
    private func utcDate(_ year: Int, _ month: Int, _ day: Int, _ hour: Int, _ minute: Int = 0) -> Date {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = utc
        return calendar.date(from: DateComponents(year: year, month: month, day: day, hour: hour, minute: minute))!
    }

    func testNewBudgetIsUnused() {
        let budget = DeepBudget()
        XCTAssertEqual(budget.day, "")
        XCTAssertEqual(budget.used, 0)
        XCTAssertEqual(budget.remaining(limit: 10, now: Date(), timeZone: singapore), 10)
        XCTAssertEqual(budget.usedToday(now: Date(), timeZone: singapore), 0)
    }

    func testRecordingCountsDownToZero() {
        var budget = DeepBudget()
        let now = utcDate(2026, 10, 7, 3)
        for expected in stride(from: 2, through: 0, by: -1) {
            budget.record(now: now, timeZone: singapore)
            XCTAssertEqual(budget.remaining(limit: 3, now: now, timeZone: singapore), expected)
        }
        budget.record(now: now, timeZone: singapore)
        XCTAssertEqual(budget.used, 4)
        XCTAssertEqual(budget.remaining(limit: 3, now: now, timeZone: singapore), 0, "never negative")
        XCTAssertEqual(budget.remaining(limit: 0, now: now, timeZone: singapore), 0)
        XCTAssertEqual(budget.remaining(limit: -5, now: now, timeZone: singapore), 0)
    }

    func testDayRollsOverAtLocalMidnight() {
        var budget = DeepBudget()
        // 23:30 on 7 October in Singapore is 15:30 UTC.
        let lateEvening = utcDate(2026, 10, 7, 15, 30)
        budget.record(now: lateEvening, timeZone: singapore)
        budget.record(now: lateEvening, timeZone: singapore)
        XCTAssertEqual(budget.day, "2026-10-07")
        XCTAssertEqual(budget.remaining(limit: 10, now: lateEvening, timeZone: singapore), 8)

        // 00:30 on 8 October in Singapore, while it's still 7 October in UTC.
        let afterMidnight = utcDate(2026, 10, 7, 16, 30)
        XCTAssertEqual(DeepBudget.dayKey(for: afterMidnight, timeZone: singapore), "2026-10-08")
        XCTAssertEqual(DeepBudget.dayKey(for: afterMidnight, timeZone: utc), "2026-10-07")
        XCTAssertEqual(budget.remaining(limit: 10, now: afterMidnight, timeZone: singapore), 10)
        XCTAssertEqual(budget.usedToday(now: afterMidnight, timeZone: singapore), 0)

        budget.record(now: afterMidnight, timeZone: singapore)
        XCTAssertEqual(budget.day, "2026-10-08")
        XCTAssertEqual(budget.used, 1)
        XCTAssertEqual(budget.remaining(limit: 10, now: afterMidnight, timeZone: singapore), 9)
        // The day before no longer counts anything.
        XCTAssertEqual(budget.remaining(limit: 10, now: lateEvening, timeZone: singapore), 10)
    }

    func testTheSameMomentFallsOnDifferentDaysInDifferentZones() {
        var budget = DeepBudget()
        // 02:00 UTC on 8 October is 22:00 on 7 October in New York (EDT, UTC-4).
        let moment = utcDate(2026, 10, 8, 2)
        budget.record(now: moment, timeZone: newYork)
        XCTAssertEqual(budget.day, "2026-10-07")
        XCTAssertEqual(budget.remaining(limit: 10, now: moment, timeZone: newYork), 9)
        XCTAssertEqual(budget.remaining(limit: 10, now: moment, timeZone: utc), 10, "after travel the count follows the new zone")
    }

    func testDayKeyIsGregorianAndZeroPadded() {
        XCTAssertEqual(DeepBudget.dayKey(for: utcDate(2027, 1, 2, 12), timeZone: utc), "2027-01-02")
        XCTAssertEqual(DeepBudget.dayKey(for: utcDate(2026, 12, 31, 23, 59), timeZone: utc), "2026-12-31")
        XCTAssertEqual(DeepBudget.dayKey(for: utcDate(2026, 12, 31, 23, 59), timeZone: singapore), "2027-01-01")
    }

    func testCodableRoundTripAndTolerantDecoding() throws {
        var budget = DeepBudget()
        budget.record(now: utcDate(2026, 10, 7, 3), timeZone: singapore)
        let data = try JSONEncoder().encode(budget)
        XCTAssertEqual(try JSONDecoder().decode(DeepBudget.self, from: data), budget)

        XCTAssertEqual(try JSONDecoder().decode(DeepBudget.self, from: Data("{}".utf8)), DeepBudget())
        XCTAssertEqual(
            try JSONDecoder().decode(DeepBudget.self, from: Data(#"{"day":"2026-10-07","used":"many"}"#.utf8)),
            DeepBudget(day: "2026-10-07", used: 0)
        )
    }
}
