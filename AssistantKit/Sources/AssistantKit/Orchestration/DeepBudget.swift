import Foundation

/// How many deep-mode replies ran today. The day is the calendar day in the user's time zone, so
/// the allowance resets at local midnight, also after travel.
public struct DeepBudget: Codable, Equatable, Sendable {
    /// The day `used` counts, as `yyyy-MM-dd`; empty before the first deep reply.
    public var day: String
    /// Deep replies started on `day`.
    public var used: Int

    public init() {
        self.init(day: "", used: 0)
    }

    public init(day: String, used: Int) {
        self.day = day
        self.used = used
    }

    /// The Gregorian calendar day of `now` in `timeZone`, as `yyyy-MM-dd`.
    public static func dayKey(for now: Date, timeZone: TimeZone) -> String {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = timeZone
        let parts = calendar.dateComponents([.year, .month, .day], from: now)
        return "\(padded(parts.year ?? 0, 4))-\(padded(parts.month ?? 0, 2))-\(padded(parts.day ?? 0, 2))"
    }

    private static func padded(_ number: Int, _ width: Int) -> String {
        let digits = String(number)
        return String(repeating: "0", count: max(0, width - digits.count)) + digits
    }

    /// Deep replies started on the day of `now`.
    public func usedToday(now: Date, timeZone: TimeZone) -> Int {
        day == Self.dayKey(for: now, timeZone: timeZone) ? max(0, used) : 0
    }

    /// Deep replies still allowed on the day of `now`; never negative.
    public func remaining(limit: Int, now: Date, timeZone: TimeZone) -> Int {
        max(0, limit - usedToday(now: now, timeZone: timeZone))
    }

    /// Counts one deep reply, starting a new count when the day has changed.
    public mutating func record(now: Date, timeZone: TimeZone) {
        let today = Self.dayKey(for: now, timeZone: timeZone)
        if day != today {
            day = today
            used = 0
        }
        used = max(0, used) + 1
    }

    private enum CodingKeys: String, CodingKey {
        case day, used
    }

    /// Missing or malformed fields decode as an unused budget.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        day = (try? container.decodeIfPresent(String.self, forKey: .day)) ?? ""
        used = (try? container.decodeIfPresent(Int.self, forKey: .used)) ?? 0
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(day, forKey: .day)
        try container.encode(used, forKey: .used)
    }
}
