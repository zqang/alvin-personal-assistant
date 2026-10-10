import Foundation

/// Reads and writes the ISO 8601 local times that tools exchange with models, e.g.
/// `2026-10-07T17:00`: wall-clock time in the user's time zone, without an offset.
///
/// Parsing also accepts what models write in practice: seconds (`17:00:00`, with or without a
/// fraction), a space instead of `T`, a date alone, and an explicit `Z` or `+08:00` / `+0800` /
/// `+08` offset, which then takes precedence over the user's time zone. It never depends on the
/// device's locale or calendar settings.
public enum LocalDateTime {
    public struct Parsed: Equatable, Sendable {
        public var date: Date
        /// False when the text was a date alone; `date` is then the start of that day.
        public var hasTime: Bool

        public init(date: Date, hasTime: Bool) {
            self.date = date
            self.hasTime = hasTime
        }
    }

    /// The moment `text` names, reading times without an offset in `timeZone`, or nil if `text`
    /// isn't one of the accepted forms or names a date that doesn't exist (e.g. 2026-02-30).
    public static func parse(_ text: String, timeZone: TimeZone) -> Parsed? {
        var scanner = TextCursor(text.trimmingCharacters(in: .whitespacesAndNewlines))

        guard let year = scanner.digits(4), scanner.take("-"),
              let month = scanner.digits(2), scanner.take("-"),
              let day = scanner.digits(2),
              (1...9999).contains(year), (1...12).contains(month),
              (1...daysIn(month: month, year: year)).contains(day)
        else { return nil }

        var components = DateComponents(year: year, month: month, day: day, hour: 0, minute: 0, second: 0)
        if scanner.isAtEnd {
            guard let date = calendar(timeZone).date(from: components) else { return nil }
            return Parsed(date: date, hasTime: false)
        }

        guard scanner.take("T") || scanner.take("t") || scanner.take(" "),
              let hour = scanner.digits(2), scanner.take(":"),
              let minute = scanner.digits(2),
              (0...23).contains(hour), (0...59).contains(minute)
        else { return nil }
        components.hour = hour
        components.minute = minute

        var fraction: TimeInterval = 0
        if scanner.take(":") {
            guard let second = scanner.digits(2), (0...59).contains(second) else { return nil }
            components.second = second
            if scanner.take(".") || scanner.take(",") {
                let digits = scanner.digitRun()
                guard !digits.isEmpty else { return nil }
                fraction = Double("0." + digits) ?? 0
            }
        }

        var offset: Int?
        if scanner.take("Z") || scanner.take("z") {
            offset = 0
        } else if let sign = scanner.takeSign() {
            guard let hours = scanner.digits(2), hours <= 18 else { return nil }
            var minutes = 0
            if !scanner.isAtEnd {
                _ = scanner.take(":")
                guard let value = scanner.digits(2), value <= 59 else { return nil }
                minutes = value
            }
            offset = sign * (hours * 3600 + minutes * 60)
        }
        guard scanner.isAtEnd else { return nil }

        let date: Date?
        if let offset {
            // The wall-clock components are in the given offset: read them as UTC, then shift.
            date = calendar(utc).date(from: components).map { $0.addingTimeInterval(TimeInterval(-offset)) }
        } else {
            date = calendar(timeZone).date(from: components)
        }
        return date.map { Parsed(date: $0.addingTimeInterval(fraction), hasTime: true) }
    }

    /// `date` as wall-clock time in `timeZone`: `2026-10-07T17:00`, or `2026-10-07` without the
    /// time. Seconds are dropped, not rounded.
    public static func format(_ date: Date, timeZone: TimeZone, includeTime: Bool = true) -> String {
        let parts = calendar(timeZone).dateComponents([.year, .month, .day, .hour, .minute], from: date)
        let day = "\(pad(parts.year ?? 0, 4))-\(pad(parts.month ?? 1, 2))-\(pad(parts.day ?? 1, 2))"
        guard includeTime else { return day }
        return "\(day)T\(pad(parts.hour ?? 0, 2)):\(pad(parts.minute ?? 0, 2))"
    }

    // MARK: - Helpers

    private static let utc = TimeZone(secondsFromGMT: 0) ?? TimeZone(identifier: "UTC")!

    private static func calendar(_ timeZone: TimeZone) -> Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.locale = Locale(identifier: "en_US_POSIX")
        calendar.timeZone = timeZone
        return calendar
    }

    private static func daysIn(month: Int, year: Int) -> Int {
        switch month {
        case 2:
            let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
            return leap ? 29 : 28
        case 4, 6, 9, 11:
            return 30
        default:
            return 31
        }
    }

    private static func pad(_ value: Int, _ width: Int) -> String {
        let digits = String(value)
        return digits.count >= width ? digits : String(repeating: "0", count: width - digits.count) + digits
    }

    /// Reads ASCII tokens from the front of a string.
    private struct TextCursor {
        private let bytes: [UInt8]
        private var position = 0

        init(_ text: String) {
            bytes = Array(text.utf8)
        }

        var isAtEnd: Bool { position == bytes.count }

        /// Consumes `character` if it comes next.
        mutating func take(_ character: Character) -> Bool {
            guard let ascii = character.asciiValue, position < bytes.count, bytes[position] == ascii else { return false }
            position += 1
            return true
        }

        /// Consumes `+` (1), `-` (-1) or the Unicode minus sign (-1) if one comes next.
        mutating func takeSign() -> Int? {
            if take("+") { return 1 }
            if take("-") { return -1 }
            let minus: [UInt8] = [0xE2, 0x88, 0x92]
            if bytes.count - position >= 3, Array(bytes[position..<position + 3]) == minus {
                position += 3
                return -1
            }
            return nil
        }

        /// Consumes exactly `count` ASCII digits and returns their value.
        mutating func digits(_ count: Int) -> Int? {
            guard bytes.count - position >= count else { return nil }
            var value = 0
            for byte in bytes[position..<position + count] {
                guard (48...57).contains(byte) else { return nil }
                value = value * 10 + Int(byte - 48)
            }
            position += count
            return value
        }

        /// Consumes every ASCII digit that comes next.
        mutating func digitRun() -> String {
            let start = position
            while position < bytes.count, (48...57).contains(bytes[position]) { position += 1 }
            return String(decoding: bytes[start..<position], as: UTF8.self)
        }
    }
}
