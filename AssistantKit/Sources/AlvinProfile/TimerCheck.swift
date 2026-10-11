import AssistantKit
import Foundation

/// This app's `ToolCallGuard`, for the on-device model's `set_timer` calls.
public enum TimerCheck {
    /// Stops the measured failure (plan §9): asked how long a timer had left, Woof started a
    /// one-second timer. A `set_timer` call runs when the user stated an amount the guard can
    /// read (`statesAnAmount`). Without one, it doesn't run when the user's words ask about a
    /// timer that is already running (`asksAboutRunningTimer`), or when `seconds` is under
    /// `shortestUnstatedSeconds`, a duration nobody wants without saying so (in any
    /// language).
    ///
    /// Any other call runs. Amounts are read only in English, Chinese and digits, and the
    /// app takes typed and spoken requests in every language speech recognition offers:
    /// objecting to every call without a readable amount would refuse "pon un temporizador
    /// de diez minutos" every time, and offline it could never be set at all.
    public static let `guard` = ToolCallGuard { call, userText in
        switch call.name {
        case setTimerName:
            return timerObjection(seconds: seconds(of: call), userText: userText)
        default:
            return nil
        }
    }

    public static let setTimerName = "set_timer"
    /// A timer shorter than this (in seconds) runs only when the user stated an amount.
    public static let shortestUnstatedSeconds = 5

    /// Why a `set_timer` call for `seconds` (nil when unreadable) doesn't fit `userText`.
    static func timerObjection(seconds: Int?, userText: String) -> String? {
        if statesAnAmount(userText) {
            return nil
        }
        if asksAboutRunningTimer(userText) {
            return "the user asked about a timer that is already running, and set_timer only starts a new one"
        }
        if let seconds, seconds < shortestUnstatedSeconds {
            return "the user didn't say how long the timer should run"
        }
        return nil
    }

    /// The call's `seconds` as a whole number, also when the model wrote it as a string
    /// ("600"), as on-device models do.
    static func seconds(of call: PendingToolCall) -> Int? {
        guard let value = call.input?["seconds"] else { return nil }
        if let whole = value.intValue {
            return whole
        }
        guard let text = value.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines) else { return nil }
        if let whole = Int(text) {
            return whole
        }
        guard let number = Double(text), number.isFinite, abs(number) < 9e15 else { return nil }
        return Int(number.rounded(.down))
    }

    /// Whether `text` asks about a timer that is already running: in English "left",
    /// "remaining", "how long", "how much"; in Chinese 剩 ("还剩", "剩下"), 多久, 多长时间,
    /// 多少时间 or 几 before a unit of time ("还有几分钟"). Other languages aren't recognised.
    public static func asksAboutRunningTimer(_ text: String) -> Bool {
        let words = text.lowercased().split { !$0.isLetter }.map(String.init)
        if words.contains(where: { runningTimerWords.contains($0) }) {
            return true
        }
        for (index, word) in words.enumerated().dropLast() {
            if word == "how", runningTimerFollowers.contains(words[index + 1]) {
                return true
            }
        }
        if runningTimerChinese.contains(where: { text.contains($0) }) {
            return true
        }
        let characters = Array(text)
        for (index, character) in characters.enumerated() where character == "几" || character == "幾" {
            var next = index + 1
            while next < characters.count, chineseBetween.contains(characters[next]) {
                next += 1
            }
            if next < characters.count, chineseTimeUnits.contains(characters[next]) {
                return true
            }
        }
        return false
    }

    private static let runningTimerWords: Set<String> = ["left", "remaining", "remain", "remains", "elapsed"]
    /// Words after "how" that ask for a duration: "how long", "how much (time)".
    private static let runningTimerFollowers: Set<String> = ["long", "much"]
    private static let runningTimerChinese = ["剩", "多久", "多长", "多長", "多少时间", "多少時間"]

    /// Whether `text` states an amount: a decimal digit in any script ("10", "３"), a Chinese
    /// numeral before a unit of time ("二十分钟", "一个半小时"; not the "一" of "一下" or
    /// "一个计时器"), an English number word ("ten", "half", "couple"), or "a"/"an" before a
    /// unit of time ("an hour").
    public static func statesAnAmount(_ text: String) -> Bool {
        if text.unicodeScalars.contains(where: { $0.properties.numericType == .decimal }) || hasChineseDuration(text) {
            return true
        }
        let words = text.lowercased().split { !$0.isLetter }.map(String.init)
        for (index, word) in words.enumerated() {
            if numberWords.contains(word) {
                return true
            }
            if word == "a" || word == "an", index + 1 < words.count, timeUnits.contains(words[index + 1]) {
                return true
            }
        }
        return false
    }

    /// A Chinese numeral followed by a unit of time, possibly after more numerals or a
    /// measure word ("一个半小时", "十多分钟", "一百二十秒"). "几分钟" doesn't count: it is
    /// how a question about a running timer asks ("还剩几分钟").
    private static func hasChineseDuration(_ text: String) -> Bool {
        let characters = Array(text)
        for (index, character) in characters.enumerated() where chineseNumerals.contains(character) {
            var next = index + 1
            while next < characters.count, chineseNumerals.contains(characters[next]) || chineseBetween.contains(characters[next]) {
                next += 1
            }
            if next < characters.count, chineseTimeUnits.contains(characters[next]) {
                return true
            }
        }
        return false
    }

    private static let chineseNumerals: Set<Character> = Set("〇零一二两兩三四五六七八九十百千万萬半廿卅")
    /// Measure words and "more" that may sit between a numeral and its unit.
    private static let chineseBetween: Set<Character> = Set("个個多来來")
    /// Seconds, minutes, hours (小时/小時, 钟头/鐘頭), quarter hours.
    private static let chineseTimeUnits: Set<Character> = Set("秒分小钟鐘刻")

    private static let numberWords: Set<String> = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
        "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
        "half", "quarter", "couple", "few", "several", "dozen",
    ]

    private static let timeUnits: Set<String> = [
        "second", "sec", "minute", "min", "hour", "hr",
    ]
}
