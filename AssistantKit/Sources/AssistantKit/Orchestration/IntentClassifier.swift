import Foundation

/// Cheap, rule-based hints about what an utterance asks for, in English and Chinese.
///
/// The router uses them to choose an engine and a mode; they never change what a model is told.
/// Matching is a substring search of the lowercased text, with word boundaries at the ends of a
/// phrase in Latin script, so "plan" doesn't fire on "explanation" or "planet" and "hi" doesn't fire
/// on "this". In the fresh-facts, device-action and complex tables the last word may be inflected:
/// every phrase may take a plural ("reminders", "searches"), and the phrases in `verbPhrases` also
/// take verb endings ("scored", "searching", "scheduled", "planning"); see `endings(for:)`. Nouns
/// don't take verb endings, so "alarming" isn't a device action. Chinese, Japanese and Korean
/// phrases match anywhere, because those scripts don't separate words.
public struct IntentClassifier: Sendable {
    public struct Intents: OptionSet, Hashable, Sendable {
        public let rawValue: UInt8

        public init(rawValue: UInt8) {
            self.rawValue = rawValue
        }

        /// Needs current information from the internet: news, weather, prices, scores, search.
        public static let freshFacts = Intents(rawValue: 1 << 0)
        /// Acts on the device or reads from it: reminders, calendar, timers, alarms.
        public static let deviceAction = Intents(rawValue: 1 << 1)
        /// The user asks for a careful, in-depth answer.
        public static let explicitDepth = Intents(rawValue: 1 << 2)
        /// Long, multi-part, or a comparison or plan.
        public static let complex = Intents(rawValue: 1 << 3)
        /// A short greeting, thanks, acknowledgement, clock question or simple sum.
        public static let smallTalk = Intents(rawValue: 1 << 4)
    }

    /// Small talk has at most this many words and at most `smallTalkMaxCJK` CJK characters. Chinese
    /// written without spaces counts as one word, so each script is held to its own limit.
    public static let smallTalkMaxWords = 8
    public static let smallTalkMaxCJK = 12
    /// More words than this makes a request complex.
    public static let complexMinWords = 46
    /// More CJK characters than this makes a request complex.
    public static let complexMinCJK = 81
    /// This many question marks or more makes a request complex.
    public static let complexMinQuestions = 2

    // MARK: Phrase tables (lowercase). Tests pin them; tune them together.

    static let freshFactsPhrases = [
        "weather", "forecast", "news", "headline", "score", "stock", "share price", "price of",
        "exchange rate", "traffic", "flight", "open now", "opening hours", "latest", "today's",
        "right now", "search", "look up", "look it up", "look that up", "google",
        "天气", "新闻", "股价", "汇率", "比分", "最新", "今天的", "搜索", "查一下",
    ]

    /// Phrases in the inflected tables that are also verbs, which take verb endings as well as plurals:
    /// "scored", "searching", "googled", "scheduled", "compared", "planning".
    static let verbPhrases: Set<String> = ["score", "search", "google", "schedule", "compare", "plan"]

    /// Fresh-facts phrases that only qualify time. They don't count in a question about the clock
    /// itself ("what time is it right now", "what's today's date").
    static let timeQualifierPhrases = ["today's", "right now", "今天的"]

    static let deviceActionPhrases = [
        "remind me", "reminder", "to-do", "todo", "calendar", "schedule", "meeting", "appointment",
        "what's on", "am i free", "timer", "alarm", "wake me",
        "提醒", "日程", "日历", "会议", "计时", "闹钟", "定时",
    ]

    static let explicitDepthPhrases = [
        "think hard", "think harder", "think carefully", "think deeply", "think it through",
        "take your time", "deep dive", "really think", "research this", "in depth", "in-depth",
        "thoroughly", "dig into",
        "认真想", "仔细想", "好好想想", "深入", "详细分析",
    ]

    static let complexPhrases = [
        "compare", "comparison", "pros and cons", "step by step", "step-by-step", "plan", "strategy",
        "strategies", "explain why",
        "比较", "优缺点", "计划",
    ]

    /// Questions about the time or date, which the on-device model answers from the clock.
    static let clockPhrases = [
        "what time is it", "what's the time", "what is the time", "what's the date", "what is the date",
        "what day is it", "today's date",
        "几点了", "几点钟", "现在几点", "今天几号", "今天是几号", "今天星期几",
    ]

    static let smallTalkPhrases = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening", "good night", "goodnight",
        "thanks", "thank you", "thx", "ok", "okay", "how are you", "how's it going", "what's up",
        "bye", "goodbye", "see you", "nice to meet you",
        "你好", "您好", "谢谢", "多谢", "早上好", "早安", "晚上好", "晚安", "再见",
    ] + clockPhrases

    public init() {}

    public func classify(_ utterance: String) -> Intents {
        let text = Self.normalize(utterance)
        guard !text.isEmpty else { return [] }
        let scalars = Array(text.unicodeScalars)
        let words = Self.wordCount(text)
        let cjk = Self.cjkCount(scalars)
        let isClockQuestion = Self.matchesAny(scalars, Self.clockPhrases)

        var intents: Intents = []
        let fresh = isClockQuestion
            ? Self.freshFactsPhrases.filter { !Self.timeQualifierPhrases.contains($0) }
            : Self.freshFactsPhrases
        if Self.matchesAny(scalars, fresh, inflected: true) { intents.insert(.freshFacts) }
        if Self.matchesAny(scalars, Self.deviceActionPhrases, inflected: true) { intents.insert(.deviceAction) }
        if Self.matchesAny(scalars, Self.explicitDepthPhrases) { intents.insert(.explicitDepth) }

        let questions = scalars.filter { $0 == "?" || $0 == "？" }.count
        if words >= Self.complexMinWords || cjk >= Self.complexMinCJK || questions >= Self.complexMinQuestions
            || Self.matchesAny(scalars, Self.complexPhrases, inflected: true) {
            intents.insert(.complex)
        }

        let isShort = words <= Self.smallTalkMaxWords && cjk <= Self.smallTalkMaxCJK
        if isShort, Self.matchesAny(scalars, Self.smallTalkPhrases) || Self.isSimpleArithmetic(text) {
            intents.insert(.smallTalk)
        }
        return intents
    }

    // MARK: Text measures

    /// Lowercased, with curly apostrophes straightened and runs of whitespace collapsed to one space.
    static func normalize(_ utterance: String) -> String {
        let lowered = utterance.lowercased()
            .replacingOccurrences(of: "\u{2019}", with: "'")
            .replacingOccurrences(of: "\u{2018}", with: "'")
        return lowered.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    }

    /// Space-separated tokens that hold at least one letter or digit. A run of Chinese text without
    /// spaces counts as one word; `cjkCount` measures it instead.
    static func wordCount(_ text: String) -> Int {
        text.split(separator: " ").filter { token in
            token.unicodeScalars.contains { $0.properties.isAlphabetic || $0.properties.numericType != nil }
        }.count
    }

    static func cjkCount(_ scalars: [Unicode.Scalar]) -> Int {
        scalars.filter(isCJK).count
    }

    static func isCJK(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x3040...0x30FF,  // Hiragana, Katakana
             0x3400...0x4DBF,  // CJK Extension A
             0x4E00...0x9FFF,  // CJK Unified Ideographs
             0xAC00...0xD7AF,  // Hangul syllables
             0xF900...0xFAFF,  // CJK Compatibility Ideographs
             0x20000...0x2FA1F:  // CJK Extensions B and later
            return true
        default:
            return false
        }
    }

    /// A letter or digit of a script that separates words with spaces.
    static func isWordScalar(_ scalar: Unicode.Scalar) -> Bool {
        guard !isCJK(scalar) else { return false }
        return scalar.properties.isAlphabetic || ("0"..."9").contains(scalar)
    }

    // MARK: Matching

    static func matchesAny(_ text: [Unicode.Scalar], _ phrases: [String], inflected: Bool = false) -> Bool {
        phrases.contains { matches(text, phrase: $0, inflected: inflected) }
    }

    /// Whether `phrase` occurs in `text`; with `inflected`, also with the endings its last word may
    /// take (see `endings(for:)`).
    static func matches(_ text: [Unicode.Scalar], phrase: String, inflected: Bool) -> Bool {
        let scalars = Array(phrase.unicodeScalars)
        guard inflected else { return contains(text, phrase: scalars) }
        if contains(text, phrase: scalars, endings: endings(for: phrase)) { return true }
        // A verb's final "e" drops before "ing": "score" → "scoring".
        return verbPhrases.contains(phrase) && phrase.hasSuffix("e") && scalars.count > 2
            && contains(text, phrase: Array(scalars.dropLast()), endings: ["ing"])
    }

    /// The letters the last word of `phrase` may gain in an inflected table, "" being none: a plural
    /// ("es" after s, x, z, ch or sh, otherwise "s"), and for a verb "d" after a final "e", otherwise
    /// "ed" and "ing", also after a doubled final consonant. So "plan" matches "plans", "planned"
    /// and "planning" but not "planes", and "alarm" matches "alarms" but not "alarming".
    static func endings(for phrase: String) -> Set<String> {
        guard let last = phrase.last else { return [""] }
        let sibilant = ["s", "x", "z", "ch", "sh"].contains { phrase.hasSuffix($0) }
        var endings: Set<String> = ["", sibilant ? "es" : "s"]
        guard verbPhrases.contains(phrase) else { return endings }
        if last == "e" {
            endings.insert("d")
        } else {
            endings.formUnion(["ed", "ing", "\(last)ed", "\(last)ing"])
        }
        return endings
    }

    /// Whether `phrase` occurs in `text`. A start of the phrase that is a letter or digit must sit at
    /// a word boundary. So must its end, after one of `endings`: the letters that may follow the
    /// phrase within its last word, where "" is the bare phrase. An end that isn't a letter or digit
    /// (Chinese, punctuation) needs no boundary.
    static func contains(_ text: [Unicode.Scalar], phrase: [Unicode.Scalar], endings: Set<String> = [""]) -> Bool {
        guard let first = phrase.first, let last = phrase.last, phrase.count <= text.count else { return false }
        let boundedStart = isWordScalar(first)
        let boundedEnd = isWordScalar(last)
        var start = 0
        while start + phrase.count <= text.count {
            defer { start += 1 }
            guard text[start] == first, text[start..<start + phrase.count].elementsEqual(phrase) else { continue }
            if boundedStart, start > 0, isWordScalar(text[start - 1]) { continue }
            guard boundedEnd else { return true }
            let end = start + phrase.count
            var wordEnd = end
            while wordEnd < text.count, isWordScalar(text[wordEnd]) { wordEnd += 1 }
            var ending = String.UnicodeScalarView()
            ending.append(contentsOf: text[end..<wordEnd])
            if endings.contains(String(ending)) { return true }
        }
        return false
    }

    /// A bare sum such as "12 x 4", "what's 2 plus 2?" or "3加5等于几": only digits, spaces, dots and
    /// the operators + - * / x × ÷, with at least one digit and one operator
    /// (`^[\d\s+\-*/x×÷.]+$` once the question words are removed).
    static func isSimpleArithmetic(_ normalized: String) -> Bool {
        var text = normalized
        while let last = text.last, "?？=.。!！ ".contains(last) {
            text.removeLast()
        }
        for prefix in ["what's ", "what is ", "how much is ", "calculate "] where text.hasPrefix(prefix) {
            text = String(text.dropFirst(prefix.count))
            break
        }
        for suffix in ["等于多少", "等于几", "是多少"] where text.hasSuffix(suffix) {
            text = String(text.dropLast(suffix.count))
            break
        }
        let operatorWords = [
            ("divided by", "/"), ("multiplied by", "*"), ("plus", "+"), ("minus", "-"), ("times", "*"),
            ("除以", "/"), ("乘以", "*"), ("加", "+"), ("减", "-"), ("乘", "*"),
        ]
        for (word, symbol) in operatorWords {
            text = text.replacingOccurrences(of: word, with: " \(symbol) ")
        }
        var hasDigit = false
        var hasOperator = false
        for scalar in text.unicodeScalars {
            if scalar.properties.numericType == .decimal {
                hasDigit = true
            } else if "+-*/x×÷".unicodeScalars.contains(scalar) {
                hasOperator = true
            } else if scalar != "." && !scalar.properties.isWhitespace {
                return false
            }
        }
        return hasDigit && hasOperator
    }
}
