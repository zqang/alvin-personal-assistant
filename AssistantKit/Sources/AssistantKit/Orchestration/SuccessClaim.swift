import Foundation

/// Whether a reply says an action was carried out, in English or Chinese: "I've set a timer",
/// "Reminder set for 5 pm", "已经为你设置好了明天的提醒", "会议已加入您的日程".
///
/// The on-device tool loop asks this of a reply to an action request (`IntentClassifier`'s
/// `.deviceAction`) that called no tool, which therefore claims something that didn't happen. Like
/// `IntentClassifier` it is a table of phrases, matched the same way (lowercased; word boundaries
/// in Latin script, anywhere in CJK). The text is split into clauses at sentence ends and commas,
/// and a clause that is a question (it ends with "?", "？" or "吗") never counts, so offers such as
/// "Want me to add it?" and "要我帮你设置提醒吗？" don't. Neither do forms that name no finished
/// action ("I can set one").
///
/// The tables here are generic; the app adds the nouns its actions make and its own phrases
/// (`AssistantProfile.claims`).
public struct SuccessClaim: Sendable {
    // MARK: Phrase tables (lowercase). Tests pin them; tune them together.

    /// Participles that report an action as done: after "is", "it's", "has been" or a noun such as
    /// "timer", and at the start of a clause ("Added buy milk to your list.").
    static let participles = ["set", "added", "scheduled", "created", "booked", "saved", "started", "done"]

    /// Verbs that, after "I'll" or "I will", promise an action the reply has no means to carry out.
    static let promisedVerbs = ["set", "add", "schedule", "create", "book", "put", "start", "save"]

    /// The English claims: `participles` after "I", "is" and the like, and after each of `nouns`;
    /// `verbs` after "I'll" and "I will".
    static func englishPhrases(participles: [String], verbs: [String], nouns: [String]) -> [String] {
        func after(_ firsts: [String], _ lasts: [String]) -> [String] {
            firsts.flatMap { first in ["", "just ", "already ", "also "].flatMap { adverb in lasts.map { "\(first) \(adverb)\($0)" } } }
        }
        return after(["i", "i've", "i have"], participles + ["put"]) + after(["i'll", "i will"], verbs)
            + ["is", "are", "it's", "that's", "has been", "have been"].flatMap { verb in participles.map { "\(verb) \($0)" } }
            + nouns.flatMap { noun in participles.map { "\(noun) \($0)" } }
            + ["all set", "added to your", "added it to"]
    }

    static let chinesePhrases = [
        "已经", "已为", "已帮", "已给", "已将", "已设", "已添加", "已加入", "已创建", "已安排", "已记", "已预约",
        "已启动", "已成功", "设置好", "设好了", "加好了", "记好了", "安排好了", "设置成功", "添加成功",
        "设置了", "添加了", "创建了", "安排了", "加入了", "加到了", "记下了", "预约了", "我会在",
    ]

    /// The generic participles and the app's.
    let allParticiples: [String]
    let phrases: [String]

    /// - Parameters:
    ///   - nouns: what the app's actions make, which a claim names before a participle ("reminder
    ///     set", "timer started").
    ///   - participles: more participles that report an action as done ("published").
    ///   - verbs: more verbs that "I'll" promises ("remind").
    ///   - phrases: more phrases that claim an action, such as promises in Chinese ("会提醒你").
    public init(nouns: [String] = [], participles: [String] = [], verbs: [String] = [], phrases: [String] = []) {
        // Matching lowercases the text, not the phrases.
        allParticiples = Self.participles + participles.map(IntentClassifier.normalize)
        let verbs = Self.promisedVerbs + verbs.map(IntentClassifier.normalize)
        self.phrases = Self.englishPhrases(participles: allParticiples, verbs: verbs, nouns: nouns.map(IntentClassifier.normalize))
            + Self.chinesePhrases + phrases.map(IntentClassifier.normalize)
    }

    /// Whether `text` says, outside a question, that an action was done or will be done by the
    /// assistant itself.
    public func claimsDone(_ text: String) -> Bool {
        var clause = ""
        for character in text + "\n" {
            guard "。！？.!?\n，,；;".contains(character) else {
                clause.append(character)
                continue
            }
            let normalized = IntentClassifier.normalize(clause)
            let isQuestion = character == "?" || character == "？" || normalized.hasSuffix("吗")
            if !isQuestion, isClaim(normalized) { return true }
            clause = ""
        }
        return false
    }

    /// Whether a normalized clause that isn't a question claims an action.
    func isClaim(_ clause: String) -> Bool {
        if let first = clause.split(whereSeparator: { !$0.isLetter }).first, allParticiples.contains(String(first)) {
            return true
        }
        return IntentClassifier.matchesAny(Array(clause.unicodeScalars), phrases)
    }

    /// What to say when a requested action couldn't be done: in Chinese when `userText` contains
    /// Chinese, Japanese or Korean script (`IntentClassifier.isCJK`), otherwise in English.
    public static func actionFailedMessage(for userText: String) -> String {
        userText.unicodeScalars.contains(where: IntentClassifier.isCJK)
            ? "抱歉，这个操作我没能完成。你可以再说一次，或者换个说法。"
            : "Sorry, I couldn't do that. Please try again or rephrase."
    }
}
