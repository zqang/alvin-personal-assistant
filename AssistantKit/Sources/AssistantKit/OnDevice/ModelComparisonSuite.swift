import Foundation

/// A small, fixed diagnostic for comparing local models under the same generation settings.
/// This is not a standardized capability benchmark. Every case is an independent conversation;
/// scoring includes compliance with the explicit answer format, not just semantic correctness.
public enum ModelComparisonSuite {
    public static let version = "local-comparison-v1"

    public struct Prompt: Identifiable, Sendable {
        public let id: String
        public let category: String
        public let language: String
        public let prompt: String
        fileprivate let rule: Rule

        /// The complete required answer, with canonical JSON key ordering for JSON cases.
        public var expected: String { rule.expected }
    }

    public struct Evaluation: Equatable, Sendable {
        public let passed: Bool
        public let expected: String
    }

    fileprivate enum Rule: Sendable {
        case exact(String)
        case json(JSONValue)

        var expected: String {
            switch self {
            case .exact(let text): return text
            case .json(let value):
                // These fixed fixtures contain only ordinary finite JSON values.
                return String(decoding: try! value.serialized(), as: UTF8.self)
            }
        }
    }

    /// 24 cases, balanced across six categories and English/Chinese (12 each).
    /// Expected answers are specified before either model is run.
    public static let prompts: [Prompt] = [
        .init(id: "arithmetic-en-1", category: "Arithmetic", language: "en",
              prompt: "There are 3 boxes with 12 pencils in each box, plus 7 loose pencils. How many pencils are there in total? Reply with only the integer, no units or explanation.",
              rule: .exact("43")),
        .init(id: "arithmetic-en-2", category: "Arithmetic", language: "en",
              prompt: "An item costs 120 dollars before a 25 percent discount. What is the price after that discount? Reply with only the integer, no currency sign or explanation.",
              rule: .exact("90")),
        .init(id: "arithmetic-zh-1", category: "Arithmetic", language: "zh",
              prompt: "有3辆车，每辆坐18人。后来一共下车11人，现在车上总共有多少人？只输出阿拉伯数字整数，不要单位或解释。",
              rule: .exact("43")),
        .init(id: "arithmetic-zh-2", category: "Arithmetic", language: "zh",
              prompt: "商店原有48本书，先卖掉原有数量的四分之一，再补进9本。现在有多少本？只输出阿拉伯数字整数，不要单位或解释。",
              rule: .exact("45")),

        .init(id: "reading-en-1", category: "Grounded reading", language: "en",
              prompt: "Use only this story: Maya put the key in the drawer. Tom moved the key from the drawer into a tin. Maya then moved the empty drawer to the garage. Where is the key now? Reply with exactly one word: drawer, tin, or garage.",
              rule: .exact("tin")),
        .init(id: "reading-en-2", category: "Grounded reading", language: "en",
              prompt: "Use only this note: The delivery was first scheduled for 12 June. It was rescheduled to 14 June. A later message confirmed 14 June without further changes. What is the final delivery date? Reply only in the form 'number June', without quotation marks.",
              rule: .exact("14 June")),
        .init(id: "reading-zh-1", category: "Grounded reading", language: "zh",
              prompt: "只根据资料回答，不要猜测。资料：李明今年12岁，他有一个姐姐，姐姐喜欢游泳。姐姐今年几岁？如果资料没有给出年龄，只输出“未知”（不含引号）；否则只输出年龄数字。",
              rule: .exact("未知")),
        .init(id: "reading-zh-2", category: "Grounded reading", language: "zh",
              prompt: "只根据资料回答：小文原来记录为64分，复核后更正为74分；小林得71分，小佳得69分。按更正后的分数，这三人中谁排第二？只输出姓名，不要标点或解释。",
              rule: .exact("小林")),

        .init(id: "json-en-1", category: "JSON extraction", language: "en",
              prompt: "Parse this reminder request without taking any action. Today is 2026-10-13. Request: Remind me tomorrow at 09:15 to Call Mei; do not repeat. Output only one JSON object with exactly these keys: title (string), at (string in YYYY-MM-DDTHH:MM format), repeats (boolean). Preserve the title 'Call Mei'. No Markdown or explanation.",
              rule: .json(.object(["title": .string("Call Mei"), "at": .string("2026-10-14T09:15"), "repeats": .bool(false)]))),
        .init(id: "json-en-2", category: "JSON extraction", language: "en",
              prompt: "Records in order: ticket 3 is open; ticket 5 is closed; ticket 8 is open. Output only a JSON array containing the integer IDs of the open tickets, preserving their order. No Markdown or explanation.",
              rule: .json(.array([.int(3), .int(8)]))),
        .init(id: "json-zh-1", category: "JSON extraction", language: "zh",
              prompt: "解析计时器请求，不执行操作。请求：设一个2分30秒的计时器，标签为泡茶。只输出一个JSON对象，恰好包含seconds（总秒数，整数）和label（标签，字符串）两个键。不要Markdown或解释。",
              rule: .json(.object(["seconds": .int(150), "label": .string("泡茶")]))),
        .init(id: "json-zh-2", category: "JSON extraction", language: "zh",
              prompt: "记录只有：姓名李明，电话未提供。只输出一个JSON对象，恰好包含name和phone两个键。name使用记录中的姓名字符串，phone使用JSON的null，不要猜测。不要Markdown或解释。",
              rule: .json(.object(["name": .string("李明"), "phone": .null]))),

        .init(id: "instructions-en-1", category: "Exact instructions", language: "en",
              prompt: "Sort these words in ascending alphabetical order: zebra, apple, lemon. Output only the three lowercase words separated by commas, with no spaces, quotation marks, or final punctuation.",
              rule: .exact("apple,lemon,zebra")),
        .init(id: "instructions-en-2", category: "Exact instructions", language: "en",
              prompt: "Read the sequence: red blue red green blue red. Output only the lowercase color word that occurs exactly three times. No punctuation or explanation.",
              rule: .exact("red")),
        .init(id: "instructions-zh-1", category: "Exact instructions", language: "zh",
              prompt: "对字符串“春3夏8秋1冬”删除所有数字，按原顺序保留汉字。只输出处理后的字符串，不要引号、空格、标点或解释。",
              rule: .exact("春夏秋冬")),
        .init(id: "instructions-zh-2", category: "Exact instructions", language: "zh",
              prompt: "规则：标记为绿色就输出GO，标记为红色就输出STOP。当前标记为红色。只输出规则要求的大写英文单词，不要引号、标点或解释。",
              rule: .exact("STOP")),

        .init(id: "translation-en-1", category: "Translation selection", language: "en",
              prompt: "Choose the Chinese translation of 'She has not arrived yet.' A: 她刚刚离开。 B: 她还没到。 C: 她已经到了。 D: 她从未来过。 Reply with only the uppercase option letter.",
              rule: .exact("B")),
        .init(id: "translation-en-2", category: "Translation selection", language: "en",
              prompt: "Choose the Chinese translation of 'I used to live here.' A: 我将要住在这里。 B: 我从未住在这里。 C: 我以前住在这里。 D: 我必须住在这里。 Reply with only the uppercase option letter.",
              rule: .exact("C")),
        .init(id: "translation-zh-1", category: "Translation selection", language: "zh",
              prompt: "选择“请在周五之前回复。”的正确英文翻译。A: Please reply before Friday. B: Please reply after Friday. C: Please reply only on Friday. D: Please do not reply. 只输出选项的大写英文字母，不要标点或解释。",
              rule: .exact("A")),
        .init(id: "translation-zh-2", category: "Translation selection", language: "zh",
              prompt: "选择“除员工外禁止进入。”的正确英文翻译。A: Everyone may enter. B: Staff may not enter. C: Visitors only. D: Staff only. 只输出选项的大写英文字母，不要标点或解释。",
              rule: .exact("D")),

        .init(id: "knowledge-en-1", category: "Logic and knowledge", language: "en",
              prompt: "Which planet has the greatest mass in our Solar System? A: Earth. B: Mars. C: Jupiter. D: Venus. Reply with only the uppercase option letter.",
              rule: .exact("C")),
        .init(id: "knowledge-en-2", category: "Logic and knowledge", language: "en",
              prompt: "In this fictional world, every ravon is blue, and no blue object is heavy. Can a ravon be heavy under these rules? Reply with exactly 'yes' or 'no', without quotation marks or explanation.",
              rule: .exact("no")),
        .init(id: "knowledge-zh-1", category: "Logic and knowledge", language: "zh",
              prompt: "以下哪一种动物是哺乳动物？A：鲨鱼。B：鲸鱼。C：章鱼。D：金鱼。只输出选项的大写英文字母，不要标点或解释。",
              rule: .exact("B")),
        .init(id: "knowledge-zh-2", category: "Logic and knowledge", language: "zh",
              prompt: "已知所有方块都是红色，并且有一个红色物体。仅根据这些条件，能否确定这个物体一定是方块？只输出“能”或“不能”，不要引号、标点或解释。",
              rule: .exact("不能")),
    ]

    /// Surrounding whitespace is ignored. Exact cases otherwise require the entire answer;
    /// JSON cases allow key order/whitespace differences but require identical values and types.
    /// Fences, explanations, missing/extra keys, and extra answers are not accepted.
    public static func evaluate(_ prompt: Prompt, reply: String) -> Evaluation {
        let answer = reply.trimmingCharacters(in: .whitespacesAndNewlines)
        let passed: Bool
        switch prompt.rule {
        case .exact(let expected): passed = answer == expected
        case .json(let expected): passed = (try? JSONValue.parse(answer)) == expected
        }
        return Evaluation(passed: passed, expected: prompt.expected)
    }
}
