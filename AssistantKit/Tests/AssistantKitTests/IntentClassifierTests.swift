import XCTest
@testable import AssistantKit

final class IntentClassifierTests: XCTestCase {
    private typealias Intents = IntentClassifier.Intents
    private let classifier = IntentClassifier()

    private func assertIntents(_ cases: [(String, Intents)], file: StaticString = #filePath, line: UInt = #line) {
        for (utterance, expected) in cases {
            XCTAssertEqual(classifier.classify(utterance), expected, "\"\(utterance)\"", file: file, line: line)
        }
    }

    func testFreshFacts() {
        assertIntents([
            ("What's the weather like in Singapore?", .freshFacts),
            ("Any news about the election", .freshFacts),
            ("Read me the headlines", .freshFacts),
            ("How did Arsenal score last night", .freshFacts),
            ("What's the Apple share price", .freshFacts),
            ("Exchange rate from euros to dollars", .freshFacts),
            ("Is my flight delayed", .freshFacts),
            ("Is the pharmacy open now", .freshFacts),
            ("Search for vegan restaurants nearby", .freshFacts),
            ("Can you look it up", .freshFacts),
            ("Google the latest iPhone", .freshFacts),
            ("明天天气怎么样", .freshFacts),
            ("今天有什么新闻", .freshFacts),
            ("帮我查一下汇率", .freshFacts),
        ])
    }

    func testDeviceActions() {
        assertIntents([
            ("Remind me to call mum at five", .deviceAction),
            ("Show my reminders", .deviceAction),
            ("Add milk to my to-do list", .deviceAction),
            ("What's on my calendar tomorrow", .deviceAction),
            ("Am I free on Friday afternoon", .deviceAction),
            ("Set a timer for ten minutes", .deviceAction),
            ("Wake me at 6:30", .deviceAction),
            ("Move my meetings to Monday", .deviceAction),
            ("明天早上提醒我开会", .deviceAction),
            ("设一个十分钟的计时", .deviceAction),
            ("帮我定一个闹钟", .deviceAction),
        ])
    }

    func testExplicitDepth() {
        assertIntents([
            ("Think hard about whether I should move", .explicitDepth),
            ("Take your time and tell me what you make of this poem", .explicitDepth),
            ("Give me an in-depth look at sourdough fermentation", .explicitDepth),
            ("Dig into why my bread is dense", .explicitDepth),
            ("请认真想一下这个问题", .explicitDepth),
            ("详细分析一下这篇文章", .explicitDepth),
        ])
    }

    func testComplex() {
        assertIntents([
            ("Compare the two job offers", .complex),
            ("What are the pros and cons of solar panels", .complex),
            ("Walk me through it step by step", .complex),
            ("Help me plan a trip to Japan", .complex),
            ("Why is the sky blue? And why are sunsets red?", .complex),
            ("比较一下这两个手机的优缺点", .complex),
        ])
        let long = Array(repeating: "word", count: 46).joined(separator: " ")
        XCTAssertEqual(classifier.classify(long), .complex)
        XCTAssertEqual(classifier.classify(Array(repeating: "word", count: 45).joined(separator: " ")), [])
        XCTAssertEqual(classifier.classify(String(repeating: "好", count: 81)), .complex)
        XCTAssertEqual(classifier.classify(String(repeating: "嗯", count: 80)), [])
    }

    func testSmallTalk() {
        assertIntents([
            ("Hi", .smallTalk),
            ("Hello there!", .smallTalk),
            ("Thanks, that's great", .smallTalk),
            ("Thank you so much", .smallTalk),
            ("OK", .smallTalk),
            ("How are you today?", .smallTalk),
            ("Good night Alvin", .smallTalk),
            ("What time is it", .smallTalk),
            ("What time is it right now?", .smallTalk),
            ("What\u{2019}s today\u{2019}s date?", .smallTalk),
            ("12 x 4", .smallTalk),
            ("what's 7 plus 8?", .smallTalk),
            ("2 + 2 =", .smallTalk),
            ("你好", .smallTalk),
            ("谢谢你", .smallTalk),
            ("早上好！", .smallTalk),
            ("现在几点了？", .smallTalk),
            ("今天几号", .smallTalk),
            ("3加5等于几", .smallTalk),
        ])
    }

    func testNegatives() {
        assertIntents([
            ("", []),
            ("Tell me a joke", []),
            ("What's the capital of Peru", []),
            ("What's his name again", []),
            ("This is interesting", []),
            ("Give me an explanation of photosynthesis", []),
            ("I did some research on mitochondria", []),
            ("What's one plus one in binary", []),
            ("Who wrote Pride and Prejudice", []),
            ("The underscore character", []),
            ("Is Stockholm in Sweden", []),
            ("That sounds alarming", []),
            ("Planets orbit the sun", []),
            ("1.5", []),
            ("给我讲个故事", []),
        ])
    }

    func testCombinedIntents() {
        XCTAssertEqual(classifier.classify("Hey, set a timer for five minutes"), [.deviceAction, .smallTalk])
        XCTAssertEqual(classifier.classify("Hi! What's the weather today?"), [.freshFacts, .smallTalk])
        XCTAssertEqual(classifier.classify("Think hard and compare these plans"), [.explicitDepth, .complex])
        XCTAssertEqual(classifier.classify("Remind me to check the news"), [.freshFacts, .deviceAction])
    }

    func testSmallTalkMustBeShort() {
        XCTAssertEqual(classifier.classify("Thanks, can you tell me a long story about a dragon"), [])
        XCTAssertEqual(classifier.classify("Thanks for that"), .smallTalk)
        // Twelve CJK characters still count as short; thirteen don't.
        XCTAssertEqual(classifier.classify("谢谢你帮我做了这么多事情"), .smallTalk)
        XCTAssertEqual(classifier.classify("谢谢你帮我做了这么多的事情"), [])
    }

    func testPhrasesMatchWholeWordsWithPlurals() {
        let text = Array("what's on my calendar".unicodeScalars)
        XCTAssertTrue(IntentClassifier.contains(text, phrase: Array("what's on".unicodeScalars)))
        XCTAssertFalse(IntentClassifier.contains(Array("what's one".unicodeScalars), phrase: Array("what's on".unicodeScalars)))
        XCTAssertTrue(IntentClassifier.contains(Array("two timers".unicodeScalars), phrase: Array("timer".unicodeScalars), allowPlural: true))
        XCTAssertFalse(IntentClassifier.contains(Array("two timers".unicodeScalars), phrase: Array("timer".unicodeScalars)))
        XCTAssertFalse(IntentClassifier.contains(Array("this".unicodeScalars), phrase: Array("hi".unicodeScalars), allowPlural: true))
        XCTAssertTrue(IntentClassifier.contains(Array("hi你好".unicodeScalars), phrase: Array("hi".unicodeScalars)))
        XCTAssertTrue(IntentClassifier.contains(Array("帮我查一下".unicodeScalars), phrase: Array("查一下".unicodeScalars)))
    }

    func testMeasures() {
        XCTAssertEqual(IntentClassifier.normalize("  Hello\n  THERE\u{2019}s  "), "hello there's")
        XCTAssertEqual(IntentClassifier.wordCount("hello , there 42"), 3)
        XCTAssertEqual(IntentClassifier.cjkCount(Array("ab你好カナ한".unicodeScalars)), 5)
        XCTAssertTrue(IntentClassifier.isSimpleArithmetic("3 × 4 ÷ 2"))
        XCTAssertTrue(IntentClassifier.isSimpleArithmetic("how much is 10 divided by 4?"))
        XCTAssertFalse(IntentClassifier.isSimpleArithmetic("42"))
        XCTAssertFalse(IntentClassifier.isSimpleArithmetic("x + y"))
        XCTAssertFalse(IntentClassifier.isSimpleArithmetic("2 + two"))
    }

    func testTablesAreLowercaseAndUnique() {
        let tables = [
            IntentClassifier.freshFactsPhrases, IntentClassifier.deviceActionPhrases, IntentClassifier.explicitDepthPhrases,
            IntentClassifier.complexPhrases, IntentClassifier.smallTalkPhrases, IntentClassifier.clockPhrases,
        ]
        for table in tables {
            XCTAssertEqual(Set(table).count, table.count)
            for phrase in table {
                XCTAssertEqual(phrase, IntentClassifier.normalize(phrase))
            }
        }
        XCTAssertTrue(Set(IntentClassifier.timeQualifierPhrases).isSubset(of: IntentClassifier.freshFactsPhrases))
    }
}
