import XCTest
@testable import AssistantKit

final class SentenceChunkerTests: XCTestCase {
    func testSplitsSentencesAsTheyStream() {
        var chunker = SentenceChunker()
        XCTAssertEqual(chunker.append("Hello there"), [])
        XCTAssertEqual(chunker.append("! How are"), ["Hello there!"])
        XCTAssertEqual(chunker.append(" you today? I'm"), ["How are you today?"])
        XCTAssertEqual(chunker.flush(), ["I'm"])
    }

    func testKeepsAbbreviationsAndDecimalsTogether() {
        var chunker = SentenceChunker()
        XCTAssertEqual(
            chunker.append("Dr. Tan said the price is 3.50 dollars. Next"),
            ["Dr. Tan said the price is 3.50 dollars."]
        )
        XCTAssertEqual(chunker.flush(), ["Next"])
    }

    func testWaitsForTheCharacterAfterAPeriod() {
        var chunker = SentenceChunker()
        XCTAssertEqual(chunker.append("Hmm."), [])
        XCTAssertEqual(chunker.flush(), ["Hmm."])
    }

    func testSplitsChineseSentences() {
        var chunker = SentenceChunker()
        XCTAssertEqual(chunker.append("今天天气很好。我们去公园吧！"), ["今天天气很好。", "我们去公园吧！"])
    }

    func testFirstChunkMayEndAtACommaToStartSpeakingSooner() {
        var chunker = SentenceChunker()
        XCTAssertEqual(
            chunker.append("Well, that is a really interesting question to think about, and here is why"),
            ["Well, that is a really interesting question to think about,"]
        )
    }

    func testSplitsOnNewlines() {
        var chunker = SentenceChunker()
        XCTAssertEqual(chunker.append("- Milk\n- Eggs\n"), ["- Milk", "- Eggs"])
    }

    func testCutsVeryLongRunsAtASpace() {
        var chunker = SentenceChunker()
        chunker.hardLimit = 20
        chunker.firstChunkSoftLimit = 1000
        XCTAssertEqual(chunker.append("one two three four five six seven"), ["one two three four"])
    }
}

final class SpeechTextCleanerTests: XCTestCase {
    func testStripsMarkdown() {
        XCTAssertEqual(SpeechTextCleaner.clean("**Bold** and *italic* with `code`"), "Bold and italic with code")
        XCTAssertEqual(SpeechTextCleaner.clean("## Heading\n- item one\n1. item two"), "Heading item one item two")
    }

    func testRemovesLinksAndURLs() {
        XCTAssertEqual(SpeechTextCleaner.clean("See [the docs](https://example.com) or https://example.com/page"), "See the docs or")
    }

    func testRemovesEmoji() {
        XCTAssertEqual(SpeechTextCleaner.clean("Great job 🎉👍 today!"), "Great job today!")
        XCTAssertEqual(SpeechTextCleaner.clean("你好！😊"), "你好！")
    }

    func testFlattensTables() {
        XCTAssertEqual(SpeechTextCleaner.clean("| Name | Age |\n|---|---|\n| Tan | 30 |"), "Name, Age, Tan, 30")
    }
}

final class BargeInDetectorTests: XCTestCase {
    let detector = BargeInDetector()

    func testIgnoresEchoOfTheAssistantsOwnSpeech() {
        let spoken = "The weather in Singapore is sunny with a high of thirty two degrees."
        XCTAssertFalse(detector.isInterruption(heard: "weather in Singapore is sunny", assistantSpeech: spoken))
        XCTAssertTrue(detector.isInterruption(heard: "what about tomorrow", assistantSpeech: spoken))
        XCTAssertFalse(detector.isInterruption(heard: "um", assistantSpeech: spoken))
        XCTAssertTrue(detector.isInterruption(heard: "stop", assistantSpeech: spoken))
    }

    func testChinese() {
        let spoken = "今天新加坡天气晴朗，最高气温三十二度。"
        XCTAssertFalse(detector.isInterruption(heard: "新加坡天气晴朗", assistantSpeech: spoken))
        XCTAssertTrue(detector.isInterruption(heard: "那明天呢", assistantSpeech: spoken))
        XCTAssertTrue(detector.isInterruption(heard: "等一下", assistantSpeech: spoken))
        XCTAssertFalse(detector.isInterruption(heard: "嗯", assistantSpeech: spoken))
    }

    func testVerdicts() {
        let spoken = "The weather in Singapore is sunny today."
        XCTAssertEqual(detector.evaluate(heard: "um", assistantSpeech: spoken), .tooShort)
        XCTAssertEqual(detector.evaluate(heard: "Singapore is sunny", assistantSpeech: spoken), .echo)
        XCTAssertEqual(detector.evaluate(heard: "hold on a second", assistantSpeech: spoken), .interruption)
    }

    func testMonitorJudgesNewWordsAfterEcho() {
        var monitor = InterruptionMonitor()
        let spoken = "The weather in Singapore is sunny with a high of thirty two degrees and light winds in the afternoon."
        XCTAssertNil(monitor.interruption(in: "weather in Singapore is sunny with a high", assistantSpeech: spoken))
        // Judged on the whole transcript this would count as echo (8 of 12 words match).
        XCTAssertEqual(
            monitor.interruption(in: "weather in Singapore is sunny with a high wait what about tomorrow", assistantSpeech: spoken),
            "wait what about tomorrow"
        )
    }

    func testMonitorHearsAStopCommandAfterEcho() {
        var monitor = InterruptionMonitor()
        let spoken = "Here are three ideas for dinner tonight."
        XCTAssertNil(monitor.interruption(in: "three ideas for dinner", assistantSpeech: spoken))
        XCTAssertEqual(monitor.interruption(in: "three ideas for dinner stop", assistantSpeech: spoken), "stop")
        monitor.reset()
        XCTAssertEqual(monitor.userWords(in: "three ideas for dinner stop"), "three ideas for dinner stop")
    }

    func testTokenizer() {
        XCTAssertEqual(SpeechTokenizer.tokens("What's up, Tan 你好!"), ["what's", "up", "tan", "你", "好"])
        XCTAssertEqual(SpeechTokenizer.matchingUnits("hi 你好吗"), ["hi", "你好", "好吗"])
    }
}

final class TurnDetectorTests: XCTestCase {
    func testTurnEndsAfterSilence() {
        var detector = TurnDetector(silenceTimeout: 1.0)
        detector.transcriptChanged("what's the weather", at: 0)
        XCTAssertFalse(detector.shouldEndTurn(at: 0.5))
        XCTAssertTrue(detector.shouldEndTurn(at: 1.05))
    }

    func testNoTranscriptNeverEndsTheTurn() {
        var detector = TurnDetector(silenceTimeout: 1.0)
        detector.audioLevel(-20, at: 0)
        XCTAssertFalse(detector.shouldEndTurn(at: 100))
    }

    func testRecentVoiceDelaysTheEndOfTurn() {
        var detector = TurnDetector(silenceTimeout: 1.0)
        detector.transcriptChanged("tell me a story", at: 0)
        detector.audioLevel(-20, at: 0.9)
        XCTAssertFalse(detector.shouldEndTurn(at: 1.2))
        XCTAssertTrue(detector.shouldEndTurn(at: 1.6))
    }

    func testStableTranscriptEndsTheTurnDespiteNoise() {
        var detector = TurnDetector(silenceTimeout: 1.0)
        detector.transcriptChanged("tell me a story", at: 0)
        detector.audioLevel(-20, at: 2.55)
        XCTAssertTrue(detector.shouldEndTurn(at: 2.6))
    }

    func testWaitsLongerAfterAWordThatPromisesMore() {
        var detector = TurnDetector(silenceTimeout: 1.0)
        detector.transcriptChanged("I want to go there because", at: 0)
        XCTAssertFalse(detector.shouldEndTurn(at: 1.2))
        XCTAssertTrue(detector.shouldEndTurn(at: 1.7))
        XCTAssertEqual(TurnDetector.timeout(for: "我想去那里然后", base: 1.0), 1.6, accuracy: 0.001)
    }

    func testFinishedQuestionEndsSooner() {
        var detector = TurnDetector(silenceTimeout: 1.0)
        detector.transcriptChanged("What time is it?", at: 0)
        XCTAssertTrue(detector.shouldEndTurn(at: 0.85))
    }

    func testResetCanContinueFromEarlierSpeech() {
        var detector = TurnDetector(silenceTimeout: 1.0)
        detector.reset(transcript: "hello", at: 5)
        XCTAssertFalse(detector.shouldEndTurn(at: 5.5))
        XCTAssertTrue(detector.shouldEndTurn(at: 6.1))
    }

    func testVADAdaptsToSteadyNoise() {
        var vad = EnergyVAD()
        for _ in 0..<2000 {
            _ = vad.isVoice(level: -40)
        }
        XCTAssertFalse(vad.isVoice(level: -40))
        XCTAssertTrue(vad.isVoice(level: -18))
    }

    func testVADIgnoresSilence() {
        var vad = EnergyVAD()
        XCTAssertFalse(vad.isVoice(level: -70))
    }
}
