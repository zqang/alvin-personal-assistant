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

    func testLastChangeIsWhenTheTranscriptLastChanged() {
        var detector = TurnDetector()
        XCTAssertNil(detector.lastChange)
        detector.transcriptChanged("hello", at: 1)
        detector.transcriptChanged("hello there", at: 2)
        detector.transcriptChanged("hello there", at: 3)
        detector.audioLevel(-20, at: 4)
        XCTAssertEqual(detector.lastChange, 2)
        detector.reset()
        XCTAssertNil(detector.lastChange)
        detector.reset(transcript: "carried over", at: 7)
        XCTAssertEqual(detector.lastChange, 7)
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

final class FinalTranscriptTests: XCTestCase {
    func testKeepsRefinedTextUnlessItLooksBroken() {
        XCTAssertEqual(FinalTranscript.pick(live: "我想去新加波", refined: "我想去新加坡。"), "我想去新加坡。")
        XCTAssertEqual(FinalTranscript.pick(live: "what's the whether", refined: "What's the weather?"), "What's the weather?")
        XCTAssertEqual(FinalTranscript.pick(live: "what's the whether", refined: nil), "what's the whether")
        XCTAssertEqual(FinalTranscript.pick(live: "hi there", refined: "  "), "hi there")
        XCTAssertEqual(FinalTranscript.pick(live: "hi there", refined: String(repeating: "thank you ", count: 10)), "hi there")
        XCTAssertEqual(FinalTranscript.pick(live: "turn on the kitchen lights please right now", refined: "now"), "turn on the kitchen lights please right now")
    }

    func testChecksOnlyTheRefinedSegmentOfAResumedTurn() {
        let live = "remind me to call mom at five pm"
        XCTAssertEqual(FinalTranscript.pick(live: live, carried: "remind me to call mom", refined: "at 5 p.m."), "remind me to call mom at 5 p.m.")
        // An empty or broken second pass mustn't swallow the new words.
        XCTAssertEqual(FinalTranscript.pick(live: live, carried: "remind me to call mom", refined: ""), live)
        XCTAssertEqual(FinalTranscript.pick(live: live, carried: "remind me to call mom", refined: String(repeating: "hmm ", count: 20)), live)
    }

    func testClipKeepsALeadInButNeverCutsSpeech() {
        let second = 16_000
        func start(_ speechStart: Double, _ end: Double) -> Int {
            FinalTranscript.clipStart(speechStart: Int(speechStart * 16_000), end: Int(end * 16_000), maxLength: 30 * second, leadIn: 5 * second, minLeadIn: 3 * second / 2)
        }
        // Five seconds before the first words when it fits.
        XCTAssertEqual(start(10, 15), 5 * second)
        // Less lead-in to stay within the limit.
        XCTAssertEqual(start(10, 38), 8 * second)
        // Speech too long for the limit (counting the words' lag): the clip runs over and is rejected.
        XCTAssertEqual(start(10, 39.5), 8 * second + second / 2)
        XCTAssertGreaterThan(Int(39.5 * 16_000) - start(10, 39.5), 30 * second)
        XCTAssertEqual(start(1, 5), 0)
    }

    func testVoicedRangeTrimsSilence() throws {
        let second = 16_000
        let tone = (0..<second).map { Float(sin(Double($0) * 0.2)) * 0.1 }
        let clip = [Float](repeating: 0, count: 2 * second) + tone + [Float](repeating: 0, count: second)
        let range = try XCTUnwrap(FinalTranscript.voicedRange(of: clip, margin: 4_800))
        // To within one 30 ms frame.
        XCTAssertLessThanOrEqual(abs(range.lowerBound - (2 * second - 4_800)), 480)
        XCTAssertLessThanOrEqual(abs(range.upperBound - (3 * second + 4_800)), 480)
        XCTAssertNil(FinalTranscript.voicedRange(of: [Float](repeating: 0, count: second)))
    }

    func testSampleWindowKeepsTheLatestAudioAtStablePositions() {
        var window = SampleWindow(capacity: 10)
        func append(_ values: ClosedRange<Int>) {
            let floats = values.map(Float.init)
            floats.withUnsafeBufferPointer { window.append($0) }
        }
        append(0...5)
        append(6...11)   // full: the older half (0...2) is discarded
        XCTAssertEqual(window.count, 12)
        XCTAssertEqual(window.samples(from: 3), (3...11).map(Float.init))
        XCTAssertEqual(window.samples(from: 9), [9, 10, 11])
        XCTAssertEqual(window.samples(from: 2), [], "discarded")
        XCTAssertEqual(window.samples(from: 12), [], "past the end")
        window.reset()
        XCTAssertEqual(window.count, 0)
    }

    func testOnlyEnglishAndMainlandChineseUseQwen() {
        XCTAssertEqual(FinalTranscript.qwenLanguage(forLocale: "zh-CN"), "Chinese")
        XCTAssertEqual(FinalTranscript.qwenLanguage(forLocale: "en-SG"), "English")
        XCTAssertNil(FinalTranscript.qwenLanguage(forLocale: "zh-TW"))
        XCTAssertNil(FinalTranscript.qwenLanguage(forLocale: "ms-MY"))
        var settings = AssistantSettings()
        settings.qwenListening = true
        settings.speechLocale = "zh-TW"
        XCTAssertFalse(settings.usesQwenListening)
        settings.speechLocale = "en-SG"
        XCTAssertTrue(settings.usesQwenListening)
    }
}
