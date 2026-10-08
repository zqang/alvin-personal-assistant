import XCTest
@testable import AssistantKit

final class TurnTextTests: XCTestCase {
    func testNormalizesWidthCasePunctuationAndSpaces() {
        XCTAssertEqual(TurnText.normalized("  What's the Weather?  "), "whatstheweather")
        // NFKC folds full-width forms and ligatures.
        XCTAssertEqual(TurnText.normalized("ＨＥＬＬＯ，ｗｏｒｌｄ！"), "helloworld")
        XCTAssertEqual(TurnText.normalized("ﬁve ﬂights"), "fiveflights")
        XCTAssertEqual(TurnText.normalized("明天 天气 怎么样？"), "明天天气怎么样")
        XCTAssertEqual(TurnText.normalized(" .,!? "), "")
    }

    func testSameRequest() {
        XCTAssertTrue(TurnText.sameRequest("what's the weather", "What's the weather?"))
        XCTAssertTrue(TurnText.sameRequest("明天天气怎么样", "明天天气怎么样？"))
        XCTAssertTrue(TurnText.sameRequest("ok   google", "OK, Google."))
        XCTAssertFalse(TurnText.sameRequest("what's the weather", "what's the weather in Tokyo"))
        XCTAssertFalse(TurnText.sameRequest("call mom at five pm", "call mom at 5 p.m."))
    }
}

final class EarlyReplyTests: XCTestCase {
    typealias Action = EarlyReplyCoordinator.Action

    /// Ticks every `step` seconds from the clock's time up to `end`, returning each action
    /// with the time it was produced.
    private func ticks(
        _ early: inout EarlyReplyCoordinator,
        _ clock: ManualClock,
        until end: TimeInterval,
        step: TimeInterval = 0.01
    ) -> [(time: TimeInterval, action: Action)] {
        let start = clock.now
        var produced: [(time: TimeInterval, action: Action)] = []
        var index = 1
        while start + Double(index) * step <= end + 1e-9 {
            clock.advance(to: start + Double(index) * step)
            produced += early.tick(at: clock.now).map { (time: clock.now, action: $0) }
            index += 1
        }
        return produced
    }

    func testTentativeReplyStartsWhenTheTranscriptHasBeenStableForTheDelay() throws {
        let clock = ManualClock(now: 10)
        var early = EarlyReplyCoordinator(enabled: true)
        XCTAssertEqual(early.currentDelay, 0.35)
        XCTAssertEqual(early.transcriptChanged("what's the", at: clock.now), [])
        clock.advance(by: 0.2)
        XCTAssertEqual(early.transcriptChanged("what's the weather", at: clock.now), [])
        let produced = ticks(&early, clock, until: 12)
        XCTAssertEqual(produced.map(\.action), [.startTentative(text: "what's the weather")], "once per text")
        let started = try XCTUnwrap(produced.first?.time)
        XCTAssertEqual(started, 10.2 + 0.35, accuracy: 0.0101)
        XCTAssertGreaterThanOrEqual(started, 10.2 + 0.35 - 1e-9)
        XCTAssertEqual(early.tentativeText, "what's the weather")
    }

    func testSentenceEndShortensTheWait() throws {
        let clock = ManualClock()
        var early = EarlyReplyCoordinator(enabled: true)
        _ = early.transcriptChanged("What time is it?", at: clock.now)
        let produced = ticks(&early, clock, until: 1)
        XCTAssertEqual(produced.map(\.action), [.startTentative(text: "What time is it?")])
        XCTAssertEqual(try XCTUnwrap(produced.first?.time), 0.25, accuracy: 0.0101)
        XCTAssertEqual(early.wait(for: "我想知道时间。"), 0.25, accuracy: 1e-9)
        XCTAssertEqual(early.wait(for: "what time is it"), 0.35, accuracy: 1e-9)
    }

    func testWaitsForEnoughWords() {
        for (text, starts) in [
            ("hello", false),
            ("hi there", true),
            ("你好", false),
            ("你好吗", true),
            ("ok?", false),
            ("  ", false),
        ] {
            let clock = ManualClock()
            var early = EarlyReplyCoordinator(enabled: true)
            _ = early.transcriptChanged(text, at: clock.now)
            let produced = ticks(&early, clock, until: 3)
            XCTAssertEqual(produced.map(\.action), starts ? [.startTentative(text: text.trimmingCharacters(in: .whitespaces))] : [], text)
        }
    }

    func testThresholdsComeFromThePolicy() {
        var policy = EarlyReplyPolicy()
        policy.minimumWords = 3
        policy.minimumCJKCharacters = 2
        let clock = ManualClock()
        var early = EarlyReplyCoordinator(policy: policy, enabled: true)
        _ = early.transcriptChanged("hi there", at: clock.now)
        XCTAssertEqual(ticks(&early, clock, until: 2).map(\.action), [])
        _ = early.transcriptChanged("你好", at: clock.now)
        XCTAssertEqual(ticks(&early, clock, until: 4).map(\.action), [.startTentative(text: "你好")])
    }

    func testADifferentRequestCancelsTheTentativeReply() {
        let clock = ManualClock()
        var early = EarlyReplyCoordinator(enabled: true)
        _ = early.transcriptChanged("what's the weather", at: clock.now)
        XCTAssertEqual(ticks(&early, clock, until: 0.5).map(\.action), [.startTentative(text: "what's the weather")])
        // Only punctuation and capitals changed: the same request keeps running.
        XCTAssertEqual(early.transcriptChanged("What's the weather?", at: clock.now), [])
        XCTAssertEqual(early.tentativeText, "what's the weather")
        XCTAssertEqual(ticks(&early, clock, until: 1).map(\.action), [])

        clock.advance(to: 1.2)
        XCTAssertEqual(early.transcriptChanged("What's the weather in Tokyo", at: clock.now), [.cancelTentative])
        XCTAssertNil(early.tentativeText)
        let produced = ticks(&early, clock, until: 2)
        XCTAssertEqual(produced.map(\.action), [.startTentative(text: "What's the weather in Tokyo")])
        XCTAssertEqual(produced.first?.time ?? 0, 1.55, accuracy: 0.0101)
    }

    func testAdoptsTheSameRequestAtCommit() {
        let clock = ManualClock()
        var early = EarlyReplyCoordinator(enabled: true)
        _ = early.transcriptChanged("what's the weather in tokyo", at: clock.now)
        _ = ticks(&early, clock, until: 0.5)
        // The refined transcript adds punctuation and capitals.
        XCTAssertEqual(early.commit(finalText: "What's the weather in Tokyo?", at: 0.9), [.adopt])
        XCTAssertNil(early.tentativeText)
    }

    func testStartsFreshWhenTheCommittedTextDiffers() {
        let clock = ManualClock()
        var early = EarlyReplyCoordinator(enabled: true)
        _ = early.transcriptChanged("remind me to call mom at five pm", at: clock.now)
        _ = ticks(&early, clock, until: 0.5)
        XCTAssertEqual(
            early.commit(finalText: "Remind me to call Mom at 5 p.m.", at: 0.9),
            [.cancelTentative, .startFresh(text: "Remind me to call Mom at 5 p.m.")]
        )
        XCTAssertNil(early.tentativeText)
    }

    func testCommitWithoutATentativeReplyStartsFresh() {
        var early = EarlyReplyCoordinator(enabled: true)
        _ = early.transcriptChanged("hi", at: 0)
        XCTAssertEqual(early.tick(at: 5), [])
        XCTAssertEqual(early.commit(finalText: " hi ", at: 5), [.startFresh(text: "hi")])
    }

    func testNeverAdoptsBeforeCommitAndRunsOneTentativeAtATime() {
        // Partial results as a recognizer streams them, with pauses mid-sentence.
        let partials: [(TimeInterval, String)] = [
            (0.0, "set"), (0.2, "set a"), (0.4, "set a timer"), (0.9, "set a timer for"),
            (1.0, "set a timer for ten"), (1.5, "set a timer for ten minutes"),
            (1.6, "Set a timer for ten minutes."), (2.4, "set a timer for 10 minutes"),
            (3.0, "set a timer for 10 minutes please"),
        ]
        let clock = ManualClock()
        var early = EarlyReplyCoordinator(enabled: true)
        var actions: [Action] = []
        for (time, text) in partials {
            actions += ticks(&early, clock, until: time).map(\.action)
            actions += early.transcriptChanged(text, at: time)
        }
        actions += ticks(&early, clock, until: 4).map(\.action)
        XCTAssertFalse(actions.contains(.adopt))
        var running = false
        for action in actions {
            switch action {
            case .startTentative:
                XCTAssertFalse(running, "two tentative replies at once")
                running = true
            case .cancelTentative:
                XCTAssertTrue(running, "cancelled nothing")
                running = false
            case .adopt, .startFresh:
                XCTFail("only commit adopts or starts fresh")
            }
        }
        XCTAssertTrue(running)
        XCTAssertEqual(actions.filter { $0 == .cancelTentative }.count, 4)
        XCTAssertEqual(early.tentativeText, "set a timer for 10 minutes please")
        XCTAssertEqual(early.commit(finalText: "Set a timer for 10 minutes, please.", at: 4), [.adopt])
    }

    func testARequestStartsOncePerTurn() {
        let clock = ManualClock()
        var early = EarlyReplyCoordinator(enabled: true)
        _ = early.transcriptChanged("turn on the lights", at: clock.now)
        XCTAssertEqual(ticks(&early, clock, until: 0.5).count, 1)
        XCTAssertEqual(early.transcriptChanged("turn on the light", at: clock.now), [.cancelTentative])
        XCTAssertEqual(early.transcriptChanged("turn on the lights", at: clock.now), [])
        // The recognizer flipped back; the request already had its tentative reply this turn.
        XCTAssertEqual(ticks(&early, clock, until: 2).map(\.action), [])
        XCTAssertEqual(early.commit(finalText: "turn on the lights", at: 2), [.startFresh(text: "turn on the lights")])
        // A new turn may start the same request again.
        _ = early.transcriptChanged("turn on the lights", at: clock.now)
        XCTAssertEqual(ticks(&early, clock, until: 3).map(\.action), [.startTentative(text: "turn on the lights")])
    }

    func testDisabledNeverStartsTentativeReplies() {
        let clock = ManualClock()
        var early = EarlyReplyCoordinator(enabled: false)
        _ = early.transcriptChanged("what's on my calendar today", at: clock.now)
        XCTAssertEqual(ticks(&early, clock, until: 3).map(\.action), [])
        XCTAssertEqual(early.commit(finalText: "what's on my calendar today", at: 3), [.startFresh(text: "what's on my calendar today")])
    }

    func testTurningOffStillSettlesTheRunningTentative() {
        let clock = ManualClock()
        var early = EarlyReplyCoordinator(enabled: true)
        _ = early.transcriptChanged("what's on my calendar", at: clock.now)
        _ = ticks(&early, clock, until: 0.5)
        early.enabled = false
        XCTAssertEqual(early.commit(finalText: "What's on my calendar?", at: 1), [.adopt])
    }

    func testResetForgetsTheTurnButNotTheTuner() {
        let clock = ManualClock()
        var early = EarlyReplyCoordinator(enabled: true)
        for _ in 0..<10 {
            _ = early.transcriptChanged("tell me a joke", at: clock.now)
            XCTAssertEqual(ticks(&early, clock, until: clock.now + 0.5).count, 1)
            early.reset()
            XCTAssertNil(early.tentativeText)
        }
        // Dropped replies aren't judged.
        XCTAssertEqual(early.currentDelay, 0.35)
        XCTAssertEqual(early.tick(at: clock.now + 10), [], "reset forgot the transcript")
    }

    // MARK: Tuner

    /// Runs one turn whose tentative reply is adopted or discarded at commit.
    private func settle(_ early: inout EarlyReplyCoordinator, _ clock: ManualClock, discarded: Bool) {
        _ = early.transcriptChanged("how far is the moon", at: clock.now)
        let started = ticks(&early, clock, until: clock.now + 1).map(\.action)
        XCTAssertEqual(started, [.startTentative(text: "how far is the moon")])
        _ = early.commit(finalText: discarded ? "how far is the moon from mars" : "How far is the moon?", at: clock.now)
    }

    func testTunerRaisesTheDelayWhenTentativeRepliesAreOftenDiscarded() {
        let clock = ManualClock()
        var early = EarlyReplyCoordinator(enabled: true)
        var delays: [TimeInterval] = []
        for _ in 0..<12 {
            settle(&early, clock, discarded: true)
            delays.append(early.currentDelay)
        }
        // It waits for five settled replies, then steps up to the bound and stays there.
        XCTAssertEqual(delays, [0.35, 0.35, 0.35, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.6, 0.6, 0.6])

        // The tuned delay is the one used, and the sentence-end wait moves with it.
        _ = early.transcriptChanged("is it far", at: clock.now)
        let start = clock.now
        let produced = ticks(&early, clock, until: start + 1)
        XCTAssertEqual((produced.first?.time ?? 0) - start, 0.6, accuracy: 0.0101)
        XCTAssertEqual(early.wait(for: "Is it far?"), 0.5, accuracy: 1e-9)
    }

    func testTunerLowersTheDelayWhenTentativeRepliesAreAlmostAlwaysAdopted() {
        let clock = ManualClock()
        var early = EarlyReplyCoordinator(enabled: true)
        var delays: [TimeInterval] = []
        for _ in 0..<8 {
            settle(&early, clock, discarded: false)
            delays.append(early.currentDelay)
        }
        XCTAssertEqual(delays, [0.35, 0.35, 0.35, 0.35, 0.3, 0.25, 0.25, 0.25])
        XCTAssertEqual(early.wait(for: "Is it far?"), 0.15, accuracy: 1e-9)
    }

    func testTunerHoldsTheDelayInsideTheBand() {
        let clock = ManualClock()
        var early = EarlyReplyCoordinator(enabled: true)
        // One discard in five: 20%, between 10% and 30%.
        for index in 0..<40 {
            settle(&early, clock, discarded: index % 5 == 4)
            XCTAssertEqual(early.currentDelay, 0.35)
        }
    }

    func testTunerRaisesTheDelayAboveThirtyPercentDiscards() {
        let clock = ManualClock()
        var early = EarlyReplyCoordinator(enabled: true)
        var delays: [TimeInterval] = []
        // Two discards in every five: 40%.
        for index in 0..<12 {
            settle(&early, clock, discarded: index % 5 == 0 || index % 5 == 2)
            delays.append(early.currentDelay)
        }
        XCTAssertEqual(delays, [0.35, 0.35, 0.35, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.6, 0.6, 0.6])
    }

    func testTunerThresholdsAreStrict() {
        let clock = ManualClock()
        var early = EarlyReplyCoordinator(enabled: true)
        // Every prefix from five replies on stays within 10%...30%, ending at exactly 30%.
        for discarded in [false, false, true, false, false, false, true, false, false, true] {
            settle(&early, clock, discarded: discarded)
            XCTAssertEqual(early.currentDelay, 0.35)
        }

        early = EarlyReplyCoordinator(enabled: true)
        // One discard, then adopts: 20% down to exactly 10% after ten replies, then 1 in 11.
        for index in 0..<10 {
            settle(&early, clock, discarded: index == 0)
            XCTAssertEqual(early.currentDelay, 0.35)
        }
        settle(&early, clock, discarded: false)
        XCTAssertEqual(early.currentDelay, 0.3)
    }

    func testTunerStaysWithinBoundsForAnyHistory() {
        let clock = ManualClock()
        var early = EarlyReplyCoordinator(enabled: true)
        var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
        var seen: Set<TimeInterval> = []
        for _ in 0..<300 {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            settle(&early, clock, discarded: (seed >> 33) % 3 == 0)
            XCTAssertGreaterThanOrEqual(early.currentDelay, 0.25)
            XCTAssertLessThanOrEqual(early.currentDelay, 0.6)
            seen.insert(early.currentDelay)
        }
        XCTAssertGreaterThan(seen.count, 1, "the tuner moved")
    }

    func testInitialDelayIsClampedToTheBounds() {
        var policy = EarlyReplyPolicy()
        policy.delay = 2
        XCTAssertEqual(EarlyReplyCoordinator(policy: policy, enabled: true).currentDelay, 0.6)
    }

    // MARK: Simulated voice turn

    private struct TurnResult {
        var earlyStart: TimeInterval?
        var commit: TimeInterval
        var firstAudio: TimeInterval
    }

    /// A voice turn driven like `VoiceSession`: a 50 ms ticker, `TurnDetector` deciding when the
    /// turn ends, and a reply that needs `readyAfter` seconds from its start to its first audio.
    /// A tentative reply's audio is held until it is adopted.
    private func simulate(
        earlyStart: Bool,
        readyAfter: TimeInterval,
        partials: [(TimeInterval, String)],
        finalText: ((String) -> String)? = nil
    ) throws -> TurnResult {
        let clock = ManualClock()
        var detector = TurnDetector(silenceTimeout: 0.9)
        var early = EarlyReplyCoordinator(enabled: earlyStart)
        var pending = partials[...]
        var replyStart: TimeInterval?
        var earlyAt: TimeInterval?
        for index in 0...200 {
            clock.advance(to: Double(index) * 0.05)
            let now = clock.now
            while let next = pending.first, next.0 <= now + 1e-9 {
                pending = pending.dropFirst()
                let (time, text) = next
                detector.transcriptChanged(text, at: time)
                for action in early.transcriptChanged(text, at: time) where action == .cancelTentative {
                    replyStart = nil
                }
            }
            for case .startTentative in early.tick(at: now) {
                replyStart = now
                earlyAt = earlyAt ?? now
            }
            guard detector.shouldEndTurn(at: now) else { continue }
            let text = finalText?(detector.transcript) ?? detector.transcript
            var firstAudio: TimeInterval?
            for action in early.commit(finalText: text, at: now) {
                switch action {
                case .adopt:
                    let start = try XCTUnwrap(replyStart)
                    // The commit gate opens now; audio plays once the reply is ready.
                    firstAudio = max(start + readyAfter, now)
                case .startFresh:
                    firstAudio = now + readyAfter
                case .cancelTentative:
                    replyStart = nil
                case .startTentative:
                    XCTFail("commit never starts a tentative reply")
                }
            }
            return TurnResult(earlyStart: earlyAt, commit: now, firstAudio: try XCTUnwrap(firstAudio))
        }
        throw TurnNeverEnded()
    }

    private struct TurnNeverEnded: Error {}

    private let weather: [(TimeInterval, String)] = [
        (0.0, "what's"), (0.3, "what's the weather"), (0.6, "what's the weather in Tokyo"),
    ]

    func testEarlyStartBringsFirstAudioForwardByCommitMinusEarlyStart() throws {
        for readyAfter in [0.55, 0.8, 1.2, 2.5] {
            let baseline = try simulate(earlyStart: false, readyAfter: readyAfter, partials: weather)
            let result = try simulate(earlyStart: true, readyAfter: readyAfter, partials: weather)
            let early = try XCTUnwrap(result.earlyStart)
            XCTAssertNil(baseline.earlyStart)
            XCTAssertEqual(result.commit, baseline.commit, accuracy: 1e-9, "early start doesn't move the commit")
            XCTAssertEqual(early, 0.95, accuracy: 0.051)
            XCTAssertEqual(result.commit, 1.5, accuracy: 0.051)
            XCTAssertEqual(baseline.firstAudio - result.firstAudio, result.commit - early, accuracy: 1e-9, "\(readyAfter)")
        }
    }

    func testAReplyReadyBeforeTheCommitPlaysAtTheCommit() throws {
        let baseline = try simulate(earlyStart: false, readyAfter: 0.3, partials: weather)
        let result = try simulate(earlyStart: true, readyAfter: 0.3, partials: weather)
        XCTAssertEqual(result.firstAudio, result.commit, accuracy: 1e-9)
        XCTAssertEqual(baseline.firstAudio - result.firstAudio, 0.3, accuracy: 1e-9)
    }

    func testRefinedTranscriptOfTheSameRequestIsAdopted() throws {
        let baseline = try simulate(earlyStart: false, readyAfter: 1.2, partials: weather) { _ in "What's the weather in Tokyo?" }
        let result = try simulate(earlyStart: true, readyAfter: 1.2, partials: weather) { _ in "What's the weather in Tokyo?" }
        let early = try XCTUnwrap(result.earlyStart)
        XCTAssertEqual(baseline.firstAudio - result.firstAudio, result.commit - early, accuracy: 1e-9)
    }

    func testALateChangeCostsNothingComparedWithNoEarlyStart() throws {
        let partials = weather + [(1.0, "what's the weather in Tokyo tomorrow")]
        let baseline = try simulate(earlyStart: false, readyAfter: 1.2, partials: partials)
        let result = try simulate(earlyStart: true, readyAfter: 1.2, partials: partials)
        // The first tentative reply was cancelled at 1.0; the second started at 1.35.
        XCTAssertEqual(result.commit, 1.9, accuracy: 0.051)
        XCTAssertLessThanOrEqual(result.firstAudio, baseline.firstAudio + 1e-9)
        XCTAssertEqual(baseline.firstAudio - result.firstAudio, 0.55, accuracy: 0.051)
    }

    func testACorrectedRequestStartsFreshWithoutLosingTime() throws {
        // The second pass hears something else: the tentative reply is replaced at the commit.
        let corrected: (String) -> String = { _ in "What's the weather in Kyoto?" }
        let baseline = try simulate(earlyStart: false, readyAfter: 1.2, partials: weather, finalText: corrected)
        let result = try simulate(earlyStart: true, readyAfter: 1.2, partials: weather, finalText: corrected)
        XCTAssertEqual(result.firstAudio, baseline.firstAudio, accuracy: 1e-9)
    }
}
