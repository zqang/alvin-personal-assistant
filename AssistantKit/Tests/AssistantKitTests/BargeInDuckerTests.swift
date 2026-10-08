import XCTest
@testable import AssistantKit

final class BargeInDuckerTests: XCTestCase {
    /// Microphone levels arrive every 20 ms.
    private let frame: TimeInterval = 0.02
    private let quiet: Float = -70
    private let voice: Float = -35

    /// A ducker whose noise floor has settled on `floor` dBFS.
    private func settledDucker(on floor: Float = -70) -> BargeInDucker {
        var ducker = BargeInDucker()
        for index in 0..<3000 {
            XCTAssertEqual(ducker.level(floor, at: Double(index) * frame - 100, speaking: false), BargeInDucker.Output.none)
        }
        return ducker
    }

    /// Feeds `level` every frame from `start` (inclusive) to `end` (exclusive) and returns the
    /// outputs other than `.none`, each with its time.
    private func feed(
        _ ducker: inout BargeInDucker,
        _ level: Float,
        from start: TimeInterval,
        to end: TimeInterval,
        speaking: Bool = true
    ) -> [(time: TimeInterval, output: BargeInDucker.Output)] {
        var outputs: [(time: TimeInterval, output: BargeInDucker.Output)] = []
        var index = 0
        while start + Double(index) * frame < end - 1e-9 {
            let time = start + Double(index) * frame
            let output = ducker.level(level, at: time, speaking: speaking)
            if output != .none { outputs.append((time: time, output: output)) }
            index += 1
        }
        return outputs
    }

    func testDucksAfterSustainedVoiceWhileSpeaking() throws {
        var ducker = settledDucker()
        XCTAssertEqual(ducker.noiseFloor, quiet, accuracy: 0.5)
        XCTAssertTrue(feed(&ducker, quiet, from: 0, to: 1).isEmpty)
        let outputs = feed(&ducker, voice, from: 1, to: 1.5)
        XCTAssertEqual(outputs.map(\.output), [.duck])
        let ducked = try XCTUnwrap(outputs.first?.time)
        // The first frame at or after 150 ms of voice.
        XCTAssertGreaterThanOrEqual(ducked - 1, 0.15 - 1e-9)
        XCTAssertLessThan(ducked - 1, 0.15 + frame)
        XCTAssertTrue(ducker.isDucked)
    }

    func testShortBurstsDontDuck() {
        var ducker = settledDucker()
        var time: TimeInterval = 0
        for _ in 0..<10 {
            // 100 ms of voice, 40 ms of quiet.
            XCTAssertTrue(feed(&ducker, voice, from: time, to: time + 0.1).isEmpty)
            XCTAssertTrue(feed(&ducker, quiet, from: time + 0.1, to: time + 0.14).isEmpty)
            time += 0.14
        }
        XCTAssertFalse(ducker.isDucked)
    }

    func testNeedsTheMarginOverTheNoiseFloor() {
        // A fan at -50 dBFS: 12 dB over it isn't the user, 22 dB over it is.
        var ducker = settledDucker(on: -50)
        XCTAssertEqual(ducker.noiseFloor, -50, accuracy: 0.5)
        XCTAssertTrue(feed(&ducker, -38, from: 0, to: 1).isEmpty)
        XCTAssertEqual(feed(&ducker, -28, from: 1, to: 2).map(\.output), [.duck])
    }

    func testMarginIsConfigurable() {
        var ducker = settledDucker(on: -50)
        ducker.marginDB = 10
        XCTAssertEqual(feed(&ducker, -38, from: 0, to: 1).map(\.output), [.duck])
    }

    func testIgnoresLevelsBelowTheVoiceThreshold() {
        // In a silent room 30 dB over the floor can still be too quiet to be a voice.
        var ducker = settledDucker(on: -100)
        XCTAssertTrue(feed(&ducker, -60, from: 0, to: 1).isEmpty)
    }

    func testNeverDucksWhileTheAssistantIsSilent() {
        var ducker = settledDucker()
        XCTAssertTrue(feed(&ducker, voice, from: 0, to: 2, speaking: false).isEmpty)
        XCTAssertFalse(ducker.isDucked)
    }

    func testRestoresOnlyAfterQuiet() throws {
        var ducker = settledDucker()
        XCTAssertEqual(feed(&ducker, voice, from: 0, to: 0.3).map(\.output), [.duck])
        // Still talking long after the duck, with a short pause: playback stays down.
        XCTAssertTrue(feed(&ducker, voice, from: 0.3, to: 1.2).isEmpty)
        XCTAssertTrue(feed(&ducker, quiet, from: 1.2, to: 1.6).isEmpty)
        XCTAssertTrue(feed(&ducker, voice, from: 1.6, to: 2.0).isEmpty, "already ducked")
        let lastVoice = 2.0 - frame
        let outputs = feed(&ducker, quiet, from: 2.0, to: 3.0)
        XCTAssertEqual(outputs.map(\.output), [.restore])
        let restored = try XCTUnwrap(outputs.first?.time)
        XCTAssertGreaterThanOrEqual(restored - lastVoice, 0.6 - 1e-9)
        XCTAssertLessThan(restored - lastVoice, 0.6 + frame)
        XCTAssertFalse(ducker.isDucked)
    }

    func testDucksAgainOnlyAfterAFreshRunOfVoice() throws {
        var ducker = settledDucker()
        XCTAssertEqual(feed(&ducker, voice, from: 0, to: 0.3).map(\.output), [.duck])
        XCTAssertEqual(feed(&ducker, quiet, from: 0.3, to: 1.0).map(\.output), [.restore])
        let outputs = feed(&ducker, voice, from: 1.0, to: 1.5)
        XCTAssertEqual(outputs.map(\.output), [.duck])
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(outputs.first?.time) - 1.0, 0.15 - 1e-9)
    }

    func testConfirmedInterruptionKeepsPlaybackDownUntilReset() {
        var ducker = settledDucker()
        XCTAssertEqual(feed(&ducker, voice, from: 0, to: 0.3).map(\.output), [.duck])
        ducker.interruptionConfirmed()
        XCTAssertTrue(feed(&ducker, quiet, from: 0.3, to: 2).isEmpty)
        XCTAssertTrue(feed(&ducker, quiet, from: 2, to: 2.5, speaking: false).isEmpty)
        XCTAssertTrue(feed(&ducker, voice, from: 2.5, to: 3).isEmpty)
        let floor = ducker.noiseFloor
        ducker.reset()
        XCTAssertFalse(ducker.isDucked)
        XCTAssertEqual(ducker.noiseFloor, floor, "reset keeps the noise floor")
        XCTAssertEqual(feed(&ducker, voice, from: 3, to: 3.5).map(\.output), [.duck])
    }

    func testRestoresWhenPlaybackStopsWhileDucked() {
        var ducker = settledDucker()
        XCTAssertEqual(feed(&ducker, voice, from: 0, to: 0.3).map(\.output), [.duck])
        XCTAssertEqual(feed(&ducker, voice, from: 0.3, to: 1, speaking: false).map(\.output), [.restore])
        XCTAssertFalse(ducker.isDucked)
    }

    func testTimingsAreConfigurable() {
        var ducker = settledDucker()
        ducker.minimumVoice = 0.3
        ducker.restoreAfter = 1.0
        let ducked = feed(&ducker, voice, from: 0, to: 1)
        XCTAssertEqual(ducked.map(\.output), [.duck])
        XCTAssertEqual(ducked.first?.time ?? 0, 0.3, accuracy: frame)
        let lastVoice = 1 - frame
        let restored = feed(&ducker, quiet, from: 1, to: 3)
        XCTAssertEqual(restored.map(\.output), [.restore])
        XCTAssertEqual((restored.first?.time ?? 0) - lastVoice, 1.0, accuracy: frame)
    }
}
