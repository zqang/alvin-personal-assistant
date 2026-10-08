import XCTest
@testable import AssistantKit

final class CuePolicyTests: XCTestCase {
    private enum Step {
        case event(AssistantEvent)
        /// Ticks every 100 ms up to this time.
        case ticks
    }

    private struct Case {
        var name: String
        var enabled = true
        var isVoice = true
        var deep = false
        var fillerDelay: TimeInterval = 1.8
        /// Expected seconds to the first text, passed to `replyStarted` at time 0; nil skips the call.
        var expected: TimeInterval?? = .some(nil)
        var steps: [(TimeInterval, Step)]
        var cues: [(TimeInterval, SpokenCue)]
    }

    /// Plays `steps` in time order, ticking every 100 ms between them, and returns each cue
    /// with its time rounded to milliseconds.
    private func play(_ testCase: Case) -> [(TimeInterval, SpokenCue)] {
        var policy = CuePolicy(enabled: testCase.enabled, isVoice: testCase.isVoice, fillerDelay: testCase.fillerDelay, deep: testCase.deep)
        if case .some(let expected) = testCase.expected {
            policy.replyStarted(at: 0, expectedFirstText: expected)
        }
        var cues: [(TimeInterval, SpokenCue)] = []
        var tick = 0
        for (time, step) in testCase.steps {
            while Double(tick) * 0.1 <= time + 1e-9 {
                let now = Double(tick) * 0.1
                if let cue = policy.tick(at: now) { cues.append(((now * 1000).rounded() / 1000, cue)) }
                tick += 1
            }
            if case .event(let event) = step, let cue = policy.received(event, at: time) {
                cues.append((time, cue))
            }
        }
        return cues
    }

    private let deepRoute = AssistantEvent.routed(RouteDecision(engine: .cloud, mode: .deep, reason: .deepRequested))
    private let standardRoute = AssistantEvent.routed(RouteDecision(engine: .cloud, reason: .cloudDefault, fallback: .local))

    func testMatrix() {
        let cases: [Case] = [
            Case(name: "typed input never cues", isVoice: false, deep: true, expected: 5,
                 steps: [(0, .event(.cue(.deepThinking))), (0.2, .event(.cue(.stillThinking))), (5, .ticks)],
                 cues: []),
            Case(name: "cues turned off", enabled: false,
                 steps: [(0.3, .event(.cue(.checking))), (5, .ticks)],
                 cues: []),
            Case(name: "a tool cue is the one cue",
                 steps: [(0.0, .event(standardRoute)), (0.3, .event(.cue(.checking))), (0.4, .event(.cue(.checking))),
                         (0.5, .event(.cue(.lookingUp))), (5, .ticks)],
                 cues: [(0.3, .checking)]),
            Case(name: "filler after the filler delay",
                 steps: [(0.2, .event(.progress(.responseStarted))), (0.4, .event(.reply(.activity("Thinking")))), (5, .ticks)],
                 cues: [(1.8, .working)]),
            Case(name: "filler early when the reply is expected to be slow", expected: 3.0,
                 steps: [(1.0, .event(.cue(.checking))), (5, .ticks)],
                 cues: [(0.6, .working)]),
            Case(name: "no early filler at exactly 2.5 s", expected: 2.5,
                 steps: [(5, .ticks)],
                 cues: [(1.8, .working)]),
            Case(name: "a short filler delay is kept for a slow reply", fillerDelay: 0.4, expected: 3.0,
                 steps: [(5, .ticks)],
                 cues: [(0.4, .working)]),
            Case(name: "no filler before the reply starts", expected: .none,
                 steps: [(5, .ticks)],
                 cues: []),
            Case(name: "a negative filler delay turns the filler off", fillerDelay: -1,
                 steps: [(5, .ticks), (5.1, .event(.cue(.checking)))],
                 cues: [(5.1, .checking)]),
            Case(name: "a cue after the filler is dropped",
                 steps: [(2.0, .event(.cue(.handingOff))), (5, .ticks)],
                 cues: [(1.8, .working)]),
            Case(name: "text closes the policy",
                 steps: [(1.0, .event(.reply(.text("Sure,")))), (1.2, .event(.cue(.checking))), (5, .ticks)],
                 cues: []),
            Case(name: "the end of the reply closes the policy",
                 steps: [(0.5, .event(.reply(.finished(.completed)))), (0.6, .event(.cue(.checking))), (5, .ticks)],
                 cues: []),
            Case(name: "tool rounds and progress don't close it",
                 steps: [(0.1, .event(.progress(.firstToken))),
                         (0.2, .event(.toolRound(ToolRound(calls: [ToolCallRecord(id: "t1", name: "get_current_time", input: [:], result: "{}")])))),
                         (0.3, .event(.reply(.activity(nil)))), (0.4, .event(.cue(.checking))), (5, .ticks)],
                 cues: [(0.4, .checking)]),
            Case(name: "deep mode may add still thinking", deep: true,
                 steps: [(0, .event(.cue(.deepThinking))), (12.5, .event(.cue(.stillThinking))), (13, .event(.cue(.stillThinking))),
                         (14, .event(.cue(.checking))), (20, .event(.reply(.text("Here's")))), (21, .event(.cue(.stillThinking)))],
                 cues: [(0, .deepThinking), (12.5, .stillThinking)]),
            Case(name: "still thinking before any other cue in deep mode", deep: true,
                 steps: [(0.5, .event(.cue(.stillThinking))), (1.0, .event(.cue(.deepThinking))), (5, .ticks)],
                 cues: [(0.5, .stillThinking), (1.0, .deepThinking)]),
            Case(name: "still thinking is an ordinary cue outside deep mode",
                 steps: [(0.2, .event(.cue(.checking))), (5, .event(.cue(.stillThinking)))],
                 cues: [(0.2, .checking)]),
            Case(name: "a deep route allows still thinking",
                 steps: [(0, .event(deepRoute)), (0, .event(.cue(.deepThinking))), (12.5, .event(.cue(.stillThinking)))],
                 cues: [(0, .deepThinking), (12.5, .stillThinking)]),
            Case(name: "a standard route doesn't",
                 steps: [(0, .event(standardRoute)), (0, .event(.cue(.loadingModel))), (12.5, .event(.cue(.stillThinking)))],
                 cues: [(0, .loadingModel)]),
            Case(name: "a filler in deep mode still leaves room for still thinking", deep: true,
                 steps: [(5, .ticks), (12.5, .event(.cue(.stillThinking)))],
                 cues: [(1.8, .working), (12.5, .stillThinking)]),
        ]
        for testCase in cases {
            let cues = play(testCase)
            XCTAssertEqual(cues.map(\.1), testCase.cues.map(\.1), testCase.name)
            XCTAssertEqual(cues.map(\.0), testCase.cues.map(\.0), testCase.name)
        }
    }

    func testFillerPlaysOnceRelativeToTheReplyStart() {
        var policy = CuePolicy(enabled: true, isVoice: true, fillerDelay: 1.8)
        XCTAssertNil(policy.tick(at: 100), "the reply hasn't started")
        policy.replyStarted(at: 200, expectedFirstText: 1.0)
        XCTAssertNil(policy.tick(at: 201.7))
        XCTAssertEqual(policy.tick(at: 201.8), .working)
        XCTAssertNil(policy.tick(at: 202))
        XCTAssertNil(policy.tick(at: 300))
        XCTAssertNil(policy.received(.cue(.checking), at: 300), "the filler was the one cue")
    }

    func testDefaultIsNotDeep() {
        var policy = CuePolicy(enabled: true, isVoice: true, fillerDelay: 1.8)
        XCTAssertEqual(policy.received(.cue(.stillThinking), at: 0), .stillThinking)
        XCTAssertNil(policy.received(.cue(.stillThinking), at: 1))
    }
}
