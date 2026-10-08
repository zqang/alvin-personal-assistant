import XCTest
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
@testable import AssistantKit

/// Events a consumer task has received so far, readable from the test while it runs.
private final class EventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [AssistantEvent] = []

    var events: [AssistantEvent] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    func append(_ event: AssistantEvent) {
        lock.lock()
        recorded.append(event)
        lock.unlock()
    }

    /// Consumes `stream` on a new task, recording each event.
    func consume(_ stream: AsyncThrowingStream<AssistantEvent, Error>) -> Task<Error?, Never> {
        Task {
            do {
                for try await event in stream { append(event) }
                return nil
            } catch {
                return error
            }
        }
    }
}

/// Yields `before`, waits until `gate` opens, then yields `after` and finishes.
private final class GatedProvider: AssistantProvider, @unchecked Sendable {
    let before: [AssistantEvent]
    let after: [AssistantEvent]
    let gate = CommitGate()
    private let lock = NSLock()
    private var terminations = 0

    init(before: [AssistantEvent], after: [AssistantEvent]) {
        self.before = before
        self.after = after
    }

    var terminationCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return terminations
    }

    func streamEvents(system: String, turns: [ChatTurn]) -> AsyncThrowingStream<AssistantEvent, Error> {
        let before = before, after = after, gate = gate
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for event in before { continuation.yield(event) }
                    try await gate.wait()
                    for event in after { continuation.yield(event) }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { [self] _ in
                task.cancel()
                lock.lock()
                terminations += 1
                lock.unlock()
            }
        }
    }
}

final class OrchestratorTests: XCTestCase {
    private let turns = [ChatTurn(role: .user, text: "Hello")]
    private let answer: [AssistantEvent] = [.reply(.text("Hi there.")), .reply(.finished(.completed))]
    private let cloudFirst = RouteDecision(engine: .cloud, reason: .cloudDefault, fallback: .local)
    private let localFirst = RouteDecision(engine: .local, reason: .localFirst, fallback: .cloud)
    private let networkFallback = RouteDecision(engine: .local, reason: .networkFallback)
    private let escalated = RouteDecision(engine: .cloud, reason: .escalated)

    private static func seconds(_ duration: Duration) -> TimeInterval {
        let parts = duration.components
        return TimeInterval(parts.seconds) + TimeInterval(parts.attoseconds) / 1e18
    }

    private func orchestrator(
        _ decision: RouteDecision,
        cloud: (any AssistantProvider)? = nil,
        deep: (any AssistantProvider)? = nil,
        local: (any AssistantProvider)? = nil,
        timeout: Duration? = .seconds(8),
        clock: ManualClock = ManualClock()
    ) -> Orchestrator {
        Orchestrator(
            decision: decision,
            engines: Orchestrator.Engines(cloud: cloud, deep: deep, local: local),
            firstEventTimeout: timeout
        ) { duration in
            try await clock.sleep(for: Self.seconds(duration))
        }
    }

    private func expectError(
        _ stream: AsyncThrowingStream<AssistantEvent, Error>
    ) async -> (events: [AssistantEvent], error: Error?) {
        var events: [AssistantEvent] = []
        do {
            for try await event in stream { events.append(event) }
            return (events, nil)
        } catch {
            return (events, error)
        }
    }

    // MARK: Routing

    func testRoutedComesFirstThenThePrimarysEvents() async throws {
        let cloud = ScriptedProvider([.progress(.responseStarted)] + answer)
        let local = ScriptedProvider(answer)
        let events = try await collectEvents(orchestrator(cloudFirst, cloud: cloud, local: local).streamEvents(system: "S", turns: turns))
        XCTAssertEqual(events, [.routed(cloudFirst), .progress(.responseStarted)] + answer)
        XCTAssertEqual(cloud.receivedSystems, ["S"])
        XCTAssertEqual(cloud.receivedTurns, [turns])
        XCTAssertEqual(local.callCount, 0)
    }

    func testEachDecisionUsesItsEngine() async throws {
        let cloud = ScriptedProvider([.reply(.text("cloud"))])
        let deep = ScriptedProvider([.cue(.deepThinking), .reply(.text("deep"))])
        let local = ScriptedProvider([.reply(.text("local"))])
        let deepDecision = RouteDecision(engine: .cloud, mode: .deep, reason: .deepRequested, fallback: .local)

        let deepEvents = try await collectEvents(orchestrator(deepDecision, cloud: cloud, deep: deep, local: local).streamEvents(system: "S", turns: turns))
        XCTAssertEqual(deepEvents, [.routed(deepDecision), .cue(.deepThinking), .reply(.text("deep"))])
        let localEvents = try await collectEvents(orchestrator(localFirst, cloud: cloud, deep: deep, local: local).streamEvents(system: "S", turns: turns))
        XCTAssertEqual(localEvents, [.routed(localFirst), .reply(.text("local"))])
        XCTAssertEqual([cloud.callCount, deep.callCount, local.callCount], [0, 1, 1])
    }

    func testMissingEngineThrowsAfterRouting() async {
        let decision = RouteDecision(engine: .cloud, mode: .deep, reason: .deepRequested)
        let result = await expectError(orchestrator(decision, cloud: ScriptedProvider(answer)).streamEvents(system: "S", turns: turns))
        XCTAssertEqual(result.events, [.routed(decision)])
        XCTAssertEqual(result.error as? AssistantError, .missingConfiguration(Orchestrator.missingEngineMessage(.cloud)))
    }

    func testStreamReplyKeepsOnlyReplyEvents() async throws {
        let cloud = ScriptedProvider([.progress(.responseStarted), .cue(.lookingUp)] + answer)
        let replies = try await collect(orchestrator(cloudFirst, cloud: cloud).streamReply(system: "S", turns: turns))
        XCTAssertEqual(replies, [.text("Hi there."), .finished(.completed)])
    }

    // MARK: Connectivity fallback

    func testConnectivityFailureBeforeCommitFallsBackToLocal() async throws {
        let early: [AssistantEvent] = [.progress(.responseStarted), .cue(.lookingUp), .reply(.activity("Searching the web")), .reply(.text(""))]
        let cloud = ScriptedProvider(early, failAfter: early.count, error: URLError(.networkConnectionLost))
        let local = ScriptedProvider(answer)
        let events = try await collectEvents(orchestrator(cloudFirst, cloud: cloud, local: local).streamEvents(system: "S", turns: turns))
        XCTAssertEqual(events, [.routed(cloudFirst)] + early + [.reply(.activity(nil)), .routed(networkFallback)] + answer)
        XCTAssertEqual(local.receivedSystems, ["S"])
        XCTAssertEqual(local.receivedTurns, [turns])
    }

    func testDeepConnectivityFailureFallsBackToStandardLocal() async throws {
        let decision = RouteDecision(engine: .cloud, mode: .deep, reason: .deepAutomatic, fallback: .local)
        let deep = ScriptedProvider([], failAfter: 0, error: URLError(.notConnectedToInternet))
        let local = ScriptedProvider(answer)
        let events = try await collectEvents(orchestrator(decision, deep: deep, local: local).streamEvents(system: "S", turns: turns))
        XCTAssertEqual(events, [.routed(decision), .reply(.activity(nil)), .routed(networkFallback)] + answer)
    }

    func testConnectivityFailureWithoutALocalFallbackIsRethrown() async {
        let noFallback = RouteDecision(engine: .cloud, reason: .complex)
        let local = ScriptedProvider(answer)
        var result = await expectError(
            orchestrator(noFallback, cloud: ScriptedProvider([], failAfter: 0), local: local).streamEvents(system: "S", turns: turns)
        )
        XCTAssertEqual((result.error as? URLError)?.code, .notConnectedToInternet)
        XCTAssertEqual(result.events, [.routed(noFallback)])

        // A local fallback with no on-device engine to run.
        result = await expectError(orchestrator(cloudFirst, cloud: ScriptedProvider([], failAfter: 0)).streamEvents(system: "S", turns: turns))
        XCTAssertEqual((result.error as? URLError)?.code, .notConnectedToInternet)
        XCTAssertEqual(result.events, [.routed(cloudFirst)])
        XCTAssertEqual(local.callCount, 0)
    }

    func testErrorsAfterTextAreRethrown() async {
        let cloud = ScriptedProvider([.reply(.text("Half an "))], failAfter: 1, error: URLError(.networkConnectionLost))
        let local = ScriptedProvider(answer)
        let result = await expectError(orchestrator(cloudFirst, cloud: cloud, local: local).streamEvents(system: "S", turns: turns))
        XCTAssertEqual((result.error as? URLError)?.code, .networkConnectionLost)
        XCTAssertEqual(result.events, [.routed(cloudFirst), .reply(.text("Half an "))])
        XCTAssertEqual(local.callCount, 0)
    }

    func testErrorsAfterAToolRoundAreRethrown() async {
        let round = ToolRound(calls: [ToolCallRecord(id: "toolu_1", name: "set_timer", input: ["seconds": 60], result: "{}")])
        let cloud = ScriptedProvider([.progress(.responseStarted), .toolRound(round)], failAfter: 2, error: URLError(.timedOut))
        let local = ScriptedProvider(answer)
        let result = await expectError(orchestrator(cloudFirst, cloud: cloud, local: local).streamEvents(system: "S", turns: turns))
        XCTAssertEqual((result.error as? URLError)?.code, .timedOut)
        XCTAssertEqual(result.events, [.routed(cloudFirst), .progress(.responseStarted), .toolRound(round)])
        XCTAssertEqual(local.callCount, 0, "falling back would run the timer tool twice")
    }

    func testUnauthorizedNeverFallsBack() async {
        let transport = MockTransport([
            .failure(HTTPStatusError(statusCode: 401, body: #"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#)),
        ])
        let cloud = LegacyProviderAdapter(ClaudeProvider(configuration: ClaudeConfiguration(apiKey: "bad"), transport: transport))
        let local = ScriptedProvider(answer)
        let result = await expectError(orchestrator(cloudFirst, cloud: cloud, local: local).streamEvents(system: "S", turns: turns))
        guard case .api(_, let status, let type, _)? = result.error as? AssistantError else {
            return XCTFail("Expected an API error, got \(String(describing: result.error))")
        }
        XCTAssertEqual(status, 401)
        XCTAssertEqual(type, "authentication_error")
        XCTAssertEqual(result.events, [.routed(cloudFirst)])
        XCTAssertEqual(transport.requests.count, 1)
        XCTAssertEqual(local.callCount, 0)
    }

    func testServerErrorsDoNotFallBack() async {
        let overloaded = AssistantError.stream(type: "overloaded_error", message: "Overloaded")
        let local = ScriptedProvider(answer)
        let result = await expectError(
            orchestrator(cloudFirst, cloud: ScriptedProvider([], failAfter: 0, error: overloaded), local: local).streamEvents(system: "S", turns: turns)
        )
        XCTAssertEqual(result.error as? AssistantError, overloaded)
        XCTAssertEqual(local.callCount, 0)
    }

    func testFallbackErrorsPropagate() async {
        let cloud = ScriptedProvider([], failAfter: 0, error: URLError(.cannotFindHost))
        let local = ScriptedProvider([.reply(.text("Partial"))], failAfter: 1, error: AssistantError.invalidResponse("model crashed"))
        let result = await expectError(orchestrator(cloudFirst, cloud: cloud, local: local).streamEvents(system: "S", turns: turns))
        XCTAssertEqual(result.error as? AssistantError, .invalidResponse("model crashed"))
        XCTAssertEqual(result.events, [.routed(cloudFirst), .reply(.activity(nil)), .routed(networkFallback), .reply(.text("Partial"))])
        XCTAssertEqual(cloud.callCount, 1, "the fallback runs once and is never retried")
    }

    // MARK: Handoff

    func testHandoffFallsBackToTheCloudWithACue() async throws {
        let local = ScriptedProvider(
            [.progress(.responseStarted), .progress(.toolCallStarted(name: "handoff_to_cloud"))],
            failAfter: 2,
            error: ReplyHandoff(reason: "needs the news")
        )
        let cloud = ScriptedProvider(answer)
        let events = try await collectEvents(orchestrator(localFirst, cloud: cloud, local: local).streamEvents(system: "S", turns: turns))
        XCTAssertEqual(events, [
            .routed(localFirst), .progress(.responseStarted), .progress(.toolCallStarted(name: "handoff_to_cloud")),
            .reply(.activity(nil)), .routed(escalated), .cue(.handingOff),
        ] + answer)
        XCTAssertEqual(cloud.receivedTurns, [turns])
    }

    func testHandoffWithoutACloudFallbackIsRethrown() async {
        let decision = RouteDecision(engine: .local, reason: .noCloudKey)
        let result = await expectError(
            orchestrator(decision, cloud: ScriptedProvider(answer), local: ScriptedProvider([], failAfter: 0, error: ReplyHandoff(reason: "x")))
                .streamEvents(system: "S", turns: turns)
        )
        XCTAssertEqual(result.error as? ReplyHandoff, ReplyHandoff(reason: "x"))
        XCTAssertEqual(result.events, [.routed(decision)])
    }

    func testHandoffAfterTextIsRethrown() async {
        let cloud = ScriptedProvider(answer)
        let local = ScriptedProvider([.reply(.text("Well, "))], failAfter: 1, error: ReplyHandoff(reason: "x"))
        let result = await expectError(orchestrator(localFirst, cloud: cloud, local: local).streamEvents(system: "S", turns: turns))
        XCTAssertEqual(result.error as? ReplyHandoff, ReplyHandoff(reason: "x"))
        XCTAssertEqual(cloud.callCount, 0)
    }

    func testOnDeviceConnectivityErrorsDoNotFallBack() async {
        let cloud = ScriptedProvider(answer)
        let local = ScriptedProvider([], failAfter: 0, error: URLError(.notConnectedToInternet))
        let result = await expectError(orchestrator(localFirst, cloud: cloud, local: local).streamEvents(system: "S", turns: turns))
        XCTAssertEqual((result.error as? URLError)?.code, .notConnectedToInternet)
        XCTAssertEqual(cloud.callCount, 0)
    }

    // MARK: Watchdog

    func testWatchdogFiresOnAHangingTransport() async throws {
        let transport = HangingTransport()
        let cloud = LegacyProviderAdapter(ClaudeProvider(configuration: ClaudeConfiguration(apiKey: "k"), transport: transport))
        let local = ScriptedProvider(answer)
        let clock = ManualClock()
        let consumer = Task { [turns] in
            try await collectEvents(orchestrator(cloudFirst, cloud: cloud, local: local, clock: clock).streamEvents(system: "S", turns: turns))
        }
        let waiting = await waitUntil { transport.requests.count == 1 && clock.sleeperCount == 1 }
        XCTAssertTrue(waiting)
        clock.advance(by: 7)
        XCTAssertEqual(clock.sleeperCount, 1)
        XCTAssertEqual(local.callCount, 0)
        clock.advance(by: 1)

        let events = try await consumer.value
        XCTAssertEqual(events, [.routed(cloudFirst), .reply(.activity(nil)), .routed(networkFallback)] + answer)
        let cancelled = await waitUntil { transport.terminationCount == 1 }
        XCTAssertTrue(cancelled, "the silent request must be cancelled")
    }

    func testWatchdogDoesNotFireAfterAFirstEvent() async {
        let cloud = GatedProvider(before: [.progress(.responseStarted)], after: answer)
        let local = ScriptedProvider(answer)
        let clock = ManualClock()
        let recorder = EventRecorder()
        let consumer = recorder.consume(orchestrator(cloudFirst, cloud: cloud, local: local, clock: clock).streamEvents(system: "S", turns: turns))
        let started = await waitUntil { recorder.events.count == 2 && clock.sleeperCount == 1 }
        XCTAssertTrue(started)

        clock.advance(by: 30)
        XCTAssertEqual(clock.sleeperCount, 0)
        try? await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertEqual(recorder.events, [.routed(cloudFirst), .progress(.responseStarted)])
        XCTAssertEqual(cloud.terminationCount, 0, "a primary that has started is never abandoned")

        cloud.gate.open()
        let error = await consumer.value
        XCTAssertNil(error)
        XCTAssertEqual(recorder.events, [.routed(cloudFirst), .progress(.responseStarted)] + answer)
        XCTAssertEqual(local.callCount, 0)
    }

    func testWatchdogAppliesOnlyWithALocalFallback() async {
        let local = ScriptedProvider(answer)
        let cloud = ScriptedProvider(answer)
        XCTAssertEqual(orchestrator(cloudFirst, cloud: cloud, local: local).watchdogTimeout, .seconds(8))
        XCTAssertEqual(orchestrator(cloudFirst, cloud: cloud, local: local, timeout: .seconds(3)).watchdogTimeout, .seconds(3))
        XCTAssertNil(orchestrator(cloudFirst, cloud: cloud, local: local, timeout: nil).watchdogTimeout)
        XCTAssertNil(orchestrator(cloudFirst, cloud: cloud, local: local, timeout: .zero).watchdogTimeout)
        XCTAssertNil(orchestrator(cloudFirst, cloud: cloud).watchdogTimeout, "no on-device engine to switch to")
        XCTAssertNil(orchestrator(localFirst, cloud: cloud, local: local).watchdogTimeout)
        XCTAssertNil(orchestrator(RouteDecision(engine: .cloud, reason: .complex), cloud: cloud, local: local).watchdogTimeout)
        XCTAssertEqual(Orchestrator(decision: cloudFirst, engines: .init(cloud: cloud, local: local)).firstEventTimeout, .seconds(8))

        // A silent local primary is left alone.
        let silentLocal = GatedProvider(before: [], after: answer)
        let clock = ManualClock()
        let recorder = EventRecorder()
        let consumer = recorder.consume(orchestrator(localFirst, cloud: cloud, local: silentLocal, clock: clock).streamEvents(system: "S", turns: turns))
        let routed = await waitUntil { recorder.events.count == 1 }
        XCTAssertTrue(routed)
        try? await Task.sleep(nanoseconds: 20_000_000)
        XCTAssertEqual(clock.sleeperCount, 0)
        silentLocal.gate.open()
        let error = await consumer.value
        XCTAssertNil(error)
        XCTAssertEqual(recorder.events, [.routed(localFirst)] + answer)
        XCTAssertEqual(cloud.callCount, 0)
    }

    func testRealTimeWatchdog() async throws {
        let cloud = ScriptedProvider([], hangs: true)
        let local = ScriptedProvider(answer)
        let orchestrator = Orchestrator(
            decision: cloudFirst,
            engines: .init(cloud: cloud, local: local),
            firstEventTimeout: .milliseconds(50)
        )
        let events = try await collectEvents(orchestrator.streamEvents(system: "S", turns: turns))
        XCTAssertEqual(events, [.routed(cloudFirst), .reply(.activity(nil)), .routed(networkFallback)] + answer)
        let ended = await waitUntil { cloud.terminationCount == 1 }
        XCTAssertTrue(ended)
    }

    // MARK: Cancellation

    func testCancellationReachesThePrimary() async {
        let cloud = ScriptedProvider([.progress(.responseStarted)], hangs: true)
        let local = ScriptedProvider(answer)
        let clock = ManualClock()
        let recorder = EventRecorder()
        let consumer = recorder.consume(orchestrator(cloudFirst, cloud: cloud, local: local, clock: clock).streamEvents(system: "S", turns: turns))
        let started = await waitUntil { recorder.events.count == 2 }
        XCTAssertTrue(started)
        XCTAssertEqual(cloud.terminationCount, 0)

        consumer.cancel()
        _ = await consumer.value
        let ended = await waitUntil { cloud.terminationCount == 1 }
        XCTAssertTrue(ended, "cancelling the outer stream must end the primary's stream")
        let watchdogEnded = await waitUntil { clock.sleeperCount == 0 }
        XCTAssertTrue(watchdogEnded)
        XCTAssertEqual(local.callCount, 0)
    }

    func testCancellationReachesTheFallback() async {
        let cloud = ScriptedProvider([], failAfter: 0, error: URLError(.notConnectedToInternet))
        let local = ScriptedProvider([.reply(.text("Thinking"))], hangs: true)
        let recorder = EventRecorder()
        let consumer = recorder.consume(orchestrator(cloudFirst, cloud: cloud, local: local).streamEvents(system: "S", turns: turns))
        let answering = await waitUntil { recorder.events.last == .reply(.text("Thinking")) }
        XCTAssertTrue(answering)

        consumer.cancel()
        _ = await consumer.value
        let ended = await waitUntil { local.terminationCount == 1 }
        XCTAssertTrue(ended)
    }

    // MARK: Connectivity errors

    func testConnectivityFailureClassification() {
        let connectivity: [URLError.Code] = [
            .notConnectedToInternet, .networkConnectionLost, .timedOut, .cannotFindHost, .cannotConnectToHost,
            .dnsLookupFailed, .internationalRoamingOff, .dataNotAllowed, .secureConnectionFailed,
        ]
        for code in connectivity {
            XCTAssertTrue(URLError(code).isConnectivityFailure, "\(code)")
        }
        for code in [URLError.Code.cancelled, .badServerResponse, .badURL, .userAuthenticationRequired] {
            XCTAssertFalse(URLError(code).isConnectivityFailure, "\(code)")
        }
        XCTAssertTrue(NSError(domain: NSURLErrorDomain, code: URLError.Code.notConnectedToInternet.rawValue).isConnectivityFailure)
        XCTAssertFalse(NSError(domain: NSURLErrorDomain, code: URLError.Code.cancelled.rawValue).isConnectivityFailure)
        XCTAssertFalse(NSError(domain: "Other", code: URLError.Code.notConnectedToInternet.rawValue).isConnectivityFailure)
        XCTAssertFalse(HTTPStatusError(statusCode: 401, body: "").isConnectivityFailure)
        XCTAssertFalse(AssistantError.api(service: "Anthropic", status: 401, type: "authentication_error", message: "no").isConnectivityFailure)
        XCTAssertFalse(AssistantError.missingAPIKey(service: "Anthropic").isConnectivityFailure)
        XCTAssertFalse(ReplyHandoff(reason: "x").isConnectivityFailure)
        XCTAssertFalse(CancellationError().isConnectivityFailure)
    }
}
