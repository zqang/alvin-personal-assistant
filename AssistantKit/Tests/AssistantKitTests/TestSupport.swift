import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
@testable import AssistantKit

/// A transport that replays scripted responses and records the requests it receives.
final class MockTransport: HTTPStreamingTransport, @unchecked Sendable {
    enum Response {
        case lines([String])
        case failure(Error)
        /// Yields these lines, then never ends; the stream ends only when its reader cancels.
        case hang([String])
    }

    private let lock = NSLock()
    private var responses: [Response]
    private var recorded: [URLRequest] = []

    init(_ responses: [Response]) {
        self.responses = responses
    }

    var requests: [URLRequest] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    func lines(for request: URLRequest) async throws -> AsyncThrowingStream<String, Error> {
        try Self.stream(nextResponse(for: request))
    }

    private func nextResponse(for request: URLRequest) -> Response {
        lock.lock()
        defer { lock.unlock() }
        recorded.append(request)
        return responses.isEmpty ? .failure(URLError(.resourceUnavailable)) : responses.removeFirst()
    }

    /// The body stream for `response`. `onTermination` runs when the stream ends for any reason.
    static func stream(
        _ response: Response,
        onTermination: @escaping @Sendable (AsyncThrowingStream<String, Error>.Continuation.Termination) -> Void = { _ in }
    ) throws -> AsyncThrowingStream<String, Error> {
        switch response {
        case .failure(let error):
            throw error
        case .lines(let lines):
            return AsyncThrowingStream { continuation in
                continuation.onTermination = onTermination
                for line in lines { continuation.yield(line) }
                continuation.finish()
            }
        case .hang(let lines):
            return AsyncThrowingStream { continuation in
                continuation.onTermination = onTermination
                for line in lines { continuation.yield(line) }
            }
        }
    }

    /// Formats event payloads as an Anthropic-style SSE body.
    static func sse(_ events: [String]) -> [String] {
        events.flatMap { event -> [String] in
            let type = (try? JSONValue.parse(event))?["type"]?.stringValue ?? "message"
            return ["event: \(type)", "data: \(event)", ""]
        }
    }

    /// The event payloads (for `sse`) of one streamed `tool_use` block: its start with an empty
    /// input, `json` split into two `input_json_delta`s, and its stop.
    static func toolUse(index: Int, id: String, name: String, json: String) -> [String] {
        let split = json.index(json.startIndex, offsetBy: json.count / 2)
        let parts = [String(json[..<split]), String(json[split...])]
        let block: JSONValue = ["type": "tool_use", "id": .string(id), "name": .string(name), "input": .object([:])]
        var events = [payload(["type": "content_block_start", "index": .int(index), "content_block": block])]
        for part in parts {
            let delta: JSONValue = ["type": "input_json_delta", "partial_json": .string(part)]
            events.append(payload(["type": "content_block_delta", "index": .int(index), "delta": delta]))
        }
        events.append(payload(["type": "content_block_stop", "index": .int(index)]))
        return events
    }

    private static func payload(_ value: JSONValue) -> String {
        String(decoding: (try? value.serialized()) ?? Data(), as: UTF8.self)
    }
}

/// A transport whose responses never arrive. Each body stream yields `prefix`, then waits until
/// its reader cancels it.
final class HangingTransport: HTTPStreamingTransport, @unchecked Sendable {
    private let prefix: [String]
    private let lock = NSLock()
    private var recorded: [URLRequest] = []
    private var terminations = 0

    init(prefix: [String] = []) {
        self.prefix = prefix
    }

    var requests: [URLRequest] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    /// How many body streams have ended (each one only by being cancelled).
    var terminationCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return terminations
    }

    func lines(for request: URLRequest) async throws -> AsyncThrowingStream<String, Error> {
        lock.withTestLock { recorded.append(request) }
        return try MockTransport.stream(.hang(prefix)) { [self] _ in
            lock.withTestLock { terminations += 1 }
        }
    }
}

/// A transport that answers each request with whatever `responder` returns for it and its
/// parsed JSON body (`.null` when there is none).
final class ScriptedTransport: HTTPStreamingTransport, @unchecked Sendable {
    private let responder: @Sendable (URLRequest, JSONValue) -> MockTransport.Response
    private let lock = NSLock()
    private var recorded: [URLRequest] = []
    private var cancellations = 0

    init(responder: @escaping @Sendable (URLRequest, JSONValue) -> MockTransport.Response) {
        self.responder = responder
    }

    var requests: [URLRequest] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    /// The parsed bodies of `requests`, in order.
    var bodies: [JSONValue] {
        requests.map(Self.body)
    }

    /// How many body streams were cancelled by their reader before they ended.
    var cancellationCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return cancellations
    }

    func lines(for request: URLRequest) async throws -> AsyncThrowingStream<String, Error> {
        lock.withTestLock { recorded.append(request) }
        return try MockTransport.stream(responder(request, Self.body(of: request))) { [self] termination in
            guard case .cancelled = termination else { return }
            lock.withTestLock { cancellations += 1 }
        }
    }

    static func body(of request: URLRequest) -> JSONValue {
        request.httpBody.flatMap { try? JSONValue.parse($0) } ?? .null
    }
}

/// An `AssistantProvider` that streams scripted events, optionally slowly, optionally failing
/// after some of them, and optionally never finishing.
final class ScriptedProvider: AssistantProvider, @unchecked Sendable {
    private let events: [AssistantEvent]
    private let failure: (after: Int, error: Error)?
    private let delay: TimeInterval
    private let hangs: Bool
    private let lock = NSLock()
    private var calls: [(system: String, turns: [ChatTurn])] = []
    private var terminations = 0

    /// - Parameters:
    ///   - failAfter: throw `error` after this many events (0 throws before the first).
    ///   - delay: seconds to wait before each event.
    ///   - hangs: after the events, wait until cancelled instead of finishing.
    init(
        _ events: [AssistantEvent],
        failAfter: Int? = nil,
        error: Error = URLError(.notConnectedToInternet),
        delay: TimeInterval = 0,
        hangs: Bool = false
    ) {
        self.events = events
        self.failure = failAfter.map { (after: $0, error: error) }
        self.delay = delay
        self.hangs = hangs
    }

    var callCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return calls.count
    }

    /// The `turns` of each call, in order.
    var receivedTurns: [[ChatTurn]] {
        lock.lock()
        defer { lock.unlock() }
        return calls.map(\.turns)
    }

    /// The `system` of each call, in order.
    var receivedSystems: [String] {
        lock.lock()
        defer { lock.unlock() }
        return calls.map(\.system)
    }

    /// How many returned streams have ended, for any reason.
    var terminationCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return terminations
    }

    func streamEvents(system: String, turns: [ChatTurn]) -> AsyncThrowingStream<AssistantEvent, Error> {
        lock.lock()
        calls.append((system: system, turns: turns))
        lock.unlock()
        let events = events, failure = failure, delay = delay, hangs = hangs
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for (index, event) in events.enumerated() {
                        if let failure, index == failure.after { throw failure.error }
                        if delay > 0 { try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000)) }
                        continuation.yield(event)
                    }
                    if let failure, failure.after >= events.count { throw failure.error }
                    while hangs {
                        try await Task.sleep(nanoseconds: 1_000_000_000)
                    }
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

/// An ordered record of tool runs starting and ending, shared by fake tools and executors.
final class ToolActivityLog: @unchecked Sendable {
    enum Entry: Equatable {
        case start(String)
        case end(String)
    }

    private let lock = NSLock()
    private var recorded: [(entry: Entry, time: TimeInterval)] = []

    var entries: [Entry] {
        lock.lock()
        defer { lock.unlock() }
        return recorded.map(\.entry)
    }

    /// Monotonic times (seconds) of `entries`.
    var times: [TimeInterval] {
        lock.lock()
        defer { lock.unlock() }
        return recorded.map(\.time)
    }

    func record(_ entry: Entry) {
        lock.lock()
        recorded.append((entry: entry, time: ProcessInfo.processInfo.systemUptime))
        lock.unlock()
    }

    /// The most runs that were in progress at once.
    var maxConcurrency: Int {
        var running = 0
        var peak = 0
        for entry in entries {
            switch entry {
            case .start: running += 1
            case .end: running -= 1
            }
            peak = max(peak, running)
        }
        return peak
    }
}

/// A tool with a fixed output (or error), an optional delay, and a record of its runs.
final class FakeTool: AssistantTool, @unchecked Sendable {
    static let emptySchema: JSONValue = [
        "type": "object", "properties": .object([:]), "required": .array([]), "additionalProperties": false,
    ]

    let definition: ToolDefinition
    let effect: ToolEffect
    let presentation: ToolPresentation
    /// Records `start(name)` and `end(name)` around each run.
    let log: ToolActivityLog
    private let output: ToolOutput
    private let error: Error?
    private let delay: TimeInterval
    private let lock = NSLock()
    private var receivedInputs: [[String: JSONValue]] = []

    init(
        name: String,
        effect: ToolEffect = .readOnly,
        output: ToolOutput = .ok(["ok": true]),
        error: Error? = nil,
        delay: TimeInterval = 0,
        description: String? = nil,
        schema: JSONValue = FakeTool.emptySchema,
        presentation: ToolPresentation = .generic,
        log: ToolActivityLog = ToolActivityLog()
    ) {
        definition = ToolDefinition(name: name, description: description ?? "Fake tool \(name).", inputSchema: schema)
        self.effect = effect
        self.output = output
        self.error = error
        self.delay = delay
        self.presentation = presentation
        self.log = log
    }

    var name: String { definition.name }

    /// The inputs of every run so far, in the order the runs started.
    var inputs: [[String: JSONValue]] {
        lock.lock()
        defer { lock.unlock() }
        return receivedInputs
    }

    var runCount: Int { inputs.count }

    func run(_ input: [String: JSONValue], context: ToolContext) async throws -> ToolOutput {
        lock.withTestLock { receivedInputs.append(input) }
        log.record(.start(name))
        defer { log.record(.end(name)) }
        if delay > 0 { try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000)) }
        if let error { throw error }
        return output
    }
}

/// Runs `FakeTool`s by name, serially or all at once, and records each round it is asked to run.
/// Unknown names give an "Unknown tool" error record; unparsed input gives `{"INVALID_JSON": raw}`;
/// a thrown error gives `{"error": description}`.
final class FakeToolExecutor: ToolExecutor, @unchecked Sendable {
    let tools: [FakeTool]
    let concurrent: Bool
    /// Records `start(call id)` and `end(call id)` around each call.
    let log = ToolActivityLog()
    private let lock = NSLock()
    private var received: [(calls: [PendingToolCall], context: ToolContext)] = []

    init(_ tools: [FakeTool], concurrent: Bool = false) {
        self.tools = tools
        self.concurrent = concurrent
    }

    var definitions: [ToolDefinition] {
        tools.map(\.definition).sorted { $0.name < $1.name }
    }

    func presentation(for name: String) -> ToolPresentation {
        tools.first { $0.name == name }?.presentation ?? .generic
    }

    /// The calls of each round, in order.
    var rounds: [[PendingToolCall]] {
        lock.lock()
        defer { lock.unlock() }
        return received.map(\.calls)
    }

    /// The context of each round, in order.
    var contexts: [ToolContext] {
        lock.lock()
        defer { lock.unlock() }
        return received.map(\.context)
    }

    func run(_ calls: [PendingToolCall], context: ToolContext) async -> ToolRound {
        lock.withTestLock { received.append((calls: calls, context: context)) }
        guard concurrent else {
            var records: [ToolCallRecord] = []
            for call in calls {
                records.append(await runOne(call, context: context))
            }
            return ToolRound(calls: records)
        }
        let records = await withTaskGroup(of: (Int, ToolCallRecord).self) { group -> [ToolCallRecord] in
            for (index, call) in calls.enumerated() {
                group.addTask { (index, await self.runOne(call, context: context)) }
            }
            var byIndex: [Int: ToolCallRecord] = [:]
            for await (index, record) in group { byIndex[index] = record }
            return calls.indices.compactMap { byIndex[$0] }
        }
        return ToolRound(calls: records)
    }

    private func runOne(_ call: PendingToolCall, context: ToolContext) async -> ToolCallRecord {
        guard let tool = tools.first(where: { $0.name == call.name }) else {
            return ToolCallRecord(call: call, output: .error("Unknown tool"))
        }
        guard case .object(let input)? = call.input else {
            return ToolCallRecord(call: call, output: ToolOutput(content: ["INVALID_JSON": .string(call.rawInput ?? "")], isError: true))
        }
        log.record(.start(call.id))
        defer { log.record(.end(call.id)) }
        do {
            return ToolCallRecord(call: call, output: try await tool.run(input, context: context))
        } catch {
            return ToolCallRecord(call: call, output: .error(String(describing: error)))
        }
    }
}

/// A clock that moves only when a test advances it. Times are seconds from an arbitrary origin,
/// matching the `at time: TimeInterval` parameters of the speech and latency types.
final class ManualClock: @unchecked Sendable {
    private struct Sleeper {
        var deadline: TimeInterval
        var continuation: CheckedContinuation<Void, Error>
    }

    private let lock = NSLock()
    private var current: TimeInterval
    private var sleepers: [UInt64: Sleeper] = [:]
    private var nextSleeper: UInt64 = 0

    init(now: TimeInterval = 0) {
        current = now
    }

    var now: TimeInterval {
        lock.lock()
        defer { lock.unlock() }
        return current
    }

    /// Tasks currently suspended in `sleep`.
    var sleeperCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return sleepers.count
    }

    /// Moves the clock forward and wakes every sleeper whose deadline has passed.
    func advance(by seconds: TimeInterval) {
        lock.lock()
        let target = current + seconds
        lock.unlock()
        advance(to: target)
    }

    /// Moves the clock to `time` (never backwards) and wakes every sleeper whose deadline has passed.
    func advance(to time: TimeInterval) {
        lock.lock()
        current = max(current, time)
        let due = sleepers.filter { $0.value.deadline <= current }
        for id in due.keys { sleepers.removeValue(forKey: id) }
        lock.unlock()
        for sleeper in due.values.sorted(by: { $0.deadline < $1.deadline }) {
            sleeper.continuation.resume()
        }
    }

    /// Suspends until the clock reaches `deadline`. Throws `CancellationError` if the task is cancelled.
    func sleep(until deadline: TimeInterval) async throws {
        let id = lock.withTestLock { () -> UInt64 in
            nextSleeper += 1
            return nextSleeper
        }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                lock.lock()
                if current >= deadline {
                    lock.unlock()
                    continuation.resume()
                } else if Task.isCancelled {
                    lock.unlock()
                    continuation.resume(throwing: CancellationError())
                } else {
                    sleepers[id] = Sleeper(deadline: deadline, continuation: continuation)
                    lock.unlock()
                }
            }
        } onCancel: {
            lock.lock()
            let sleeper = sleepers.removeValue(forKey: id)
            lock.unlock()
            sleeper?.continuation.resume(throwing: CancellationError())
        }
    }

    func sleep(for seconds: TimeInterval) async throws {
        try await sleep(until: now + seconds)
    }
}

func collect(_ stream: AsyncThrowingStream<ReplyEvent, Error>) async throws -> [ReplyEvent] {
    var events: [ReplyEvent] = []
    for try await event in stream {
        events.append(event)
    }
    return events
}

func collectEvents(_ stream: AsyncThrowingStream<AssistantEvent, Error>) async throws -> [AssistantEvent] {
    var events: [AssistantEvent] = []
    for try await event in stream {
        events.append(event)
    }
    return events
}

/// Polls `condition` every 5 ms until it holds or `timeout` seconds pass; returns whether it held.
@discardableResult
func waitUntil(timeout: TimeInterval = 2, _ condition: () -> Bool) async -> Bool {
    let deadline = Date().addingTimeInterval(timeout)
    while !condition() {
        guard Date() < deadline else { return false }
        try? await Task.sleep(nanoseconds: 5_000_000)
    }
    return true
}

extension NSLock {
    /// Runs `body` holding the lock. Synchronous, so async test code can use it.
    func withTestLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}
