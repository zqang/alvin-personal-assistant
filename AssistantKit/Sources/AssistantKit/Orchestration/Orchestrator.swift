import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Runs one routed reply and falls back to the other engine when the first fails early.
///
/// - It yields `.routed(decision)` first.
/// - It passes the primary engine's events through. The reply counts as committed once a non-empty
///   `.reply(.text)`, a `.toolRound` or a `.reply(.finished)` has been passed on.
/// - Before commit, and only then, it falls back once:
///   - on a connectivity failure, when the fallback is local: `.reply(.activity(nil))`, then
///     `.routed(local, standard, networkFallback)`, then the on-device reply;
///   - on `ReplyHandoff`, when the fallback is the cloud: `.reply(.activity(nil))`, then
///     `.routed(cloud, standard, escalated)`, then `.cue(.handingOff)`, then the cloud reply;
///   - when the first-event watchdog fires (fallback local only): as for a connectivity failure.
///     The watchdog cancels a primary that has produced no event of any kind within
///     `firstEventTimeout`, which catches captive portals and dead Wi-Fi that still look online.
/// - Every other error, and every error after commit, is rethrown unchanged; a 401 never falls
///   back. Falling back after a tool round would run its tools twice.
/// - Ending or cancelling the returned stream cancels the inner providers' streams.
public struct Orchestrator: AssistantProvider {
    public struct Engines: Sendable {
        /// The standard Claude reply.
        public var cloud: (any AssistantProvider)?
        /// The deep-mode Claude reply.
        public var deep: (any AssistantProvider)?
        /// The on-device reply.
        public var local: (any AssistantProvider)?

        public init(
            cloud: (any AssistantProvider)? = nil,
            deep: (any AssistantProvider)? = nil,
            local: (any AssistantProvider)? = nil
        ) {
            self.cloud = cloud
            self.deep = deep
            self.local = local
        }

        /// The provider for `engine` in `mode`. Deep mode exists only in the cloud.
        public func provider(for engine: ReplyEngine, mode: ReplyMode) -> (any AssistantProvider)? {
            switch engine {
            case .cloud: return mode == .deep ? deep : cloud
            case .local: return local
            }
        }
    }

    public let decision: RouteDecision
    public let engines: Engines
    /// How long the primary may stay silent before the watchdog switches to the on-device model.
    /// Used only when the fallback is local and an on-device engine is present; nil (or a value of
    /// zero or less) disables it.
    public let firstEventTimeout: Duration?
    private let sleep: @Sendable (Duration) async throws -> Void

    public init(decision: RouteDecision, engines: Engines, firstEventTimeout: Duration? = .seconds(8)) {
        self.init(decision: decision, engines: engines, firstEventTimeout: firstEventTimeout) { duration in
            try await Task.sleep(for: duration)
        }
    }

    /// `sleep` lets tests drive the watchdog with a manual clock.
    init(
        decision: RouteDecision,
        engines: Engines,
        firstEventTimeout: Duration?,
        sleep: @escaping @Sendable (Duration) async throws -> Void
    ) {
        self.decision = decision
        self.engines = engines
        self.firstEventTimeout = firstEventTimeout
        self.sleep = sleep
    }

    public func streamEvents(system: String, turns: [ChatTurn]) -> AsyncThrowingStream<AssistantEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await run(system: system, turns: turns, continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: Running

    typealias Continuation = AsyncThrowingStream<AssistantEvent, Error>.Continuation

    private func run(system: String, turns: [ChatTurn], continuation: Continuation) async throws {
        continuation.yield(.routed(decision))
        guard let primary = engines.provider(for: decision.engine, mode: decision.mode) else {
            throw AssistantError.missingConfiguration(Self.missingEngineMessage(decision.engine, mode: decision.mode))
        }

        let attempt = Attempt()
        let reason: RouteReason
        switch await relay(primary, system: system, turns: turns, attempt: attempt, continuation: continuation) {
        case .finished:
            return
        case .timedOut:
            reason = .networkFallback
        case .failed(let error):
            guard !attempt.isCommitted, !Task.isCancelled, let allowed = fallbackReason(for: error) else { throw error }
            reason = allowed
        }
        try Task.checkCancellation()

        let engine: ReplyEngine = reason == .escalated ? .cloud : .local
        guard let fallback = engines.provider(for: engine, mode: .standard) else {
            throw AssistantError.missingConfiguration(Self.missingEngineMessage(engine))
        }
        continuation.yield(.reply(.activity(nil)))
        continuation.yield(.routed(RouteDecision(engine: engine, mode: .standard, reason: reason)))
        if reason == .escalated {
            continuation.yield(.cue(.handingOff))
        }
        for try await event in fallback.streamEvents(system: system, turns: turns) {
            continuation.yield(event)
        }
    }

    /// The fallback a pre-commit `error` allows, as the reason of the new route.
    func fallbackReason(for error: Error) -> RouteReason? {
        if error is ReplyHandoff {
            return canFallBack(to: .cloud) ? .escalated : nil
        }
        if error.isConnectivityFailure {
            return canFallBack(to: .local) ? .networkFallback : nil
        }
        return nil
    }

    private func canFallBack(to engine: ReplyEngine) -> Bool {
        decision.fallback == engine && decision.engine != engine && engines.provider(for: engine, mode: .standard) != nil
    }

    /// The watchdog's timeout, when it applies to this decision. A timeout of zero or less disables it.
    var watchdogTimeout: Duration? {
        guard let timeout = firstEventTimeout, timeout > .zero, canFallBack(to: .local) else { return nil }
        return timeout
    }

    private enum Signal: Sendable {
        case finished
        case watchdog
    }

    private enum Outcome {
        case finished
        case timedOut
        case failed(Error)
    }

    /// Streams `primary` into `continuation`, racing the first-event watchdog when it applies.
    private func relay(
        _ primary: any AssistantProvider,
        system: String,
        turns: [ChatTurn],
        attempt: Attempt,
        continuation: Continuation
    ) async -> Outcome {
        let timeout = watchdogTimeout
        let sleep = sleep
        do {
            return try await withThrowingTaskGroup(of: Signal.self) { group -> Outcome in
                group.addTask {
                    for try await event in primary.streamEvents(system: system, turns: turns) {
                        guard attempt.forward(event, to: continuation) else { break }
                    }
                    return .finished
                }
                if let timeout {
                    group.addTask {
                        try await sleep(timeout)
                        return .watchdog
                    }
                }
                while let signal = try await group.next() {
                    switch signal {
                    case .finished:
                        group.cancelAll()
                        return .finished
                    case .watchdog:
                        // The primary spoke in time; let it run.
                        guard attempt.expireIfSilent() else { continue }
                        group.cancelAll()
                        // Let the cancelled primary wind down before the fallback starts.
                        while true {
                            do {
                                guard try await group.next() != nil else { break }
                            } catch {
                                continue
                            }
                        }
                        return .timedOut
                    }
                }
                return .finished
            }
        } catch {
            return .failed(error)
        }
    }

    static func missingEngineMessage(_ engine: ReplyEngine, mode: ReplyMode = .standard) -> String {
        switch (engine, mode) {
        case (.cloud, .deep): return "Deep thinking isn't available right now."
        case (.cloud, .standard): return "Add your Anthropic API key in Settings to use Claude."
        case (.local, _): return "Download an on-device model in Settings to answer on this iPhone."
        }
    }

    /// Whether the primary has produced anything, and whether that commits the reply. The watchdog
    /// and the relay settle under one lock, so an event is either passed on or dropped for good.
    private final class Attempt: @unchecked Sendable {
        private let lock = NSLock()
        private var started = false
        private var committed = false
        private var expired = false

        var isCommitted: Bool {
            lock.lock()
            defer { lock.unlock() }
            return committed
        }

        /// Yields `event` unless the watchdog has already given up on this attempt.
        func forward(_ event: AssistantEvent, to continuation: Continuation) -> Bool {
            lock.lock()
            defer { lock.unlock() }
            guard !expired else { return false }
            started = true
            if Attempt.commits(event) { committed = true }
            continuation.yield(event)
            return true
        }

        /// Gives up on the attempt if it hasn't produced an event yet. Returns whether it gave up.
        func expireIfSilent() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            guard !started else { return false }
            expired = true
            return true
        }

        static func commits(_ event: AssistantEvent) -> Bool {
            switch event {
            case .reply(.text(let text)): return !text.isEmpty
            case .reply(.finished), .toolRound: return true
            case .reply(.activity), .cue, .routed, .progress: return false
            }
        }
    }
}

/// The URL error codes that mean the network, not the server, failed.
private let connectivityFailureCodes: [URLError.Code] = [
    .notConnectedToInternet, .networkConnectionLost, .timedOut, .cannotFindHost, .cannotConnectToHost,
    .dnsLookupFailed, .internationalRoamingOff, .dataNotAllowed, .secureConnectionFailed,
]

extension Error {
    /// Whether this error means the network couldn't be used (no route, lost connection, timeout,
    /// DNS, roaming or cellular data off, a TLS handshake broken by a captive portal). HTTP status
    /// errors, such as a 401, are never connectivity failures.
    public var isConnectivityFailure: Bool {
        if let urlError = self as? URLError {
            return connectivityFailureCodes.contains(urlError.code)
        }
        let error = self as NSError
        guard error.domain == NSURLErrorDomain else { return false }
        // `URLError.Code(rawValue:)` is failable on Linux only, so compare raw values.
        return connectivityFailureCodes.contains { $0.rawValue == error.code }
    }
}
