import Foundation

/// A one-shot gate that holds side effects back until the user's turn is final.
///
/// A reply may start before the user has finished speaking. Its read-only work runs at once, but
/// anything that changes the world (adding a reminder, setting a timer) first awaits `wait()`.
/// The voice session calls `open()` when it adopts the early reply, or `cancel()` when it
/// discards it. The first of the two wins; later calls do nothing.
public final class CommitGate: @unchecked Sendable {
    private enum State {
        case pending
        case open
        case cancelled
    }

    private let lock = NSLock()
    private var state = State.pending
    private var waiters: [UInt64: CheckedContinuation<Void, Error>] = [:]
    private var nextWaiter: UInt64 = 0

    public init() {}

    public var isOpen: Bool {
        locked { state == .open }
    }

    public var isCancelled: Bool {
        locked { state == .cancelled }
    }

    /// How many tasks are suspended in `wait()`. Lets tests wait for a waiter to register.
    var waiterCount: Int {
        locked { waiters.count }
    }

    /// Lets every current and future waiter through. Does nothing once the gate is open or cancelled.
    public func open() {
        settle(.open)
    }

    /// Makes every current and future waiter throw `CancellationError`. Does nothing once the gate
    /// is open or cancelled.
    public func cancel() {
        settle(.cancelled)
    }

    /// Returns once the gate is open; at once if it already is.
    /// Throws `CancellationError` if the gate is cancelled or the waiting task is cancelled.
    public func wait() async throws {
        let id = locked { () -> UInt64 in
            nextWaiter += 1
            return nextWaiter
        }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                lock.lock()
                switch state {
                case .open:
                    lock.unlock()
                    continuation.resume()
                case .cancelled:
                    lock.unlock()
                    continuation.resume(throwing: CancellationError())
                case .pending:
                    // Checked under the lock: a cancellation that lands after this check finds
                    // the waiter registered and resumes it from the handler.
                    if Task.isCancelled {
                        lock.unlock()
                        continuation.resume(throwing: CancellationError())
                    } else {
                        waiters[id] = continuation
                        lock.unlock()
                    }
                }
            }
        } onCancel: {
            let continuation = locked { waiters.removeValue(forKey: id) }
            continuation?.resume(throwing: CancellationError())
        }
    }

    private func settle(_ final: State) {
        let resumed: [CheckedContinuation<Void, Error>] = locked {
            guard state == .pending else { return [] }
            state = final
            let pending = Array(waiters.values)
            waiters.removeAll()
            return pending
        }
        for continuation in resumed {
            if final == .open {
                continuation.resume()
            } else {
                continuation.resume(throwing: CancellationError())
            }
        }
    }

    private func locked<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }
}

/// `base`, once `gate` is open: the stream waits for that, and fails with `CancellationError`
/// when the gate is cancelled instead, without ever calling `base`.
///
/// A reply started early wraps every provider whose request leaves a durable record (a cloud
/// request, a backend that keeps its own history) in one, so nothing is sent for a turn the user
/// didn't finish. The wait counts against the orchestrator's first-event watchdog.
public struct CommitGatedProvider: AssistantProvider {
    public let base: any AssistantProvider
    public let gate: CommitGate

    public init(base: any AssistantProvider, gate: CommitGate) {
        self.base = base
        self.gate = gate
    }

    public func streamEvents(system: String, turns: [ChatTurn]) -> AsyncThrowingStream<AssistantEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await gate.wait()
                    // The gate may open just as the stream is dropped: send nothing then.
                    // ponytail: narrows that race to the microseconds before this check (measured: a base
                    // call in 100 of 100 open-then-drop runs without it, 3–7 with it). Closing it fully
                    // would need this task on the actor that opens and drops the gate.
                    try Task.checkCancellation()
                    for try await event in base.streamEvents(system: system, turns: turns) {
                        continuation.yield(event)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
