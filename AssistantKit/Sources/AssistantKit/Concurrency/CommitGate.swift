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
