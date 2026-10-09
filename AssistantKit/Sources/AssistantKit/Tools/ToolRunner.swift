import Foundation

/// Runs a model's tool calls against a `ToolRegistry`.
///
/// For each round:
/// - every call is checked first: an unknown, turned-off or blocked tool, or input that fails
///   `ToolInputValidator`, gets an error record without running anything;
/// - read-only tools run concurrently;
/// - side-effect tools run one at a time in the model's order, alongside the read-only ones. Each
///   first waits on `context.commitGate`; if the gate is cancelled (or the round is), the call is
///   not run and gets `cancelledMessage`;
/// - before running, a tool's `authorize` gets what it needs from the user, such as a first-time
///   permission alert. It has no time limit, so a slow answer doesn't cancel the action the user
///   just allowed; cancelling the round stops the wait at once (`cancelledMessage`), and an error
///   it throws is the call's result;
/// - each run is limited to `timeout`. A tool that doesn't finish in time gets an error record; its
///   task is cancelled, but the runner doesn't wait for it to stop;
/// - a side-effect tool that timed out may still be acting, so no later side-effect tool starts
///   until it has ended: not later in the round, nor in a later round run by this runner or a copy
///   of it. That wait is also limited to `timeout`; past it, the later call isn't run and gets
///   `earlierActionRunningMessage`;
/// - a result is the output's `content` as JSON with sorted keys, cut to `maxResultCharacters`
///   characters plus `"…(truncated)"` when longer.
///
/// Records come back in the calls' order, and `run` never throws.
public struct ToolRunner: ToolExecutor {
    public static let unknownToolMessage = "Unknown tool"
    public static let cancelledMessage = "The request was cancelled before the action ran."
    public static let truncationMarker = "…(truncated)"
    static let timeoutMessage = "The tool didn't finish in time."
    static let sideEffectTimeoutMessage = "The tool didn't finish in time; the action may or may not have happened. Don't repeat it; tell the user it may not have gone through."
    static let earlierActionRunningMessage = "An earlier action is still in progress, so this one wasn't started."

    public let registry: ToolRegistry
    public let timeout: Duration
    public let maxResultCharacters: Int
    public let lenientInput: Bool
    /// Side-effect runs that timed out but may still be acting. Copies of the runner share it.
    private let abandoned = AbandonedSideEffects()

    /// - Parameters:
    ///   - timeout: the longest one tool may run; zero or less means no limit.
    ///   - lenientInput: coerce slightly mistyped input (see `ToolInputValidator`), for on-device models.
    public init(registry: ToolRegistry, timeout: Duration = .seconds(10), maxResultCharacters: Int = 2_000, lenientInput: Bool = false) {
        self.registry = registry
        self.timeout = timeout
        self.maxResultCharacters = max(maxResultCharacters, 0)
        self.lenientInput = lenientInput
    }

    public var definitions: [ToolDefinition] {
        registry.definitions
    }

    public func presentation(for name: String) -> ToolPresentation {
        registry.presentation(for: name)
    }

    public func run(_ calls: [PendingToolCall], context: ToolContext) async -> ToolRound {
        var outputs = [ToolOutput?](repeating: nil, count: calls.count)
        var reads: [Job] = []
        var writes: [Job] = []
        for (index, call) in calls.enumerated() {
            switch prepare(call, index: index) {
            case .finished(let output):
                outputs[index] = output
            case .run(let job):
                if job.tool.effect == .sideEffect {
                    writes.append(job)
                } else {
                    reads.append(job)
                }
            }
        }

        if !reads.isEmpty || !writes.isEmpty {
            await withTaskGroup(of: [(Int, ToolOutput)].self) { group in
                for job in reads {
                    group.addTask { [(job.index, await self.runReadOnly(job, context: context))] }
                }
                if !writes.isEmpty {
                    let serial = writes
                    group.addTask {
                        var finished: [(Int, ToolOutput)] = []
                        for job in serial {
                            finished.append((job.index, await self.runSideEffect(job, context: context)))
                        }
                        return finished
                    }
                }
                for await finished in group {
                    for (index, output) in finished {
                        outputs[index] = output
                    }
                }
            }
        }

        let records = calls.enumerated().map { index, call in
            record(call, output: outputs[index] ?? .error(Self.cancelledMessage))
        }
        return ToolRound(calls: records)
    }

    // MARK: - Steps

    private struct Job: Sendable {
        var index: Int
        var tool: any AssistantTool
        var input: [String: JSONValue]
    }

    private enum Preparation {
        case finished(ToolOutput)
        case run(Job)
    }

    /// Resolves the tool and validates the input; anything that stops the call here is final.
    private func prepare(_ call: PendingToolCall, index: Int) -> Preparation {
        switch registry.resolve(call.name) {
        case .unknown:
            return .finished(.error(Self.unknownToolMessage))
        case .unavailable(let message):
            return .finished(.error(message))
        case .runnable(let tool):
            switch ToolInputValidator.validate(call, schema: tool.definition.inputSchema, lenient: lenientInput) {
            case .success(let input):
                return .run(Job(index: index, tool: tool, input: input))
            case .failure(let error):
                return .finished(ToolOutput(content: ToolInputValidator.errorContent(error), isError: true))
            }
        }
    }

    private func runSideEffect(_ job: Job, context: ToolContext) async -> ToolOutput {
        guard !Task.isCancelled else { return .error(Self.cancelledMessage) }
        if let gate = context.commitGate {
            do {
                try await gate.wait()
            } catch {
                return .error(Self.cancelledMessage)
            }
        }
        // An open gate lets a cancelled task through; the round was abandoned, so don't act.
        guard !Task.isCancelled else { return .error(Self.cancelledMessage) }
        // Only a committed turn asks the user, and the answer doesn't count against the timeout.
        if let refused = await authorize(job, context: context) { return refused }
        guard !Task.isCancelled else { return .error(Self.cancelledMessage) }
        // Never act while an earlier action that timed out may still be acting.
        let clear = await abandoned.waitUntilEnded(limit: timeout)
        guard !Task.isCancelled else { return .error(Self.cancelledMessage) }
        guard clear else { return .error(Self.earlierActionRunningMessage) }
        return await execute(job, context: context)
    }

    private func runReadOnly(_ job: Job, context: ToolContext) async -> ToolOutput {
        if let refused = await authorize(job, context: context) { return refused }
        return await execute(job, context: context)
    }

    /// Runs the tool's `authorize` in its own task, with no time limit. Returns nil when the tool
    /// may run, or else the call's output: the error `authorize` threw, or `cancelledMessage` as
    /// soon as the caller is cancelled. A cancelled `authorize` isn't waited for, since it may be
    /// waiting on the user (a permission alert stays up until answered).
    private func authorize(_ job: Job, context: ToolContext) async -> ToolOutput? {
        guard !Task.isCancelled else { return .error(Self.cancelledMessage) }
        let outcome = FirstValue<Authorization>()
        let tool = job.tool
        let input = job.input
        let work = Task {
            do {
                try await tool.authorize(input, context: context)
                outcome.offer(.granted)
            } catch {
                outcome.offer(.refused(Self.failure(error)))
            }
        }
        let result = await withTaskCancellationHandler {
            await outcome.value()
        } onCancel: {
            work.cancel()
            outcome.offer(.refused(.error(Self.cancelledMessage)))
        }
        switch result {
        case .granted:
            return nil
        case .refused(let output):
            return output
        }
    }

    private enum Authorization: Sendable {
        case granted
        case refused(ToolOutput)
    }

    /// Runs the tool in its own task and returns its output, or a timeout error when `timeout`
    /// passes first. Cancelling the caller cancels the tool's task; its output (often an error)
    /// is still awaited, within the timeout, so the record says what actually happened. A
    /// side-effect run that times out is kept in `abandoned` until it ends.
    private func execute(_ job: Job, context: ToolContext) async -> ToolOutput {
        let outcome = FirstValue<Outcome>()
        let tool = job.tool
        let input = job.input
        let work = Task {
            let output: ToolOutput
            do {
                output = try await tool.run(input, context: context)
            } catch {
                output = Self.failure(error)
            }
            outcome.offer(.finished(output))
        }
        var timer: Task<Void, Never>?
        if timeout > .zero {
            let limit = timeout
            timer = Task {
                do {
                    try await Task.sleep(for: limit)
                } catch {
                    return
                }
                outcome.offer(.timedOut)
            }
        }
        let result = await withTaskCancellationHandler {
            await outcome.value()
        } onCancel: {
            work.cancel()
        }
        work.cancel()
        timer?.cancel()
        switch result {
        case .finished(let output):
            return output
        case .timedOut:
            guard tool.effect == .sideEffect else { return .error(Self.timeoutMessage) }
            abandoned.add(work)
            return .error(Self.sideEffectTimeoutMessage)
        }
    }

    private enum Outcome: Sendable {
        case finished(ToolOutput)
        case timedOut
    }

    private static func failure(_ error: Error) -> ToolOutput {
        if error is CancellationError {
            return .error("The request was cancelled.")
        }
        if let described = (error as? LocalizedError)?.errorDescription, !described.isEmpty {
            return .error(described)
        }
        return .error(String(describing: error))
    }

    private func record(_ call: PendingToolCall, output: ToolOutput) -> ToolCallRecord {
        var record = ToolCallRecord(call: call, output: output)
        record.result = Self.truncate(record.result, to: maxResultCharacters)
        return record
    }

    /// The first `limit` characters of `text` plus the marker, or `text` itself when it fits.
    static func truncate(_ text: String, to limit: Int) -> String {
        guard text.count > limit else { return text }
        return String(text.prefix(limit)) + truncationMarker
    }
}

/// Side-effect runs the runner stopped waiting for after a timeout. A tool may ignore
/// cancellation (or wait on a permission alert), so such a run can still act; later side effects
/// wait for it to end, which keeps actions from ever running at the same time.
private final class AbandonedSideEffects: @unchecked Sendable {
    private let lock = NSLock()
    private var runs: [Task<Void, Never>] = []

    func add(_ run: Task<Void, Never>) {
        lock.lock()
        runs.append(run)
        lock.unlock()
    }

    /// Waits until every run added so far has ended, for at most `limit` (no limit when it is zero
    /// or less). Returns false when `limit` passes first or the waiting task is cancelled.
    func waitUntilEnded(limit: Duration) async -> Bool {
        let pending = current()
        guard !pending.isEmpty else { return true }

        let ended = FirstValue<Bool>()
        let watcher = Task {
            for run in pending {
                await run.value
            }
            ended.offer(true)
        }
        var timer: Task<Void, Never>?
        if limit > .zero {
            timer = Task {
                do {
                    try await Task.sleep(for: limit)
                } catch {
                    return
                }
                ended.offer(false)
            }
        }
        let result = await withTaskCancellationHandler {
            await ended.value()
        } onCancel: {
            ended.offer(false)
        }
        // A run that hasn't ended keeps `watcher` waiting; its late offer is ignored.
        watcher.cancel()
        timer?.cancel()
        if result {
            forget(pending)
        }
        return result
    }

    private func current() -> [Task<Void, Never>] {
        lock.lock()
        defer { lock.unlock() }
        return runs
    }

    private func forget(_ ended: [Task<Void, Never>]) {
        lock.lock()
        runs.removeAll { ended.contains($0) }
        lock.unlock()
    }
}

/// Holds the first value offered, and hands it to the one task waiting for it.
private final class FirstValue<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Value?
    private var waiter: CheckedContinuation<Value, Never>?

    /// Keeps `candidate` if no value arrived before it; later offers are ignored.
    func offer(_ candidate: Value) {
        lock.lock()
        guard stored == nil else {
            lock.unlock()
            return
        }
        stored = candidate
        let waiting = waiter
        waiter = nil
        lock.unlock()
        waiting?.resume(returning: candidate)
    }

    /// Waits for the first value. Call it once.
    func value() async -> Value {
        await withCheckedContinuation { (continuation: CheckedContinuation<Value, Never>) in
            lock.lock()
            if let stored {
                lock.unlock()
                continuation.resume(returning: stored)
            } else {
                waiter = continuation
                lock.unlock()
            }
        }
    }
}
