import AssistantKit
import Foundation
import MLX
import MLXLMCommon

/// The on-device reply engine ("Alvin engine"): one loaded model, its live session, and a
/// serial queue that runs all of its MLX work (plan §4.1, §4.11).
///
/// - `reply` prepares the session for a request (reusing every cached token it can), then
///   decodes with the configured generator, streaming `EngineEvent`s.
/// - `continueReply(after:)` appends a tool round's results to the reply in progress and
///   decodes again.
/// - Every GPU job runs between `hooks.beginGPU()` and `hooks.endGPU()`; `hooks.isAllowed()` and
///   the stream's termination are checked before every decode step and prefill chunk, and stop
///   the reply with `.cancelled`, leaving the ledger exact.
///
/// Public async APIs hop onto the engine queue; the queue block keeps the engine alive until
/// it finishes.
public final class InferenceEngine: @unchecked Sendable {
    public let loaded: LoadedModel
    /// The live session (engine queue only).
    public let session: LiveSession
    let runtime: SessionRuntime
    let queue = DispatchQueue(label: "alvin.engine", qos: .userInitiated)
    /// The tool-call format replies are parsed with.
    public let toolCallFormat: ToolCallFormat

    /// Engine queue only.
    private(set) var configuration: EngineConfiguration
    private var preparedExtensions: [ObjectIdentifier: Bool] = [:]

    private let infoLock = NSLock()
    private var currentInfo: EngineInfo

    // MARK: Loading

    /// Loads the model in `directory` (already downloaded) with CPU-resident weights: the
    /// `HybridQwen35` fork for Qwen3.5 text models (`EngineModelRegistry`), the stock model
    /// code otherwise. Run it inside `Device.withDefaultDevice(.cpu)`; `warmUp()` then runs
    /// on the GPU.
    public static func load(directory: URL, modelID: String, configuration: EngineConfiguration) async throws -> InferenceEngine {
        let loaded = try await ModelLoader.load(directory: directory, id: modelID, typeRegistry: EngineModelRegistry.makeTypeRegistry())
        return InferenceEngine(loaded: loaded, configuration: configuration)
    }

    /// An engine around an already loaded model. `target` defaults to a `HybridTarget` for
    /// fork models and a `StockTarget` otherwise (tests pass their own).
    public init(loaded: LoadedModel, target: (any TargetModel)? = nil, configuration: EngineConfiguration = EngineConfiguration()) {
        let target = target
            ?? EngineModelRegistry.makeTarget(for: loaded)
            ?? StockTarget(model: loaded.model, directory: loaded.directory)
        let renderer = try? TurnRenderer(renderer: loaded.renderer, chatContext: configuration.chatContext)
        let session = LiveSession(target: target, noReuse: renderer == nil)
        self.loaded = loaded
        self.session = session
        self.configuration = configuration
        self.toolCallFormat = loaded.configuration.toolCallFormat ?? .json
        self.runtime = SessionRuntime(
            session: session, renderer: renderer, templates: loaded.renderer, modelID: loaded.id,
            revision: loaded.directory.lastPathComponent, configuration: configuration)
        self.currentInfo = EngineInfo(
            modelID: loaded.id, isHybrid: target.layout.isHybrid, forked: EngineModelRegistry.isFork(loaded.model),
            vocabularySize: target.vocabularySize, bytesPerCheckpoint: 0, stopTokenIDs: loaded.stopTokenIDs,
            toolCallFormat: (loaded.configuration.toolCallFormat ?? .json).rawValue)
    }

    public var info: EngineInfo {
        infoLock.lock()
        defer { infoLock.unlock() }
        return currentInfo
    }

    /// The model the engine decodes with.
    public var target: any TargetModel { session.target }

    // MARK: Lifecycle

    /// Compiles the kernels the engine uses (single-row and four-row forwards and the sampler,
    /// on scratch positions of the live cache) and prepares the extensions.
    public func warmUp() async throws {
        try await onQueue { [self] in
            try withGPU {
                try withError {
                    let probe = self.loaded.renderer.encodeRaw("Hi").first ?? 0
                    let sampler = FastSampler(configuration: self.configuration, greedy: false)
                    var bytes = 0
                    self.session.withScratch {
                        let position = self.session.ledger.count
                        if let logits = self.session.feed([probe], rows: .last).logits {
                            let (token, top1) = sampler.sample(logits, positions: position ..< (position + 1))
                            eval(token, top1)
                        }
                        if let logits = self.session.feed([probe, probe, probe, probe], rows: .all).logits {
                            let (tokens, top1) = sampler.sample(logits, positions: (position + 1) ..< (position + 5))
                            eval(tokens, top1)
                        }
                        bytes = self.session.bytesPerCheckpoint
                    }
                    self.updateInfo { $0.bytesPerCheckpoint = bytes }
                    self.prepareExtensionsIfNeeded()
                }
                Memory.clearCache()
            }
        }
    }

    /// Changes the configuration; takes effect for the next job.
    public func updateConfiguration(_ change: @escaping @Sendable (inout EngineConfiguration) -> Void) async {
        _ = try? await onQueue { [self] in
            var configuration = self.configuration
            change(&configuration)
            self.configuration = configuration
            self.runtime.apply(configuration)
            let current = Set(configuration.extensions.map { ObjectIdentifier($0) })
            self.preparedExtensions = self.preparedExtensions.filter { current.contains($0.key) }
        }
    }

    /// Makes the system prefix of `system` and `tools` resident: kept if live, else loaded from
    /// disk, else prefilled (and saved when the disk prefix is on).
    public func prewarm(system: String, tools: [ToolDefinition]) async throws {
        try await onQueue { [self] in
            try withGPU {
                let hooks = self.configuration.hooks
                do {
                    try withError {
                        try self.runtime.prewarm(system: system, tools: tools, isAllowed: hooks.isAllowed)
                    }
                } catch is PrefillInterrupted {
                    self.runtime.abandon()
                    throw EngineError.leftForeground
                } catch {
                    self.runtime.abandon()
                    throw error
                }
                self.updateInfo { $0.bytesPerCheckpoint = self.session.bytesPerCheckpoint }
                Memory.clearCache()
            }
        }
    }

    /// Answers `request`: see `EngineEvent` for the order of events.
    public func reply(_ request: EngineRequest) -> AsyncThrowingStream<EngineEvent, Error> {
        stream { [self] continuation, cancellation in
            self.generate(.reply(request), continuation: continuation, cancellation: cancellation)
        }
    }

    /// Continues the reply whose stream ended with `.toolCalls`, after `round` (the results of
    /// those calls). Valid only right after `.toolCalls`; otherwise the stream throws
    /// `EngineError.busy`.
    public func continueReply(after round: ToolRound) -> AsyncThrowingStream<EngineEvent, Error> {
        stream { [self] continuation, cancellation in
            self.generate(.continuation(round), continuation: continuation, cancellation: cancellation)
        }
    }

    /// Forgets the live conversation; the next reply starts from an empty cache (or the disk
    /// prefix).
    public func invalidateSession() async {
        _ = try? await onQueue { [self] in
            self.runtime.invalidate()
        }
    }

    /// Releases rewind checkpoints (memory pressure). Rewinds then re-feed from the deepest
    /// one left (or from the start), which stays exact.
    public func dropCheckpoints(keepSystem: Bool) async {
        _ = try? await onQueue { [self] in
            self.session.checkpoints.removeAll(keepSystem: keepSystem)
        }
    }

    /// Returns once every job queued before the call has finished.
    public func waitUntilIdle() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            queue.async {
                continuation.resume()
            }
        }
    }

    /// One line on the session: warm or not, ledger size, checkpoints, disk prefix.
    public func sessionSummary() async -> String {
        (try? await onQueue { [self] in self.summary() }) ?? "unavailable"
    }

    /// Runs `body` with the live session on the engine queue, between other jobs (tests,
    /// probes and self-checks).
    public func withSession<T>(_ body: @escaping (LiveSession) throws -> T) async throws -> T {
        try await onQueue { [self] in try body(self.session) }
    }

    // MARK: Generating

    private enum Job {
        case reply(EngineRequest)
        case continuation(ToolRound)
    }

    private func generate(
        _ job: Job, continuation: AsyncThrowingStream<EngineEvent, Error>.Continuation, cancellation: StreamCancellation
    ) {
        let hooks = configuration.hooks
        guard hooks.beginGPU() else {
            continuation.finish(throwing: EngineError.leftForeground)
            return
        }
        defer { hooks.endGPU() }
        let isAllowed: () -> Bool = { !cancellation.isCancelled && hooks.isAllowed() }
        let requestStart = Date()
        Memory.peakMemory = 0
        continuation.yield(.progress(.responseStarted))

        do {
            prepareExtensionsIfNeeded()
            let prepared: SessionRuntime.Prepared = try withError {
                switch job {
                case .reply(let request):
                    return try self.runtime.prepare(request, isAllowed: isAllowed)
                case .continuation(let round):
                    return try self.runtime.prepareContinuation(after: round, isAllowed: isAllowed)
                }
            }
            guard let request = runtime.reply?.request else {
                throw EngineError.busy
            }
            continuation.yield(.progress(.prefillDone(prefilledTokens: prepared.prefilledTokens, reusedTokens: prepared.reusedTokens)))

            let streamer = TextStreamer(
                tokenizer: loaded.tokenizer, renderer: loaded.renderer, format: toolCallFormat,
                tools: JSONBridge.templateToolsOrNil(request.tools))
            let drafters = activeExtensions().flatMap { $0.drafters(for: request, engine: self) }
            for drafter in drafters {
                drafter.reset(ledger: session.ledger, request: request)
            }
            let confidence = ConfidenceRecorder()
            let context = GeneratorContext(
                session: session, sampler: FastSampler(configuration: configuration, greedy: request.greedy),
                firstLogits: prepared.firstLogits, stopTokens: loaded.stopTokenIDs,
                maxTokens: request.maxTokens ?? configuration.maxTokens, request: request, drafters: drafters,
                insideToolCall: { streamer.insideToolCall }, isAllowed: isAllowed, renderer: loaded.renderer,
                toolCallFormat: toolCallFormat, confidence: confidence)
            let generator = (configuration.generatorFactory ?? PlainGeneratorFactory()).makeGenerator(context)

            let decodeStart = Date()
            var phases = prepared.phases
            var firstToken = false
            var firstText: Date?
            var generated = 0
            var reason: EngineFinish.Reason = .stop
            while true {
                let (emitted, finished) = try withError { try generator.step() }
                if !firstToken && (!emitted.isEmpty || finished == .stop || finished == .length) {
                    firstToken = true
                    phases.firstToken = Date().timeIntervalSince(decodeStart)
                    continuation.yield(.progress(.firstToken))
                }
                generated += emitted.count
                for output in streamer.append(emitted) {
                    switch output {
                    case .text(let text):
                        if firstText == nil { firstText = Date() }
                        continuation.yield(.text(text))
                    case .toolCallStarted(let name):
                        continuation.yield(.progress(.toolCallStarted(name: name)))
                    }
                }
                if let finished {
                    reason = finished
                    break
                }
            }
            let flushed = hooks.isAllowed()
            if flushed {
                try withError { try generator.flush() }
            }
            let generateTime = Date().timeIntervalSince(decodeStart)

            let (tail, calls) = streamer.finish()
            if !tail.isEmpty {
                if firstText == nil { firstText = Date() }
                continuation.yield(.text(tail))
            }
            runtime.finishGeneration(
                text: streamer.visibleText, calls: calls, startedCall: !streamer.startedCalls.isEmpty, reason: reason,
                incomplete: !flushed)
            if !calls.isEmpty && reason != .cancelled {
                reason = .toolCalls
            }
            updateInfo { $0.bytesPerCheckpoint = self.session.bytesPerCheckpoint }

            let speculation = generator.speculation
            let stats = LocalGenerationStats(
                timeToFirstText: firstText.map { $0.timeIntervalSince(requestStart) },
                promptTokens: prepared.prefilledTokens,
                promptTime: prepared.phases.prefill,
                generatedTokens: generated,
                generateTime: generateTime,
                reusedSession: prepared.reusedTokens > 0,
                draftTokens: speculation?.totalDrafted,
                acceptedDraftTokens: speculation?.totalAccepted,
                peakMemoryBytes: Memory.peakMemory,
                engine: "alvin",
                phases: phases,
                prefilledTokens: prepared.prefilledTokens,
                reusedTokens: prepared.reusedTokens,
                planReason: prepared.reason,
                speculation: speculation,
                confidence: confidence.summary)
            if reason == .toolCalls {
                continuation.yield(.toolCalls(calls))
            }
            continuation.yield(.finished(EngineFinish(reason: reason, stats: stats)))
            continuation.finish()

            // Off the critical path: the reply has been delivered.
            runtime.persistPrefixIfNeeded(isAllowed: hooks.isAllowed)
            Memory.clearCache()
        } catch is PrefillInterrupted {
            runtime.interrupted()
            Memory.clearCache()
            let stats = LocalGenerationStats(engine: "alvin")
            continuation.yield(.finished(EngineFinish(reason: .cancelled, stats: stats)))
            continuation.finish()
        } catch EngineError.busy {
            // Nothing was changed: the session stays as it was.
            continuation.finish(throwing: EngineError.busy)
        } catch {
            runtime.abandon()
            Memory.clearCache()
            continuation.finish(throwing: error)
        }
    }

    // MARK: Extensions

    /// The configured extensions that prepared without error.
    func activeExtensions() -> [any EngineExtension] {
        configuration.extensions.filter { preparedExtensions[ObjectIdentifier($0)] == true }
    }

    /// Prepares extensions that haven't been yet; one that throws stays disabled.
    func prepareExtensionsIfNeeded() {
        for engineExtension in configuration.extensions where preparedExtensions[ObjectIdentifier(engineExtension)] == nil {
            do {
                try engineExtension.prepare(self)
                preparedExtensions[ObjectIdentifier(engineExtension)] = true
            } catch {
                preparedExtensions[ObjectIdentifier(engineExtension)] = false
                print("LocalEngine: extension \(engineExtension.name) disabled: \(error)")
            }
        }
    }

    // MARK: Helpers

    private func stream(
        _ body: @escaping (AsyncThrowingStream<EngineEvent, Error>.Continuation, StreamCancellation) -> Void
    ) -> AsyncThrowingStream<EngineEvent, Error> {
        AsyncThrowingStream { continuation in
            let cancellation = StreamCancellation()
            continuation.onTermination = { _ in cancellation.cancel() }
            queue.async {
                body(continuation, cancellation)
            }
        }
    }

    /// Runs `body` on the engine queue.
    func onQueue<T>(_ body: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<T, Error>) in
            queue.async {
                continuation.resume(with: Result { try body() })
            }
        }
    }

    /// Runs `body` between `beginGPU` and `endGPU`; throws `leftForeground` if refused.
    private func withGPU<T>(_ body: () throws -> T) throws -> T {
        let hooks = configuration.hooks
        guard hooks.beginGPU() else { throw EngineError.leftForeground }
        defer { hooks.endGPU() }
        return try body()
    }

    private func updateInfo(_ change: (inout EngineInfo) -> Void) {
        infoLock.lock()
        defer { infoLock.unlock() }
        change(&currentInfo)
    }

    private func summary() -> String {
        let snapshot = session.snapshot
        let marks = session.checkpoints.marks.sorted { $0.value < $1.value }.map { "\($0.key.rawValue)=\($0.value)" }
        let prefixOnDisk = snapshot.map { runtime.prefixStore?.contains(key: $0.prefixKey) ?? false } ?? false
        var parts = [
            snapshot == nil ? "cold" : "warm",
            "ledger \(session.ledger.count) tokens",
            "turns \(snapshot?.turns.count ?? 0)",
            "marks [\(marks.joined(separator: ", "))]",
            "checkpoints \(session.checkpoints.bytes / (1 << 20)) MiB",
            "prefix on disk \(prefixOnDisk ? "yes" : "no")",
            currentInfo.forked ? "fork" : "stock",
        ]
        if session.noReuse { parts.append("no reuse") }
        return parts.joined(separator: ", ")
    }
}

/// Set when a reply's stream is terminated by its consumer.
final class StreamCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    func cancel() {
        lock.lock()
        cancelled = true
        lock.unlock()
    }
}
