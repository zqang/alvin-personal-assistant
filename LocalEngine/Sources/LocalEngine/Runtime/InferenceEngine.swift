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
///   decodes again; `retryContinuation` decodes that continuation again from the same point.
/// - A reply to a request with tools opens with a tool call when `ToolChoice` requires one or
///   the model samples it; the engine then feeds the call's opening, its function name picked
///   among the request's tools, and the generator decodes the arguments (all of it under
///   `ToolChoice.native`). A continuation does so under its own tool choice.
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

    /// Answers `request`: see `EngineEvent` for the order of events. A request that doesn't
    /// end with a user turn throws `renderFailed` and changes nothing.
    public func reply(_ request: EngineRequest) -> AsyncThrowingStream<EngineEvent, Error> {
        stream { [self] continuation, cancellation in
            self.generate(.reply(request), continuation: continuation, cancellation: cancellation)
        }
    }

    /// Continues the reply whose stream ended with `.toolCalls`, after `round` (the results of
    /// those calls), opening under `toolChoice` (by default the model alone, `.native`). Valid
    /// only right after `.toolCalls`; otherwise the stream throws `EngineError.busy`.
    public func continueReply(after round: ToolRound, toolChoice: ToolChoice = .native) -> AsyncThrowingStream<EngineEvent, Error> {
        stream { [self] continuation, cancellation in
            self.generate(.continuation(round, toolChoice), continuation: continuation, cancellation: cancellation)
        }
    }

    /// Decodes the reply's last continuation again, opening under `toolChoice`: the session goes
    /// back to the end of that continuation's tool results, so what it wrote leaves the cache and
    /// the reply. Valid only right after a continuation that asked for no tools (it finished, or
    /// its stream was ended); otherwise the stream throws `EngineError.busy`.
    public func retryContinuation(toolChoice: ToolChoice) -> AsyncThrowingStream<EngineEvent, Error> {
        stream { [self] continuation, cancellation in
            self.generate(.retry(toolChoice), continuation: continuation, cancellation: cancellation)
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
        case continuation(ToolRound, ToolChoice)
        /// The last continuation again (`retryContinuation`).
        case retry(ToolChoice)
    }

    private func generate(
        _ job: Job, continuation: AsyncThrowingStream<EngineEvent, Error>.Continuation, cancellation: StreamCancellation
    ) {
        // A request that can't be answered is refused before anything changes: the live
        // conversation and a reply waiting for tool results stay as they are.
        if case .reply(let request) = job {
            do {
                try SessionRuntime.validate(request)
            } catch {
                continuation.finish(throwing: error)
                return
            }
        }
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
                case .continuation(let round, _):
                    return try self.runtime.prepareContinuation(after: round, isAllowed: isAllowed)
                case .retry:
                    return try self.runtime.prepareRetry(isAllowed: isAllowed)
                }
            }
            guard let request = runtime.reply?.request else {
                throw EngineError.busy
            }
            // How the first token is chosen: a reply's by its request, a continuation by its own
            // choice; nil for a continuation left to the model, which opens as before tool choice.
            let toolChoice: ToolChoice?
            switch job {
            case .reply: toolChoice = request.toolChoice
            case .continuation(_, let choice), .retry(let choice): toolChoice = choice == .native ? nil : choice
            }
            continuation.yield(.progress(.prefillDone(prefilledTokens: prepared.prefilledTokens, reusedTokens: prepared.reusedTokens)))

            let streamer = TextStreamer(
                tokenizer: loaded.tokenizer, renderer: loaded.renderer, format: toolCallFormat,
                tools: JSONBridge.templateToolsOrNil(request.tools))
            let sampler = FastSampler(configuration: configuration, greedy: request.greedy)
            var decodeStart = Date()
            var phases = prepared.phases
            var firstToken = false
            var firstText: Date?
            var generated = 0
            /// Hands emitted tokens to the streamer and yields what they show.
            func deliver(_ emitted: [Int], finished: EngineFinish.Reason? = nil) {
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
            }

            // The first token, when the request offers tools: a tool call's opening is fed here,
            // the arguments are then decoded by the generator.
            var firstLogits = prepared.firstLogits
            var toolCallProbability: Double?
            var forcedToolCall = false
            if let toolChoice, !request.tools.isEmpty, let start = streamer.startToken {
                let opening = try withError {
                    try self.openReply(
                        firstLogits: prepared.firstLogits, start: start, request: request, toolChoice: toolChoice, sampler: sampler,
                        isAllowed: isAllowed, deliver: { deliver($0) })
                }
                firstLogits = opening.firstLogits
                toolCallProbability = opening.toolCallProbability
                forcedToolCall = opening.forced
            }

            let drafters = activeExtensions().flatMap { $0.drafters(for: request, engine: self) }
            for drafter in drafters {
                drafter.reset(ledger: session.ledger, request: request)
            }
            let confidence = ConfidenceRecorder()
            let context = GeneratorContext(
                session: session, sampler: sampler,
                firstLogits: firstLogits, stopTokens: loaded.stopTokenIDs,
                maxTokens: max(0, (request.maxTokens ?? configuration.maxTokens) - generated), request: request, drafters: drafters,
                insideToolCall: { streamer.insideToolCall }, isAllowed: isAllowed, renderer: loaded.renderer,
                toolCallFormat: toolCallFormat, confidence: confidence)
            let generator = (configuration.generatorFactory ?? PlainGeneratorFactory()).makeGenerator(context)
            if generated == 0 {
                // No opening was fed: the decode is timed from the generator's start.
                decodeStart = Date()
            }

            var reason: EngineFinish.Reason = .stop
            while true {
                let (emitted, finished) = try withError { try generator.step() }
                deliver(emitted, finished: finished)
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
                confidence: confidence.summary,
                toolCallProbability: toolCallProbability,
                forcedToolCall: forcedToolCall)
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
            // The cache may hold more than the ledger: start the session over.
            runtime.abandon()
            Memory.clearCache()
            continuation.finish(throwing: error)
        }
    }

    // MARK: Opening a tool call

    /// How a generation of a reply to a request with tools starts: the probability the model
    /// gives the tool-call start token at the first position (raw softmax), whether `toolChoice`
    /// forces a call, and the logits the generator starts from.
    ///
    /// - A forced call, or one sampled here (only when the start token is in the sampler's kept
    ///   set, so it could be), is opened by `openToolCall`: the logits after its opening.
    /// - A text token sampled here: the first logits with every other token at `-inf`, so the
    ///   generator emits that token (its confidence then reads 1).
    /// - Otherwise, and always for `.native`, the first logits, unchanged.
    private func openReply(
        firstLogits: MLXArray, start: Int, request: EngineRequest, toolChoice: ToolChoice, sampler: FastSampler,
        isAllowed: () -> Bool, deliver: ([Int]) -> Void
    ) throws -> (firstLogits: MLXArray, toolCallProbability: Double, forced: Bool) {
        let row = firstLogits.reshaped(-1).asType(.float32)
        let probability = Double(exp(row[start] - logSumExp(row, axis: -1)).item(Float.self))
        if toolChoice.forcesCall(probability: probability) {
            return try (openToolCall(start: start, request: request, isAllowed: isAllowed, deliver: deliver), probability, true)
        }
        guard toolChoice != .native, sampler.keptDistribution(firstLogits).contains(where: { $0.token == start }) else {
            return (firstLogits, probability, false)
        }
        let position = session.ledger.count
        let token = sampler.sample(firstLogits, positions: position ..< (position + 1)).tokens.item(Int.self)
        if token == start {
            return try (openToolCall(start: start, request: request, isAllowed: isAllowed, deliver: deliver), probability, false)
        }
        return (which(MLXArray.arange(row.dim(0)) .== token, firstLogits, MLXArray(-Float.infinity)), probability, false)
    }

    /// Feeds a tool call's opening (`callOpenings`): the start token, then a function name among
    /// the request's tools, taking the model's most likely allowed token where the openings
    /// branch. Each token is delivered (to the streamer) as it is fed; returns the logits after
    /// the opening. Throws `PrefillInterrupted` when the GPU stops being allowed between feeds.
    private func openToolCall(start: Int, request: EngineRequest, isAllowed: () -> Bool, deliver: ([Int]) -> Void) throws -> MLXArray {
        let openings = Self.callOpenings(tools: request.tools, format: toolCallFormat, start: start, renderer: loaded.renderer)
        var fed = 0
        func feed(_ path: [Int]) throws -> MLXArray {
            guard isAllowed() else { throw PrefillInterrupted() }
            let tokens = Array(path[fed...])
            deliver(tokens)
            fed = path.count
            guard let logits = session.feed(tokens, rows: .last).logits else {
                preconditionFailure("A `.last` forward returned no logits.")
            }
            return logits
        }
        let opening = try Self.selectOpening(openings) { prefix, allowed in
            let scores = try feed(prefix).take(MLXArray(allowed.map { Int32($0) })).asType(.float32).asArray(Float.self)
            return allowed[scores.indices.max { scores[$0] < scores[$1] } ?? 0]
        }
        return try feed(opening)
    }

    /// The tokens a call to one of `tools` opens with in `format`, up to the function's name:
    /// `<tool_call>\n<function=NAME>\n`, or `<tool_call>\n{"name": "NAME", "arguments":` (a
    /// trailing space is left to the model, which writes it with what follows). Each starts with
    /// `start`; `[[start]]` when the format has no skeletons.
    static func callOpenings(tools: [ToolDefinition], format: ToolCallFormat, start: Int, renderer: any ChatTemplateRendering) -> [[Int]] {
        let openings = ExemplarDrafter.skeletons(tools: tools, format: format, withParameter: false)
            .map { renderer.encodeRaw($0.replacingOccurrences(of: " +$", with: "", options: .regularExpression)) }
            .filter { $0.first == start }
        return openings.isEmpty ? [[start]] : openings
    }

    /// Walks `openings` (a trie, as token lists) from their shared start: a token every remaining
    /// opening has next is taken; where they differ, `choose(prefix, allowed)` picks among the
    /// `allowed` next tokens. Ends where an opening ends.
    static func selectOpening(_ openings: [[Int]], choose: (_ prefix: [Int], _ allowed: [Int]) throws -> Int) rethrows -> [Int] {
        var candidates = openings
        var path: [Int] = []
        while !candidates.isEmpty, !candidates.contains(where: { $0.count == path.count }) {
            var allowed: [Int] = []
            for candidate in candidates where !allowed.contains(candidate[path.count]) {
                allowed.append(candidate[path.count])
            }
            let token = try allowed.count == 1 ? allowed[0] : choose(path, allowed)
            path.append(token)
            candidates = candidates.filter { $0[path.count - 1] == token }
        }
        return path
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
