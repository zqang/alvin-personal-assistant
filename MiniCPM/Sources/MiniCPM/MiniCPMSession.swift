import AssistantKit
import Foundation
import CMiniCPM

/// Execution-only tuning. All profiles keep the same weights, top-8 routing and
/// exact target-greedy acceptance. The baseline reproduces the original benchmark.
public struct MiniCPMRuntimeConfiguration: Codable, Equatable, Sendable {
    public let name: String
    public let checkpointLayerInterval: Int
    public let draftMaximum: Int
    public let expertCacheSize: Int
    public let directGreedy: Bool
    public let cpuThreads: Int
    public let ioThreads: Int

    init(name: String, checkpointLayerInterval: Int, draftMaximum: Int, expertCacheSize: Int,
         directGreedy: Bool, cpuThreads: Int = 6, ioThreads: Int = 4) {
        self.name = name
        self.checkpointLayerInterval = checkpointLayerInterval
        self.draftMaximum = draftMaximum
        self.expertCacheSize = expertCacheSize
        self.directGreedy = directGreedy
        self.cpuThreads = cpuThreads
        self.ioThreads = ioThreads
    }

    private enum CodingKeys: String, CodingKey {
        case name, checkpointLayerInterval, draftMaximum, expertCacheSize, directGreedy, cpuThreads, ioThreads
    }
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        name = try values.decode(String.self, forKey: .name)
        checkpointLayerInterval = try values.decode(Int.self, forKey: .checkpointLayerInterval)
        draftMaximum = try values.decode(Int.self, forKey: .draftMaximum)
        expertCacheSize = try values.decode(Int.self, forKey: .expertCacheSize)
        directGreedy = try values.decode(Bool.self, forKey: .directGreedy)
        cpuThreads = try values.decodeIfPresent(Int.self, forKey: .cpuThreads) ?? 6
        ioThreads = try values.decodeIfPresent(Int.self, forKey: .ioThreads) ?? 4
    }

    public static let baseline = Self(name: "baseline", checkpointLayerInterval: 1, draftMaximum: 3, expertCacheSize: 32, directGreedy: false)
    public static let checkpoints4 = Self(name: "checkpoints4", checkpointLayerInterval: 4, draftMaximum: 3, expertCacheSize: 32, directGreedy: false)
    public static let cache48 = Self(name: "cache48", checkpointLayerInterval: 1, draftMaximum: 3, expertCacheSize: 48, directGreedy: false)
    public static let balanced = Self(name: "balanced", checkpointLayerInterval: 4, draftMaximum: 3, expertCacheSize: 48, directGreedy: true)
    public static let checkpoints8 = Self(name: "checkpoints8", checkpointLayerInterval: 8, draftMaximum: 3, expertCacheSize: 48, directGreedy: true)
    public static let draft2 = Self(name: "draft2", checkpointLayerInterval: 4, draftMaximum: 2, expertCacheSize: 48, directGreedy: true)
    public static let cache64 = Self(name: "cache64", checkpointLayerInterval: 4, draftMaximum: 3, expertCacheSize: 64, directGreedy: true)
    public static let io1 = Self(name: "io1", checkpointLayerInterval: 4, draftMaximum: 3, expertCacheSize: 48, directGreedy: true, ioThreads: 1)
    public static let io8 = Self(name: "io8", checkpointLayerInterval: 4, draftMaximum: 3, expertCacheSize: 48, directGreedy: true, ioThreads: 8)
    public static let threads4 = Self(name: "threads4", checkpointLayerInterval: 4, draftMaximum: 3, expertCacheSize: 48, directGreedy: true, cpuThreads: 4)
    public static let profiles: [Self] = [.baseline, .checkpoints4, .cache48, .balanced, .checkpoints8, .draft2, .cache64, .io1, .io8, .threads4]

    /// What the app runs the 35B with: target-only, with the measured `baseline` execution. The
    /// device ABBA run (D22) kept it: neither DSpark profile met decision 13 (first text 3.5–4.2 %
    /// slower, Swift decode −4.1 % or +6.9 % against the +10 % bar, peak resident 5.2–6.0 GB against
    /// 4.5 GB). So the draft isn't hosted and its manifest entry keeps no `sourceURL`: a fresh
    /// install never downloads it, and a draft already on the phone stays unused.
    public static let production: (configuration: MiniCPMRuntimeConfiguration, useDraft: Bool) = (.baseline, false)

    /// The native runtime this package links: llama.cpp-omni's commit and the first 8 hex digits of
    /// the local patch's SHA-256 (`scripts/prepare_minicpm.sh`). A prefix file another runtime wrote
    /// never loads. Change it with the `llama` binary target.
    // ponytail: the patch (Metal autorelease pools) made no difference on the iPhone (D20 A/B,
    // p = 0.82); drop it with the next runtime release, which needs a new asset, tag and checksum
    // (owner gate G1).
    public static let runtimeRevision = "96bbf61710d44421ea54e63f5194f9b17c20d877-p60c9974a"
}

public struct MiniCPMBenchmarkResult: Codable, Equatable, Sendable {
    public let text: String
    /// Text tokens plus image embedding tokens evaluated for the prompt.
    public let promptTokens: Int
    /// Sampled output tokens, excluding the end-of-generation token.
    public let generatedTokens: Int
    /// Committed target tokens evaluated, excluding prompt prefill and rejected drafts.
    public let decodedTokens: Int
    /// Includes image I/O, tokenization, image encoding and prompt evaluation.
    public let prefillSeconds: Double
    /// Synchronized decoding. DSpark includes drafting, verification, injection and rollback.
    public let decodeSeconds: Double
    /// Run start to first visible complete UTF-8 text; -1 if the model emitted none. For the
    /// streams (`reply`, `describePhoto`), measured where Swift receives the text.
    public let firstTextSeconds: Double
    public let totalSeconds: Double
    public let finishReason: String
    public let lesEnabled: Bool
    public let dsparkEnabled: Bool
    /// Proposed draft tokens submitted for verification; excludes unused tail rows at the token cap.
    public let draftTokens: Int
    public let acceptedDraftTokens: Int
    public let speculationRounds: Int
    public let draftSeconds: Double
    public let runtimeConfiguration: MiniCPMRuntimeConfiguration
    /// For the streams: the native runtime's own first-text time, from the start of the native
    /// call that generated (for `reply`, after the prefill). Nil for `run`.
    public var nativeFirstTextSeconds: Double?
    public var tokensPerSecond: Double { decodeSeconds > 0 ? Double(decodedTokens) / decodeSeconds : 0 }
}

extension MiniCPMBenchmarkResult {
    /// `measured` replaces the native timings with what a Swift stream saw (a nil prefill keeps
    /// the native one); the native first text is then kept in `nativeFirstTextSeconds`.
    init(_ native: minicpm_result, text: String, profile: String,
         measured: (prefill: Double?, firstText: Double, total: Double)? = nil) {
        self.init(
            text: text,
            promptTokens: Int(native.prompt_tokens), generatedTokens: Int(native.generated_tokens),
            decodedTokens: Int(native.decoded_tokens), prefillSeconds: measured?.prefill ?? native.prefill_seconds,
            decodeSeconds: native.decode_seconds, firstTextSeconds: measured?.firstText ?? native.first_text_seconds,
            totalSeconds: measured?.total ?? native.total_seconds, finishReason: native.finish_reason == 0 ? "endOfGeneration" : "tokenLimit",
            lesEnabled: native.les_enabled, dsparkEnabled: native.dspark_enabled,
            draftTokens: Int(native.draft_tokens), acceptedDraftTokens: Int(native.accepted_draft_tokens),
            speculationRounds: Int(native.speculation_rounds), draftSeconds: native.draft_seconds,
            runtimeConfiguration: MiniCPMRuntimeConfiguration(name: profile,
                checkpointLayerInterval: Int(native.runtime_config.checkpoint_layer_interval),
                draftMaximum: Int(native.runtime_config.draft_max), expertCacheSize: Int(native.runtime_config.expert_cache_size),
                directGreedy: native.runtime_config.direct_greedy,
                cpuThreads: Int(native.runtime_config.cpu_threads), ioThreads: Int(native.runtime_config.io_threads)),
            nativeFirstTextSeconds: measured == nil ? nil : native.first_text_seconds)
    }
}

/// What a loaded session holds.
public struct MiniCPMInfo: Equatable, Sendable {
    /// Mapped weights (`llama_model_size`, target plus draft). `phys_footprint` doesn't count them.
    public let weightsBytes: UInt64
    /// One recurrent-state checkpoint; 0 until one is taken, and for a pure-attention model.
    public let checkpointBytes: UInt64
    public let nCtx: Int
    /// Tokens in the cache.
    public let nPast: Int
    public let recurrent: Bool
    public let vision: Bool
    public let dspark: Bool

    init(_ native: minicpm_info) {
        weightsBytes = native.weights_bytes
        checkpointBytes = native.checkpoint_bytes
        nCtx = Int(native.n_ctx)
        nPast = Int(native.n_past)
        recurrent = native.recurrent
        vision = native.vision
        dspark = native.dspark
    }
}

public enum MiniCPMError: LocalizedError {
    case notLoaded
    case alreadyLoaded
    case invalidInput(String)
    case runtime(String)

    public var errorDescription: String? {
        switch self {
        case .notLoaded: return "Load the MiniCPM model before running the benchmark."
        case .alreadyLoaded: return "Unload the current MiniCPM model before loading another."
        case .invalidInput(let message), .runtime(let message): return message
        }
    }
}

/// This handle is separate from the actor's inference state so cancellation never
/// needs to wait for a long native call to yield. Native atomics and a condition
/// variable guard its cancellation and idle state.
private final class CancellationControl: @unchecked Sendable {
    let pointer: OpaquePointer

    init() {
        guard let pointer = minicpm_control_create() else {
            preconditionFailure("Could not allocate MiniCPM cancellation control")
        }
        self.pointer = pointer
    }
    deinit { minicpm_control_free(pointer) }
    func cancel() { minicpm_cancel(pointer) }
    func check() throws {
        if minicpm_is_cancelled(pointer) { throw CancellationError() }
    }
    /// Whether `run` was cancelled: a stream's body checks it before any native call, since the
    /// stream may have ended (and its caller moved on) before the body got the queue.
    func check(run: UInt64) throws {
        if minicpm_run_is_cancelled(pointer, run) { throw CancellationError() }
    }
    func wait(timeout: TimeInterval) -> Bool { minicpm_wait_until_idle(pointer, timeout) }
    func beginRun() -> UInt64 { minicpm_begin_run(pointer) }
    func cancel(run: UInt64) { minicpm_cancel_run(pointer, run) }
}

/// Receives a stream's text from the native callback, on the session's queue.
private final class TextSink {
    let continuation: AsyncThrowingStream<MiniCPMEvent, Error>.Continuation
    let start: ContinuousClock.Instant
    /// Runs before the first piece is delivered.
    var beforeFirstPiece: (() -> Void)?
    private(set) var text = ""
    private(set) var firstText: Duration?

    init(_ continuation: AsyncThrowingStream<MiniCPMEvent, Error>.Continuation, start: ContinuousClock.Instant) {
        self.continuation = continuation
        self.start = start
    }

    func receive(_ piece: String) -> Bool {
        beforeFirstPiece?()
        beforeFirstPiece = nil
        if firstText == nil, piece.contains(where: { !$0.isWhitespace }) {
            firstText = start.duration(to: .now)
            continuation.yield(.firstText)
        }
        text += piece
        if case .terminated = continuation.yield(.text(piece)) { return false }
        return true
    }

    /// The `minicpm_text_fn` for a sink passed as `user`.
    static let callback: minicpm_text_fn = { bytes, length, user in
        guard let bytes, let user else { return true }
        let sink = Unmanaged<TextSink>.fromOpaque(user).takeUnretainedValue()
        return sink.receive(String(decoding: UnsafeRawBufferPointer(start: bytes, count: Int(length)), as: UTF8.self))
    }
}

/// MiniCPM-V 4.7 dense 1B or 35B-A3B LES with optional DSpark.
/// Configuration: Metal, 4096 context, 512 batch, 128 microbatch, greedy sampling,
/// one overview image slice for `run`. GPU layer checkpoints make lifecycle cancellation
/// responsive; their synchronization overhead is included in timings.
///
/// `run` (the benchmark) starts each prompt with empty target and draft caches. `reply` keeps
/// the conversation in the cache and reuses it; `describePhoto` reads a photo from memory,
/// starting fresh each time.
///
/// `cancel()` is latched, for unload and background: create a fresh session for new work;
/// `unload()` remains valid and releases the cancelled session's model. Ending a stream, or
/// cancelling `prewarm`'s task, cancels only that call and keeps the model loaded.
public actor MiniCPMSession {
    /// Native calls run on this serial queue, the actor's executor, so a long call blocks no
    /// Swift concurrency thread and no two calls ever overlap.
    private nonisolated let queue = DispatchSerialQueue(label: "MiniCPMSession")
    public nonisolated var unownedExecutor: UnownedSerialExecutor { queue.asUnownedSerialExecutor() }

    private nonisolated let cancellation = CancellationControl()
    private var handle: OpaquePointer?
    private var runtimeConfiguration: MiniCPMRuntimeConfiguration = .baseline
    /// The model's identity in prefix keys: its file name, then the runtime and the size and
    /// modification time of the model, expert manifest and draft (a changed file, setup or runtime
    /// never loads a stale prefix).
    private var modelIdentity = (id: "", revision: "")
    /// The id of `<|im_start|>`.
    private var turnStart = 0
    /// The conversation the cache holds, for `reply`.
    private var snapshot: SessionSnapshot?
    /// The planner rule of the last `reply` and how long its rewind took, for tests.
    private(set) var lastPlan: SessionPlan.Reason?
    private(set) var lastRewindSeconds = 0.0

    /// Memory for a reply's checkpoints (§3.2 step 5): all three fit for the 35B.
    static let checkpointBudgetBytes = 256 << 20
    /// Turns a rebuilt cache starts from; a window that doesn't fit is shortened. A rebuild fills
    /// at most half the budget (only a lone turn may use it all), so the appends after it fit for
    /// a while: a rebuild re-reads its window, tens of seconds on the 35B.
    static let windows = [12, 6, 3, 1]

    public init() {}
    deinit { if let handle { autoreleasepool { minicpm_free(handle) } } }

    public nonisolated func cancel() { cancellation.cancel() }

    /// Call cancel first. A false result means native work is still retiring;
    /// callers must not start another GPU engine until it has become idle.
    public nonisolated func waitUntilIdle(timeout: TimeInterval) -> Bool {
        cancellation.wait(timeout: timeout.isFinite ? max(0, timeout) : 0)
    }

    /// A nil projector runs text only. `expertsURL` is the split expert manifest.
    /// DSpark uses up to three drafts, target top-8 experts, a shared 256-slot prefill
    /// bank, a configurable exact expert cache and four lookahead experts. It fails closed
    /// if the draft, hidden-state extraction or rollback cannot be initialized.
    public func load(modelURL: URL, projectorURL: URL?, expertsURL: URL? = nil, draftURL: URL? = nil,
                     configuration: MiniCPMRuntimeConfiguration = .baseline) async throws {
        try Task.checkCancellation()
        try cancellation.check()
        guard handle == nil else { throw MiniCPMError.alreadyLoaded }
        guard (1...8).contains(configuration.checkpointLayerInterval),
              (1...3).contains(configuration.draftMaximum),
              (32...80).contains(configuration.expertCacheSize),
              (1...8).contains(configuration.cpuThreads), (1...16).contains(configuration.ioThreads) else {
            throw MiniCPMError.invalidInput("Runtime profile is outside the validated tuning bounds.")
        }
        try Self.validateFile(modelURL, name: "MiniCPM model")
        if let projectorURL { try Self.validateFile(projectorURL, name: "MiniCPM vision projector") }
        if let expertsURL { try Self.validateFile(expertsURL, name: "MiniCPM expert manifest") }
        if let draftURL {
            guard expertsURL != nil else { throw MiniCPMError.invalidInput("DSpark requires a matching LES expert manifest.") }
            try Self.validateFile(draftURL, name: "MiniCPM DSpark draft")
        }
        let control = cancellation
        let nativeConfiguration = minicpm_runtime_config(
            checkpoint_layer_interval: Int32(configuration.checkpointLayerInterval),
            draft_max: Int32(configuration.draftMaximum), expert_cache_size: Int32(configuration.expertCacheSize),
            direct_greedy: configuration.directGreedy, cpu_threads: Int32(configuration.cpuThreads), io_threads: Int32(configuration.ioThreads))
        try await withTaskCancellationHandler {
            try control.check()
            guard handle == nil else { throw MiniCPMError.alreadyLoaded }
            var error: UnsafeMutablePointer<CChar>?
            // Swift concurrency workers do not guarantee an autorelease-pool
            // boundary per actor operation. Retire temporary Metal/ObjC owners
            // before another model can be loaded on a different worker.
            let loaded = autoreleasepool {
                modelURL.path.withCString { modelPath in
                    Self.withOptionalPath(projectorURL) { projectorPath in
                        Self.withOptionalPath(expertsURL) { expertsPath in
                            Self.withOptionalPath(draftURL) { draftPath in
                                minicpm_load_with_runtime_config(modelPath, projectorPath, expertsPath, draftPath,
                                                                nativeConfiguration, control.pointer, &error)
                            }
                        }
                    }
                }
            }
            defer { minicpm_string_free(error) }
            guard let loaded else {
                try control.check()
                throw MiniCPMError.runtime(error.map { String(cString: $0) } ?? "MiniCPM model loading failed.")
            }
            // Cancellation may arrive after native load completes but before it is
            // installed on the actor. Release that result instead of keeping it.
            do {
                try Task.checkCancellation()
                try control.check()
                handle = loaded
                runtimeConfiguration = configuration
                snapshot = nil
                func stamp(_ url: URL?) -> String {
                    guard let values = try? url?.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey]) else { return "none" }
                    return "\(values.fileSize ?? 0)-\(values.contentModificationDate?.timeIntervalSince1970 ?? 0)"
                }
                modelIdentity = (modelURL.lastPathComponent, [MiniCPMRuntimeConfiguration.runtimeRevision, stamp(modelURL),
                                                              "experts:" + stamp(expertsURL), "draft:" + stamp(draftURL)].joined(separator: "-"))
                let start = try tokens([.init(text: MiniCPMChat.turnStart, special: true)])
                guard start.count == 1 else { throw MiniCPMError.runtime("The model's vocabulary has no ChatML turn markers.") }
                turnStart = start[0]
            } catch {
                handle = nil
                autoreleasepool { minicpm_free(loaded) }
                throw error
            }
        } onCancel: {
            control.cancel()
        }
    }

    public func run(prompt: String, imageURL: URL? = nil, maxTokens: Int = 128) async throws -> MiniCPMBenchmarkResult {
        try Task.checkCancellation()
        try cancellation.check()
        guard handle != nil else { throw MiniCPMError.notLoaded }
        guard (1...2048).contains(maxTokens), !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              !prompt.utf8.contains(0), prompt.utf8.count <= 32768 else {
            throw MiniCPMError.invalidInput("Use a nonempty prompt up to 32768 UTF-8 bytes and 1–2048 output tokens.")
        }
        if let imageURL { try Self.validateFile(imageURL, name: "Benchmark image") }
        let control = cancellation
        return try await withTaskCancellationHandler {
            try control.check()
            guard let handle else { throw MiniCPMError.notLoaded }
            snapshot = nil
            var native = minicpm_result()
            var error: UnsafeMutablePointer<CChar>?
            let status = autoreleasepool {
                prompt.withCString { promptPointer in
                    if let imageURL {
                        return imageURL.path.withCString { imagePath in
                            minicpm_run(handle, promptPointer, imagePath, Int32(maxTokens), &native, &error)
                        }
                    }
                    return minicpm_run(handle, promptPointer, nil, Int32(maxTokens), &native, &error)
                }
            }
            defer {
                minicpm_string_free(native.text)
                minicpm_string_free(error)
            }
            try Task.checkCancellation()
            try control.check()
            if status == 2 { throw CancellationError() }
            guard status == 0 else {
                throw MiniCPMError.runtime(error.map { String(cString: $0) } ?? "MiniCPM benchmark failed.")
            }
            return MiniCPMBenchmarkResult(native, text: native.text.map { String(cString: $0) } ?? "", profile: runtimeConfiguration.name)
        } onCancel: {
            control.cancel()
        }
    }

    public func unload() {
        if let handle { autoreleasepool { minicpm_free(handle) } }
        handle = nil
        snapshot = nil
    }

    public var info: MiniCPMInfo {
        var native = minicpm_info()
        minicpm_get_info(handle, &native)
        return MiniCPMInfo(native)
    }

    // MARK: Streaming chat with cache reuse

    /// Makes the system prefix of `system` resident: kept if the live conversation starts with it,
    /// else loaded from `prefixDirectory`, else prefilled and saved there.
    public func prewarm(system: String, prefixDirectory: URL) async throws {
        try Task.checkCancellation()
        try cancellation.check()
        guard let handle else { throw MiniCPMError.notLoaded }
        let control = cancellation
        let run = control.beginRun()
        try await withTaskCancellationHandler {
            let key = prefixKey(system: system)
            let prefix = try tokens(MiniCPMChat.system(system))
            guard !prefix.isEmpty else { return }
            if let snapshot, snapshot.prefixKey == key, snapshot.systemEnd > 0, ledger(handle).count >= snapshot.systemEnd { return }
            snapshot = nil
            let url = prefixURL(key: key, in: prefixDirectory)
            do {
                if loadPrefix(handle, from: url, expecting: prefix) {
                    try native { minicpm_checkpoint(handle, 0, $0) }
                } else {
                    try feedTokens(handle, run: run, prefix)
                    try native { minicpm_checkpoint(handle, 0, $0) }
                    savePrefix(handle, to: url)
                }
            } catch {
                minicpm_reset(handle)
                throw error
            }
            snapshot = SessionSnapshot(prefixKey: key, systemEnd: prefix.count, firstTurnIndex: 0, turns: [], tokenCount: prefix.count)
        } onCancel: {
            control.cancel(run: run)
        }
    }

    /// Streams a reply to `turns` (the last one the user's) as LocalEngine's `SessionRuntime`
    /// does, without tools or drafters: plan with `SessionPlanner`, rewind or load the disk
    /// prefix, feed only what the cache lacks (checkpointing at `CheckpointPolicy`'s marks),
    /// generate. An empty `system` renders no system turn, as the benchmark's `run` sends.
    /// Ending the stream cancels the reply; the model stays loaded.
    public nonisolated func reply(system: String, turns: [ChatTurn], maxTokens: Int,
                                  prefixDirectory: URL) -> AsyncThrowingStream<MiniCPMEvent, Error> {
        let control = cancellation
        let run = control.beginRun()
        let (stream, continuation) = AsyncThrowingStream<MiniCPMEvent, Error>.makeStream()
        continuation.onTermination = { _ in control.cancel(run: run) }
        Task {
            await self.streamReply(run: run, system: system, turns: turns, maxTokens: maxTokens,
                                   prefixDirectory: prefixDirectory, continuation: continuation)
        }
        return stream
    }

    /// Streams what the vision model reads in an image (JPEG or PNG bytes), with up to
    /// `maxSlices` 448-px slices. Starts from an empty cache each time.
    public nonisolated func describePhoto(jpeg: Data, system: String, instruction: String, maxTokens: Int,
                                          maxSlices: Int) -> AsyncThrowingStream<MiniCPMEvent, Error> {
        let control = cancellation
        let run = control.beginRun()
        let (stream, continuation) = AsyncThrowingStream<MiniCPMEvent, Error>.makeStream()
        continuation.onTermination = { _ in control.cancel(run: run) }
        Task {
            await self.streamDescription(run: run, jpeg: jpeg, system: system, instruction: instruction,
                                         maxTokens: maxTokens, maxSlices: maxSlices, continuation: continuation)
        }
        return stream
    }

    private func streamReply(run: UInt64, system: String, turns: [ChatTurn], maxTokens: Int, prefixDirectory: URL,
                             continuation: AsyncThrowingStream<MiniCPMEvent, Error>.Continuation) {
        let start = ContinuousClock.now
        do {
            try cancellation.check(run: run)
            guard let handle else { throw MiniCPMError.notLoaded }
            guard (1...2048).contains(maxTokens), turns.last?.role == .user else {
                throw MiniCPMError.invalidInput("Reply to a conversation that ends with the user, with 1–2048 output tokens.")
            }
            let prefilled = try prefill(handle, run: run, system: system, turns: turns, maxTokens: maxTokens,
                                        prefixDirectory: prefixDirectory)
            continuation.yield(.prefilled(fed: prefilled.fed, reused: prefilled.reused))
            let prefillEnd = start.duration(to: .now).seconds

            let sink = TextSink(continuation, start: start)
            let result = UnsafeMutablePointer<minicpm_result>.allocate(capacity: 1)
            result.initialize(to: minicpm_result())
            var error: UnsafeMutablePointer<CChar>?
            defer {
                minicpm_string_free(result.pointee.text)
                result.deallocate()
                minicpm_string_free(error)
            }
            let status = autoreleasepool {
                minicpm_generate(handle, run, Int32(maxTokens), TextSink.callback,
                                 Unmanaged.passUnretained(sink).toOpaque(), result, &error)
            }
            // The ledger holds every token delivered, so the reply is recorded as delivered, even
            // when it was cut short.
            snapshot?.recordReply(ChatTurn(role: .assistant, text: sink.text.trimmingCharacters(in: .whitespacesAndNewlines)),
                                  tokenCount: ledger(handle).count)
            if status == 2 { throw CancellationError() }
            guard status == 0 else {
                abandon(handle)
                throw MiniCPMError.runtime(error.map { String(cString: $0) } ?? "MiniCPM generation failed.")
            }
            continuation.yield(.finished(MiniCPMBenchmarkResult(
                result.pointee, text: sink.text, profile: runtimeConfiguration.name,
                measured: (prefillEnd, sink.firstText?.seconds ?? -1, start.duration(to: .now).seconds))))
            continuation.finish()
        } catch {
            continuation.finish(throwing: error)
        }
    }

    private func streamDescription(run: UInt64, jpeg: Data, system: String, instruction: String, maxTokens: Int,
                                   maxSlices: Int, continuation: AsyncThrowingStream<MiniCPMEvent, Error>.Continuation) {
        let start = ContinuousClock.now
        do {
            try cancellation.check(run: run)
            guard let handle else { throw MiniCPMError.notLoaded }
            guard (1...2048).contains(maxTokens), (1...9).contains(maxSlices), !jpeg.isEmpty,
                  !instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  !instruction.utf8.contains(0), !system.utf8.contains(0) else {
                throw MiniCPMError.invalidInput("Read a nonempty image with an instruction, 1–9 slices and 1–2048 output tokens.")
            }
            snapshot = nil
            let sink = TextSink(continuation, start: start)
            let result = UnsafeMutablePointer<minicpm_result>.allocate(capacity: 1)
            result.initialize(to: minicpm_result())
            var error: UnsafeMutablePointer<CChar>?
            defer {
                minicpm_string_free(result.pointee.text)
                result.deallocate()
                minicpm_string_free(error)
            }
            var prefilled = false
            sink.beforeFirstPiece = {
                prefilled = true
                continuation.yield(.prefilled(fed: Int(result.pointee.prompt_tokens), reused: 0))
            }
            let status = autoreleasepool {
                jpeg.withUnsafeBytes { bytes in
                    system.withCString { systemPointer in
                        instruction.withCString { instructionPointer in
                            minicpm_describe_image(handle, run, systemPointer, instructionPointer,
                                                   bytes.bindMemory(to: UInt8.self).baseAddress, bytes.count,
                                                   Int32(maxSlices), Int32(maxTokens), TextSink.callback,
                                                   Unmanaged.passUnretained(sink).toOpaque(), result, &error)
                        }
                    }
                }
            }
            if status == 2 { throw CancellationError() }
            guard status == 0 else { throw MiniCPMError.runtime(error.map { String(cString: $0) } ?? "MiniCPM photo reading failed.") }
            if !prefilled { continuation.yield(.prefilled(fed: Int(result.pointee.prompt_tokens), reused: 0)) }
            continuation.yield(.finished(MiniCPMBenchmarkResult(
                result.pointee, text: sink.text, profile: runtimeConfiguration.name,
                measured: (nil, sink.firstText?.seconds ?? -1, start.duration(to: .now).seconds))))
            continuation.finish()
        } catch {
            continuation.finish(throwing: error)
        }
    }

    /// Brings the cache to `turns` and their generation prompt (§3.2 steps 1–5), leaving
    /// `snapshot` describing it, or what stays reusable after a cancellation.
    private func prefill(_ handle: OpaquePointer, run: UInt64, system: String, turns: [ChatTurn], maxTokens: Int,
                         prefixDirectory: URL) throws -> (fed: Int, reused: Int) {
        // 1. The key. 2. The plan, rendered against what its base leaves in the cache: the reuse the
        // planner picks if it really fits, else a rebuild from the longest window that fits half the
        // budget (`windows`).
        let key = prefixKey(system: system)
        let prefixFile = prefixURL(key: key, in: prefixDirectory)
        let systemTokens = try tokens(MiniCPMChat.system(system))
        let current = ledger(handle)
        let live = snapshot.flatMap { $0.tokenCount <= current.count ? $0 : nil }
        let hasPersistedPrefix = !systemTokens.isEmpty && FileManager.default.fileExists(atPath: prefixFile.path)
        let budget = info.nCtx - maxTokens - 8
        var chosen: (plan: SessionPlan, base: [Int], feed: [Int], systemEnd: Int)?
        // The planner's own choice first, then rebuilds: a zero token budget makes it rebuild from
        // the window. A rebuild it chose itself is tried in that order too, so only once.
        for (keepTurns, limit) in [(Self.windows[0], budget)] + Self.windows.map({ ($0, 0) }) {
            guard let plan = SessionPlanner.plan(live: live, prefixKey: key, hasPersistedPrefix: hasPersistedPrefix,
                                                 turns: turns, limits: .init(maxTokens: limit, keepTurns: keepTurns)) else {
                throw MiniCPMError.invalidInput("The conversation must end with the user.")
            }
            let reuses = [.append, .replaceLastReply, .replaceLastUserTurn].contains(plan.reason)
            if limit > 0, !reuses { continue }
            var base: [Int] = []
            var systemEnd = 0
            switch plan.base {
            case .keep(let position):
                base = Array(current.prefix(position))
                systemEnd = min(live?.systemEnd ?? 0, position)
            case .persistedPrefix:
                base = systemTokens
                systemEnd = systemTokens.count
            case .empty:
                break
            }
            let feed = try render(plan.pieces, system: system, systemTokens: systemTokens, after: base, systemEnd: &systemEnd)
            if base.count + feed.count <= (reuses || keepTurns == 1 ? budget : budget / 2) {
                chosen = (plan, base, feed, systemEnd)
                break
            }
        }
        guard let chosen else { throw MiniCPMError.invalidInput("The message is too long for the on-device model.") }
        var plan = chosen.plan
        var base = chosen.base
        var feed = chosen.feed
        let systemEnd = chosen.systemEnd
        lastPlan = plan.reason
        snapshot = nil

        // 3. The base.
        var needsPrefixSave = false
        lastRewindSeconds = 0
        switch plan.base {
        case .keep:
            do {
                let rewound = ContinuousClock.now
                try native { minicpm_rewind(handle, run, Int32(base.count), $0) }
                lastRewindSeconds = rewound.duration(to: .now).seconds
            } catch is CancellationError {
                // The re-feed stopped short: keep the deepest cut of the conversation the ledger holds.
                snapshot = Self.cut(of: live, describingAtMost: ledger(handle).count, key: key)
                throw CancellationError()
            } catch {
                abandon(handle)
                throw error
            }
        case .persistedPrefix:
            if !loadPrefix(handle, from: prefixFile, expecting: systemTokens) {
                feed = systemTokens + feed
                base = []
                plan.base = .empty
                needsPrefixSave = true
            }
        case .empty:
            minicpm_reset(handle)
            needsPrefixSave = systemEnd > 0 && !FileManager.default.fileExists(atPath: prefixFile.path)
        }

        // 5. Checkpoints at the marks the budget keeps; and, for a base inside the conversation,
        // one at the base until the prefill completes (slot 2, which `replyStart` then takes), so
        // that a cancelled prefill (a replaced early-start turn) rewinds there without re-feeding.
        let start = base.count
        let replyStart = start + feed.count
        let userStart = TurnDelta.lastUserTurnStart(in: (base + feed)[...], turnStart: turnStart)
        var marks: [CheckpointMark: Int] = [.replyStart: replyStart]
        if systemEnd > 0, systemEnd >= start { marks[.systemEnd] = systemEnd }
        if let userStart, userStart >= start, userStart > 0 { marks[.lastUserStart] = userStart }
        let kept = CheckpointPolicy.marksToKeep(marks, bytesPerMark: Int(info.checkpointBytes), budgetBytes: Self.checkpointBudgetBytes)
        var slots: [Int: Int32] = [:]
        for mark in CheckpointMark.allCases.reversed() {
            guard let position = kept[mark] else { continue }
            slots[position] = mark == .systemEnd ? 0 : mark == .lastUserStart ? 1 : 2
        }
        if case .keep = plan.base, start > 0, slots[start] == nil, let live {
            let newest = live.newestUserTurnIndex.map { live.turns[$0] }
            if ![live.systemEnd, newest?.start, newest?.replyStart].contains(start) { slots[start] = 2 }
        }
        let resumable = Self.cut(of: live, base: plan.base, kept: start, key: key, systemEnd: systemEnd)
        let prefilled = SessionSnapshot.afterPrefill(prefixKey: key, systemEnd: systemEnd, plan: plan, turns: turns,
                                                     userStart: userStart, replyStart: replyStart)

        // 4 and 6. Feed, split at the checkpoints.
        do {
            var fed = 0
            for (position, slot) in slots.sorted(by: { $0.key < $1.key }) {
                try feedTokens(handle, run: run, Array(feed[fed ..< position - start]))
                fed = position - start
                try native { minicpm_checkpoint(handle, slot, $0) }
                if needsPrefixSave, position == systemEnd {
                    // ponytail: writes the prefix (about 73 MB for the 35B, plus a buffer of that size
                    // at its memory peak) on the reply's path, once per system prompt and only when no
                    // prewarm saved it. On the iPhone the write took 48–175 ms (D19) and a cold reply
                    // without the file 22.8–27.2 s to first audio, mostly the system prefill (D23). If
                    // the write ever costs, save after the reply from the slot-0 checkpoint and the
                    // attention cells below `systemEnd` (a token-count variant of minicpm_prefix_save).
                    savePrefix(handle, to: prefixFile)
                }
            }
            if fed < feed.count { try feedTokens(handle, run: run, Array(feed[fed...])) }
        } catch is CancellationError {
            // Once everything is in, the `replyStart` checkpoint has taken the base's slot.
            snapshot = ledger(handle).count == replyStart ? prefilled : resumable
            throw CancellationError()
        } catch {
            abandon(handle)
            throw error
        }
        snapshot = prefilled
        return (feed.count, start)
    }

    /// The tokens `pieces` add after `base` (§3.2 step 4): first turns without the system prefix,
    /// a continuation without the part the ledger already ends with (`TurnDelta.overlap`).
    private func render(_ pieces: [FeedPiece], system: String, systemTokens: [Int], after base: [Int],
                        systemEnd: inout Int) throws -> [Int] {
        var tokens: [Int] = []
        for piece in pieces {
            switch piece {
            case .systemPrefix:
                if base.isEmpty, tokens.isEmpty, !systemTokens.isEmpty {
                    tokens = systemTokens
                    systemEnd = systemTokens.count
                }
            case .firstTurns(let turns):
                let full = try self.tokens(MiniCPMChat.fullRender(system: system, turns: turns))
                guard let rest = TurnDelta.dropPrefix(base + tokens, from: full) else {
                    throw MiniCPMError.runtime("The conversation's render doesn't start with the cached system prompt.")
                }
                tokens += rest
            case .continuation(let turns):
                let delta = try self.tokens(MiniCPMChat.continuation(turns))
                let tail = (base.suffix(4) + tokens).suffix(4)
                tokens += delta.dropFirst(TurnDelta.overlap(ledgerTail: tail, delta: delta))
            case .assistantText(let text):
                tokens += try self.tokens(MiniCPMChat.assistantText(text))
            }
        }
        return tokens
    }

    /// Feeds `tokens`; an empty feed still feeds the pending token.
    private func feedTokens(_ handle: OpaquePointer, run: UInt64, _ tokens: [Int]) throws {
        let native = tokens.map(Int32.init)
        try self.native { minicpm_feed(handle, run, native, Int32(native.count), $0) }
    }

    /// Forgets the conversation after a failure: the cache starts empty next time.
    private func abandon(_ handle: OpaquePointer) {
        minicpm_reset(handle)
        snapshot = nil
    }

    /// Drops the live conversation (tests use it for a fresh render).
    func invalidate() {
        if let handle { abandon(handle) }
    }

    // MARK: Native helpers

    /// Runs a native call that reports a status (0 ok, 1 error, 2 cancelled) and an error string.
    private func native(_ body: (UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> Int32) throws {
        var error: UnsafeMutablePointer<CChar>?
        let status = autoreleasepool { body(&error) }
        defer { minicpm_string_free(error) }
        if status == 2 { throw CancellationError() }
        guard status == 0 else { throw MiniCPMError.runtime(error.map { String(cString: $0) } ?? "MiniCPM call failed.") }
    }

    /// The tokens of `pieces`, each tokenized on its own.
    func tokens(_ pieces: [MiniCPMChat.Piece]) throws -> [Int] {
        guard let handle else { throw MiniCPMError.notLoaded }
        var tokens: [Int] = []
        for piece in pieces where !piece.text.isEmpty {
            // A NUL would end the C string early.
            let text = piece.text.replacingOccurrences(of: "\0", with: "")
            let capacity = text.utf8.count + 8
            var buffer = [Int32](repeating: 0, count: capacity)
            var count: Int32 = 0
            try native { error in
                text.withCString { minicpm_tokenize(handle, $0, piece.special, &buffer, Int32(capacity), &count, error) }
            }
            tokens += buffer.prefix(Int(count)).map(Int.init)
        }
        return tokens
    }

    /// The tokens the cache holds.
    private func ledger(_ handle: OpaquePointer) -> [Int] {
        var count: Int32 = 0
        minicpm_ledger(handle, nil, 0, &count)
        let capacity = count
        var tokens = [Int32](repeating: 0, count: Int(capacity))
        guard minicpm_ledger(handle, &tokens, capacity, &count) == 0 else { return [] }
        return tokens.map(Int.init)
    }

    private func prefixKey(system: String) -> String {
        PrefixKey.make(modelID: modelIdentity.id, revision: modelIdentity.revision, system: system, toolsJSON: "[]",
                       context: PrefixKey.canonicalContext(["enable_thinking": false]), formatVersion: 1)
    }

    private func prefixURL(key: String, in directory: URL) -> URL {
        directory.appendingPathComponent(key + ".prefix")
    }

    /// Resets the cache and loads the prefix saved at `url` if it holds exactly `tokens`. A file
    /// that can't (another model, setup or runtime, or damaged) is deleted; one that failed for
    /// another reason, such as memory, is kept.
    private func loadPrefix(_ handle: OpaquePointer, from url: URL, expecting tokens: [Int]) -> Bool {
        guard !tokens.isEmpty, FileManager.default.fileExists(atPath: url.path) else {
            minicpm_reset(handle)
            return false
        }
        let expected = tokens.map(Int32.init)
        var error: UnsafeMutablePointer<CChar>?
        defer { minicpm_string_free(error) }
        let status = autoreleasepool {
            url.path.withCString { minicpm_prefix_load(handle, $0, expected, Int32(expected.count), &error) }
        }
        if status == 3 { try? FileManager.default.removeItem(at: url) }
        return status == 0
    }

    /// Saves the cache (exactly a system prefix) to `url` through a temporary file, keeping the two
    /// newest prefixes in the directory and removing temporary files a stopped save left. Best
    /// effort: a failure only costs a later prefill.
    private func savePrefix(_ handle: OpaquePointer, to url: URL) {
        let manager = FileManager.default
        var directory = url.deletingLastPathComponent()
        guard (try? manager.createDirectory(at: directory, withIntermediateDirectories: true)) != nil else { return }
        var excluded = URLResourceValues()
        excluded.isExcludedFromBackup = true
        try? directory.setResourceValues(excluded)
        let temporary = directory.appendingPathComponent(".\(UUID().uuidString).tmp")
        var error: UnsafeMutablePointer<CChar>?
        defer { minicpm_string_free(error) }
        let status = autoreleasepool { temporary.path.withCString { minicpm_prefix_save(handle, $0, &error) } }
        guard status == 0, rename(temporary.path, url.path) == 0 else {
            try? manager.removeItem(at: temporary)
            return
        }
        func modified(_ file: URL) -> Date {
            (try? file.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
        }
        let files = (try? manager.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        // The host saves from one session at a time, so a temporary file here is a stopped save's.
        for leftover in files where leftover.pathExtension == "tmp" { try? manager.removeItem(at: leftover) }
        let prefixes = files.filter { $0.pathExtension == "prefix" }
        for stale in prefixes.sorted(by: { modified($0) > modified($1) }).dropFirst(2) { try? manager.removeItem(at: stale) }
    }

    // MARK: Snapshot cuts (as LocalEngine's SessionRuntime)

    /// The snapshot describing the first `kept` tokens after a plan's base ran: the live snapshot
    /// cut back to `kept`, or the system prefix alone. Nil if no snapshot describes them.
    static func cut(of live: SessionSnapshot?, base: SessionPlan.Base, kept: Int, key: String, systemEnd: Int) -> SessionSnapshot? {
        switch base {
        case .keep:
            guard let live else { return nil }
            if kept == live.tokenCount { return live }
            if kept == live.systemEnd && kept > 0 {
                return SessionSnapshot(prefixKey: key, systemEnd: live.systemEnd, firstTurnIndex: live.firstTurnIndex, turns: [], tokenCount: kept)
            }
            guard let index = live.newestUserTurnIndex else { return nil }
            var cut = live
            if kept == live.turns[index].replyStart {
                cut.turns = Array(live.turns[...index])
            } else if kept == live.turns[index].start {
                cut.turns = Array(live.turns[..<index])
            } else {
                return nil
            }
            cut.tokenCount = kept
            return cut
        case .persistedPrefix, .empty:
            guard kept > 0, kept == systemEnd else { return nil }
            return SessionSnapshot(prefixKey: key, systemEnd: systemEnd, firstTurnIndex: 0, turns: [], tokenCount: kept)
        }
    }

    /// The deepest cut of `live` that describes at most `limit` tokens: what stays reusable after
    /// a rewind stopped short. Nil if none does.
    static func cut(of live: SessionSnapshot?, describingAtMost limit: Int, key: String) -> SessionSnapshot? {
        guard let live else { return nil }
        var cuts = [live.tokenCount, live.systemEnd]
        if let index = live.newestUserTurnIndex {
            cuts += [live.turns[index].replyStart, live.turns[index].start].compactMap { $0 }
        }
        for kept in cuts.filter({ $0 <= limit }).sorted(by: >) {
            if let cut = cut(of: live, base: .keep(kept), kept: kept, key: key, systemEnd: live.systemEnd) {
                return cut
            }
        }
        return nil
    }

    private static func withOptionalPath<T>(_ url: URL?, _ body: (UnsafePointer<CChar>?) -> T) -> T {
        guard let url else { return body(nil) }
        return url.path.withCString(body)
    }

    private static func validateFile(_ url: URL, name: String) throws {
        guard url.isFileURL, !url.path.utf8.contains(0), FileManager.default.isReadableFile(atPath: url.path) else {
            throw MiniCPMError.invalidInput("\(name) is not a readable local file.")
        }
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values.isRegularFile == true, (values.fileSize ?? 0) > 0 else {
            throw MiniCPMError.invalidInput("\(name) must be a nonempty regular file.")
        }
    }
}

private extension Duration {
    var seconds: Double {
        let parts = components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}
