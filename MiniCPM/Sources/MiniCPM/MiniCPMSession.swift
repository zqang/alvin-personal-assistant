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
    /// Run start to first visible complete UTF-8 text; -1 if the model emitted none.
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
    public var tokensPerSecond: Double { decodeSeconds > 0 ? Double(decodedTokens) / decodeSeconds : 0 }
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
    func wait(timeout: TimeInterval) -> Bool { minicpm_wait_until_idle(pointer, timeout) }
}

/// MiniCPM-V 4.7 dense 1B or 35B-A3B LES with optional DSpark.
/// Each run starts with independent target and draft KV caches.
/// Configuration: Metal, 4096 context, 512 batch, 128 microbatch, greedy sampling,
/// one overview image slice. GPU layer checkpoints make lifecycle cancellation
/// responsive; their synchronization overhead is included in timings.
///
/// Cancellation is latched: after `cancel()`, create a fresh session for new work.
/// `unload()` remains valid and releases the cancelled session's model.
public actor MiniCPMSession {
    private nonisolated let cancellation = CancellationControl()
    private var handle: OpaquePointer?
    private var runtimeConfiguration: MiniCPMRuntimeConfiguration = .baseline

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
            } catch {
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
            return MiniCPMBenchmarkResult(
                text: native.text.map { String(cString: $0) } ?? "",
                promptTokens: Int(native.prompt_tokens), generatedTokens: Int(native.generated_tokens),
                decodedTokens: Int(native.decoded_tokens), prefillSeconds: native.prefill_seconds,
                decodeSeconds: native.decode_seconds, firstTextSeconds: native.first_text_seconds,
                totalSeconds: native.total_seconds, finishReason: native.finish_reason == 0 ? "endOfGeneration" : "tokenLimit",
                lesEnabled: native.les_enabled, dsparkEnabled: native.dspark_enabled,
                draftTokens: Int(native.draft_tokens), acceptedDraftTokens: Int(native.accepted_draft_tokens),
                speculationRounds: Int(native.speculation_rounds), draftSeconds: native.draft_seconds,
                runtimeConfiguration: MiniCPMRuntimeConfiguration(name: runtimeConfiguration.name,
                    checkpointLayerInterval: Int(native.runtime_config.checkpoint_layer_interval),
                    draftMaximum: Int(native.runtime_config.draft_max), expertCacheSize: Int(native.runtime_config.expert_cache_size),
                    directGreedy: native.runtime_config.direct_greedy,
                    cpuThreads: Int(native.runtime_config.cpu_threads), ioThreads: Int(native.runtime_config.io_threads))
            )
        } onCancel: {
            control.cancel()
        }
    }

    public func unload() {
        if let handle { autoreleasepool { minicpm_free(handle) } }
        handle = nil
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
