import AssistantKit
import Foundation
import LocalEngine

/// LocalEngine's fast kernels (`FastKernelsExtension`, plan WP41) as the app drives them: the
/// request in the engine configuration, the verdict of the extension's `prepare` (run in
/// `warmUp()`), and a dry-run comparison for Settings.
///
/// The kernels reach LocalEngine in the same wave as the app's engine files, so only launch setup
/// names them. It puts one `FastKernelsExtension` in `EngineSetup.extraExtensions` and declares
/// the conformances at file scope; the extension's `report` and `measure(_:)` and every member
/// of its report already match:
///
/// ```swift
/// extension FastKernelsExtension: FastKernelsProviding {
///     func request(_ requested: Bool, in configuration: inout EngineConfiguration) {
///         configuration.fastKernelsRequested = requested
///     }
/// }
/// extension FastKernelsReport: FastKernelsVerdict {}
/// ```
protocol FastKernelsProviding: EngineExtension, Sendable {
    associatedtype Verdict: FastKernelsVerdict

    /// Asks for the kernels in `configuration`, or stops asking. The request is kept among
    /// `configuration.extensions`, so it is made after they are assigned.
    func request(_ requested: Bool, in configuration: inout EngineConfiguration)

    /// What the latest `prepare` found (for whichever engine it prepared); nil before any.
    var report: Verdict? { get }

    /// Runs the kernel self-test and compares the cost curves with and without the kernels on
    /// `engine` now, whatever was requested, and leaves the model's layers as they were.
    func measure(_ engine: InferenceEngine) async throws -> Verdict
}

/// What the fast kernels found for one model.
protocol FastKernelsVerdict: Sendable {
    var modelID: String { get }
    /// How much cheaper an 8-row forward was with the kernels: `1 − c(8) with ÷ c(8) without`.
    /// Nil when nothing was compared (not requested, unavailable, or the self-test failed).
    var gain: Double? { get }
    /// Whether the model runs the kernels (only ever after a `prepare` that kept them).
    var isEnabled: Bool { get }
    /// One line for the engine summary, with the before/after c(8) when they were compared.
    var summary: String { get }
}

/// How the app configures the on-device engine (plan §4, WP40 instruction 4).
@MainActor
enum EngineSetup {
    /// Engine extensions added at launch (the fast kernels, an MTP drafter). Every load passes
    /// those `wantsExtension` keeps.
    static var extraExtensions: [any EngineExtension] = []

    /// Whether `extraExtensions` member `engineExtension` is used under `settings`; by default
    /// every one is. The extensions can't read the settings, so launch setup uses this to keep an
    /// MTP drafter to `localMTP`; such a change applies from the next reply. The fast kernels
    /// need no filter: they read `localFastKernels` from the configuration's request
    /// (`FastKernelsProviding`), and a change of it reloads the model.
    static var wantsExtension: @MainActor (_ engineExtension: any EngineExtension, _ settings: AssistantSettings) -> Bool = { _, _ in true }

    /// The fast kernels among `extraExtensions`, when launch setup put them there.
    static var fastKernels: (any FastKernelsProviding)? {
        for engineExtension in extraExtensions {
            if let kernels = engineExtension as? any FastKernelsProviding {
                return kernels
            }
        }
        return nil
    }

    /// The fast kernels count as helping when they make an 8-row forward at least this much
    /// cheaper (plan WP41: c(8) improved by ≥ 25%, `FastKernelsExtension`'s default
    /// `requiredGain`).
    static let kernelGainThreshold = 0.25

    /// The engine configuration for `settings`, with `host`'s measured cost curve, loaded draft
    /// model and GPU hooks.
    static func configuration(settings: AssistantSettings, host: LocalModelHost) -> EngineConfiguration {
        var configuration = EngineConfiguration()
        // The app's sampling, as the stock path uses it.
        configuration.temperature = 0.7
        configuration.topP = 0.8
        configuration.topK = 20
        configuration.maxTokens = 1024
        configuration.prefixCacheDirectory = prefixCacheDirectory(settings: settings)
        configuration.generatorFactory = generatorFactory(settings: settings, host: host)
        configuration.extensions = extensions(for: settings)
        // After `extensions`: assigning them drops the request.
        fastKernels?.request(settings.localFastKernels, in: &configuration)
        configuration.hooks = LocalModelHost.engineHooks
        return configuration
    }

    /// Each reply's generator: speculative decoding in `localSpeculation`'s mode, with `host`'s
    /// measured cost curve (the stock MLX curve until one is measured), the drafting corpus of the
    /// model, and the draft model `host` loaded (only for a catalog model with a draft model while
    /// `localSpeculativeDecoding` is on). More than 4 drafts per round are allowed only while the
    /// model runs the fast kernels and the curve was measured with them; the draft policy then
    /// still caps them at 4 unless the curve gives c(9)/c(1) ≤ 2.
    static func generatorFactory(settings: AssistantSettings, host: LocalModelHost) -> SpeculativeGeneratorFactory {
        let measured = host.costCurve
        return SpeculativeGeneratorFactory(
            mode: settings.localSpeculation,
            curve: measured ?? .stockMLXDefault,
            maxDraft: host.kernelsInstalled && measured != nil ? 8 : 4,
            corpusURL: corpusURL(modelID: settings.localModelID),
            draftModel: host.draftModel
        )
    }

    /// The extensions of `extraExtensions` that `settings` wants (without the fast-kernel
    /// request, which `configuration(settings:host:)` adds).
    static func extensions(for settings: AssistantSettings) -> [any EngineExtension] {
        extraExtensions.filter { wantsExtension($0, settings) }
    }

    /// Where the system prefix is saved between launches; nil while `localPrefixCache` is off.
    static func prefixCacheDirectory(settings: AssistantSettings) -> URL? {
        settings.localPrefixCache ? cacheDirectory() : nil
    }

    // MARK: Files

    /// `Application Support/EngineCache`, created and excluded from backup.
    nonisolated static func cacheDirectory() -> URL {
        var folder = URL.applicationSupportDirectory.appending(path: "EngineCache", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? folder.setResourceValues(values)
        return folder
    }

    /// The drafting corpus (past replies and tool results, as token ids) of `modelID`, in the
    /// engine cache. One file per model, because the token ids only mean something to the
    /// tokenizer that made them.
    nonisolated static func corpusURL(modelID: String) -> URL {
        let name = String(modelID.map { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "." ? $0 : "_" })
        return cacheDirectory().appending(path: "corpus-\(name).bin", directoryHint: .notDirectory)
    }

    // MARK: Checkpoint contents

    /// Whether the checkpoint in `directory` holds multi-token-prediction weights (`mtp.*`
    /// tensors, the same test the engine's sanitize uses). Reads the shard index, else the header
    /// of each safetensors file, never the tensors themselves.
    nonisolated static func hasMTPWeights(in directory: URL) -> Bool {
        let index = directory.appending(path: "model.safetensors.index.json", directoryHint: .notDirectory)
        if let data = try? Data(contentsOf: index),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let map = json["weight_map"] as? [String: Any]
        {
            return map.keys.contains(where: isMTPTensor)
        }
        let files = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        return files.contains { file in
            file.pathExtension == "safetensors" && safetensorsTensorNames(file).contains(where: isMTPTensor)
        }
    }

    nonisolated static func isMTPTensor(_ name: String) -> Bool {
        name.contains("mtp.")
    }

    /// The tensor names in a safetensors file's header: 8 bytes of little-endian header length,
    /// then that much JSON. Empty when the file can't be read as safetensors.
    nonisolated static func safetensorsTensorNames(_ url: URL) -> [String] {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return [] }
        defer { try? handle.close() }
        guard let prefix = try? handle.read(upToCount: 8), prefix.count == 8 else { return [] }
        var length: UInt64 = 0
        for (index, byte) in prefix.enumerated() {
            length |= UInt64(byte) << (8 * UInt64(index))
        }
        // Real headers are kilobytes; anything huge isn't a safetensors header.
        guard length > 1, length < 64 << 20,
              let header = try? handle.read(upToCount: Int(length)), header.count == Int(length),
              let json = try? JSONSerialization.jsonObject(with: header) as? [String: Any]
        else { return [] }
        return json.keys.filter { $0 != "__metadata__" }
    }
}
