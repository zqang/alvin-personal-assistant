import AssistantKit
import Foundation
import LocalEngine

/// How the app configures the on-device engine (plan §4, WP40 instruction 4).
@MainActor
enum EngineSetup {
    /// Engine extensions added at launch (the fast kernels, an MTP drafter). Every load passes
    /// those `wantsExtension` keeps.
    static var extraExtensions: [any EngineExtension] = []

    /// Whether `extraExtensions` member `engineExtension` is used under `settings`; by default
    /// every one is. The extensions can't read the settings, so launch setup uses this to keep
    /// the fast kernels to `localFastKernels` and an MTP drafter to `localMTP`. A change of
    /// `localFastKernels` reloads the model; other changes apply from the next reply.
    static var wantsExtension: @MainActor (_ engineExtension: any EngineExtension, _ settings: AssistantSettings) -> Bool = { _, _ in true }

    /// The fast kernels count as helping when an 8-row forward runs at least this much faster
    /// with them (plan WP41: a ≥ 25% gain in c(8)).
    static let kernelGainThreshold = 1.25

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
        configuration.hooks = LocalModelHost.engineHooks
        return configuration
    }

    /// Each reply's generator: speculative decoding in `localSpeculation`'s mode, with `host`'s
    /// measured cost curve (the stock MLX curve until one is measured), the drafting corpus of the
    /// model, and the draft model `host` loaded (only for a catalog model with a draft model while
    /// `localSpeculativeDecoding` is on). More than 4 drafts per round are allowed only with the
    /// fast kernels and a measured curve; the draft policy then still caps them at 4 unless the
    /// curve gives c(9)/c(1) ≤ 2.
    static func generatorFactory(settings: AssistantSettings, host: LocalModelHost) -> SpeculativeGeneratorFactory {
        let measured = host.costCurve
        return SpeculativeGeneratorFactory(
            mode: settings.localSpeculation,
            curve: measured ?? .stockMLXDefault,
            maxDraft: settings.localFastKernels && measured != nil ? 8 : 4,
            corpusURL: corpusURL(modelID: settings.localModelID),
            draftModel: host.draftModel
        )
    }

    /// The extensions of `extraExtensions` that `settings` wants.
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
