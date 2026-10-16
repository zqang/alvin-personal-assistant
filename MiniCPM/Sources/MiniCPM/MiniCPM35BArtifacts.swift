import Foundation

/// The 35B-A3B behind "Most accurate on device": the phone-specific LES split, its 40 expert shards
/// and their manifest, pinned by SHA-256 in the bundled `MiniCPM35BManifest.json`, which
/// `scripts/download_minicpm_35b.py` reads too. The set reads no photos (the 1B does), so it has no
/// projector. The DSpark draft is optional: it has no source until it's hosted, so it's used only
/// when present and verified, and a wrong one is never used.
public actor MiniCPM35BArtifacts {
    public typealias Artifact = MiniCPMArtifacts.Artifact

    public struct Paths: Sendable {
        public let model: URL
        /// The LES JSON manifest, not the directory containing it.
        public let experts: URL
        /// Nil when the draft is absent or fails its check: the model then runs target-only.
        public let draft: URL?
    }

    private struct Manifest: Decodable {
        let revision: String
        let artifacts: [Artifact]
        let draftProvenance: DraftProvenance
    }

    public struct DraftProvenance: Codable, Sendable {
        public let sourceRepository: String
        public let sourceRevision: String
        public let sourceWeightsSHA256: String
        public let sourceConfigSHA256: String
        public let pairedTargetSHA256: String
        public let runtimeRevision: String
        public let converter: String
        public let conversion: String
        public let tokenizer: String
        public let tensorCount: Int
        public let parameterCount: Int
        public let sizeBytes: Int64
        public let sha256: String
    }

    private static let manifest: Manifest = {
        let url = Bundle.module.url(forResource: "MiniCPM35BManifest", withExtension: "json")!
        return try! JSONDecoder().decode(Manifest.self, from: Data(contentsOf: url))
    }()
    public static let revision = manifest.revision
    public static let artifacts = manifest.artifacts
    public static let draftProvenance = manifest.draftProvenance
    /// What a download fetches: every file with a source, so the draft only once it's hosted.
    public static let totalBytes = artifacts.reduce(Int64(0)) { $0 + ($1.sourceURL == nil ? 0 : $1.sizeBytes) }
    // ponytail: see MiniCPMArtifacts.directory on the benchmark-era name.
    public static let directory = URL.applicationSupportDirectory.appending(path: "MiniCPM35BBenchmark", directoryHint: .isDirectory)

    /// Whether an iPhone with `physicalMemory` bytes can run the 35B. A 12 GB iPhone reports about
    /// 12.26e9 bytes, so a "12 GiB" line would reject it.
    public static func eligible(physicalMemory: UInt64) -> Bool {
        physicalMemory >= 11_000_000_000
    }

    private let directory: URL
    private let entries: [Artifact]
    private let downloader: MiniCPMArtifacts.Downloader
    private let freeBytes: MiniCPMArtifacts.FreeBytes
    private var isPreparing = false

    public init(directory: URL = MiniCPM35BArtifacts.directory) {
        self.init(directory: directory, artifacts: Self.artifacts)
    }

    init(directory: URL, artifacts: [Artifact],
         downloader: @escaping MiniCPMArtifacts.Downloader = { try await MiniCPMArtifacts.download(from: $0, to: $1, progress: $2) },
         freeBytes: @escaping MiniCPMArtifacts.FreeBytes = { try MiniCPMArtifacts.availableStorage(in: $0) }) {
        self.directory = directory
        self.entries = artifacts
        self.downloader = downloader
        self.freeBytes = freeBytes
    }

    /// Makes the set present and verified, downloading what's missing, and returns its paths.
    public func prepare(progress: @escaping @Sendable (Double) -> Void = { _ in }) async throws -> Paths {
        guard !isPreparing else { throw MiniCPMArtifacts.ArtifactError.alreadyPreparing }
        isPreparing = true
        defer { isPreparing = false }
        // The benchmark copied a projector in; nothing reads it, and removing it frees 1.12 GB.
        try? FileManager.default.removeItem(at: directory.appending(path: "mmproj-model-f16.gguf"))
        let found = try await MiniCPMArtifacts.fetch(entries, into: directory, downloader: downloader,
                                                     freeBytes: freeBytes, progress: progress)
        guard let model = found["model"], let experts = found["expertManifest"] else {
            throw MiniCPMArtifacts.ArtifactError.invalidExpertManifest
        }
        try Self.validateExpertManifest(experts, artifacts: entries)
        return Paths(model: model, experts: experts, draft: found["draft"])
    }

    static func validateExpertManifest(_ path: URL, artifacts: [Artifact]) throws {
        struct ExpertManifest: Decodable {
            let schema_version: Int
            let layout: String
            struct Model: Decodable { let arch: String; let expert_count: Int; let expert_used_count: Int }
            let model: Model
            struct Entry: Decodable {
                let layer: Int
                let tensor_family: String
                let repacked_file: String
                let repacked_offset: Int64
                let expert_stride: Int64
                let bytes_per_expert: Int64
            }
            let entries: [Entry]
        }
        let manifest = try JSONDecoder().decode(ExpertManifest.self, from: Data(contentsOf: path))
        let shards = Dictionary(uniqueKeysWithValues: artifacts.filter { $0.role == "expert" }.map { ($0.path, $0.sizeBytes) })
        guard manifest.schema_version == 1, manifest.layout == "layer_major_expert", manifest.model.arch == "qwen35moe",
              manifest.model.expert_count == 256, manifest.model.expert_used_count == 8,
              manifest.entries.count == 120, shards.count == 40 else { throw MiniCPMArtifacts.ArtifactError.invalidExpertManifest }
        var seen: Set<String> = []
        let families: Set<String> = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
        for entry in manifest.entries {
            guard (0..<40).contains(entry.layer), families.contains(entry.tensor_family),
                  entry.repacked_file == String(format: "layer_%03d.bin", entry.layer),
                  let size = shards["experts/" + entry.repacked_file],
                  entry.repacked_offset >= 0, entry.repacked_offset < size,
                  entry.bytes_per_expert > 0, entry.bytes_per_expert <= size,
                  entry.expert_stride > 0, entry.expert_stride <= size / 255,
                  entry.repacked_offset + 255 * entry.expert_stride + entry.bytes_per_expert <= size,
                  seen.insert("\(entry.layer)/\(entry.tensor_family)").inserted else {
                throw MiniCPMArtifacts.ArtifactError.invalidExpertManifest
            }
        }
    }
}
