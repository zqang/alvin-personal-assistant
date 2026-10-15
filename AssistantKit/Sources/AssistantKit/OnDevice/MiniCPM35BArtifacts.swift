import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
#if canImport(CryptoKit)
import CryptoKit
#endif

/// The phone-specific LES split and DSpark draft, independently pinned by SHA256.
/// The Python preparation script consumes the same bundled manifest. The draft is
/// locally converted from the published OpenBMB checkpoint, never silently omitted.
public actor MiniCPM35BArtifacts {
    public struct Artifact: Codable, Equatable, Sendable {
        public let path: String
        public let sizeBytes: Int64
        public let sha256: String
        public let role: String
        public let sourceURL: URL?
    }

    public struct Paths: Sendable {
        public let model: URL
        public let projector: URL
        /// The LES JSON manifest, not the directory containing it.
        public let experts: URL
        public let draft: URL
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
    public static var totalBytes: Int64 { artifacts.reduce(0) { $0 + $1.sizeBytes } }
    public static let modelFilename = "MiniCPM-V-4_7-35B-A3B-mix_q3gud_im-les.gguf"
    public static let draftFilename = "MiniCPMV47-DSpark-Q8_0.gguf"

    public enum ArtifactError: LocalizedError, Equatable {
        case alreadyPreparing
        case unsafePath(String)
        case invalidArtifact(String)
        case invalidExpertManifest
        case missingLocalDraft
        case insufficientStorage(required: Int64)
        case verificationUnavailable
        case unexpectedHTTPStatus(Int)

        public var errorDescription: String? {
            switch self {
            case .alreadyPreparing: return "MiniCPM 35B preparation is already running."
            case .unsafePath(let path): return "Unsafe MiniCPM artifact path: \(path)."
            case .invalidArtifact(let path): return "MiniCPM artifact failed its pinned size, header, or SHA256 check: \(path)."
            case .invalidExpertManifest: return "The LES manifest does not match the pinned expert shards."
            case .missingLocalDraft: return "The pinned DSpark draft is missing. Prepare it on the Mac with scripts/convert_minicpm_dspark.py and copy it into the MiniCPM 35B model directory."
            case .insufficientStorage(let required): return "MiniCPM 35B needs at least \(required / 1_000_000) MB of free storage to finish preparation."
            case .verificationUnavailable: return "MiniCPM artifact verification requires CryptoKit."
            case .unexpectedHTTPStatus(let status): return "MiniCPM 35B download returned HTTP \(status)."
            }
        }
    }

    typealias Downloader = @Sendable (URL, URL, @escaping @Sendable (Int64) -> Void) async throws -> Void
    private let directory: URL
    private let entries: [Artifact]
    private let downloader: Downloader
    private var isPreparing = false

    public init(directory: URL) {
        self.directory = directory
        self.entries = Self.artifacts
        self.downloader = { source, destination, progress in
            try await Self.download(from: source, to: destination, progress: progress)
        }
    }

    init(directory: URL, artifacts: [Artifact], downloader: @escaping Downloader) {
        self.directory = directory
        self.entries = artifacts
        self.downloader = downloader
    }

    public func prepare(progress: @escaping @Sendable (Double) -> Void = { _ in }) async throws -> Paths {
        guard !isPreparing else { throw ArtifactError.alreadyPreparing }
        isPreparing = true
        defer { isPreparing = false }
        try Task.checkCancellation()
        #if !canImport(CryptoKit)
        throw ArtifactError.verificationUnavailable
        #else
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        try Self.excludeFromBackup(directory)
        guard entries.contains(where: { $0.role == "draft" }) else { throw ArtifactError.missingLocalDraft }
        let total = entries.reduce(Int64(0)) { $0 + $1.sizeBytes }
        var completed: Int64 = 0
        var missing: [Artifact] = []
        var paths: [String: URL] = [:]
        progress(0)
        for artifact in entries {
            try Task.checkCancellation()
            let target = try Self.safePath(artifact.path, in: directory)
            paths[artifact.role] = target
            if try Self.valid(target, artifact: artifact) {
                completed += artifact.sizeBytes
                progress(Double(completed) / Double(total))
            } else {
                guard artifact.sourceURL != nil else { throw ArtifactError.missingLocalDraft }
                missing.append(artifact)
            }
        }
        if !missing.isEmpty {
            let required = missing.reduce(Int64(0)) { $0 + $1.sizeBytes } + 536_870_912
            let free = try Self.availableStorage(in: directory)
            guard free >= required else { throw ArtifactError.insufficientStorage(required: required) }
        }
        for artifact in missing {
            try Task.checkCancellation()
            let target = try Self.safePath(artifact.path, in: directory)
            try manager.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            let staged = try Self.safePath(artifact.path + ".download", in: directory)
            defer { try? manager.removeItem(at: staged) }
            let previous = completed
            try await downloader(artifact.sourceURL!, staged) { received in
                progress(min(0.99, Double(previous + min(artifact.sizeBytes, max(0, received))) / Double(total)))
            }
            guard try Self.valid(staged, artifact: artifact) else { throw ArtifactError.invalidArtifact(artifact.path) }
            try Task.checkCancellation()
            try Self.excludeFromBackup(staged)
            if manager.fileExists(atPath: target.path) {
                _ = try manager.replaceItemAt(target, withItemAt: staged)
            } else {
                try manager.moveItem(at: staged, to: target)
            }
            completed += artifact.sizeBytes
            progress(Double(completed) / Double(total))
        }
        guard let model = paths["model"], let projector = paths["projector"],
              let experts = paths["expertManifest"], let draft = paths["draft"] else {
            throw ArtifactError.invalidExpertManifest
        }
        try Self.validateExpertManifest(experts, artifacts: entries)
        return Paths(model: model, projector: projector, experts: experts, draft: draft)
        #endif
    }

    private static func availableStorage(in directory: URL) throws -> Int64 {
        #if !os(Linux)
        // This download is explicitly requested by the user. Important-use capacity
        // includes storage iOS can reclaim; raw free bytes can be much smaller.
        if let values = try? directory.resourceValues(forKeys: [
            .volumeAvailableCapacityForImportantUsageKey, .volumeAvailableCapacityKey
        ]) {
            if let important = values.volumeAvailableCapacityForImportantUsage, important >= 0 {
                return important
            }
            if let available = values.volumeAvailableCapacity, available >= 0 {
                return Int64(available)
            }
        }
        #endif
        let attributes = try FileManager.default.attributesOfFileSystem(forPath: directory.path)
        return max(0, (attributes[.systemFreeSize] as? NSNumber)?.int64Value ?? 0)
    }

    static func safePath(_ relative: String, in directory: URL) throws -> URL {
        let parts = relative.split(separator: "/", omittingEmptySubsequences: false)
        guard !relative.isEmpty, !relative.contains("\\"), parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw ArtifactError.unsafePath(relative)
        }
        let root = directory.resolvingSymlinksInPath().standardizedFileURL
        var path = root
        // Resolving a nonexistent leaf can leave its symlinked parent unresolved.
        // Reject links at every component before creating directories or files.
        for part in parts {
            path.appendPathComponent(String(part))
            if (try? FileManager.default.attributesOfItem(atPath: path.path)[.type]) as? FileAttributeType == .typeSymbolicLink {
                throw ArtifactError.unsafePath(relative)
            }
        }
        return path
    }

    static func valid(_ path: URL, artifact: Artifact) throws -> Bool {
        try Task.checkCancellation()
        guard FileManager.default.fileExists(atPath: path.path) else { return false }
        let attributes = try FileManager.default.attributesOfItem(atPath: path.path)
        guard (attributes[.size] as? NSNumber)?.int64Value == artifact.sizeBytes else { return false }
        let file = try FileHandle(forReadingFrom: path)
        defer { try? file.close() }
        if artifact.path.hasSuffix(".gguf") {
            guard try file.read(upToCount: 4) == Data("GGUF".utf8) else { return false }
            try file.seek(toOffset: 0)
        }
        #if canImport(CryptoKit)
        var hasher = SHA256()
        while true {
            // FileHandle can autorelease its backing NSData. A synchronous hash
            // of all 17 GB otherwise retains those objects until this actor job
            // returns, despite reading only a megabyte at a time.
            let consumed = try Self.withTemporaryObjects {
                try Task.checkCancellation()
                guard let chunk = try file.read(upToCount: 1_048_576), !chunk.isEmpty else { return false }
                hasher.update(data: chunk)
                return true
            }
            if !consumed { break }
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined() == artifact.sha256
        #else
        throw ArtifactError.verificationUnavailable
        #endif
    }

    private static func withTemporaryObjects<T>(_ operation: () throws -> T) rethrows -> T {
        #if canImport(ObjectiveC)
        return try autoreleasepool(invoking: operation)
        #else
        return try operation()
        #endif
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
              manifest.entries.count == 120, shards.count == 40 else { throw ArtifactError.invalidExpertManifest }
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
                throw ArtifactError.invalidExpertManifest
            }
        }
    }

    private static func excludeFromBackup(_ path: URL) throws {
        #if os(iOS)
        var path = path
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try path.setResourceValues(values)
        #endif
    }

    private static func download(from source: URL, to destination: URL, progress: @escaping @Sendable (Int64) -> Void) async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.allowsCellularAccess = false
        configuration.timeoutIntervalForResource = 21_600
        #if !os(Linux)
        configuration.waitsForConnectivity = true
        configuration.allowsExpensiveNetworkAccess = false
        #endif
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        let resumeFile = destination.appendingPathExtension("resume")
        let delegate = MiniCPM35BDownloadProgress(progress)
        for attempt in 0..<2 {
            do {
                let result: (URL, URLResponse)
                if let resume = try? Data(contentsOf: resumeFile) {
                    result = try await session.download(resumeFrom: resume, delegate: delegate)
                } else {
                    var request = URLRequest(url: source)
                    request.timeoutInterval = 90
                    result = try await session.download(for: request, delegate: delegate)
                }
                let (temporary, response) = result
                defer { try? FileManager.default.removeItem(at: temporary) }
                guard let http = response as? HTTPURLResponse, http.statusCode == 200 || http.statusCode == 206 else {
                    throw ArtifactError.unexpectedHTTPStatus((response as? HTTPURLResponse)?.statusCode ?? -1)
                }
                try Task.checkCancellation()
                try? FileManager.default.removeItem(at: destination)
                try FileManager.default.moveItem(at: temporary, to: destination)
                try? FileManager.default.removeItem(at: resumeFile)
                return
            } catch {
                if let resume = (error as NSError).userInfo["NSURLSessionDownloadTaskResumeData"] as? Data {
                    try? resume.write(to: resumeFile, options: .atomic)
                    try? excludeFromBackup(resumeFile)
                } else {
                    try? FileManager.default.removeItem(at: resumeFile)
                }
                try Task.checkCancellation()
                if attempt == 1 { throw error }
            }
        }
    }
}

private final class MiniCPM35BDownloadProgress: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let progress: @Sendable (Int64) -> Void
    init(_ progress: @escaping @Sendable (Int64) -> Void) { self.progress = progress }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {}
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        progress(totalBytesWritten)
    }
}
