import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
#if canImport(CryptoKit)
import CryptoKit
#endif

/// The pinned GGUF pair used by the experimental MiniCPM-V 4.7 1B benchmark.
/// Downloads are foreground-only and disallow cellular access. Every use checks the
/// complete file checksum, including cached files; verification reads bounded chunks.
public actor MiniCPMArtifacts {
    public struct Manifest: Equatable, Sendable {
        public let filename: String
        public let sizeBytes: Int64
        public let sha256: String
        public let sourceURL: URL
    }

    public static let revision = "8716e5209f308fd53f197ccd7d50c6fdcc2dc37d"
    public static let model = Manifest(
        filename: "MiniCPM-V-4_7-1B-Q4_K_M.gguf",
        sizeBytes: 672_003_584,
        sha256: "bdc2c0da2320129a4fd91b0d464c63f8047b511a7ef7f7e1d143b433c7165cd5",
        sourceURL: artifactURL("MiniCPM-V-4_7-1B-Q4_K_M.gguf")
    )
    public static let projector = Manifest(
        filename: "mmproj-model-f16.gguf",
        sizeBytes: 1_108_746_816,
        sha256: "febb329c29ec9f1643c9d690ca74d35f486b83fa48c2f54c108b135729f635fd",
        sourceURL: artifactURL("mmproj-model-f16.gguf")
    )
    public static let manifests = [model, projector]
    public static let totalBytes = model.sizeBytes + projector.sizeBytes

    public enum ArtifactError: LocalizedError, Equatable {
        case alreadyPreparing
        case verificationUnavailable
        case unexpectedHTTPStatus(Int)
        case wrongSize(filename: String, expected: Int64, actual: Int64)
        case invalidGGUF(String)
        case checksumMismatch(String)

        public var errorDescription: String? {
            switch self {
            case .alreadyPreparing: return "MiniCPM model preparation is already running."
            case .verificationUnavailable: return "MiniCPM artifact verification requires CryptoKit on this platform."
            case .unexpectedHTTPStatus(let code): return "MiniCPM download returned HTTP \(code)."
            case .wrongSize(let filename, let expected, let actual):
                return "\(filename) is incomplete: expected \(expected) bytes, found \(actual)."
            case .invalidGGUF(let filename): return "\(filename) is not a GGUF model file."
            case .checksumMismatch(let filename): return "The checksum of \(filename) does not match the pinned model."
            }
        }
    }

    typealias Downloader = @Sendable (URL, URL, @escaping @Sendable (Int64) -> Void) async throws -> Void
    private let directory: URL
    private let artifacts: [Manifest]
    private let downloader: Downloader
    private var isPreparing = false

    public init(directory: URL) {
        self.directory = directory
        self.artifacts = Self.manifests
        self.downloader = { source, destination, progress in
            try await Self.download(from: source, to: destination, progress: progress)
        }
    }

    // Injectable transport and manifests keep corruption/cancellation tests small and offline.
    init(directory: URL, artifacts: [Manifest], downloader: @escaping Downloader) {
        precondition(artifacts.count == 2)
        self.directory = directory
        self.artifacts = artifacts
        self.downloader = downloader
    }

    public func prepare(progress: @escaping @Sendable (Double) -> Void = { _ in }) async throws -> (model: URL, projector: URL) {
        guard !isPreparing else { throw ArtifactError.alreadyPreparing }
        isPreparing = true
        defer { isPreparing = false }
        try Task.checkCancellation()
        #if !canImport(CryptoKit)
        throw ArtifactError.verificationUnavailable
        #else
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Self.excludeFromBackup(directory)
        let totalBytes = artifacts.reduce(Int64(0)) { $0 + $1.sizeBytes }
        var completedBytes: Int64 = 0
        var paths: [URL] = []
        progress(0)
        for artifact in artifacts {
            try Task.checkCancellation()
            let target = directory.appendingPathComponent(artifact.filename)
            if try !Self.isValidCache(target, artifact: artifact) {
                // A unique staging file in the same directory permits an atomic replacement.
                let staged = directory.appendingPathComponent(".\(artifact.filename).\(UUID().uuidString).download")
                defer { try? FileManager.default.removeItem(at: staged) }
                let previousBytes = completedBytes
                try await downloader(artifact.sourceURL, staged) { received in
                    let bytes = previousBytes + min(artifact.sizeBytes, max(0, received))
                    progress(min(0.99, Double(bytes) / Double(totalBytes)))
                }
                try Self.validate(staged, artifact: artifact)
                try Task.checkCancellation()
                try Self.excludeFromBackup(staged)
                if FileManager.default.fileExists(atPath: target.path) {
                    _ = try FileManager.default.replaceItemAt(target, withItemAt: staged)
                } else {
                    try FileManager.default.moveItem(at: staged, to: target)
                }
            }
            completedBytes += artifact.sizeBytes
            paths.append(target)
            progress(Double(completedBytes) / Double(totalBytes))
        }
        return (model: paths[0], projector: paths[1])
        #endif
    }

    private static func artifactURL(_ filename: String) -> URL {
        URL(string: "https://huggingface.co/tc-mb/MiniCPM-V-Apps-gguf/resolve/\(revision)/MiniCPM-V-4.7-1B/\(filename)")!
    }

    private static func isValidCache(_ path: URL, artifact: Manifest) throws -> Bool {
        guard FileManager.default.fileExists(atPath: path.path) else { return false }
        do {
            try validate(path, artifact: artifact)
            return true
        } catch ArtifactError.wrongSize { return false
        } catch ArtifactError.invalidGGUF { return false
        } catch ArtifactError.checksumMismatch { return false }
    }

    static func validate(_ path: URL, artifact: Manifest) throws {
        try Task.checkCancellation()
        let attributes = try FileManager.default.attributesOfItem(atPath: path.path)
        let size = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        guard size == artifact.sizeBytes else {
            throw ArtifactError.wrongSize(filename: artifact.filename, expected: artifact.sizeBytes, actual: size)
        }
        let file = try FileHandle(forReadingFrom: path)
        defer { try? file.close() }
        guard try file.read(upToCount: 4) == Data("GGUF".utf8) else {
            throw ArtifactError.invalidGGUF(artifact.filename)
        }
        try file.seek(toOffset: 0)
        #if canImport(CryptoKit)
        var hasher = SHA256()
        while true {
            // Keep FileHandle's autoreleased backing data bounded across large files.
            let consumed = try Self.withTemporaryObjects {
                try Task.checkCancellation()
                guard let chunk = try file.read(upToCount: 1_048_576), !chunk.isEmpty else { return false }
                hasher.update(data: chunk)
                return true
            }
            if !consumed { break }
        }
        let actual = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        guard actual == artifact.sha256 else { throw ArtifactError.checksumMismatch(artifact.filename) }
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
        configuration.timeoutIntervalForResource = 3_600
        #if !os(Linux)
        configuration.waitsForConnectivity = true
        configuration.allowsExpensiveNetworkAccess = false
        #endif
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        var request = URLRequest(url: source)
        request.timeoutInterval = 60
        let delegate = DownloadProgress(progress)
        let (temporary, response) = try await session.download(for: request, delegate: delegate)
        defer { try? FileManager.default.removeItem(at: temporary) }
        guard let response = response as? HTTPURLResponse, response.statusCode == 200 else {
            throw ArtifactError.unexpectedHTTPStatus((response as? HTTPURLResponse)?.statusCode ?? -1)
        }
        try Task.checkCancellation()
        try FileManager.default.moveItem(at: temporary, to: destination)
    }
}

private final class DownloadProgress: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let progress: @Sendable (Int64) -> Void
    init(_ progress: @escaping @Sendable (Int64) -> Void) { self.progress = progress }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {}
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        progress(totalBytesWritten)
    }
}
