import CryptoKit
import Foundation

/// The pinned MiniCPM-V 4.7 1B model and vision projector that read photos, plus the file mechanics
/// `MiniCPM35BArtifacts` shares. Downloads are foreground and Wi-Fi only, keep their resume data per
/// file, and are installed only after their exact size, GGUF header and SHA-256 check out. A file is
/// hashed once: `.verified.json` then records its size and modification date, and later checks
/// compare only those.
public actor MiniCPMArtifacts {
    /// One pinned file: its path in the model folder, exact size and SHA-256, and where it downloads
    /// from. A file without a `sourceURL` is optional: it's used when present and verified, and never
    /// downloaded.
    public struct Artifact: Codable, Equatable, Sendable {
        public let path: String
        public let sizeBytes: Int64
        public let sha256: String
        public let role: String
        public let sourceURL: URL?
    }

    public enum ArtifactError: LocalizedError, Equatable {
        case alreadyPreparing
        case unsafePath(String)
        case invalidArtifact(String)
        case invalidExpertManifest
        case insufficientStorage(required: Int64)
        case unexpectedHTTPStatus(Int)

        public var errorDescription: String? {
            switch self {
            case .alreadyPreparing: return "The MiniCPM model files are already being prepared."
            case .unsafePath(let path): return "Unsafe MiniCPM file path: \(path)."
            case .invalidArtifact(let path): return "\(path) failed its pinned size, header or SHA-256 check."
            case .invalidExpertManifest: return "The MiniCPM 35B expert manifest doesn't match its pinned expert files."
            case .insufficientStorage(let required):
                return String(format: "Not enough storage: the download needs %.1f GB free.", Double(required) / 1e9)
            case .unexpectedHTTPStatus(let status): return "The MiniCPM download returned HTTP \(status)."
            }
        }
    }

    public static let revision = "8716e5209f308fd53f197ccd7d50c6fdcc2dc37d"
    public static let model = pinned("MiniCPM-V-4_7-1B-Q4_K_M.gguf", role: "model", 672_003_584,
                                     "bdc2c0da2320129a4fd91b0d464c63f8047b511a7ef7f7e1d143b433c7165cd5")
    public static let projector = pinned("mmproj-model-f16.gguf", role: "projector", 1_108_746_816,
                                         "febb329c29ec9f1643c9d690ca74d35f486b83fa48c2f54c108b135729f635fd")
    public static let artifacts = [model, projector]
    public static let totalBytes = model.sizeBytes + projector.sizeBytes
    // ponytail: both model folders keep their benchmark-era names, so the files already on the
    // owner's iPhone are reused without migration code. Renaming one needs a one-time move.
    public static let directory = URL.applicationSupportDirectory.appending(path: "MiniCPMBenchmark", directoryHint: .isDirectory)
    /// The free space a download leaves beyond the files it fetches.
    static let storageMargin: Int64 = 1_000_000_000

    typealias Downloader = @Sendable (_ source: URL, _ destination: URL, _ progress: @escaping @Sendable (Int64) -> Void) async throws -> Void
    typealias FreeBytes = @Sendable (URL) throws -> Int64
    private let directory: URL
    private let files: [Artifact]
    private let downloader: Downloader
    private let freeBytes: FreeBytes
    private var isPreparing = false

    public init(directory: URL = MiniCPMArtifacts.directory) {
        self.init(directory: directory, artifacts: Self.artifacts)
    }

    // Injectable files, transport and free space keep the tests small and offline.
    init(directory: URL, artifacts: [Artifact],
         downloader: @escaping Downloader = { try await MiniCPMArtifacts.download(from: $0, to: $1, progress: $2) },
         freeBytes: @escaping FreeBytes = { try MiniCPMArtifacts.availableStorage(in: $0) }) {
        self.directory = directory
        self.files = artifacts
        self.downloader = downloader
        self.freeBytes = freeBytes
    }

    /// Makes both files present and verified, downloading what's missing, and returns their paths.
    public func prepare(progress: @escaping @Sendable (Double) -> Void = { _ in }) async throws -> (model: URL, projector: URL) {
        guard !isPreparing else { throw ArtifactError.alreadyPreparing }
        isPreparing = true
        defer { isPreparing = false }
        let found = try await Self.fetch(files, into: directory, downloader: downloader, freeBytes: freeBytes, progress: progress)
        guard let model = found["model"], let projector = found["projector"] else { throw ArtifactError.invalidArtifact("model or projector") }
        return (model, projector)
    }

    private static func pinned(_ filename: String, role: String, _ size: Int64, _ sha256: String) -> Artifact {
        Artifact(path: filename, sizeBytes: size, sha256: sha256, role: role,
                 sourceURL: URL(string: "https://huggingface.co/tc-mb/MiniCPM-V-Apps-gguf/resolve/\(revision)/MiniCPM-V-4.7-1B/\(filename)")!)
    }
}

// MARK: - Shared by both models

extension MiniCPMArtifacts {
    /// Deletes a model folder, the `directory` of either model, with everything in it, and returns
    /// the bytes that freed. Cancel its download and unload its model first: a mapped file gives its
    /// space back only once it's unmapped.
    public static func delete(in directory: URL) -> Int64 {
        let before = bytesOnDisk(in: directory)
        try? FileManager.default.removeItem(at: directory)
        return before - bytesOnDisk(in: directory)
    }

    /// What a model folder takes on disk, partial downloads included.
    public static func bytesOnDisk(in directory: URL) -> Int64 {
        var total: Int64 = 0
        let files = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.totalFileAllocatedSizeKey])
        while let file = files?.nextObject() as? URL {
            total += Int64((try? file.resourceValues(forKeys: [.totalFileAllocatedSizeKey]).totalFileAllocatedSize) ?? 0)
        }
        return total
    }

    /// Makes each of `files` present and verified in `directory`, downloading the missing ones that
    /// have a source, and returns the verified files by role. An optional file (no source) that is
    /// missing or fails its check is left out, so it's never used.
    static func fetch(_ files: [Artifact], into directory: URL, downloader: Downloader, freeBytes: FreeBytes,
                      progress: @escaping @Sendable (Double) -> Void) async throws -> [String: URL] {
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        try excludeFromBackup(directory)
        let markerURL = directory.appending(path: ".verified.json")
        var marker = (try? JSONDecoder().decode([String: Stamp].self, from: Data(contentsOf: markerURL))) ?? [:]
        func record(_ file: Artifact, at url: URL) throws {
            try excludeFromBackup(url)
            marker[file.path] = Stamp(url, sha256: file.sha256)
            try JSONEncoder().encode(marker).write(to: markerURL, options: .atomic)
        }

        let total = max(1, files.reduce(Int64(0)) { $0 + ($1.sourceURL == nil ? 0 : $1.sizeBytes) })
        var completed: Int64 = 0
        var found: [String: URL] = [:]
        var missing: [(file: Artifact, url: URL, source: URL)] = []
        progress(0)
        for file in files {
            try Task.checkCancellation()
            let url = try safePath(file.path, in: directory)
            // Trust the marker only while the file keeps the size and date it had when it was hashed.
            if let stamp = Stamp(url, sha256: file.sha256), marker[file.path] == stamp {
                found[file.role] = url
            } else if try valid(url, artifact: file) {
                try record(file, at: url)
                found[file.role] = url
            } else if let source = file.sourceURL {
                missing.append((file, url, source))
                continue
            }
            if file.sourceURL != nil {
                completed += file.sizeBytes
                progress(Double(completed) / Double(total))
            }
        }
        let needed = missing.reduce(storageMargin) { $0 + $1.file.sizeBytes }
        if !missing.isEmpty, try freeBytes(directory) < needed { throw ArtifactError.insufficientStorage(required: needed) }

        for (file, url, source) in missing {
            try Task.checkCancellation()
            try manager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            // One fixed staging name per file, so its resume data (`<staged>.resume`) outlives a stop.
            let staged = try safePath(file.path + ".download", in: directory)
            defer { try? manager.removeItem(at: staged) }
            let previous = completed
            try await downloader(source, staged) { received in
                progress(min(0.99, Double(previous + min(file.sizeBytes, max(0, received))) / Double(total)))
            }
            guard try valid(staged, artifact: file) else { throw ArtifactError.invalidArtifact(file.path) }
            if manager.fileExists(atPath: url.path) {
                _ = try manager.replaceItemAt(url, withItemAt: staged)
            } else {
                try manager.moveItem(at: staged, to: url)
            }
            try? manager.removeItem(at: staged.appendingPathExtension("resume"))
            try record(file, at: url)
            found[file.role] = url
            completed += file.sizeBytes
            progress(Double(completed) / Double(total))
        }
        progress(1)
        return found
    }

    /// A verified file's size and modification date with the SHA-256 it matched, as `.verified.json`
    /// records them.
    private struct Stamp: Codable, Equatable {
        let size: Int64
        let modified: Date
        let sha256: String

        init?(_ url: URL, sha256: String) {
            guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
                  let size = (attributes[.size] as? NSNumber)?.int64Value,
                  let modified = attributes[.modificationDate] as? Date
            else { return nil }
            self.size = size
            self.modified = modified
            self.sha256 = sha256
        }
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

    /// Whether `path` has the artifact's exact size, GGUF header (for a .gguf) and SHA-256.
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
        var hasher = SHA256()
        while true {
            // FileHandle can autorelease its backing NSData. A synchronous hash of 15 GB otherwise
            // retains those objects until the job returns, despite reading a megabyte at a time.
            let consumed = try autoreleasepool {
                try Task.checkCancellation()
                guard let chunk = try file.read(upToCount: 1_048_576), !chunk.isEmpty else { return false }
                hasher.update(data: chunk)
                return true
            }
            if !consumed { break }
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined() == artifact.sha256
    }

    private static func excludeFromBackup(_ path: URL) throws {
        #if os(iOS)
        var path = path
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try path.setResourceValues(values)
        #endif
    }

    static func availableStorage(in directory: URL) throws -> Int64 {
        // The user asked for this download. Important-use capacity includes storage iOS can
        // reclaim; raw free bytes can be much smaller.
        let values = try? directory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey, .volumeAvailableCapacityKey])
        if let important = values?.volumeAvailableCapacityForImportantUsage, important >= 0 { return important }
        if let available = values?.volumeAvailableCapacity, available >= 0 { return Int64(available) }
        let attributes = try FileManager.default.attributesOfFileSystem(forPath: directory.path)
        return max(0, (attributes[.systemFreeSize] as? NSNumber)?.int64Value ?? 0)
    }

    /// Downloads `source` to `destination` in the foreground, over Wi-Fi only (without it, it fails
    /// at once with a `URLError` whose `networkUnavailableReason` says why), retrying a network
    /// failure, or a resume the server refuses, once. Whenever a transfer stops early, cancellation
    /// included, its resume data goes to `<destination>.resume` and the next attempt continues
    /// from there.
    // ponytail: a foreground session stops when iOS suspends the app; the host restarts it on
    // return. A background URLSession is the upgrade path. Resume data replays Hugging Face's
    // signed CDN URL, which expires after about an hour (HTTP 403): the retry then starts the file
    // over from `source`, losing the partial file; Range requests on `source` would keep it.
    static func download(from source: URL, to destination: URL, progress: @escaping @Sendable (Int64) -> Void) async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.allowsCellularAccess = false
        configuration.allowsExpensiveNetworkAccess = false
        configuration.allowsConstrainedNetworkAccess = false
        configuration.timeoutIntervalForResource = 21_600
        let resumeFile = destination.appendingPathExtension("resume")
        for attempt in 0..<2 {
            let transfer = Transfer(destination: destination, resumeFile: resumeFile, progress: progress)
            let session = URLSession(configuration: configuration, delegate: transfer, delegateQueue: nil)
            defer { session.finishTasksAndInvalidate() }
            let resumeData = try? Data(contentsOf: resumeFile)
            let task = resumeData.map(session.downloadTask(withResumeData:))
                ?? session.downloadTask(with: URLRequest(url: source, timeoutInterval: 90))
            do {
                return try await transfer.run(task)
            } catch {
                try Task.checkCancellation()
                // The failed transfer has removed its resume data, so the retry starts from `source`.
                let refusedResume = resumeData != nil && error is ArtifactError
                if attempt == 1 || !(error is URLError || refusedResume) { throw error }
            }
        }
    }
}

/// One file's transfer: reports bytes, moves the finished file to `destination`, and writes resume
/// data to `resumeFile` whenever the transfer stops early.
private final class Transfer: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let destination: URL
    private let resumeFile: URL
    private let progress: @Sendable (Int64) -> Void
    private let outcome: AsyncThrowingStream<Void, Error>
    private let finish: AsyncThrowingStream<Void, Error>.Continuation
    private var failure: Error? // Only touched on the session's serial delegate queue.

    init(destination: URL, resumeFile: URL, progress: @escaping @Sendable (Int64) -> Void) {
        self.destination = destination
        self.resumeFile = resumeFile
        self.progress = progress
        let (outcome, finish) = AsyncThrowingStream<Void, Error>.makeStream()
        self.outcome = outcome
        self.finish = finish
    }

    func run(_ task: URLSessionDownloadTask) async throws {
        finish.onTermination = { [resumeFile] termination in
            // The caller was cancelled: stop the transfer but keep what it has.
            guard case .cancelled = termination else { return }
            task.cancel { data in try? data?.write(to: resumeFile, options: .atomic) }
        }
        task.resume()
        for try await _ in outcome {}
        try Task.checkCancellation()
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        progress(totalBytesWritten)
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        let status = (downloadTask.response as? HTTPURLResponse)?.statusCode ?? -1
        do {
            guard status == 200 || status == 206 else { throw MiniCPMArtifacts.ArtifactError.unexpectedHTTPStatus(status) }
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: location, to: destination)
        } catch { failure = error }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let data = (error as? URLError)?.downloadTaskResumeData {
            try? data.write(to: resumeFile, options: .atomic)
        } else if (error as? URLError)?.code != .cancelled {
            // Finished or failed for good: the old resume data is spent.
            try? FileManager.default.removeItem(at: resumeFile)
        }
        if let error = error ?? failure { finish.finish(throwing: error) } else { finish.finish() }
    }
}
