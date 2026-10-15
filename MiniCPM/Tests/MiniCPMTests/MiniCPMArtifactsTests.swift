import Foundation
import XCTest
@testable import AssistantKit
#if canImport(CryptoKit)
import CryptoKit

final class MiniCPMArtifactsTests: XCTestCase {
    private let modelBytes = Data("GGUFtest-model".utf8)
    private let projectorBytes = Data("GGUFtest-projector".utf8)

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    private func manifest(_ name: String, bytes: Data) -> MiniCPMArtifacts.Manifest {
        .init(filename: name, sizeBytes: Int64(bytes.count),
              sha256: SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined(),
              sourceURL: URL(string: "https://example.invalid/\(name)")!)
    }

    private var manifests: [MiniCPMArtifacts.Manifest] {
        [manifest("model.gguf", bytes: modelBytes), manifest("projector.gguf", bytes: projectorBytes)]
    }

    func testPinnedPairHasExactRevisionSizesAndChecksums() {
        XCTAssertEqual(MiniCPMArtifacts.totalBytes, 1_780_750_400)
        for artifact in MiniCPMArtifacts.manifests {
            XCTAssertEqual(artifact.sha256.count, 64)
            XCTAssertTrue(artifact.sourceURL.path.contains("/resolve/\(MiniCPMArtifacts.revision)/"))
            XCTAssertEqual(artifact.sourceURL.lastPathComponent, artifact.filename)
        }
    }

    func testValidCacheAvoidsDownloadAndSameSizeCorruptionIsReplaced() async throws {
        let directory = try temporaryDirectory()
        let downloads = FixtureDownloads(files: ["model.gguf": modelBytes, "projector.gguf": projectorBytes])
        let artifacts = MiniCPMArtifacts(directory: directory, artifacts: manifests) { source, target, progress in
            try await downloads.download(source, target, progress)
        }
        let first = try await artifacts.prepare()
        _ = try await artifacts.prepare()
        let initialCount = await downloads.count
        XCTAssertEqual(initialCount, 2)
        XCTAssertEqual(try Data(contentsOf: first.model), modelBytes)
        // Keep the correct header and length, so only full checksum validation catches this.
        try Data("GGUFbad!-model".utf8).write(to: first.model)
        XCTAssertEqual(try Data(contentsOf: first.model).count, modelBytes.count)
        let replacement = try await artifacts.prepare()
        let finalCount = await downloads.count
        XCTAssertEqual(finalCount, 3)
        XCTAssertEqual(try Data(contentsOf: replacement.model), modelBytes)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted(), ["model.gguf", "projector.gguf"])
    }

    func testPartialCacheIsRedownloaded() async throws {
        let directory = try temporaryDirectory()
        try Data("GGUF".utf8).write(to: directory.appendingPathComponent("model.gguf"))
        let downloads = FixtureDownloads(files: ["model.gguf": modelBytes, "projector.gguf": projectorBytes])
        let artifacts = MiniCPMArtifacts(directory: directory, artifacts: manifests) { source, target, progress in
            try await downloads.download(source, target, progress)
        }
        let paths = try await artifacts.prepare()
        XCTAssertEqual(try Data(contentsOf: paths.model), modelBytes)
        let count = await downloads.count
        XCTAssertEqual(count, 2)
    }

    func testBadDownloadIsNeverInstalledAndStagingIsRemoved() async throws {
        let directory = try temporaryDirectory()
        let artifacts = MiniCPMArtifacts(directory: directory, artifacts: manifests) { _, target, _ in
            try Data("GGUFbad!-model".utf8).write(to: target)
        }
        do {
            _ = try await artifacts.prepare()
            XCTFail("Corrupted GGUF must not be installed")
        } catch let error as MiniCPMArtifacts.ArtifactError {
            XCTAssertEqual(error, .checksumMismatch("model.gguf"))
        }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), [])
    }

    func testCancellationCleansPartialFileAndAllowsRetry() async throws {
        let directory = try temporaryDirectory()
        let (started, signal) = AsyncStream<Void>.makeStream()
        let artifacts = MiniCPMArtifacts(directory: directory, artifacts: manifests) { _, target, _ in
            try Data("GGUF".utf8).write(to: target)
            signal.yield(())
            try await Task.sleep(for: .seconds(60))
        }
        let task = Task { try await artifacts.prepare() }
        var iterator = started.makeAsyncIterator()
        _ = await iterator.next()
        do {
            _ = try await artifacts.prepare()
            XCTFail("A second concurrent preparation must be rejected")
        } catch let error as MiniCPMArtifacts.ArtifactError {
            XCTAssertEqual(error, .alreadyPreparing)
        }
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("Expected cancellation")
        } catch is CancellationError { }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), [])
        let retry = Task { try await artifacts.prepare() }
        _ = await iterator.next()
        retry.cancel()
        do { _ = try await retry.value } catch is CancellationError { }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), [])
        signal.finish()
    }
}

private actor FixtureDownloads {
    let files: [String: Data]
    var count = 0
    init(files: [String: Data]) { self.files = files }
    func download(_ source: URL, _ target: URL, _ progress: @escaping @Sendable (Int64) -> Void) async throws {
        try Task.checkCancellation()
        count += 1
        let bytes = try XCTUnwrap(files[source.lastPathComponent])
        try bytes.write(to: target)
        progress(Int64(bytes.count))
    }
}
#endif
