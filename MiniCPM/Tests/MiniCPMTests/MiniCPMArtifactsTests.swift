import CryptoKit
import Foundation
import XCTest
@testable import MiniCPM

final class MiniCPMArtifactsTests: XCTestCase {
    private let modelBytes = Data("GGUFtest-model".utf8)
    private let projectorBytes = Data("GGUFtest-projector".utf8)

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    private func artifact(_ name: String, role: String, bytes: Data) -> MiniCPMArtifacts.Artifact {
        .init(path: name, sizeBytes: Int64(bytes.count),
              sha256: SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined(),
              role: role, sourceURL: URL(string: "https://example.invalid/\(name)")!)
    }

    private var pair: [MiniCPMArtifacts.Artifact] {
        [artifact("model.gguf", role: "model", bytes: modelBytes), artifact("projector.gguf", role: "projector", bytes: projectorBytes)]
    }

    private func makeLoader(_ directory: URL, _ downloads: FixtureDownloads,
                        free: @escaping MiniCPMArtifacts.FreeBytes = { _ in .max }) -> MiniCPMArtifacts {
        MiniCPMArtifacts(directory: directory, artifacts: pair, downloader: { source, target, progress in
            try await downloads.download(source, target, progress)
        }, freeBytes: free)
    }

    private func contents(_ directory: URL) throws -> [String] {
        try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
    }

    func testPinnedPairHasExactRevisionSizesAndChecksums() {
        XCTAssertEqual(MiniCPMArtifacts.totalBytes, 1_780_750_400)
        XCTAssertEqual(MiniCPMArtifacts.directory.lastPathComponent, "MiniCPMBenchmark")
        for artifact in MiniCPMArtifacts.artifacts {
            XCTAssertEqual(artifact.sha256.count, 64)
            XCTAssertEqual(artifact.sourceURL?.path.contains("/resolve/\(MiniCPMArtifacts.revision)/"), true)
            XCTAssertEqual(artifact.sourceURL?.lastPathComponent, artifact.path)
        }
    }

    func testValidCacheAvoidsDownloadAndSameSizeCorruptionIsReplaced() async throws {
        let directory = try temporaryDirectory()
        let downloads = FixtureDownloads(files: ["model.gguf": modelBytes, "projector.gguf": projectorBytes])
        let artifacts = makeLoader(directory, downloads)
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
        XCTAssertEqual(try contents(directory), [".verified.json", "model.gguf", "projector.gguf"])
    }

    func testMarkerSkipsHashingUntilSizeOrDateChanges() async throws {
        let directory = try temporaryDirectory()
        let downloads = FixtureDownloads(files: ["model.gguf": modelBytes, "projector.gguf": projectorBytes])
        let artifacts = makeLoader(directory, downloads)
        let model = try await artifacts.prepare().model
        // A new date makes a good file hash again (setAttributes keeps whole seconds exactly).
        let date = Date(timeIntervalSinceReferenceDate: 800_000_000)
        try FileManager.default.setAttributes([.modificationDate: date], ofItemAtPath: model.path)
        _ = try await artifacts.prepare()
        // Same size and date: the marker vouches for the file, so the corruption goes unseen.
        try Data("GGUFbad!-model".utf8).write(to: model)
        try FileManager.default.setAttributes([.modificationDate: date], ofItemAtPath: model.path)
        _ = try await artifacts.prepare()
        var count = await downloads.count
        XCTAssertEqual(count, 2)
        XCTAssertEqual(try Data(contentsOf: model), Data("GGUFbad!-model".utf8))
        // A new date makes it hash again, which catches the corruption.
        try FileManager.default.setAttributes([.modificationDate: date.addingTimeInterval(1)], ofItemAtPath: model.path)
        _ = try await artifacts.prepare()
        count = await downloads.count
        XCTAssertEqual(count, 3)
        XCTAssertEqual(try Data(contentsOf: model), modelBytes)
    }

    func testPartialCacheIsRedownloaded() async throws {
        let directory = try temporaryDirectory()
        try Data("GGUF".utf8).write(to: directory.appendingPathComponent("model.gguf"))
        let downloads = FixtureDownloads(files: ["model.gguf": modelBytes, "projector.gguf": projectorBytes])
        let paths = try await makeLoader(directory, downloads).prepare()
        XCTAssertEqual(try Data(contentsOf: paths.model), modelBytes)
        let count = await downloads.count
        XCTAssertEqual(count, 2)
    }

    func testStorageNeedsTheMissingBytesPlusOneGigabyte() async throws {
        let directory = try temporaryDirectory()
        // The projector is already here, so only the model is missing.
        try projectorBytes.write(to: directory.appendingPathComponent("projector.gguf"))
        let needed = Int64(modelBytes.count) + 1_000_000_000
        let downloads = FixtureDownloads(files: ["model.gguf": modelBytes])
        do {
            _ = try await makeLoader(directory, downloads, free: { _ in needed - 1 }).prepare()
            XCTFail("A download must leave 1 GB free")
        } catch let error as MiniCPMArtifacts.ArtifactError {
            XCTAssertEqual(error, .insufficientStorage(required: needed))
        }
        var count = await downloads.count
        XCTAssertEqual(count, 0)
        _ = try await makeLoader(directory, downloads, free: { _ in needed }).prepare()
        count = await downloads.count
        XCTAssertEqual(count, 1)
    }

    func testResumeDataSurvivesAStopAndIsDroppedOnceInstalled() async throws {
        let directory = try temporaryDirectory()
        try projectorBytes.write(to: directory.appendingPathComponent("projector.gguf"))
        let destinations = Destinations()
        let modelBytes = modelBytes
        let resuming = MiniCPMArtifacts(directory: directory, artifacts: pair, downloader: { _, target, _ in
            let resume = target.appendingPathExtension("resume")
            if await destinations.add(target) == 1 {
                try Data("resume".utf8).write(to: resume)
                throw URLError(.networkConnectionLost)
            }
            XCTAssertEqual(try Data(contentsOf: resume), Data("resume".utf8))
            try modelBytes.write(to: target)
        }, freeBytes: { _ in .max })
        do { _ = try await resuming.prepare(); XCTFail("The first transfer stops") } catch is URLError { }
        XCTAssertEqual(try contents(directory), [".verified.json", "model.gguf.download.resume", "projector.gguf"])
        let model = try await resuming.prepare().model
        XCTAssertEqual(try Data(contentsOf: model), modelBytes)
        let seen = await destinations.all
        XCTAssertEqual(seen.count, 2)
        XCTAssertEqual(Set(seen).count, 1, "Each file keeps one staging name, so its resume data is found again")
        XCTAssertEqual(try contents(directory), [".verified.json", "model.gguf", "projector.gguf"])
    }

    func testDeleteRemovesTheFolderAndReportsTheBytesFreed() async throws {
        let directory = try temporaryDirectory()
        let downloads = FixtureDownloads(files: ["model.gguf": modelBytes, "projector.gguf": projectorBytes])
        _ = try await makeLoader(directory, downloads).prepare()
        let used = MiniCPMArtifacts.bytesOnDisk(in: directory)
        XCTAssertGreaterThanOrEqual(used, Int64(modelBytes.count + projectorBytes.count))
        XCTAssertEqual(MiniCPMArtifacts.delete(in: directory), used)
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
        XCTAssertEqual(MiniCPMArtifacts.delete(in: directory), 0)
    }

    func testBadDownloadIsNeverInstalledAndStagingIsRemoved() async throws {
        let directory = try temporaryDirectory()
        let artifacts = MiniCPMArtifacts(directory: directory, artifacts: pair, downloader: { _, target, _ in
            try Data("GGUFbad!-model".utf8).write(to: target)
        }, freeBytes: { _ in .max })
        do {
            _ = try await artifacts.prepare()
            XCTFail("Corrupted GGUF must not be installed")
        } catch let error as MiniCPMArtifacts.ArtifactError {
            XCTAssertEqual(error, .invalidArtifact("model.gguf"))
        }
        XCTAssertEqual(try contents(directory), [])
    }

    func testCancellationCleansPartialFileAndAllowsRetry() async throws {
        let directory = try temporaryDirectory()
        let (started, signal) = AsyncStream<Void>.makeStream()
        let artifacts = MiniCPMArtifacts(directory: directory, artifacts: pair, downloader: { _, target, _ in
            try Data("GGUF".utf8).write(to: target)
            signal.yield(())
            try await Task.sleep(for: .seconds(60))
        }, freeBytes: { _ in .max })
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
        XCTAssertEqual(try contents(directory), [])
        let retry = Task { try await artifacts.prepare() }
        _ = await iterator.next()
        retry.cancel()
        do { _ = try await retry.value } catch is CancellationError { }
        XCTAssertEqual(try contents(directory), [])
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

private actor Destinations {
    var all: [URL] = []
    func add(_ url: URL) -> Int {
        all.append(url)
        return all.count
    }
}
