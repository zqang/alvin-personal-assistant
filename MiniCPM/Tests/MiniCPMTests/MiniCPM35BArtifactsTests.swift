import CryptoKit
import Darwin
import Foundation
import XCTest
@testable import MiniCPM

final class MiniCPM35BArtifactsTests: XCTestCase {
    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    private func artifact(_ path: String, role: String, bytes: Data, local: Bool = false) -> MiniCPM35BArtifacts.Artifact {
        .init(path: path, sizeBytes: Int64(bytes.count),
              sha256: SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined(),
              role: role, sourceURL: local ? nil : URL(string: "https://example.invalid/" + path)!)
    }

    private func fixture() throws -> ([MiniCPM35BArtifacts.Artifact], [String: Data]) {
        var files: [String: Data] = ["model.gguf": Data("GGUFmodel".utf8), "draft.gguf": Data("GGUFdraft".utf8)]
        var entries: [[String: Any]] = []
        var artifacts = [artifact("model.gguf", role: "model", bytes: files["model.gguf"]!),
                         artifact("draft.gguf", role: "draft", bytes: files["draft.gguf"]!, local: true)]
        for layer in 0..<40 {
            let name = String(format: "layer_%03d.bin", layer)
            let path = "experts/" + name
            let bytes = Data(repeating: UInt8(layer), count: 768)
            files[path] = bytes
            artifacts.append(artifact(path, role: "expert", bytes: bytes))
            for (offset, family) in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"].enumerated() {
                entries.append(["layer": layer, "tensor_family": family, "repacked_file": name,
                                "repacked_offset": offset, "expert_stride": 3, "bytes_per_expert": 1])
            }
        }
        let metadata: [String: Any] = ["schema_version": 1, "layout": "layer_major_expert",
                                      "model": ["arch": "qwen35moe", "expert_count": 256, "expert_used_count": 8], "entries": entries]
        let manifest = try JSONSerialization.data(withJSONObject: metadata, options: .sortedKeys)
        files["experts/manifest.json"] = manifest
        artifacts.append(artifact("experts/manifest.json", role: "expertManifest", bytes: manifest))
        return (artifacts, files)
    }

    private func makeLoader(_ directory: URL, _ entries: [MiniCPM35BArtifacts.Artifact], _ downloads: MiniCPM35BFixtureDownloads) -> MiniCPM35BArtifacts {
        MiniCPM35BArtifacts(directory: directory, artifacts: entries, downloader: { source, destination, progress in
            try await downloads.download(source, destination, progress)
        }, freeBytes: { _ in .max })
    }

    func testPinnedSetHasFortyRawShardsNoProjectorAndAnUnhostedDraft() {
        let files = MiniCPM35BArtifacts.artifacts
        XCTAssertEqual(files.filter { $0.role == "expert" }.count, 40)
        XCTAssertFalse(files.contains { $0.role == "projector" })
        let draft = files.first { $0.role == "draft" }
        XCTAssertEqual(draft?.sha256, MiniCPM35BArtifacts.draftProvenance.sha256)
        XCTAssertEqual(draft?.sizeBytes, MiniCPM35BArtifacts.draftProvenance.sizeBytes)
        XCTAssertNil(draft?.sourceURL)
        XCTAssertEqual(files.first { $0.role == "model" }?.sha256, MiniCPM35BArtifacts.draftProvenance.pairedTargetSHA256)
        // The download: the model, 40 shards and their manifest, without the unhosted draft.
        XCTAssertEqual(MiniCPM35BArtifacts.totalBytes, 15_504_009_668)
        XCTAssertEqual(MiniCPM35BArtifacts.directory.lastPathComponent, "MiniCPM35BBenchmark")
        for file in files where file.role != "draft" {
            XCTAssertEqual(file.sha256.count, 64)
            XCTAssertTrue(file.sourceURL?.path.contains("/resolve/\(MiniCPM35BArtifacts.revision)/") == true)
        }
    }

    func testEligibilityNeedsElevenBillionBytesOfMemory() {
        XCTAssertTrue(MiniCPM35BArtifacts.eligible(physicalMemory: 12_262_113_280)) // iPhone 17 Pro Max
        XCTAssertTrue(MiniCPM35BArtifacts.eligible(physicalMemory: 11_000_000_000))
        XCTAssertFalse(MiniCPM35BArtifacts.eligible(physicalMemory: 10_999_999_999))
        XCTAssertFalse(MiniCPM35BArtifacts.eligible(physicalMemory: 8_000_000_000))
    }

    func testHashingDoesNotRetainAWholeFileOfAutoreleasedChunks() throws {
        let directory = try temporaryDirectory()
        let file = directory.appendingPathComponent("large.bin")
        FileManager.default.createFile(atPath: file.path, contents: nil)
        let writer = try FileHandle(forWritingTo: file)
        let size: UInt64 = 256 * 1_048_576
        try writer.truncate(atOffset: size) // A sparse zero-filled fixture, not 256 MB of test data in RAM.
        try writer.close()
        let chunk = Data(repeating: 0, count: 1_048_576)
        var hasher = SHA256()
        for _ in 0..<256 { hasher.update(data: chunk) }
        let artifact = MiniCPM35BArtifacts.Artifact(path: "large.bin", sizeBytes: Int64(size),
            sha256: hasher.finalize().map { String(format: "%02x", $0) }.joined(), role: "expert", sourceURL: nil)
        let growth: UInt64 = try autoreleasepool {
            let before = try processFootprint()
            XCTAssertTrue(try MiniCPMArtifacts.valid(file, artifact: artifact))
            let after = try processFootprint()
            return after > before ? after - before : 0
        }
        // The previous loop retained ~256 MB until this outer pool drained.
        XCTAssertLessThan(growth, 32 * 1_048_576, "Hashing must retain only bounded temporary data")
    }

    private func processFootprint() throws -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        XCTAssertEqual(status, KERN_SUCCESS)
        return info.phys_footprint
    }

    func testUnsafePathsAndSymlinkEscapeAreRejected() throws {
        let directory = try temporaryDirectory()
        for name in ["/tmp/file", "../file", "experts/../../file", "experts//file", "experts/./file", "experts\\file", ""] {
            XCTAssertThrowsError(try MiniCPMArtifacts.safePath(name, in: directory))
        }
        try FileManager.default.createSymbolicLink(at: directory.appendingPathComponent("escape"), withDestinationURL: directory.deletingLastPathComponent())
        XCTAssertThrowsError(try MiniCPMArtifacts.safePath("escape/file", in: directory))
    }

    func testCacheIsVerifiedAndCorruptShardIsReplaced() async throws {
        let directory = try temporaryDirectory()
        let (entries, files) = try fixture()
        let downloads = MiniCPM35BFixtureDownloads(files: files)
        let loader = makeLoader(directory, entries, downloads)
        let paths = try await loader.prepare()
        _ = try await loader.prepare()
        let initialCount = await downloads.count
        XCTAssertEqual(initialCount, 42)
        XCTAssertEqual(paths.experts.lastPathComponent, "manifest.json")
        let shard = directory.appendingPathComponent("experts/layer_007.bin")
        try Data(repeating: 99, count: 768).write(to: shard)
        _ = try await loader.prepare()
        let finalCount = await downloads.count
        XCTAssertEqual(finalCount, 43)
        XCTAssertEqual(try Data(contentsOf: shard), files["experts/layer_007.bin"])
    }

    func testDraftIsOptionalAndAWrongOneIsNeverUsed() async throws {
        let directory = try temporaryDirectory()
        let (entries, files) = try fixture()
        let downloads = MiniCPM35BFixtureDownloads(files: files)
        let loader = makeLoader(directory, entries, downloads)
        let draft = directory.appendingPathComponent("draft.gguf")
        // Absent: the target downloads and runs without it; the draft is never fetched.
        var paths = try await loader.prepare()
        XCTAssertNil(paths.draft)
        XCTAssertEqual(paths.model.lastPathComponent, "model.gguf")
        let count = await downloads.count
        XCTAssertEqual(count, 42)
        // Present but wrong: left alone and not used.
        try Data("GGUFdrafT".utf8).write(to: draft) // Right size, wrong hash.
        paths = try await loader.prepare()
        XCTAssertNil(paths.draft)
        XCTAssertTrue(FileManager.default.fileExists(atPath: draft.path))
        // Present and right: used.
        try files["draft.gguf"]!.write(to: draft)
        paths = try await loader.prepare()
        XCTAssertEqual(paths.draft?.lastPathComponent, "draft.gguf")
        let finalCount = await downloads.count
        XCTAssertEqual(finalCount, 42)
    }

    func testAnOldProjectorIsDeleted() async throws {
        let directory = try temporaryDirectory()
        let (entries, files) = try fixture()
        let projector = directory.appendingPathComponent("mmproj-model-f16.gguf")
        try Data("GGUFprojector".utf8).write(to: projector)
        _ = try await makeLoader(directory, entries, MiniCPM35BFixtureDownloads(files: files)).prepare()
        XCTAssertFalse(FileManager.default.fileExists(atPath: projector.path))
    }

    func testExpertManifestRejectsTraversalAndOutOfBoundsAndDuplicateTensors() throws {
        let directory = try temporaryDirectory()
        let (artifacts, files) = try fixture()
        let original = try XCTUnwrap(try JSONSerialization.jsonObject(with: files["experts/manifest.json"]!) as? [String: Any])
        let path = directory.appendingPathComponent("manifest.json")
        for patch: [String: Any] in [["repacked_file": "../layer_000.bin"], ["bytes_per_expert": 800], ["expert_stride": Int64.max], ["tensor_family": "ffn_up_exps"]] {
            var metadata = original
            var entries = metadata["entries"] as! [[String: Any]]
            entries[0].merge(patch) { _, new in new }
            metadata["entries"] = entries
            try JSONSerialization.data(withJSONObject: metadata).write(to: path)
            XCTAssertThrowsError(try MiniCPM35BArtifacts.validateExpertManifest(path, artifacts: artifacts))
        }
    }

    func testCancellationRejectsConcurrentPreparationAndCleansStaging() async throws {
        let directory = try temporaryDirectory()
        let (entries, files) = try fixture()
        try files["draft.gguf"]!.write(to: directory.appendingPathComponent("draft.gguf"))
        let (started, signal) = AsyncStream<Void>.makeStream()
        let loader = MiniCPM35BArtifacts(directory: directory, artifacts: entries, downloader: { _, target, _ in
            try Data("GGUF".utf8).write(to: target)
            signal.yield(())
            try await Task.sleep(for: .seconds(60))
        }, freeBytes: { _ in .max })
        let task = Task { try await loader.prepare() }
        var iterator = started.makeAsyncIterator()
        _ = await iterator.next()
        do {
            _ = try await loader.prepare()
            XCTFail("Concurrent preparation must not share staging files")
        } catch let error as MiniCPMArtifacts.ArtifactError {
            XCTAssertEqual(error, .alreadyPreparing)
        }
        task.cancel()
        do { _ = try await task.value; XCTFail("Expected cancellation") } catch is CancellationError { }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted(), [".verified.json", "draft.gguf"])
        signal.finish()
    }
}

private actor MiniCPM35BFixtureDownloads {
    let files: [String: Data]
    var count = 0
    init(files: [String: Data]) { self.files = files }
    func download(_ source: URL, _ target: URL, _ progress: @escaping @Sendable (Int64) -> Void) throws {
        count += 1
        let bytes = try XCTUnwrap(files[String(source.path.dropFirst())])
        try bytes.write(to: target)
        progress(Int64(bytes.count))
    }
}
