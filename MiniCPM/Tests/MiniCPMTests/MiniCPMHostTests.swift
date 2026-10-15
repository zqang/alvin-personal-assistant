import XCTest
@testable import MiniCPM

/// `MiniCPMHost` runs through the app's GPU lease, here a recording no-op as on macOS.
final class MiniCPMHostTests: XCTestCase {
    /// Replaces the lease with one that records each job's memory need, for one test.
    @MainActor
    private func recordLeases() -> () -> [Int] {
        var needs: [Int] = []
        let saved = MiniCPMHost.withGPU
        MiniCPMHost.withGPU = { need, body in
            needs.append(need)
            try await body()
        }
        addTeardownBlock { @MainActor in MiniCPMHost.withGPU = saved }
        return { needs }
    }

    private func read(_ stream: AsyncThrowingStream<String, Error>) async throws -> String {
        var text = ""
        for try await piece in stream { text += piece }
        return text
    }

    /// Without its files, a read fails before the lease, and not as a cancellation: the chat stores
    /// the "couldn't be read" note instead of offering to read again.
    @MainActor
    func testAReadWithoutTheModelFailsBeforeTheLease() async throws {
        let leases = recordLeases()
        let absent = FileManager.default.temporaryDirectory.appendingPathComponent("minicpm-host-\(UUID().uuidString)")
        let host = MiniCPMHost(photoReaderDirectory: absent, accurateDirectory: absent)
        XCTAssertEqual(host.status[.photoReader], .notDownloaded)
        do {
            _ = try await read(host.readPhoto(Data([0xFF, 0xD8])))
            XCTFail("A read without the model must fail")
        } catch is CancellationError {
            XCTFail("A missing model is a failure, not a stop")
        } catch {}
        XCTAssertEqual(leases(), [])
        XCTAssertFalse(FileManager.default.fileExists(atPath: absent.path), "No check or download runs by itself")
    }

    /// K4: reads with the real 1B. A load takes the lease with the load's memory need, a run on the
    /// loaded model with the margin only; ending a stream stops just that read; a host-side stop
    /// ends the read with `CancellationError`.
    @MainActor
    func testReal1BReadsThroughTheLease() async throws {
        guard let directory = ProcessInfo.processInfo.environment["MINICPM_BENCH_MODEL_DIR"] else {
            throw XCTSkip("Set MINICPM_BENCH_MODEL_DIR to read photos with the real 1B.")
        }
        let leases = recordLeases()
        let absent = FileManager.default.temporaryDirectory.appendingPathComponent("minicpm-host-\(UUID().uuidString)")
        let host = MiniCPMHost(photoReaderDirectory: URL(fileURLWithPath: directory), accurateDirectory: absent)
        // The launch check verifies the files (hashing them the first time only).
        while host.status[.photoReader] == .verifying { try await Task.sleep(for: .milliseconds(50)) }
        XCTAssertEqual(host.status[.photoReader], .ready)
        XCTAssertFalse(host.isLoaded(.photoReader))
        let fixtures = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        let jpeg = try Data(contentsOf: fixtures.appendingPathComponent("receipt.jpg"))

        let note = try await read(host.readPhoto(jpeg))
        print("HOST_NOTE:\n\(note)")
        XCTAssertTrue(note.contains("Text:"), "the reader prompt's layout")
        XCTAssertTrue(note.localizedCaseInsensitiveContains("TOTAL") && note.contains("8.75"))
        XCTAssertTrue(host.isLoaded(.photoReader))
        XCTAssertEqual(leases(), [MiniCPMHost.loadBytes(.photoReader)])

        // The caller ends a read after two pieces; the next read runs on the loaded model.
        var pieces = 0
        for try await _ in host.readPhoto(jpeg) {
            pieces += 1
            if pieces == 2 { break }
        }
        let again = try await read(host.readPhoto(jpeg))
        XCTAssertEqual(again, note, "greedy reads of one photo agree")
        XCTAssertEqual(leases(), [MiniCPMHost.loadBytes(.photoReader), MiniCPMHost.runBytes, MiniCPMHost.runBytes])

        // A host-side stop (here an unload, as a Woof load does) ends a read in progress with
        // CancellationError, so the photo can be read again.
        var reading = host.readPhoto(jpeg).makeAsyncIterator()
        _ = try await reading.next()
        await host.unloadAll()
        do {
            while try await reading.next() != nil {}
            XCTFail("A read stopped by the host must end with CancellationError")
        } catch is CancellationError {}
        XCTAssertFalse(host.isLoaded(.photoReader))
        XCTAssertEqual(host.status[.photoReader], .ready)
    }
}
