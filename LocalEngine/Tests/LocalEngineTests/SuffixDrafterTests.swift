import AssistantKit
import Foundation
import LocalEngine
import XCTest

/// Deleting a conversation erases the drafting corpora (`SuffixDrafter.eraseAllCorpora`): what
/// the live drafters hold and every corpus file, while the drafters keep working. No GPU needed.
final class SuffixDrafterTests: XCTestCase {
    private var directory: URL!
    private let request = EngineRequest(system: "", turns: [ChatTurn(role: .user, text: "Hi")])
    /// A ledger, then the same ledger with a tool result (`<tool_response>` 8 … `</tool_response>` 9).
    private let ledger = [1, 20, 21, 22]
    private var withResult: [Int] { ledger + [8, 40, 41, 42, 43, 44, 9, 2] }

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("suffix-drafter-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func exists(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path)
    }

    /// A drafter saving at `url` that has been handed one tool result (its changes queued).
    private func drafter(at url: URL) -> SuffixDrafter {
        let drafter = SuffixDrafter(url: url, excluded: [], toolResponseTags: (start: 8, end: 9))
        drafter.reset(ledger: ledger, request: request)
        drafter.reset(ledger: withResult, request: request)
        return drafter
    }

    func testErasingEmptiesTheLiveCorporaAndDeletesEveryCorpusFile() throws {
        let url = directory.appendingPathComponent("corpus-model-a.bin")
        let live = drafter(at: url)
        live.waitForBackgroundWork()
        XCTAssertEqual(live.corpus?.tokenCount, 5)
        XCTAssertNotNil(live.propose(context: [40, 41, 42][...], maxTokens: 4))
        XCTAssertTrue(exists(url))

        // Another model's corpus from an earlier launch, and files that aren't corpora.
        let other = directory.appendingPathComponent("corpus-model-b.bin")
        var saved = SuffixCorpus(capacityTokens: 64)
        saved.add(document: [50, 51, 52, 53])
        try saved.encoded().write(to: other)
        let prefix = directory.appendingPathComponent("key.safetensors")
        let index = directory.appendingPathComponent("prefix-index.json")
        try Data("x".utf8).write(to: prefix)
        try Data("[]".utf8).write(to: index)

        SuffixDrafter.eraseAllCorpora(savedIn: directory)
        live.waitForBackgroundWork()
        XCTAssertEqual(live.corpus?.tokenCount, 0)
        XCTAssertNil(live.propose(context: [40, 41, 42][...], maxTokens: 4))
        XCTAssertFalse(exists(url))
        XCTAssertFalse(exists(other))
        XCTAssertTrue(exists(prefix))
        XCTAssertTrue(exists(index))

        // The drafter keeps working: later results are kept and saved again, and only they.
        live.reset(ledger: withResult + [8, 60, 61, 62, 9], request: request)
        live.waitForBackgroundWork()
        XCTAssertEqual(live.corpus?.documents, [[60, 61, 62]])
        let reloaded = try XCTUnwrap(SuffixCorpus(data: Data(contentsOf: url)))
        XCTAssertEqual(reloaded.documents, [[60, 61, 62]])
    }

    /// Changes queued before the erase (a reply that just ended) are erased too, file included.
    func testChangesQueuedBeforeTheEraseAreErased() throws {
        let url = directory.appendingPathComponent("corpus-model-a.bin")
        let live = drafter(at: url)
        SuffixDrafter.eraseAllCorpora(savedIn: directory)
        live.waitForBackgroundWork()
        XCTAssertEqual(live.corpus?.tokenCount, 0)
        XCTAssertFalse(exists(url))
    }

    /// A drafter that keeps its corpus in memory only is emptied as well.
    func testErasingEmptiesACorpusKeptInMemory() {
        let memory = SuffixDrafter(url: nil, excluded: [], toolResponseTags: (start: 8, end: 9))
        memory.reset(ledger: ledger, request: request)
        memory.reset(ledger: withResult, request: request)
        memory.waitForBackgroundWork()
        XCTAssertEqual(memory.corpus?.tokenCount, 5)
        SuffixDrafter.eraseAllCorpora(savedIn: directory)
        memory.waitForBackgroundWork()
        XCTAssertEqual(memory.corpus?.tokenCount, 0)
    }
}
