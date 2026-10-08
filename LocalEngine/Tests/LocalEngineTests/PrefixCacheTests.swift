import AssistantKit
import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLMCommon
import XCTest

/// The system prefix on disk: a saved prefix loads with exactly equal next logits (§6.4), a key
/// or token mismatch is refused, at most three files are kept, and an engine starts from it.
final class PrefixCacheTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        try MetalAvailability.require()
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("prefix-cache-\(UUID().uuidString)")
    }

    override func tearDownWithError() throws {
        if let directory {
            try? FileManager.default.removeItem(at: directory)
        }
    }

    private static func target(for model: any LanguageModel) -> any TargetModel {
        if let fork = model as? any HybridQwen35Forwarding {
            return HybridTarget(model: fork)
        }
        return StockTarget(model: model)
    }

    /// A session holding `prefix` with a `systemEnd` checkpoint, saved under `key`.
    private func savedSession(
        model: any LanguageModel, prefix: [Int], key: String, store: PrefixCacheStore, modelID: String = "tiny"
    ) throws -> LiveSession {
        let session = LiveSession(target: Self.target(for: model))
        session.feed(prefix, rows: .none)
        session.checkpoint(.systemEnd)
        // More tokens after the prefix: the file must hold the first `prefix.count` only.
        session.feed(TinyModels.tokens(5, seed: 99), rows: .none)
        eval(session.target.cache)
        let checkpoint = try XCTUnwrap(session.checkpoints.snapshot(for: .systemEnd))
        var kvState: [Int: [MLXArray]] = [:]
        for index in session.layout.attention {
            kvState[index] = session.target.cache[index].state
        }
        try store.save(
            snapshot: checkpoint, position: prefix.count, kvState: kvState, layout: session.layout, key: key, tokens: prefix,
            modelID: modelID)
        session.rewind(to: prefix.count)
        return session
    }

    func testSavedPrefixLoadsWithExactlyEqualNextLogits() throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let model = try EngineTestHarness.makeModel(tiny, seed: 51)
            let prefix = TinyModels.tokens(23, seed: 52)
            let store = PrefixCacheStore(directory: directory)
            let live = try savedSession(model: model, prefix: prefix, key: "key-\(tiny.rawValue)", store: store)
            XCTAssertTrue(store.contains(key: "key-\(tiny.rawValue)"))

            let loaded = LiveSession(target: Self.target(for: model))
            let cache = try XCTUnwrap(store.load(key: "key-\(tiny.rawValue)", expectedTokens: prefix, layout: loaded.layout))
            try loaded.adopt(cache, holding: prefix)

            let next = TinyModels.tokens(4, seed: 53)
            let expected = try XCTUnwrap(live.feed(next, rows: .all).logits)
            let actual = try XCTUnwrap(loaded.feed(next, rows: .all).logits)
            eval(expected, actual)
            XCTAssertEqual(LogitCheck.maxAbsDifference(actual, expected), 0, "\(tiny)")
            XCTAssertTrue(loaded.assertConsistent().isConsistent(), "\(tiny)")
        }
    }

    func testMismatchedKeyOrTokensAreRefused() throws {
        let model = try EngineTestHarness.makeModel(.hybrid, seed: 61)
        let prefix = TinyModels.tokens(12, seed: 62)
        let store = PrefixCacheStore(directory: directory)
        let session = try savedSession(model: model, prefix: prefix, key: "good", store: store)

        // Another key has no file.
        XCTAssertNil(store.load(key: "other", expectedTokens: prefix, layout: session.layout))
        // Other tokens (the token hash differs): refused and deleted.
        var other = prefix
        other[3] = other[3] == 40 ? 41 : 40
        XCTAssertNil(store.load(key: "good", expectedTokens: other, layout: session.layout))
        XCTAssertFalse(store.contains(key: "good"))

        // Another length: refused.
        _ = try savedSession(model: model, prefix: prefix, key: "good", store: store)
        XCTAssertNil(store.load(key: "good", expectedTokens: Array(prefix.dropLast()), layout: session.layout))

        // Another layout (a Qwen3 cache for a hybrid file): refused.
        _ = try savedSession(model: model, prefix: prefix, key: "good", store: store)
        let qwen3Layout = CacheLayout(attention: [0, 1, 2, 3], recurrent: [])
        XCTAssertNil(store.load(key: "good", expectedTokens: prefix, layout: qwen3Layout))

        // A file renamed to another key: its metadata names the old key.
        _ = try savedSession(model: model, prefix: prefix, key: "good", store: store)
        try FileManager.default.copyItem(at: store.url(for: "good"), to: store.url(for: "renamed"))
        XCTAssertNil(store.load(key: "renamed", expectedTokens: prefix, layout: session.layout))
        XCTAssertNotNil(store.load(key: "good", expectedTokens: prefix, layout: session.layout))
    }

    func testAtMostThreeFilesAndOtherModelsArePruned() throws {
        let model = try EngineTestHarness.makeModel(.qwen3, seed: 71)
        let store = PrefixCacheStore(directory: directory)
        for (index, key) in ["a", "b", "c", "d"].enumerated() {
            _ = try savedSession(model: model, prefix: TinyModels.tokens(6 + index, seed: UInt64(72 + index)), key: key, store: store)
            Thread.sleep(forTimeInterval: 0.01)
        }
        XCTAssertEqual(Set(store.keys), ["b", "c", "d"], "the least recently used file went first")
        XCTAssertFalse(FileManager.default.fileExists(atPath: store.url(for: "a").path))

        _ = try savedSession(model: model, prefix: TinyModels.tokens(6, seed: 80), key: "other-model", store: store, modelID: "other")
        store.pruneOtherModels(modelID: "other")
        XCTAssertEqual(store.keys, ["other-model"])
        let files = try FileManager.default.contentsOfDirectory(atPath: directory.path).filter { $0.hasSuffix(".safetensors") }
        XCTAssertEqual(files, ["other-model.safetensors"])
    }

    /// An engine saves its system prefix after the first reply; a second engine over the same
    /// weights starts from the file (`disk`) and answers exactly as the first did.
    func testEngineStartsFromTheDiskPrefix() async throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            var configuration = EngineTestHarness.testConfiguration(maxTokens: 12)
            configuration.prefixCacheDirectory = directory.appendingPathComponent(tiny.rawValue)
            let first = try EngineTestHarness.makeEngine(tiny: tiny, seed: 81, configuration: configuration)
            let request = EngineRequest(
                system: "You are Alvin.", tools: [ToolDefinition(name: "get_time", description: "The time.", inputSchema: ["type": "object"])],
                turns: [ChatTurn(role: .user, text: "What time is it?")])
            let firstEvents = try await EngineTestHarness.collect(first.reply(request))
            let firstFinish = try XCTUnwrap(EngineTestHarness.finish(firstEvents))
            XCTAssertEqual(firstFinish.stats.phases?.start, "cold", "\(tiny)")
            let summary = await first.sessionSummary()
            XCTAssertTrue(summary.contains("prefix on disk yes"), "\(tiny): \(summary)")

            let second = try EngineTestHarness.makeEngine(tiny: tiny, seed: 81, configuration: configuration)
            let secondEvents = try await EngineTestHarness.collect(second.reply(request))
            let secondFinish = try XCTUnwrap(EngineTestHarness.finish(secondEvents))
            XCTAssertEqual(secondFinish.stats.phases?.start, "disk", "\(tiny)")
            let systemEnd = try await second.withSession { $0.snapshot?.systemEnd ?? 0 }
            XCTAssertGreaterThan(systemEnd, 0)
            XCTAssertEqual(secondFinish.stats.reusedTokens, systemEnd, "\(tiny)")
            XCTAssertEqual(EngineTestHarness.text(secondEvents), EngineTestHarness.text(firstEvents), "\(tiny)")
            let report = try await second.withSession { $0.assertConsistent() }
            XCTAssertTrue(report.isConsistent(), "\(tiny): \(report)")

            // Prewarming a third engine loads the prefix without a reply.
            let third = try EngineTestHarness.makeEngine(tiny: tiny, seed: 81, configuration: configuration)
            try await third.prewarm(system: request.system, tools: request.tools)
            let warm = try await third.withSession { ($0.ledger.count, $0.snapshot?.systemEnd) }
            XCTAssertEqual(warm.0, systemEnd)
            XCTAssertEqual(warm.1, systemEnd)
        }
    }
}
