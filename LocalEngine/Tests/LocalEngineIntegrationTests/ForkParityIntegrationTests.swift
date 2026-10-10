import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

/// The `HybridQwen35` fork on real checkpoints: loaded through `EngineModelRegistry`, it must
/// decode exactly like the stock model code (first-step logits bitwise equal, 64 greedy tokens
/// identical on 4 prompts), and the engine path (`HybridTarget`, `.last` rows) must agree with it
/// up to near-ties (plan §6.4).
///
/// One exception is expected: a checkpoint that kept `mtp.*` tensors with an already converted
/// conv1d loads as garbage in stock 3.31.4 (F7) and correctly in the fork. The test detects that
/// layout from the safetensors headers and then requires the two to differ.
final class ForkParityIntegrationTests: XCTestCase {
    private static let chatContext: [String: any Sendable] = ["enable_thinking": false]
    private static let generatedCount = 64

    private static let conversations: [[[String: any Sendable]]] = [
        [["role": "user", "content": "What is the capital of France? Answer in one sentence."]],
        [["role": "user", "content": "Write a short poem about the sea."]],
        [
            ["role": "system", "content": "You are Alvin, a concise voice assistant. Answer in plain words."],
            ["role": "user", "content": "Remind me how to boil an egg."],
        ],
        [["role": "user", "content": "Repeat this sentence exactly: the quick brown fox jumps over the lazy dog."]],
    ]

    override func setUpWithError() throws {
        try IntegrationEnvironment.requireEnabled()
        try MetalAvailability.require()
    }

    func testForkMatchesStockOnQwen35_0_8B() async throws {
        try await assertForkParity("mlx-community/Qwen3.5-0.8B-MLX-4bit")
    }

    func testForkMatchesStockOnQwen35_2B() async throws {
        try await assertForkParity("mlx-community/Qwen3.5-2B-4bit")
    }

    func testForkMatchesStockOnWoof() async throws {
        guard IntegrationEnvironment.woof else {
            throw XCTSkip("Woof runs only with LOCAL_ENGINE_WOOF=1 ([ci woof]).")
        }
        try await assertForkParity(IntegrationEnvironment.woofRepo)
    }

    // MARK: The check

    private struct Run {
        let tokens: [Int]
        /// Float32 logits `[V]` predicting the first generated token.
        let first: MLXArray
        let seconds: Double
    }

    private func assertForkParity(_ repo: String) async throws {
        let directory = try await IntegrationEnvironment.snapshot(repo)
        let facts = try Self.checkpointFacts(directory)
        // F7: stock 3.31.4 shifts the norms of a converted checkpoint that kept `mtp.*` tensors.
        let stockIsGarbage = facts.mtpTensors > 0 && !facts.unsanitizedConv1d

        // The stock model first; it is released before the fork loads.
        var prompts: [[Int]] = []
        var stockRuns: [Run] = []
        var stockStops: Set<Int> = []
        var stockModelType = ""
        do {
            let stock = try await ModelLoader.load(directory: directory, id: repo)
            XCTAssertFalse(EngineModelRegistry.isFork(stock.model), "\(repo): the stock load built the fork")
            XCTAssertNil(EngineModelRegistry.makeTarget(for: stock))
            stockStops = stock.stopTokenIDs
            stockModelType = stock.modelType
            for messages in Self.conversations {
                let prompt = try stock.renderer.renderTokens(
                    messages: messages, tools: nil, context: Self.chatContext, addGenerationPrompt: true)
                prompts.append(prompt)
                stockRuns.append(Self.greedy(stock.model, prompt: prompt, count: Self.generatedCount))
            }
        }

        let fork = try await ModelLoader.load(directory: directory, id: repo, typeRegistry: EngineModelRegistry.makeTypeRegistry())
        XCTAssertTrue(EngineModelRegistry.isFork(fork.model), "\(repo): \(type(of: fork.model)) is not the fork")
        XCTAssertEqual(fork.modelType, stockModelType)
        XCTAssertEqual(fork.stopTokenIDs, stockStops)
        let target = try XCTUnwrap(EngineModelRegistry.makeTarget(for: fork))
        XCTAssertTrue(target is HybridTarget)

        var tally = NearTieTally(tolerance: NearTieTally.quantizedTolerance)
        var rows: [[String]] = []
        for (index, prompt) in prompts.enumerated() {
            let label = "\(repo) prompt \(index + 1)"
            let stockRun = stockRuns[index]
            let forkRun = Self.greedy(fork.model, prompt: prompt, count: Self.generatedCount)
            let difference = LogitCheck.maxAbsDifference(forkRun.first, stockRun.first)
            let sameTokens = forkRun.tokens == stockRun.tokens

            if stockIsGarbage {
                XCTAssertGreaterThan(difference, 0, "\(label): stock should differ (F7: mtp.* kept, conv1d converted)")
            } else {
                XCTAssertEqual(difference, 0, "\(label): first-step logits differ, max |Δ| = \(difference)")
                XCTAssertTrue(sameTokens, "\(label): greedy tokens differ\nstock \(stockRun.tokens)\nfork  \(forkRun.tokens)")
            }

            // The engine path: `.last` rows through `HybridTarget`, against the fork's stock path.
            target.resetCache()
            let engineRun = Self.engineGreedy(target, prompt: prompt, count: Self.generatedCount)
            let margins = TeacherForcing.run(model: fork.model, prompt: prompt, continuation: forkRun.tokens).map { $0.margin }
            let outcome = tally.record(label, reference: forkRun.tokens, candidate: engineRun.tokens, referenceMargins: margins)
            let engineFirst = LogitCheck.maxAbsDifference(engineRun.first, forkRun.first)

            rows.append([
                "\(index + 1)", "\(prompt.count)", sameTokens ? "identical" : "differ",
                "\(difference)", "\(engineFirst)", Self.describe(outcome),
                String(format: "%.1f / %.1f", Double(Self.generatedCount) / stockRun.seconds, Double(Self.generatedCount) / engineRun.seconds),
            ])
        }
        tally.assertPassed()

        let checkpoint = "`mtp.*` tensors: \(facts.mtpTensors); conv1d \(facts.conv1dShape.map { "\($0)" } ?? "none") (\(facts.unsanitizedConv1d ? "unconverted" : "converted")); vision tensors: \(facts.visionTensors)"
        EngineReport.appendTable(
            title: "HybridQwen35 fork vs stock: \(repo) (\(fork.modelType); \(checkpoint)\(stockIsGarbage ? "; stock loads garbage, F7" : ""))",
            header: ["Prompt", "Prompt tokens", "64 greedy tokens fork vs stock", "First-step max |Δ| fork vs stock", "Engine `.last` first-step max |Δ|", "Engine path vs fork", "tok/s stock / engine"],
            rows: rows)
        EngineReport.append("- Engine path near-tie tally (\(repo)): \(tally.summary)")
    }

    // MARK: Decoding

    /// Prefills `prompt` with the model's own forward, then takes `count` greedy tokens one at a
    /// time (stop tokens don't end the run).
    private static func greedy(_ model: any LanguageModel, prompt: [Int], count: Int) -> Run {
        let cache = model.newCache(parameters: nil)
        let prefill = model(MLXArray(prompt.map { Int32($0) })[.newAxis], cache: cache)
        let first = prefill[0, prefill.dim(1) - 1].asType(.float32)
        eval(first)
        let start = Date()
        var row = first
        var tokens: [Int] = []
        for _ in 0 ..< count {
            let token = row.argMax().item(Int.self)
            tokens.append(token)
            let logits = model(MLXArray([Int32(token)])[.newAxis], cache: cache)
            row = logits[0, logits.dim(1) - 1].asType(.float32)
        }
        return Run(tokens: tokens, first: first, seconds: Date().timeIntervalSince(start))
    }

    /// The same through the engine target: `.last` rows only.
    private static func engineGreedy(_ target: any TargetModel, prompt: [Int], count: Int) -> Run {
        let prefill = target.forward(MLXArray(prompt.map { Int32($0) }), rows: .last, captureForRollback: false, wantHidden: false)
        let first = prefill.logits![0].asType(.float32)
        eval(first)
        let start = Date()
        var row = first
        var tokens: [Int] = []
        for _ in 0 ..< count {
            let token = row.argMax().item(Int.self)
            tokens.append(token)
            let step = target.forward(MLXArray([Int32(token)]), rows: .last, captureForRollback: false, wantHidden: false)
            row = step.logits![0].asType(.float32)
        }
        return Run(tokens: tokens, first: first, seconds: Date().timeIntervalSince(start))
    }

    private static func describe(_ outcome: NearTieTally.Outcome) -> String {
        switch outcome {
        case .identical:
            return "identical"
        case .nearTie(let position, let margin):
            return "near-tie at \(position) (margin \(margin))"
        case .mismatch(let position, _, _, let margin):
            return "MISMATCH at \(position) (margin \(margin.map { "\($0)" } ?? "?"))"
        }
    }

    // MARK: Checkpoint layout

    private struct CheckpointFacts {
        var mtpTensors = 0
        var visionTensors = 0
        var conv1dShape: [Int]?
        var unsanitizedConv1d = false
    }

    /// Reads tensor names and shapes from the safetensors headers (8-byte little-endian length,
    /// then JSON) without loading any weights.
    private static func checkpointFacts(_ directory: URL) throws -> CheckpointFacts {
        var facts = CheckpointFacts()
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
        XCTAssertFalse(files.isEmpty, "no safetensors in \(directory.path)")
        for file in files {
            let handle = try FileHandle(forReadingFrom: file)
            defer { try? handle.close() }
            let prefix = try handle.read(upToCount: 8) ?? Data()
            guard prefix.count == 8 else { continue }
            var length: UInt64 = 0
            for (shift, byte) in prefix.enumerated() {
                length |= UInt64(byte) << (8 * UInt64(shift))
            }
            let header = try handle.read(upToCount: Int(length)) ?? Data()
            let entries = try JSONSerialization.jsonObject(with: header) as? [String: Any] ?? [:]
            for (name, value) in entries where name != "__metadata__" {
                if name.contains("mtp.") {
                    facts.mtpTensors += 1
                }
                if name.hasPrefix("vision_tower") || name.contains("visual") {
                    facts.visionTensors += 1
                }
                if name.contains("conv1d.weight"), let info = value as? [String: Any], let shape = info["shape"] as? [Int] {
                    facts.conv1dShape = shape
                    if shape.last != 1 {
                        facts.unsanitizedConv1d = true
                    }
                }
            }
        }
        return facts
    }
}
