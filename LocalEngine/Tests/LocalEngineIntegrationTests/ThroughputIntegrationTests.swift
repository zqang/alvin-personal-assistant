import AssistantKit
import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

/// The engine's plain pipelined decode must keep at least 0.8× the stock `TokenIterator`'s
/// greedy speed on the same model (WP20 acceptance); the ratio goes into the report.
final class ThroughputIntegrationTests: XCTestCase {
    private static let minimumRatio = 0.8
    private static let prompt = "Write a long, detailed story about a robot who learns to paint. Keep going."
    private static let tokens = 64

    override func setUpWithError() throws {
        try IntegrationEnvironment.requireEnabled()
        try MetalAvailability.require()
    }

    func testThroughputOnQwen3_0_6B() async throws {
        try await compare("mlx-community/Qwen3-0.6B-4bit")
    }

    func testThroughputOnQwen35_0_8B() async throws {
        try await compare("mlx-community/Qwen3.5-0.8B-MLX-4bit")
    }

    private func compare(_ repo: String) async throws {
        let directory = try await IntegrationEnvironment.snapshot(repo)
        var configuration = EngineConfiguration()
        configuration.temperature = 0
        let engine = try await InferenceEngine.load(directory: directory, modelID: repo, configuration: configuration)
        try await engine.warmUp()
        let loaded = engine.loaded
        let context: [String: any Sendable] = ["enable_thinking": false]
        let user: [String: any Sendable] = ["role": "user", "content": Self.prompt]
        let prompt = try loaded.renderer.renderTokens(messages: [user], tools: nil, context: context, addGenerationPrompt: true)

        // Stock: warm up, then time the tokens after the first (prefill excluded).
        _ = try stockDecode(loaded, prompt: prompt, count: 8)
        var stockRates: [Double] = []
        var engineRates: [Double] = []
        for _ in 0 ..< 3 {
            let (generated, seconds) = try stockDecode(loaded, prompt: prompt, count: Self.tokens)
            stockRates.append(Double(generated) / seconds)

            // Engine: a fresh session each time (same prefill work), the same timing window: from
            // the first token to the last.
            await engine.invalidateSession()
            let request = EngineRequest(system: "", turns: [ChatTurn(role: .user, text: Self.prompt)], maxTokens: Self.tokens + 1, greedy: true)
            let events = try await EngineTestHarness.collect(engine.reply(request))
            let stats = try XCTUnwrap(EngineTestHarness.finish(events)?.stats)
            let firstToken = stats.phases?.firstToken ?? 0
            let steady = stats.generateTime - firstToken
            XCTAssertGreaterThanOrEqual(stats.generatedTokens, 32, "\(repo): the reply stopped too early to measure")
            if stats.generatedTokens > 1 && steady > 0 {
                engineRates.append(Double(stats.generatedTokens - 1) / steady)
            }
        }
        let stock = stockRates.sorted()[stockRates.count / 2]
        let engineRate = try XCTUnwrap(engineRates.sorted().dropFirst(engineRates.count / 2).first)
        let ratio = engineRate / stock
        EngineReport.appendTable(
            title: "Plain decode throughput, \(repo) (greedy, median of 3)",
            header: ["Stock TokenIterator tok/s", "Engine DecodeLoop tok/s", "Ratio", "Engine"],
            rows: [[String(format: "%.1f", stock), String(format: "%.1f", engineRate), String(format: "%.2f", ratio), engine.info.forked ? "fork" : "stock"]])
        XCTAssertGreaterThanOrEqual(ratio, Self.minimumRatio, "\(repo): engine \(engineRate) tok/s vs stock \(stock) tok/s")
    }

    /// Prefills `prompt` on a fresh cache and takes the first token (which waits for the
    /// prefill), then times `count` more greedy tokens. Stop tokens don't end the run.
    private func stockDecode(_ loaded: LoadedModel, prompt: [Int], count: Int) throws -> (generated: Int, seconds: Double) {
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map { Int32($0) })), model: loaded.model, cache: nil,
            parameters: GenerateParameters(maxTokens: count + 1, temperature: 0))
        guard iterator.next() != nil else { return (0, 0) }
        let start = Date()
        var generated = 0
        while iterator.next() != nil {
            generated += 1
        }
        return (generated, Date().timeIntervalSince(start))
    }
}
