import AssistantKit
import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLMCommon
import XCTest

/// `EngineSelfTest` on real models: it must pass on Qwen3.5-0.8B (the hybrid fork, which the app
/// gates on it) and is reported for Qwen3-0.6B and, with `[ci woof]`, Woof 4B.
final class SelfTestIntegrationTests: XCTestCase {
    override func setUpWithError() throws {
        try IntegrationEnvironment.requireEnabled()
        try MetalAvailability.require()
    }

    func testSelfTestPassesOnQwen35_0_8B() async throws {
        let result = try await run("mlx-community/Qwen3.5-0.8B-MLX-4bit", speculative: true)
        XCTAssertTrue(result.passed, result.detail)
    }

    func testSelfTestOnQwen3_0_6B() async throws {
        let result = try await run("mlx-community/Qwen3-0.6B-4bit", speculative: false)
        XCTAssertTrue(result.passed, result.detail)
    }

    func testSelfTestOnWoof() async throws {
        guard IntegrationEnvironment.woof else {
            throw XCTSkip("Woof runs only with LOCAL_ENGINE_WOOF=1 ([ci woof]).")
        }
        let result = try await run(IntegrationEnvironment.woofRepo, speculative: true)
        XCTAssertTrue(result.passed, result.detail)
    }

    private func run(_ repo: String, speculative: Bool) async throws -> EngineSelfTest.Result {
        let directory = try await IntegrationEnvironment.snapshot(repo)
        var configuration = EngineConfiguration()
        if speculative {
            configuration.generatorFactory = SpeculativeGeneratorFactory(mode: .automatic, curve: .stockMLXDefault, corpusURL: nil)
        }
        let engine = try await InferenceEngine.load(directory: directory, modelID: repo, configuration: configuration)
        try await engine.warmUp()
        let result = await EngineSelfTest.run(engine: engine)
        EngineReport.appendTable(
            title: "Self-test, \(repo) (engine \(engine.info.forked ? "fork" : "stock")\(speculative ? ", speculative factory" : ""))",
            header: ["Passed", "Seconds", "Detail"],
            rows: [[result.passed ? "yes" : "NO", String(format: "%.2f", result.seconds),
                    result.detail.replacingOccurrences(of: "\n", with: "<br>").replacingOccurrences(of: "|", with: "/")]])
        let summary = await engine.sessionSummary()
        XCTAssertTrue(summary.hasPrefix("cold"), "the self-test invalidates the session: \(summary)")
        return result
    }
}
