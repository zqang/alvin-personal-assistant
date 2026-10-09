import AssistantKit
import Foundation
@testable import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

/// The small-M kernel on the CI GPU (WP41): kernel versus stock time per row count at Woof 4B's
/// layer widths, and `FastKernelsExtension` on Qwen3.5-0.8B (and Woof 4B when enabled): its
/// self-test, the before/after cost curves, the 25% decision, and speculative decoding with the
/// kernel against the stock layers (wall time, plus the teacher-forced check). Reports only;
/// only correctness fails a test.
final class KernelSpeedIntegrationTests: XCTestCase {
    override func setUpWithError() throws {
        try IntegrationEnvironment.requireEnabled()
        try MetalAvailability.require()
        try XCTSkipUnless(SmallMQuantizedMatmul.canRunOnDefaultDevice, "Custom Metal kernels can't run on this device.")
    }

    // MARK: One layer

    func testKernelVersusStockTimePerRowCount() throws {
        let shapes = [(2560, 9216), (9216, 2560), (2560, 8192), (1024, 3584)]
        var summary: [[String]] = []
        for (index, (k, n)) in shapes.enumerated() {
            let stock = Self.randomLayer(inputDimensions: k, outputDimensions: n, dtype: .bfloat16, seed: 4120 + UInt64(index))
            let fast = FastQuantizedLinear(stock)
            var rows: [[String]] = []
            var stockOne = 0.0
            var kernelEight = 0.0
            var stockEight = 0.0
            for m in 1 ... 9 {
                let x = MLXRandom.normal([1, m, k]).asType(.bfloat16)
                let expected = stock(x)
                let actual = fast(x)
                try checkedEval(expected, actual)
                let scale = Double(MLX.abs(expected.asType(.float32)).max().item(Float.self))
                let tolerance = KernelSelfTest.tolerance(for: .bfloat16)
                XCTAssertTrue(
                    LogitCheck.isClose(actual, expected, rtol: tolerance, atol: tolerance * scale),
                    "\(k)→\(n), M = \(m): max |Δ| = \(LogitCheck.maxAbsDifference(actual, expected)) of \(scale)")

                let stockMs = try Self.milliseconds(stock, x)
                let fastMs = try Self.milliseconds(fast, x)
                if m == 1 { stockOne = stockMs }
                if m == 8 {
                    stockEight = stockMs
                    kernelEight = fastMs
                }
                rows.append([
                    "\(m)", String(format: "%.3f", stockMs), String(format: "%.3f", fastMs),
                    String(format: "%.2f×", stockMs / max(fastMs, 1e-9)),
                    String(format: "%.2f", stockMs / max(stockOne, 1e-9)), String(format: "%.2f", fastMs / max(stockOne, 1e-9)),
                    fast.usesKernel(for: x) ? "kernel" : "stock",
                ])
            }
            EngineReport.appendTable(
                title: "Small-M kernel vs stock quantizedMM, \(k)→\(n), 4-bit g64, bfloat16 (ms per product, median of 5 × 16)",
                header: ["M", "Stock ms", "Fast ms", "Speed-up", "Stock / stock M=1", "Fast / stock M=1", "Path"], rows: rows)
            summary.append([
                "\(k)→\(n)", String(format: "%.2f", stockEight / max(stockOne, 1e-9)),
                String(format: "%.2f", kernelEight / max(stockOne, 1e-9)),
            ])
        }
        EngineReport.appendTable(
            title: "Small-M kernel: cost of 8 rows relative to one stock row", header: ["Shape", "Stock c(8)", "Kernel c(8)"], rows: summary)
    }

    // MARK: Whole models

    func testFastKernelsExtensionOnQwen35_0_8B() async throws {
        try await runExtension(repo: "mlx-community/Qwen3.5-0.8B-MLX-4bit")
    }

    func testFastKernelsExtensionOnWoof4B() async throws {
        try XCTSkipUnless(IntegrationEnvironment.woof, "Woof 4B runs only with LOCAL_ENGINE_WOOF=1.")
        try await runExtension(repo: IntegrationEnvironment.woofRepo)
    }

    private static let system = "You are a careful copy editor. Return only the corrected text."
    private static let editPrompt = """
        Fix the typos in this paragraph and return the whole paragraph, otherwise unchanged:

        The engine keeps the conversation resident in memory, so a continued chat only reads the new \
        tokens. When the reply repeats the prompt, the drafter guesess the next tokens from earlier \
        text and the model checks several of them in one pass. Accepted tokens are emited at once; a \
        rejected one is replaced by the models own choice, so the anwser is exactly what plain \
        decoding would have produced.
        """

    /// Loads `repo` with the kernels requested and any gain accepted (so the kernels are in for
    /// the speculation runs), reports the extension's verdict as the default 25% rule would have
    /// made it, then decodes the same edit with prompt-lookup rounds of 4 and 8 drafts on the
    /// fast and on the stock layers.
    private func runExtension(repo: String) async throws {
        let directory = try await IntegrationEnvironment.snapshot(repo)
        let fastKernels = FastKernelsExtension(requiredGain: -.infinity)
        var configuration = EngineConfiguration()
        configuration.temperature = 0
        configuration.extensions = [fastKernels]
        configuration.fastKernelsRequested = true
        let engine = try await InferenceEngine.load(directory: directory, modelID: repo, configuration: configuration)
        let model: Module = engine.loaded.model
        let stockLayers = FastKernelsExtension.eligibleLayers(in: model)
        try await engine.warmUp()

        let report = try XCTUnwrap(fastKernels.report)
        XCTAssertEqual(report.selfTest?.passed, true, report.selfTest?.detail ?? "no self-test")
        XCTAssertEqual(report.outcome, .faster, report.summary)
        XCTAssertTrue(report.kernelsInstalled)
        XCTAssertEqual(report.layers, stockLayers.count)
        let before = try XCTUnwrap(report.before)
        let after = try XCTUnwrap(report.after)
        let gain = try XCTUnwrap(report.gain)
        let widths = before.seconds.keys.sorted()
        EngineReport.appendTable(
            title: "Fast kernels, \(repo): CostProbe before and after the swap (\(report.layers) layers)",
            header: ["Rows S"] + widths.map(String.init),
            rows: [
                ["stock ms"] + widths.map { String(format: "%.2f", (before.seconds[$0] ?? 0) * 1000) },
                ["fast ms"] + widths.map { String(format: "%.2f", (after.seconds[$0] ?? 0) * 1000) },
                ["stock c(S)"] + widths.map { String(format: "%.2f", before.relative($0)) },
                ["fast c(S)"] + widths.map { String(format: "%.2f", after.relative($0)) },
            ])
        let verdict = gain >= 0.25 ? "would be enabled" : "would stay off"
        EngineReport.append(
            "- Fast kernels, \(repo): c(8) gain \(String(format: "%.0f%%", gain * 100)), so the default 25% rule \(verdict). "
                + "Self-test: \(report.selfTest?.detail ?? "–"), " + String(format: "%.2f s", report.selfTest?.seconds ?? 0))

        // Speculation with the kernel against the stock layers.
        let fastLayers = model.leafModules().flattened().compactMap { (path, module) -> (String, QuantizedLinear)? in
            guard let layer = module as? FastQuantizedLinear else { return nil }
            return (path, layer as QuantizedLinear)
        }
        let originalLayers = stockLayers.map { ($0.path, $0.layer) }
        XCTAssertEqual(fastLayers.count, stockLayers.count)
        let request = EngineRequest(system: Self.system, turns: [ChatTurn(role: .user, text: Self.editPrompt)], maxTokens: 160, greedy: true)
        var rows: [[String]] = []
        var tally = NearTieTally(tolerance: NearTieTally.quantizedTolerance)
        for draftLength in [4, 8] {
            var seconds: [Bool: Double] = [:]
            for kernels in [true, false] {
                FastKernelsExtension.install(kernels ? fastLayers : originalLayers, in: model)
                await engine.updateConfiguration { $0.generatorFactory = PromptLookupFactory(draftLength: draftLength) }
                await engine.invalidateSession()
                let run = try await Self.generate(engine, request: request)
                seconds[kernels] = run.seconds
                // Teacher forcing on the stock layers.
                FastKernelsExtension.install(originalLayers, in: model)
                let forced = TeacherForcing.run(model: engine.loaded.model, prompt: run.context, continuation: run.generated)
                tally.record(
                    "\(repo) K=\(draftLength) \(kernels ? "kernel" : "stock")", reference: forced.map(\.argmax), candidate: run.generated,
                    referenceMargins: forced.map(\.margin))
                rows.append([
                    "K = \(draftLength), \(kernels ? "fast kernels" : "stock layers")", "\(run.stats.generatedTokens)",
                    "\(run.stats.speculation?.rounds ?? 0)",
                    run.stats.speculation?.meanTokensPerRound.map { String(format: "%.2f", $0) } ?? "–",
                    String(format: "%.2f", run.seconds),
                    kernels ? "–" : String(format: "%.2f×", (seconds[false] ?? 0) / max(seconds[true] ?? 0, 1e-9)),
                ])
            }
        }
        EngineReport.appendTable(
            title: "Speculation with and without the fast kernels, \(repo), \"edit this paragraph\" (greedy, forced prompt lookup)",
            header: ["Run", "Tokens", "Rounds", "Tokens/round", "Decode s", "Kernel speed-up"], rows: rows)
        EngineReport.append("- Teacher-forced check with fast kernels (\(repo)): \(tally.summary)")
        tally.assertPassed()
    }

    // MARK: Helpers

    /// A stock 4-bit, group-64 layer with random packed weights, scales and biases in `dtype`.
    static func randomLayer(inputDimensions k: Int, outputDimensions n: Int, dtype: DType, seed: UInt64) -> QuantizedLinear {
        let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 4)
        let words = [n, k / 8]
        let groups = [n, k / 64]
        let high = MLXRandom.randInt(UInt32(0) ..< UInt32(1 << 16), words, key: keys[0])
        let low = MLXRandom.randInt(UInt32(0) ..< UInt32(1 << 16), words, key: keys[1])
        let scales = MLXRandom.uniform(Float(0.002) ..< Float(0.02), groups, key: keys[2])
        let biases = scales * Float(-8) + MLXRandom.uniform(Float(-0.01) ..< Float(0.01), groups, key: keys[3])
        let layer = QuantizedLinear(
            weight: (high << UInt32(16)) | low, bias: nil, scales: scales.asType(dtype), biases: biases.asType(dtype),
            groupSize: 64, bits: 4, mode: .affine)
        eval(layer)
        return layer
    }

    /// Median milliseconds per product of `layer(x)`: 5 timings of 16 products evaluated together
    /// (so launch and sync overhead don't dominate), after a warm-up.
    static func milliseconds(_ layer: QuantizedLinear, _ x: MLXArray) throws -> Double {
        try checkedEval(layer(x))
        var samples: [Double] = []
        for _ in 0 ..< 5 {
            let started = ProcessInfo.processInfo.systemUptime
            let outputs = (0 ..< 16).map { _ in layer(x) }
            try withError { eval(outputs) }
            samples.append((ProcessInfo.processInfo.systemUptime - started) * 1000 / 16)
        }
        samples.sort()
        return samples[samples.count / 2]
    }

    struct Run {
        let context: [Int]
        let generated: [Int]
        let seconds: Double
        let stats: LocalGenerationStats
    }

    /// One reply from a fresh session, timed from the first token to the last.
    static func generate(_ engine: InferenceEngine, request: EngineRequest) async throws -> Run {
        let events = try await EngineTestHarness.collect(engine.reply(request))
        let stats = try XCTUnwrap(EngineTestHarness.finish(events)?.stats)
        let (context, generated) = try await engine.withSession { session -> ([Int], [Int]) in
            let replyStart = session.snapshot?.turns.last(where: { $0.turn.role == .user })?.replyStart ?? session.ledger.count
            return (Array(session.ledger[..<replyStart]), Array(session.ledger[replyStart...]))
        }
        let seconds = stats.generateTime - (stats.phases?.firstToken ?? 0)
        return Run(context: context, generated: generated, seconds: seconds, stats: stats)
    }
}

/// Prompt-lookup speculation with a fixed draft length whenever there is a match.
final class PromptLookupFactory: GeneratorFactory, @unchecked Sendable {
    let draftLength: Int

    init(draftLength: Int) {
        self.draftLength = draftLength
    }

    func makeGenerator(_ context: GeneratorContext) -> TokenGenerator {
        let drafter = PromptLookupDrafter(renderer: context.renderer)
        drafter.reset(ledger: context.session.ledger, request: context.request)
        return SpeculativeLoop(
            context, drafters: [drafter], policy: SharedDraftPolicy(curve: .stockMLXDefault, maxDraft: 8),
            options: SpeculativeLoop.Options(forcedDraftLength: draftLength))
    }
}
