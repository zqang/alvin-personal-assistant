import AssistantKit
import Foundation
@testable import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLMCommon
import MLXNN
import XCTest

/// The small-M quantized matmul kernel (WP41): it matches `quantizedMM` for every shape, row
/// count and dtype; ineligible products fall back to the stock path bitwise; fast layers inside
/// a hybrid model give the stock logits; and `FastKernelsExtension` swaps them in only when asked,
/// keeps them only with the required gain, and undoes the swap otherwise.
///
/// The kernel itself runs only on a Metal GPU; elsewhere those tests are skipped and the rest
/// check the stock fallbacks.
final class SmallMKernelTests: XCTestCase {
    override func setUpWithError() throws {
        try MetalAvailability.require()
    }

    private func requireKernel() throws {
        try XCTSkipUnless(SmallMQuantizedMatmul.canRunOnDefaultDevice, "Custom Metal kernels can't run on this device.")
    }

    // MARK: The kernel against the stock product

    func testKernelMatchesTheStockProductAcrossShapesRowsAndTypes() throws {
        try requireKernel()
        let dtypes: [DType] = [.float32, .float16, .bfloat16]
        let result = KernelSelfTest.run(options: KernelSelfTest.Options(dtypes: dtypes))
        XCTAssertTrue(result.passed, result.detail + "\n" + result.failures.joined(separator: "\n"))
        let shapes = KernelSelfTest.defaultShapes.count
        XCTAssertEqual(result.cases, shapes * 9 * dtypes.count, "every shape × M ∈ 1…9 × dtype")
        XCTAssertEqual(result.kernelCases, shapes * 8 * dtypes.count, "M = 2…9 run on the kernel")
        XCTAssertTrue(result.failures.isEmpty)
        EngineReport.append(
            "- Small-M kernel self-test (\(KernelSelfTest.defaultShapes.map(\.description).joined(separator: ", ")); "
                + "float32, float16, bfloat16; M 1…9): \(result.detail), "
                + String(format: "%.2f s", result.seconds))
    }

    /// The raw kernel, without `FastQuantizedLinear`, on a layer quantized from float weights.
    func testKernelDirectlyAgainstQuantizedMM() throws {
        try requireKernel()
        MLXRandom.seed(4101)
        let (k, n) = (1024, 64)
        let (weight, scales, biases) = MLX.quantized(MLXRandom.normal([n, k]) * Float(0.05), groupSize: 64, bits: 4, mode: .affine)
        let unwrappedBiases = try XCTUnwrap(biases)
        for rows in SmallMQuantizedMatmul.rows {
            let x = MLXRandom.normal([rows, k])
            let fast = SmallMQuantizedMatmul.multiply(x, weight: weight, scales: scales, biases: unwrappedBiases)
            let stock = quantizedMM(x, weight, scales: scales, biases: unwrappedBiases, transpose: true, groupSize: 64, bits: 4, mode: .affine)
            try checkedEval(fast, stock)
            XCTAssertEqual(fast.shape, [rows, n])
            XCTAssertEqual(fast.dtype, .float32)
            let scale = Double(MLX.abs(stock).max().item(Float.self))
            XCTAssertTrue(
                LogitCheck.isClose(fast, stock, rtol: 1e-3, atol: 1e-3 * scale),
                "M = \(rows): max |Δ| = \(LogitCheck.maxAbsDifference(fast, stock)) of \(scale)")
        }
    }

    func testSelfTestCatchesAWrongKernel() throws {
        try requireKernel()
        var options = KernelSelfTest.Options(shapes: [KernelSelfTest.Shape(512, 12, bias: true)])
        options.debugBreakKernel = true
        let result = KernelSelfTest.run(options: options)
        XCTAssertFalse(result.passed)
        XCTAssertEqual(result.cases, 9)
        XCTAssertEqual(result.failures.count, 8, "every kernel product (M = 2…9) is caught; M = 1 is stock: \(result.failures)")
        XCTAssertTrue(result.detail.contains("differ"), result.detail)
    }

    func testSelfTestStopsWhenTheGPUIsNoLongerAllowed() throws {
        try requireKernel()
        let result = KernelSelfTest.run(options: KernelSelfTest.Options(shapes: [KernelSelfTest.Shape(512, 12)]), isAllowed: { false })
        XCTAssertFalse(result.passed)
        XCTAssertEqual(result.cases, 0)
        XCTAssertTrue(result.detail.hasPrefix("stopped"), result.detail)
    }

    func testSelfTestFailsWithoutTheGPU() throws {
        let result = Device.withDefaultDevice(.cpu) {
            KernelSelfTest.run(options: KernelSelfTest.Options(shapes: [KernelSelfTest.Shape(512, 12)]))
        }
        XCTAssertFalse(result.passed)
        XCTAssertEqual(result.cases, 0)
        XCTAssertEqual(result.kernelCases, 0)
    }

    // MARK: Fallback to the stock layer

    /// Products the kernel doesn't handle run `super`, bitwise equal to the stock layer.
    func testIneligibleProductsFallBackToTheStockLayerBitwise() throws {
        MLXRandom.seed(4102)
        struct Case {
            let label: String
            let layer: QuantizedLinear
            let x: MLXArray
        }
        func layer(_ k: Int, _ n: Int, groupSize: Int = 64, bits: Int = 4, bias: Bool = false) -> QuantizedLinear {
            QuantizedLinear(
                weight: MLXRandom.normal([n, k]) * Float(0.05), bias: bias ? MLXRandom.normal([n]) : nil,
                groupSize: groupSize, bits: bits, mode: .affine)
        }
        let eligible = layer(512, 16, bias: true)
        let cases = [
            Case(label: "M = 1", layer: eligible, x: MLXRandom.normal([1, 1, 512])),
            Case(label: "M = 10", layer: eligible, x: MLXRandom.normal([1, 10, 512])),
            Case(label: "M = 16, 2-D x", layer: eligible, x: MLXRandom.normal([16, 512])),
            Case(label: "1-D x", layer: eligible, x: MLXRandom.normal([512])),
            Case(label: "K = 576, not a multiple of 512", layer: layer(576, 16), x: MLXRandom.normal([1, 4, 576])),
            Case(label: "N = 10, not a multiple of 4", layer: layer(512, 10), x: MLXRandom.normal([1, 4, 512])),
            Case(label: "8 bits", layer: layer(512, 16, bits: 8), x: MLXRandom.normal([1, 4, 512])),
            Case(label: "group size 32", layer: layer(512, 16, groupSize: 32), x: MLXRandom.normal([1, 4, 512])),
            Case(label: "x float16, scales float32", layer: eligible, x: MLXRandom.normal([1, 4, 512]).asType(.float16)),
        ]
        for testCase in cases {
            let fast = FastQuantizedLinear(testCase.layer)
            XCTAssertFalse(fast.usesKernel(for: testCase.x), testCase.label)
            let expected = testCase.layer(testCase.x)
            let actual = fast(testCase.x)
            eval(expected, actual)
            XCTAssertTrue(LogitCheck.isExactlyEqual(actual, expected), "\(testCase.label): max |Δ| = \(LogitCheck.maxAbsDifference(actual, expected))")
        }
        XCTAssertTrue(FastQuantizedLinear.isEligible(eligible))
        for ineligible in [layer(576, 16), layer(512, 10), layer(512, 16, bits: 8), layer(512, 16, groupSize: 32)] {
            XCTAssertFalse(FastQuantizedLinear.isEligible(ineligible))
        }
    }

    /// Custom kernels run only on the GPU: on the CPU device even eligible products take the
    /// stock path.
    func testTheCPUDeviceUsesTheStockPath() throws {
        MLXRandom.seed(4103)
        let stock = QuantizedLinear(weight: MLXRandom.normal([16, 512]) * Float(0.05), bias: nil, groupSize: 64, bits: 4, mode: .affine)
        let fast = FastQuantizedLinear(stock)
        let x = MLXRandom.normal([1, 4, 512])
        eval(stock, x)
        let (expected, actual, usesKernel) = Device.withDefaultDevice(.cpu) { () -> (MLXArray, MLXArray, Bool) in
            let usesKernel = fast.usesKernel(for: x)
            let expected = stock(x)
            let actual = fast(x)
            eval(expected, actual)
            return (expected, actual, usesKernel)
        }
        XCTAssertFalse(usesKernel)
        XCTAssertTrue(LogitCheck.isExactlyEqual(actual, expected))
    }

    /// The fast layer shares the stock layer's arrays (no copies) and is frozen like it.
    func testTheFastLayerSharesTheStockArrays() throws {
        MLXRandom.seed(4104)
        let stock = QuantizedLinear(
            weight: MLXRandom.normal([16, 512]), bias: MLXRandom.normal([16]), groupSize: 64, bits: 4, mode: .affine)
        let fast = FastQuantizedLinear(stock)
        XCTAssertTrue(fast.weight === stock.weight)
        XCTAssertTrue(fast.scales === stock.scales)
        XCTAssertTrue(fast.biases === stock.biases)
        XCTAssertTrue(fast.bias === stock.bias)
        XCTAssertEqual(fast.groupSize, 64)
        XCTAssertEqual(fast.bits, 4)
        XCTAssertEqual(fast.mode, .affine)
        XCTAssertTrue(fast.trainableParameters().flattened().isEmpty, "frozen like the stock layer")
        XCTAssertEqual(
            Set(fast.parameters().flattened().map(\.0)), Set(stock.parameters().flattened().map(\.0)),
            "the same parameter keys, so weights load into either")
    }

    // MARK: Inside a model

    /// Every eligible layer of a hybrid model swapped for `FastQuantizedLinear`: logits allClose
    /// to the stock model's for every row count (bitwise where the kernel isn't used), also
    /// across cached chunks; swapping back restores the stock logits bitwise.
    func testFastLayersInsideTheHybridGiveTheStockLogits() throws {
        let model = try Self.makeKernelSizedHybrid(seed: 4105)
        let candidates = FastKernelsExtension.eligibleLayers(in: model)
        XCTAssertEqual(candidates.count, Self.kernelSizedEligibleLayers, candidates.map(\.path).joined(separator: ", "))
        let tokens = TinyModels.tokens(12, seed: 4106)
        let lengths = [1, 2, 3, 5, 8, 9, 12]
        let chunks = [5, 1, 9, 3, 12]

        func run() -> (single: [MLXArray], chunked: [MLXArray]) {
            let single = lengths.map { length -> MLXArray in
                let logits = TinyForkModels.engineLogits(model, Array(tokens.prefix(length)), cache: model.newCache(parameters: nil)).logits!
                eval(logits)
                return logits
            }
            let cache = model.newCache(parameters: nil)
            let chunked = chunks.enumerated().map { index, count -> MLXArray in
                let chunk = TinyModels.tokens(count, seed: 4107 + UInt64(index))
                let logits = TinyForkModels.engineLogits(model, chunk, cache: cache).logits!
                eval(logits)
                return logits
            }
            return (single, chunked)
        }

        let stock = run()
        let fastLayers = candidates.map { ($0.path, FastQuantizedLinear($0.layer) as QuantizedLinear) }
        FastKernelsExtension.install(fastLayers, in: model)
        let swapped = model.leafModules().flattened().filter { $0.1 is FastQuantizedLinear }.map(\.0)
        XCTAssertEqual(Set(swapped), Set(candidates.map(\.path)))
        XCTAssertTrue(FastKernelsExtension.eligibleLayers(in: model).isEmpty, "swapped layers aren't candidates again")

        let fast = run()
        let kernel = SmallMQuantizedMatmul.canRunOnDefaultDevice
        func compare(_ label: String, rows: Int, _ candidate: MLXArray, _ reference: MLXArray) {
            if kernel && SmallMQuantizedMatmul.rows.contains(rows) {
                let scale = Double(MLX.abs(reference).max().item(Float.self))
                XCTAssertTrue(
                    LogitCheck.isClose(candidate, reference, rtol: 1e-3, atol: 1e-3 * scale),
                    "\(label): max |Δ| = \(LogitCheck.maxAbsDifference(candidate, reference)) of \(scale)")
            } else {
                XCTAssertTrue(LogitCheck.isExactlyEqual(candidate, reference), "\(label): the stock path is bitwise the same")
            }
        }
        for (index, length) in lengths.enumerated() {
            compare("\(length) tokens", rows: length, fast.single[index], stock.single[index])
        }
        for (index, count) in chunks.enumerated() {
            compare("chunk \(index) (\(count) tokens)", rows: count, fast.chunked[index], stock.chunked[index])
        }

        FastKernelsExtension.install(candidates.map { ($0.path, $0.layer) }, in: model)
        XCTAssertTrue(model.leafModules().flattened().allSatisfy { !($0.1 is FastQuantizedLinear) })
        let restored = run()
        for index in lengths.indices {
            XCTAssertTrue(LogitCheck.isExactlyEqual(restored.single[index], stock.single[index]), "\(lengths[index]) tokens")
        }
        for index in chunks.indices {
            XCTAssertTrue(LogitCheck.isExactlyEqual(restored.chunked[index], stock.chunked[index]), "chunk \(index)")
        }
    }

    /// Layers are set on their parents by property key: a model whose first block has nothing to
    /// swap still swaps the second block's layer, and layers held in an array property are left
    /// alone.
    func testSwappingSetsLayersOnTheirParents() {
        MLXRandom.seed(4108)
        func layer(_ k: Int) -> QuantizedLinear {
            QuantizedLinear(weight: MLXRandom.normal([16, k]) * Float(0.05), bias: nil, groupSize: 64, bits: 4, mode: .affine)
        }
        let stack = KernelTestStack(blocks: [KernelTestBlock(layer(576)), KernelTestBlock(layer(512))], heads: [layer(512)])
        let firstBlock = stack.blocks[0].proj
        let head = stack.heads[0]
        let pairs = FastKernelsExtension.layerPairs(in: stack)
        XCTAssertEqual(pairs.map(\.path), ["blocks.1.proj"])

        FastKernelsExtension.install(pairs, fast: true, in: stack)
        XCTAssertTrue(stack.blocks[1].proj === pairs[0].fast)
        XCTAssertTrue(stack.blocks[0].proj === firstBlock)
        XCTAssertTrue(stack.heads[0] === head)
        XCTAssertEqual(stack.heads.count, 1)

        let again = FastKernelsExtension.layerPairs(in: stack)
        XCTAssertEqual(again.map(\.path), ["blocks.1.proj"])
        XCTAssertEqual(again.map(\.startsFast), [true])
        XCTAssertTrue(again[0].fast === pairs[0].fast)
        FastKernelsExtension.install(again, fast: false, in: stack)
        XCTAssertFalse(stack.blocks[1].proj is FastQuantizedLinear)
        FastKernelsExtension.restore(again, in: stack)
        XCTAssertTrue(stack.blocks[1].proj === pairs[0].fast, "restore puts back what the model started with")
        FastKernelsExtension.restore(pairs, in: stack)
        XCTAssertTrue(stack.blocks[1].proj === pairs[0].stock)
    }

    // MARK: The extension

    func testTheRequestFlagIsAValueOfTheConfiguration() {
        var configuration = EngineConfiguration()
        XCTAssertFalse(configuration.fastKernelsRequested)
        let fastKernels = FastKernelsExtension()
        configuration.extensions = [fastKernels]
        configuration.fastKernelsRequested = true
        XCTAssertTrue(configuration.fastKernelsRequested)
        configuration.fastKernelsRequested = true
        XCTAssertEqual(configuration.extensions.count, 2, "one marker however often it is set")
        XCTAssertTrue(configuration.extensions.first === fastKernels)

        let copy = configuration
        configuration.fastKernelsRequested = false
        XCTAssertFalse(configuration.fastKernelsRequested)
        XCTAssertTrue(copy.fastKernelsRequested, "copies keep their own value")
        XCTAssertEqual(configuration.extensions.count, 1)
        XCTAssertTrue(configuration.extensions.first === fastKernels)
    }

    func testTheExtensionDoesNothingUnlessRequested() async throws {
        let fastKernels = FastKernelsExtension()
        let (engine, model) = try Self.makeEngine(seed: 4110, fastKernels: fastKernels, requested: false)
        try await engine.warmUp()
        XCTAssertEqual(fastKernels.report?.outcome, .notRequested)
        XCTAssertFalse(fastKernels.isEnabled)
        XCTAssertEqual(fastKernels.summary, "Fast kernels off (not requested)")
        XCTAssertEqual(Self.fastLayerCount(model), 0)
    }

    /// Without the required gain the probe's verdict undoes the swap: the stock layers (the
    /// same objects) are back and the logits are bitwise the stock ones.
    func testTheExtensionUndoesTheSwapWithoutTheGain() async throws {
        let fastKernels = FastKernelsExtension(requiredGain: .infinity)
        fastKernels.selfTestShapes = [KernelSelfTest.Shape(512, 12, bias: true)]
        let (engine, model) = try Self.makeEngine(seed: 4111, fastKernels: fastKernels, requested: true)
        let before = Self.layerIdentities(model)
        let stockLogits = try await Self.nextLogits(engine)
        try await engine.warmUp()

        let report = try XCTUnwrap(fastKernels.report)
        XCTAssertEqual(Self.fastLayerCount(model), 0)
        XCTAssertEqual(Self.layerIdentities(model), before, "the original layer objects are back")
        let logits = try await Self.nextLogits(engine)
        XCTAssertTrue(LogitCheck.isExactlyEqual(logits, stockLogits))
        guard SmallMQuantizedMatmul.canRunOnDefaultDevice else {
            XCTAssertEqual(report.outcome, .unavailable)
            XCTAssertTrue(report.summary.hasPrefix("Fast kernels unavailable"), report.summary)
            return
        }
        XCTAssertEqual(report.outcome, .notFaster, report.summary)
        XCTAssertEqual(report.layers, Self.kernelSizedEligibleLayers)
        XCTAssertEqual(report.selfTest?.passed, true, report.selfTest?.detail ?? "")
        XCTAssertEqual(Set(try XCTUnwrap(report.before).seconds.keys), [1, 2, 3, 4, 6, 8])
        XCTAssertEqual(Set(try XCTUnwrap(report.after).seconds.keys), [1, 2, 3, 4, 6, 8])
        XCTAssertNotNil(report.gain)
        XCTAssertTrue(report.summary.hasPrefix("Fast kernels off: c(8) "), report.summary)
        EngineReport.append("- Fast kernels on the kernel-sized tiny hybrid (gain required: impossible): \(report.summary)")
    }

    /// With the gain the kernels stay in: every eligible layer is fast; speculative rounds of up
    /// to 9 rows (on the kernel) decode exactly what a stock engine with the same weights decodes
    /// plainly, and the ledger stays consistent.
    func testTheExtensionKeepsTheKernelsWithTheGain() async throws {
        try requireKernel()
        let fastKernels = FastKernelsExtension(requiredGain: -.infinity)
        fastKernels.selfTestShapes = [KernelSelfTest.Shape(512, 12, bias: true)]
        let (engine, model) = try Self.makeEngine(seed: 4112, fastKernels: fastKernels, requested: true)
        try await engine.warmUp()

        let report = try XCTUnwrap(fastKernels.report)
        XCTAssertEqual(report.outcome, .faster, report.summary)
        XCTAssertTrue(report.kernelsInstalled)
        XCTAssertFalse(report.measuredOnly)
        XCTAssertTrue(fastKernels.isEnabled)
        XCTAssertEqual(Self.fastLayerCount(model), Self.kernelSizedEligibleLayers)
        XCTAssertTrue(report.summary.hasPrefix("Fast kernels on (\(Self.kernelSizedEligibleLayers) layers): c(8) "), report.summary)

        // A dry run on the enabled engine measures the same comparison and leaves the fast layers in.
        let identities = Self.layerIdentities(model)
        let measured = try await fastKernels.measure(engine)
        XCTAssertEqual(measured.outcome, .faster, measured.summary)
        XCTAssertTrue(measured.measuredOnly)
        XCTAssertTrue(measured.kernelsInstalled)
        XCTAssertEqual(measured.layers, Self.kernelSizedEligibleLayers)
        XCTAssertEqual(Self.layerIdentities(model), identities, "the same fast layer objects stay in")
        XCTAssertEqual(fastKernels.report, report, "a dry run doesn't change the report")

        let (stockEngine, _) = try Self.makeEngine(seed: 4112, fastKernels: FastKernelsExtension(), requested: false)
        try await stockEngine.warmUp()
        let fastLogits = try await Self.nextLogits(engine)
        let stockLogits = try await Self.nextLogits(stockEngine)
        let scale = Double(MLX.abs(stockLogits).max().item(Float.self))
        XCTAssertTrue(
            LogitCheck.isClose(fastLogits, stockLogits, rtol: 1e-3, atol: 1e-3 * scale),
            "max |Δ| = \(LogitCheck.maxAbsDifference(fastLogits, stockLogits)) of \(scale)")

        let request = EngineRequest(
            system: "You are Alvin.", turns: [ChatTurn(role: .user, text: "Copy this: the quick brown fox.")], maxTokens: 24)
        for draftLength in [1, 4, 8] {
            let factory = ForcedDraftFactory(draftLength: draftLength)
            await engine.updateConfiguration { $0.generatorFactory = factory }
            await engine.invalidateSession()
            await stockEngine.invalidateSession()
            let fastEvents = try await EngineTestHarness.collect(engine.reply(request))
            let stockEvents = try await EngineTestHarness.collect(stockEngine.reply(request))
            XCTAssertEqual(EngineTestHarness.text(fastEvents), EngineTestHarness.text(stockEvents), "K = \(draftLength)")
            let speculation = EngineTestHarness.finish(fastEvents)?.stats.speculation
            XCTAssertGreaterThan(speculation?.rounds ?? 0, 0, "K = \(draftLength): rounds ran")
            let consistency = try await engine.withSession { $0.assertConsistent() }
            XCTAssertTrue(consistency.isConsistent(), "K = \(draftLength): \(consistency)")
        }
        EngineReport.append("- Fast kernels on the kernel-sized tiny hybrid (any gain accepted): \(report.summary)")
    }

    /// A failing self-test keeps the stock layers, whatever the gain would be.
    func testAFailingSelfTestKeepsTheStockLayers() async throws {
        try requireKernel()
        let fastKernels = FastKernelsExtension(requiredGain: -.infinity)
        fastKernels.selfTestShapes = [KernelSelfTest.Shape(512, 12, bias: true)]
        fastKernels.debugBreakSelfTest = true
        let (engine, model) = try Self.makeEngine(seed: 4115, fastKernels: fastKernels, requested: true)
        let identities = Self.layerIdentities(model)
        try await engine.warmUp()
        let report = try XCTUnwrap(fastKernels.report)
        XCTAssertEqual(report.outcome, .selfTestFailed, report.summary)
        XCTAssertFalse(report.kernelsInstalled)
        XCTAssertNil(report.before, "nothing was measured")
        XCTAssertEqual(Self.layerIdentities(model), identities)
        XCTAssertTrue(report.summary.hasPrefix("Fast kernels off: self-test failed"), report.summary)
        // The engine still answers.
        let events = try await EngineTestHarness.collect(engine.reply(
            EngineRequest(system: "You are Alvin.", turns: [ChatTurn(role: .user, text: "Hello")])))
        XCTAssertNotNil(EngineTestHarness.finish(events))
    }

    /// `measure` runs whatever the flag says and leaves the stock layers (the same objects) in.
    func testMeasureIsADryRun() async throws {
        let fastKernels = FastKernelsExtension(requiredGain: -.infinity)
        fastKernels.selfTestShapes = [KernelSelfTest.Shape(512, 12, bias: true)]
        let (engine, model) = try Self.makeEngine(seed: 4114, fastKernels: fastKernels, requested: false)
        try await engine.warmUp()
        XCTAssertEqual(fastKernels.report?.outcome, .notRequested)
        let identities = Self.layerIdentities(model)

        let measured = try await fastKernels.measure(engine)
        XCTAssertTrue(measured.measuredOnly)
        XCTAssertFalse(measured.kernelsInstalled)
        XCTAssertEqual(Self.layerIdentities(model), identities)
        XCTAssertEqual(fastKernels.report?.outcome, .notRequested, "a dry run doesn't change the report")
        guard SmallMQuantizedMatmul.canRunOnDefaultDevice else {
            XCTAssertEqual(measured.outcome, .unavailable)
            return
        }
        XCTAssertEqual(measured.outcome, .faster, measured.summary)
        XCTAssertNotNil(measured.before)
        XCTAssertNotNil(measured.after)
        XCTAssertTrue(measured.summary.hasPrefix("Fast kernels would help (\(Self.kernelSizedEligibleLayers) layers): c(8) "), measured.summary)

        // The GPU refused: nothing runs.
        await engine.updateConfiguration { $0.hooks = EngineHooks(beginGPU: { false }, endGPU: {}, isAllowed: { true }) }
        do {
            _ = try await fastKernels.measure(engine)
            XCTFail("measured without the GPU")
        } catch {
            XCTAssertEqual(error as? EngineError, .leftForeground)
        }
    }

    func testTheExtensionReportsUnavailableOffTheGPU() throws {
        let fastKernels = FastKernelsExtension()
        let (engine, model) = try Self.makeEngine(seed: 4113, fastKernels: fastKernels, requested: true)
        // The extension decides on MLX's default device when it prepares (normally on the
        // engine queue, inside `warmUp()`); nothing else uses the engine here.
        XCTAssertThrowsError(try Device.withDefaultDevice(.cpu) { try fastKernels.prepare(engine) }) { error in
            guard case .unavailable = error as? FastKernelsError else {
                return XCTFail("unexpected error \(error)")
            }
        }
        XCTAssertEqual(fastKernels.report?.outcome, .unavailable)
        XCTAssertEqual(Self.fastLayerCount(model), 0)
    }

    // MARK: Helpers

    /// A Qwen3.5 hybrid (one gated-delta layer, one attention layer) whose every projection the
    /// kernel handles: K is 512 or 1024 and N a multiple of 4.
    static let kernelSizedHybridJSON = """
        {
          "model_type": "qwen3_5_text",
          "hidden_size": 512,
          "num_hidden_layers": 2,
          "full_attention_interval": 2,
          "intermediate_size": 1024,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 128,
          "linear_num_key_heads": 4,
          "linear_num_value_heads": 8,
          "linear_key_head_dim": 64,
          "linear_value_head_dim": 64,
          "linear_conv_kernel_dim": 4,
          "rms_norm_eps": 1e-6,
          "vocab_size": 128,
          "max_position_embeddings": 4096,
          "tie_word_embeddings": true,
          "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25
          }
        }
        """

    /// Gated-delta layer: in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj; attention
    /// layer: q, k, v, o; two MLPs: gate, up, down.
    static let kernelSizedEligibleLayers = 5 + 4 + 2 * 3

    /// The kernel-sized hybrid on the engine's fork, sharpened and quantized to 4 bits.
    static func makeKernelSizedHybrid(seed: UInt64) throws -> HybridQwen35TextModel {
        let configuration = try JSONDecoder.json5().decode(HybridQwen35TextConfiguration.self, from: Data(kernelSizedHybridJSON.utf8))
        MLXRandom.seed(seed)
        let model = HybridQwen35TextModel(configuration)
        eval(model)
        TinyModels.sharpen(model)
        TinyModels.quantize4bit(model)
        return model
    }

    static func makeEngine(seed: UInt64, fastKernels: FastKernelsExtension, requested: Bool) throws -> (InferenceEngine, HybridQwen35TextModel) {
        let model = try makeKernelSizedHybrid(seed: seed)
        let loaded = try EngineTestHarness.loadedModel(model, id: "tiny-kernel-\(seed)", modelType: "qwen3_5_text")
        var configuration = EngineTestHarness.testConfiguration(maxTokens: 24)
        configuration.extensions = [fastKernels]
        configuration.fastKernelsRequested = requested
        return (InferenceEngine(loaded: loaded, configuration: configuration), model)
    }

    static func fastLayerCount(_ model: Module) -> Int {
        model.leafModules().flattened().filter { $0.1 is FastQuantizedLinear }.count
    }

    static func layerIdentities(_ model: Module) -> [String: ObjectIdentifier] {
        var identities: [String: ObjectIdentifier] = [:]
        for (path, module) in model.leafModules().flattened() {
            identities[path] = ObjectIdentifier(module)
        }
        return identities
    }

    /// The logits of a 4-token forward on a scratch extension of the session (`[4, V]`).
    static func nextLogits(_ engine: InferenceEngine) async throws -> MLXArray {
        try await engine.withSession { session in
            session.withScratch { () -> MLXArray in
                let logits = session.feed([65, 66, 67, 68], rows: .all).logits!.asType(.float32)
                eval(logits)
                return logits
            }
        }
    }
}

/// A block with one projection, set by property key.
final class KernelTestBlock: Module {
    @ModuleInfo(key: "proj") var proj: Linear

    init(_ proj: Linear) {
        self._proj.wrappedValue = proj
        super.init()
    }
}

/// Blocks in an array, plus projections held directly in an array property.
final class KernelTestStack: Module {
    let blocks: [KernelTestBlock]
    @ModuleInfo(key: "heads") var heads: [Linear]

    init(blocks: [KernelTestBlock], heads: [Linear]) {
        self.blocks = blocks
        self._heads.wrappedValue = heads
        super.init()
    }
}

/// Speculates on every token with `draftLength` drafts from `RepeatDrafter`, so every round
/// verifies `draftLength + 1` rows.
final class ForcedDraftFactory: GeneratorFactory, @unchecked Sendable {
    let draftLength: Int

    init(draftLength: Int) {
        self.draftLength = draftLength
    }

    func makeGenerator(_ context: GeneratorContext) -> TokenGenerator {
        let drafter = RepeatDrafter()
        drafter.reset(ledger: context.session.ledger, request: context.request)
        return SpeculativeLoop(
            context, drafters: [drafter], policy: SharedDraftPolicy(curve: .stockMLXDefault, maxDraft: 8),
            options: SpeculativeLoop.Options(forcedDraftLength: draftLength))
    }
}

/// Guesses that the last token repeats (a printable token when the last one is special):
/// mostly wrong, which is fine, since verification keeps the output exact either way.
final class RepeatDrafter: Drafter {
    var source: DraftSource { .mtp }
    var costPerToken: Double { 0 }
    var wantsHidden: Bool { false }

    func reset(ledger: [Int], request: EngineRequest) {}

    func propose(context: ArraySlice<Int>, maxTokens: Int) -> DraftProposal? {
        guard maxTokens > 0 else { return nil }
        let last = context.last ?? 65
        let token = last >= 32 && last < FakeChatMLTokenizer.vocabularySize ? last : 65
        return DraftProposal(tokens: Array(repeating: token, count: maxTokens), source: .mtp, matchLength: 4)
    }

    func observe(_ round: RoundObservation) {}
}
