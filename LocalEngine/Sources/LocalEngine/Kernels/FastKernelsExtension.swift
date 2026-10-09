import AssistantKit
import Foundation
import MLX
import MLXNN

extension EngineConfiguration {
    /// Whether the app asks for the small-M kernel (`AssistantSettings.localFastKernels`; plan
    /// WP41). `FastKernelsExtension` reads it when it prepares (in `warmUp()`), so a change takes
    /// effect at the next load; the app reloads the model when the setting changes.
    ///
    /// It is kept as a marker in `extensions` (a value, like every other field), so set it
    /// **after** assigning `extensions`: assigning a new array drops it.
    public var fastKernelsRequested: Bool {
        get { extensions.contains { $0 is FastKernelsRequest } }
        set {
            extensions.removeAll { $0 is FastKernelsRequest }
            if newValue {
                extensions.append(FastKernelsRequest())
            }
        }
    }
}

/// The marker behind `EngineConfiguration.fastKernelsRequested`: an extension that does nothing.
final class FastKernelsRequest: EngineExtension {
    let name = "fast-kernels-request"

    func prepare(_ engine: InferenceEngine) throws {}

    func drafters(for request: EngineRequest, engine: InferenceEngine) -> [any Drafter] { [] }
}

/// What `FastKernelsExtension` found for one engine, with the cost curves it measured.
public struct FastKernelsReport: Sendable, Equatable {
    public enum Outcome: String, Sendable {
        /// `fastKernelsRequested` is off: nothing was measured or swapped.
        case notRequested
        /// Custom Metal kernels can't run here (not a Metal platform, or the default device isn't
        /// the GPU).
        case unavailable
        /// The model has no 4-bit layer the kernel handles.
        case noEligibleLayers
        /// The kernel disagreed with the stock product, or MLX raised an error.
        case selfTestFailed
        /// The probe measured at least the required c(8) gain.
        case faster
        /// The probe measured less than the required gain.
        case notFaster
        /// The probe or the swap failed (for example the app left the foreground).
        case failed
    }

    public var outcome: Outcome
    public var modelID: String
    /// Layers the kernel handles.
    public var layers: Int
    /// Whether the model runs the fast layers afterwards: true after `prepare` only with
    /// `.faster`; `measure` leaves the layers as it found them.
    public var kernelsInstalled: Bool
    /// Whether this came from `measure` (a dry run) rather than `prepare`.
    public var measuredOnly: Bool
    public var selfTest: KernelSelfTest.Result?
    /// The cost curve with the stock layers, and with the fast ones.
    public var before: CostCurve?
    public var after: CostCurve?
    /// `1 − c(8) after / c(8) before`, from the relative costs (one row costs 1).
    public var gain: Double?
    public var requiredGain: Double
    public var detail: String

    /// Whether the fast layers are in the model.
    public var isEnabled: Bool { kernelsInstalled }

    /// One line for the app's engine summary: the outcome and the before/after c(8).
    public var summary: String {
        let comparison: String? = {
            guard let before, let after, let gain else { return nil }
            let width = FastKernelsExtension.decisionWidth
            return "c(\(width)) "
                + String(
                    format: "%.2f× → %.2f× one row (%+.0f%%, needs %+.0f%%)", before.relative(width), after.relative(width),
                    -gain * 100, -requiredGain * 100)
        }()
        switch outcome {
        case .notRequested:
            return "Fast kernels off (not requested)"
        case .unavailable:
            return "Fast kernels unavailable: \(detail)"
        case .noEligibleLayers:
            return "Fast kernels off: no layer the kernel handles"
        case .selfTestFailed:
            return "Fast kernels off: self-test failed (\(detail))"
        case .faster:
            var text = kernelsInstalled ? "Fast kernels on (\(layers) layers)" : "Fast kernels would help (\(layers) layers)"
            if let comparison { text += ": " + comparison }
            if let selfTest { text += "; self-test \(selfTest.cases) products in " + String(format: "%.1f s", selfTest.seconds) }
            return text
        case .notFaster:
            let prefix = kernelsInstalled ? "Fast kernels on, but not faster now: " : "Fast kernels off: "
            return prefix + (comparison ?? detail)
        case .failed:
            return "Fast kernels off: \(detail)"
        }
    }
}

/// Puts the small-M kernel into the model when it is asked for, correct here and measurably
/// faster (plan WP41 instruction 4).
///
/// `prepare(engine)` runs once per engine, on the engine queue (from `warmUp()`):
/// 1. Without `EngineConfiguration.fastKernelsRequested` it does nothing.
/// 2. It runs `KernelSelfTest` (float32 and the model's scale dtypes). Any mismatch or MLX error
///    keeps the stock layers (and `prepare` throws, which the engine logs).
/// 3. It measures the cost curve with `CostProbe`, swaps every eligible `QuantizedLinear` for a
///    `FastQuantizedLinear` (via `leafModules()` and `update(modules:)`), compiles the kernel for
///    every row count, and measures again.
/// 4. It keeps the swap only when c(8) improved by at least `requiredGain` (25%); otherwise it
///    puts the original layers back.
///
/// `measure(engine)` does steps 2 and 3 at any time, whatever the flag says, and leaves the
/// layers as it found them: a settings screen can tell whether the kernel would help.
///
/// The verdict, the self-test and both curves are in `report`; `summary` is the line for the
/// app's engine summary. One instance may serve several engines in turn (the app installs it once
/// in `EngineSetup.extraExtensions`); `report` describes the latest `prepare`.
public final class FastKernelsExtension: EngineExtension, @unchecked Sendable {
    public let name = "fast-kernels"
    /// The row count whose cost decides: a verify pass of 7 drafts plus the bonus row.
    public static let decisionWidth = 8
    /// The c(8) improvement needed to keep the kernels.
    public let requiredGain: Double
    /// The widths both probes measure.
    public let probeWidths: [Int]
    /// The shapes the self-test checks (tests use small ones).
    var selfTestShapes = KernelSelfTest.defaultShapes
    /// Tests: makes the self-test see wrong kernel products.
    var debugBreakSelfTest = false

    private let lock = NSLock()
    private var latest: FastKernelsReport?

    public init(requiredGain: Double = 0.25, probeWidths: [Int] = [1, 2, 3, 4, 6, 8]) {
        precondition(probeWidths.contains(1) && probeWidths.contains(Self.decisionWidth), "The probes must measure widths 1 and 8.")
        self.requiredGain = requiredGain
        self.probeWidths = probeWidths
    }

    /// The verdict of the latest `prepare`; nil before any.
    public var report: FastKernelsReport? {
        lock.lock()
        defer { lock.unlock() }
        return latest
    }

    /// One line for the app's engine summary.
    public var summary: String {
        report?.summary ?? "Fast kernels: not prepared yet"
    }

    /// Whether the latest `prepare` kept the kernels.
    public var isEnabled: Bool { report?.isEnabled ?? false }

    public func prepare(_ engine: InferenceEngine) throws {
        var report = emptyReport(engine, measuredOnly: false)
        defer { record(report) }
        guard engine.configuration.fastKernelsRequested else { return }
        try evaluate(engine, keepIfFaster: true, report: &report)
    }

    /// The self-test and the before/after probes on `engine` now, whatever
    /// `fastKernelsRequested` says, leaving the model's layers as they were. Runs on the engine
    /// queue between other jobs, under the engine's GPU hooks; throws
    /// `EngineError.leftForeground` when the GPU is refused. Every other problem is in the
    /// returned report. `report` is not changed.
    public func measure(_ engine: InferenceEngine) async throws -> FastKernelsReport {
        try await engine.onQueue { [self] in
            let hooks = engine.configuration.hooks
            guard hooks.beginGPU() else { throw EngineError.leftForeground }
            defer { hooks.endGPU() }
            var report = self.emptyReport(engine, measuredOnly: true)
            do {
                try self.evaluate(engine, keepIfFaster: false, report: &report)
            } catch {
                // The report says what happened.
            }
            return report
        }
    }

    public func drafters(for request: EngineRequest, engine: InferenceEngine) -> [any Drafter] { [] }

    // MARK: Evaluating

    /// Steps 2–4 of the type's documentation, on the engine queue. With `keepIfFaster` the fast
    /// layers stay exactly when the gain is reached; without it the layers end as they started.
    /// Fills `report`; throws after recording an outcome that keeps the stock layers.
    private func evaluate(_ engine: InferenceEngine, keepIfFaster: Bool, report: inout FastKernelsReport) throws {
        guard SmallMQuantizedMatmul.canRunOnDefaultDevice else {
            report.outcome = .unavailable
            report.detail = SmallMQuantizedMatmul.isAvailable
                ? "MLX's default device is not the GPU" : "custom Metal kernels don't exist on this platform"
            throw FastKernelsError.unavailable(report.detail)
        }

        let model = engine.target.model
        let pairs = Self.layerPairs(in: model)
        report.layers = pairs.count
        report.kernelsInstalled = pairs.contains { $0.startsFast }
        guard !pairs.isEmpty else {
            report.outcome = .noEligibleLayers
            report.detail = "no 4-bit, group-64 layer with K a multiple of 512"
            return
        }

        let hooks = engine.configuration.hooks
        var dtypes: [DType] = [.float32]
        for pair in pairs where !dtypes.contains(pair.stock.scales.dtype) {
            dtypes.append(pair.stock.scales.dtype)
        }
        var options = KernelSelfTest.Options(shapes: selfTestShapes, dtypes: dtypes)
        options.debugBreakKernel = debugBreakSelfTest
        let selfTest = KernelSelfTest.run(options: options, isAllowed: hooks.isAllowed)
        report.selfTest = selfTest
        guard selfTest.passed else {
            report.outcome = .selfTestFailed
            report.detail = selfTest.detail
            if keepIfFaster {
                Self.install(pairs, fast: false, in: model)
                report.kernelsInstalled = false
            }
            throw FastKernelsError.selfTestFailed(selfTest.detail)
        }

        let filler = CostProbe.fillerTokens(session: engine.session, renderer: engine.loaded.renderer)
        do {
            try withError {
                Self.install(pairs, fast: false, in: model)
                let before = try CostProbe.measure(
                    session: engine.session, filler: filler, widths: probeWidths, isAllowed: hooks.isAllowed)
                report.before = before

                Self.install(pairs, fast: true, in: model)
                try Self.compileKernels(pairs.map(\.fast), isAllowed: hooks.isAllowed)
                let after = try CostProbe.measure(
                    session: engine.session, filler: filler, widths: probeWidths, isAllowed: hooks.isAllowed)
                report.after = after

                let gain = Self.gain(before: before, after: after)
                report.gain = gain
                let faster = gain >= requiredGain
                report.outcome = faster ? .faster : .notFaster
                if !faster {
                    report.detail = String(format: "c(8) improved by %.0f%%, below %.0f%%", gain * 100, requiredGain * 100)
                }
            }
        } catch {
            report.outcome = .failed
            report.detail = (error as? LocalizedError)?.errorDescription ?? "\(error)"
        }

        // Where the layers end.
        if keepIfFaster {
            let keep = report.outcome == .faster
            Self.install(pairs, fast: keep, in: model)
            report.kernelsInstalled = keep
        } else {
            Self.restore(pairs, in: model)
        }
        Memory.clearCache()
        if report.outcome == .failed {
            throw FastKernelsError.failed(report.detail)
        }
    }

    // MARK: Swapping

    /// One eligible layer: its stock form, its fast form, and which of them the model had.
    struct LayerPair {
        let path: String
        let stock: QuantizedLinear
        let fast: FastQuantizedLinear
        let startsFast: Bool
    }

    /// The model's eligible layers. A stock `QuantizedLinear` (exactly that class, not a subclass)
    /// that the kernel handles pairs with a new `FastQuantizedLinear` over its arrays; a layer
    /// already swapped pairs with a new stock layer over the same arrays. Layers held in an array
    /// property (a numeric last path component) are left alone: replacing one element of an
    /// array through `update(modules:)` would replace the whole array.
    static func layerPairs(in model: Module) -> [LayerPair] {
        model.leafModules().flattened().compactMap { (path, module) -> LayerPair? in
            guard Int(split(path).key) == nil else { return nil }
            if let fast = module as? FastQuantizedLinear {
                let stock = QuantizedLinear(
                    weight: fast.weight, bias: fast.bias, scales: fast.scales, biases: fast.biases,
                    groupSize: fast.groupSize, bits: fast.bits, mode: fast.mode)
                stock.freeze()
                return LayerPair(path: path, stock: stock, fast: fast, startsFast: true)
            }
            guard type(of: module) == QuantizedLinear.self, let layer = module as? QuantizedLinear,
                  FastQuantizedLinear.isEligible(layer)
            else { return nil }
            return LayerPair(path: path, stock: layer, fast: FastQuantizedLinear(layer), startsFast: false)
        }
    }

    /// The stock `QuantizedLinear` layers of `model` the kernel handles, with their paths.
    static func eligibleLayers(in model: Module) -> [(path: String, layer: QuantizedLinear)] {
        layerPairs(in: model).filter { !$0.startsFast }.map { (path: $0.path, layer: $0.stock) }
    }

    /// Puts the fast (or the stock) layer of every pair at its path.
    static func install(_ pairs: [LayerPair], fast: Bool, in model: Module) {
        install(pairs.map { ($0.path, fast ? $0.fast as QuantizedLinear : $0.stock) }, in: model)
    }

    /// Puts back the layer each pair started with.
    static func restore(_ pairs: [LayerPair], in model: Module) {
        install(pairs.map { ($0.path, $0.startsFast ? $0.fast as QuantizedLinear : $0.stock) }, in: model)
    }

    /// Puts `layers` at their paths in `model`.
    ///
    /// Each layer is set on its parent module, by property key. One `update(modules:)` on the
    /// root with all paths would need an entry for every element of every module array on the
    /// way (`update` refuses an array with gaps), which holds only when every decoder layer has
    /// a layer to swap.
    static func install(_ layers: [(String, QuantizedLinear)], in model: Module) {
        guard !layers.isEmpty else { return }
        let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        var children: [String: [(String, Module)]] = [:]
        for (path, layer) in layers {
            let (parent, key) = split(path)
            children[parent, default: []].append((key, layer))
        }
        for (path, updates) in children {
            guard let parent = path.isEmpty ? model : modules[path] else {
                preconditionFailure("The model has no module at \(path).")
            }
            parent.update(modules: ModuleChildren.unflattened(updates))
        }
    }

    /// `a.b.c` → (`a.b`, `c`); `c` → (``, `c`).
    static func split(_ path: String) -> (parent: String, key: String) {
        guard let dot = path.lastIndex(of: ".") else { return ("", path) }
        return (String(path[..<dot]), String(path[path.index(after: dot)...]))
    }

    /// Runs every kernel instance the engine can need (each K and dtype, every row count) once,
    /// so no reply pays for compiling one.
    static func compileKernels(_ layers: [FastQuantizedLinear], isAllowed: () -> Bool) throws {
        var seen: [String] = []
        for layer in layers {
            let k = layer.weight.dim(1) * 32 / layer.bits
            let dtype = layer.scales.dtype
            let key = "\(k) \(dtype)"
            guard !seen.contains(key) else { continue }
            seen.append(key)
            for rows in SmallMQuantizedMatmul.rows {
                guard isAllowed() else { throw EngineError.leftForeground }
                try checkedEval(layer(MLXArray.zeros([1, rows, k], dtype: dtype)))
            }
        }
    }

    /// `1 − c(8) after / c(8) before`, from the relative costs.
    static func gain(before: CostCurve, after: CostCurve) -> Double {
        let previous = before.relative(decisionWidth)
        guard previous > 0 else { return 0 }
        return 1 - after.relative(decisionWidth) / previous
    }

    private func emptyReport(_ engine: InferenceEngine, measuredOnly: Bool) -> FastKernelsReport {
        FastKernelsReport(
            outcome: .notRequested, modelID: engine.info.modelID, layers: 0, kernelsInstalled: false,
            measuredOnly: measuredOnly, selfTest: nil, before: nil, after: nil, gain: nil, requiredGain: requiredGain,
            detail: "")
    }

    private func record(_ report: FastKernelsReport) {
        lock.lock()
        latest = report
        lock.unlock()
    }
}

public enum FastKernelsError: Error, Equatable, CustomStringConvertible {
    case unavailable(String)
    case selfTestFailed(String)
    case failed(String)

    public var description: String {
        switch self {
        case .unavailable(let detail): return "fast kernels unavailable: \(detail)"
        case .selfTestFailed(let detail): return "kernel self-test failed: \(detail)"
        case .failed(let detail): return "fast kernels failed: \(detail)"
        }
    }
}
