import AssistantKit
import Foundation
import MLX
import MLXNN

extension EngineConfiguration {
    /// Whether the app asks for the small-M kernel (`AssistantSettings.localFastKernels`; plan
    /// WP41), for a `FastKernelsExtension` with the trigger `.flag` (the default). The extension
    /// reads it when it prepares (in `warmUp()`), so a change takes effect at the next load; the
    /// app reloads the model when the setting changes.
    ///
    /// It is kept as a marker in `extensions` (a value, like every other field), so set it
    /// **after** assigning `extensions`: assigning a new array drops it. In the app's
    /// `EngineSetup.configuration(settings:host:)` that is the line
    /// `configuration.fastKernelsRequested = settings.localFastKernels` right after
    /// `configuration.extensions = extensions(for: settings)`. An app that instead adds the
    /// extension only while the setting is on uses the trigger `.presence` and needs no flag.
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
        /// The trigger is `.flag` and `fastKernelsRequested` is off: nothing was measured or
        /// swapped.
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
    /// Whether the fast layers are in the model afterwards: true after `prepare` only with
    /// `.faster`; `measure` leaves the layers as it found them. `kernelsActive` decides whether
    /// they use the kernel.
    public var kernelsInstalled: Bool
    /// Whether this came from `measure` (a dry run) rather than `prepare`.
    public var measuredOnly: Bool
    public var selfTest: KernelSelfTest.Result?
    /// The cost curve with the stock layers, and with the fast ones.
    public var before: CostCurve?
    public var after: CostCurve?
    /// The fraction of c(8) the kernels saved: `1 − c(8) after / c(8) before`, from the relative
    /// costs (one row costs 1). 0.25 means an 8-row forward costs a quarter less.
    public var gain: Double?
    /// The same comparison as a speed-up: `c(8) before / c(8) after` (1.33 means an 8-row
    /// forward runs 1.33× as fast). This is the value the app stores as its kernel gain
    /// (`LocalModelHost.recordKernelGain`, "c(8) without ÷ with") and compares with
    /// `requiredSpeedup`.
    public var speedup: Double?
    /// The `gain` the kernels need to be kept.
    public var requiredGain: Double
    public var detail: String

    /// `requiredGain` as a speed-up: `1 / (1 − requiredGain)`, 1.33 for the default 25%.
    public var requiredSpeedup: Double { FastKernelsExtension.speedup(forGain: requiredGain) }

    /// Whether the fast layers are in the model.
    public var isEnabled: Bool { kernelsInstalled }

    /// One line for the app's engine summary: the outcome and the before/after c(8).
    public var summary: String {
        let comparison: String? = {
            guard let before, let after, let gain, let speedup else { return nil }
            let width = FastKernelsExtension.decisionWidth
            return "c(\(width)) "
                + String(
                    format: "%.2f× → %.2f× one row (%+.0f%%, %.2f× as fast; needs %+.0f%%, %.2f×)", before.relative(width),
                    after.relative(width), -gain * 100, speedup, -requiredGain * 100, requiredSpeedup)
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
/// 1. Unless the kernels are asked for (see `trigger`) it does nothing.
/// 2. It runs `KernelSelfTest` (float32 and the model's scale dtypes). Any mismatch or MLX error
///    keeps the stock layers (and `prepare` throws, which the engine logs).
/// 3. It swaps every eligible `QuantizedLinear` for a `FastQuantizedLinear` (via `leafModules()`
///    and `update(modules:)`), measures the cost curve with `CostProbe` while they run the stock
///    product, compiles the kernel for every row count, and measures again on the kernel.
/// 4. It keeps the swap only when c(8) improved by at least `requiredGain` (25%, a 1.33×
///    speed-up); otherwise it puts the original layers back.
///
/// `measure(engine)` does steps 2 and 3 at any time, whatever the trigger says, and leaves the
/// layers as it found them: a settings screen can tell whether the kernel would help, and store
/// `FastKernelsReport.speedup`.
///
/// **What asks for the kernels** (`trigger`): by default `EngineConfiguration.fastKernelsRequested`,
/// which the app sets from `localFastKernels`; with `.presence`, the extension being in the
/// engine's `extensions` at load (for an app that adds it only while `localFastKernels` is on).
///
/// **Every path over the model.** The fast layers replace layers of `engine.target.model`, the
/// same object MLX's `ChatSession` runs through `engine.loaded.container`. That path's one-row
/// decode and longer prefills still run the stock code (`super`), but its speculative verify
/// passes and prefills of 2–9 tokens use the kernel. A host that wants that path stock (a fallback, or the stock side of
/// an engine comparison) sets `kernelsActive = false` for it and back to true for engine replies:
/// the switch is read at every forward and never touches the model's modules.
///
/// **When it touches the model.** `prepare` swaps modules on the engine queue in `warmUp()`,
/// before the host serves any reply. `measure` swaps the fast layers in and back out when the
/// model has the stock ones (a model that already runs the fast layers isn't touched): the
/// engine queue serializes that with engine replies, but not with a `ChatSession` reply, so the
/// host calls it only while no stock-path reply runs.
///
/// The verdict, the self-test and both curves are in `report`; `summary` is the line for the
/// app's engine summary. One instance may serve several engines in turn (the app installs it once
/// in `EngineSetup.extraExtensions`); `report` describes the latest `prepare`.
public final class FastKernelsExtension: EngineExtension, @unchecked Sendable {
    /// What asks for the kernels.
    public enum Trigger: Sendable, Equatable {
        /// `EngineConfiguration.fastKernelsRequested` (the default): the extension can stay in
        /// `extensions` for good and does nothing (not even a measurement) while the flag is off.
        case flag
        /// Being in the engine's `extensions` when it warms up: for an app that adds the
        /// extension only while `localFastKernels` is on (as `EngineSetup.wantsExtension` can).
        case presence
    }

    public let name = "fast-kernels"
    /// The row count whose cost decides: a verify pass of 7 drafts plus the bonus row.
    public static let decisionWidth = 8
    /// The plan's threshold: c(8) must cost at least 25% less with the kernels.
    public static let defaultRequiredGain = 0.25
    /// `defaultRequiredGain` as a speed-up of c(8): 1 / 0.75 ≈ 1.33. The app's threshold for
    /// a stored `FastKernelsReport.speedup`.
    public static let defaultRequiredSpeedup = speedup(forGain: defaultRequiredGain)
    /// The c(8) improvement needed to keep the kernels.
    public let requiredGain: Double
    /// The widths both probes measure.
    public let probeWidths: [Int]
    public let trigger: Trigger
    /// The shapes the self-test checks (tests use small ones).
    var selfTestShapes = KernelSelfTest.defaultShapes
    /// Tests: makes the self-test see wrong kernel products.
    var debugBreakSelfTest = false
    /// Shared by every fast layer this instance puts in.
    let kernelSwitch = FastKernelSwitch()

    private let lock = NSLock()
    private var latest: FastKernelsReport?

    public init(
        requiredGain: Double = FastKernelsExtension.defaultRequiredGain, probeWidths: [Int] = [1, 2, 3, 4, 6, 8],
        trigger: Trigger = .flag
    ) {
        precondition(probeWidths.contains(1) && probeWidths.contains(Self.decisionWidth), "The probes must measure widths 1 and 8.")
        self.requiredGain = requiredGain
        self.probeWidths = probeWidths
        self.trigger = trigger
    }

    /// Whether the fast layers this instance put in use the kernel (true, the default) or the
    /// stock product, bitwise the stock layers. Safe to set from any thread at any time; it
    /// applies from the next forward and touches no module. `prepare` and `measure` leave it as
    /// they found it (and still test and time the kernel).
    public var kernelsActive: Bool {
        get { kernelSwitch.isOn }
        set { kernelSwitch.isOn = newValue }
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
        guard trigger == .presence || engine.configuration.fastKernelsRequested else { return }
        try evaluate(engine, keepIfFaster: true, report: &report)
    }

    /// The self-test and the before/after probes on `engine` now, whatever the trigger says,
    /// leaving the model's layers (and `kernelsActive`) as they were. Runs on the engine queue
    /// between other jobs, under the engine's GPU hooks; throws `EngineError.leftForeground` when
    /// the GPU is refused. Every other problem is in the returned report. `report` is not
    /// changed.
    ///
    /// Call it only while no `ChatSession` reply runs over `engine.loaded.container`: unless the
    /// fast layers are already in, it swaps them into that same model and back out.
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
        let pairs = Self.layerPairs(in: model, kernelSwitch: kernelSwitch)
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

        // Both probes run the fast layers, first switched to the stock product and then to the
        // kernel, so a model that already runs them isn't touched.
        let filler = CostProbe.fillerTokens(session: engine.session, renderer: engine.loaded.renderer)
        let startsStock = pairs.contains { !$0.startsFast }
        let switches = Self.switches(of: pairs)
        let switchStates = switches.map(\.isOn)
        do {
            try withError {
                switches.forEach { $0.isOn = false }
                if startsStock {
                    Self.install(pairs, fast: true, in: model)
                }
                let before = try CostProbe.measure(
                    session: engine.session, filler: filler, widths: probeWidths, isAllowed: hooks.isAllowed)
                report.before = before

                switches.forEach { $0.isOn = true }
                try Self.compileKernels(pairs.map(\.fast), isAllowed: hooks.isAllowed)
                let after = try CostProbe.measure(
                    session: engine.session, filler: filler, widths: probeWidths, isAllowed: hooks.isAllowed)
                report.after = after

                let gain = Self.gain(before: before, after: after)
                report.gain = gain
                report.speedup = Self.speedup(before: before, after: after)
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
        for (layerSwitch, isOn) in zip(switches, switchStates) {
            layerSwitch.isOn = isOn
        }

        // Where the layers end.
        if keepIfFaster {
            let keep = report.outcome == .faster
            Self.install(pairs, fast: keep, in: model)
            report.kernelsInstalled = keep
        } else if startsStock {
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
    /// that the kernel handles pairs with a new `FastQuantizedLinear` over its arrays, reading
    /// `kernelSwitch`; a layer already swapped pairs with a new stock layer over the same arrays.
    /// Layers held in an array property (a numeric last path component) are left alone:
    /// replacing one element of an array through `update(modules:)` would replace the whole
    /// array.
    static func layerPairs(in model: Module, kernelSwitch: FastKernelSwitch = FastKernelSwitch()) -> [LayerPair] {
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
            return LayerPair(
                path: path, stock: layer, fast: FastQuantizedLinear(layer, kernelSwitch: kernelSwitch), startsFast: false)
        }
    }

    /// The distinct switches the fast layers of `pairs` read (one, unless some layers were put in
    /// by another instance).
    static func switches(of pairs: [LayerPair]) -> [FastKernelSwitch] {
        var switches: [FastKernelSwitch] = []
        for pair in pairs where !switches.contains(where: { $0 === pair.fast.kernelSwitch }) {
            switches.append(pair.fast.kernelSwitch)
        }
        return switches
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

    /// `1 − c(8) after / c(8) before`, from the relative costs; 0 without a usable c(8).
    public static func gain(before: CostCurve, after: CostCurve) -> Double {
        let previous = before.relative(decisionWidth)
        let current = after.relative(decisionWidth)
        guard previous > 0, current > 0 else { return 0 }
        return 1 - current / previous
    }

    /// `c(8) before / c(8) after`, from the relative costs; 1 without a usable c(8). Reaches
    /// `speedup(forGain: g)` exactly when `gain` reaches `g` (up to rounding).
    public static func speedup(before: CostCurve, after: CostCurve) -> Double {
        let previous = before.relative(decisionWidth)
        let current = after.relative(decisionWidth)
        guard previous > 0, current > 0 else { return 1 }
        return previous / current
    }

    /// The speed-up a c(8) `gain` (below 1) amounts to: `1 / (1 − gain)`.
    public static func speedup(forGain gain: Double) -> Double {
        1 / (1 - gain)
    }

    private func emptyReport(_ engine: InferenceEngine, measuredOnly: Bool) -> FastKernelsReport {
        FastKernelsReport(
            outcome: .notRequested, modelID: engine.info.modelID, layers: 0, kernelsInstalled: false,
            measuredOnly: measuredOnly, selfTest: nil, before: nil, after: nil, gain: nil, speedup: nil,
            requiredGain: requiredGain, detail: "")
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
