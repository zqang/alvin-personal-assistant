import AssistantKit
import Darwin
import HuggingFace
import LocalEngine
import MiniCPM
import MLX
import Observation
import SwiftUI
import UIKit

enum ModelComparisonVariant: String, Encodable {
    case dense1B, moe35B

    var title: String {
        self == .dense1B ? "MiniCPM-V 4.7 1B" : "MiniCPM-V 4.7 35B-A3B · LES"
    }
    var modelID: String {
        "tc-mb/MiniCPM-V-Apps-gguf/" + (self == .dense1B ? "MiniCPM-V-4.7-1B" : "MiniCPM-V-4.7-35B-A3B")
    }
    var reportDirectory: String { self == .dense1B ? "ModelComparison" : "ModelComparison35B" }
    var hostModel: MiniCPMHost.Model { self == .dense1B ? .photoReader : .accurate }
}

/// One MiniCPM arm of a comparison: an execution profile, with or without the DSpark draft.
/// `--minicpm-profile targetOnly` runs the 35B without the draft (`baseline` execution); a name
/// from `MiniCPMRuntimeConfiguration.profiles` runs that profile with the draft.
struct MiniCPMArm: Encodable, Equatable {
    let name: String
    let configuration: MiniCPMRuntimeConfiguration
    let draft: Bool

    /// What the app runs (decision 13): without the flag, the comparison measures this.
    static var production: Self {
        let (configuration, draft) = MiniCPMRuntimeConfiguration.production
        return Self(name: draft ? configuration.name : "targetOnly", configuration: configuration, draft: draft)
    }

    static func named(_ name: String) -> Self? {
        if name == "targetOnly" { return Self(name: name, configuration: .baseline, draft: false) }
        return MiniCPMRuntimeConfiguration.profiles.first { $0.name == name }.map { Self(name: $0.name, configuration: $0, draft: true) }
    }
}

/// The process's memory, as both benchmark screens sample it.
struct ProcessMemory: Equatable {
    /// `phys_footprint`: what iOS counts against the app. llama.cpp's memory-mapped weights aren't in it.
    var footprint: UInt64 = 0
    /// `resident_size`: pages in RAM, the mapped weights that are paged in included.
    var resident: UInt64 = 0

    static var now: Self {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Self(footprint: info.phys_footprint, resident: info.resident_size) : Self()
    }

    mutating func raise(to sample: Self) {
        footprint = max(footprint, sample.footprint)
        resident = max(resident, sample.resident)
    }
}

/// Waits while the app's MiniCPM host downloads or verifies `model`'s files (it checks them at
/// launch), so a benchmark never writes the same folder and finds them verified.
// ponytail: polls once a second; observation tracking is the upgrade if the wait ever matters.
@MainActor
func waitForMiniCPMFiles(_ model: MiniCPMHost.Model, progress: (String) -> Void) async throws {
    while true {
        switch MiniCPMHost.shared.status[model] {
        case .downloading?, .verifying?:
            progress("Waiting for the app to finish checking MiniCPM's files…")
            try await Task.sleep(for: .seconds(1))
        default:
            return
        }
    }
}

/// A MiniCPM stream as the app sees it: first text is when this caller receives the first visible
/// piece, the same criterion as the current model's (the runtime's own time stays in
/// `result.nativeFirstTextSeconds`).
struct MiniCPMStream {
    let text: String
    let firstTextSeconds: Double?
    let totalSeconds: Double
    /// Tokens prefilled, and tokens the cache already held.
    let fed: Int
    let reused: Int
    let result: MiniCPMBenchmarkResult

    static func measure(_ events: () -> AsyncThrowingStream<MiniCPMEvent, Error>) async throws -> Self {
        let started = ContinuousClock.now
        var text = ""
        var first: Double?
        var prefilled = (fed: 0, reused: 0)
        for try await event in events() {
            switch event {
            case .prefilled(let fed, let reused): prefilled = (fed, reused)
            case .firstText: break
            case .text(let piece):
                if first == nil, isVisible(piece) { first = seconds(started.duration(to: .now)) }
                text += piece
            case .finished(let result):
                return Self(text: text, firstTextSeconds: first, totalSeconds: seconds(started.duration(to: .now)),
                            fed: prefilled.fed, reused: prefilled.reused, result: result)
            }
        }
        // Only a cancelled consumer sees the stream end without its result.
        throw CancellationError()
    }

    /// Anything beyond ASCII whitespace and DEL, as the native runtime counts visible text.
    static func isVisible(_ text: String) -> Bool {
        text.unicodeScalars.contains { $0.value > 0x20 && $0.value != 0x7F }
    }

    static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }
}

/// A launch-only probe used before transferring large benchmark bundles.
struct BenchmarkDeviceProbeView: View {
    @State private var summary = "Checking available iPhone storage…"

    var body: some View {
        Text(summary).padding().task {
            do {
                let values = try URL.applicationSupportDirectory.resourceValues(forKeys: [
                    .volumeAvailableCapacityKey, .volumeAvailableCapacityForImportantUsageKey, .volumeTotalCapacityKey
                ])
                let result: [String: Any] = [
                    "systemVersion": UIDevice.current.systemVersion,
                    "physicalMemoryBytes": ProcessInfo.processInfo.physicalMemory,
                    "availableAppMemoryBytes": os_proc_available_memory(),
                    "availableStorageBytes": values.volumeAvailableCapacity ?? 0,
                    "availableImportantStorageBytes": values.volumeAvailableCapacityForImportantUsage ?? 0,
                    "totalStorageBytes": values.volumeTotalCapacity ?? 0,
                    "lowPowerMode": ProcessInfo.processInfo.isLowPowerModeEnabled,
                    "thermalState": String(describing: ProcessInfo.processInfo.thermalState)
                ]
                let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
                try data.write(to: URL.documentsDirectory.appending(path: "BenchmarkDeviceInfo.json"), options: .atomic)
                summary = "Device check saved. Available storage: \((values.volumeAvailableCapacity ?? 0) / 1_000_000_000) GB."
                print("[BenchmarkDeviceProbe] \(String(decoding: data, as: UTF8.self))")
                fflush(nil)
            } catch { summary = error.localizedDescription }
        }
    }
}

/// A foreground-only comparison of the selected model and the experimental MiniCPM backend.
/// Settings and conversation data are never changed by this screen.
struct ModelComparisonBenchmarkView: View {
    @State private var runner: ModelComparisonBenchmarkRunner
    @State private var started = false

    init(settings: AssistantSettings, variant: ModelComparisonVariant = .dense1B) {
        _runner = State(initialValue: ModelComparisonBenchmarkRunner(settings: settings, variant: variant))
    }

    var body: some View {
        List {
            Section {
                Text("Current model versus \(runner.variant.title) · \(runner.armDescription)")
                Text(runner.progress).font(.footnote).foregroundStyle(.secondary)
                if runner.isRunning {
                    ProgressView()
                    Button("Stop", role: .cancel) { runner.cancel() }
                } else if started {
                    Button("Run again") { runner.run() }
                }
            } footer: {
                Text("Keep Assistant open. Both models answer the same prompts with greedy decoding, thinking off, and a 128-token limit: independent prompts, long prompts, and a 5-turn conversation that reuses each model's cache. The test saves its answers and measurements after each prompt.")
            }
            // Laying out the growing report during a run can delay delivery of native
            // results on the main actor and contaminate the caller's elapsed time.
            if !runner.isRunning && !runner.report.isEmpty {
                Section("Results") {
                    Text(runner.report).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                    ShareLink(item: runner.report)
                }
            }
        }
        .navigationTitle("Model comparison")
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(runner.isRunning)
        .interactiveDismissDisabled(runner.isRunning)
        .task { startIfReady() }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)) { _ in startIfReady() }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.willResignActiveNotification)) { _ in runner.cancel(reason: "Assistant became inactive.") }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)) { _ in runner.stopForBackground() }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in runner.cancel(reason: "iOS reported memory pressure.") }
        .onDisappear { runner.cancel(reason: "The comparison screen closed.") }
    }

    private func startIfReady() {
        guard !started, UIApplication.shared.applicationState == .active else { return }
        started = true
        runner.run()
    }
}

@MainActor
@Observable
final class ModelComparisonBenchmarkRunner {
    private enum Model: String, Encodable { case current, minicpm }
    private enum Status: String, Encodable { case running, completed, cancelled, failed }

    private struct LocalMetrics: Encodable {
        let firstTextSeconds: Double?
        let promptTokens: Int
        let promptSeconds: Double
        let generatedTokens: Int
        let decodeSeconds: Double
        let tokensPerSecond: Double?
        let prefilledTokens: Int?
        let reusedTokens: Int?
        let planReason: String?
        let draftTokens: Int?
        let acceptedDraftTokens: Int?
        let peakMLXBytes: Int?

        init(_ stats: LocalGenerationStats) {
            firstTextSeconds = stats.timeToFirstText
            promptTokens = stats.promptTokens
            promptSeconds = stats.promptTime
            generatedTokens = stats.generatedTokens
            decodeSeconds = stats.generateTime
            tokensPerSecond = stats.tokensPerSecond
            prefilledTokens = stats.prefilledTokens
            reusedTokens = stats.reusedTokens
            planReason = stats.planReason
            draftTokens = stats.draftTokens
            acceptedDraftTokens = stats.acceptedDraftTokens
            peakMLXBytes = stats.peakMemoryBytes
        }
    }

    private struct Row: Encodable {
        let block: Int
        let model: Model
        let kind: String
        let id: String
        let category: String
        let language: String?
        let prompt: String
        let reply: String
        let expected: String?
        let passed: Bool?
        let startedAt: Date
        /// When the Swift caller received the first visible text, for both models.
        let firstTextSeconds: Double?
        let totalSeconds: Double
        /// Tokens prefilled for this reply, and tokens reused from the cache or the saved prefix.
        let fedTokens: Int?
        let reusedTokens: Int?
        let visibleUnicodeScalars: Int
        let visibleScalarsPerSecond: Double
        let outputTokensPerTotalSecond: Double
        let finishReason: String
        let thermalBefore: String
        let thermalAfter: String
        let lowPowerBefore: Bool
        let lowPowerAfter: Bool
        let peakProcessFootprintBytes: UInt64
        let peakResidentBytes: UInt64
        let localMetrics: LocalMetrics?
        let nativeMetrics: MiniCPMBenchmarkResult?
    }

    private struct Block: Encodable {
        let index: Int
        let model: Model
        let modelID: String
        let revision: String
        var startedAt = Date()
        var finishedAt: Date?
        var status = Status.running
        var loadSeconds: Double?
        var warmupSeconds: Double?
        var warmupReply: String?
        var peakProcessFootprintBytes: UInt64 = 0
        var peakResidentBytes: UInt64 = 0
        /// MiniCPM's memory-mapped weights (`weights_bytes`), which `phys_footprint` doesn't count.
        var weightsBytes: UInt64?
        /// One MiniCPM recurrent-state checkpoint, after the block's conversation.
        var checkpointBytes: UInt64?
        var thermalAtStart = String(describing: ProcessInfo.processInfo.thermalState)
        var thermalAtEnd: String?
        var lowPowerAtStart = ProcessInfo.processInfo.isLowPowerModeEnabled
        var lowPowerAtEnd: Bool?
        var arm: MiniCPMArm?
        var preparationSeconds: Double?
        var coolingSeconds: Double?
        var thermalAtLoad: String?
        var footprintBeforePreparation: UInt64?
        var footprintBeforeLoad: UInt64?
        var availableMemoryBeforeLoad: UInt64?
        /// What the process held once the block's model was released.
        var footprintAfterRelease: UInt64?
        var residentAfterRelease: UInt64?
    }

    private struct Export: Encodable {
        let schemaVersion = 3
        let runID = UUID().uuidString
        let startedAt = Date()
        let device: String
        let systemVersion: String
        let build: String
        let currentModelID: String
        let savedEngineMode: String
        let savedSpeculation: String
        let savedPrefixCache: Bool
        let savedFastKernels: Bool
        let variant: ModelComparisonVariant
        let nativeArtifacts: [MiniCPM35BArtifacts.Artifact]?
        let draftProvenance: MiniCPM35BArtifacts.DraftProvenance?
        let accuracySuiteVersion = ModelComparisonSuite.version
        let extendedSuiteVersion = ModelComparisonSuite.extendedVersion
        let methodology: String
        let runKind: String
        let requestedArms: [MiniCPMArm]
        let preparationPolicy = "Before every block: reclaim model memory for up to 300 seconds (the 35B's later releases took 2-90+ s on the iPhone, D20), requiring idle footprint <= max(startup+256 MB, 512 MB); then wait up to 120 seconds for nominal thermal state. Preparation is outside load/reply timings. Both backends use the same policy, and a model counts as released only once its engine is gone and the footprint is back under the same ceiling."
        let nativeRuntimeCommit = "96bbf61710d44421ea54e63f5194f9b17c20d877"
        let nativeRuntimePatches = ["minicpm-metal-worker-autorelease-v1": "60c9974a467c716eda01b1f22e5839f7d1b01ea27e7ee7cab1079349b27c505a"]
        var status = Status.running
        var finishedAt: Date?
        var error: String?
        var currentModelConfiguration: String?
        var blocks: [Block] = []
        var rows: [Row] = []
    }

    private struct Response {
        let text: String
        let firstTextSeconds: Double?
        let totalSeconds: Double
        let generatedTokens: Int
        let finishReason: String
        let fed: Int?
        let reused: Int?
        var local: LocalMetrics?
        var native: MiniCPMBenchmarkResult?
    }

    private struct NativeFiles {
        let model: URL
        let projector: URL?
        var experts: URL? = nil
        var draft: URL? = nil
    }

    private(set) var isRunning = false
    private(set) var progress = "Waiting for Assistant to become active…"
    private(set) var report = ""
    let variant: ModelComparisonVariant
    @ObservationIgnored private let settings: AssistantSettings
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var engine: InferenceEngine?
    @ObservationIgnored private var native: MiniCPMSession?
    @ObservationIgnored private var export: Export?
    @ObservationIgnored private var cancellationReason: String?
    @ObservationIgnored private var blockPeak = ProcessMemory()
    @ObservationIgnored private var rowPeak = ProcessMemory()
    @ObservationIgnored private var mlxMemoryBaseline = 0
    @ObservationIgnored private var idleFootprintBaseline: UInt64 = 0
    private let arm: MiniCPMArm
    private let pilotArms: [MiniCPMArm]
    private let configurationError: String?
    private var isPilot: Bool { !pilotArms.isEmpty }
    private var blockCount: Int { isPilot ? pilotArms.count : 4 }
    private var reportDirectory: String { isPilot ? "MiniCPM35BProfilePilot" : variant.reportDirectory }
    /// This run's saved prefixes, a folder per engine; deleted when the run ends.
    private var prefixes: URL {
        FileManager.default.temporaryDirectory.appending(path: "model-comparison-\(export?.runID ?? "none")", directoryHint: .isDirectory)
    }
    /// The footprint an idle app stays under once a model has gone.
    private var idleCeiling: UInt64 { max(idleFootprintBaseline + 256_000_000, 512_000_000) }

    var armDescription: String {
        isPilot ? "pilot: " + pilotArms.map(\.name).joined(separator: ", ") : arm.name
    }

    init(settings: AssistantSettings, variant: ModelComparisonVariant) {
        self.settings = settings
        self.variant = variant
        let arguments = ProcessInfo.processInfo.arguments
        func value(_ flag: String) -> String? {
            guard let index = arguments.firstIndex(of: flag), arguments.indices.contains(index + 1) else { return nil }
            return arguments[index + 1]
        }
        let profile = value("--minicpm-profile")
        let named = profile.flatMap(MiniCPMArm.named)
        arm = named ?? .production
        let pilotNames = value("--minicpm-pilot")?.split(separator: ",").map(String.init) ?? []
        pilotArms = pilotNames.compactMap(MiniCPMArm.named)
        configurationError = ((profile != nil && named == nil) ||
                              (arguments.contains("--minicpm-profile") && profile == nil) ||
                              pilotArms.count != pilotNames.count ||
                              (arguments.contains("--minicpm-pilot") && pilotNames.isEmpty) ||
                              (!pilotNames.isEmpty && variant != .moe35B) ||
                              (variant == .dense1B && arm.draft))
            ? "Unknown or unsupported MiniCPM benchmark profile." : nil
    }

    func run() {
        guard !isRunning else { return }
        export = Export(device: Self.machine, systemVersion: UIDevice.current.systemVersion,
                        build: Self.build, currentModelID: settings.localModelID,
                        savedEngineMode: settings.localEngineMode.rawValue,
                        savedSpeculation: settings.localSpeculation.rawValue,
                        savedPrefixCache: settings.localPrefixCache, savedFastKernels: settings.localFastKernels,
                        variant: variant,
                        nativeArtifacts: variant == .moe35B ? MiniCPM35BArtifacts.artifacts : nil,
                        draftProvenance: variant == .moe35B ? MiniCPM35BArtifacts.draftProvenance : nil,
                        methodology: Self.methodology(variant, arm: arm, pilot: isPilot),
                        runKind: isPilot ? "profilePilot" : "comparison",
                        requestedArms: isPilot ? pilotArms : [arm])
        report = ""
        cancellationReason = nil
        // latest.json always describes the newest launch, a refused one included.
        if let refusal {
            progress = refusal
            export?.status = .failed
            export?.error = refusal
            export?.finishedAt = Date()
            do { try persist() } catch { progress += " Could not save results: \(error.localizedDescription)" }
            return
        }
        isRunning = true
        let previousIdleTimer = UIApplication.shared.isIdleTimerDisabled
        UIApplication.shared.isIdleTimerDisabled = true
        task = Task {
            await measure()
            UIApplication.shared.isIdleTimerDisabled = previousIdleTimer
            isRunning = false
            task = nil
        }
    }

    /// Why the comparison can't start now; nil when it can.
    private var refusal: String? {
        if let configurationError { return configurationError }
        #if targetEnvironment(simulator)
        return "Use a real iPhone for this GPU benchmark."
        #else
        guard UIApplication.shared.applicationState == .active else { return "Keep Assistant in the foreground." }
        guard !QwenListener.shared.isWanted else { return "Close voice mode before starting the comparison." }
        return nil
        #endif
    }

    func cancel(reason: String = "Stopped by user.") {
        guard isRunning else { return }
        if cancellationReason == nil {
            cancellationReason = reason
            Self.log("Cancelling: \(reason) Process footprint: \(ProcessMemory.now.footprint / 1_000_000) MB.")
        }
        task?.cancel()
        native?.cancel()
        progress = "Stopping; completed results are saved…"
        export?.status = .cancelled
        do { try persist() } catch { Self.log("Could not save cancellation: \(error.localizedDescription)") }
    }

    func stopForBackground() {
        cancel(reason: "Assistant entered the background.")
        // LocalModelHost's lifecycle guard drains direct-engine work through engineHooks.
        _ = native?.waitUntilIdle(timeout: 3)
    }

    private func measure() async {
        var ownsGPU = false
        var sampling: Task<Void, Never>?
        defer {
            sampling?.cancel()
            try? FileManager.default.removeItem(at: prefixes)
            if ownsGPU { LocalModelHost.shared.endExternalBenchmark() }
        }
        do {
            try persist()
            try await waitForMiniCPMFiles(variant.hostModel) { progress = $0 }
            progress = "Releasing reply and speech models…"
            guard await LocalModelHost.shared.beginExternalBenchmark() else {
                throw ComparisonError.message("Another on-device benchmark is running.")
            }
            ownsGPU = true
            await QwenListener.shared.unloadForBenchmark()
            await MiniCPMHost.shared.unloadAll()
            mlxMemoryBaseline = Memory.activeMemory
            idleFootprintBaseline = ProcessMemory.now.footprint
            try checkForeground()
            let snapshot = try Self.snapshot(settings.localModelID)
            export?.currentModelConfiguration = try Self.modelConfiguration(snapshot)
            progress = "Verifying cached MiniCPM model files…"
            let files: NativeFiles
            switch variant {
            case .dense1B:
                let prepared = try await MiniCPMArtifacts().prepare()
                files = NativeFiles(model: prepared.model, projector: prepared.projector)
            case .moe35B:
                let prepared = try await MiniCPM35BArtifacts().prepare { [weak self] fraction in
                    Task { @MainActor in
                        guard let self, self.isRunning, self.task?.isCancelled == false else { return }
                        self.progress = "Verifying MiniCPM 35B files: \(Int(fraction * 100))%…"
                    }
                }
                guard prepared.draft != nil || !(export?.requestedArms ?? []).contains(where: \.draft) else {
                    throw ComparisonError.message("The DSpark draft isn't on this iPhone or failed its check. Copy it in, or run --minicpm-profile targetOnly.")
                }
                // The frozen suite contains only text; do not reserve memory for a vision encoder.
                files = NativeFiles(model: prepared.model, projector: nil, experts: prepared.experts, draft: prepared.draft)
            }
            sampling = Task { [weak self] in
                while !Task.isCancelled {
                    self?.sampleMemory()
                    do { try await Task.sleep(for: .milliseconds(100)) } catch { break }
                }
            }
            let order: [Model] = isPilot ? Array(repeating: .minicpm, count: pilotArms.count) : [.current, .minicpm, .minicpm, .current]
            for (offset, model) in order.enumerated() {
                try checkForeground()
                blockPeak = .now
                let index = offset + 1
                export?.blocks.append(Block(index: index, model: model,
                                            modelID: model == .current ? settings.localModelID : variant.modelID,
                                            revision: model == .current ? snapshot.lastPathComponent :
                                                (variant == .dense1B ? MiniCPMArtifacts.revision : MiniCPM35BArtifacts.revision)))
                let blockArm = isPilot ? pilotArms[offset] : arm
                if model == .minicpm { export?.blocks[offset].arm = blockArm }
                try persist()
                try await prepareBlock(index: index)
                blockPeak = .now
                if model == .current {
                    try await currentBlock(index: index, snapshot: snapshot, scoreAccuracy: offset == 0)
                } else {
                    try await nativeBlock(index: index, files: files, scoreAccuracy: !isPilot && offset == 1, arm: blockArm)
                }
                // The block function has returned: no local strong reference retains its model.
                let released = await releaseModels()
                let after = ProcessMemory.now
                export?.blocks[offset].footprintAfterRelease = after.footprint
                export?.blocks[offset].residentAfterRelease = after.resident
                guard released else {
                    throw ComparisonError.message("The previous model's memory did not finish releasing (\(after.footprint / 1_000_000) MB still held); the next model was not loaded.")
                }
                try checkForeground()
                export?.blocks[offset].peakProcessFootprintBytes = blockPeak.footprint
                export?.blocks[offset].peakResidentBytes = blockPeak.resident
                export?.blocks[offset].finishedAt = Date()
                export?.blocks[offset].status = .completed
                export?.blocks[offset].thermalAtEnd = Self.thermal
                export?.blocks[offset].lowPowerAtEnd = ProcessInfo.processInfo.isLowPowerModeEnabled
                try persist()
                Self.log("Block \(index)/\(blockCount) \(model.rawValue) completed")
            }
            export?.status = .completed
            progress = "Comparison finished. All model memory released."
        } catch {
            native?.cancel()
            let cancelled = Task.isCancelled || error is CancellationError
            export?.status = cancelled ? .cancelled : .failed
            export?.error = cancelled ? "\(cancellationReason ?? "Cancelled.") Completed prompts are preserved." : error.localizedDescription
            Self.log(export?.error ?? error.localizedDescription)
            if let offset = export?.blocks.indices.last, export?.blocks[offset].status == .running {
                export?.blocks[offset].status = cancelled ? .cancelled : .failed
                export?.blocks[offset].peakProcessFootprintBytes = blockPeak.footprint
                export?.blocks[offset].peakResidentBytes = blockPeak.resident
                export?.blocks[offset].finishedAt = Date()
                export?.blocks[offset].thermalAtEnd = Self.thermal
                export?.blocks[offset].lowPowerAtEnd = ProcessInfo.processInfo.isLowPowerModeEnabled
            }
            progress = export?.error ?? error.localizedDescription
            // Saved first: iOS can suspend a backgrounded app while the memory retires.
            try? persist()
            _ = await releaseModels()
        }
        export?.finishedAt = Date()
        do { try persist() } catch {
            progress += " Could not save results: \(error.localizedDescription)"
        }
        Self.log("Run \(export?.status.rawValue ?? "unknown"); Documents/\(reportDirectory)/latest.json")
    }

    /// OS/driver pages can retire after the owning model has deinitialized.
    /// Apply the same bounded reclamation and cooling policy to both backends.
    private func prepareBlock(index: Int) async throws {
        let started = ContinuousClock.now
        export?.blocks[index - 1].footprintBeforePreparation = ProcessMemory.now.footprint
        for _ in 0..<3000 {
            try checkForeground()
            await Task.detached {
                autoreleasepool {
                    Stream.gpu.synchronize()
                    Stream.cpu.synchronize()
                    Memory.clearCache()
                    _ = malloc_zone_pressure_relief(nil, 0)
                }
            }.value
            if ProcessMemory.now.footprint <= idleCeiling { break }
            progress = "Block \(index)/\(blockCount): waiting for model memory to retire…"
            try await Task.sleep(for: .milliseconds(100))
        }
        guard ProcessMemory.now.footprint <= idleCeiling else {
            throw ComparisonError.message("The previous model's process memory did not retire within 300 seconds; the next model was not loaded.")
        }
        let cooling = ContinuousClock.now
        while ProcessInfo.processInfo.thermalState != .nominal && Self.seconds(cooling.duration(to: .now)) < 120 {
            try checkForeground()
            progress = "Block \(index)/\(blockCount): cooling before loading (\(Self.thermal))…"
            try await Task.sleep(for: .seconds(1))
        }
        try checkForeground()
        export?.blocks[index - 1].coolingSeconds = Self.seconds(cooling.duration(to: .now))
        export?.blocks[index - 1].preparationSeconds = Self.seconds(started.duration(to: .now))
        export?.blocks[index - 1].thermalAtLoad = Self.thermal
        export?.blocks[index - 1].footprintBeforeLoad = ProcessMemory.now.footprint
        export?.blocks[index - 1].availableMemoryBeforeLoad = UInt64(os_proc_available_memory())
        Self.log("Block \(index) prepared: \(Self.thermal), footprint=\(ProcessMemory.now.footprint / 1_000_000) MB, available=\(os_proc_available_memory() / 1_000_000) MB.")
        try persist()
    }

    private func currentBlock(index: Int, snapshot: URL, scoreAccuracy: Bool) async throws {
        let needed = (LocalModelCatalog.option(for: settings.localModelID)?.approximateBytes ?? 2_500_000_000) + 600_000_000
        guard UInt64(os_proc_available_memory()) >= UInt64(needed) else {
            throw ComparisonError.message("Not enough available app memory to load the current model (needs about \(needed / 1_000_000) MB).")
        }
        // The app's own engine setup (speculation mode and cost curve, extensions, fast-kernel
        // request), decoding greedily. Its mechanisms run in this run's folders: a drafting corpus
        // that starts empty with each load, so no block drafts from another's identical replies,
        // and, while the setting is on, a prefix cache that a later block loads from disk.
        let host = LocalModelHost.shared
        var configuration = EngineSetup.configuration(settings: settings, host: host)
        configuration.temperature = 0
        configuration.maxTokens = 128
        configuration.chatContext = ["enable_thinking": false]
        configuration.prefixCacheDirectory = settings.localPrefixCache ? prefixes.appending(path: "current", directoryHint: .isDirectory) : nil
        let speculation = EngineSetup.generatorFactory(settings: settings, host: host)
        configuration.generatorFactory = SpeculativeGeneratorFactory(mode: speculation.mode, curve: speculation.curve, maxDraft: speculation.maxDraft,
                                                                     corpusURL: nil, draftModel: speculation.draftModel)
        let modelID = settings.localModelID
        let fixedConfiguration = configuration
        progress = "Block \(index)/\(blockCount): loading \(modelID)…"
        Self.log(progress)
        let started = ContinuousClock.now
        engine = try await Task.detached {
            try await Device.withDefaultDevice(.cpu) {
                try await InferenceEngine.load(directory: snapshot, modelID: modelID, configuration: fixedConfiguration)
            }
        }.value
        export?.blocks[index - 1].loadSeconds = Self.seconds(started.duration(to: .now))
        try checkForeground()
        let warming = ContinuousClock.now
        try await engine?.warmUp()
        let warmup = try await currentReply(system: "", turns: [ChatTurn(role: .user, text: "Reply with exactly OK.")], fresh: true)
        export?.blocks[index - 1].warmupSeconds = Self.seconds(warming.duration(to: .now))
        export?.blocks[index - 1].warmupReply = warmup.text
        try persist()
        try await prompts(model: .current, block: index, scoreAccuracy: scoreAccuracy)
    }

    private func nativeBlock(index: Int, files: NativeFiles, scoreAccuracy: Bool, arm: MiniCPMArm) async throws {
        let configuration = arm.configuration
        let extraCacheBytes = max(0, configuration.expertCacheSize - 32) * 54_067_200
        let needed = variant == .dense1B ? 2_300_000_000 : 5_000_000_000 + extraCacheBytes
        guard os_proc_available_memory() >= needed else {
            throw ComparisonError.message("Not enough available app memory for MiniCPM arm \(arm.name) (needs about \(needed / 1_000_000) MB).")
        }
        native = MiniCPMSession()
        progress = "Block \(index)/\(blockCount): loading \(variant.title) · \(arm.name)…"
        Self.log(progress)
        let started = ContinuousClock.now
        try await native?.load(modelURL: files.model, projectorURL: files.projector,
                               expertsURL: files.experts, draftURL: arm.draft ? files.draft : nil, configuration: configuration)
        export?.blocks[index - 1].loadSeconds = Self.seconds(started.duration(to: .now))
        export?.blocks[index - 1].weightsBytes = await native?.info.weightsBytes
        try checkForeground()
        let warming = ContinuousClock.now
        let warmup = try await nativeReply(system: "", turns: [ChatTurn(role: .user, text: "Reply with exactly OK.")])
        export?.blocks[index - 1].warmupSeconds = Self.seconds(warming.duration(to: .now))
        export?.blocks[index - 1].warmupReply = warmup.text
        try persist()
        try await prompts(model: .minicpm, block: index, scoreAccuracy: scoreAccuracy)
        export?.blocks[index - 1].checkpointBytes = await native?.info.checkpointBytes
    }

    /// When the reuse conversation's turns are sent, fixed so its context tags render the same in every run.
    private static let reuseStart = Date(timeIntervalSince1970: 1_791_684_000)
    private static let reuseZone = TimeZone(identifier: "Asia/Singapore") ?? .gmt

    private func prompts(model: Model, block: Int, scoreAccuracy: Bool) async throws {
        for (offset, prompt) in LocalBenchmark.prompts.enumerated() {
            if isPilot && ![0, 3, 4].contains(offset) { continue }
            try await record(model: model, block: block, kind: "performance", id: "performance-\(offset + 1)",
                             category: "English/Chinese conversation", turns: [ChatTurn(role: .user, text: prompt)])
        }
        if scoreAccuracy {
            for prompt in ModelComparisonSuite.prompts {
                try await record(model: model, block: block, kind: "accuracy", id: prompt.id, category: prompt.category,
                                 turns: [ChatTurn(role: .user, text: prompt.prompt)], accuracy: prompt)
            }
        }
        guard !isPilot else { return }
        for prompt in ModelComparisonSuite.longPrompts {
            try await record(model: model, block: block, kind: "long", id: prompt.id, category: prompt.category,
                             turns: [ChatTurn(role: .user, text: prompt.prompt)], accuracy: prompt)
        }
        // A spoken conversation under the app's system prompt (no tools), rendered as the app renders
        // history: each turn carries the replies before it, so both engines reuse their cache.
        let system = PromptBuilder.systemPrompt(userName: "", customInstructions: "", profile: .app)
        var messages: [StoredMessage] = []
        for (offset, text) in ModelComparisonSuite.reuseTurns.enumerated() {
            let sent = Self.reuseStart.addingTimeInterval(Double(offset) * 60)
            messages.append(StoredMessage(role: .user, text: text, createdAt: sent, isVoice: true))
            let reply = try await record(model: model, block: block, kind: "reuse", id: "reuse-\(offset + 1)",
                                         category: "5-turn spoken conversation", system: system,
                                         turns: PromptBuilder.turns(from: messages, timeZone: Self.reuseZone), fresh: offset == 0)
            messages.append(StoredMessage(role: .assistant, text: reply, createdAt: sent.addingTimeInterval(20), isVoice: true))
        }
    }

    /// Answers `turns` and records the row; returns the reply. A `fresh` request starts without
    /// the current model's live session (the MiniCPM planner rebuilds for a new conversation).
    @discardableResult
    private func record(model: Model, block: Int, kind: String, id: String, category: String, system: String = "",
                        turns: [ChatTurn], fresh: Bool = true, accuracy: ModelComparisonSuite.Prompt? = nil) async throws -> String {
        try checkForeground()
        progress = "Block \(block)/\(blockCount) · \(model.rawValue) · \(kind) · \(id)"
        let startedAt = Date()
        let thermalBefore = Self.thermal
        let lowPowerBefore = ProcessInfo.processInfo.isLowPowerModeEnabled
        rowPeak = .now
        blockPeak.raise(to: rowPeak)
        let result: Response
        switch model {
        case .current: result = try await currentReply(system: system, turns: turns, fresh: fresh)
        case .minicpm: result = try await nativeReply(system: system, turns: turns)
        }
        try checkForeground()
        sampleMemory()
        let evaluation = accuracy.map { ModelComparisonSuite.evaluate($0, reply: result.text) }
        let count = result.text.unicodeScalars.count
        export?.rows.append(Row(block: block, model: model, kind: kind, id: id, category: category, language: accuracy?.language,
                                prompt: turns.last?.text ?? "", reply: result.text, expected: evaluation?.expected,
                                passed: evaluation?.passed, startedAt: startedAt,
                                firstTextSeconds: result.firstTextSeconds, totalSeconds: result.totalSeconds,
                                fedTokens: result.fed, reusedTokens: result.reused,
                                visibleUnicodeScalars: count,
                                visibleScalarsPerSecond: result.totalSeconds > 0 ? Double(count) / result.totalSeconds : 0,
                                outputTokensPerTotalSecond: result.totalSeconds > 0 ? Double(result.generatedTokens) / result.totalSeconds : 0,
                                finishReason: result.finishReason, thermalBefore: thermalBefore, thermalAfter: Self.thermal,
                                lowPowerBefore: lowPowerBefore, lowPowerAfter: ProcessInfo.processInfo.isLowPowerModeEnabled,
                                peakProcessFootprintBytes: rowPeak.footprint, peakResidentBytes: rowPeak.resident,
                                localMetrics: result.local, nativeMetrics: result.native))
        export?.blocks[block - 1].peakProcessFootprintBytes = blockPeak.footprint
        export?.blocks[block - 1].peakResidentBytes = blockPeak.resident
        try persist()
        Self.log("\(model.rawValue) \(id): \(String(format: "%.3f", result.firstTextSeconds ?? -1)) s first text, \(String(format: "%.2f", result.totalSeconds)) s total, reused \(result.reused ?? 0), \(result.finishReason)\(evaluation.map { ", accuracy=\($0.passed)" } ?? "")")
        return result.text
    }

    private func currentReply(system: String, turns: [ChatTurn], fresh: Bool) async throws -> Response {
        guard let engine else { throw ComparisonError.message("The current model is not loaded.") }
        // Outside the timing window: an independent request keeps no live session (only a prefix
        // saved on disk).
        if fresh { await engine.invalidateSession() }
        try checkForeground()
        let request = EngineRequest(system: system, tools: [], turns: turns, maxTokens: 128, greedy: true, toolChoice: .native)
        let started = ContinuousClock.now
        var text = ""
        var firstText: Double?
        var finished: EngineFinish?
        for try await event in engine.reply(request) {
            try Task.checkCancellation()
            switch event {
            case .text(let chunk):
                if firstText == nil, MiniCPMStream.isVisible(chunk) {
                    firstText = Self.seconds(started.duration(to: .now))
                }
                text += chunk
            case .finished(let finish): finished = finish
            default: break
            }
        }
        let total = Self.seconds(started.duration(to: .now))
        guard let finished else { throw ComparisonError.message("The current model did not finish its reply.") }
        guard finished.reason != .cancelled else { throw CancellationError() }
        return Response(text: text, firstTextSeconds: firstText, totalSeconds: total,
                        generatedTokens: finished.stats.generatedTokens, finishReason: String(describing: finished.reason),
                        fed: finished.stats.prefilledTokens, reused: finished.stats.reusedTokens,
                        local: LocalMetrics(finished.stats))
    }

    /// Streams the reply as the app does (`MiniCPMSession.reply`), its prefix saved in this run's folder.
    private func nativeReply(system: String, turns: [ChatTurn]) async throws -> Response {
        guard let native else { throw ComparisonError.message("MiniCPM is not loaded.") }
        let directory = prefixes.appending(path: "minicpm", directoryHint: .isDirectory)
        let streamed = try await MiniCPMStream.measure {
            native.reply(system: system, turns: turns, maxTokens: 128, prefixDirectory: directory)
        }
        let result = streamed.result
        guard let arm = export?.blocks.last?.arm, result.runtimeConfiguration == arm.configuration else {
            throw ComparisonError.message("MiniCPM's active runtime settings do not match the requested benchmark profile.")
        }
        if variant == .moe35B, !result.lesEnabled || result.dsparkEnabled != arm.draft {
            throw ComparisonError.message("The 35B benchmark needs LES, and DSpark exactly when the arm uses the draft; the runtime didn't match.")
        }
        return Response(text: streamed.text, firstTextSeconds: streamed.firstTextSeconds, totalSeconds: streamed.totalSeconds,
                        generatedTokens: result.generatedTokens, finishReason: result.finishReason,
                        fed: streamed.fed, reused: streamed.reused, native: result)
    }

    /// End the receiver's async lifetime before clearing buffers returned by deinit.
    @inline(never) private func detachModels() async {
        await native?.unload()
        native = nil
        await engine?.waitUntilIdle()
        engine = nil
    }

    /// Whether both models are gone: their objects deinitialized, MLX's buffers back to the
    /// baseline, and the process footprint, which holds the native runtime's Metal buffers, back
    /// under `idleCeiling`. Waits up to 300 seconds: on the iPhone the 35B's memory often
    /// stays counted for 2–90+ s after a later unload in a process (D20, D22).
    private func releaseModels() async -> Bool {
        weak let retiringEngine = engine
        weak let retiringNative = native
        await detachModels()
        for attempt in 0..<3000 {
            let memory = await Task.detached {
                autoreleasepool {
                    Stream.gpu.synchronize()
                    Stream.cpu.synchronize()
                    Memory.clearCache()
                    _ = malloc_zone_pressure_relief(nil, 0)
                    return (active: Memory.activeMemory, cached: Memory.cacheMemory)
                }
            }.value
            let process = ProcessMemory.now
            let released = retiringEngine == nil && retiringNative == nil
                && memory.active <= mlxMemoryBaseline + 64_000_000 && memory.cached <= 64_000_000
                && process.footprint <= idleCeiling
            if released || attempt == 0 || attempt == 2999 {
                Self.log("Model release: engineReleased=\(retiringEngine == nil), nativeReleased=\(retiringNative == nil), MLX active=\(memory.active / 1_000_000) MB, cached=\(memory.cached / 1_000_000) MB, footprint=\(process.footprint / 1_000_000) MB, resident=\(process.resident / 1_000_000) MB, available=\(os_proc_available_memory() / 1_000_000) MB.")
            }
            if released { return true }
            // Cleanup must still drain when the benchmark task has been cancelled.
            await Task.detached { try? await Task.sleep(for: .milliseconds(100)) }.value
        }
        return false
    }

    private func checkForeground() throws {
        try Task.checkCancellation()
        guard UIApplication.shared.applicationState == .active else { throw CancellationError() }
    }

    private static func snapshot(_ modelID: String) throws -> URL {
        let cache = HubCache(cacheDirectory: URL.applicationSupportDirectory.appending(path: "Models", directoryHint: .isDirectory))
        guard let repo = Repo.ID(rawValue: modelID),
              let revision = cache.resolveRevision(repo: repo, kind: .model, ref: "main"),
              let directory = try? cache.snapshotPath(repo: repo, kind: .model, commitHash: revision),
              LocalModelHost.hasModelFiles(in: directory) else {
            throw ComparisonError.message("Download the selected on-device model (\(modelID)) in Settings before comparing it.")
        }
        return directory
    }

    /// Only model architecture/quantization facts, never application preferences or secrets.
    private static func modelConfiguration(_ directory: URL) throws -> String? {
        let data = try Data(contentsOf: directory.appending(path: "config.json"))
        guard let source = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        let keys = Set(["model_type", "architectures", "quantization", "quantization_config", "hidden_size", "num_hidden_layers", "num_attention_heads", "torch_dtype", "vocab_size"])
        var selected = source.filter { keys.contains($0.key) }
        if let text = source["text_config"] as? [String: Any] {
            selected["text_config"] = text.filter { keys.contains($0.key) }
        }
        return String(decoding: try JSONSerialization.data(withJSONObject: selected, options: [.prettyPrinted, .sortedKeys]), as: UTF8.self)
    }

    private func sampleMemory() {
        let now = ProcessMemory.now
        rowPeak.raise(to: now)
        blockPeak.raise(to: now)
    }

    /// Writes this run's files and replaces `latest.json` and `latest.txt` (each atomically).
    private func persist() throws {
        guard let export else { return }
        report = Self.report(export)
        let root = URL.documentsDirectory.appending(path: reportDirectory, directoryHint: .isDirectory)
        let run = root.appending(path: export.runID, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(export)
        try data.write(to: run.appending(path: "results.json"), options: .atomic)
        try data.write(to: root.appending(path: "latest.json"), options: .atomic)
        try report.write(to: run.appending(path: "report.txt"), atomically: true, encoding: .utf8)
        try report.write(to: root.appending(path: "latest.txt"), atomically: true, encoding: .utf8)
    }

    private static func report(_ export: Export) -> String {
        var lines = ["On-device model comparison · \(export.status.rawValue)",
                     "\(export.device) · iOS \(export.systemVersion) · \(export.build)",
                     "Current: \(export.currentModelID)", export.variant.title, "",
                     export.methodology, ""]
        func seconds(_ value: Double?) -> String { value.map { String(format: "%.3f", $0) } ?? "–" }
        func gigabytes(_ value: UInt64?) -> String { value.map { String(format: "%.2f", Double($0) / 1e9) } ?? "–" }
        for model in [Model.current, .minicpm] {
            let rows = export.rows.filter { $0.model == model }
            func firstText(_ id: String) -> String { seconds(median(rows.filter { $0.id == id }.compactMap(\.firstTextSeconds))) }
            let performance = rows.filter { $0.kind == "performance" }
            let accuracy = rows.filter { $0.kind == "accuracy" }
            let long = rows.filter { $0.kind == "long" }
            let blocks = export.blocks.filter { $0.model == model }
            let first = median(performance.compactMap(\.firstTextSeconds))
            let throughput = median(performance.map(\.visibleScalarsPerSecond))
            lines.append("\(model.rawValue): \(performance.count) performance prompts, first text median \(seconds(first)) s, visible throughput median \(throughput.map { String(format: "%.1f", $0) } ?? "–") Unicode scalars/s; accuracy \(accuracy.filter { $0.passed == true }.count)/\(accuracy.count)")
            lines.append("  long prompts (~500 / 1,000 / 2,000 tokens): first text median \(ModelComparisonSuite.longPrompts.map { firstText($0.id) }.joined(separator: " / ")) s; correct \(long.filter { $0.passed == true }.count)/\(long.count)")
            lines.append("  5-turn conversation: first text median by turn \(ModelComparisonSuite.reuseTurns.indices.map { firstText("reuse-\($0 + 1)") }.joined(separator: " / ")) s")
            lines.append("  peak phys_footprint \(gigabytes(blocks.map(\.peakProcessFootprintBytes).max())) GB, resident \(gigabytes(blocks.map(\.peakResidentBytes).max())) GB, mapped weights \(gigabytes(blocks.compactMap(\.weightsBytes).max())) GB")
        }
        if let error = export.error { lines += ["", "Stopped: \(error)"] }
        for row in export.rows {
            lines += ["", "Block \(row.block) · \(row.model.rawValue) · \(row.id) · \(row.finishReason)", row.prompt, "→ \(row.reply)"]
            if let passed = row.passed { lines.append("Score: \(passed ? "pass" : "fail"); expected: \(row.expected ?? "")") }
        }
        return lines.joined(separator: "\n")
    }

    private static func methodology(_ variant: ModelComparisonVariant, arm: MiniCPMArm, pilot: Bool) -> String {
        let configuration = arm.configuration
        let native = variant == .dense1B
            ? "MiniCPM 1B uses Q4_K_M with its F16 projector loaded and no speculation."
            : "MiniCPM 35B-A3B uses the published mix_q3gud_im LES split, unchanged top-8 expert routing, a 256-slot shared bank, \(configuration.expertCacheSize) cached experts per layer and lookahead prefetch depth 4; " + (arm.draft
                ? "DSpark proposes up to \(configuration.draftMaximum) draft tokens, verified by target greedy decoding, and draft attempts and target-accepted draft tokens are recorded."
                : "it runs target-only, without the DSpark draft.") + " The vision projector is not loaded for this text-only suite."
        if pilot {
            return "Native-only profile selection pilot. Each named arm (targetOnly = baseline without the draft; any other name = that profile with DSpark) reloads the model, excludes one warmup, and runs frozen performance prompts 1, 4, and 5 with greedy decoding, thinking off, and max output 128. Arms and actual runtime settings are saved per block/response. This pilot does not replace the full accuracy/performance comparison."
        }
        return "ABBA order: current, MiniCPM, MiniCPM, current. MiniCPM arm: \(arm.name). Each block reloads the model and excludes one warmup reply. Every block runs the same 8 performance prompts, 3 scored long prompts (about 500, 1,000 and 2,000 tokens), and a 5-turn spoken conversation under the app's system prompt; the first block for each model also runs the same 24 scored prompts. Greedy decoding, max output 128 tokens, thinking disabled, no tools. Independent prompts start from an empty conversation. The conversation's turns reuse each engine's cache, and its first turn starts from the system prefix an earlier block saved to disk, if any. The current model uses LocalEngine with the app's engine setup: its saved speculation mode and cost curve with a drafting corpus that starts empty in each block, its extensions, and its prefix cache when that setting is on, kept in a folder of this run. \(native) MiniCPM replies through MiniCPMSession.reply, which renders ChatML in Swift and reuses its cache as the app does. Both models are timed the same way: first text is when the Swift caller receives the first visible text (MiniCPM's runtime time from its generate call is nativeMetrics.nativeFirstTextSeconds); total time is measured around each reply. Visible throughput is Unicode scalars / total time including prefill. Tokenizers differ, so raw token rates are not directly equivalent. Memory is sampled every 100 ms: phys_footprint (what iOS counts against the app; it leaves out MiniCPM's memory-mapped weights) and resident_size (pages in RAM, mapped weights included); weightsBytes is MiniCPM's mapped weights. This is a small deterministic text test, not a general capability or tool-use evaluation."
    }

    private static func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
    }

    private static var thermal: String { String(describing: ProcessInfo.processInfo.thermalState) }
    private static var machine: String {
        var info = utsname()
        uname(&info)
        let capacity = MemoryLayout.size(ofValue: info.machine)
        return withUnsafePointer(to: &info.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: capacity) { String(cString: $0) }
        }
    }
    private static var build: String {
        #if DEBUG
        return "Debug"
        #else
        return "Release"
        #endif
    }
    private static func seconds(_ duration: Duration) -> Double { MiniCPMStream.seconds(duration) }
    private static func log(_ message: String) {
        print("[ModelComparison] \(message)")
        fflush(nil)
    }
    private enum ComparisonError: LocalizedError {
        case message(String)
        var errorDescription: String? { if case .message(let text) = self { return text }; return nil }
    }
}
