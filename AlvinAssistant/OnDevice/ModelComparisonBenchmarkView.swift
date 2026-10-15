import AssistantKit
import Darwin
import HuggingFace
import LocalEngine
import MiniCPMBenchmark
import MLX
import Observation
import SwiftUI
import UIKit

enum ModelComparisonVariant: String, Encodable {
    case dense1B, moe35BLESDSpark

    var title: String {
        self == .dense1B ? "MiniCPM-V 4.7 1B" : "MiniCPM-V 4.7 35B-A3B · LES + DSpark"
    }
    var modelID: String {
        "tc-mb/MiniCPM-V-Apps-gguf/" + (self == .dense1B ? "MiniCPM-V-4.7-1B" : "MiniCPM-V-4.7-35B-A3B")
    }
    var reportDirectory: String { self == .dense1B ? "ModelComparison" : "ModelComparison35B" }
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
                Text("Current model versus \(runner.variant.title)")
                Text(runner.progress).font(.footnote).foregroundStyle(.secondary)
                if runner.isRunning {
                    ProgressView()
                    Button("Stop", role: .cancel) { runner.cancel() }
                } else if started {
                    Button("Run again") { runner.run() }
                }
            } footer: {
                Text("Keep Assistant open. Both models answer the same independent prompts with greedy decoding, thinking off, and a 128-token limit. The test saves its answers and measurements after each prompt.")
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
        let reusedTokens: Int?
        let peakMLXBytes: Int?

        init(_ stats: LocalGenerationStats) {
            firstTextSeconds = stats.timeToFirstText
            promptTokens = stats.promptTokens
            promptSeconds = stats.promptTime
            generatedTokens = stats.generatedTokens
            decodeSeconds = stats.generateTime
            tokensPerSecond = stats.tokensPerSecond
            reusedTokens = stats.reusedTokens
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
        let firstTextSeconds: Double?
        let totalSeconds: Double
        let visibleUnicodeScalars: Int
        let visibleScalarsPerSecond: Double
        let outputTokensPerTotalSecond: Double
        let finishReason: String
        let thermalBefore: String
        let thermalAfter: String
        let lowPowerBefore: Bool
        let lowPowerAfter: Bool
        let peakProcessFootprintBytes: UInt64
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
        var thermalAtStart = String(describing: ProcessInfo.processInfo.thermalState)
        var thermalAtEnd: String?
        var lowPowerAtStart = ProcessInfo.processInfo.isLowPowerModeEnabled
        var lowPowerAtEnd: Bool?
        var runtimeConfiguration: MiniCPMRuntimeConfiguration?
        var preparationSeconds: Double?
        var coolingSeconds: Double?
        var thermalAtLoad: String?
        var footprintBeforePreparation: UInt64?
        var footprintBeforeLoad: UInt64?
        var availableMemoryBeforeLoad: UInt64?
    }

    private struct Export: Encodable {
        let schemaVersion = 2
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
        let methodology: String
        let runKind: String
        let requestedRuntimeConfiguration: MiniCPMRuntimeConfiguration
        let preparationPolicy = "Before every block: reclaim model memory for up to 10 seconds, requiring idle footprint <= max(startup+256 MB, 512 MB); then wait up to 120 seconds for nominal thermal state. Preparation is outside load/reply timings. Both backends use the same policy."
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
    @ObservationIgnored private var blockPeak: UInt64 = 0
    @ObservationIgnored private var rowPeak: UInt64 = 0
    @ObservationIgnored private var mlxMemoryBaseline = 0
    @ObservationIgnored private var idleFootprintBaseline: UInt64 = 0
    private let configuration: MiniCPMRuntimeConfiguration
    private let pilotProfiles: [MiniCPMRuntimeConfiguration]
    private let configurationError: String?
    private var isPilot: Bool { !pilotProfiles.isEmpty }
    private var blockCount: Int { isPilot ? pilotProfiles.count : 4 }
    private var reportDirectory: String { isPilot ? "MiniCPM35BProfilePilot" : variant.reportDirectory }

    init(settings: AssistantSettings, variant: ModelComparisonVariant) {
        self.settings = settings
        self.variant = variant
        let arguments = ProcessInfo.processInfo.arguments
        func value(_ flag: String) -> String? {
            guard let index = arguments.firstIndex(of: flag), arguments.indices.contains(index + 1) else { return nil }
            return arguments[index + 1]
        }
        let name = value("--minicpm-profile") ?? "baseline"
        configuration = MiniCPMRuntimeConfiguration.profiles.first { $0.name == name } ?? .baseline
        let pilotNames = value("--minicpm-pilot")?.split(separator: ",").map(String.init) ?? []
        pilotProfiles = pilotNames.compactMap { name in MiniCPMRuntimeConfiguration.profiles.first { $0.name == name } }
        configurationError = (!MiniCPMRuntimeConfiguration.profiles.contains { $0.name == name } ||
                              (arguments.contains("--minicpm-profile") && value("--minicpm-profile") == nil) ||
                              pilotProfiles.count != pilotNames.count ||
                              (arguments.contains("--minicpm-pilot") && pilotNames.isEmpty) ||
                              (!pilotNames.isEmpty && variant != .moe35BLESDSpark))
            ? "Unknown or unsupported MiniCPM benchmark profile." : nil
    }

    func run() {
        guard !isRunning else { return }
        if let configurationError { progress = configurationError; return }
        #if targetEnvironment(simulator)
        progress = "Use a real iPhone for this GPU benchmark."
        return
        #else
        guard UIApplication.shared.applicationState == .active else {
            progress = "Keep Assistant in the foreground."
            return
        }
        guard !QwenListener.shared.isWanted else {
            progress = "Close voice mode before starting the comparison."
            return
        }
        export = Export(device: Self.machine, systemVersion: UIDevice.current.systemVersion,
                        build: Self.build, currentModelID: settings.localModelID,
                        savedEngineMode: settings.localEngineMode.rawValue,
                        savedSpeculation: settings.localSpeculation.rawValue,
                        savedPrefixCache: settings.localPrefixCache, savedFastKernels: settings.localFastKernels,
                        variant: variant,
                        nativeArtifacts: variant == .moe35BLESDSpark ? MiniCPM35BArtifacts.artifacts : nil,
                        draftProvenance: variant == .moe35BLESDSpark ? MiniCPM35BArtifacts.draftProvenance : nil,
                        methodology: Self.methodology(variant, configuration: configuration, pilot: isPilot),
                        runKind: isPilot ? "profilePilot" : "comparison",
                        requestedRuntimeConfiguration: configuration)
        report = ""
        cancellationReason = nil
        isRunning = true
        let previousIdleTimer = UIApplication.shared.isIdleTimerDisabled
        UIApplication.shared.isIdleTimerDisabled = true
        task = Task {
            await measure()
            UIApplication.shared.isIdleTimerDisabled = previousIdleTimer
            isRunning = false
            task = nil
        }
        #endif
    }

    func cancel(reason: String = "Stopped by user.") {
        guard isRunning else { return }
        if cancellationReason == nil {
            cancellationReason = reason
            Self.log("Cancelling: \(reason) Process footprint: \(Self.memoryFootprint() / 1_000_000) MB.")
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
            if ownsGPU { LocalModelHost.shared.endExternalBenchmark() }
        }
        do {
            try persist()
            progress = "Releasing reply and speech models…"
            guard await LocalModelHost.shared.beginExternalBenchmark() else {
                throw ComparisonError.message("Another on-device benchmark is running.")
            }
            ownsGPU = true
            await QwenListener.shared.unloadForBenchmark()
            mlxMemoryBaseline = Memory.activeMemory
            idleFootprintBaseline = Self.memoryFootprint()
            try checkForeground()
            let snapshot = try Self.snapshot(settings.localModelID)
            export?.currentModelConfiguration = try Self.modelConfiguration(snapshot)
            progress = "Verifying cached MiniCPM model files…"
            let files: NativeFiles
            switch variant {
            case .dense1B:
                let directory = URL.applicationSupportDirectory.appending(path: "MiniCPMBenchmark", directoryHint: .isDirectory)
                let prepared = try await MiniCPMArtifacts(directory: directory).prepare()
                files = NativeFiles(model: prepared.model, projector: prepared.projector)
            case .moe35BLESDSpark:
                let directory = URL.applicationSupportDirectory.appending(path: "MiniCPM35BBenchmark", directoryHint: .isDirectory)
                let prepared = try await MiniCPM35BArtifacts(directory: directory).prepare { [weak self] fraction in
                    Task { @MainActor in
                        guard let self, self.isRunning, self.task?.isCancelled == false else { return }
                        self.progress = "Verifying MiniCPM 35B files: \(Int(fraction * 100))%…"
                    }
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
            let order: [Model] = isPilot ? Array(repeating: .minicpm, count: pilotProfiles.count) : [.current, .minicpm, .minicpm, .current]
            for (offset, model) in order.enumerated() {
                try checkForeground()
                blockPeak = Self.memoryFootprint()
                let index = offset + 1
                export?.blocks.append(Block(index: index, model: model,
                                            modelID: model == .current ? settings.localModelID : variant.modelID,
                                            revision: model == .current ? snapshot.lastPathComponent :
                                                (variant == .dense1B ? MiniCPMArtifacts.revision : MiniCPM35BArtifacts.revision)))
                let blockConfiguration = isPilot ? pilotProfiles[offset] : configuration
                if model == .minicpm { export?.blocks[offset].runtimeConfiguration = blockConfiguration }
                try persist()
                try await prepareBlock(index: index)
                blockPeak = Self.memoryFootprint()
                if model == .current {
                    try await currentBlock(index: index, snapshot: snapshot, scoreAccuracy: offset == 0)
                } else {
                    try await nativeBlock(index: index, files: files, scoreAccuracy: !isPilot && offset == 1, configuration: blockConfiguration)
                }
                // The block function has returned: no local strong reference retains its model.
                guard await releaseModels() else {
                    throw ComparisonError.message("The previous model's buffers did not finish releasing; the next model was not loaded.")
                }
                try checkForeground()
                export?.blocks[offset].peakProcessFootprintBytes = blockPeak
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
            _ = await releaseModels()
            let cancelled = Task.isCancelled || error is CancellationError
            export?.status = cancelled ? .cancelled : .failed
            export?.error = cancelled ? "\(cancellationReason ?? "Cancelled.") Completed prompts are preserved." : error.localizedDescription
            Self.log(export?.error ?? error.localizedDescription)
            if let offset = export?.blocks.indices.last, export?.blocks[offset].status == .running {
                export?.blocks[offset].status = cancelled ? .cancelled : .failed
                export?.blocks[offset].peakProcessFootprintBytes = blockPeak
                export?.blocks[offset].finishedAt = Date()
                export?.blocks[offset].thermalAtEnd = Self.thermal
                export?.blocks[offset].lowPowerAtEnd = ProcessInfo.processInfo.isLowPowerModeEnabled
            }
            progress = export?.error ?? error.localizedDescription
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
        export?.blocks[index - 1].footprintBeforePreparation = Self.memoryFootprint()
        let ceiling = max(idleFootprintBaseline + 256_000_000, 512_000_000)
        for _ in 0..<100 {
            try checkForeground()
            await Task.detached {
                autoreleasepool {
                    Stream.gpu.synchronize()
                    Stream.cpu.synchronize()
                    Memory.clearCache()
                    _ = malloc_zone_pressure_relief(nil, 0)
                }
            }.value
            if Self.memoryFootprint() <= ceiling { break }
            progress = "Block \(index)/\(blockCount): waiting for model memory to retire…"
            try await Task.sleep(for: .milliseconds(100))
        }
        guard Self.memoryFootprint() <= ceiling else {
            throw ComparisonError.message("The previous model's process memory did not retire within 10 seconds; the next model was not loaded.")
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
        export?.blocks[index - 1].footprintBeforeLoad = Self.memoryFootprint()
        export?.blocks[index - 1].availableMemoryBeforeLoad = UInt64(os_proc_available_memory())
        Self.log("Block \(index) prepared: \(Self.thermal), footprint=\(Self.memoryFootprint() / 1_000_000) MB, available=\(os_proc_available_memory() / 1_000_000) MB.")
        try persist()
    }

    private func currentBlock(index: Int, snapshot: URL, scoreAccuracy: Bool) async throws {
        let needed = (LocalModelCatalog.option(for: settings.localModelID)?.approximateBytes ?? 2_500_000_000) + 600_000_000
        guard UInt64(os_proc_available_memory()) >= UInt64(needed) else {
            throw ComparisonError.message("Not enough available app memory to load the current model (needs about \(needed / 1_000_000) MB).")
        }
        var configuration = EngineConfiguration()
        configuration.temperature = 0
        configuration.maxTokens = 128
        configuration.chatContext = ["enable_thinking": false]
        configuration.prefixCacheDirectory = nil
        configuration.generatorFactory = nil
        configuration.extensions = []
        configuration.hooks = LocalModelHost.engineHooks
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
        let warmup = try await currentReply("Reply with exactly OK.")
        export?.blocks[index - 1].warmupSeconds = Self.seconds(warming.duration(to: .now))
        export?.blocks[index - 1].warmupReply = warmup.text
        try persist()
        try await prompts(model: .current, block: index, scoreAccuracy: scoreAccuracy)
    }

    private func nativeBlock(index: Int, files: NativeFiles, scoreAccuracy: Bool,
                             configuration: MiniCPMRuntimeConfiguration) async throws {
        let extraCacheBytes = max(0, configuration.expertCacheSize - 32) * 54_067_200
        let needed = variant == .dense1B ? 2_300_000_000 : 5_000_000_000 + extraCacheBytes
        guard os_proc_available_memory() >= needed else {
            throw ComparisonError.message("Not enough available app memory for MiniCPM profile \(configuration.name) (needs about \(needed / 1_000_000) MB).")
        }
        native = MiniCPMSession()
        progress = "Block \(index)/\(blockCount): loading \(variant.title)…"
        Self.log(progress)
        let started = ContinuousClock.now
        try await native?.load(modelURL: files.model, projectorURL: files.projector,
                               expertsURL: files.experts, draftURL: files.draft, configuration: configuration)
        export?.blocks[index - 1].loadSeconds = Self.seconds(started.duration(to: .now))
        try checkForeground()
        let warming = ContinuousClock.now
        let warmup = try await nativeReply("Reply with exactly OK.")
        export?.blocks[index - 1].warmupSeconds = Self.seconds(warming.duration(to: .now))
        export?.blocks[index - 1].warmupReply = warmup.text
        try persist()
        try await prompts(model: .minicpm, block: index, scoreAccuracy: scoreAccuracy)
    }

    private func prompts(model: Model, block: Int, scoreAccuracy: Bool) async throws {
        for (offset, prompt) in LocalBenchmark.prompts.enumerated() {
            if isPilot && ![0, 3, 4].contains(offset) { continue }
            try await record(model: model, block: block, kind: "performance", id: "performance-\(offset + 1)",
                             category: "English/Chinese conversation", prompt: prompt, accuracy: nil)
        }
        if scoreAccuracy {
            for prompt in ModelComparisonSuite.prompts {
                try await record(model: model, block: block, kind: "accuracy", id: prompt.id,
                                 category: prompt.category, prompt: prompt.prompt, accuracy: prompt)
            }
        }
    }

    private func record(model: Model, block: Int, kind: String, id: String, category: String,
                        prompt: String, accuracy: ModelComparisonSuite.Prompt?) async throws {
        try checkForeground()
        progress = "Block \(block)/\(blockCount) · \(model.rawValue) · \(kind) · \(id)"
        let startedAt = Date()
        let thermalBefore = Self.thermal
        let lowPowerBefore = ProcessInfo.processInfo.isLowPowerModeEnabled
        rowPeak = Self.memoryFootprint()
        blockPeak = max(blockPeak, rowPeak)
        let result: Response
        switch model {
        case .current: result = try await currentReply(prompt)
        case .minicpm: result = try await nativeReply(prompt)
        }
        try checkForeground()
        sampleMemory()
        let evaluation = accuracy.map { ModelComparisonSuite.evaluate($0, reply: result.text) }
        let count = result.text.unicodeScalars.count
        export?.rows.append(Row(block: block, model: model, kind: kind, id: id, category: category, language: accuracy?.language,
                                prompt: prompt, reply: result.text, expected: evaluation?.expected,
                                passed: evaluation?.passed, startedAt: startedAt,
                                firstTextSeconds: result.firstTextSeconds, totalSeconds: result.totalSeconds,
                                visibleUnicodeScalars: count,
                                visibleScalarsPerSecond: result.totalSeconds > 0 ? Double(count) / result.totalSeconds : 0,
                                outputTokensPerTotalSecond: result.totalSeconds > 0 ? Double(result.generatedTokens) / result.totalSeconds : 0,
                                finishReason: result.finishReason, thermalBefore: thermalBefore, thermalAfter: Self.thermal,
                                lowPowerBefore: lowPowerBefore, lowPowerAfter: ProcessInfo.processInfo.isLowPowerModeEnabled,
                                peakProcessFootprintBytes: rowPeak, localMetrics: result.local, nativeMetrics: result.native))
        export?.blocks[block - 1].peakProcessFootprintBytes = blockPeak
        try persist()
        Self.log("\(model.rawValue) \(id): \(String(format: "%.3f", result.firstTextSeconds ?? -1)) s first text, \(String(format: "%.2f", result.totalSeconds)) s total, \(result.finishReason)\(evaluation.map { ", accuracy=\($0.passed)" } ?? "")")
    }

    private func currentReply(_ prompt: String) async throws -> Response {
        guard let engine else { throw ComparisonError.message("The current model is not loaded.") }
        // Outside the timing window, but no cache contents survive into the timed request.
        await engine.invalidateSession()
        try checkForeground()
        let request = EngineRequest(system: "", tools: [], turns: [ChatTurn(role: .user, text: prompt)],
                                    maxTokens: 128, greedy: true, toolChoice: .native)
        let started = ContinuousClock.now
        var text = ""
        var firstText: Double?
        var finished: EngineFinish?
        for try await event in engine.reply(request) {
            try Task.checkCancellation()
            switch event {
            case .text(let chunk):
                // Match MiniCPM's complete visible UTF-8 criterion (skip ASCII whitespace).
                if firstText == nil, chunk.unicodeScalars.contains(where: { $0.value > 0x20 && $0.value != 0x7F }) {
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
                        local: LocalMetrics(finished.stats))
    }

    private func nativeReply(_ prompt: String) async throws -> Response {
        guard let native else { throw ComparisonError.message("MiniCPM is not loaded.") }
        let started = ContinuousClock.now
        let result = try await native.run(prompt: prompt, maxTokens: 128)
        let total = Self.seconds(started.duration(to: .now))
        if variant == .moe35BLESDSpark, !result.lesEnabled || !result.dsparkEnabled {
            throw ComparisonError.message("The 35B benchmark requires active LES and DSpark; the runtime did not enable both.")
        }
        guard result.runtimeConfiguration == export?.blocks.last?.runtimeConfiguration else {
            throw ComparisonError.message("MiniCPM's active runtime settings do not match the requested benchmark profile.")
        }
        return Response(text: result.text, firstTextSeconds: result.firstTextSeconds >= 0 ? result.firstTextSeconds : nil,
                        totalSeconds: total, generatedTokens: result.generatedTokens, finishReason: result.finishReason,
                        native: result)
    }

    /// End the receiver's async lifetime before clearing buffers returned by deinit.
    @inline(never) private func detachModels() async {
        await native?.unload()
        native = nil
        await engine?.waitUntilIdle()
        engine = nil
    }

    private func releaseModels() async -> Bool {
        weak var retiringEngine = engine
        await detachModels()
        for attempt in 0..<30 {
            let memory = await Task.detached {
                autoreleasepool {
                    Stream.gpu.synchronize()
                    Stream.cpu.synchronize()
                    Memory.clearCache()
                    return (active: Memory.activeMemory, cached: Memory.cacheMemory)
                }
            }.value
            let released = retiringEngine == nil && memory.active <= mlxMemoryBaseline + 64_000_000 && memory.cached <= 64_000_000
            if released || attempt == 0 || attempt == 29 {
                Self.log("Model release: engineReleased=\(retiringEngine == nil), MLX active=\(memory.active / 1_000_000) MB, cached=\(memory.cached / 1_000_000) MB, footprint=\(Self.memoryFootprint() / 1_000_000) MB, available=\(os_proc_available_memory() / 1_000_000) MB.")
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
        let bytes = Self.memoryFootprint()
        rowPeak = max(rowPeak, bytes)
        blockPeak = max(blockPeak, bytes)
    }

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
        for model in [Model.current, .minicpm] {
            let rows = export.rows.filter { $0.model == model }
            let performance = rows.filter { $0.kind == "performance" }
            let accuracy = rows.filter { $0.kind == "accuracy" }
            let first = median(performance.compactMap(\.firstTextSeconds))
            let throughput = median(performance.map(\.visibleScalarsPerSecond))
            lines.append("\(model.rawValue): \(performance.count) performance prompts, first text median \(first.map { String(format: "%.3f", $0) } ?? "–") s, visible throughput median \(throughput.map { String(format: "%.1f", $0) } ?? "–") Unicode scalars/s; accuracy \(accuracy.filter { $0.passed == true }.count)/\(accuracy.count)")
        }
        if let error = export.error { lines += ["", "Stopped: \(error)"] }
        for row in export.rows {
            lines += ["", "Block \(row.block) · \(row.model.rawValue) · \(row.id) · \(row.finishReason)", row.prompt, "→ \(row.reply)"]
            if let passed = row.passed { lines.append("Score: \(passed ? "pass" : "fail"); expected: \(row.expected ?? "")") }
        }
        return lines.joined(separator: "\n")
    }

    private static func methodology(_ variant: ModelComparisonVariant, configuration: MiniCPMRuntimeConfiguration, pilot: Bool) -> String {
        let native = variant == .dense1B
            ? "MiniCPM 1B uses Q4_K_M with its F16 projector loaded and no speculation."
            : "MiniCPM 35B-A3B uses the published mix_q3gud_im LES split, unchanged top-8 expert routing, a 256-slot shared bank, \(configuration.expertCacheSize) cached experts per layer, lookahead prefetch depth 4, and DSpark with a maximum of \(configuration.draftMaximum) draft tokens verified by target greedy decoding. The vision projector is not loaded for this text-only suite. Draft attempts and target-accepted draft tokens are recorded."
        if pilot {
            return "Native-only profile selection pilot. Each named profile reloads the model, excludes one warmup, and runs frozen performance prompts 1, 4, and 5 with greedy decoding, thinking off, and max output 128. Profiles and actual runtime settings are saved per block/response. This pilot does not replace the full accuracy/performance comparison."
        }
        return "ABBA order: current, MiniCPM, MiniCPM, current. Each block reloads the model, excludes one warmup reply, then runs the same 8 performance prompts. The first block for each model also runs the same 24 scored prompts. Greedy decoding, max output 128 tokens, thinking disabled, no tools, no conversation/prefix reuse. The current model has no speculation or optional kernel extensions. \(native) Current model uses LocalEngine's normal model implementation with an empty system message; MiniCPM uses its native user-only template. First-text time is caller-to-first-streamed-text for LocalEngine and native-run-to-first-visible-text for MiniCPM (the native API returns completed replies). Total time is independently measured around each public reply call. Visible throughput is Unicode scalars / total time including prefill. Native decode timing and LocalEngine decode timing have different boundaries, and tokenizers differ, so raw token rates are not directly equivalent. Process memory is sampled every 100 ms and includes the app and runtime. This is a small deterministic text test, not a general capability or tool-use evaluation."
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
    private static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }
    private static func memoryFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? info.phys_footprint : 0
    }
    private static func log(_ message: String) {
        print("[ModelComparison] \(message)")
        fflush(nil)
    }
    private enum ComparisonError: LocalizedError {
        case message(String)
        var errorDescription: String? { if case .message(let text) = self { return text }; return nil }
    }
}
