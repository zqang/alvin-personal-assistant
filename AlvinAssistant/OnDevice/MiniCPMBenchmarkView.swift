import AssistantKit
import Darwin
import MiniCPM
import Observation
import SwiftUI
import UIKit

/// A separate experimental backend: selecting it never changes the chat model.
struct MiniCPMBenchmarkView: View {
    enum LaunchMode: String {
        case text, image, all

        static var requested: Self? {
            let arguments = ProcessInfo.processInfo.arguments
            guard let index = arguments.firstIndex(of: "--minicpm-benchmark"),
                  arguments.indices.contains(index + 1) else { return nil }
            return Self(rawValue: arguments[index + 1])
        }

        var scenarios: [MiniCPMBenchmarkRunner.Scenario] {
            switch self {
            case .text: return [.text]
            case .image: return [.image]
            case .all: return [.text, .image]
            }
        }
    }

    let launchMode: LaunchMode?
    @State private var runner = MiniCPMBenchmarkRunner()
    @State private var scenario = MiniCPMBenchmarkRunner.Scenario.image
    @State private var automationStarted = false

    init(launchMode: LaunchMode? = nil) { self.launchMode = launchMode }

    var body: some View {
        List {
            Section {
                LabeledContent("Model", value: "MiniCPM-V 4.7 1B · Q4_K_M")
                Picker("Scenario", selection: $scenario) {
                    Text("Image understanding × 3").tag(MiniCPMBenchmarkRunner.Scenario.image)
                    Text("English and Chinese · 8 prompts").tag(MiniCPMBenchmarkRunner.Scenario.text)
                }
                .disabled(runner.isRunning)
                if scenario == .image {
                    Image(uiImage: MiniCPMBenchmarkRunner.fixture())
                        .resizable()
                        .scaledToFit()
                        .accessibilityLabel("Test image: SINGAPORE 47, a red circle, a blue square, and three green bars")
                    Text("Expected: SINGAPORE 47, one red circle, one blue square, three green bars. Inspect the generated answers to judge accuracy.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                if runner.isRunning {
                    ProgressView()
                    Button("Stop", role: .cancel) { runner.cancel() }
                } else {
                    Button("Download if needed and run") { runner.run(scenario) }
                }
                if let progress = runner.progress {
                    Text(progress).font(.footnote).foregroundStyle(.secondary)
                }
            } footer: {
                Text("Uses the photo reader's model, and downloads it (about 1.8 GB over Wi-Fi) if it isn't on this iPhone yet; delete it in Settings › Photo reading. Each prompt starts with an empty conversation, thinking off, and a 128-token limit. Image tests use one overview slice. Keep the app open and use a Release build on a real iPhone for useful timings.")
            }
            if !runner.report.isEmpty {
                Section("Results") {
                    Text(runner.report)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                    Button("Copy results") { UIPasteboard.general.string = runner.report }
                    ShareLink(item: runner.report)
                }
            }
        }
        .navigationTitle("MiniCPM-V benchmark")
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(runner.isRunning)
        .interactiveDismissDisabled(runner.isRunning)
        .task { startAutomationIfReady() }
        .onChange(of: runner.currentScenario) { _, current in
            if let current { scenario = current }
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)) { _ in
            startAutomationIfReady()
        }
        .onDisappear { runner.cancel() }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.willResignActiveNotification)) { _ in
            runner.cancel()
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)) { _ in
            runner.stopForBackground()
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in
            runner.cancel()
        }
    }

    private func startAutomationIfReady() {
        guard let launchMode, !automationStarted,
              UIApplication.shared.applicationState == .active else { return }
        // Consumed before starting: interruption or foregrounding never silently reruns it.
        automationStarted = true
        scenario = launchMode.scenarios[0]
        runner.run(launchMode.scenarios)
    }
}

@MainActor
@Observable
final class MiniCPMBenchmarkRunner {
    enum Scenario: String, Codable { case text, image }
    private enum Status: String, Codable { case pending, running, completed, failed, cancelled }
    private struct PromptExport: Encodable {
        let index: Int
        let prompt: String
        let thermalState: String
        /// When the Swift caller received the first visible text.
        let firstTextSeconds: Double?
        let tokensPerSecond: Double
        let metrics: MiniCPMBenchmarkResult
    }
    private struct ScenarioExport: Encodable {
        let scenario: Scenario
        var status = Status.pending
        var startedAt: Date?
        var finishedAt: Date?
        var loadSeconds: Double?
        var peakProcessFootprintBytes: UInt64 = 0
        var peakResidentBytes: UInt64 = 0
        /// The memory-mapped weights (`weights_bytes`), which `phys_footprint` doesn't count.
        var weightsBytes: UInt64?
        var prompts: [PromptExport] = []
        var report = ""
        var error: String?
    }
    private struct BenchmarkExport: Encodable {
        let schemaVersion = 2
        let runID: String
        let device: String
        let systemVersion: String
        let configuration: String
        let modelRevision = MiniCPMArtifacts.revision
        let runtimeCommit = "96bbf61710d44421ea54e63f5194f9b17c20d877"
        let lowPowerMode: Bool
        let startedAt: Date
        var finishedAt: Date?
        var status = Status.running
        var scenarios: [ScenarioExport]
    }
    private(set) var isRunning = false
    private(set) var currentScenario: Scenario?
    private(set) var progress: String?
    private(set) var report = ""
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var session: MiniCPMSession?
    @ObservationIgnored private var export: BenchmarkExport?

    func cancel() {
        task?.cancel()
        session?.cancel()
        if isRunning {
            progress = "Stopping…"
            export?.status = .cancelled
            if let index = export?.scenarios.firstIndex(where: { $0.status == .running }) {
                export?.scenarios[index].status = .cancelled
            }
            // Save the interruption before iOS can suspend the app. finishedAt remains
            // unset until native work has drained and the task actually returns.
            do { try persistExport() } catch {
                Self.log("Could not save cancellation: \(error.localizedDescription)")
            }
            Self.log("Cancellation requested")
        }
    }

    func stopForBackground() {
        cancel()
        // Native cancellation stops future GPU submissions. Drain the current operation before
        // iOS removes GPU access; the native wait does not depend on the main actor.
        _ = session?.waitUntilIdle(timeout: 3)
    }

    func run(_ scenario: Scenario) {
        run([scenario])
    }

    func run(_ scenarios: [Scenario]) {
        guard !isRunning else { return }
        guard !scenarios.isEmpty else { return }
        report = ""
        export = BenchmarkExport(
            runID: UUID().uuidString, device: Self.machine,
            systemVersion: UIDevice.current.systemVersion, configuration: Self.configuration,
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled, startedAt: Date(),
            scenarios: scenarios.map { ScenarioExport(scenario: $0) })
        Self.log("Starting \(export!.runID): \(scenarios.map(\.rawValue).joined(separator: ", "))")
        do { try persistExport() } catch {
            progress = "Could not save benchmark results: \(error.localizedDescription)"
            return
        }
        #if targetEnvironment(simulator)
        failBeforeStarting("Use a real iPhone for this GPU benchmark.")
        return
        #else
        guard !QwenListener.shared.isWanted else {
            failBeforeStarting("Close voice mode before starting the benchmark.")
            return
        }
        guard UIApplication.shared.applicationState == .active else {
            failBeforeStarting("Keep the app in the foreground to run the benchmark.")
            return
        }
        isRunning = true
        let previousIdleTimer = UIApplication.shared.isIdleTimerDisabled
        UIApplication.shared.isIdleTimerDisabled = true
        task = Task {
            var outcome = Status.completed
            for (index, scenario) in scenarios.enumerated() {
                if Task.isCancelled { outcome = .cancelled; break }
                outcome = await measure(scenario, exportIndex: index)
                if outcome != .completed { break }
            }
            export?.status = outcome
            export?.finishedAt = Date()
            do { try persistExport() } catch {
                progress = "Could not save benchmark results: \(error.localizedDescription)"
            }
            Self.log("Run \(outcome.rawValue); Documents/MiniCPMBenchmark/latest.json")
            UIApplication.shared.isIdleTimerDisabled = previousIdleTimer
            isRunning = false
            task = nil
        }
        #endif
    }

    private func measure(_ scenario: Scenario, exportIndex: Int) async -> Status {
        currentScenario = scenario
        report = ""
        var rows: [String] = []
        var answers: [String] = []
        var sampleTask: Task<Void, Never>?
        var peak = ProcessMemory()
        var ownsGPU = false
        let working = FileManager.default.temporaryDirectory.appendingPathComponent("minicpm-bench-\(UUID().uuidString)")
        defer {
            sampleTask?.cancel()
            if ownsGPU { LocalModelHost.shared.endExternalBenchmark() }
            try? FileManager.default.removeItem(at: working)
        }
        do {
            Self.log("Scenario \(scenario.rawValue): checking model files")
            export?.scenarios[exportIndex].status = .running
            export?.scenarios[exportIndex].startedAt = Date()
            try persistExport()
            try await waitForMiniCPMFiles(.photoReader) { progress = $0 }
            progress = "Releasing the reply and speech models…"
            guard await LocalModelHost.shared.beginExternalBenchmark() else { throw BenchmarkError.busy }
            ownsGPU = true
            await QwenListener.shared.unloadForBenchmark()
            await MiniCPMHost.shared.unloadAll()
            try Task.checkCancellation()

            progress = "Checking or downloading model files…"
            let artifacts = MiniCPMArtifacts()
            let files = try await artifacts.prepare { [weak self] fraction in
                Task { @MainActor in
                    guard let self, self.isRunning, self.task?.isCancelled == false, self.session == nil else { return }
                    self.progress = "Checking or downloading model files: \(Int(fraction * 100))%"
                }
            }
            try Task.checkCancellation()
            guard UIApplication.shared.applicationState == .active else { throw CancellationError() }
            // The official demo recommends at least 4 GB device RAM. Check current headroom
            // as well, after all MLX work has drained, before allocating native model buffers.
            guard os_proc_available_memory() >= 2_300_000_000 else {
                throw BenchmarkError.memory
            }
            var image: Data?
            if scenario == .image {
                guard let png = Self.fixture().pngData() else { throw BenchmarkError.fixture }
                image = png
            }
            let native = MiniCPMSession()
            session = native
            sampleTask = Task {
                while !Task.isCancelled {
                    peak.raise(to: .now)
                    do { try await Task.sleep(for: .milliseconds(100)) } catch { break }
                }
            }
            progress = "Loading MiniCPM-V 4.7 1B…"
            Self.log("Scenario \(scenario.rawValue): loading model")
            let loadStart = ContinuousClock.now
            try await native.load(modelURL: files.model, projectorURL: files.projector)
            let loadSeconds = Self.seconds(loadStart.duration(to: .now))
            let weightsBytes = await native.info.weightsBytes
            export?.scenarios[exportIndex].loadSeconds = loadSeconds
            export?.scenarios[exportIndex].weightsBytes = weightsBytes
            let prompts = scenario == .text ? LocalBenchmark.prompts : Array(repeating:
                "Read the text at the top exactly. Name the red and blue shapes. How many green bars are there? Answer briefly.", count: 3)
            report = Self.header(loadSeconds: loadSeconds, scenario: scenario)
            for (index, prompt) in prompts.enumerated() {
                try Task.checkCancellation()
                guard UIApplication.shared.applicationState == .active else { throw CancellationError() }
                progress = "Prompt \(index + 1) of \(prompts.count)…"
                // Through the streams the app reads from, so first text is when Swift receives it.
                let streamed = try await MiniCPMStream.measure {
                    if let image {
                        return native.describePhoto(jpeg: image, system: "", instruction: prompt, maxTokens: 128, maxSlices: 1)
                    }
                    return native.reply(system: "", turns: [ChatTurn(role: .user, text: prompt)], maxTokens: 128, prefixDirectory: working)
                }
                let result = streamed.result
                peak.raise(to: .now)
                let thermal = String(describing: ProcessInfo.processInfo.thermalState)
                let firstText = streamed.firstTextSeconds.map(Self.format) ?? "–"
                rows.append("\(index + 1) | \(firstText) | \(result.promptTokens) | \(Self.format(result.prefillSeconds)) | \(result.generatedTokens) | \(Self.format(result.tokensPerSecond)) | \(result.finishReason) | \(thermal)")
                answers.append("\(index + 1). \(prompt)\n→ \(result.text)")
                report = Self.header(loadSeconds: loadSeconds, scenario: scenario)
                    + rows.joined(separator: "\n")
                    + "\n\nSampled peaks: phys_footprint \(Self.format(Double(peak.footprint) / 1e9)) GB, resident_size \(Self.format(Double(peak.resident) / 1e9)) GB\n"
                    + "Mapped weights (weights_bytes, not in phys_footprint): \(Self.format(Double(weightsBytes) / 1e9)) GB\n"
                    + "Includes the app and native buffers; sampled every 100 ms, not MLX's allocation metric.\n\n"
                    + answers.joined(separator: "\n\n")
                export?.scenarios[exportIndex].prompts.append(PromptExport(
                    index: index + 1, prompt: prompt, thermalState: thermal, firstTextSeconds: streamed.firstTextSeconds,
                    tokensPerSecond: result.tokensPerSecond, metrics: result))
                export?.scenarios[exportIndex].peakProcessFootprintBytes = peak.footprint
                export?.scenarios[exportIndex].peakResidentBytes = peak.resident
                export?.scenarios[exportIndex].report = report
                try persistExport()
                Self.log("\(scenario.rawValue) prompt \(index + 1)/\(prompts.count): first text \(firstText) s, \(Self.format(result.tokensPerSecond)) tokens/s, \(thermal)")
            }
            await native.unload()
            session = nil
            try Task.checkCancellation()
            export?.scenarios[exportIndex].status = .completed
            export?.scenarios[exportIndex].finishedAt = Date()
            try persistExport()
            progress = "Finished. Model memory released."
            Self.log("Scenario \(scenario.rawValue): completed")
            return .completed
        } catch {
            session?.cancel()
            await session?.unload()
            session = nil
            let cancelled = Task.isCancelled || error is CancellationError
            let reason = cancelled ? "Stopped; completed prompts are preserved." : error.localizedDescription
            progress = reason
            report += "\n\n\(reason)"
            export?.scenarios[exportIndex].status = cancelled ? .cancelled : .failed
            export?.scenarios[exportIndex].finishedAt = Date()
            export?.scenarios[exportIndex].peakProcessFootprintBytes = peak.footprint
            export?.scenarios[exportIndex].peakResidentBytes = peak.resident
            export?.scenarios[exportIndex].report = report
            export?.scenarios[exportIndex].error = reason
            Self.log("Scenario \(scenario.rawValue): \(reason)")
            return cancelled ? .cancelled : .failed
        }
    }

    private func failBeforeStarting(_ reason: String) {
        progress = reason
        export?.status = .failed
        export?.finishedAt = Date()
        if export?.scenarios.isEmpty == false {
            export?.scenarios[0].status = .failed
            export?.scenarios[0].error = reason
        }
        do { try persistExport() } catch {
            progress = "\(reason) Results could not be saved: \(error.localizedDescription)"
        }
    }

    /// `latest.json` is atomically replaced after each prompt. Each run also retains its
    /// own JSON and text files so a later run cannot erase a completed device measurement.
    private func persistExport() throws {
        guard let export else { return }
        let root = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
                                               appropriateFor: nil, create: true)
            .appendingPathComponent("MiniCPMBenchmark", isDirectory: true)
        let directory = root.appendingPathComponent(export.runID, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(export)
        for scenario in export.scenarios where !scenario.report.isEmpty {
            try scenario.report.write(to: directory.appendingPathComponent("\(scenario.scenario.rawValue).txt"),
                                      atomically: true, encoding: .utf8)
        }
        try data.write(to: directory.appendingPathComponent("results.json"), options: .atomic)
        try data.write(to: root.appendingPathComponent("latest.json"), options: .atomic)
    }

    private static var machine: String {
        var info = utsname()
        uname(&info)
        let capacity = MemoryLayout.size(ofValue: info.machine)
        return withUnsafePointer(to: &info.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: capacity) { String(cString: $0) }
        }
    }

    private static var configuration: String {
        #if DEBUG
        return "Debug (use Release for representative speed)"
        #else
        return "Release"
        #endif
    }

    private static func header(loadSeconds: Double, scenario: Scenario) -> String {
        return """
        MiniCPM-V 4.7 1B benchmark
        Model: Q4_K_M + F16 vision projector
        Model revision: \(MiniCPMArtifacts.revision)
        Runtime: llama.cpp-omni 96bbf61710d4 · Metal
        Device: \(machine) · iOS \(UIDevice.current.systemVersion) · \(configuration)
        Low Power Mode: \(ProcessInfo.processInfo.isLowPowerModeEnabled)
        Scenario: \(scenario == .image ? "fixed image, 3 independent repeats" : "8 independent English/Chinese prompts")
        Greedy decoding · thinking off · context 4096 · max output 128 · image slices 1
        No conversation reuse, tools, or speculative decoding. First prompt includes cold kernel work.
        Streamed as the app reads it: first text is when Swift receives the first visible text
        (the runtime's own time is nativeFirstTextSeconds in the JSON).
        GPU layer cancellation checkpoints enabled; decode rate counts synchronized forward passes.
        Load time (download and verification excluded): \(format(loadSeconds)) s

        run | first text s | prompt tok | prefill incl. vision s | output tok | decode tok/s | finish | thermal

        """
    }

    private static func log(_ message: String) {
        print("[MiniCPMBenchmark] \(message)")
        fflush(nil)
    }

    private static func format(_ value: Double) -> String { String(format: "%.2f", value) }
    private static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }

    static func fixture() -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        return UIGraphicsImageRenderer(size: CGSize(width: 720, height: 480), format: format).image { context in
            UIColor.white.setFill()
            context.fill(CGRect(x: 0, y: 0, width: 720, height: 480))
            ("SINGAPORE 47" as NSString).draw(at: CGPoint(x: 50, y: 35), withAttributes: [
                .font: UIFont.boldSystemFont(ofSize: 64), .foregroundColor: UIColor.black,
            ])
            UIColor.red.setFill()
            UIBezierPath(ovalIn: CGRect(x: 80, y: 150, width: 150, height: 150)).fill()
            UIColor.blue.setFill()
            UIBezierPath(rect: CGRect(x: 430, y: 150, width: 150, height: 150)).fill()
            UIColor.systemGreen.setFill()
            for x in [180, 310, 440] {
                UIBezierPath(rect: CGRect(x: x, y: 370, width: 80, height: 50)).fill()
            }
        }
    }

    private enum BenchmarkError: LocalizedError {
        case memory, fixture, busy
        var errorDescription: String? {
            switch self {
            case .memory: return "MiniCPM needs about 2.3 GB of available app memory. Close other apps and try again."
            case .fixture: return "The test image could not be created."
            case .busy: return "Another on-device benchmark is running."
            }
        }
    }
}
