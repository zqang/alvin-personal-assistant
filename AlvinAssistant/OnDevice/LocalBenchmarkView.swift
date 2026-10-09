import AssistantKit
import LocalEngine
import SwiftUI
import UIKit

/// Plays a scripted scenario (`LocalBenchmark.scenarios`) through the on-device model and reports
/// time to first text, prefill and generation speed, cache reuse, speculative decoding, tool-call
/// accuracy and peak memory. "Compare engines" plays the scenario with the Alvin engine, then with
/// MLX's stock session over the same loaded model.
struct LocalBenchmarkView: View {
    @Bindable var store: SettingsStore
    @State private var runner = LocalBenchmarkRunner()
    @State private var scenarioID = LocalBenchmark.continued.id

    private var scenario: LocalBenchmark.Scenario {
        LocalBenchmark.scenario(id: scenarioID) ?? LocalBenchmark.continued
    }

    var body: some View {
        List {
            Section {
                Picker("Scenario", selection: $scenarioID) {
                    ForEach(LocalBenchmark.scenarios) { scenario in
                        Text(scenario.title).tag(scenario.id)
                    }
                }
                .disabled(runner.isRunning)
                Button(runner.isRunning ? "Running…" : "Run") {
                    runner.run(scenario, settings: store.settings, compareEngines: false)
                }
                .disabled(runner.isRunning)
                Button("Compare engines") {
                    runner.run(scenario, settings: store.settings, compareEngines: true)
                }
                .disabled(runner.isRunning)
                if let progress = runner.progress {
                    Text(progress)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            } footer: {
                Text("Loads \(LocalModelCatalog.option(for: store.settings.localModelID)?.displayName ?? "the model") fresh, then plays “\(scenario.title)” (\(scenario.turnCount) turns). Tool calls are recorded, never carried out. Keep the app open; use a Release build for realistic speed.")
            }

            Section {
                Button("Measure speed (cost curve)") {
                    runner.measureSpeed(settings: store.settings)
                }
                .disabled(runner.isRunning)
                Button("Run self-test") {
                    runner.runSelfTest(settings: store.settings)
                }
                .disabled(runner.isRunning)
                if LocalModelHost.shared.fastKernelsAvailable {
                    Button("Test fast kernels") {
                        runner.measureFastKernels(settings: store.settings)
                    }
                    .disabled(runner.isRunning)
                }
            } footer: {
                Text("The cost curve is how long checking several tokens at once takes on this iPhone; speculative decoding uses it. The self-test checks that the Alvin engine gives the same answers as a fresh run. The fast-kernel test compares the cost curve with and without the fast GPU kernels.")
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
        .navigationTitle("Benchmark")
        .navigationBarTitleDisplayMode(.inline)
        .onDisappear { runner.cancel() }
    }
}

// MARK: - Runner

@MainActor
@Observable
final class LocalBenchmarkRunner {
    private(set) var isRunning = false
    private(set) var progress: String?
    private(set) var report = ""
    @ObservationIgnored private var task: Task<Void, Never>?

    /// Plays `scenario` with the engine `settings` choose, or, with `compareEngines`, with the
    /// Alvin engine and then the stock session.
    func run(_ scenario: LocalBenchmark.Scenario, settings: AssistantSettings, compareEngines: Bool) {
        start { [self] in
            await benchmark(scenario, settings: settings, compareEngines: compareEngines)
        }
    }

    /// Measures the engine's cost curve on this iPhone and stores it.
    func measureSpeed(settings: AssistantSettings) {
        start { [self] in
            guard await ensureLoaded(settings) else { return "" }
            progress = "Measuring…"
            defer { progress = nil }
            do {
                let curve = try await LocalModelHost.shared.measureSpeed()
                return Self.curveReport(curve, model: Self.modelName(settings))
            } catch {
                return "The speed measurement stopped: \(error.localizedDescription)"
            }
        }
    }

    /// Runs the engine self-test and stores its result.
    func runSelfTest(settings: AssistantSettings) {
        start { [self] in
            guard await ensureLoaded(settings) else { return "" }
            progress = "Checking the engine…"
            defer { progress = nil }
            guard let result = await LocalModelHost.shared.runSelfTest() else {
                return "The self-test was interrupted. Keep the app open and try again."
            }
            return Self.selfTestReport(result, model: Self.modelName(settings))
        }
    }

    /// Compares the cost curve with and without the fast kernels and stores the gain.
    func measureFastKernels(settings: AssistantSettings) {
        start { [self] in
            guard await ensureLoaded(settings) else { return "" }
            progress = "Testing the fast kernels…"
            defer { progress = nil }
            do {
                let summary = try await LocalModelHost.shared.measureFastKernels()
                return [
                    "On-device fast kernels",
                    "Model: \(Self.modelName(settings))",
                    "Device: \(Self.deviceModel())",
                    "",
                    summary,
                    LocalModelHost.shared.fastKernelsSummary,
                ].joined(separator: "\n")
            } catch {
                return "The fast-kernel test stopped: \(error.localizedDescription)"
            }
        }
    }

    func cancel() {
        task?.cancel()
    }

    private func start(_ work: @escaping @MainActor () async -> String) {
        guard !isRunning else { return }
        isRunning = true
        report = ""
        progress = nil
        task = Task {
            let text = await work()
            report = text
            isRunning = false
        }
    }

    /// Loads the model if it isn't; false, with `progress` saying why, when it can't.
    private func ensureLoaded(_ settings: AssistantSettings) async -> Bool {
        let host = LocalModelHost.shared
        progress = "Downloading or loading the model…"
        host.prepare(settings, prewarm: false)
        await host.settle()
        if case .failed(let message) = host.status {
            progress = message
            return false
        }
        guard host.status == .ready else {
            progress = Task.isCancelled ? "Stopped." : "The model didn't load. Keep the app open and try again."
            return false
        }
        progress = nil
        return true
    }

    // MARK: Scenarios

    private func benchmark(_ scenario: LocalBenchmark.Scenario, settings: AssistantSettings, compareEngines: Bool) async -> String {
        let host = LocalModelHost.shared
        // A fresh load, so its time is measured and no earlier session is reused.
        host.unload(stopDownload: true)
        guard await ensureLoaded(settings) else { return "" }
        let loadTime = host.lastLoadTime

        let modes: [LocalEngineMode] = compareEngines ? [.alvin, .stock] : [settings.localEngineMode]
        var reports: [String] = []
        for mode in modes {
            var runSettings = settings
            runSettings.localEngineMode = mode
            let label: String? = compareEngines ? (mode == .alvin ? "Alvin engine" : "MLX stock session") : nil
            let (rows, stopped) = await play(scenario, settings: runSettings, label: label)
            var title = scenario.title
            if let label { title += " · " + label }
            if let stopped { title += " (\(stopped))" }
            reports.append(LocalBenchmark.report(
                model: Self.modelName(settings),
                device: Self.deviceModel(),
                speculative: host.speculation,
                loadTime: loadTime,
                rows: rows,
                scenario: title
            ))
            if stopped != nil { break }
        }
        progress = nil
        return reports.joined(separator: "\n\n" + String(repeating: "─", count: 32) + "\n\n")
    }

    /// Plays the scenario's actions; returns the rows and, if it stopped early, why.
    private func play(_ scenario: LocalBenchmark.Scenario, settings: AssistantSettings, label: String?) async -> (rows: [LocalBenchmark.Row], stopped: String?) {
        let host = LocalModelHost.shared
        var settings = settings
        let system = PromptBuilder.systemPrompt(userName: settings.userName, customInstructions: "")
        let tools = BenchmarkToolExecutor(definitions: DeviceTools.localRegistry(settings: settings, handoff: false).definitions)
        // The continued chat ends with its last turn played again without the cache.
        let replayLast = scenario.id == LocalBenchmark.continued.id
        let total = scenario.turnCount + (replayLast ? 1 : 0)
        let prefix = label.map { "\($0): " } ?? ""
        var turns: [ChatTurn] = []
        var rows: [LocalBenchmark.Row] = []
        var number = 0
        do {
            for action in scenario.actions {
                try Task.checkCancellation()
                switch action {
                case .turn(let turn):
                    number += 1
                    progress = "\(prefix)turn \(number) of \(total)"
                    turns.append(ChatTurn(role: .user, text: turn.prompt, context: turn.context))
                    let outcome = try await reply(turns: turns, system: system, settings: settings, tools: tools, cancelAfter: turn.cancelAfterTokens)
                    // What the app would keep: after a barge-in, only the words spoken so far.
                    let kept = turn.storedWords.map { LocalBenchmark.spokenPrefix(of: outcome.text, words: $0) } ?? outcome.text
                    if !kept.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !outcome.rounds.isEmpty {
                        turns.append(ChatTurn(role: .assistant, text: kept, toolRounds: outcome.rounds))
                    }
                    var prompt = turn.prompt
                    if outcome.cutOff, let tokens = turn.cancelAfterTokens {
                        prompt += " (cut off after \(tokens) tokens)"
                    }
                    let call = outcome.rounds.first?.calls.first
                    let score = turn.expectation.map { $0.score(name: call?.name, arguments: call?.input) }
                    rows.append(LocalBenchmark.Row(prompt: prompt, reply: kept, stats: outcome.stats, toolScore: score))
                case .reloadModel(let usePrefixCache):
                    progress = "\(prefix)reloading the model…"
                    settings.localPrefixCache = usePrefixCache
                    host.unload(stopDownload: true)
                    guard await ensureLoaded(settings) else {
                        return (rows, progress ?? "the model didn't load")
                    }
                case .newConversation:
                    turns = []
                }
            }
            if replayLast, turns.last?.role == .assistant {
                number += 1
                progress = "\(prefix)turn \(number) of \(total)"
                turns.removeLast()
                host.resetSession()
                let outcome = try await reply(turns: turns, system: system, settings: settings, tools: tools, cancelAfter: nil)
                rows.append(LocalBenchmark.Row(prompt: "(last turn again, without the cache)", reply: outcome.text, stats: outcome.stats))
            }
            return (rows, nil)
        } catch {
            return (rows, Task.isCancelled ? "stopped" : "stopped: \(error.localizedDescription)")
        }
    }

    private struct Outcome {
        var text: String
        var stats: LocalGenerationStats
        var rounds: [ToolRound]
        /// The reply was cancelled on purpose after `cancelAfter` tokens.
        var cutOff: Bool
    }

    /// One reply; with `cancelAfter`, cancelled once its text reaches that many tokens, as a
    /// barge-in would.
    private func reply(turns: [ChatTurn], system: String, settings: AssistantSettings, tools: BenchmarkToolExecutor, cancelAfter: Int?) async throws -> Outcome {
        let host = LocalModelHost.shared
        let recorder = ReplyRecorder()
        let reply = Task { @MainActor in
            try await host.respond(system: system, turns: turns, settings: settings, tools: tools) { event in
                recorder.record(event)
                if let cancelAfter, !recorder.cutOff, (host.tokenCount(recorder.text) ?? recorder.chunks) >= cancelAfter {
                    recorder.cutOff = true
                    withUnsafeCurrentTask { $0?.cancel() }
                }
            }
        }
        // Stopping the benchmark stops the reply.
        let result = await withTaskCancellationHandler {
            await reply.result
        } onCancel: {
            reply.cancel()
        }
        switch result {
        case .success(let value):
            return Outcome(text: value.text, stats: value.stats, rounds: recorder.rounds, cutOff: false)
        case .failure(let error):
            guard recorder.cutOff, !Task.isCancelled else { throw error }
            let stats = LocalGenerationStats(
                generatedTokens: host.tokenCount(recorder.text) ?? recorder.chunks,
                engine: host.usesEngine(settings) ? "alvin" : "stock"
            )
            return Outcome(text: recorder.text, stats: stats, rounds: recorder.rounds, cutOff: true)
        }
    }

    // MARK: Reports

    static func curveReport(_ curve: CostCurve, model: String) -> String {
        var lines = [
            "On-device cost curve",
            "Model: \(model)",
            "Device: \(deviceModel())",
            "",
            "rows | ms per forward | × one row",
        ]
        for (rows, seconds) in curve.seconds.sorted(by: { $0.key < $1.key }) {
            lines.append("\(rows) | \(String(format: "%.1f", seconds * 1000)) | \(String(format: "%.2f", curve.relative(rows)))")
        }
        return lines.joined(separator: "\n")
    }

    static func selfTestReport(_ result: EngineSelfTest.Result, model: String) -> String {
        [
            "On-device engine self-test",
            "Model: \(model)",
            "Device: \(deviceModel())",
            "Result: \(result.passed ? "passed" : "FAILED") in \(String(format: "%.1f", result.seconds)) s (format \(result.formatVersion))",
            "",
            result.detail,
        ].joined(separator: "\n")
    }

    private static func modelName(_ settings: AssistantSettings) -> String {
        LocalModelCatalog.option(for: settings.localModelID)?.displayName ?? settings.localModelID
    }

    /// The hardware identifier and iOS version, e.g. "iPhone17,1, iOS 26.0".
    static func deviceModel() -> String {
        var info = utsname()
        uname(&info)
        let machine = withUnsafeBytes(of: &info.machine) { bytes in
            String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
        }
        return "\(machine), iOS \(UIDevice.current.systemVersion)"
    }
}

/// What a benchmark reply streamed.
@MainActor
private final class ReplyRecorder {
    var text = ""
    var chunks = 0
    var rounds: [ToolRound] = []
    var cutOff = false

    func record(_ event: AssistantEvent) {
        switch event {
        case .reply(.text(let chunk)):
            text += chunk
            chunks += 1
        case .toolRound(let round):
            rounds.append(round)
        default:
            break
        }
    }
}

/// Offers the on-device tools but carries none out: every call gets a neutral result, so a
/// benchmark never changes the user's reminders, calendar or timers.
struct BenchmarkToolExecutor: ToolExecutor {
    let definitions: [ToolDefinition]

    func presentation(for name: String) -> ToolPresentation {
        .generic
    }

    func run(_ calls: [PendingToolCall], context: ToolContext) async -> ToolRound {
        ToolRound(calls: calls.map { call in
            ToolCallRecord(call: call, output: .ok(.object(["ok": .bool(true)])))
        })
    }
}
