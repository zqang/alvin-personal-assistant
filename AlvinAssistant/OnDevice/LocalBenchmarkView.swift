import AssistantKit
import SwiftUI
import UIKit

/// Plays a scripted conversation through the on-device model and reports time to first text,
/// prefill and generation speed, draft acceptance, cache reuse, and peak memory.
struct LocalBenchmarkView: View {
    @Bindable var store: SettingsStore
    @State private var runner = LocalBenchmarkRunner()

    var body: some View {
        List {
            Section {
                Button(runner.isRunning ? "Running…" : "Run benchmark") {
                    runner.run(settings: store.settings)
                }
                .disabled(runner.isRunning)
                if let progress = runner.progress {
                    Text(progress)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            } footer: {
                Text("Loads \(LocalModelCatalog.option(for: store.settings.localModelID)?.displayName ?? "the model") fresh, then plays \(LocalBenchmark.prompts.count) short turns in English and Chinese as one conversation, plus one turn without the cached session for comparison. Keep the app open; use a Release build for realistic speed.")
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

@MainActor
@Observable
final class LocalBenchmarkRunner {
    private(set) var isRunning = false
    private(set) var progress: String?
    private(set) var report = ""
    @ObservationIgnored private var task: Task<Void, Never>?

    func run(settings: AssistantSettings) {
        guard !isRunning else { return }
        isRunning = true
        report = ""
        task = Task {
            await benchmark(settings: settings)
            isRunning = false
        }
    }

    func cancel() {
        task?.cancel()
    }

    private func benchmark(settings: AssistantSettings) async {
        let host = LocalModelHost.shared
        let option = LocalModelCatalog.option(for: settings.localModelID)
        // A fresh load, so its time is measured and no earlier session is reused.
        host.unload(stopDownload: true)
        progress = "Downloading or loading the model…"
        host.prepare(settings)
        await host.settle()
        if case .failed(let message) = host.status {
            progress = message
            return
        }

        let system = PromptBuilder.systemPrompt(userName: settings.userName, customInstructions: "")
        var turns: [ChatTurn] = []
        var rows: [LocalBenchmark.Row] = []
        do {
            for (index, prompt) in LocalBenchmark.prompts.enumerated() {
                progress = "Turn \(index + 1) of \(LocalBenchmark.prompts.count + 1)"
                turns.append(ChatTurn(role: .user, text: prompt, context: "<context>input: spoken</context>"))
                let result = try await host.respond(system: system, turns: turns, settings: settings) { _ in }
                turns.append(ChatTurn(role: .assistant, text: result.text))
                rows.append(LocalBenchmark.Row(prompt: prompt, reply: result.text, stats: result.stats))
            }
            // The last turn again, re-reading the whole conversation, to show what the cache saves.
            progress = "Turn \(LocalBenchmark.prompts.count + 1) of \(LocalBenchmark.prompts.count + 1)"
            turns.removeLast()
            host.resetSession()
            let result = try await host.respond(system: system, turns: turns, settings: settings) { _ in }
            rows.append(LocalBenchmark.Row(prompt: "(last turn again, without the cache)", reply: result.text, stats: result.stats))
            progress = nil
        } catch {
            progress = Task.isCancelled ? "Stopped." : "Stopped: \(error.localizedDescription)"
        }
        report = LocalBenchmark.report(
            model: option?.displayName ?? settings.localModelID,
            device: Self.deviceModel(),
            speculative: host.speculation,
            loadTime: host.lastLoadTime,
            rows: rows
        )
    }

    /// The hardware identifier, e.g. "iPhone17,1".
    private static func deviceModel() -> String {
        var info = utsname()
        uname(&info)
        let machine = withUnsafeBytes(of: &info.machine) { bytes in
            String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
        }
        return "\(machine), iOS \(UIDevice.current.systemVersion)"
    }
}
