import AssistantKit
import SwiftUI

/// Settings › On-device engine: which generator answers on-device replies, speculative decoding,
/// the system prompt kept on disk, the draft model, multi-token prediction and the fast kernels,
/// with the engine's self-test, its measured speed and the last reply's drafting.
struct OnDeviceEngineSection: View {
    @Bindable var store: SettingsStore

    private var host: LocalModelHost { LocalModelHost.shared }

    var body: some View {
        Section {
            Picker("Engine", selection: $store.settings.localEngineMode) {
                Text("Automatic").tag(LocalEngineMode.automatic)
                Text("Alvin engine").tag(LocalEngineMode.alvin)
                Text("MLX stock").tag(LocalEngineMode.stock)
            }
            Picker("Speculative decoding", selection: $store.settings.localSpeculation) {
                Text("Off").tag(LocalSpeculationMode.off)
                Text("Tool calls only").tag(LocalSpeculationMode.toolsOnly)
                Text("Automatic").tag(LocalSpeculationMode.automatic)
            }
            Toggle("Keep the system prompt on disk", isOn: $store.settings.localPrefixCache)
            if LocalModelCatalog.option(for: store.settings.localModelID)?.supportsSpeculativeDecoding == true {
                Toggle("Draft model", isOn: $store.settings.localSpeculativeDecoding)
            }
            if host.hasMTPWeights {
                Toggle("Multi-token prediction", isOn: $store.settings.localMTP)
            }
            // Offered once a measurement on this iPhone showed the kernels help; always
            // switchable off.
            Toggle("Fast GPU kernels", isOn: $store.settings.localFastKernels)
                .disabled(!host.fastKernelsHelp && !store.settings.localFastKernels)

            if let statusText = host.statusText {
                Text(statusText)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            LabeledContent("Self-test", value: host.selfTestSummary)
            LabeledContent("Speed", value: host.costCurveSummary)
            LabeledContent("Last reply", value: host.lastReplySummary)
            DisclosureGroup("Details") {
                Text(host.engineSummary)
                    .font(.footnote.monospaced())
                    .textSelection(.enabled)
            }
            Button("Run self-test") {
                Task { _ = await host.runSelfTest() }
            }
            .disabled(!canCheck)
            Button("Measure speed") {
                Task { _ = try? await host.measureSpeed() }
            }
            .disabled(!canCheck)
            NavigationLink("Benchmark") {
                LocalBenchmarkView(store: store)
            }
        } header: {
            Text("On-device engine")
        } footer: {
            Text("The Alvin engine keeps the conversation in memory between replies, keeps the processed system prompt on disk, checks several drafted words at once, and lets the model use your reminders, calendar and timers. “Automatic” uses it once its self-test has passed on this iPhone for this model and app version, and MLX's stock session until then. Speed is measured once per model on this iPhone. Checks need the model loaded.")
        }
    }

    private var canCheck: Bool {
        host.status == .ready && !host.isChecking
    }
}
