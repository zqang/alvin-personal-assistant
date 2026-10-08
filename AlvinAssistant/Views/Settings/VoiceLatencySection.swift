import AssistantKit
import SwiftUI

/// Settings for how soon voice mode answers: early start, spoken cues, the turn chime, ducking,
/// and a report of recent voice turns' latency.
struct VoiceLatencySection: View {
    @Bindable var store: SettingsStore

    var body: some View {
        Section {
            Picker("Start answering early", selection: $store.settings.earlyReplyStart) {
                Text("Off").tag(EarlyStartSetting.off)
                Text("On-device only").tag(EarlyStartSetting.onDeviceOnly)
                Text("Automatic").tag(EarlyStartSetting.automatic)
            }
            Toggle("Spoken cues", isOn: $store.settings.spokenCues)
            if store.settings.spokenCues {
                VStack(alignment: .leading) {
                    Text("Say “One moment” after: \(store.settings.fillerDelay, format: .number.precision(.fractionLength(1))) s")
                    Slider(value: $store.settings.fillerDelay, in: 1.0...4.0, step: 0.1)
                }
            }
            Toggle("Chime when I finish speaking", isOn: $store.settings.turnChime)
            Toggle("Lower the voice when I talk over it", isOn: $store.settings.bargeInDucking)
            NavigationLink("Latency report") {
                LatencyReportScreen()
            }
        } header: {
            Text("Voice responsiveness")
        } footer: {
            Text("Starting early prepares the answer while you pause, before your turn is over; nothing is said or done until it's final. “On-device only” does this only for on-device replies, so no extra cloud requests are made. Spoken cues are short phrases like “Let me check” while an answer is on its way; they need echo cancellation. Lowering the voice needs echo cancellation and interrupting by talking.")
        }
    }
}

/// p50 and p90 latency of recent voice turns, per engine.
private struct LatencyReportScreen: View {
    private var log: LatencyLog { LatencyLog.shared }

    var body: some View {
        ScrollView {
            Text(log.summary())
                .font(.footnote.monospaced())
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
        }
        .navigationTitle("Latency report")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                ShareLink(item: log.summary())
            }
            ToolbarItem(placement: .bottomBar) {
                Button("Clear", role: .destructive) { log.clear() }
                    .disabled(log.traces.isEmpty)
            }
        }
    }
}
