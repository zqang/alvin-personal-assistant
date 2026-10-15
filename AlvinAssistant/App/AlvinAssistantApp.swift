import SwiftData
import SwiftUI

@main
struct AlvinAssistantApp: App {
    @State private var store: SettingsStore
    private let benchmarkLaunchMode = MiniCPMBenchmarkView.LaunchMode.requested
    private let compareLocalModels = ProcessInfo.processInfo.arguments.contains("--compare-local-models")
    private let compare35B = ProcessInfo.processInfo.arguments.contains("--compare-local-models-35b")
    private let probeDevice = ProcessInfo.processInfo.arguments.contains("--benchmark-device-probe")

    init() {
        // The store first, then the launch setup that reads it.
        let store = SettingsStore()
        AppSetup.install(store: store)
        _store = State(initialValue: store)
    }

    var body: some Scene {
        WindowGroup {
            Group {
                if probeDevice {
                    BenchmarkDeviceProbeView()
                } else if compareLocalModels || compare35B {
                    NavigationStack {
                        ModelComparisonBenchmarkView(settings: store.settings, variant: compare35B ? .moe35B : .dense1B)
                    }
                } else if let benchmarkLaunchMode {
                    NavigationStack {
                        MiniCPMBenchmarkView(launchMode: benchmarkLaunchMode)
                    }
                } else {
                    RootView()
                }
            }
            .environment(store)
        }
        .modelContainer(for: [Conversation.self, ChatMessage.self])
    }
}

extension String {
    var trimmed: String { trimmingCharacters(in: .whitespacesAndNewlines) }
}
