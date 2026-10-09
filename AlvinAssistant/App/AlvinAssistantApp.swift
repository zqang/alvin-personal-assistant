import SwiftData
import SwiftUI

@main
struct AlvinAssistantApp: App {
    @State private var store: SettingsStore

    init() {
        // The store first, then the launch setup that reads it.
        let store = SettingsStore()
        AppSetup.install(store: store)
        _store = State(initialValue: store)
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(store)
        }
        .modelContainer(for: [Conversation.self, ChatMessage.self])
    }
}

extension String {
    var trimmed: String { trimmingCharacters(in: .whitespacesAndNewlines) }
}
