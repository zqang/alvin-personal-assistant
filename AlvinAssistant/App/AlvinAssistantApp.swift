import SwiftData
import SwiftUI

@main
struct AlvinAssistantApp: App {
    @State private var store = SettingsStore()

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
