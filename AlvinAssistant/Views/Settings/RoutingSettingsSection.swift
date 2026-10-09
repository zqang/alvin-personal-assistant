import AssistantKit
import SwiftUI

/// Settings › Routing: whether each request goes to Claude or to the on-device model, decided per
/// request by the connection, what is asked and how fast each has been answering.
struct RoutingSettingsSection: View {
    @Bindable var store: SettingsStore

    var body: some View {
        Section {
            Toggle("Automatic routing", isOn: automatic)
            if store.settings.routingMode == .automatic {
                Toggle("Prefer on-device", isOn: $store.settings.preferOnDevice)
                Toggle("Answer small talk on device", isOn: $store.settings.fastLocalSmallTalk)
            }
        } header: {
            Text("Routing")
        } footer: {
            Text(footer)
        }
    }

    private var automatic: Binding<Bool> {
        Binding(
            get: { store.settings.routingMode == .automatic },
            set: { store.settings.routingMode = $0 ? .automatic : .single }
        )
    }

    private var footer: String {
        guard store.settings.routingMode == .automatic else {
            return "Claude answers every request. Turn on automatic routing to answer on this iPhone when you're offline, and when on-device is quicker."
        }
        var text = "Claude answers questions that need the web or careful thought, and the on-device model answers when you're offline or Claude can't be reached."
        if store.settings.preferOnDevice {
            text += " Simple requests, reminders, calendar and timers stay on this iPhone."
        }
        if store.settings.fastLocalSmallTalk {
            text += " In voice chat, quick small talk is answered on this iPhone for speed."
        }
        return text + " The on-device model needs to be downloaded first."
    }
}
