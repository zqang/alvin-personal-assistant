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

    /// Where requests go and what Claude sees. It promises no more than routing does: online, the
    /// on-device preferences apply only while the model is loaded, and Claude is sent the whole
    /// conversation, on-device replies and tool results included.
    private var footer: String {
        guard store.settings.routingMode == .automatic else {
            return "Claude answers every request and sees the whole conversation, including what the tools read, such as your events. Turn on automatic routing to answer on this iPhone when you're offline, and when on-device is quicker."
        }
        var text = "Claude answers questions that need the web or careful thought, and the on-device model answers when you're offline or Claude can't be reached."
        if store.settings.preferOnDevice {
            text += " While the on-device model is loaded, it also answers simple requests, and reminders, calendar and timers if it runs on the Alvin engine with the tools on. It's unloaded when you leave the app; until it's loaded again, or when it hands a request over, Claude answers these."
        }
        if store.settings.fastLocalSmallTalk {
            text += " In voice chat, quick small talk is answered on this iPhone for speed while the model is loaded."
        }
        text += " Whenever Claude answers, it sees the whole conversation, including replies from this iPhone and what the tools read, such as your events."
        return text + " The on-device model needs to be downloaded first."
    }
}
