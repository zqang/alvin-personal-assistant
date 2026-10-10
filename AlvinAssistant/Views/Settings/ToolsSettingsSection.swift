import AssistantKit
import EventKit
import SwiftUI
import UIKit

/// Settings › Tools: switches for the device tools, each with whether iOS lets the app use what it
/// needs and a button to allow it.
struct ToolsSettingsSection: View {
    @Bindable var store: SettingsStore

    @Environment(\.openURL) private var openURL
    @Environment(\.scenePhase) private var scenePhase
    @State private var access: [DeviceTools.Category: DeviceToolAccess] = [:]
    @State private var asking: DeviceTools.Category? = nil

    var body: some View {
        Section {
            Toggle("Reminders, calendar and timers", isOn: $store.settings.deviceToolsEnabled)
                // On one row only: modifiers on a Section apply to every row.
                .task { await refreshAccess() }
                .onChange(of: scenePhase) {
                    // Back from the Settings app, where the user may have changed access.
                    if scenePhase == .active {
                        Task { await refreshAccess() }
                    }
                }
            if store.settings.deviceToolsEnabled {
                ForEach(DeviceTools.Category.allCases) { category in
                    row(for: category)
                }
            }
        } header: {
            Text("Tools")
        } footer: {
            Text("The assistant can add reminders and calendar events, read your schedule and set timers when you ask, with Claude and on this iPhone. It asks iOS for access the first time.")
        }
    }

    private func row(for category: DeviceTools.Category) -> some View {
        let state = access[category] ?? .unknown
        return VStack(alignment: .leading, spacing: 6) {
            Toggle(isOn: enabled(category)) {
                Label(category.title, systemImage: category.systemImage)
            }
            HStack(spacing: 8) {
                Text(Self.statusText(state, category: category))
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                Spacer(minLength: 8)
                if Self.canAllow(state) {
                    Button("Allow access") { allow(category, state: state) }
                        .font(.footnote)
                        .buttonStyle(.borderless)
                        .disabled(asking != nil)
                }
            }
        }
        .padding(.vertical, 2)
    }

    private func enabled(_ category: DeviceTools.Category) -> Binding<Bool> {
        Binding(
            get: { DeviceTools.isEnabled(category, in: store.settings) },
            set: { isOn in
                DeviceTools.setEnabled(isOn, category, in: &store.settings)
                store.save()
            }
        )
    }

    // MARK: Access

    private func refreshAccess() async {
        var states: [DeviceTools.Category: DeviceToolAccess] = [:]
        for category in DeviceTools.Category.allCases {
            states[category] = await Self.currentAccess(category)
        }
        access = states
    }

    /// Asks iOS the first time; afterwards only the Settings app can change the answer.
    private func allow(_ category: DeviceTools.Category, state: DeviceToolAccess) {
        guard state == .notDetermined else {
            let link = category == .timers ? UIApplication.openNotificationSettingsURLString : UIApplication.openSettingsURLString
            if let url = URL(string: link) {
                openURL(url)
            }
            return
        }
        asking = category
        Task {
            let result: DeviceToolAccess
            switch category {
            case .reminders:
                result = await EventStoreService.shared.requestAccess(.reminder)
            case .calendar:
                result = await EventStoreService.shared.requestAccess(.event)
            case .timers:
                result = await TimerService.shared.requestAccess()
            }
            access[category] = result
            asking = nil
        }
    }

    private static func currentAccess(_ category: DeviceTools.Category) async -> DeviceToolAccess {
        switch category {
        case .reminders:
            return EventStoreService.access(for: .reminder)
        case .calendar:
            return EventStoreService.access(for: .event)
        case .timers:
            return await TimerService.shared.access()
        }
    }

    private static func canAllow(_ state: DeviceToolAccess) -> Bool {
        switch state {
        case .notDetermined, .denied, .addOnly:
            return true
        case .unknown, .granted, .restricted:
            return false
        }
    }

    private static func statusText(_ state: DeviceToolAccess, category: DeviceTools.Category) -> String {
        switch state {
        case .unknown:
            return "Checking access…"
        case .notDetermined:
            return "iOS asks for access the first time it's used."
        case .granted:
            return category == .timers ? "Notifications allowed." : "Access allowed."
        case .addOnly:
            return "Can add events but not read them."
        case .denied:
            return category == .timers ? "Notifications are off in iOS Settings." : "Access is off in iOS Settings."
        case .restricted:
            return "Restricted on this iPhone."
        }
    }
}
