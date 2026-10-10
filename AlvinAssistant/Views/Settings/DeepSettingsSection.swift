import AssistantKit
import SwiftUI

/// Settings › Deep thinking: when replies take longer to answer better, how, how many a day, and
/// with which model the parallel workers run.
struct DeepSettingsSection: View {
    @Bindable var store: SettingsStore

    var body: some View {
        Section {
            Picker("Deep thinking", selection: $store.settings.deepMode) {
                Text("Off").tag(DeepModeSetting.off)
                Text("When I ask").tag(DeepModeSetting.onRequest)
                Text("Automatic").tag(DeepModeSetting.automatic)
            }
            if store.settings.deepMode != .off {
                Picker("Strategy", selection: $store.settings.deepStrategy) {
                    Text("One careful answer").tag(DeepStrategy.single)
                    Text("Three perspectives").tag(DeepStrategy.parallel)
                }
                Stepper("Daily limit: \(store.settings.deepDailyLimit)", value: $store.settings.deepDailyLimit, in: 1...50)
                LabeledContent("Used today", value: "\(store.deepRunsToday()) of \(store.settings.deepDailyLimit)")
                if store.settings.deepStrategy == .parallel {
                    Picker("Worker model", selection: $store.settings.deepWorkerModel) {
                        Text("Same as the chat model").tag("")
                        ForEach(ClaudeModelCatalog.options) { option in
                            Text(option.displayName).tag(option.id)
                        }
                        if !isKnownWorkerModel {
                            Text(store.settings.deepWorkerModel).tag(store.settings.deepWorkerModel)
                        }
                    }
                }
            }
        } header: {
            Text("Deep thinking")
        } footer: {
            Text(footer)
        }
    }

    /// Whether the worker model setting is empty or one of the catalog's models.
    private var isKnownWorkerModel: Bool {
        let id = store.settings.deepWorkerModel
        return id.isEmpty || ClaudeModelCatalog.options.contains { $0.id == id }
    }

    private var footer: String {
        switch store.settings.deepMode {
        case .off:
            return "Replies always answer at the effort chosen above."
        case .onRequest, .automatic:
            var text = "Use “Think deeper” in the chat or in voice mode for a slower, more thorough answer from Claude"
            text += store.settings.deepMode == .automatic ? "; typed requests that look complex get one automatically." : "."
            switch store.settings.deepStrategy {
            case .single:
                text += " One careful answer thinks longer before replying."
            case .parallel:
                text += " Three perspectives asks a researcher, a reasoner and a critic at once, then merges their notes; it costs about four requests. A cheaper worker model lowers the cost but doesn't share the chat's cached conversation."
            }
            return text + " Voice never goes deep on its own. After the daily limit, replies are standard."
        }
    }
}
