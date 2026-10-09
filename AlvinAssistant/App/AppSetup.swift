import AssistantKit
import Foundation
import LocalEngine
import UserNotifications

/// Wires the app together at launch: the reply pipeline, the on-device host's prompt and network
/// seams, the engine extensions and the timer notifications. `AlvinAssistantApp.init` calls
/// `install(store:)` once, with the store the app then uses.
@MainActor
enum AppSetup {
    static func install(store: SettingsStore) {
        // Routing, client tools, deep mode and early start for the chat and voice screens.
        ReplyPipelines.make = { OrchestratedReplyPipeline(store: $0) }

        // The prewarm renders exactly the system prompt and tools that replies send.
        LocalModelHost.systemProvider = { settings in
            let prompt = LocalPromptFactory.make(settings: settings, store: store)
            return (system: prompt.system, tools: prompt.tools.definitions)
        }
        LocalModelHost.isOnline = { Connectivity.shared.isOnline }

        // The small-M kernel (used only when asked for, correct here and measurably faster). No
        // catalog checkpoint carries multi-token-prediction weights, so there is no MTP drafter
        // to add (plan WP42 was skipped).
        EngineSetup.extraExtensions = [FastKernelsExtension()]

        // Timer notifications show (with their sound) while the app is open.
        UNUserNotificationCenter.current().delegate = TimerNotificationDelegate.shared

        // Start watching the network now, so the first reply has a real path to route by.
        _ = Connectivity.shared
    }
}

// The fast kernels as `EngineSetup` drives them (see `FastKernelsProviding`): the request lives in
// the engine configuration, and the extension's report is the verdict.
extension FastKernelsExtension: FastKernelsProviding {
    typealias Verdict = FastKernelsReport

    func request(_ requested: Bool, in configuration: inout EngineConfiguration) {
        configuration.fastKernelsRequested = requested
    }
}

extension FastKernelsReport: FastKernelsVerdict {}
