import AssistantKit
import Foundation

/// Replies from the model running on the iPhone (see `LocalModelHost`).
///
/// With the Alvin engine, `tools` are offered to the model and run in up to three rounds per
/// reply; side-effect tools wait for `commitGate` when one is given. `handoff_to_cloud` hands the
/// request to the cloud assistant (the stream throws `ReplyHandoff`) only when
/// `handoffAvailable`. The stock session offers no tools.
///
/// As a plain `ChatProvider` (`streamReply`) it streams only the `.reply` events.
struct LocalProvider: AssistantProvider {
    let settings: AssistantSettings
    let tools: (any ToolExecutor)?
    let handoffAvailable: Bool
    let commitGate: CommitGate?

    init(settings: AssistantSettings, tools: (any ToolExecutor)? = nil, handoffAvailable: Bool = false, commitGate: CommitGate? = nil) {
        self.settings = settings
        self.tools = tools
        self.handoffAvailable = handoffAvailable
        self.commitGate = commitGate
    }

    func streamEvents(system: String, turns: [ChatTurn]) -> AsyncThrowingStream<AssistantEvent, Error> {
        let settings = settings
        let tools = tools
        let handoffAvailable = handoffAvailable
        let commitGate = commitGate
        return AsyncThrowingStream { continuation in
            let task = Task { @MainActor in
                do {
                    try await LocalModelHost.shared.respond(
                        system: system,
                        turns: turns,
                        settings: settings,
                        tools: tools,
                        handoffAvailable: handoffAvailable,
                        commitGate: commitGate
                    ) { event in
                        continuation.yield(event)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
