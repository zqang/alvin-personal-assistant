import AssistantKit
import Foundation

/// Replies from the model running on the iPhone (see `LocalEngineHost`), the one
/// `settings.localModelID` picks.
///
/// With the Alvin engine, `tools` are offered to the model and run in up to three rounds per
/// reply; side-effect tools wait for `commitGate` when one is given. `handoff_to_cloud` hands the
/// request to the cloud assistant (the stream throws `ReplyHandoff`) only when
/// `handoffAvailable`. The stock session offers no tools.
public struct LocalProvider: AssistantProvider {
    public let settings: AssistantSettings
    public let tools: (any ToolExecutor)?
    public let handoffAvailable: Bool
    public let commitGate: CommitGate?

    public init(settings: AssistantSettings, tools: (any ToolExecutor)?, handoffAvailable: Bool, commitGate: CommitGate?) {
        self.settings = settings
        self.tools = tools
        self.handoffAvailable = handoffAvailable
        self.commitGate = commitGate
    }

    public func streamEvents(system: String, turns: [ChatTurn]) -> AsyncThrowingStream<AssistantEvent, Error> {
        let settings = settings
        let tools = tools
        let handoffAvailable = handoffAvailable
        let commitGate = commitGate
        return AsyncThrowingStream { continuation in
            let task = Task { @MainActor in
                do {
                    try await LocalEngineHost.shared.respond(
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
