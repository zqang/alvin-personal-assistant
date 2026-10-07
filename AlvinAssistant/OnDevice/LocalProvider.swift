import AssistantKit
import Foundation

/// Replies from the model running on the iPhone (see `LocalModelHost`).
struct LocalProvider: ChatProvider {
    let settings: AssistantSettings

    func streamReply(system: String, turns: [ChatTurn]) -> AsyncThrowingStream<ReplyEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { @MainActor in
                do {
                    try await LocalModelHost.shared.respond(system: system, turns: turns, settings: settings) { event in
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
