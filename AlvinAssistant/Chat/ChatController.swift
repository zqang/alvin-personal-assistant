import AssistantKit
import Foundation
import Observation

/// Sends typed messages and streams the replies into the conversation.
@MainActor
@Observable
final class ChatController {
    private(set) var isResponding = false
    private(set) var activity: String?
    @ObservationIgnored private var task: Task<Void, Never>?

    func send(_ text: String, in conversation: Conversation, store: SettingsStore) {
        let text = text.trimmed
        guard !text.isEmpty, !isResponding else { return }
        conversation.append(ChatMessage(role: .user, text: text, isVoice: false))
        respond(in: conversation, store: store)
    }

    func retry(_ failed: ChatMessage, in conversation: Conversation, store: SettingsStore) {
        guard !isResponding else { return }
        conversation.remove(failed)
        respond(in: conversation, store: store)
    }

    func stop() {
        task?.cancel()
    }

    private func respond(in conversation: Conversation, store: SettingsStore) {
        let stream = ReplyService.stream(for: conversation, store: store)
        let reply = ChatMessage(role: .assistant, text: "", isVoice: false, status: .streaming)
        conversation.append(reply)
        isResponding = true
        activity = nil

        task = Task {
            var failure: Error?
            do {
                for try await event in stream {
                    switch event {
                    case .text(let text):
                        reply.text += text
                        activity = nil
                    case .activity(let description):
                        activity = description
                    case .finished(let stop):
                        ReplyOutcome.apply(stop, to: reply)
                    }
                }
            } catch {
                failure = error
            }

            if Task.isCancelled {
                ReplyOutcome.keepStopped(reply, in: conversation)
            } else if let failure {
                reply.status = .failed
                reply.errorText = failure.localizedDescription
            } else if reply.status == .streaming {
                reply.status = .complete
            }
            activity = nil
            isResponding = false
            conversation.updatedAt = .now
            try? conversation.modelContext?.save()
        }
    }
}
