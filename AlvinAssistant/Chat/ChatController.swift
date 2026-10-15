import AssistantKit
import Foundation
import MiniCPM
import Observation

/// Sends typed messages and streams the replies into the conversation, through the app's
/// `ReplyPipeline`.
@MainActor
@Observable
final class ChatController {
    private(set) var isResponding = false
    private(set) var activity: String?
    /// The engine answering the latest reply, once the pipeline reported its route (and again
    /// after a fallback).
    private(set) var routedEngine: ReplyEngine?
    /// The last route of each reply this controller streamed, by message id, for the "On device"
    /// badge. Only replies streamed while this chat is open have one.
    private(set) var routes: [UUID: RouteDecision] = [:]
    /// The photo sent with the latest user message while its read finishes, for the live note.
    private(set) var reading: (message: UUID, photo: PendingPhoto)?
    /// Replies this controller streamed for a "Think deeper" request, so a retry asks again.
    @ObservationIgnored private var deepReplies: Set<UUID> = []
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var cachedPipeline: (store: SettingsStore, pipeline: any ReplyPipeline)?

    /// The pipeline replies go through, made once per store by `ReplyPipelines.make`.
    func pipeline(for store: SettingsStore) -> any ReplyPipeline {
        if let cachedPipeline, cachedPipeline.store === store {
            return cachedPipeline.pipeline
        }
        let made = ReplyPipelines.make(store)
        cachedPipeline = (store, made)
        return made
    }

    /// Sends `text`, with `photo` when one is attached, and streams the reply. A photo without
    /// words asks what's in it. Returns false (and sends nothing) while a reply is in progress or
    /// when there's neither text nor a photo.
    @discardableResult
    func send(_ text: String, photo: PendingPhoto? = nil, in conversation: Conversation, store: SettingsStore) -> Bool {
        send(text, photo: photo, deep: false, in: conversation, store: store)
    }

    /// Sends `text` and asks for a deep-mode reply ("Think deeper"). The router still decides:
    /// without a cloud key, offline, or with today's deep replies used up, the reply is a standard
    /// one.
    @discardableResult
    func sendDeep(_ text: String, photo: PendingPhoto? = nil, in conversation: Conversation, store: SettingsStore) -> Bool {
        send(text, photo: photo, deep: true, in: conversation, store: store)
    }

    /// Answers again after a failed reply. A failed reply that ran tool rounds stays, as an
    /// interrupted reply without text: its actions happened, so the history keeps them. Any other
    /// failed reply is removed. A reply that asked for deep mode, or was routed to it, asks again;
    /// the router still checks the budget and the connection. A photo whose read didn't finish, or
    /// failed (perhaps for want of memory, or before its model was downloaded), is read again.
    /// Does nothing when `canRetry` is false.
    func retry(_ failed: ChatMessage, in conversation: Conversation, store: SettingsStore) {
        guard !isResponding, canRetry(failed) else { return }
        for message in conversation.orderedMessages where message.photoNote == PendingPhoto.unreadable {
            message.photoNote = nil
        }
        let deep = deepReplies.contains(failed.id) || routes[failed.id]?.mode == .deep
        deepReplies.remove(failed.id)
        if failed.toolRounds.isEmpty {
            routes[failed.id] = nil
            conversation.remove(failed)
        } else {
            // Its partial text never counted in the history (a failed reply's text doesn't), so
            // it doesn't now either; the retry's answer takes its place.
            failed.text = ""
            failed.status = .interrupted
            failed.errorText = nil
        }
        respond(in: conversation, store: store, deep: deep)
    }

    /// Whether `failed` may be answered again: not when it called a tool that acts, such as
    /// adding a reminder, an event or a timer. The retry's request ends at the user's message
    /// without the reply's rounds (`PromptBuilder` drops a trailing reply), so the model would
    /// act a second time. Tools that only read may run again.
    func canRetry(_ failed: ChatMessage) -> Bool {
        let rounds = failed.toolRounds
        guard !rounds.isEmpty else { return true }
        let repeatable = Self.repeatableToolNames
        return rounds.allSatisfy { round in
            round.calls.allSatisfy { repeatable.contains($0.name) }
        }
    }

    /// The tools that change nothing outside the conversation. A name the app doesn't know
    /// counts as one that acts.
    private static var repeatableToolNames: Set<String> {
        var names = Set(DeviceTools.allTools.filter { $0.effect == .readOnly }.map(\.definition.name))
        names.insert(HandoffTool.name)
        return names
    }

    func stop() {
        task?.cancel()
    }

    /// The engine that answered `message`, when this controller streamed it.
    func engine(for message: ChatMessage) -> ReplyEngine? {
        routes[message.id]?.engine
    }

    /// What the iPhone has read so far in `message`'s photo, while the read runs.
    func livePhotoNote(for message: ChatMessage) -> String? {
        reading?.message == message.id ? reading?.photo.note : nil
    }

    private func send(_ text: String, photo: PendingPhoto?, deep: Bool, in conversation: Conversation, store: SettingsStore) -> Bool {
        var text = text.trimmed
        if text.isEmpty, photo != nil {
            text = PendingPhoto.defaultQuestion(localeIdentifier: store.settings.speechLocale)
        }
        guard !text.isEmpty, !isResponding else { return false }
        let message = ChatMessage(role: .user, text: text, isVoice: false)
        message.photoData = photo?.jpeg
        conversation.append(message)
        respond(in: conversation, store: store, deep: deep, attached: photo.map { (message, $0) })
        return true
    }

    private func respond(in conversation: Conversation, store: SettingsStore, deep: Bool, attached: (user: ChatMessage, photo: PendingPhoto)? = nil) {
        let pipeline = pipeline(for: store)
        let messages = conversation.orderedMessages
        let reply = ChatMessage(role: .assistant, text: "", isVoice: false, status: .streaming)
        conversation.append(reply)
        if deep {
            deepReplies.insert(reply.id)
        }
        isResponding = true
        activity = nil
        routedEngine = nil
        reading = attached.map { ($0.user.id, $0.photo) }

        task = Task {
            var failure: Error?
            do {
                // A photo's note goes in with its turn, so the request waits for the read.
                try await PendingPhoto.noteUnread(messages, attached: attached, read: { MiniCPMHost.shared.readPhoto($0) }) {
                    activity = "Reading the photo…"
                }
                reading = nil
                activity = nil
                // Reads the conversation now; the empty reply above is no turn of its own yet.
                let stream = pipeline.stream(ReplyRequest(conversation: conversation, inputIsVoice: false, deep: deep))
                for try await event in stream {
                    switch event {
                    case .reply(.text(let text)):
                        reply.text += text
                        activity = nil
                    case .reply(.activity(let description)):
                        activity = description
                    case .reply(.finished(let stop)):
                        ReplyOutcome.apply(stop, to: reply)
                    case .toolRound(let round):
                        reply.toolRounds.append(round)
                    case .routed(let decision):
                        routedEngine = decision.engine
                        routes[reply.id] = decision
                    case .cue, .progress:
                        // Spoken cues are for voice; latency marks aren't shown.
                        break
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
            reading = nil
            isResponding = false
            conversation.updatedAt = .now
            try? conversation.modelContext?.save()
        }
    }
}
