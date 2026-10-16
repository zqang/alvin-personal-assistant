import AssistantKit
import Foundation

/// How a finished or stopped reply is recorded.
@MainActor
public enum ReplyOutcome {
    public static func apply(_ stop: ReplyStop, to reply: any VoiceMessage) {
        switch stop {
        case .refused:
            // A declined reply's partial text shouldn't be kept or sent back to the model.
            reply.text = ""
            reply.status = .refused
            reply.errorText = String(localized: "The model declined to answer this. Try rephrasing.", bundle: .voiceKit)
        case .completed, .truncated, .other:
            reply.status = .complete
        }
    }

    /// Keeps a stopped reply's text, or removes the reply if nothing arrived. A reply that ran
    /// tool rounds is kept even without text: its actions happened, so the history must show
    /// them. Returns false if removed.
    @discardableResult
    public static func keepStopped(_ reply: any VoiceMessage, text: String? = nil, in conversation: any VoiceConversation) -> Bool {
        let kept = (text ?? reply.text).trimmed
        guard !kept.isEmpty || !reply.toolRounds.isEmpty else {
            conversation.remove(reply)
            return false
        }
        reply.text = kept
        reply.status = .interrupted
        return true
    }
}
