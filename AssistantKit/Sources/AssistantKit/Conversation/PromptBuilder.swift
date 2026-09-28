import Foundation

/// A persisted chat message, as the prompt builder needs it.
public struct StoredMessage: Equatable, Sendable {
    public enum Status: String, Codable, Sendable {
        case complete
        case streaming
        /// The user cut the reply off; `text` holds what was spoken.
        case interrupted
        case failed
        case refused
    }

    public var role: ChatRole
    public var text: String
    public var createdAt: Date
    public var isVoice: Bool
    public var status: Status

    public init(role: ChatRole, text: String, createdAt: Date, isVoice: Bool, status: Status = .complete) {
        self.role = role
        self.text = text
        self.createdAt = createdAt
        self.isVoice = isVoice
        self.status = status
    }
}

public enum PromptBuilder {
    /// Converts stored messages into alternating turns that start and end with the user.
    ///
    /// The result is a pure function of the stored messages, so each request's history is a
    /// byte-identical prefix of the next one. That keeps the prompt cache warm.
    public static func turns(from messages: [StoredMessage], timeZone: TimeZone = .current) -> [ChatTurn] {
        let formatter = contextDateFormatter(timeZone: timeZone)
        var turns: [ChatTurn] = []
        var lastReplyInterrupted = false
        var mergedTurnFollowsInterruption = false

        for message in messages {
            let text = message.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            switch message.role {
            case .user:
                if let last = turns.last, last.role == .user {
                    // A reply failed in between: fold this message into the pending user turn.
                    let context = contextTag(for: message, formatter: formatter, timeZone: timeZone, afterInterruption: mergedTurnFollowsInterruption)
                    turns[turns.count - 1] = ChatTurn(role: .user, text: last.text + "\n\n" + text, context: context)
                } else {
                    mergedTurnFollowsInterruption = lastReplyInterrupted
                    let context = contextTag(for: message, formatter: formatter, timeZone: timeZone, afterInterruption: lastReplyInterrupted)
                    turns.append(ChatTurn(role: .user, text: text, context: context))
                }
                lastReplyInterrupted = false
            case .assistant:
                guard message.status == .complete || message.status == .interrupted else { continue }
                lastReplyInterrupted = message.status == .interrupted
                if let last = turns.last, last.role == .assistant {
                    turns[turns.count - 1].text = last.text + "\n\n" + text
                } else {
                    turns.append(ChatTurn(role: .assistant, text: text))
                }
            }
        }

        while turns.first?.role == .assistant { turns.removeFirst() }
        while turns.last?.role == .assistant { turns.removeLast() }
        return turns
    }

    static func contextTag(for message: StoredMessage, formatter: DateFormatter, timeZone: TimeZone, afterInterruption: Bool) -> String {
        var parts = [
            "time: \(formatter.string(from: message.createdAt)) \(timeZone.identifier)",
            "input: \(message.isVoice ? "spoken" : "typed")",
        ]
        if afterInterruption {
            parts.append("note: the user interrupted your previous reply")
        }
        return "<context>\(parts.joined(separator: "; "))</context>"
    }

    static func contextDateFormatter(timeZone: TimeZone) -> DateFormatter {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = timeZone
        formatter.dateFormat = "EEEE d MMMM yyyy, HH:mm"
        return formatter
    }

    /// The system prompt. It contains nothing that changes per request, so it stays cacheable.
    public static func systemPrompt(userName: String, customInstructions: String) -> String {
        let name = userName.trimmingCharacters(in: .whitespacesAndNewlines)
        let owner = name.isEmpty ? "" : " for \(name)"
        var prompt = """
        You are a warm, capable personal assistant\(owner), running as an iPhone app. The user talks to you by voice or types.

        Each user message starts with a <context> tag giving the local time it was sent and whether it was spoken or typed. Use the time when it matters, and never mention the tag.

        When the input is spoken, your reply is read aloud by a text-to-speech voice:
        - Talk like a thoughtful person in conversation: natural, warm, and to the point. Usually one to three short sentences; go longer only when asked for detail, a story, or step-by-step help.
        - Write plain spoken sentences only: no Markdown, lists, headings, tables, code, emoji, or URLs. Say numbers, dates, units, and symbols the way you would read them aloud.
        - The words come from speech recognition and may contain errors. Infer the likely meaning, and ask a short clarifying question only when it really matters.
        - If the user interrupted your previous reply, respond to what they said rather than repeating yourself.

        When the input is typed, you may use light Markdown and go into more detail when it helps.

        Reply in the language the user is using. When you search the web, answer in your own words; don't read out links or source names unless asked.

        Keep responses focused, brief, and concise to avoid overwhelming the person. Latency-sensitive; begin your visible answer immediately.
        """
        let instructions = customInstructions.trimmingCharacters(in: .whitespacesAndNewlines)
        if !instructions.isEmpty {
            prompt += "\n\n<user_instructions>\n\(instructions)\n</user_instructions>"
        }
        return prompt
    }
}
