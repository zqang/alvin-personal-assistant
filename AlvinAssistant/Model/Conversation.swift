import AssistantKit
import Foundation
import SwiftData

@Model
final class Conversation {
    static let untitled = "New chat"

    var id: UUID
    var title: String
    var createdAt: Date
    var updatedAt: Date
    @Relationship(deleteRule: .cascade, inverse: \ChatMessage.conversation)
    var messages: [ChatMessage] = []

    init() {
        id = UUID()
        title = Conversation.untitled
        createdAt = .now
        updatedAt = .now
    }

    /// Messages in the order they were added; SwiftData doesn't preserve relationship order.
    var orderedMessages: [ChatMessage] {
        messages.sorted { $0.sequence < $1.sequence }
    }

    func append(_ message: ChatMessage) {
        message.sequence = (messages.map(\.sequence).max() ?? -1) + 1
        modelContext?.insert(message)
        messages.append(message)
        updatedAt = .now
        if title == Conversation.untitled, message.role == .user {
            title = Conversation.makeTitle(from: message.text)
        }
    }

    func remove(_ message: ChatMessage) {
        messages.removeAll { $0.id == message.id }
        modelContext?.delete(message)
    }

    static func makeTitle(from text: String) -> String {
        let firstLine = text.split(whereSeparator: \.isNewline).first.map(String.init) ?? text
        return firstLine.count > 48 ? String(firstLine.prefix(47)) + "…" : firstLine
    }
}

@Model
final class ChatMessage {
    var id: UUID
    var roleRaw: String
    var text: String
    var createdAt: Date
    var sequence: Int
    var isVoice: Bool
    var statusRaw: String
    var errorText: String?
    /// JSON for `toolRounds`; nil when the reply ran no tools.
    var toolRoundsData: Data? = nil
    var conversation: Conversation?

    init(role: ChatRole, text: String, isVoice: Bool, status: StoredMessage.Status = .complete) {
        id = UUID()
        roleRaw = role.rawValue
        self.text = text
        createdAt = .now
        sequence = 0
        self.isVoice = isVoice
        statusRaw = status.rawValue
    }

    var role: ChatRole { ChatRole(rawValue: roleRaw) ?? .user }

    var status: StoredMessage.Status {
        get { StoredMessage.Status(rawValue: statusRaw) ?? .complete }
        set { statusRaw = newValue.rawValue }
    }

    /// Client tool rounds the reply ran, in order.
    var toolRounds: [ToolRound] {
        get {
            guard let toolRoundsData, let rounds = try? JSONDecoder().decode([ToolRound].self, from: toolRoundsData) else { return [] }
            return rounds
        }
        set {
            guard !newValue.isEmpty else {
                toolRoundsData = nil
                return
            }
            let encoder = JSONEncoder()
            encoder.outputFormatting = .sortedKeys
            toolRoundsData = try? encoder.encode(newValue)
        }
    }

    var stored: StoredMessage {
        StoredMessage(role: role, text: text, createdAt: createdAt, isVoice: isVoice, status: status, toolRounds: toolRounds)
    }
}
