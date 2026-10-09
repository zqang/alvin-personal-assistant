import AssistantKit
import Foundation
import MLXLMCommon

/// Renders the token pieces a session plan feeds (plan §4.5), through the model's own chat
/// template so every piece equals that segment of a real render:
///
/// - **system prefix**: the tokens of `[system, user]` (with tools, no generation prompt) before
///   the last `<|im_start|>`;
/// - **first turns**: `[system] + turns` with the generation prompt, minus the system prefix the
///   ledger already holds;
/// - **continuation**: the sentinel delta. Render `[system, user(SU), assistant(SA)] + turns`
///   with the generation prompt, decode it keeping special tokens, cut at the first
///   `<|im_end|>` after `SA` and encode the rest. The cut starts with a special token, where BPE
///   always splits, so the tokens equal that segment of a real render. A tool round renders
///   `assistant(SA, tool_calls:)` and the `tool` messages instead;
/// - **assistant text**: a replacement reply, encoded as raw text (no turn end).
///
/// Chat turns become messages with `DefaultMessageGenerator`: a user turn's content is its
/// context tag and text (`LocalSessionPlan.content(of:)`); an assistant turn with tool rounds is
/// `assistant("", tool_calls:)` plus one `tool` message per call for each round, then
/// `assistant(text)` if it has text. All of that text goes through `escaper` first, so no text in
/// the conversation becomes a control token (the system prompt and tools are the app's own).
public struct TurnRenderer {
    public static let turnStartToken = "<|im_start|>"
    public static let turnEndToken = "<|im_end|>"

    public let renderer: any ChatTemplateRendering
    public let context: [String: any Sendable]
    /// The ids of `<|im_start|>` and `<|im_end|>`.
    public let turnStart: Int
    public let turnEnd: Int
    /// Escapes `renderer`'s added-token literals in the conversation's text.
    public let escaper: SpecialTokenEscaper

    /// Throws `EngineError.notChatML` when the vocabulary has no ChatML turn markers.
    public init(renderer: any ChatTemplateRendering, chatContext: [String: Bool]) throws {
        guard let start = renderer.tokenID(Self.turnStartToken), let end = renderer.tokenID(Self.turnEndToken) else {
            throw EngineError.notChatML
        }
        self.renderer = renderer
        self.context = JSONBridge.context(chatContext)
        self.turnStart = start
        self.turnEnd = end
        self.escaper = SpecialTokenEscaper(renderer: renderer)
    }

    // MARK: Messages

    /// The chat-template messages of `turns`, escaped with `escaper`.
    public func messages(_ turns: [ChatTurn]) -> [Chat.Message] {
        Self.messages(turns, escaper: escaper)
    }

    /// The chat-template messages of `turns`, every text escaped with `escaper`.
    public static func messages(_ turns: [ChatTurn], escaper: SpecialTokenEscaper) -> [Chat.Message] {
        var messages: [Chat.Message] = []
        for turn in turns {
            switch turn.role {
            case .user:
                messages.append(.user(escaper.escape(LocalSessionPlan.content(of: turn))))
            case .assistant:
                for round in turn.toolRounds where !round.calls.isEmpty {
                    messages += roundMessages(content: "", round: round, escaper: escaper)
                }
                if !turn.text.isEmpty || turn.toolRounds.allSatisfy({ $0.calls.isEmpty }) {
                    messages.append(.assistant(escaper.escape(turn.text)))
                }
            }
        }
        return messages
    }

    /// `assistant(content, tool_calls:)` and the round's `tool` result messages. The calls and
    /// results are escaped with `escaper`; `content` is the renderer's own (empty, or the sentinel).
    public static func roundMessages(content: String, round: ToolRound, escaper: SpecialTokenEscaper) -> [Chat.Message] {
        let calls = round.calls.map { escaper.escape($0) }
        return [.assistant(content, toolCalls: calls.map(JSONBridge.toolCall))]
            + calls.map { Chat.Message.tool($0.result, id: $0.id) }
    }

    /// Raw template messages (`[[String: any Sendable]]`) for `system` + `messages`.
    public static func raw(system: String, _ messages: [Chat.Message]) -> [[String: any Sendable]] {
        DefaultMessageGenerator().generate(messages: [.system(system)] + messages)
    }

    // MARK: Pieces

    /// The system prefix for `system` and `tools`; nil if the template gives none (the cache
    /// then has no `systemEnd` and every rebuild starts empty).
    public func systemPrefix(system: String, tools: [ToolDefinition]) -> [Int]? {
        let messages = Self.raw(system: system, [.user(TurnDelta.userSentinel)])
        guard let tokens = try? render(messages, tools: tools, generationPrompt: false) else { return nil }
        return TurnDelta.systemPrefix(in: tokens, turnStart: turnStart)
    }

    /// `[system] + turns` with the generation prompt.
    public func fullRender(system: String, tools: [ToolDefinition], turns: [ChatTurn]) throws -> [Int] {
        try render(Self.raw(system: system, messages(turns)), tools: tools, generationPrompt: true)
    }

    /// `fullRender` minus `ledgerPrefix`, the system prefix the ledger holds (the whole render
    /// when it is empty). Throws `renderFailed` if the render doesn't start with it.
    public func firstTurns(system: String, tools: [ToolDefinition], turns: [ChatTurn], after ledgerPrefix: [Int]) throws -> [Int] {
        let full = try fullRender(system: system, tools: tools, turns: turns)
        guard let rest = TurnDelta.dropPrefix(ledgerPrefix, from: full) else {
            throw EngineError.renderFailed("the conversation's render doesn't start with the cached system prefix")
        }
        return rest
    }

    /// The sentinel delta that follows a cached reply with `turns`; it starts with
    /// `<|im_end|>` and ends with the generation prompt.
    public func continuation(system: String, tools: [ToolDefinition], turns: [ChatTurn]) throws -> [Int] {
        let sentinelMessages: [Chat.Message] = [.user(TurnDelta.userSentinel), .assistant(TurnDelta.assistantSentinel)] + messages(turns)
        return try delta(Self.raw(system: system, sentinelMessages), tools: tools)
    }

    /// The sentinel delta that follows a reply that ended with `round`'s tool calls: the turn
    /// end, the tool results and the generation prompt.
    public func toolRoundContinuation(system: String, tools: [ToolDefinition], round: ToolRound) throws -> [Int] {
        let messages: [Chat.Message] = [.user(TurnDelta.userSentinel)]
            + Self.roundMessages(content: TurnDelta.assistantSentinel, round: round, escaper: escaper)
        return try delta(Self.raw(system: system, messages), tools: tools)
    }

    /// A replacement reply's text (escaped), without a turn end.
    public func assistantText(_ text: String) -> [Int] {
        renderer.encodeRaw(escaper.escape(text))
    }

    /// `[system] + turns` with the generation prompt, through any template (no ChatML markers
    /// needed): the render of a reply that reuses nothing.
    public static func fullRender(
        renderer: any ChatTemplateRendering, chatContext: [String: Bool], system: String, tools: [ToolDefinition],
        turns: [ChatTurn]
    ) throws -> [Int] {
        let escaped = messages(turns, escaper: SpecialTokenEscaper(renderer: renderer))
        return try render(
            renderer: renderer, context: JSONBridge.context(chatContext), raw(system: system, escaped), tools: tools,
            generationPrompt: true)
    }

    // MARK: Helpers

    func render(_ messages: [[String: any Sendable]], tools: [ToolDefinition], generationPrompt: Bool) throws -> [Int] {
        try Self.render(renderer: renderer, context: context, messages, tools: tools, generationPrompt: generationPrompt)
    }

    static func render(
        renderer: any ChatTemplateRendering, context: [String: any Sendable], _ messages: [[String: any Sendable]],
        tools: [ToolDefinition], generationPrompt: Bool
    ) throws -> [Int] {
        do {
            return try renderer.renderTokens(
                messages: messages, tools: JSONBridge.templateToolsOrNil(tools), context: context,
                addGenerationPrompt: generationPrompt)
        } catch let error as EngineError {
            throw error
        } catch {
            throw EngineError.renderFailed(String(describing: error))
        }
    }

    private func delta(_ messages: [[String: any Sendable]], tools: [ToolDefinition]) throws -> [Int] {
        let tokens = try render(messages, tools: tools, generationPrompt: true)
        let text = renderer.decodeRaw(tokens)
        guard let cut = TurnDelta.continuation(in: text, after: TurnDelta.assistantSentinel, turnEnd: Self.turnEndToken) else {
            throw EngineError.renderFailed("the sentinel render has no unique turn end after the reply")
        }
        let delta = renderer.encodeRaw(cut)
        guard delta.first == turnEnd else {
            throw EngineError.renderFailed("the continuation doesn't start with \(Self.turnEndToken)")
        }
        return delta
    }
}
