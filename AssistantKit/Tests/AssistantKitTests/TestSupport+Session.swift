import Foundation
@testable import AssistantKit

/// The chat template and tokenizer of LocalEngine's `FakeChatMLTokenizer` (WP02), without MLX, so
/// the session logic can be checked against real renders on Linux.
///
/// - Ids: 0 unknown, 10 `\n`, 32...126 printable ASCII; added tokens 1 `<|im_start|>`,
///   2 `<|im_end|>`, 3 `<|endoftext|>`, 4 `<think>`, 5 `</think>`, 6 `<tool_call>`,
///   7 `</tool_call>`, 8 `<tool_response>`, 9 `</tool_response>`.
/// - Template: a system block holding the tools JSON; history assistant turns without think
///   blocks; tool calls as `<tool_call>\n{json}\n</tool_call>`; consecutive `tool` messages grouped
///   into one user turn of `<tool_response>` blocks; the generation prompt
///   `<|im_start|>assistant\n`, plus `<think>\n\n</think>\n\n` when thinking is off.
struct FakeChatMLTemplate {
    static let unknown = 0
    static let imStart = 1
    static let imEnd = 2
    static let endOfText = 3
    static let newline = 10

    static let turnStart = "<|im_start|>"
    static let turnEnd = "<|im_end|>"

    struct ToolCall: Equatable {
        var name: String
        var arguments: JSONValue
    }

    struct Message: Equatable {
        var role: String
        var content: String
        var toolCalls: [ToolCall] = []

        static func system(_ content: String) -> Message { Message(role: "system", content: content) }
        static func user(_ content: String) -> Message { Message(role: "user", content: content) }
        static func assistant(_ content: String, toolCalls: [ToolCall] = []) -> Message { Message(role: "assistant", content: content, toolCalls: toolCalls) }
        static func tool(_ content: String) -> Message { Message(role: "tool", content: content) }
    }

    private static let addedTokens: [(id: Int, text: String)] = [
        (0, "<unk>"), (1, "<|im_start|>"), (2, "<|im_end|>"), (3, "<|endoftext|>"), (4, "<think>"),
        (5, "</think>"), (6, "<tool_call>"), (7, "</tool_call>"), (8, "<tool_response>"), (9, "</tool_response>"),
    ]

    /// Tools in the template's form (`{"type": "function", "function": {...}}`), as JSON values.
    var tools: [JSONValue] = []
    var enableThinking = false

    // MARK: Tokenizer

    func encode(_ text: String) -> [Int] {
        let scalars = Array(text.unicodeScalars)
        let added = Self.addedTokens
            .map { (id: $0.id, scalars: Array($0.text.unicodeScalars)) }
            .sorted { $0.scalars.count > $1.scalars.count }
        var ids: [Int] = []
        var index = 0
        scan: while index < scalars.count {
            if scalars[index] == "<" {
                for token in added where index + token.scalars.count <= scalars.count && Array(scalars[index ..< index + token.scalars.count]) == token.scalars {
                    ids.append(token.id)
                    index += token.scalars.count
                    continue scan
                }
            }
            let value = scalars[index].value
            ids.append(value == 10 ? Self.newline : (32 ... 126).contains(value) ? Int(value) : Self.unknown)
            index += 1
        }
        return ids
    }

    /// Decodes with special tokens kept.
    func decode(_ tokens: [Int]) -> String {
        var text = ""
        for id in tokens {
            if let token = Self.addedTokens.first(where: { $0.id == id }) {
                text += token.text
            } else if id == Self.newline {
                text += "\n"
            } else if (32 ... 126).contains(id), let scalar = Unicode.Scalar(UInt32(id)) {
                text.unicodeScalars.append(scalar)
            }
        }
        return text
    }

    // MARK: Template

    func render(_ messages: [Message], addGenerationPrompt: Bool) -> String {
        var out = ""
        var first = 0
        let startsWithSystem = messages.first?.role == "system"

        if !tools.isEmpty {
            out += "<|im_start|>system\n"
            if startsWithSystem {
                out += messages[0].content + "\n\n"
                first = 1
            }
            out += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            out += "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            for tool in tools {
                out += "\n" + Self.json(tool)
            }
            out += "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
            out += "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n"
        } else if startsWithSystem {
            out += "<|im_start|>system\n" + messages[0].content + "<|im_end|>\n"
            first = 1
        }

        for index in first ..< messages.count {
            let message = messages[index]
            switch message.role {
            case "assistant":
                var text = message.content
                if let end = text.range(of: "</think>", options: .backwards) {
                    text = String(text[end.upperBound...])
                    while text.hasPrefix("\n") { text.removeFirst() }
                }
                out += "<|im_start|>assistant\n" + text
                for (number, call) in message.toolCalls.enumerated() {
                    if number > 0 || !text.isEmpty { out += "\n" }
                    out += "<tool_call>\n{\"name\": \(Self.json(.string(call.name))), \"arguments\": \(Self.json(call.arguments))}\n</tool_call>"
                }
                out += "<|im_end|>\n"
            case "tool":
                let previousIsTool = index > 0 && messages[index - 1].role == "tool"
                let nextIsTool = index + 1 < messages.count && messages[index + 1].role == "tool"
                if !previousIsTool { out += "<|im_start|>user" }
                out += "\n<tool_response>\n\(message.content)\n</tool_response>"
                if !nextIsTool { out += "<|im_end|>\n" }
            default:
                out += "<|im_start|>\(message.role)\n\(message.content)<|im_end|>\n"
            }
        }

        if addGenerationPrompt {
            out += "<|im_start|>assistant\n"
            if !enableThinking { out += "<think>\n\n</think>\n\n" }
        }
        return out
    }

    func renderTokens(_ messages: [Message], addGenerationPrompt: Bool) -> [Int] {
        encode(render(messages, addGenerationPrompt: addGenerationPrompt))
    }

    /// The messages LocalEngine's `TurnRenderer` (WP20) builds: user content with its context tag;
    /// an assistant turn with rounds as `assistant("", tool_calls)` plus one `tool` message per call
    /// for each round, then `assistant(text)` when the text isn't empty.
    static func messages(system: String?, turns: [ChatTurn]) -> [Message] {
        var messages: [Message] = system.map { [.system($0)] } ?? []
        for turn in turns {
            switch turn.role {
            case .user:
                messages.append(.user(LocalSessionPlan.content(of: turn)))
            case .assistant:
                for round in turn.toolRounds {
                    messages.append(.assistant("", toolCalls: round.calls.map { ToolCall(name: $0.name, arguments: $0.input) }))
                    messages += round.calls.map { .tool($0.result) }
                }
                if turn.toolRounds.isEmpty || !turn.text.isEmpty {
                    messages.append(.assistant(turn.text))
                }
            }
        }
        return messages
    }

    /// Compact JSON with sorted keys and Python-style `", "` / `": "` separators, as the fake
    /// template prints tools and arguments.
    static func json(_ value: JSONValue) -> String {
        switch value {
        case .null: return "null"
        case .bool(let flag): return flag ? "true" : "false"
        case .int(let number): return String(number)
        case .double(let number): return number.isFinite ? String(number) : "null"
        case .string(let text): return quoted(text)
        case .array(let items): return "[" + items.map(json).joined(separator: ", ") + "]"
        case .object(let members):
            return "{" + members.keys.sorted().map { quoted($0) + ": " + json(members[$0] ?? .null) }.joined(separator: ", ") + "}"
        }
    }

    private static func quoted(_ text: String) -> String {
        var out = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "\""
    }
}

// MARK: - Session fixtures

enum SessionFixtures {
    static let system = "You are helpful."
    static let key = PrefixKey.make(modelID: "test/model", revision: "rev1", system: system, toolsJSON: "[]", context: "enable_thinking=false", formatVersion: 1)

    static func user(_ text: String, context: String? = "<context>input: spoken</context>") -> ChatTurn {
        ChatTurn(role: .user, text: text, context: context)
    }

    static func assistant(_ text: String, rounds: [ToolRound] = []) -> ChatTurn {
        ChatTurn(role: .assistant, text: text, toolRounds: rounds)
    }

    static func round(_ name: String = "get_current_time", id: String = "call_1", input: JSONValue = [:], result: String = #"{"time":"16:05"}"#) -> ToolRound {
        ToolRound(calls: [ToolCallRecord(id: id, name: name, input: input, result: result)])
    }

    /// A snapshot as the engine leaves it after replying to `turns` (which ends with a user turn,
    /// optionally followed by the reply): the newest user turn is marked at `userStart` and
    /// `replyStart`.
    static func snapshot(
        _ turns: [ChatTurn],
        firstTurnIndex: Int = 0,
        systemEnd: Int = 100,
        userStart: Int = 180,
        replyStart: Int = 200,
        tokenCount: Int = 240,
        prefixKey: String = key
    ) -> SessionSnapshot {
        var cached = turns.map { CachedTurn(turn: $0) }
        if let index = cached.lastIndex(where: { $0.turn.role == .user }) {
            cached[index].start = userStart
            cached[index].replyStart = replyStart
        }
        return SessionSnapshot(prefixKey: prefixKey, systemEnd: systemEnd, firstTurnIndex: firstTurnIndex, turns: cached, tokenCount: tokenCount)
    }

    /// `count` alternating turns starting with a user turn: q0, a0, q1, a1, …
    static func conversation(_ count: Int, replyLength: Int = 10) -> [ChatTurn] {
        (0 ..< count).map { index in
            index.isMultiple(of: 2)
                ? user("q\(index / 2)")
                : assistant("a\(index / 2) " + String(repeating: "x", count: replyLength))
        }
    }
}
