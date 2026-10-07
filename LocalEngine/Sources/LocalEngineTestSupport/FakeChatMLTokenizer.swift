import Foundation
import LocalEngine
import MLXLMCommon

/// A deterministic ChatML tokenizer with a hard-coded Qwen3-like chat template, for engine tests
/// that run without tokenizer files. Its 128 ids fit the tiny models' vocabulary:
/// - 0 `<unk>` (any character outside the table), 10 `\n`, 32...126 printable ASCII;
/// - added tokens 1 `<|im_start|>`, 2 `<|im_end|>`, 3 `<|endoftext|>`, 4 `<think>`, 5 `</think>`,
///   6 `<tool_call>`, 7 `</tool_call>`, 8 `<tool_response>`, 9 `</tool_response>`.
///
/// Ids 0...3 are special (dropped by `skipSpecialTokens`); 4...9 are kept, as in Qwen. Unused ids
/// (11...31, 127) decode to nothing.
///
/// The template: a system block that holds the tools JSON; history assistant turns without think
/// blocks; tool calls as `<tool_call>\n{json}\n</tool_call>`; consecutive `tool` messages grouped
/// into one user turn of `<tool_response>` blocks; and the generation prompt
/// `<|im_start|>assistant\n`, followed by `<think>\n\n</think>\n\n` when `enable_thinking` is false.
public struct FakeChatMLTokenizer: MLXLMCommon.Tokenizer, ChatTemplateRendering {
    public static let vocabularySize = 128

    public static let unknown = 0
    public static let imStart = 1
    public static let imEnd = 2
    public static let endOfText = 3
    public static let thinkStart = 4
    public static let thinkEnd = 5
    public static let toolCallStart = 6
    public static let toolCallEnd = 7
    public static let toolResponseStart = 8
    public static let toolResponseEnd = 9
    public static let newline = 10

    /// The template context the engine uses: answer without a reasoning block.
    public static let chatContext: [String: any Sendable] = ["enable_thinking": false]

    public enum TemplateError: Error, Equatable {
        case unknownRole(String)
    }

    private static let addedTokens: [Int: String] = [
        unknown: "<unk>",
        imStart: "<|im_start|>",
        imEnd: "<|im_end|>",
        endOfText: "<|endoftext|>",
        thinkStart: "<think>",
        thinkEnd: "</think>",
        toolCallStart: "<tool_call>",
        toolCallEnd: "</tool_call>",
        toolResponseStart: "<tool_response>",
        toolResponseEnd: "</tool_response>",
    ]

    /// Added tokens as scalars, longest first, for greedy matching while encoding.
    private static let addedTokenScalars: [(id: Int, scalars: [Unicode.Scalar])] = addedTokens
        .map { (id: $0.key, scalars: Array($0.value.unicodeScalars)) }
        .sorted { $0.scalars.count != $1.scalars.count ? $0.scalars.count > $1.scalars.count : $0.id < $1.id }

    /// When false, rendering throws `MLXLMCommon.TokenizerError.missingChatTemplate`, like a
    /// tokenizer shipped without a template.
    public var templateAvailable: Bool

    public init(templateAvailable: Bool = true) {
        self.templateAvailable = templateAvailable
    }

    // MARK: MLXLMCommon.Tokenizer

    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        // Qwen adds no BOS, so addSpecialTokens changes nothing.
        let scalars = Array(text.unicodeScalars)
        var ids: [Int] = []
        ids.reserveCapacity(scalars.count)
        var index = 0
        scan: while index < scalars.count {
            if scalars[index] == "<" {
                for token in Self.addedTokenScalars where Self.matches(scalars, at: index, token.scalars) {
                    ids.append(token.id)
                    index += token.scalars.count
                    continue scan
                }
            }
            ids.append(Self.id(of: scalars[index]))
            index += 1
        }
        return ids
    }

    public func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        var text = ""
        for id in tokenIds {
            if let token = Self.addedTokens[id] {
                if skipSpecialTokens && id <= Self.endOfText { continue }
                text += token
            } else if let character = Self.character(of: id) {
                text.unicodeScalars.append(character)
            }
        }
        return text
    }

    public func convertTokenToId(_ token: String) -> Int? {
        if let added = Self.addedTokens.first(where: { $0.value == token }) {
            return added.key
        }
        let scalars = Array(token.unicodeScalars)
        guard scalars.count == 1 else { return nil }
        let id = Self.id(of: scalars[0])
        return id == Self.unknown ? nil : id
    }

    public func convertIdToToken(_ id: Int) -> String? {
        if let token = Self.addedTokens[id] { return token }
        return Self.character(of: id).map { String($0) }
    }

    public var bosToken: String? { nil }
    public var eosToken: String? { "<|im_end|>" }
    public var unknownToken: String? { "<unk>" }

    public func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] {
        try renderTokens(messages: messages, tools: tools, context: additionalContext, addGenerationPrompt: true)
    }

    // MARK: ChatTemplateRendering

    public func renderTokens(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                             context: [String: any Sendable]?, addGenerationPrompt: Bool) throws -> [Int] {
        encode(text: try renderText(messages: messages, tools: tools, context: context, addGenerationPrompt: addGenerationPrompt), addSpecialTokens: false)
    }

    public func encodeRaw(_ text: String) -> [Int] {
        encode(text: text, addSpecialTokens: false)
    }

    public func decodeRaw(_ tokens: [Int]) -> String {
        decode(tokenIds: tokens, skipSpecialTokens: false)
    }

    public func tokenID(_ token: String) -> Int? {
        convertTokenToId(token)
    }

    // MARK: Template

    /// The rendered template text, before tokenization.
    public func renderText(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                           context: [String: any Sendable]?, addGenerationPrompt: Bool) throws -> String {
        guard templateAvailable else {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
        let messages: [[String: Any]] = messages
        var out = ""
        var first = 0
        let startsWithSystem = messages.first.map { Self.role(of: $0) == "system" } ?? false

        if let tools, !tools.isEmpty {
            out += "<|im_start|>system\n"
            if startsWithSystem {
                out += Self.content(of: messages[0]) + "\n\n"
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
            out += "<|im_start|>system\n" + Self.content(of: messages[0]) + "<|im_end|>\n"
            first = 1
        }

        for index in first ..< messages.count {
            let message = messages[index]
            let role = Self.role(of: message)
            let content = Self.content(of: message)
            switch role {
            case "user", "system":
                out += "<|im_start|>\(role)\n\(content)<|im_end|>\n"

            case "assistant":
                // History turns are shown without their reasoning.
                var text = content
                if let end = text.range(of: "</think>", options: .backwards) {
                    text = String(text[end.upperBound...])
                    while text.hasPrefix("\n") { text.removeFirst() }
                }
                out += "<|im_start|>assistant\n" + text
                for (number, call) in Self.dictionaries(message["tool_calls"]).enumerated() {
                    if number > 0 || !text.isEmpty { out += "\n" }
                    let function = (call["function"] as? [String: Any]) ?? call
                    let name = function["name"] as? String ?? ""
                    let arguments = (function["arguments"] as? String) ?? Self.json(function["arguments"] ?? [String: Any]())
                    out += "<tool_call>\n{\"name\": \(Self.quoted(name)), \"arguments\": \(arguments)}\n</tool_call>"
                }
                out += "<|im_end|>\n"

            case "tool":
                let previousIsTool = index > 0 && Self.role(of: messages[index - 1]) == "tool"
                let nextIsTool = index + 1 < messages.count && Self.role(of: messages[index + 1]) == "tool"
                if !previousIsTool { out += "<|im_start|>user" }
                out += "\n<tool_response>\n\(content)\n</tool_response>"
                if !nextIsTool { out += "<|im_end|>\n" }

            default:
                throw TemplateError.unknownRole(role)
            }
        }

        if addGenerationPrompt {
            out += "<|im_start|>assistant\n"
            if (context?["enable_thinking"] as? Bool) == false {
                out += "<think>\n\n</think>\n\n"
            }
        }
        return out
    }

    /// Compact JSON with sorted keys and Python-style `", "` / `": "` separators.
    public static func json(_ value: Any) -> String {
        switch value {
        case is NSNull:
            return "null"
        case let value as String:
            return quoted(value)
        case let value as Bool:
            return value ? "true" : "false"
        case let value as Int:
            return String(value)
        case let value as Double:
            return value.isFinite ? String(value) : "null"
        case let value as Float:
            return value.isFinite ? String(value) : "null"
        case let value as MLXLMCommon.JSONValue:
            return json(value.anyValue)
        case let value as [String: Any]:
            let members = value.keys.sorted().map { key in quoted(key) + ": " + json(value[key] ?? NSNull()) }
            return "{" + members.joined(separator: ", ") + "}"
        case let value as [Any]:
            return "[" + value.map { json($0) }.joined(separator: ", ") + "]"
        default:
            return quoted(String(describing: value))
        }
    }

    // MARK: Helpers

    private static func matches(_ scalars: [Unicode.Scalar], at index: Int, _ token: [Unicode.Scalar]) -> Bool {
        guard index + token.count <= scalars.count else { return false }
        for offset in 0 ..< token.count where scalars[index + offset] != token[offset] {
            return false
        }
        return true
    }

    private static func id(of scalar: Unicode.Scalar) -> Int {
        if scalar == "\n" { return newline }
        if (32 ... 126).contains(scalar.value) { return Int(scalar.value) }
        return unknown
    }

    private static func character(of id: Int) -> Unicode.Scalar? {
        if id == newline { return "\n" }
        guard (32 ... 126).contains(id) else { return nil }
        return Unicode.Scalar(UInt8(id))
    }

    private static func role(of message: [String: Any]) -> String {
        message["role"] as? String ?? ""
    }

    private static func content(of message: [String: Any]) -> String {
        message["content"] as? String ?? ""
    }

    private static func dictionaries(_ value: Any?) -> [[String: Any]] {
        guard let value else { return [] }
        return value as? [[String: Any]] ?? []
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
