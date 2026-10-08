import AssistantKit
import Foundation
import MLXLMCommon

/// Converts between the app's JSON (`AssistantKit.JSONValue`), mlx-swift-lm's
/// (`MLXLMCommon.JSONValue`, used by `ToolCall`) and the plain Swift values chat templates read.
public enum JSONBridge {
    // MARK: AssistantKit → MLXLMCommon

    /// The same value as mlx-swift-lm JSON. Object members that are `null` are dropped and `null`
    /// array elements become the string "null": chat templates get tool-call arguments through
    /// `ToolCall.Function`, whose null becomes `NSNull`, which the Jinja runtime can't convert.
    public static func mlx(_ value: AssistantKit.JSONValue) -> MLXLMCommon.JSONValue {
        switch value {
        case .null: return .string("null")
        case .bool(let value): return .bool(value)
        case .int(let value): return .int(value)
        case .double(let value): return .double(value)
        case .string(let value): return .string(value)
        case .array(let values): return .array(values.map(mlx))
        case .object(let members):
            var result: [String: MLXLMCommon.JSONValue] = [:]
            for (key, member) in members where !member.isNull {
                result[key] = mlx(member)
            }
            return .object(result)
        }
    }

    /// The members of a tool call's input object, as `ToolCall.Function` arguments. A
    /// non-object input gives no arguments.
    public static func arguments(_ input: AssistantKit.JSONValue) -> [String: MLXLMCommon.JSONValue] {
        guard case .object = input, case .object(let members) = mlx(input) else { return [:] }
        return members
    }

    // MARK: MLXLMCommon → AssistantKit

    public static func assistant(_ value: MLXLMCommon.JSONValue) -> AssistantKit.JSONValue {
        switch value {
        case .null: return .null
        case .bool(let value): return .bool(value)
        case .int(let value): return .int(value)
        case .double(let value): return .double(value)
        case .string(let value): return .string(value)
        case .array(let values): return .array(values.map(assistant))
        case .object(let members): return .object(members.mapValues(assistant))
        }
    }

    /// A parsed call's arguments as the app's tool input object.
    public static func input(_ arguments: [String: MLXLMCommon.JSONValue]) -> AssistantKit.JSONValue {
        .object(arguments.mapValues(assistant))
    }

    // MARK: Template values

    /// A plain Swift value for a chat template: strings, numbers, booleans, `[any Sendable]` and
    /// `[String: any Sendable]`. `null` becomes an empty `String?`, which the Jinja runtime reads
    /// as `none` (it can't read `NSNull`).
    public static func sendable(_ value: AssistantKit.JSONValue) -> any Sendable {
        switch value {
        case .null: return String?.none
        case .bool(let value): return value
        case .int(let value): return value
        case .double(let value): return value
        case .string(let value): return value
        case .array(let values): return values.map(sendable)
        case .object(let members): return members.mapValues(sendable) as [String: any Sendable]
        }
    }

    /// Tools in the OpenAI function format chat templates expect:
    /// `[{"type": "function", "function": {"name", "description", "parameters"}}]`.
    public static func templateTools(_ definitions: [ToolDefinition]) -> [[String: any Sendable]] {
        definitions.map { definition in
            [
                "type": "function",
                "function": [
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": sendable(definition.inputSchema),
                ] as [String: any Sendable],
            ]
        }
    }

    /// `templateTools`, or nil when there are none (so templates leave out their tools block).
    public static func templateToolsOrNil(_ definitions: [ToolDefinition]) -> [[String: any Sendable]]? {
        definitions.isEmpty ? nil : templateTools(definitions)
    }

    /// The template context for `chatContext`.
    public static func context(_ chatContext: [String: Bool]) -> [String: any Sendable] {
        chatContext.mapValues { $0 as any Sendable }
    }

    // MARK: Tool calls

    /// A tool call the model made, for the app: the parser's id (or a fresh `call_…` id) and its
    /// arguments as the input object.
    public static func pendingCall(_ call: MLXLMCommon.ToolCall) -> PendingToolCall {
        let id = call.id.flatMap { $0.isEmpty ? nil : $0 } ?? newCallID()
        return PendingToolCall(id: id, name: call.function.name, input: input(call.function.arguments), rawInput: nil)
    }

    /// A fresh tool-call id: `call_` and 32 lowercase hex digits.
    public static func newCallID() -> String {
        "call_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    }

    /// A recorded call, for a chat template's `tool_calls`.
    public static func toolCall(_ record: ToolCallRecord) -> MLXLMCommon.ToolCall {
        MLXLMCommon.ToolCall(function: .init(name: record.name, arguments: arguments(record.input)), id: record.id)
    }
}
