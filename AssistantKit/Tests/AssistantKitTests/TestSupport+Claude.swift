import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
@testable import AssistantKit

/// Event payloads of a streamed Messages API response, for `MockTransport.sse`.
enum ClaudeSSE {
    static func messageStart(model: String = "claude-opus-5-5") -> String {
        payload(["type": "message_start", "message": ["id": "msg_test", "model": .string(model), "content": .array([])]])
    }

    /// A text block streamed as an empty start, one delta and a stop.
    static func text(index: Int, _ text: String) -> [String] {
        [
            payload(["type": "content_block_start", "index": .int(index), "content_block": ["type": "text", "text": ""]]),
            payload(["type": "content_block_delta", "index": .int(index), "delta": ["type": "text_delta", "text": .string(text)]]),
            payload(["type": "content_block_stop", "index": .int(index)]),
        ]
    }

    /// An empty (omitted) thinking block that carries `signature`, as Opus 5.5 sends between tool calls.
    static func thinking(index: Int, signature: String) -> [String] {
        [
            payload(["type": "content_block_start", "index": .int(index), "content_block": ["type": "thinking", "thinking": "", "signature": ""]]),
            payload(["type": "content_block_delta", "index": .int(index), "delta": ["type": "signature_delta", "signature": .string(signature)]]),
            payload(["type": "content_block_stop", "index": .int(index)]),
        ]
    }

    /// A `web_search` server tool call at `index` and its (empty) result at `index + 1`.
    static func webSearch(index: Int, id: String, query: String) -> [String] {
        let input = String(decoding: (try? JSONValue.object(["query": .string(query)]).serialized()) ?? Data(), as: UTF8.self)
        let call: JSONValue = ["type": "server_tool_use", "id": .string(id), "name": "web_search", "input": .object([:])]
        let result: JSONValue = ["type": "web_search_tool_result", "tool_use_id": .string(id), "content": .array([])]
        return [
            payload(["type": "content_block_start", "index": .int(index), "content_block": call]),
            payload(["type": "content_block_delta", "index": .int(index), "delta": ["type": "input_json_delta", "partial_json": .string(input)]]),
            payload(["type": "content_block_stop", "index": .int(index)]),
            payload(["type": "content_block_start", "index": .int(index + 1), "content_block": result]),
            payload(["type": "content_block_stop", "index": .int(index + 1)]),
        ]
    }

    static func stop(_ reason: String, details: JSONValue? = nil) -> String {
        var delta: [String: JSONValue] = ["stop_reason": .string(reason)]
        if let details { delta["stop_details"] = details }
        return payload(["type": "message_delta", "delta": .object(delta)])
    }

    /// One scripted response made of `parts` (each a list of payloads), in order.
    static func response(_ parts: [String]...) -> MockTransport.Response {
        .lines(MockTransport.sse(parts.flatMap { $0 }))
    }

    static func payload(_ value: JSONValue) -> String {
        String(decoding: (try? value.serialized()) ?? Data(), as: UTF8.self)
    }
}

extension URLRequest {
    /// The parsed JSON body, or `.null`.
    var jsonBody: JSONValue {
        ScriptedTransport.body(of: self)
    }

    /// The body's `messages`.
    var sentMessages: [JSONValue] {
        jsonBody["messages"]?.arrayValue ?? []
    }
}

extension JSONValue {
    /// The `type` of each block in this message's `content`.
    var blockTypes: [String] {
        (self["content"]?.arrayValue ?? []).map { $0["type"]?.stringValue ?? "?" }
    }

    /// Compact JSON with sorted keys, as text.
    var serializedText: String {
        String(decoding: (try? serialized()) ?? Data(), as: UTF8.self)
    }
}

/// A Claude configuration for tests: a key, the given model, and web search off unless asked for.
func claudeConfiguration(
    model: String = "claude-opus-5-5",
    effort: String? = "low",
    webSearch: Bool = false,
    baseURL: String = "https://api.anthropic.com"
) -> ClaudeConfiguration {
    ClaudeConfiguration(apiKey: "sk-test", model: model, effort: effort, webSearchEnabled: webSearch, baseURL: URL(string: baseURL)!)
}

/// A strict object schema with the given string properties, all required.
func stringSchema(_ names: [String]) -> JSONValue {
    var properties: [String: JSONValue] = [:]
    for name in names { properties[name] = ["type": "string"] }
    return [
        "type": "object",
        "properties": .object(properties),
        "required": .array(names.map { .string($0) }),
        "additionalProperties": false,
    ]
}
