import Foundation

/// Turns Messages API stream events into `ReplyEvent`s and rebuilds the assistant's content blocks,
/// which go back verbatim when a server-tool turn pauses (`stop_reason: "pause_turn"`).
public struct ClaudeStreamDecoder: Sendable {
    /// Model that produced the message; differs from the requested one after a refusal fallback.
    public private(set) var model: String?
    public private(set) var stopReason: String?
    public private(set) var stopDetails: JSONValue?

    private var blocks: [Int: [String: JSONValue]] = [:]
    private var order: [Int] = []
    private var texts: [Int: String] = [:]
    private var thinking: [Int: String] = [:]
    private var partialInputs: [Int: String] = [:]
    /// Whether the last activity event is still on screen; the next text clears it.
    var showingActivity = false

    public init() {}

    /// Handles one event (a decoded `data:` payload). Throws for a mid-stream `error` event.
    public mutating func handle(_ event: JSONValue) throws -> [ReplyEvent] {
        switch event["type"]?.stringValue {
        case "message_start":
            model = event["message"]?["model"]?.stringValue
            return []
        case "content_block_start":
            guard let index = event["index"]?.intValue,
                  let block = event["content_block"]?.objectValue else { return [] }
            return start(block, at: index)
        case "content_block_delta":
            guard let index = event["index"]?.intValue, let delta = event["delta"] else { return [] }
            return apply(delta, at: index)
        case "content_block_stop":
            guard let index = event["index"]?.intValue else { return [] }
            return stop(at: index)
        case "message_delta":
            if let reason = event["delta"]?["stop_reason"]?.stringValue { stopReason = reason }
            if let details = event["delta"]?["stop_details"], !details.isNull { stopDetails = details }
            return []
        case "error":
            let type = event["error"]?["type"]?.stringValue ?? "error"
            let message = event["error"]?["message"]?.stringValue ?? "Unknown streaming error"
            throw AssistantError.stream(type: type, message: message)
        default:
            // message_stop, ping, and event types added later.
            return []
        }
    }

    /// The assistant content so far, shaped for sending back on the next request.
    public func continuationBlocks() -> [JSONValue] {
        let ordered = order.compactMap { materializedBlock(at: $0) }
        let lastFallback = ordered.lastIndex { $0["type"]?.stringValue == "fallback" }
        let answeredToolUseIDs = Set(ordered.compactMap { block -> String? in
            guard block["type"]?.stringValue?.hasSuffix("_tool_result") == true else { return nil }
            return block["tool_use_id"]?.stringValue
        })

        var result: [JSONValue] = []
        for (position, block) in ordered.enumerated() {
            let type = block["type"]?.stringValue ?? ""
            if type == "text", (block["text"]?.stringValue ?? "").isEmpty { continue }
            // Before the last fallback boundary, the output came from a model that declined mid-reply:
            // only its text and completed server-tool pairs carry over.
            if let boundary = lastFallback, position < boundary,
               !Self.keepsBeforeFallback(block, type: type, answered: answeredToolUseIDs) {
                continue
            }
            result.append(.object(block))
        }
        return result
    }

    static func activityDescription(tool: String?, input: JSONValue?) -> String {
        switch tool {
        case "web_search":
            if let query = input?["query"]?.stringValue, !query.isEmpty {
                return "Searching the web for “\(query)”"
            }
            return "Searching the web"
        case "web_fetch":
            return "Reading a web page"
        default:
            return "Working on it"
        }
    }

    // MARK: - Private

    private mutating func start(_ block: [String: JSONValue], at index: Int) -> [ReplyEvent] {
        if blocks[index] == nil { order.append(index) }
        blocks[index] = block
        switch block["type"]?.stringValue {
        case "text":
            let text = block["text"]?.stringValue ?? ""
            texts[index] = text
            return textEvents(text)
        case "thinking":
            thinking[index] = block["thinking"]?.stringValue ?? ""
            return []
        case "server_tool_use":
            showingActivity = true
            return [.activity(Self.activityDescription(tool: block["name"]?.stringValue, input: nil))]
        default:
            return []
        }
    }

    private mutating func apply(_ delta: JSONValue, at index: Int) -> [ReplyEvent] {
        guard blocks[index] != nil else { return [] }
        switch delta["type"]?.stringValue {
        case "text_delta":
            let text = delta["text"]?.stringValue ?? ""
            texts[index, default: ""] += text
            return textEvents(text)
        case "thinking_delta":
            thinking[index, default: ""] += delta["thinking"]?.stringValue ?? ""
        case "signature_delta":
            if let signature = delta["signature"] {
                blocks[index]?["signature"] = signature
            }
        case "input_json_delta":
            partialInputs[index, default: ""] += delta["partial_json"]?.stringValue ?? ""
        case "citations_delta":
            if let citation = delta["citation"] {
                let existing = blocks[index]?["citations"]?.arrayValue ?? []
                blocks[index]?["citations"] = .array(existing + [citation])
            }
        default:
            break
        }
        return []
    }

    private mutating func stop(at index: Int) -> [ReplyEvent] {
        guard let partial = partialInputs.removeValue(forKey: index) else { return [] }
        let trimmed = partial.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty, let input = try? JSONValue.parse(trimmed) {
            blocks[index]?["input"] = input
        }
        guard blocks[index]?["type"]?.stringValue == "server_tool_use" else { return [] }
        showingActivity = true
        let description = Self.activityDescription(
            tool: blocks[index]?["name"]?.stringValue,
            input: blocks[index]?["input"]
        )
        return [.activity(description)]
    }

    private mutating func textEvents(_ text: String) -> [ReplyEvent] {
        guard !text.isEmpty else { return [] }
        if showingActivity {
            showingActivity = false
            return [.activity(nil), .text(text)]
        }
        return [.text(text)]
    }

    private func materializedBlock(at index: Int) -> [String: JSONValue]? {
        guard var block = blocks[index] else { return nil }
        if let text = texts[index] { block["text"] = .string(text) }
        if let thought = thinking[index] { block["thinking"] = .string(thought) }
        if block["type"]?.stringValue == "text", (block["citations"]?.arrayValue ?? []).isEmpty {
            block.removeValue(forKey: "citations")
        }
        return block
    }

    private static func keepsBeforeFallback(_ block: [String: JSONValue], type: String, answered: Set<String>) -> Bool {
        switch type {
        case "text", "fallback":
            return true
        case "server_tool_use":
            guard let id = block["id"]?.stringValue else { return false }
            return answered.contains(id)
        default:
            return type.hasSuffix("_tool_result")
        }
    }
}
