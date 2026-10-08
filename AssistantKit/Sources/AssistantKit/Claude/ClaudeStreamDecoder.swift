import Foundation

/// Turns Messages API stream events into `AssistantEvent`s and rebuilds the assistant's content
/// blocks, which go back verbatim when a server-tool turn pauses (`stop_reason: "pause_turn"`) or
/// when client tool calls (`stop_reason: "tool_use"`) are answered.
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
    /// Client `tool_use` input that didn't parse to a JSON object, as the model wrote it, by block index.
    private var invalidInputs: [Int: String] = [:]
    /// How each client tool shows while it runs, by name. Unlisted tools use `ToolPresentation.generic`.
    private let clientTools: [String: ToolPresentation]
    /// Whether the last activity event is still on screen; the next text clears it.
    var showingActivity = false
    /// Whether `.progress(.responseStarted)` has been emitted. A provider carries it across the
    /// requests of one reply, so the mark appears once.
    var responseStartReported = false

    public init(clientTools: [String: ToolPresentation] = [:]) {
        self.clientTools = clientTools
    }

    /// Handles one event (a decoded `data:` payload). Throws for a mid-stream `error` event.
    public mutating func handle(_ event: JSONValue) throws -> [AssistantEvent] {
        switch event["type"]?.stringValue {
        case "message_start":
            model = event["message"]?["model"]?.stringValue
            guard !responseStartReported else { return [] }
            responseStartReported = true
            return [.progress(.responseStarted)]
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

    /// The client tool calls in `continuationBlocks()`, in order, so a call made before a fallback
    /// boundary is dropped with the rest of the declined output. A call whose input didn't parse to
    /// a JSON object, or never finished streaming, has no `input` and keeps its text in `rawInput`.
    public func clientToolCalls() -> [PendingToolCall] {
        var rawInputs: [String: String] = [:]
        for (index, raw) in invalidInputs.merging(partialInputs, uniquingKeysWith: { invalid, _ in invalid }) {
            guard blocks[index]?["type"]?.stringValue == "tool_use",
                  let id = blocks[index]?["id"]?.stringValue else { continue }
            rawInputs[id] = raw
        }
        return continuationBlocks().compactMap { block -> PendingToolCall? in
            guard block["type"]?.stringValue == "tool_use",
                  let id = block["id"]?.stringValue,
                  let name = block["name"]?.stringValue else { return nil }
            if let raw = rawInputs[id] {
                return PendingToolCall(id: id, name: name, input: nil, rawInput: raw)
            }
            let input = JSONValue.object(block["input"]?.objectValue ?? [:])
            return PendingToolCall(id: id, name: name, input: input)
        }
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

    private mutating func start(_ block: [String: JSONValue], at index: Int) -> [AssistantEvent] {
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
            let activity = Self.activityDescription(tool: block["name"]?.stringValue, input: nil)
            return [.cue(.lookingUp), .reply(.activity(activity))]
        case "tool_use":
            let name = block["name"]?.stringValue
            let presentation = name.flatMap { clientTools[$0] } ?? .generic
            showingActivity = true
            var events: [AssistantEvent] = [.progress(.toolCallStarted(name: name))]
            if let cue = presentation.cue {
                events.append(.cue(cue))
            }
            events.append(.reply(.activity(presentation.activity)))
            return events
        default:
            return []
        }
    }

    private mutating func apply(_ delta: JSONValue, at index: Int) -> [AssistantEvent] {
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

    private mutating func stop(at index: Int) -> [AssistantEvent] {
        guard let partial = partialInputs.removeValue(forKey: index) else { return [] }
        let trimmed = partial.trimmingCharacters(in: .whitespacesAndNewlines)
        let parsed: JSONValue? = trimmed.isEmpty ? nil : try? JSONValue.parse(trimmed)
        switch blocks[index]?["type"]?.stringValue {
        case "tool_use":
            // Client tool input streams unvalidated. Input that isn't a JSON object is answered
            // with INVALID_JSON; its block goes back with an empty input.
            guard !trimmed.isEmpty else { return [] }
            if let parsed, parsed.objectValue != nil {
                blocks[index]?["input"] = parsed
            } else {
                invalidInputs[index] = partial
                blocks[index]?["input"] = .object([:])
            }
            return []
        case "server_tool_use":
            if let parsed {
                blocks[index]?["input"] = parsed
            }
            showingActivity = true
            let description = Self.activityDescription(
                tool: blocks[index]?["name"]?.stringValue,
                input: blocks[index]?["input"]
            )
            return [.reply(.activity(description))]
        default:
            if let parsed {
                blocks[index]?["input"] = parsed
            }
            return []
        }
    }

    private mutating func textEvents(_ text: String) -> [AssistantEvent] {
        guard !text.isEmpty else { return [] }
        if showingActivity {
            showingActivity = false
            return [.reply(.activity(nil)), .reply(.text(text))]
        }
        return [.reply(.text(text))]
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
