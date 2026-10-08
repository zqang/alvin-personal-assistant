import AssistantKit
import Foundation
import MLXLMCommon

/// Turns emitted tokens into what the user sees, and finds the tool calls (plan §4.6):
/// `NaiveStreamingDetokenizer` → `ToolCallProcessor` → `ThinkingFilter` → visible text.
///
/// It also reports tool calls as they start: once when the start token (`<tool_call>`) is
/// emitted (`toolCallStarted(nil)`) and once the function's name can be read from
/// `<function=NAME>` or `"name": "NAME"` (`toolCallStarted(NAME)`).
///
/// Tool-call markup is never shown. `ToolCallProcessor` hands a closed call it can't parse
/// (malformed arguments, a tool the request didn't declare) back as plain text; the streamer
/// takes it out and reports it as a call with no `input` and the model's text in `rawInput`,
/// so the tool runner answers it with an error the model can act on.
///
/// Stop tokens never reach the streamer. Not thread-safe: engine queue only.
public final class TextStreamer {
    public enum Output: Equatable {
        case text(String)
        case toolCallStarted(String?)
    }

    private var detokenizer: NaiveStreamingDetokenizer
    private let processor: ToolCallProcessor
    private var thinking = ThinkingFilter()
    private let startTag: String?
    private let endTag: String?
    private let startToken: Int?
    private let endToken: Int?

    /// Raw text of the tool call being written, for reading its name.
    private var callText = ""
    private var nameReported = false
    /// Every tool call seen to start, with its name once known.
    public private(set) var startedCalls: [String?] = []
    /// The reply's tool calls so far, in the order they were written (of calls that end in the
    /// same detokenizer chunk, the parsed ones come first).
    private var calls: [PendingToolCall] = []
    /// How many of the processor's parsed calls are in `calls`.
    private var parsedCount = 0
    /// Whether the text so far is inside a tool call.
    public private(set) var insideToolCall = false
    /// All visible text so far.
    public private(set) var visibleText = ""

    public init(tokenizer: any MLXLMCommon.Tokenizer, renderer: any ChatTemplateRendering, format: ToolCallFormat, tools: [[String: any Sendable]]?) {
        self.detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        self.processor = ToolCallProcessor(format: format, tools: tools)
        let parser = format.createParser()
        self.startTag = parser.startTag
        self.endTag = parser.endTag
        self.startToken = parser.startTag.flatMap { renderer.tokenID($0) }
        self.endToken = parser.endTag.flatMap { renderer.tokenID($0) }
    }

    /// Feeds emitted tokens; returns the visible text and tool-call progress they produced.
    public func append(_ tokens: [Int]) -> [Output] {
        var outputs: [Output] = []
        for token in tokens {
            let isStart = startToken == token
            if isStart {
                beginCall(&outputs)
            }
            detokenizer.append(token: token)
            guard let chunk = detokenizer.next(), !chunk.isEmpty else {
                if endToken == token { insideToolCall = false }
                continue
            }
            if startToken == nil, let startTag, chunk.contains(startTag), !insideToolCall {
                beginCall(&outputs)
            }
            if insideToolCall {
                callText += chunk
                readName(&outputs)
                if endToken == token || (endToken == nil && endTag.map { callText.contains($0) } == true) {
                    insideToolCall = false
                }
            }
            let text = processor.processChunk(chunk)
            collectParsedCalls()
            if let text {
                emit(thinking.feed(takeUnparsedCalls(from: text)), into: &outputs)
            }
        }
        return outputs
    }

    /// Ends the stream: parses anything still buffered and returns the remaining visible text
    /// and the reply's tool calls.
    public func finish() -> (text: String, toolCalls: [PendingToolCall]) {
        var outputs: [Output] = []
        // Text held back as a possible tool call is shown, unless it is the start of a call that
        // was cut off (by the length limit or a cancellation): markup is never shown.
        let rest = processor.processEOS(returnBufferedText: true)
        collectParsedCalls()
        if let rest = rest.map(takeUnparsedCalls(from:)),
           !(startTag.map { rest.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix($0) } ?? false)
        {
            emit(thinking.feed(rest), into: &outputs)
        }
        emit(thinking.finish(), into: &outputs)
        let text = outputs.compactMap { output -> String? in
            if case .text(let text) = output { return text }
            return nil
        }.joined()
        insideToolCall = false
        return (text, calls)
    }

    /// Moves the calls the processor parsed since the last look into `calls`.
    private func collectParsedCalls() {
        let parsed = processor.toolCalls
        guard parsed.count > parsedCount else { return }
        calls += parsed[parsedCount...].map(JSONBridge.pendingCall)
        parsedCount = parsed.count
    }

    /// `text` from the processor without the closed tool calls it couldn't parse (each
    /// `startTag … endTag`), which are added to `calls` with their payload as `rawInput`.
    private func takeUnparsedCalls(from text: String) -> String {
        guard let startTag, let endTag, text.contains(startTag) else { return text }
        var visible = ""
        var rest = text[...]
        while let start = rest.range(of: startTag),
              let end = rest.range(of: endTag, range: start.upperBound ..< rest.endIndex)
        {
            visible += String(rest[..<start.lowerBound])
            let payload = rest[start.upperBound ..< end.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
            calls.append(PendingToolCall(id: JSONBridge.newCallID(), name: Self.functionName(in: payload) ?? "", input: nil, rawInput: payload))
            rest = rest[end.upperBound...]
        }
        visible += String(rest)
        return visible
    }

    private func beginCall(_ outputs: inout [Output]) {
        insideToolCall = true
        callText = ""
        nameReported = false
        startedCalls.append(nil)
        outputs.append(.toolCallStarted(nil))
    }

    private func readName(_ outputs: inout [Output]) {
        guard !nameReported, let name = Self.functionName(in: callText) else { return }
        nameReported = true
        if !startedCalls.isEmpty {
            startedCalls[startedCalls.count - 1] = name
        }
        outputs.append(.toolCallStarted(name))
    }

    private func emit(_ text: String, into outputs: inout [Output]) {
        guard !text.isEmpty else { return }
        visibleText += text
        outputs.append(.text(text))
    }

    /// The function name in the text of a tool call being written, once it is complete:
    /// `<function=NAME>` (xml-function format) or `"name": "NAME"` (JSON format).
    static func functionName(in text: String) -> String? {
        if let start = text.range(of: "<function="),
           let end = text.range(of: ">", range: start.upperBound ..< text.endIndex)
        {
            let name = text[start.upperBound ..< end.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
            return name.isEmpty ? nil : name
        }
        if let key = text.range(of: "\"name\"") {
            var index = key.upperBound
            while index < text.endIndex, text[index].isWhitespace { index = text.index(after: index) }
            guard index < text.endIndex, text[index] == ":" else { return nil }
            index = text.index(after: index)
            while index < text.endIndex, text[index].isWhitespace { index = text.index(after: index) }
            guard index < text.endIndex, text[index] == "\"" else { return nil }
            let nameStart = text.index(after: index)
            guard let close = text[nameStart...].firstIndex(of: "\"") else { return nil }
            let name = String(text[nameStart ..< close])
            return name.isEmpty ? nil : name
        }
        return nil
    }
}
