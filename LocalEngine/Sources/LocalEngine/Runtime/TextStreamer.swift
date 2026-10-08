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
            if let text = processor.processChunk(chunk) {
                emit(thinking.feed(text), into: &outputs)
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
        if let rest = processor.processEOS(returnBufferedText: true),
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
        return (text, processor.toolCalls.map(JSONBridge.pendingCall))
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
