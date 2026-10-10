import Foundation

/// Removes `<think>…</think>` reasoning from streamed local-model text, so it is never shown or
/// spoken. Qwen-family models can emit it even with thinking turned off, and tags may be split
/// across chunks.
public struct ThinkingFilter: Sendable {
    private static let open = "<think>"
    private static let close = "</think>"

    private var inside = false
    /// Text held back because it could be the start of a tag.
    private var pending = ""
    /// Whether any visible text has been emitted; leading whitespace after a reasoning block is dropped.
    private var emittedText = false

    public init() {}

    /// The visible part of `chunk`, possibly empty.
    public mutating func feed(_ chunk: String) -> String {
        var input = pending + chunk
        pending = ""
        var output = ""
        while !input.isEmpty {
            let tag = inside ? Self.close : Self.open
            if let range = input.range(of: tag) {
                if !inside { output += input[..<range.lowerBound] }
                inside.toggle()
                input = String(input[range.upperBound...])
            } else {
                let keep = Self.partialTagSuffixLength(of: input, tag: tag)
                if !inside { output += input.dropLast(keep) }
                pending = String(input.suffix(keep))
                input = ""
            }
        }
        return visible(output)
    }

    /// Whatever was held back at the end of the stream.
    public mutating func finish() -> String {
        defer { pending = "" }
        return inside ? "" : visible(pending)
    }

    private mutating func visible(_ text: String) -> String {
        var text = text
        if !emittedText {
            text = String(text.drop { $0.isWhitespace })
        }
        if !text.isEmpty { emittedText = true }
        return text
    }

    /// Length of the longest suffix of `text` that is a proper prefix of `tag`.
    static func partialTagSuffixLength(of text: String, tag: String) -> Int {
        for length in stride(from: min(tag.count - 1, text.count), to: 0, by: -1)
        where tag.hasPrefix(text.suffix(length)) {
            return length
        }
        return 0
    }
}
