import AssistantKit
import Foundation

/// Neutralizes added-token literals in conversation text before it is rendered: user turns,
/// replies, tool calls and tool results. The tokenizer reads such a literal as its control token
/// wherever it appears, so an event title holding `<|im_end|><|im_start|>user …` would forge a
/// turn. A zero-width space goes after each literal's first character: the text reads the same
/// to the model, and no literal is left whole.
///
/// The escape is a pure function of the text, so every render of the same turns (full renders,
/// sentinel deltas, replacement replies) gets the same tokens and session reuse stays exact.
public struct SpecialTokenEscaper: Sendable {
    /// What goes after the first character of each literal.
    public static let separator: Unicode.Scalar = "\u{200B}"

    /// The literals, as scalars: at least two each (one character can't be split), none holding
    /// the separator.
    private let literals: [[Unicode.Scalar]]
    private let firstScalars: Set<Unicode.Scalar>

    public init<S: Sequence>(literals: S) where S.Element == String {
        self.literals = Array(Set(literals.map { Array($0.unicodeScalars) }).filter { $0.count >= 2 && !$0.contains(Self.separator) })
        self.firstScalars = Set(self.literals.map { $0[0] })
    }

    /// The literals of `renderer`'s vocabulary (`addedTokenLiterals`).
    public init(renderer: any ChatTemplateRendering) {
        self.init(literals: renderer.addedTokenLiterals)
    }

    /// `text` with the separator after the first character of every literal it holds
    /// (overlapping ones included).
    public func escape(_ text: String) -> String {
        guard text.unicodeScalars.contains(where: firstScalars.contains) else { return text }
        let scalars = Array(text.unicodeScalars)
        var escaped = String.UnicodeScalarView()
        var changed = false
        for index in scalars.indices {
            escaped.append(scalars[index])
            if firstScalars.contains(scalars[index]), literals.contains(where: { Self.matches($0, in: scalars, at: index) }) {
                escaped.append(Self.separator)
                changed = true
            }
        }
        return changed ? String(escaped) : text
    }

    /// `value` with every string and object key escaped (tool-call arguments).
    public func escape(_ value: JSONValue) -> JSONValue {
        switch value {
        case .string(let text):
            return .string(escape(text))
        case .array(let values):
            return .array(values.map { escape($0) })
        case .object(let members):
            var escaped: [String: JSONValue] = [:]
            for (key, member) in members {
                escaped[escape(key)] = escape(member)
            }
            return .object(escaped)
        case .null, .bool, .int, .double:
            return value
        }
    }

    /// `record` with its id, name, input and result escaped.
    public func escape(_ record: ToolCallRecord) -> ToolCallRecord {
        var record = record
        record.id = escape(record.id)
        record.name = escape(record.name)
        record.input = escape(record.input)
        record.result = escape(record.result)
        return record
    }

    private static func matches(_ literal: [Unicode.Scalar], in scalars: [Unicode.Scalar], at index: Int) -> Bool {
        guard index + literal.count <= scalars.count else { return false }
        for offset in literal.indices where scalars[index + offset] != literal[offset] {
            return false
        }
        return true
    }
}
