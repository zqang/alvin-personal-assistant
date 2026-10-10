import AssistantKit
import Foundation
import MLXLMCommon

/// Drafts the fixed structure of a tool call (plan §4.7): one skeleton per tool of the request,
/// in the model's tool-call format, wrapped in `ExemplarSeeds`. Once the model writes the
/// opening `<tool_call>`, the skeletons predict the markup and the tool's name, then the first
/// required parameter's key.
///
/// - `.xmlFunction`: `<tool_call>\n<function=NAME>\n<parameter=P>\n`;
/// - `.json`: `<tool_call>\n{"name": "NAME", "arguments": {"P": `;
/// - other formats: no skeletons (the drafter never proposes).
///
/// Skeletons contain special tokens; the speculative loop lets this drafter (and only this one)
/// draft them, and asks it only while the reply is inside a tool call.
public final class ExemplarDrafter: Drafter {
    public var source: DraftSource { .exemplar }
    public var costPerToken: Double { 0 }
    public var wantsHidden: Bool { false }

    /// The skeletons' token sequences, one per tool.
    public let seeds: ExemplarSeeds

    public init(tools: [ToolDefinition], format: ToolCallFormat, renderer: any ChatTemplateRendering) {
        let sequences = Self.skeletons(tools: tools, format: format)
            .map { renderer.encodeRaw($0) }
            .filter { $0.count >= 2 }
        seeds = ExemplarSeeds(sequences: sequences)
    }

    /// Whether the drafter has any skeleton.
    public var isEmpty: Bool { seeds.sequences.isEmpty }

    /// The skeleton texts of `tools` in `format`, in tool order.
    public static func skeletons(tools: [ToolDefinition], format: ToolCallFormat) -> [String] {
        tools.compactMap { tool in
            let parameter = firstRequiredParameter(tool.inputSchema)
            switch format {
            case .xmlFunction:
                var text = "<tool_call>\n<function=\(tool.name)>\n"
                if let parameter { text += "<parameter=\(parameter)>\n" }
                return text
            case .json:
                var text = "<tool_call>\n{\"name\": \(quoted(tool.name)), \"arguments\": "
                if let parameter { text += "{\(quoted(parameter)): " }
                return text
            default:
                return nil
            }
        }
    }

    /// The first entry of the schema's `required` list.
    static func firstRequiredParameter(_ schema: AssistantKit.JSONValue) -> String? {
        guard case .array(let required)? = schema["required"] else { return nil }
        return required.lazy.compactMap(\.stringValue).first { !$0.isEmpty }
    }

    /// A JSON string literal, as the templates write names and keys (which are identifiers, so
    /// only quotes, backslashes and control characters need escaping).
    private static func quoted(_ text: String) -> String {
        var out = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
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

    public func reset(ledger: [Int], request: EngineRequest) {}

    public func propose(context: ArraySlice<Int>, maxTokens: Int) -> DraftProposal? {
        seeds.propose(context: context, maxTokens: maxTokens)
    }

    public func observe(_ round: RoundObservation) {}
}
