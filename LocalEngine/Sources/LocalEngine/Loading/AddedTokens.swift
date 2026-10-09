import Foundation

/// A tokenizer's added tokens: strings it encodes as one token wherever they appear in text, so
/// conversation text must not carry them into a prompt (`SpecialTokenEscaper`).
public enum AddedTokens {
    /// The chat-format tokens of the Qwen family: the ChatML markers, think and tool markup, and
    /// the vision, box and fill-in-the-middle markers. Every tokenizer that has them escapes them.
    public static let chatFormat = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>", "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
        "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
        "<|image_pad|>", "<|video_pad|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
        "<|repo_name|>", "<|file_sep|>",
    ]

    /// The `content` of every `added_tokens` entry of the tokenizer.json in `directory`: what
    /// swift-transformers splits text on before anything else. Nil when the file is missing or
    /// doesn't parse.
    public static func read(from directory: URL) -> [String]? {
        let url = directory.appending(component: "tokenizer.json")
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              let file = try? JSONDecoder().decode(TokenizerFile.self, from: data)
        else { return nil }
        return (file.addedTokens ?? []).map(\.content)
    }

    /// The part of tokenizer.json read here.
    private struct TokenizerFile: Decodable {
        struct AddedToken: Decodable {
            let content: String
        }

        let addedTokens: [AddedToken]?

        enum CodingKeys: String, CodingKey {
            case addedTokens = "added_tokens"
        }
    }
}
