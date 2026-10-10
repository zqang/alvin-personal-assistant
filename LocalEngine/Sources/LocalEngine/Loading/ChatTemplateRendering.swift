import Foundation

/// What the engine needs from a tokenizer beyond `MLXLMCommon.Tokenizer`: rendering a chat
/// template with or without the generation prompt, and token-exact encoding and decoding with
/// special tokens kept. Session reuse depends on these being exact (plan §4.5).
public protocol ChatTemplateRendering: Sendable {
    /// Renders `messages` through the model's chat template and tokenizes the result.
    /// `context` carries template variables such as `enable_thinking`.
    func renderTokens(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                      context: [String: any Sendable]?, addGenerationPrompt: Bool) throws -> [Int]
    /// Encodes text without adding special tokens (`addSpecialTokens: false`).
    func encodeRaw(_ text: String) -> [Int]
    /// Decodes tokens keeping special tokens (`skipSpecialTokens: false`).
    func decodeRaw(_ tokens: [Int]) -> String
    /// The id of `token` when it is a single entry of the vocabulary, else nil (never the
    /// unknown token's id).
    func tokenID(_ token: String) -> Int?
    /// The strings encoded as one added token wherever they appear in text, such as
    /// `<|im_start|>` and `<tool_call>`. Defaults to the `AddedTokens.chatFormat` entries of the
    /// vocabulary.
    var addedTokenLiterals: [String] { get }
}

extension ChatTemplateRendering {
    public var addedTokenLiterals: [String] {
        chatFormatTokenLiterals
    }

    /// The `AddedTokens.chatFormat` entries that are single entries of the vocabulary.
    var chatFormatTokenLiterals: [String] {
        AddedTokens.chatFormat.filter { tokenID($0) != nil }
    }

    /// The tokens that end a reply: the configuration's EOS ids, the tokenizer's EOS token, and
    /// `<|im_end|>` and `<|endoftext|>` when the vocabulary has them.
    public func stopTokenIDs(eosTokenIds: Set<Int>, eosToken: String?) -> Set<Int> {
        var ids = eosTokenIds
        for token in [eosToken, "<|im_end|>", "<|endoftext|>"] {
            if let token, let id = tokenID(token) {
                ids.insert(id)
            }
        }
        return ids
    }

    /// Whether the vocabulary has the ChatML turn markers that token-exact session reuse cuts at.
    public var isChatML: Bool {
        tokenID("<|im_start|>") != nil && tokenID("<|im_end|>") != nil
    }
}
