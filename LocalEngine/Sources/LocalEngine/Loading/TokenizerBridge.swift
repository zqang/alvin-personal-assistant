import Foundation
import MLXLMCommon
import Tokenizers

/// Loads tokenizers with swift-transformers, as MLXHuggingFace's macro would, without the macro.
/// Copied from the app's `LocalModelHost.swift`.
public struct TransformersTokenizerLoader: MLXLMCommon.TokenizerLoader {
    public init() {}

    public func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        TokenizerBridge(upstream: try await AutoTokenizer.from(modelFolder: directory), addedTokens: AddedTokens.read(from: directory))
    }
}

/// A swift-transformers tokenizer seen as an `MLXLMCommon.Tokenizer`, plus the template and
/// raw-token access the engine needs (`ChatTemplateRendering`).
public struct TokenizerBridge: MLXLMCommon.Tokenizer, ChatTemplateRendering {
    public let upstream: any Tokenizers.Tokenizer
    /// The added tokens of the tokenizer's files (`AddedTokens.read(from:)`), when known:
    /// swift-transformers keeps its own list private.
    public let addedTokens: [String]?

    public init(upstream: any Tokenizers.Tokenizer, addedTokens: [String]? = nil) {
        self.upstream = upstream
        self.addedTokens = addedTokens
    }

    // MARK: MLXLMCommon.Tokenizer

    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    public func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    public func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    public func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    public var bosToken: String? { upstream.bosToken }
    public var eosToken: String? { upstream.eosToken }
    public var unknownToken: String? { upstream.unknownToken }

    public func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(messages: messages, tools: tools, additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }

    // MARK: ChatTemplateRendering

    public func renderTokens(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                             context: [String: any Sendable]?, addGenerationPrompt: Bool) throws -> [Int] {
        do {
            // The full protocol requirement (F13): the only entry point that can leave out the
            // generation prompt.
            return try upstream.applyChatTemplate(
                messages: messages,
                chatTemplate: nil,
                addGenerationPrompt: addGenerationPrompt,
                truncation: false,
                maxLength: nil,
                tools: tools,
                additionalContext: context
            )
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }

    public func encodeRaw(_ text: String) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: false)
    }

    public func decodeRaw(_ tokens: [Int]) -> String {
        upstream.decode(tokens: tokens, skipSpecialTokens: false)
    }

    public func tokenID(_ token: String) -> Int? {
        // swift-transformers maps a missing token to the unknown token's id; only accept an id
        // that maps back to the same token.
        guard let id = upstream.convertTokenToId(token), upstream.convertIdToToken(id) == token else {
            return nil
        }
        return id
    }

    /// The files' added tokens and the chat-format tokens of the vocabulary.
    public var addedTokenLiterals: [String] {
        Set(chatFormatTokenLiterals).union(addedTokens ?? []).sorted()
    }
}
