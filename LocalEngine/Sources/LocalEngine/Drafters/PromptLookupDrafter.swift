import AssistantKit
import Foundation

/// Prompt-lookup drafting over the live session (plan §4.7): finds the context's latest 2–4
/// tokens earlier in the session (system prompt, tools, history, tool results, the reply so far)
/// and proposes what followed them there. Replies that copy from the prompt (edits, quotes, tool
/// arguments) get long accepted runs this way, at no model cost.
///
/// Wraps `NGramIndex`. The index is kept across replies: `reset` syncs it to the new ledger by
/// truncating to the common prefix and appending the rest, and `propose` appends whatever the
/// context added since (within a reply the context only grows).
///
/// Never proposes `<|im_start|>`, `<|endoftext|>` or the role tokens (`user`, `assistant`,
/// `system` when they are single vocabulary entries): a continuation stops before them.
/// Not thread-safe: engine queue only.
public final class PromptLookupDrafter: Drafter {
    public var source: DraftSource { .promptLookup }
    /// A hash lookup on the CPU: free next to a target forward.
    public var costPerToken: Double { 0 }
    public var wantsHidden: Bool { false }

    public private(set) var index: NGramIndex

    public init(excluded: Set<Int>, minN: Int = 2, maxN: Int = 4) {
        index = NGramIndex(minN: minN, maxN: maxN, excluded: excluded)
    }

    /// A drafter that excludes the turn markers and role tokens of `renderer`'s vocabulary.
    public convenience init(renderer: any ChatTemplateRendering, minN: Int = 2, maxN: Int = 4) {
        self.init(excluded: Self.excludedTokens(renderer: renderer), minN: minN, maxN: maxN)
    }

    /// `<|im_start|>`, `<|endoftext|>`, and the role tokens that are single vocabulary entries.
    public static func excludedTokens(renderer: any ChatTemplateRendering) -> Set<Int> {
        let names = ["<|im_start|>", "<|endoftext|>", "user", "assistant", "system"]
        return Set(names.compactMap { renderer.tokenID($0) })
    }

    public func reset(ledger: [Int], request: EngineRequest) {
        index.sync(to: ledger)
    }

    public func propose(context: ArraySlice<Int>, maxTokens: Int) -> DraftProposal? {
        guard maxTokens > 0 else { return nil }
        sync(to: context)
        return index.propose(maxTokens: maxTokens)
    }

    public func observe(_ round: RoundObservation) {}

    /// Makes the index hold exactly `context`. The common case, a context that extends the
    /// indexed tokens, only appends; anything else falls back to `NGramIndex.sync`.
    private func sync(to context: ArraySlice<Int>) {
        let indexed = index.count
        if indexed <= context.count && Self.endsMatch(index.tokens, context, count: indexed) {
            if indexed < context.count {
                index.append(contentsOf: context.dropFirst(indexed))
            }
        } else {
            index.sync(to: Array(context))
        }
    }

    /// Whether the last few of the first `count` tokens of `context` equal the end of `tokens`
    /// (`tokens.count == count`): a cheap check that the index is a prefix of the context. The
    /// index only ever grows by contexts of the same reply, and `reset` syncs it in full, so a
    /// matching tail means a matching prefix.
    private static func endsMatch(_ tokens: [Int], _ context: ArraySlice<Int>, count: Int) -> Bool {
        let checked = min(count, 8)
        guard checked > 0 else { return true }
        for offset in 1 ... checked where tokens[count - offset] != context[context.startIndex + count - offset] {
            return false
        }
        return true
    }
}
