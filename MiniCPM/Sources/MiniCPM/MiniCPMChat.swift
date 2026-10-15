import AssistantKit
import Foundation

/// What a streamed MiniCPM reply or photo reading produces, in order: `prefilled` once the prompt
/// is in the cache, `firstText` with the first visible text, `text` pieces, then `finished`.
public enum MiniCPMEvent: Sendable {
    /// `fed` tokens were evaluated; `reused` came from the cache.
    case prefilled(fed: Int, reused: Int)
    case firstText
    case text(String)
    case finished(MiniCPMBenchmarkResult)
}

/// MiniCPM-V 4.7's ChatML, as its chat template renders it with thinking off
/// (`enable_thinking=false`): every content trimmed, and past replies with the empty think block.
///
/// Each piece is tokenized on its own: scaffolding with special tokens parsed, content as plain
/// text, where the bridge also splits the user-defined tags (`<think>`, `<tool_call>`…) that
/// llama.cpp would otherwise still match, so no text in the conversation (a photo's text included)
/// becomes a control, think or tool token. Every render (a full one, a continuation, a replacement
/// reply) splits at the same places, so session reuse stays token-exact.
///
/// Tool rounds aren't rendered: the 35B has no tools, and what Woof did shows in its reply text.
enum MiniCPMChat {
    struct Piece: Equatable {
        var text: String
        var special: Bool
    }

    static let turnStart = "<|im_start|>"
    static let turnEnd = "<|im_end|>\n"
    static let generationPrompt = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    /// `<|im_start|>system\n…<|im_end|>\n`, or nothing for an empty prompt, as `minicpm_run` sends.
    static func system(_ system: String) -> [Piece] {
        system.isEmpty ? [] : [.init(text: "<|im_start|>system\n", special: true), content(system), end]
    }

    static func turns<C: Collection>(_ turns: C) -> [Piece] where C.Element == ChatTurn {
        turns.flatMap { turn -> [Piece] in
            switch turn.role {
            case .user: return [.init(text: "<|im_start|>user\n", special: true), content(LocalSessionPlan.content(of: turn)), end]
            case .assistant: return [.init(text: generationPrompt, special: true), content(turn.text), end]
            }
        }
    }

    /// `[system] + turns` with the generation prompt.
    static func fullRender(system: String, turns: [ChatTurn]) -> [Piece] {
        Self.system(system) + Self.turns(turns) + [prompt]
    }

    /// What follows a cached reply: its turn end, `turns`, the generation prompt. Equals the
    /// sentinel delta (`TurnDelta`) of a full render.
    static func continuation(_ turns: [ChatTurn]) -> [Piece] {
        [end] + Self.turns(turns) + [prompt]
    }

    /// A replacement reply, without its turn end.
    static func assistantText(_ text: String) -> [Piece] {
        [content(text)]
    }

    static func text(of pieces: [Piece]) -> String {
        pieces.map(\.text).joined()
    }

    private static let end = Piece(text: turnEnd, special: true)
    private static let prompt = Piece(text: generationPrompt, special: true)

    private static func content(_ text: String) -> Piece {
        Piece(text: text.trimmingCharacters(in: .whitespacesAndNewlines), special: false)
    }
}
