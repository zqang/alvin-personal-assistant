import Foundation

/// A model that can answer on the iPhone through MLX.
public struct LocalModelOption: Identifiable, Equatable, Sendable {
    /// Hugging Face repository of the MLX weights.
    public let id: String
    public let displayName: String
    /// Rough resident size of the weights, for the free-memory check before loading.
    public let approximateBytes: Int
    /// A smaller model with the same tokenizer that drafts tokens for speculative decoding.
    /// Nil when the model's architecture can't use it: hybrid models such as Qwen3.5 keep a
    /// recurrent state that can't be rolled back after a rejected draft.
    public let draftModelID: String?
    public let draftApproximateBytes: Int
    public let note: String

    public init(id: String, displayName: String, approximateBytes: Int, draftModelID: String? = nil, draftApproximateBytes: Int = 0, note: String) {
        self.id = id
        self.displayName = displayName
        self.approximateBytes = approximateBytes
        self.draftModelID = draftModelID
        self.draftApproximateBytes = draftApproximateBytes
        self.note = note
    }

    public var supportsSpeculativeDecoding: Bool { draftModelID != nil }
}

public enum LocalModelCatalog {
    public static let woof4B = LocalModelOption(
        id: "ConwayResearch/Underdog-Woof-4B-1.1",
        displayName: "Underdog Woof 4B",
        approximateBytes: 2_500_000_000,
        note: "Conway Research's open model (Qwen3.5-based), tuned for tool calls. About 2.5 GB."
    )

    public static let qwen35_2B = LocalModelOption(
        id: "mlx-community/Qwen3.5-2B-4bit",
        displayName: "Qwen3.5 2B",
        approximateBytes: 1_600_000_000,
        note: "Smaller and faster general chat model, same family as Woof."
    )

    public static let qwen3_4B = LocalModelOption(
        id: "mlx-community/Qwen3-4B-4bit",
        displayName: "Qwen3 4B + 0.6B draft",
        approximateBytes: 2_300_000_000,
        draftModelID: "mlx-community/Qwen3-0.6B-4bit",
        draftApproximateBytes: 400_000_000,
        note: "Standard attention, so a 0.6B draft model can speed it up (speculative decoding)."
    )

    public static let options = [woof4B, qwen35_2B, qwen3_4B]
    public static let defaultModelID = woof4B.id

    public static func option(for id: String) -> LocalModelOption? {
        options.first { $0.id == id }
    }
}
