import Foundation

public struct ClaudeModelOption: Identifiable, Hashable, Sendable {
    public let id: String
    public let displayName: String
    public let summary: String
}

/// Request features that differ between Claude models.
public struct ClaudeModelCapabilities: Equatable, Sendable {
    /// Accepts `output_config.effort`.
    public var supportsEffort: Bool
    /// Web search tool version to declare.
    public var webSearchToolType: String
    /// Accepts `fallbacks: "default"`, which re-runs a classifier-declined request on another model server-side.
    public var supportsDefaultFallbacks: Bool
}

public enum ClaudeModelCatalog {
    public static let defaultModelID = "claude-opus-5"

    public static let options: [ClaudeModelOption] = [
        ClaudeModelOption(id: "claude-opus-5", displayName: "Claude Opus 5", summary: "Most capable Opus model"),
        ClaudeModelOption(id: "claude-sonnet-5", displayName: "Claude Sonnet 5", summary: "Fast and capable at a lower price"),
        ClaudeModelOption(id: "claude-haiku-4-5", displayName: "Claude Haiku 4.5", summary: "Fastest and cheapest"),
        ClaudeModelOption(id: "claude-fable-5-1", displayName: "Claude Fable 5.1", summary: "Most capable overall; slower and pricier"),
    ]

    /// Effort trades answer depth for latency and cost; `low` suits conversation.
    public static let effortLevels = ["low", "medium", "high"]

    public static func capabilities(for modelID: String) -> ClaudeModelCapabilities {
        let id = modelID.lowercased()
        if id.contains("haiku") {
            return ClaudeModelCapabilities(
                supportsEffort: false,
                webSearchToolType: "web_search_20250305",
                supportsDefaultFallbacks: false
            )
        }
        return ClaudeModelCapabilities(
            supportsEffort: true,
            webSearchToolType: "web_search_20260209",
            supportsDefaultFallbacks: id == "claude-opus-5" || id == "claude-fable-5-1"
        )
    }

    public static func displayName(for modelID: String) -> String {
        options.first { $0.id == modelID }?.displayName ?? modelID
    }
}
