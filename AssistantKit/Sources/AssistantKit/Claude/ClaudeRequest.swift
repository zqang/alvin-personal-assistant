import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

public struct ClaudeConfiguration: Equatable, Sendable {
    public var apiKey: String
    public var model: String
    /// `output_config.effort`; ignored for models that don't accept it.
    public var effort: String?
    public var webSearchEnabled: Bool
    /// Caps thinking plus visible text for one response.
    public var maxTokens: Int
    /// IANA time zone used to localize web search results.
    public var timeZoneIdentifier: String?
    public var baseURL: URL

    public init(
        apiKey: String,
        model: String = ClaudeModelCatalog.defaultModelID,
        effort: String? = "low",
        webSearchEnabled: Bool = true,
        maxTokens: Int = 16_000,
        timeZoneIdentifier: String? = nil,
        baseURL: URL = URL(string: "https://api.anthropic.com")!
    ) {
        self.apiKey = apiKey
        self.model = model
        self.effort = effort
        self.webSearchEnabled = webSearchEnabled
        self.maxTokens = maxTokens
        self.timeZoneIdentifier = timeZoneIdentifier
        self.baseURL = baseURL
    }
}

/// Builds Messages API requests (raw HTTP; Swift has no official Anthropic SDK).
enum ClaudeRequest {
    static let apiVersion = "2023-06-01"
    static let fallbackBeta = "server-side-fallback-2026-07-01"
    static let webSearchMaxUses = 5

    static func messages(from turns: [ChatTurn]) -> [JSONValue] {
        turns.map { turn -> JSONValue in
            var blocks: [JSONValue] = []
            if turn.role == .user, let context = turn.context, !context.isEmpty {
                blocks.append(textBlock(context))
            }
            blocks.append(textBlock(turn.text))
            let message: [String: JSONValue] = ["role": .string(turn.role.rawValue), "content": .array(blocks)]
            return .object(message)
        }
    }

    static func textBlock(_ text: String) -> JSONValue {
        let block: [String: JSONValue] = ["type": .string("text"), "text": .string(text)]
        return .object(block)
    }

    static func body(configuration: ClaudeConfiguration, system: String, messages: [JSONValue]) -> JSONValue {
        let capabilities = ClaudeModelCatalog.capabilities(for: configuration.model)
        let cacheControl: JSONValue = ["type": "ephemeral"]
        var systemBlock: [String: JSONValue] = ["type": .string("text"), "text": .string(system)]
        systemBlock["cache_control"] = cacheControl
        // The system prompt is static (no timestamps), so it caches; the top-level marker
        // caches the growing conversation, which is only ever appended to.
        var body: [String: JSONValue] = [
            "model": .string(configuration.model),
            "max_tokens": .int(configuration.maxTokens),
            "stream": .bool(true),
            "system": .array([.object(systemBlock)]),
            "cache_control": cacheControl,
            "messages": .array(messages),
        ]
        if capabilities.supportsEffort, let effort = configuration.effort, !effort.isEmpty {
            let outputConfig: [String: JSONValue] = ["effort": .string(effort)]
            body["output_config"] = .object(outputConfig)
        }
        if configuration.webSearchEnabled {
            var tool: [String: JSONValue] = [
                "type": .string(capabilities.webSearchToolType),
                "name": .string("web_search"),
                "max_uses": .int(webSearchMaxUses),
            ]
            if let zone = configuration.timeZoneIdentifier, zone.contains("/") {
                let location: [String: JSONValue] = ["type": .string("approximate"), "timezone": .string(zone)]
                tool["user_location"] = .object(location)
            }
            body["tools"] = .array([.object(tool)])
        }
        if capabilities.supportsDefaultFallbacks {
            body["fallbacks"] = .string("default")
        }
        return .object(body)
    }

    static func urlRequest(configuration: ClaudeConfiguration, body: JSONValue) throws -> URLRequest {
        var request = URLRequest(url: configuration.baseURL.appendingPathComponent("v1/messages"))
        request.httpMethod = "POST"
        request.timeoutInterval = 120
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.setValue(configuration.apiKey, forHTTPHeaderField: "x-api-key")
        request.setValue(apiVersion, forHTTPHeaderField: "anthropic-version")
        if ClaudeModelCatalog.capabilities(for: configuration.model).supportsDefaultFallbacks {
            request.setValue(fallbackBeta, forHTTPHeaderField: "anthropic-beta")
        }
        request.httpBody = try body.serialized()
        return request
    }
}
