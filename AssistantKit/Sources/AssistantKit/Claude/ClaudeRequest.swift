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
///
/// Everything here is a pure function of its inputs and serializes with sorted keys, so the same
/// history always renders to the same bytes and the prompt cache stays warm.
enum ClaudeRequest {
    static let apiVersion = "2023-06-01"
    static let fallbackBeta = "server-side-fallback-2026-07-01"
    static let perMessageEffortBeta = "mid-conversation-output-config-2026-07-01"
    static let webSearchMaxUses = 5
    /// `eager_input_streaming` is sent only to this host; proxies may reject the field.
    static let directHost = "api.anthropic.com"
    static let toolIDPrefix = "toolu_"
    /// Server tools' names, which a client tool may not take (only `web_search` is ever declared).
    static let serverToolNames: Set<String> = ["web_search", "web_fetch"]
    static let maxToolNameLength = 64

    // MARK: - Options

    /// `options` with an effort that would change nothing cleared: an empty one, one equal to the
    /// configuration's, or any effort for a model that doesn't take one.
    static func resolved(_ options: ClaudeTurnOptions, for configuration: ClaudeConfiguration) -> ClaudeTurnOptions {
        var resolved = options
        let capabilities = ClaudeModelCatalog.capabilities(for: configuration.model)
        if let effort = nonEmpty(options.effort), capabilities.supportsEffort, effort != nonEmpty(configuration.effort) {
            resolved.effort = effort
        } else {
            resolved.effort = nil
        }
        return resolved
    }

    // MARK: - Messages

    /// Renders `turns` as Messages API messages, then applies `options` around the last user turn.
    ///
    /// An assistant turn with tool rounds becomes, per round, an assistant message of `tool_use`
    /// blocks and a user message of their `tool_result` blocks, then an assistant message with its
    /// text. When it has no text, the next user turn's blocks join the last results message instead,
    /// so roles keep alternating.
    ///
    /// `options.effort` is inserted as a per-message effort change wherever `capabilities` allow it:
    /// pass options from `resolved(_:for:)` so an effort equal to the configuration's isn't repeated.
    static func messages(
        from turns: [ChatTurn],
        options: ClaudeTurnOptions = ClaudeTurnOptions(),
        capabilities: ClaudeModelCapabilities = ClaudeModelCatalog.capabilities(for: ClaudeModelCatalog.defaultModelID)
    ) -> [JSONValue] {
        var messages: [[String: JSONValue]] = []
        // Whether the last message holds tool results that the next user turn's blocks should join.
        var resultsAwaitUserTurn = false

        for turn in turns {
            switch turn.role {
            case .user:
                var blocks: [JSONValue] = []
                if let context = turn.context, !context.isEmpty {
                    blocks.append(textBlock(context))
                }
                blocks.append(textBlock(turn.text))
                if resultsAwaitUserTurn, let last = messages.indices.last {
                    messages[last]["content"] = .array((messages[last]["content"]?.arrayValue ?? []) + blocks)
                } else {
                    messages.append(message(role: "user", content: blocks))
                }
                resultsAwaitUserTurn = false
            case .assistant:
                let rounds = turn.toolRounds.filter { !$0.calls.isEmpty }
                for round in rounds {
                    messages.append(message(role: "assistant", content: round.calls.map(toolUseBlock)))
                    messages.append(message(role: "user", content: round.calls.map { toolResultBlock($0) }))
                }
                if rounds.isEmpty || !turn.text.isEmpty {
                    messages.append(message(role: "assistant", content: [textBlock(turn.text)]))
                    resultsAwaitUserTurn = false
                } else {
                    resultsAwaitUserTurn = true
                }
            }
        }
        apply(options, capabilities: capabilities, to: &messages)
        return messages.map(JSONValue.object)
    }

    static func textBlock(_ text: String) -> JSONValue {
        let block: [String: JSONValue] = ["type": .string("text"), "text": .string(text)]
        return .object(block)
    }

    /// A stored call as the `tool_use` block that asked for it.
    static func toolUseBlock(_ record: ToolCallRecord) -> JSONValue {
        let input: JSONValue = record.input.objectValue == nil ? .object([:]) : record.input
        let block: [String: JSONValue] = [
            "type": .string("tool_use"),
            "id": .string(claudeToolID(record.id)),
            "name": .string(claudeToolName(record.name)),
            "input": input,
        ]
        return .object(block)
    }

    /// A call's result as a `tool_result` block. `is_error` is sent only when true.
    ///
    /// A stored round (`remapID` true) answers the `tool_use` that `toolUseBlock` renders, so both
    /// carry `claudeToolID(record.id)`. The live loop passes false: it appends the server's
    /// `tool_use` blocks unchanged, so each result must quote the server's own id, even one that
    /// doesn't start with `toolu_` (an Anthropic-compatible endpoint behind a custom `baseURL`).
    static func toolResultBlock(_ record: ToolCallRecord, remapID: Bool = true) -> JSONValue {
        var block: [String: JSONValue] = [
            "type": .string("tool_result"),
            "tool_use_id": .string(remapID ? claudeToolID(record.id) : record.id),
        ]
        if !record.result.isEmpty {
            block["content"] = .string(record.result)
        }
        if record.isError {
            block["is_error"] = .bool(true)
        }
        return .object(block)
    }

    /// The id Claude sees for a tool call. Ids Claude generated (`toolu_…`) pass through; others,
    /// such as the on-device model's `call_…`, become `toolu_` plus the id with every character
    /// outside `[A-Za-z0-9_-]` replaced by `_`. Deterministic, so re-rendered history stays cacheable.
    static func claudeToolID(_ id: String) -> String {
        if id.hasPrefix(toolIDPrefix) { return id }
        return toolIDPrefix + safeCharacters(id)
    }

    /// The name Claude sees for a stored call, which `ToolRegistry.covering(_:)` also declares.
    /// The API takes only 1 to 64 characters from `[A-Za-z0-9_-]`, unique among the request's
    /// tools, but the on-device model may write any name: every other character becomes `_`, the
    /// name is cut to 64 characters, an empty one becomes `unknown_tool`, and a server tool's name
    /// gets the prefix `local_`. Valid names pass through. Deterministic, so re-rendered history
    /// stays cacheable.
    static func claudeToolName(_ name: String) -> String {
        let sanitized = String(safeCharacters(name).prefix(maxToolNameLength))
        if sanitized.isEmpty { return "unknown_tool" }
        return serverToolNames.contains(sanitized) ? "local_" + sanitized : sanitized
    }

    /// `text` with every character outside `[A-Za-z0-9_-]` replaced by `_`.
    private static func safeCharacters(_ text: String) -> String {
        var sanitized = ""
        for scalar in text.unicodeScalars {
            switch scalar.value {
            case 0x30...0x39, 0x41...0x5A, 0x61...0x7A, 0x2D, 0x5F: // 0-9, A-Z, a-z, "-", "_"
                sanitized.unicodeScalars.append(scalar)
            default:
                sanitized += "_"
            }
        }
        return sanitized
    }

    static func message(role: String, content: [JSONValue]) -> [String: JSONValue] {
        ["role": .string(role), "content": .array(content)]
    }

    /// Adds `options`' data, instruction and effort around the last user message.
    private static func apply(
        _ options: ClaudeTurnOptions,
        capabilities: ClaudeModelCapabilities,
        to messages: inout [[String: JSONValue]]
    ) {
        guard let lastUser = messages.lastIndex(where: { $0["role"]?.stringValue == "user" }) else { return }
        let instruction = nonBlank(options.instruction)

        var appended: [JSONValue] = []
        if let data = nonBlank(options.userData) {
            appended.append(textBlock(data))
        }
        if let instruction, !capabilities.supportsMidConversationSystem {
            appended.append(textBlock("<instructions>\n\(instruction)\n</instructions>"))
        }
        if !appended.isEmpty {
            messages[lastUser]["content"] = .array((messages[lastUser]["content"]?.arrayValue ?? []) + appended)
        }
        if let instruction, capabilities.supportsMidConversationSystem {
            let system: [String: JSONValue] = ["role": .string("system"), "content": .string(instruction)]
            messages.insert(system, at: lastUser + 1)
        }
        if let effort = nonEmpty(options.effort), capabilities.supportsPerMessageEffort {
            let outputConfig: [String: JSONValue] = ["effort": .string(effort)]
            let change: [String: JSONValue] = [
                "role": .string("system"),
                "content": .array([]),
                "output_config": .object(outputConfig),
            ]
            messages.insert(change, at: effortPosition(lastUser: lastUser, in: messages))
        }
    }

    /// Where the effort change goes: just before the last user message, unless that message opens
    /// with tool results, which must directly follow their `tool_use` message; then just before that.
    private static func effortPosition(lastUser: Int, in messages: [[String: JSONValue]]) -> Int {
        let opensWithResults = messages[lastUser]["content"]?.arrayValue?.first?["type"]?.stringValue == "tool_result"
        if opensWithResults, lastUser > 0, messages[lastUser - 1]["role"]?.stringValue == "assistant" {
            return lastUser - 1
        }
        return lastUser
    }

    // MARK: - Body

    static func body(
        configuration: ClaudeConfiguration,
        system: String,
        messages: [JSONValue],
        clientTools: [ToolDefinition] = [],
        options: ClaudeTurnOptions = ClaudeTurnOptions()
    ) -> JSONValue {
        let capabilities = ClaudeModelCatalog.capabilities(for: configuration.model)
        let cacheControl: JSONValue = ["type": "ephemeral"]
        var systemBlock: [String: JSONValue] = ["type": .string("text"), "text": .string(system)]
        systemBlock["cache_control"] = cacheControl
        // The system prompt is static (no timestamps), so it caches; the top-level marker
        // caches the growing conversation, which is only ever appended to.
        var body: [String: JSONValue] = [
            "model": .string(configuration.model),
            "max_tokens": .int(options.maxTokens ?? configuration.maxTokens),
            "stream": .bool(true),
            "system": .array([.object(systemBlock)]),
            "cache_control": cacheControl,
            "messages": .array(messages),
        ]
        if capabilities.supportsEffort, let effort = topLevelEffort(configuration: configuration, options: options, capabilities: capabilities) {
            let outputConfig: [String: JSONValue] = ["effort": .string(effort)]
            body["output_config"] = .object(outputConfig)
        }

        // Tools render before the system prompt in the cache prefix, so their order is fixed:
        // web search, then client tools by name. `tool_choice` stays the default `auto`.
        var tools: [JSONValue] = []
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
            tools.append(.object(tool))
        }
        let eager = options.eagerToolInput && configuration.baseURL.host?.lowercased() == directHost
        for definition in clientTools.sorted(by: { $0.name < $1.name }) {
            tools.append(clientTool(definition, eagerInput: eager))
        }
        if !tools.isEmpty {
            body["tools"] = .array(tools)
        }
        if capabilities.supportsDefaultFallbacks {
            body["fallbacks"] = .string("default")
        }
        return .object(body)
    }

    /// The top-level effort: the configuration's, unless the options override it on a model that
    /// can't take the override as a per-message change.
    static func topLevelEffort(
        configuration: ClaudeConfiguration,
        options: ClaudeTurnOptions,
        capabilities: ClaudeModelCapabilities
    ) -> String? {
        if let override = nonEmpty(options.effort), !capabilities.supportsPerMessageEffort {
            return override
        }
        return nonEmpty(configuration.effort)
    }

    static func clientTool(_ definition: ToolDefinition, eagerInput: Bool) -> JSONValue {
        var tool: [String: JSONValue] = [
            "name": .string(definition.name),
            "description": .string(definition.description),
            "input_schema": definition.inputSchema,
            "strict": .bool(true),
        ]
        if eagerInput {
            tool["eager_input_streaming"] = .bool(true)
        }
        return .object(tool)
    }

    // MARK: - HTTP

    static func urlRequest(configuration: ClaudeConfiguration, body: JSONValue) throws -> URLRequest {
        var request = URLRequest(url: configuration.baseURL.appendingPathComponent("v1/messages"))
        request.httpMethod = "POST"
        request.timeoutInterval = 120
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        setAPIHeaders(on: &request, configuration: configuration)
        let betas = betas(configuration: configuration, body: body)
        if !betas.isEmpty {
            request.setValue(betas.joined(separator: ","), forHTTPHeaderField: "anthropic-beta")
        }
        request.httpBody = try body.serialized()
        return request
    }

    /// The betas `body` needs: server-side fallbacks where the model has them, and per-message
    /// effort when a message carries `output_config`.
    static func betas(configuration: ClaudeConfiguration, body: JSONValue) -> [String] {
        var betas: [String] = []
        if ClaudeModelCatalog.capabilities(for: configuration.model).supportsDefaultFallbacks {
            betas.append(fallbackBeta)
        }
        let changesEffort = body["messages"]?.arrayValue?.contains { $0["output_config"] != nil } ?? false
        if changesEffort {
            betas.append(perMessageEffortBeta)
        }
        return betas
    }

    /// `GET {baseURL}/v1/models?limit=1` with the API headers: a cheap request that opens the
    /// connection (DNS, TLS) before the user's turn needs it.
    static func prewarmRequest(configuration: ClaudeConfiguration) throws -> URLRequest {
        guard !configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AssistantError.missingAPIKey(service: "Anthropic")
        }
        let endpoint = configuration.baseURL.appendingPathComponent("v1/models")
        guard var components = URLComponents(url: endpoint, resolvingAgainstBaseURL: false) else {
            throw URLError(.badURL)
        }
        components.queryItems = [URLQueryItem(name: "limit", value: "1")]
        guard let url = components.url else { throw URLError(.badURL) }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 15
        setAPIHeaders(on: &request, configuration: configuration)
        return request
    }

    private static func setAPIHeaders(on request: inout URLRequest, configuration: ClaudeConfiguration) {
        request.setValue(configuration.apiKey, forHTTPHeaderField: "x-api-key")
        request.setValue(apiVersion, forHTTPHeaderField: "anthropic-version")
    }

    /// `text` trimmed, or nil when nothing is left.
    private static func nonEmpty(_ text: String?) -> String? {
        guard let text = text?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else { return nil }
        return text
    }

    /// `text` unchanged, or nil when it is only whitespace.
    private static func nonBlank(_ text: String?) -> String? {
        guard let text, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        return text
    }
}
