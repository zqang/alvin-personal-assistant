import Foundation

/// A client tool as offered to a model: its name, when to call it, and a JSON Schema for its input.
public struct ToolDefinition: Equatable, Sendable {
    public let name: String
    public let description: String
    public let inputSchema: JSONValue

    public init(name: String, description: String, inputSchema: JSONValue) {
        self.name = name
        self.description = description
        self.inputSchema = inputSchema
    }
}

/// Whether running a tool changes anything outside the conversation.
public enum ToolEffect: String, Equatable, Sendable {
    /// Only reads, e.g. listing events. Safe to run speculatively and in parallel.
    case readOnly
    /// Changes something, e.g. adds a reminder. Runs one at a time and waits for the commit gate.
    case sideEffect
}

/// What a tool run may depend on besides its input.
public struct ToolContext: Sendable {
    public var now: Date
    public var timeZone: TimeZone
    public var locale: Locale
    /// Side-effect tools wait on this before acting, so a reply started early (before the user's
    /// turn was final) never acts on a request the user didn't make. Nil means act immediately.
    public var commitGate: CommitGate?

    public init(now: Date = Date(), timeZone: TimeZone = .current, locale: Locale = .current, commitGate: CommitGate? = nil) {
        self.now = now
        self.timeZone = timeZone
        self.locale = locale
        self.commitGate = commitGate
    }
}

/// The result of one tool run, as returned to the model.
public struct ToolOutput: Equatable, Sendable {
    public var content: JSONValue
    public var isError: Bool
    /// A short line for the user, e.g. "Reminder: Call mum · today 17:00".
    public var summary: String?

    public init(content: JSONValue, isError: Bool = false, summary: String? = nil) {
        self.content = content
        self.isError = isError
        self.summary = summary
    }

    public static func ok(_ content: JSONValue, summary: String? = nil) -> ToolOutput {
        ToolOutput(content: content, isError: false, summary: summary)
    }

    /// `{"error": message}`, marked as an error.
    public static func error(_ message: String) -> ToolOutput {
        ToolOutput(content: ["error": .string(message)], isError: true)
    }

    /// `content` as compact JSON with sorted keys, so equal outputs give identical bytes.
    public var serializedContent: String {
        guard let data = try? content.serialized() else { return "null" }
        return String(decoding: data, as: UTF8.self)
    }
}

/// How the app shows and announces a tool while it runs.
public struct ToolPresentation: Equatable, Sendable {
    /// Shown as the reply's activity line, e.g. "Checking your calendar".
    public var activity: String
    /// Spoken before the answer in voice mode, if any.
    public var cue: SpokenCue?

    public init(activity: String, cue: SpokenCue? = nil) {
        self.activity = activity
        self.cue = cue
    }

    /// For tools that don't describe themselves: "Working on it", with the `.working` cue.
    public static let generic = ToolPresentation(activity: "Working on it", cue: .working)
}

/// A tool the assistant can call on the user's behalf.
public protocol AssistantTool: Sendable {
    var definition: ToolDefinition { get }
    var effect: ToolEffect { get }
    var presentation: ToolPresentation { get }
    /// Gets what the tool needs from the user before it runs, such as a first-time permission
    /// alert. `ToolRunner` calls it before `run` (after the commit gate, for a side effect) and
    /// doesn't count its time against the run's timeout; a throw becomes the call's result and the
    /// tool doesn't run. A cancelled call doesn't wait for it, so it must not act, and `run` must
    /// still work when it wasn't called. The default does nothing.
    func authorize(_ input: [String: JSONValue], context: ToolContext) async throws
    /// `input` has already been validated against `definition.inputSchema`.
    func run(_ input: [String: JSONValue], context: ToolContext) async throws -> ToolOutput
}

extension AssistantTool {
    public func authorize(_ input: [String: JSONValue], context: ToolContext) async throws {}
}

/// One tool call and its result, as stored with a reply and replayed to models in later turns.
public struct ToolCallRecord: Codable, Equatable, Sendable {
    /// The id the model gave the call (`toolu_…` from Claude, `call_…` on device).
    public var id: String
    public var name: String
    public var input: JSONValue
    /// The result content as sent to the model: JSON with sorted keys.
    public var result: String
    public var isError: Bool
    /// A short line for the user, if the tool gave one.
    public var summary: String?

    public init(id: String, name: String, input: JSONValue, result: String, isError: Bool = false, summary: String? = nil) {
        self.id = id
        self.name = name
        self.input = input
        self.result = result
        self.isError = isError
        self.summary = summary
    }

    /// The record of `call` with `output` as its result. A call whose input didn't parse is
    /// recorded with an empty object as its input.
    public init(call: PendingToolCall, output: ToolOutput) {
        self.init(
            id: call.id,
            name: call.name,
            input: call.input ?? .object([:]),
            result: output.serializedContent,
            isError: output.isError,
            summary: output.summary
        )
    }

    private enum CodingKeys: String, CodingKey {
        case id, name, input, result, isError, summary
    }

    /// Missing optional fields take their defaults, so records saved by other builds still load.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        input = try container.decodeIfPresent(JSONValue.self, forKey: .input) ?? .object([:])
        result = try container.decodeIfPresent(String.self, forKey: .result) ?? ""
        isError = try container.decodeIfPresent(Bool.self, forKey: .isError) ?? false
        summary = try container.decodeIfPresent(String.self, forKey: .summary)
    }
}

/// The calls a model made in one response, with their results, in the model's order.
public struct ToolRound: Codable, Equatable, Sendable {
    public var calls: [ToolCallRecord]

    public init(calls: [ToolCallRecord]) {
        self.calls = calls
    }

    /// The calls' non-empty summaries joined with "; ", or nil if there are none.
    public var summaryLine: String? {
        let summaries = calls.compactMap { $0.summary?.trimmed }.filter { !$0.isEmpty }
        return summaries.isEmpty ? nil : summaries.joined(separator: "; ")
    }

    private enum CodingKeys: String, CodingKey {
        case calls
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        calls = try container.decodeIfPresent([ToolCallRecord].self, forKey: .calls) ?? []
    }
}

/// A tool call a model has asked for but that hasn't run yet.
public struct PendingToolCall: Equatable, Sendable {
    public var id: String
    public var name: String
    /// The parsed input, or nil when it isn't valid JSON.
    public var input: JSONValue?
    /// The input as the model wrote it, when it didn't parse.
    public var rawInput: String?

    public init(id: String, name: String, input: JSONValue? = nil, rawInput: String? = nil) {
        self.id = id
        self.name = name
        self.input = input
        self.rawInput = rawInput
    }
}

/// Runs the tools a model calls.
public protocol ToolExecutor: Sendable {
    /// The tools to offer the model, sorted by name.
    var definitions: [ToolDefinition] { get }
    func presentation(for name: String) -> ToolPresentation
    /// Runs one round of calls. Never throws: a call that fails gives an error record.
    /// The records keep the calls' order.
    func run(_ calls: [PendingToolCall], context: ToolContext) async -> ToolRound
}

/// Offers no tools; any call it is asked to run gets the error record "Unknown tool".
public struct NoToolExecutor: ToolExecutor {
    public init() {}

    public var definitions: [ToolDefinition] { [] }

    public func presentation(for name: String) -> ToolPresentation {
        .generic
    }

    public func run(_ calls: [PendingToolCall], context: ToolContext) async -> ToolRound {
        ToolRound(calls: calls.map { ToolCallRecord(call: $0, output: .error("Unknown tool")) })
    }
}
