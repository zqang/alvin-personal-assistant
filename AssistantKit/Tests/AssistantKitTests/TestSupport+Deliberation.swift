import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
@testable import AssistantKit

/// Which deep-mode call a request is, told by its brief.
enum DeepCall: Hashable {
    /// A worker, by brief id.
    case worker(String)
    case merger
    /// No brief: the `.single` call or the plain answer.
    case plain
    /// A brief that matches no known worker or merger.
    case unknown(String)
}

/// Reads deep-mode requests.
enum DeepRequests {
    static let instructionsOpen = "<instructions>\n"
    static let instructionsClose = "\n</instructions>"

    /// What kind of call `body` is, matching its brief against `briefs` and `merger`.
    static func kind(
        of body: JSONValue,
        briefs: [WorkerBrief] = [.researcher, .reasoner, .critic],
        merger: String = DeliberationConfiguration.defaultMergerInstruction
    ) -> DeepCall {
        guard let instruction = instruction(of: body) else { return .plain }
        if instruction == merger { return .merger }
        if let brief = briefs.first(where: { $0.instruction == instruction }) { return .worker(brief.id) }
        return .unknown(instruction)
    }

    /// The request's brief: the text of its last system message with text content, or of its last
    /// `<instructions>` block, whichever comes later.
    static func instruction(of body: JSONValue) -> String? {
        var found: String?
        for message in body["messages"]?.arrayValue ?? [] {
            switch message["role"]?.stringValue {
            case "system":
                if let text = message["content"]?.stringValue { found = text }
            case "user":
                for block in message["content"]?.arrayValue ?? [] {
                    guard let text = block["text"]?.stringValue,
                          text.hasPrefix(instructionsOpen), text.hasSuffix(instructionsClose) else { continue }
                    found = String(text.dropFirst(instructionsOpen.count).dropLast(instructionsClose.count))
                }
            default:
                break
            }
        }
        return found
    }

    /// The text of the request's `<analyst_notes>` block, if any.
    static func analystNotes(of body: JSONValue) -> String? {
        var found: String?
        for message in body["messages"]?.arrayValue ?? [] where message["role"]?.stringValue == "user" {
            for block in message["content"]?.arrayValue ?? [] {
                if let text = block["text"]?.stringValue, text.hasPrefix("<analyst_notes>") { found = text }
            }
        }
        return found
    }

    /// Whether the request is a later step of a tool loop: its last message carries tool results.
    static func answersToolResults(_ body: JSONValue) -> Bool {
        guard let last = body["messages"]?.arrayValue?.last, last["role"]?.stringValue == "user" else { return false }
        return last.blockTypes.first == "tool_result"
    }

    /// The `tool_result` blocks of the request's last message.
    static func toolResults(_ body: JSONValue) -> [JSONValue] {
        (body["messages"]?.arrayValue?.last?["content"]?.arrayValue ?? []).filter { $0["type"]?.stringValue == "tool_result" }
    }

    /// The effort of the request's effort-only system message, if it has one.
    static func effortMessage(of body: JSONValue) -> String? {
        body["messages"]?.arrayValue?.first { $0["role"]?.stringValue == "system" && $0["output_config"] != nil }?["output_config"]?["effort"]?.stringValue
    }
}

/// A response that streams `text` and stops with `stop`.
func deepText(_ text: String, stop: String = "end_turn", details: JSONValue? = nil) -> MockTransport.Response {
    ClaudeSSE.response([ClaudeSSE.messageStart()], ClaudeSSE.text(index: 0, text), [ClaudeSSE.stop(stop, details: details)])
}

/// A response that calls the client tools `calls` (id, name) with empty input.
func deepToolCalls(_ calls: [(id: String, name: String)]) -> MockTransport.Response {
    var parts: [String] = [ClaudeSSE.messageStart()]
    for (index, call) in calls.enumerated() {
        parts += MockTransport.toolUse(index: index, id: call.id, name: call.name, json: "{}")
    }
    parts.append(ClaudeSSE.stop("tool_use"))
    return .lines(MockTransport.sse(parts))
}

/// A `ScriptedTransport` that answers by `DeepCall`.
func deepTransport(_ respond: @escaping @Sendable (DeepCall, JSONValue) -> MockTransport.Response) -> ScriptedTransport {
    ScriptedTransport { _, body in respond(DeepRequests.kind(of: body), body) }
}

/// The ids of the workers that have ended, in order.
final class WorkerEndLog: @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [String] = []

    var ids: [String] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    func record(_ id: String) {
        lock.withTestLock { recorded.append(id) }
    }
}

/// The events a consumer task has received so far, readable while it runs.
final class DeepEventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [AssistantEvent] = []

    var events: [AssistantEvent] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    /// Consumes `stream` on a new task, recording each event. The task returns the error that
    /// ended the stream, if any.
    func consume(_ stream: AsyncThrowingStream<AssistantEvent, Error>) -> Task<Error?, Never> {
        Task {
            do {
                for try await event in stream {
                    lock.withTestLock { recorded.append(event) }
                }
                return nil
            } catch {
                return error
            }
        }
    }
}

/// A clock reading that moves forward by `step` seconds each time it is read.
final class SteppingUptime: @unchecked Sendable {
    private let lock = NSLock()
    private let step: TimeInterval
    private var current: TimeInterval = 0

    init(step: TimeInterval) {
        self.step = step
    }

    func next() -> TimeInterval {
        lock.withTestLock {
            current += step
            return current
        }
    }
}

extension Duration {
    /// The duration in seconds.
    var deepSeconds: TimeInterval {
        let parts = components
        return TimeInterval(parts.seconds) + TimeInterval(parts.attoseconds) / 1e18
    }
}

/// A deliberation for tests. With `clock`, its deadline and halfway cue wait on that clock.
func makeDeliberation(
    _ configuration: DeliberationConfiguration,
    claude: ClaudeConfiguration = claudeConfiguration(),
    transport: HTTPStreamingTransport,
    tools: ToolRegistry = .empty,
    toolContext: @escaping @Sendable () -> ToolContext = { ToolContext() },
    clock: ManualClock? = nil,
    uptime: (@Sendable () -> TimeInterval)? = nil,
    workerEnds: WorkerEndLog? = nil
) -> Deliberation {
    var hooks = Deliberation.Hooks()
    if let clock {
        hooks.sleep = { duration in try await clock.sleep(for: duration.deepSeconds) }
    }
    if let uptime {
        hooks.uptime = uptime
    }
    if let workerEnds {
        hooks.workerEnded = { workerEnds.record($0) }
    }
    return Deliberation(
        configuration: configuration,
        claude: claude,
        transport: transport,
        tools: tools,
        toolContext: toolContext,
        hooks: hooks
    )
}

/// `typed()` (or `voice()`) with the given strategy.
func deepConfiguration(_ strategy: DeepStrategy, voice: Bool = false) -> DeliberationConfiguration {
    var configuration = voice ? DeliberationConfiguration.voice() : .typed()
    configuration.strategy = strategy
    return configuration
}
