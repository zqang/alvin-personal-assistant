import AssistantKit
import Foundation
import LocalEngine
import LocalEngineTestSupport
import XCTest

/// `LocalToolLoop` on a real model with its own chat template and tool-call format
/// (xml-function for Qwen3.5): reminder prompts from the lab, with the on-device tool subset of
/// plan §5.3 (schemas from `scripts/lab_prompts.json`, `handoff_to_cloud` as the app defines
/// it), the app's on-device system prompt and greedy decoding.
///
/// Whether the model calls `create_reminder` with the right arguments is reported, not
/// asserted (plan WP31: report only). What is asserted is the engine's side: the tool-call
/// format, a loop that ends normally (or hands off), and a cache that still agrees with a fresh
/// rebuild afterwards.
final class ToolCallIntegrationTests: XCTestCase {
    override func setUpWithError() throws {
        try IntegrationEnvironment.requireEnabled()
        try MetalAvailability.require()
    }

    func testReminderCallOnQwen35_0_8B() async throws {
        try await runReminderPrompts("mlx-community/Qwen3.5-0.8B-MLX-4bit")
    }

    func testReminderCallOnWoof() async throws {
        guard IntegrationEnvironment.woof else {
            throw XCTSkip("Woof runs only with LOCAL_ENGINE_WOOF=1 ([ci woof]).")
        }
        try await runReminderPrompts(IntegrationEnvironment.woofRepo)
    }

    private func runReminderPrompts(_ repo: String) async throws {
        let lab = try Lab.load()
        let prompts = lab.prompts.filter { $0.expectedName == "create_reminder" }
        XCTAssertFalse(prompts.isEmpty, "the lab has no reminder prompts")

        let directory = try await IntegrationEnvironment.snapshot(repo)
        var configuration = EngineConfiguration()
        configuration.temperature = 0
        configuration.maxTokens = 200
        let engine = try await InferenceEngine.load(directory: directory, modelID: repo, configuration: configuration)
        try await engine.warmUp()
        XCTAssertEqual(engine.info.toolCallFormat, "xml_function", "\(repo): Qwen3.5 templates write xml-function calls")

        let log = IntegrationToolLog()
        let runner = ToolRunner(registry: ToolRegistry(lab.tools.map { IntegrationTool(definition: $0, log: log) as any AssistantTool }), lenientInput: true)
        let system = PromptBuilder.localSystemPrompt(
            base: PromptBuilder.systemPrompt(userName: "", customInstructions: ""), handoffAvailable: true)
        let loop = LocalToolLoop(
            engine: engine, executor: runner, handoffAvailable: true, isOnline: { true }, toolContext: { ToolContext() })

        var rows: [[String]] = []
        var parsedReminders = 0
        for prompt in prompts {
            let turn = ChatTurn(role: .user, text: prompt.text, context: "<context>time: \(lab.contextTime); input: \(prompt.input)</context>")
            let request = EngineRequest(system: system, tools: runner.definitions, turns: [turn], greedy: true)
            let reports = IntegrationReports()
            let runsBefore = log.runs.count
            var events: [AssistantEvent] = []
            var failure: Error?
            do {
                for try await event in loop.run(request, report: reports.add) {
                    events.append(event)
                }
            } catch {
                failure = error
            }
            if let failure, !(failure is ReplyHandoff) {
                XCTFail("\(repo) \(prompt.id): the loop failed: \(failure)")
            }

            let records = events.flatMap { event -> [ToolCallRecord] in
                if case .toolRound(let round) = event { return round.calls }
                return []
            }
            let runs = Array(log.runs.dropFirst(runsBefore))
            let reminder = runs.first { $0.name == "create_reminder" }
            if reminder != nil { parsedReminders += 1 }
            let matches = reminder.map { Self.matches($0.input, prompt.expectedArguments) } ?? false

            let (report, ledger) = try await engine.withSession { ($0.assertConsistent(), $0.ledger.count) }
            XCTAssertTrue(report.agrees(margin: NearTieTally.quantizedTolerance), "\(repo) \(prompt.id): \(report)")

            let text = events.compactMap { event -> String? in
                if case .reply(.text(let text)) = event { return text }
                return nil
            }.joined()
            let outcome: String
            if let handoff = failure as? ReplyHandoff {
                outcome = "handoff (\(handoff.reason))"
            } else if let failure {
                outcome = "error: \(failure)"
            } else {
                outcome = events.compactMap { event -> String? in
                    if case .reply(.finished(let stop)) = event { return "\(stop)" }
                    return nil
                }.last ?? "–"
            }
            let stats = reports.all.first?.stats
            rows.append([
                prompt.id,
                Self.cell(records.map { "\($0.name)(\(Self.json($0.input)))\($0.isError ? " → error \($0.result)" : "")" }.joined(separator: "; ")),
                Self.cell("create_reminder(\(Self.json(prompt.expectedArguments)))"),
                reminder == nil ? "no call" : (matches ? "yes" : "args differ"),
                Self.cell(String(text.prefix(60))),
                Self.cell(outcome),
                "\(reports.all.first?.rounds ?? 0)",
                stats?.timeToFirstText.map { String(format: "%.0f", $0 * 1000) } ?? "–",
                "\(stats?.generatedTokens ?? 0)",
                "\(ledger)",
            ])
        }
        EngineReport.appendTable(
            title: "Local tool loop, \(repo) (greedy, \(engine.info.toolCallFormat), \(lab.tools.count) tools)",
            header: ["Prompt", "Calls", "Expected", "Parsed create_reminder", "Reply", "Outcome", "Rounds", "TTFT ms", "Tokens", "Ledger"],
            rows: rows)
        EngineReport.append("- Parsed create_reminder calls (\(repo)): \(parsedReminders)/\(prompts.count) (report only)")
    }

    // MARK: Helpers

    /// Whether every expected argument appears in the actual one (case-insensitive substring:
    /// "Call mum" for "call mum", "2026-10-07T17:00:00" for "2026-10-07T17:00").
    private static func matches(_ actual: [String: JSONValue], _ expected: JSONValue) -> Bool {
        guard let expected = expected.objectValue else { return true }
        for (key, value) in expected {
            guard let found = actual[key] else { return false }
            let want = (value.stringValue ?? json(value)).lowercased()
            let got = (found.stringValue ?? json(found)).lowercased()
            if !got.contains(want) { return false }
        }
        return true
    }

    private static func json(_ value: JSONValue) -> String {
        (try? value.serialized()).map { String(decoding: $0, as: UTF8.self) } ?? "?"
    }

    /// Text for a Markdown table cell.
    private static func cell(_ text: String) -> String {
        text.replacingOccurrences(of: "\n", with: " ").replacingOccurrences(of: "|", with: "/")
    }
}

/// The lab's prompts and the shared tool schemas (`scripts/lab_prompts.json`, plan §5.3).
private struct Lab {
    struct Prompt {
        let id: String
        let text: String
        /// "spoken" or "typed".
        let input: String
        let expectedName: String
        let expectedArguments: JSONValue
    }

    /// The local day and time the prompts were written for.
    let contextTime: String
    /// The on-device subset, sorted by name; `handoff_to_cloud` as the app defines it.
    let tools: [ToolDefinition]
    let prompts: [Prompt]

    struct Missing: Error, CustomStringConvertible {
        let description: String
    }

    static func load() throws -> Lab {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // LocalEngineIntegrationTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // LocalEngine
            .deletingLastPathComponent()  // the repository
            .appendingPathComponent("scripts/lab_prompts.json")
        let lab = try JSONValue.parse(Data(contentsOf: url))
        guard let contextTime = lab["context_time"]?.stringValue,
              let local = lab["local_tools"]?.arrayValue?.compactMap(\.stringValue),
              let schemas = lab["tool_schemas"]?.arrayValue,
              let cases = lab["tools"]?.arrayValue
        else {
            throw Missing(description: "\(url.path) lacks context_time, local_tools, tool_schemas or tools")
        }

        var tools: [ToolDefinition] = []
        for schema in schemas {
            guard let name = schema["name"]?.stringValue, local.contains(name) else { continue }
            if name == HandoffTool.name {
                tools.append(HandoffTool.definition)
            } else {
                tools.append(ToolDefinition(
                    name: name, description: schema["description"]?.stringValue ?? "",
                    inputSchema: schema["input_schema"] ?? .object([:])))
            }
        }

        let prompts = cases.compactMap { item -> Prompt? in
            guard let id = item["id"]?.stringValue, let text = item["text"]?.stringValue,
                  let name = item["expect"]?["name"]?.stringValue
            else { return nil }
            return Prompt(
                id: id, text: text, input: item["input"]?.stringValue ?? "spoken", expectedName: name,
                expectedArguments: item["expect"]?["args"] ?? .object([:]))
        }
        return Lab(contextTime: contextTime, tools: tools.sorted { $0.name < $1.name }, prompts: prompts)
    }
}

/// Stands in for a device tool: validates nothing itself, records its input, answers "ok".
private struct IntegrationTool: AssistantTool {
    let definition: ToolDefinition
    let log: IntegrationToolLog

    var effect: ToolEffect {
        definition.name.hasPrefix("list_") || definition.name.hasPrefix("get_") ? .readOnly : .sideEffect
    }

    var presentation: ToolPresentation { .generic }

    func run(_ input: [String: JSONValue], context: ToolContext) async throws -> ToolOutput {
        log.record(name: definition.name, input: input)
        return .ok(["ok": true, "id": .string("\(definition.name)-1")], summary: definition.name)
    }
}

private final class IntegrationToolLog: @unchecked Sendable {
    struct Run {
        let name: String
        let input: [String: JSONValue]
    }

    private let lock = NSLock()
    private var recorded: [Run] = []

    func record(name: String, input: [String: JSONValue]) {
        lock.lock()
        recorded.append(Run(name: name, input: input))
        lock.unlock()
    }

    var runs: [Run] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }
}

private final class IntegrationReports: @unchecked Sendable {
    private let lock = NSLock()
    private var reports: [LocalToolLoop.Report] = []

    var add: @Sendable (LocalToolLoop.Report) -> Void {
        { [self] report in
            lock.lock()
            reports.append(report)
            lock.unlock()
        }
    }

    var all: [LocalToolLoop.Report] {
        lock.lock()
        defer { lock.unlock() }
        return reports
    }
}
