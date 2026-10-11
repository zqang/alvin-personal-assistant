import AssistantKit
import Foundation

/// This app's tool-call benchmark: the scenario the on-device benchmark adds to
/// `LocalBenchmark.scenarios`, and the stand-in results its calls get.
public enum ToolBenchmark {
    /// Requests for this app's tools, each in a new conversation, with the expected tool and key
    /// arguments: the drafter lab's `tool-01` to `tool-12`, at the lab's fixed time. Run it with a
    /// tool executor that records the calls and answers them with `toolResult(for:now:timeZone:)`
    /// instead of acting on them.
    public static let tools = LocalBenchmark.Scenario(
        id: "tools",
        title: "Tool calls",
        steps: LocalBenchmark.separateConversations(toolPrompts.map(\.prompt)),
        context: LocalBenchmark.labSpokenContext,
        expectations: Dictionary(uniqueKeysWithValues: toolPrompts.enumerated().map { ($0.offset, $0.element.expectation) })
    )

    static let toolPrompts: [(prompt: String, expectation: LocalBenchmark.ToolExpectation)] = [
        ("Remind me at 5 to call mum.", LocalBenchmark.ToolExpectation(name: "create_reminder", arguments: ["title": "call mum", "due": "2026-10-07T17:00"])),
        ("Set a timer for ten minutes for the pasta.", LocalBenchmark.ToolExpectation(name: "set_timer", arguments: ["seconds": 600])),
        ("What's on my calendar tomorrow?", LocalBenchmark.ToolExpectation(name: "list_events", arguments: ["start": "2026-10-08"])),
        ("Put a dentist appointment in my calendar for Friday at 3pm.", LocalBenchmark.ToolExpectation(name: "create_event", arguments: ["title": "dentist", "start": "2026-10-09T15:00"])),
        ("Which of my reminders are due today?", LocalBenchmark.ToolExpectation(name: "list_reminders", arguments: ["scope": "today"])),
        ("Add buy milk to my to-do list.", LocalBenchmark.ToolExpectation(name: "create_reminder", arguments: ["title": "buy milk"])),
        ("提醒我明天早上八点去取快递。", LocalBenchmark.ToolExpectation(name: "create_reminder", arguments: ["title": "快递", "due": "2026-10-08T08:00"])),
        ("帮我设一个二十分钟的计时器。", LocalBenchmark.ToolExpectation(name: "set_timer", arguments: ["seconds": 1200])),
        ("我今天下午有什么安排？", LocalBenchmark.ToolExpectation(name: "list_events", arguments: ["start": "2026-10-07"])),
        ("下周一上午十点和王经理开会，帮我加到日历里。", LocalBenchmark.ToolExpectation(name: "create_event", arguments: ["title": "王经理", "start": "2026-10-12T10:00"])),
        ("有哪些提醒已经过期了？", LocalBenchmark.ToolExpectation(name: "list_reminders", arguments: ["scope": "overdue"])),
        ("我的计时器还剩多少时间？", LocalBenchmark.ToolExpectation(name: "list_timers")),
    ]

    /// What the benchmark answers a call with instead of running it, so the model carries on as
    /// after a real call while nothing on the phone changes: a result in the shape the app's tool
    /// returns (`AlvinAssistant/Tools`), at `now`. Lists are empty; what a call creates echoes its
    /// input, with times written as the tools write them; a tool the benchmark doesn't offer gets
    /// `ToolRunner`'s "Unknown tool" error, as in the app, so a model that invents a name can
    /// still correct itself. The app answers `get_current_time` with its real tool at the turn's
    /// `LocalBenchmark.time(ofContext:)`.
    public static func toolResult(for call: PendingToolCall, now: Date, timeZone: TimeZone) -> ToolOutput {
        let input = call.input?.objectValue ?? [:]
        func text(_ key: String) -> String? {
            guard let value = input[key]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty else { return nil }
            return value
        }
        // On-device models write numbers as strings too.
        func number(_ key: String) -> Int? {
            input[key]?.intValue ?? text(key).flatMap { Int($0) }
        }
        func local(_ date: Date, hasTime: Bool = true) -> JSONValue {
            .string(LocalDateTime.format(date, timeZone: timeZone, includeTime: hasTime))
        }
        // A time from the input as the tools write it back; as given when it can't be read.
        func time(_ key: String) -> JSONValue? {
            guard let value = text(key) else { return nil }
            return LocalDateTime.parse(value, timeZone: timeZone).map { local($0.date, hasTime: $0.hasTime) } ?? .string(value)
        }

        var content: [String: JSONValue]
        switch call.name {
        case "list_events":
            content = ["count": 0, "events": []]
            content["from"] = time("start")
            content["to"] = time("end")
        case "list_reminders":
            content = ["scope": .string(text("scope")?.lowercased() ?? "all"), "count": 0, "reminders": []]
        case "list_timers":
            content = ["count": 0, "timers": []]
        case "create_reminder":
            content = ["status": "added", "id": "benchmark", "title": .string(text("title") ?? ""), "list": "Reminders"]
            content["due"] = time("due")
        case "create_event":
            let start = text("start").flatMap { LocalDateTime.parse($0, timeZone: timeZone) }
            content = [
                "status": "added", "title": .string(text("title") ?? ""), "calendar": "Calendar",
                "all_day": .bool(input["all_day"]?.boolValue ?? (start.map { !$0.hasTime } ?? false)),
            ]
            content["start"] = time("start")
            content["end"] = time("end") ?? start.map {
                local($0.date.addingTimeInterval(TimeInterval((number("duration_minutes") ?? 60) * 60)), hasTime: $0.hasTime)
            }
        case "set_timer":
            let seconds = number("seconds") ?? 0
            content = ["status": "started", "id": "timer.benchmark", "seconds": .int(seconds), "ends": local(now.addingTimeInterval(TimeInterval(seconds)))]
            content["label"] = text("label").map(JSONValue.string)
        default:
            return .error(ToolRunner.unknownToolMessage)
        }
        return .ok(.object(content))
    }
}
