import AssistantKit
import Foundation

/// `set_timer`: starts a countdown that alerts the user with a notification.
struct SetTimerTool: AssistantTool {
    static let name = "set_timer"

    /// The longest timer, in seconds (24 hours).
    static let maxSeconds = 86_400

    static let definition = ToolDefinition(
        name: name,
        description: "Start a countdown timer on the phone. Call this whenever the user asks for a timer or to be alerted after a length of time.",
        inputSchema: DeviceToolSchema.object(
            [
                "seconds": DeviceToolSchema.integer("How long the timer runs, in seconds, from 1 to 86400. For example, ten minutes is 600."),
                "label": DeviceToolSchema.string("What the timer is for, e.g. \"Pasta\", only if the user said."),
            ],
            required: ["seconds"]
        )
    )

    var definition: ToolDefinition { Self.definition }
    var effect: ToolEffect { .sideEffect }
    var presentation: ToolPresentation { ToolPresentation(activity: "Setting a timer", cue: nil) }

    func run(_ input: [String: JSONValue], context: ToolContext) async throws -> ToolOutput {
        guard let seconds = DeviceToolSupport.integer(input, "seconds") else {
            return .error("seconds must be a whole number of seconds, e.g. 600 for ten minutes.")
        }
        guard (1...Self.maxSeconds).contains(seconds) else {
            return .error("seconds must be between 1 and \(Self.maxSeconds) (24 hours).")
        }
        let label = DeviceToolSupport.text(input, "label")

        let timer = try await TimerService.shared.start(seconds: seconds, label: label)

        var content: [String: JSONValue] = [
            "status": .string("started"),
            "id": .string(timer.id),
            "seconds": .int(seconds),
            "ends": .string(LocalDateTime.format(timer.ends, timeZone: context.timeZone)),
        ]
        if let label {
            content["label"] = .string(label)
        }
        let length = DeviceToolSupport.durationLabel(seconds: seconds)
        let summary = label.map { "Timer: \($0) · \(length)" } ?? "Timer: \(length)"
        return .ok(.object(content), summary: summary)
    }
}

/// `list_timers`: the timers that are still running.
struct ListTimersTool: AssistantTool {
    static let name = "list_timers"

    static let definition = ToolDefinition(
        name: name,
        description: "List the timers that are running, with the time left on each. Call this when the user asks about their timers, and before cancelling one.",
        inputSchema: DeviceToolSchema.object([:], required: [])
    )

    var definition: ToolDefinition { Self.definition }
    var effect: ToolEffect { .readOnly }
    var presentation: ToolPresentation { ToolPresentation(activity: "Checking your timers", cue: .checking) }

    func run(_ input: [String: JSONValue], context: ToolContext) async throws -> ToolOutput {
        let timers = await TimerService.shared.running()
        let items = timers.map { timer -> JSONValue in
            var item: [String: JSONValue] = [
                "id": .string(timer.id),
                "ends": .string(LocalDateTime.format(timer.ends, timeZone: context.timeZone)),
                "seconds_left": .int(max(0, Int(timer.ends.timeIntervalSince(context.now).rounded(.up)))),
            ]
            if let label = timer.label {
                item["label"] = .string(label)
            }
            return .object(item)
        }
        let fitted = DeviceToolSupport.fitting(items)

        var content: [String: JSONValue] = [
            "count": .int(timers.count),
            "timers": .array(fitted.kept),
        ]
        if fitted.omitted > 0 {
            content["more"] = .int(fitted.omitted)
        }
        let summary = timers.isEmpty ? "Timers: none running" : "Timers: \(timers.count) running"
        return .ok(.object(content), summary: summary)
    }
}

/// `cancel_timer`: stops a running timer.
struct CancelTimerTool: AssistantTool {
    static let name = "cancel_timer"

    static let definition = ToolDefinition(
        name: name,
        description: "Cancel a running timer. Get the id from list_timers first.",
        inputSchema: DeviceToolSchema.object(
            ["id": DeviceToolSchema.string("The timer's id, from list_timers.")],
            required: ["id"]
        )
    )

    var definition: ToolDefinition { Self.definition }
    var effect: ToolEffect { .sideEffect }
    var presentation: ToolPresentation { ToolPresentation(activity: "Cancelling a timer", cue: nil) }

    func run(_ input: [String: JSONValue], context: ToolContext) async throws -> ToolOutput {
        guard let id = DeviceToolSupport.text(input, "id") else {
            return .error("The id is empty. Call list_timers to get the timer's id.")
        }
        let timers = await TimerService.shared.running()
        guard let timer = Self.match(id, in: timers) else {
            return .error("There's no running timer with id \"\(id)\". Call list_timers to get the current ids.")
        }
        try Task.checkCancellation()
        TimerService.shared.cancel(ids: [timer.id])

        var content: [String: JSONValue] = [
            "status": .string("cancelled"),
            "id": .string(timer.id),
        ]
        if let label = timer.label {
            content["label"] = .string(label)
        }
        let summary = timer.label.map { "Timer cancelled: \($0)" } ?? "Timer cancelled"
        return .ok(.object(content), summary: summary)
    }

    /// The timer `id` names: its identifier, with or without the `timer.` prefix and in any case,
    /// or else the label of exactly one timer.
    static func match(_ id: String, in timers: [TimerInfo]) -> TimerInfo? {
        let wanted = id.lowercased()
        let prefixed = (TimerService.identifierPrefix + id).lowercased()
        if let timer = timers.first(where: { $0.id.lowercased() == wanted || $0.id.lowercased() == prefixed }) {
            return timer
        }
        let labelled = timers.filter { $0.label?.lowercased() == wanted }
        return labelled.count == 1 ? labelled[0] : nil
    }
}
