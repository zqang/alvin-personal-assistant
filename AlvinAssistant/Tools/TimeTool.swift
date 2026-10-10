import AssistantKit
import Foundation

/// `get_current_time`: the phone's local date, time and time zone.
struct CurrentTimeTool: AssistantTool {
    static let name = "get_current_time"

    static let definition = ToolDefinition(
        name: name,
        description: "Get the current local date, time and time zone of the user's phone. Call this when you need the exact time right now.",
        inputSchema: DeviceToolSchema.object([:], required: [])
    )

    var definition: ToolDefinition { Self.definition }
    var effect: ToolEffect { .readOnly }
    var presentation: ToolPresentation { ToolPresentation(activity: "Checking the time", cue: .checking) }

    func run(_ input: [String: JSONValue], context: ToolContext) async throws -> ToolOutput {
        let now = context.now
        let timeZone = context.timeZone
        let calendar = DeviceToolSupport.calendar(timeZone)
        let weekdayIndex = calendar.component(.weekday, from: now) - 1
        let weekdays = calendar.weekdaySymbols
        let weekday = weekdays.indices.contains(weekdayIndex) ? weekdays[weekdayIndex] : ""

        let content: [String: JSONValue] = [
            "now": .string(LocalDateTime.format(now, timeZone: timeZone)),
            "weekday": .string(weekday),
            "time_zone": .string(timeZone.identifier),
            "utc_offset": .string(Self.offset(timeZone.secondsFromGMT(for: now))),
        ]
        return .ok(.object(content))
    }

    /// "+08:00", "-05:30".
    static func offset(_ seconds: Int) -> String {
        let sign = seconds < 0 ? "-" : "+"
        let minutes = abs(seconds) / 60
        let hours = String(minutes / 60)
        let rest = String(minutes % 60)
        return sign + (hours.count < 2 ? "0" + hours : hours) + ":" + (rest.count < 2 ? "0" + rest : rest)
    }
}
