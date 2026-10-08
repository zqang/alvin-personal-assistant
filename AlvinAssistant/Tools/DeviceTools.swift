import AssistantKit
import Foundation

/// The tools that act on this iPhone (plan §5.3): reminders, calendar, timers and the time.
///
/// Every registry built here offers the same definitions whatever the settings say. Tools the user
/// turned off keep their definition and answer calls with "The user turned this off in Settings.",
/// so the tool list, and with it the cached prompt prefix, only changes when the app does.
@MainActor
enum DeviceTools {
    /// The switches in Settings. Each turns a group of tools on or off through `disabledTools`.
    enum Category: String, CaseIterable, Identifiable, Sendable {
        case reminders, calendar, timers

        var id: String { rawValue }

        var title: String {
            switch self {
            case .reminders: return "Reminders"
            case .calendar: return "Calendar"
            case .timers: return "Timers"
            }
        }

        var systemImage: String {
            switch self {
            case .reminders: return "checklist"
            case .calendar: return "calendar"
            case .timers: return "timer"
            }
        }

        /// The tools this switch controls, sorted by name.
        var toolNames: [String] {
            switch self {
            case .reminders: return [CompleteReminderTool.name, CreateReminderTool.name, ListRemindersTool.name]
            case .calendar: return [CreateEventTool.name, ListEventsTool.name]
            case .timers: return [CancelTimerTool.name, ListTimersTool.name, SetTimerTool.name]
            }
        }
    }

    /// The tools offered to the on-device model (plan §5.3, "Local subset"); `handoff_to_cloud` is
    /// added separately when a cloud key exists.
    static let localToolNames: Set<String> = [
        CreateEventTool.name,
        CreateReminderTool.name,
        CurrentTimeTool.name,
        ListEventsTool.name,
        ListRemindersTool.name,
        SetTimerTool.name,
    ]

    /// Every device tool.
    static var allTools: [any AssistantTool] {
        [
            CancelTimerTool(),
            CompleteReminderTool(),
            CreateEventTool(),
            CreateReminderTool(),
            CurrentTimeTool(),
            ListEventsTool(),
            ListRemindersTool(),
            ListTimersTool(),
            SetTimerTool(),
        ]
    }

    /// Names of the tools that answer with an error instead of running: those in `disabledTools`,
    /// and every device tool when `deviceToolsEnabled` is off.
    static func disabledToolNames(settings: AssistantSettings) -> Set<String> {
        var disabled = Set(settings.disabledTools)
        if !settings.deviceToolsEnabled {
            disabled.formUnion(allTools.map(\.definition.name))
        }
        return disabled
    }

    /// All device tools, for Claude.
    static func registry(settings: AssistantSettings) -> ToolRegistry {
        ToolRegistry(allTools, disabled: disabledToolNames(settings: settings))
    }

    /// The on-device subset, plus `handoff_to_cloud` when `handoff` is true.
    static func localRegistry(settings: AssistantSettings, handoff: Bool) -> ToolRegistry {
        let local = registry(settings: settings).subset(localToolNames)
        return handoff ? local.adding([HandoffTool.tool]) : local
    }

    /// Whether every tool of `category` may run.
    static func isEnabled(_ category: Category, in settings: AssistantSettings) -> Bool {
        let disabled = Set(settings.disabledTools)
        return !category.toolNames.contains { disabled.contains($0) }
    }

    /// Turns the tools of `category` on or off. Other entries of `disabledTools` are kept; the list
    /// stays sorted and free of duplicates.
    static func setEnabled(_ enabled: Bool, _ category: Category, in settings: inout AssistantSettings) {
        var disabled = Set(settings.disabledTools)
        disabled.subtract(category.toolNames)
        if !enabled {
            disabled.formUnion(category.toolNames)
        }
        settings.disabledTools = disabled.sorted()
    }
}

/// Whether the app may use what a group of device tools needs.
enum DeviceToolAccess: Equatable, Sendable {
    /// Not checked yet.
    case unknown
    /// The user hasn't been asked yet; the first use asks.
    case notDetermined
    case granted
    /// Calendar only: events may be added but not read.
    case addOnly
    /// The user said no; only the Settings app can change that.
    case denied
    /// Blocked on this iPhone, e.g. by Screen Time.
    case restricted
}

/// Builds the JSON Schemas of the device tools. Every object has `additionalProperties: false` and
/// a `required` list, as strict Claude tools need; ranges are checked by the tools themselves.
enum DeviceToolSchema {
    static func object(_ properties: [String: JSONValue], required: [String]) -> JSONValue {
        .object([
            "type": .string("object"),
            "properties": .object(properties),
            "required": .array(required.map { JSONValue.string($0) }),
            "additionalProperties": .bool(false),
        ])
    }

    static func string(_ description: String) -> JSONValue {
        .object(["type": .string("string"), "description": .string(description)])
    }

    static func integer(_ description: String) -> JSONValue {
        .object(["type": .string("integer"), "description": .string(description)])
    }

    static func boolean(_ description: String) -> JSONValue {
        .object(["type": .string("boolean"), "description": .string(description)])
    }

    static func choice(_ values: [String], _ description: String) -> JSONValue {
        .object([
            "type": .string("string"),
            "description": .string(description),
            "enum": .array(values.map { JSONValue.string($0) }),
        ])
    }
}

/// Helpers the device tools share: reading input, and the short texts shown to the user.
enum DeviceToolSupport {
    /// Characters of a result the item lists may fill. `ToolRunner` cuts results at 2,000
    /// characters, which would leave broken JSON, so lists stop well before that.
    static let listBudget = 1_500

    /// A Gregorian calendar in `timeZone` that doesn't depend on the device's settings.
    static func calendar(_ timeZone: TimeZone) -> Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.locale = Locale(identifier: "en_US_POSIX")
        calendar.timeZone = timeZone
        return calendar
    }

    // MARK: Input

    /// The trimmed string at `key`, or nil when it is absent or blank.
    static func text(_ input: [String: JSONValue], _ key: String) -> String? {
        guard let value = input[key]?.stringValue?.trimmed, !value.isEmpty else { return nil }
        return value
    }

    /// The whole number at `key`; also reads one written as a string, as small models do.
    static func integer(_ input: [String: JSONValue], _ key: String) -> Int? {
        guard let value = input[key] else { return nil }
        if let number = value.intValue { return number }
        if let text = value.stringValue?.trimmed { return Int(text) }
        return nil
    }

    /// The boolean at `key`; also reads "true" / "false" written as a string.
    static func bool(_ input: [String: JSONValue], _ key: String) -> Bool? {
        guard let value = input[key] else { return nil }
        if let flag = value.boolValue { return flag }
        switch value.stringValue?.trimmed.lowercased() {
        case "true"?: return true
        case "false"?: return false
        default: return nil
        }
    }

    // MARK: Results

    /// The leading `items` that fit in `budget` characters of JSON, and how many were left out.
    static func fitting(_ items: [JSONValue], budget: Int = listBudget) -> (kept: [JSONValue], omitted: Int) {
        var kept: [JSONValue] = []
        var used = 0
        for item in items {
            let size = ((try? item.serialized()).map { String(decoding: $0, as: UTF8.self).count } ?? 0) + 1
            if used + size > budget { break }
            used += size
            kept.append(item)
        }
        return (kept, items.count - kept.count)
    }

    // MARK: Text for the user

    /// When `date` is, for a summary line: "today 17:00", "tomorrow", "Fri 15:00",
    /// "Fri 9 Oct 15:00" (in the order and clock style of the user's locale).
    static func whenLabel(_ date: Date, hasTime: Bool, context: ToolContext) -> String {
        let gregorian = Self.calendar(context.timeZone)
        let today = gregorian.startOfDay(for: context.now)
        let day = gregorian.startOfDay(for: date)
        let offset = gregorian.dateComponents([.day], from: today, to: day).day ?? 0
        let chinese = context.locale.identifier.hasPrefix("zh")

        let dayText: String
        switch offset {
        case 0:
            dayText = chinese ? "今天" : "today"
        case 1:
            dayText = chinese ? "明天" : "tomorrow"
        case -1:
            dayText = chinese ? "昨天" : "yesterday"
        case 2...6:
            dayText = format(date, template: "EEE", context: context)
        default:
            let sameYear = gregorian.component(.year, from: date) == gregorian.component(.year, from: context.now)
            dayText = format(date, template: sameYear ? "EEEdMMM" : "dMMMy", context: context)
        }
        guard hasTime else { return dayText }
        let time = format(date, template: "jmm", context: context)
        return chinese ? dayText + time : dayText + " " + time
    }

    /// A length of time for a summary line: "45 s", "10 min", "1 h 30 min".
    static func durationLabel(seconds: Int) -> String {
        let hours = seconds / 3_600
        let minutes = (seconds % 3_600) / 60
        let rest = seconds % 60
        var parts: [String] = []
        if hours > 0 { parts.append("\(hours) h") }
        if minutes > 0 { parts.append("\(minutes) min") }
        if rest > 0 { parts.append("\(rest) s") }
        return parts.isEmpty ? "0 s" : parts.joined(separator: " ")
    }

    /// "no events", "1 event", "3 events".
    static func countLabel(_ count: Int, _ singular: String, _ plural: String) -> String {
        switch count {
        case 0: return "no \(plural)"
        case 1: return "1 \(singular)"
        default: return "\(count) \(plural)"
        }
    }

    /// The app's name as the Settings app lists it.
    static var appName: String {
        let info = Bundle.main.infoDictionary
        return (info?["CFBundleDisplayName"] as? String) ?? (info?["CFBundleName"] as? String) ?? "Assistant"
    }

    /// Where the user turns on `item` for this app, e.g. "Settings › Apps › Assistant › Reminders".
    static func settingsPath(_ item: String) -> String {
        if #available(iOS 18, *) {
            return "Settings › Apps › \(appName) › \(item)"
        }
        return "Settings › \(appName) › \(item)"
    }

    private static func format(_ date: Date, template: String, context: ToolContext) -> String {
        let formatter = DateFormatter()
        formatter.locale = context.locale
        formatter.timeZone = context.timeZone
        formatter.setLocalizedDateFormatFromTemplate(template)
        return formatter.string(from: date)
    }
}
