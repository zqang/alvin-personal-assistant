import AssistantKit
import Foundation

/// `create_reminder`: adds a reminder to the Reminders app.
struct CreateReminderTool: AssistantTool {
    static let name = "create_reminder"

    static let definition = ToolDefinition(
        name: name,
        description: "Add a reminder to the user's Reminders app. Call this whenever the user asks to be reminded of something or to add a to-do, even without a time. Times are the user's local time.",
        inputSchema: DeviceToolSchema.object(
            [
                "title": DeviceToolSchema.string("What to be reminded of, in the user's words, e.g. \"Call mum\"."),
                "due": DeviceToolSchema.string("When it is due, as a local time in ISO 8601 without an offset, e.g. 2026-10-07T17:00. Leave it out when the user gave no time."),
                "notes": DeviceToolSchema.string("Extra details, only if the user gave any."),
                "list": DeviceToolSchema.string("The name of a reminders list, only if the user named one."),
            ],
            required: ["title"]
        )
    )

    var definition: ToolDefinition { Self.definition }
    var effect: ToolEffect { .sideEffect }
    var presentation: ToolPresentation { ToolPresentation(activity: "Adding a reminder", cue: nil) }

    func run(_ input: [String: JSONValue], context: ToolContext) async throws -> ToolOutput {
        guard let title = DeviceToolSupport.text(input, "title") else {
            return .error("The title is empty; say what to remind the user of.")
        }
        var due: LocalDateTime.Parsed?
        if let text = DeviceToolSupport.text(input, "due") {
            guard let parsed = LocalDateTime.parse(text, timeZone: context.timeZone) else {
                return .error("Couldn't read the due time \"\(text)\". Use a local time like 2026-10-07T17:00, or leave it out.")
            }
            due = parsed
        }
        let draft = ReminderDraft(
            title: title,
            due: due?.date,
            dueHasTime: due?.hasTime ?? false,
            notes: DeviceToolSupport.text(input, "notes"),
            listName: DeviceToolSupport.text(input, "list")
        )

        let added = try await EventStoreService.shared.addReminder(draft, timeZone: context.timeZone)
        let reminder = added.item

        var content: [String: JSONValue] = [
            "status": .string("added"),
            "id": .string(reminder.id),
            "title": .string(reminder.title),
            "list": .string(reminder.list),
        ]
        if let date = reminder.due {
            content["due"] = .string(LocalDateTime.format(date, timeZone: context.timeZone, includeTime: reminder.dueHasTime))
        }
        var notes: [String] = []
        if !added.listFound, let name = draft.listName {
            notes.append("There's no list named \"\(name)\", so it went to \"\(reminder.list)\".")
        }
        if let due, Self.isPast(due, context: context) {
            notes.append("The due time has already passed.")
        }
        if !notes.isEmpty {
            content["note"] = .string(notes.joined(separator: " "))
        }

        var summary = "Reminder: \(title)"
        if let due {
            summary += " · " + DeviceToolSupport.whenLabel(due.date, hasTime: due.hasTime, context: context)
        }
        return .ok(.object(content), summary: summary)
    }

    /// Whether `due` is before now (a day alone: before today).
    static func isPast(_ due: LocalDateTime.Parsed, context: ToolContext) -> Bool {
        if due.hasTime { return due.date < context.now }
        return due.date < DeviceToolSupport.calendar(context.timeZone).startOfDay(for: context.now)
    }
}

/// `list_reminders`: reads the user's open reminders.
struct ListRemindersTool: AssistantTool {
    static let name = "list_reminders"

    enum Scope: String, CaseIterable {
        case today, upcoming, overdue, all
    }

    static let definition = ToolDefinition(
        name: name,
        description: "Read the user's open reminders. Call this whenever the user asks what is on their to-do list or which reminders they have.",
        inputSchema: DeviceToolSchema.object(
            [
                "scope": DeviceToolSchema.choice(
                    Scope.allCases.map(\.rawValue),
                    "Which reminders to read: due today, upcoming, overdue, or all open ones."
                ),
            ],
            required: ["scope"]
        )
    )

    var definition: ToolDefinition { Self.definition }
    var effect: ToolEffect { .readOnly }
    var presentation: ToolPresentation { ToolPresentation(activity: "Checking your reminders", cue: .checking) }

    func run(_ input: [String: JSONValue], context: ToolContext) async throws -> ToolOutput {
        let scope = Scope(rawValue: (input["scope"]?.stringValue ?? "").trimmed.lowercased()) ?? .all
        // Read-only, so not held by the commit gate; a first-time permission alert still waits for it.
        let reminders = try await EventStoreService.shared.openReminders(timeZone: context.timeZone, askAfter: context.commitGate)
        let selected = Self.select(reminders, scope: scope, context: context)

        let items = selected.map { reminder -> JSONValue in
            var item: [String: JSONValue] = [
                "id": .string(reminder.id),
                "title": .string(reminder.title),
                "list": .string(reminder.list),
            ]
            if let date = reminder.due {
                item["due"] = .string(LocalDateTime.format(date, timeZone: context.timeZone, includeTime: reminder.dueHasTime))
            }
            if let notes = reminder.notes {
                item["notes"] = .string(notes.count > 100 ? String(notes.prefix(100)) + "…" : notes)
            }
            return .object(item)
        }
        let fitted = DeviceToolSupport.fitting(items)

        var content: [String: JSONValue] = [
            "scope": .string(scope.rawValue),
            "count": .int(selected.count),
            "reminders": .array(fitted.kept),
        ]
        if fitted.omitted > 0 {
            content["more"] = .int(fitted.omitted)
        }
        return .ok(.object(content), summary: Self.summary(count: selected.count, scope: scope))
    }

    /// The reminders in `scope`, soonest first; reminders without a due date come last, by title.
    static func select(_ reminders: [ReminderItem], scope: Scope, context: ToolContext) -> [ReminderItem] {
        let calendar = DeviceToolSupport.calendar(context.timeZone)
        let today = calendar.startOfDay(for: context.now)
        let tomorrow = calendar.date(byAdding: .day, value: 1, to: today) ?? today.addingTimeInterval(86_400)
        let now = context.now

        let chosen = reminders.filter { reminder in
            guard scope != .all else { return true }
            guard let due = reminder.due else { return false }
            switch scope {
            case .today:
                return due >= today && due < tomorrow
            case .overdue:
                return reminder.dueHasTime ? due < now : due < today
            case .upcoming:
                return reminder.dueHasTime ? due >= now : due >= today
            case .all:
                return true
            }
        }
        return chosen.sorted { first, second in
            switch (first.due, second.due) {
            case let (a?, b?) where a != b:
                return a < b
            case (.some, .none):
                return true
            case (.none, .some):
                return false
            default:
                return first.title.localizedCaseInsensitiveCompare(second.title) == .orderedAscending
            }
        }
    }

    /// "Reminders: 3 due today", "Reminders: none overdue".
    static func summary(count: Int, scope: Scope) -> String {
        let amount = count == 0 ? "none" : String(count)
        switch scope {
        case .today: return "Reminders: \(amount) due today"
        case .upcoming: return "Reminders: \(amount) upcoming"
        case .overdue: return "Reminders: \(amount) overdue"
        case .all: return "Reminders: \(amount) open"
        }
    }
}

/// `complete_reminder`: marks one of the user's reminders as done.
struct CompleteReminderTool: AssistantTool {
    static let name = "complete_reminder"

    static let definition = ToolDefinition(
        name: name,
        description: "Mark one of the user's reminders as done. Call this when the user says they finished something on their list. Get the id from list_reminders first.",
        inputSchema: DeviceToolSchema.object(
            ["id": DeviceToolSchema.string("The reminder's id, from list_reminders.")],
            required: ["id"]
        )
    )

    var definition: ToolDefinition { Self.definition }
    var effect: ToolEffect { .sideEffect }
    var presentation: ToolPresentation { ToolPresentation(activity: "Completing a reminder", cue: nil) }

    func run(_ input: [String: JSONValue], context: ToolContext) async throws -> ToolOutput {
        guard let id = DeviceToolSupport.text(input, "id") else {
            return .error("The id is empty. Call list_reminders to get the reminder's id.")
        }
        let result = try await EventStoreService.shared.completeReminder(id: id, timeZone: context.timeZone)
        let content: [String: JSONValue] = [
            "status": .string(result.wasOpen ? "completed" : "already completed"),
            "id": .string(result.item.id),
            "title": .string(result.item.title),
        ]
        return .ok(.object(content), summary: "Done: \(result.item.title)")
    }
}
