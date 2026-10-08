import AssistantKit
import Foundation

/// `list_events`: reads the user's calendars between two local times.
struct ListEventsTool: AssistantTool {
    static let name = "list_events"

    /// The longest range one call reads.
    static let maxDays = 31

    static let definition = ToolDefinition(
        name: name,
        description: "Read events from the user's calendars between two local times. Call this for any question about their schedule, meetings or free time.",
        inputSchema: DeviceToolSchema.object(
            [
                "start": DeviceToolSchema.string("Start of the range, as a local time in ISO 8601 without an offset, e.g. 2026-10-08T00:00."),
                "end": DeviceToolSchema.string("End of the range, in the same format, e.g. 2026-10-09T00:00."),
            ],
            required: ["start", "end"]
        )
    )

    var definition: ToolDefinition { Self.definition }
    var effect: ToolEffect { .readOnly }
    var presentation: ToolPresentation { ToolPresentation(activity: "Checking your calendar", cue: .checking) }

    func run(_ input: [String: JSONValue], context: ToolContext) async throws -> ToolOutput {
        let timeZone = context.timeZone
        let calendar = DeviceToolSupport.calendar(timeZone)
        guard let startText = DeviceToolSupport.text(input, "start"), let endText = DeviceToolSupport.text(input, "end") else {
            return .error("Give both start and end, e.g. 2026-10-08T00:00 and 2026-10-09T00:00.")
        }
        guard let start = LocalDateTime.parse(startText, timeZone: timeZone) else {
            return CalendarToolText.unreadable("start", startText)
        }
        guard let end = LocalDateTime.parse(endText, timeZone: timeZone) else {
            return CalendarToolText.unreadable("end", endText)
        }

        // A date alone as the end includes that whole day.
        let from = start.date
        var to = end.hasTime ? end.date : CalendarToolText.dayAfter(end.date, calendar: calendar)
        guard to > from else {
            return .error("The end must be after the start.")
        }
        var clamped = false
        if let limit = calendar.date(byAdding: .day, value: Self.maxDays, to: from), to > limit {
            to = limit
            clamped = true
        }

        // Read-only, so not held by the commit gate; a first-time permission alert still waits for it.
        let events = try await EventStoreService.shared.events(from: from, to: to, askAfter: context.commitGate)
        let items = events.map { event -> JSONValue in
            let times = CalendarToolText.times(of: event, timeZone: timeZone)
            var item: [String: JSONValue] = [
                "title": .string(event.title),
                "start": .string(times.start),
                "end": .string(times.end),
                "calendar": .string(event.calendar),
            ]
            if event.isAllDay {
                item["all_day"] = .bool(true)
            }
            if let location = event.location {
                item["location"] = .string(location.count > 80 ? String(location.prefix(80)) + "…" : location)
            }
            return .object(item)
        }
        let fitted = DeviceToolSupport.fitting(items)

        var content: [String: JSONValue] = [
            "from": .string(LocalDateTime.format(from, timeZone: timeZone)),
            "to": .string(LocalDateTime.format(to, timeZone: timeZone)),
            "count": .int(events.count),
            "events": .array(fitted.kept),
        ]
        if fitted.omitted > 0 {
            content["more"] = .int(fitted.omitted)
        }
        if clamped {
            content["note"] = .string("Only the first \(Self.maxDays) days were read; ask again for later dates.")
        }

        let range = CalendarToolText.rangeLabel(from: from, to: to, calendar: calendar, context: context)
        let summary = "Calendar: " + DeviceToolSupport.countLabel(events.count, "event", "events") + " · " + range
        return .ok(.object(content), summary: summary)
    }
}

/// `create_event`: adds an event to the user's default calendar.
struct CreateEventTool: AssistantTool {
    static let name = "create_event"

    /// The longest event with a time, in minutes (two weeks); longer ones are all-day events.
    static let maxMinutes = 20_160

    static let definition = ToolDefinition(
        name: name,
        description: "Add an event to the user's calendar. Call this whenever the user asks to schedule, book or put something in their calendar. Times are the user's local time.",
        inputSchema: DeviceToolSchema.object(
            [
                "title": DeviceToolSchema.string("A short title, e.g. \"Dentist\"."),
                "start": DeviceToolSchema.string("When it starts, as a local time in ISO 8601 without an offset, e.g. 2026-10-09T15:00."),
                "end": DeviceToolSchema.string("When it ends, in the same format. Leave it out to use duration_minutes or the default of one hour."),
                "duration_minutes": DeviceToolSchema.integer("How long it lasts in minutes, if the user said and gave no end time."),
                "location": DeviceToolSchema.string("Where it is, only if the user said."),
                "notes": DeviceToolSchema.string("Extra details, only if the user gave any."),
                "all_day": DeviceToolSchema.boolean("True for an all-day event."),
            ],
            required: ["title", "start"]
        )
    )

    var definition: ToolDefinition { Self.definition }
    var effect: ToolEffect { .sideEffect }
    var presentation: ToolPresentation { ToolPresentation(activity: "Adding to your calendar", cue: nil) }

    func run(_ input: [String: JSONValue], context: ToolContext) async throws -> ToolOutput {
        let timeZone = context.timeZone
        let calendar = DeviceToolSupport.calendar(timeZone)
        guard let title = DeviceToolSupport.text(input, "title") else {
            return .error("The title is empty; give the event a short title.")
        }
        guard let startText = DeviceToolSupport.text(input, "start") else {
            return .error("Give a start time, e.g. 2026-10-09T15:00.")
        }
        guard let start = LocalDateTime.parse(startText, timeZone: timeZone) else {
            return CalendarToolText.unreadable("start", startText)
        }
        var end: LocalDateTime.Parsed?
        if let endText = DeviceToolSupport.text(input, "end") {
            guard let parsed = LocalDateTime.parse(endText, timeZone: timeZone) else {
                return CalendarToolText.unreadable("end", endText)
            }
            end = parsed
        }
        let minutes = DeviceToolSupport.integer(input, "duration_minutes")
        if let minutes, !(1...Self.maxMinutes).contains(minutes) {
            return .error("duration_minutes must be between 1 and \(Self.maxMinutes) (two weeks); use all_day for longer events.")
        }
        // A date without a time means an all-day event unless the model said otherwise.
        let isAllDay = DeviceToolSupport.bool(input, "all_day") ?? !start.hasTime

        let eventStart: Date
        let eventEnd: Date
        if isAllDay {
            let firstDay = calendar.startOfDay(for: start.date)
            var lastDay = firstDay
            if let end {
                var day = calendar.startOfDay(for: end.date)
                // An end at midnight closes the day before it: "until 2026-10-12T00:00".
                if end.hasTime, end.date == day, day > firstDay {
                    day = calendar.date(byAdding: .day, value: -1, to: day) ?? day
                }
                lastDay = max(firstDay, day)
            } else if let minutes {
                let days = max(1, (minutes + 1_439) / 1_440)
                lastDay = calendar.date(byAdding: .day, value: days - 1, to: firstDay) ?? firstDay
            }
            guard let limit = calendar.date(byAdding: .day, value: 366, to: firstDay), lastDay <= limit else {
                return .error("An all-day event can last at most a year.")
            }
            eventStart = firstDay
            eventEnd = lastDay
        } else {
            eventStart = start.date
            if let end {
                let endDate = end.hasTime ? end.date : CalendarToolText.dayAfter(end.date, calendar: calendar)
                guard endDate > eventStart else {
                    return .error("The end must be after the start.")
                }
                eventEnd = endDate
            } else {
                eventEnd = eventStart.addingTimeInterval(TimeInterval((minutes ?? 60) * 60))
            }
            guard eventEnd.timeIntervalSince(eventStart) <= TimeInterval(Self.maxMinutes * 60) else {
                return .error("An event with a time can last at most two weeks; use all_day for longer events.")
            }
        }

        let draft = EventDraft(
            title: title,
            start: eventStart,
            end: eventEnd,
            isAllDay: isAllDay,
            location: DeviceToolSupport.text(input, "location"),
            notes: DeviceToolSupport.text(input, "notes")
        )
        let event = try await EventStoreService.shared.addEvent(draft)
        let times = CalendarToolText.times(of: event, timeZone: timeZone)

        var content: [String: JSONValue] = [
            "status": .string("added"),
            "title": .string(event.title),
            "start": .string(times.start),
            "end": .string(times.end),
            "all_day": .bool(event.isAllDay),
            "calendar": .string(event.calendar),
        ]
        if let location = draft.location {
            content["location"] = .string(location)
        }
        let started = isAllDay ? eventStart < calendar.startOfDay(for: context.now) : eventStart < context.now
        if started {
            content["note"] = .string("The start time has already passed.")
        }

        let summary = "Event: \(title) · " + DeviceToolSupport.whenLabel(eventStart, hasTime: !isAllDay, context: context)
        return .ok(.object(content), summary: summary)
    }
}

/// Text the calendar tools share.
enum CalendarToolText {
    /// The error for a time the model wrote in a form `LocalDateTime` can't read.
    static func unreadable(_ field: String, _ text: String) -> ToolOutput {
        .error("Couldn't read \(field) \"\(text)\". Use a local time like 2026-10-09T15:00, or a date like 2026-10-09.")
    }

    /// The start of the day after the day that starts at `day`.
    static func dayAfter(_ day: Date, calendar: Calendar) -> Date {
        calendar.date(byAdding: .day, value: 1, to: calendar.startOfDay(for: day)) ?? day.addingTimeInterval(86_400)
    }

    /// An event's start and end as local times; all-day events give their first and last day.
    static func times(of event: EventItem, timeZone: TimeZone) -> (start: String, end: String) {
        guard event.isAllDay else {
            return (
                start: LocalDateTime.format(event.start, timeZone: timeZone),
                end: LocalDateTime.format(event.end, timeZone: timeZone)
            )
        }
        // EventKit ends an all-day event at the end of its last day (or the next midnight); one
        // second earlier is always on the last day.
        let lastMoment = max(event.start, event.end.addingTimeInterval(-1))
        return (
            start: LocalDateTime.format(event.start, timeZone: timeZone, includeTime: false),
            end: LocalDateTime.format(lastMoment, timeZone: timeZone, includeTime: false)
        )
    }

    /// The range a `list_events` call read, for its summary: "tomorrow", "today – Fri",
    /// "today 14:00 – today 18:00".
    static func rangeLabel(from: Date, to: Date, calendar: Calendar, context: ToolContext) -> String {
        let firstDay = calendar.startOfDay(for: from)
        let wholeDays = from == firstDay && to == calendar.startOfDay(for: to)
        guard wholeDays else {
            return DeviceToolSupport.whenLabel(from, hasTime: true, context: context)
                + " – " + DeviceToolSupport.whenLabel(to, hasTime: true, context: context)
        }
        if dayAfter(firstDay, calendar: calendar) == to {
            return DeviceToolSupport.whenLabel(from, hasTime: false, context: context)
        }
        return DeviceToolSupport.whenLabel(from, hasTime: false, context: context)
            + " – " + DeviceToolSupport.whenLabel(to.addingTimeInterval(-1), hasTime: false, context: context)
    }
}
