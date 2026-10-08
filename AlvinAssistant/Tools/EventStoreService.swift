import AssistantKit
import EventKit
import Foundation

/// A reminder as the tools see it, copied out of EventKit.
struct ReminderItem: Equatable, Sendable {
    /// `calendarItemIdentifier`, which `complete_reminder` takes back.
    var id: String
    var title: String
    var due: Date?
    /// False when the reminder is due on a day without a time.
    var dueHasTime: Bool
    /// The reminders list it is in.
    var list: String
    var notes: String?

    /// Reads `reminder`; a due date without a time zone is read in `timeZone`.
    init(_ reminder: EKReminder, timeZone: TimeZone) {
        id = reminder.calendarItemIdentifier
        title = (reminder.title ?? "").trimmed
        list = reminder.calendar?.title ?? ""
        let notes = (reminder.notes ?? "").trimmed
        self.notes = notes.isEmpty ? nil : notes
        let due = Self.dueDate(reminder.dueDateComponents, timeZone: timeZone)
        self.due = due?.date
        dueHasTime = due?.hasTime ?? false
    }

    /// The moment `components` name: the start of the day when they have no time.
    static func dueDate(_ components: DateComponents?, timeZone: TimeZone) -> (date: Date, hasTime: Bool)? {
        guard let components, let year = components.year, let month = components.month, let day = components.day else {
            return nil
        }
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = components.timeZone ?? timeZone
        let hasTime = components.hour != nil
        let parts = DateComponents(year: year, month: month, day: day, hour: components.hour ?? 0, minute: components.minute ?? 0)
        guard let date = calendar.date(from: parts) else { return nil }
        return (date: date, hasTime: hasTime)
    }
}

/// A calendar event as the tools see it, copied out of EventKit.
struct EventItem: Equatable, Sendable {
    var title: String
    var start: Date
    var end: Date
    var isAllDay: Bool
    var location: String?
    /// The calendar it is in.
    var calendar: String

    init(_ event: EKEvent) {
        title = (event.title ?? "").trimmed
        let start: Date = event.startDate ?? Date()
        self.start = start
        end = event.endDate ?? start
        isAllDay = event.isAllDay
        let location = (event.location ?? "").trimmed
        self.location = location.isEmpty ? nil : location
        calendar = event.calendar?.title ?? ""
    }
}

/// What `create_reminder` asks for.
struct ReminderDraft: Sendable {
    var title: String
    var due: Date?
    var dueHasTime: Bool
    var notes: String?
    /// The name of a reminders list, as the user said it.
    var listName: String?
}

/// What `create_event` asks for. All-day events start and end at the start of their first and
/// last day.
struct EventDraft: Sendable {
    var title: String
    var start: Date
    var end: Date
    var isAllDay: Bool
    var location: String?
    var notes: String?
}

/// Why a reminders or calendar tool couldn't act. The descriptions are written for the model, which
/// passes them on to the user.
enum EventStoreError: LocalizedError, Equatable {
    case denied(item: String)
    case restricted(item: String)
    case addOnly
    case accessFailed(String)
    case noReminderList
    case noCalendar
    case reminderNotFound(String)
    case saveFailed(String)

    var errorDescription: String? {
        switch self {
        case .denied(let item):
            return "The user hasn't allowed this app to use \(item). They can turn it on in \(DeviceToolSupport.settingsPath(item))."
        case .restricted(let item):
            return "Access to \(item) is restricted on this iPhone (for example by Screen Time), so this can't be done."
        case .addOnly:
            return "The app may add calendar events but not read them. The user can allow full access in \(DeviceToolSupport.settingsPath("Calendars"))."
        case .accessFailed(let message):
            return "Couldn't get permission: \(message)"
        case .noReminderList:
            return "There's no reminders list to add to; the user needs to set up the Reminders app first."
        case .noCalendar:
            return "There's no calendar to add events to; the user needs to set up the Calendar app first."
        case .reminderNotFound(let id):
            return "There's no open reminder with id \"\(id)\". Call list_reminders to get the current ids."
        case .saveFailed(let message):
            return "Couldn't save the change: \(message)"
        }
    }
}

/// The app's one `EKEventStore`, used by the reminders and calendar tools.
///
/// Every call first makes sure the user allowed access, asking the first time. The reads take the
/// turn's commit gate and ask only once it opens (see `ensureAccess`). Results are copied into
/// value types, so no EventKit object leaves the actor.
actor EventStoreService {
    static let shared = EventStoreService()

    private let store = EKEventStore()

    // MARK: Access

    /// Whether the app may read and write `type` (`.reminder` or `.event`).
    static func access(for type: EKEntityType) -> DeviceToolAccess {
        switch EKEventStore.authorizationStatus(for: type) {
        case .notDetermined:
            return .notDetermined
        case .restricted:
            return .restricted
        case .denied:
            return .denied
        case .fullAccess:
            return .granted
        case .writeOnly:
            return .addOnly
        default:
            return .unknown
        }
    }

    /// Asks for full access to `type` if the user hasn't been asked yet, and returns the outcome.
    func requestAccess(_ type: EKEntityType) async -> DeviceToolAccess {
        if Self.access(for: type) == .notDetermined {
            _ = try? await requestFullAccess(type)
        }
        return Self.access(for: type)
    }

    /// Returns when the app may use `type`, asking the user the first time; throws when it may not.
    /// With `addOnly`, write-only calendar access is enough.
    ///
    /// When the user hasn't been asked yet, it first waits for `gate` to open, and throws
    /// `CancellationError` if the gate is cancelled. Read-only tools aren't held by the commit gate,
    /// so they may run for an early reply that is later thrown away; a permission alert can't be
    /// taken back once shown, and a "Don't Allow" lasts until the user changes it in Settings.
    /// Side-effect tools pass no gate: the tool runner already waited on it.
    func ensureAccess(_ type: EKEntityType, addOnly: Bool = false, askAfter gate: CommitGate? = nil) async throws {
        if let gate, Self.access(for: type) == .notDetermined {
            try await gate.wait()
            // An open gate lets a cancelled task through; don't ask for an abandoned turn.
            try Task.checkCancellation()
        }
        let item = type == .reminder ? "Reminders" : "Calendars"
        switch Self.access(for: type) {
        case .granted:
            return
        case .addOnly where addOnly:
            return
        case .addOnly:
            throw EventStoreError.addOnly
        case .notDetermined:
            let granted: Bool
            do {
                granted = try await requestFullAccess(type)
            } catch {
                throw EventStoreError.accessFailed(error.localizedDescription)
            }
            guard granted else { throw EventStoreError.denied(item: item) }
        case .restricted:
            throw EventStoreError.restricted(item: item)
        case .denied, .unknown:
            throw EventStoreError.denied(item: item)
        }
    }

    private func requestFullAccess(_ type: EKEntityType) async throws -> Bool {
        if type == .reminder {
            return try await store.requestFullAccessToReminders()
        }
        return try await store.requestFullAccessToEvents()
    }

    // MARK: Reminders

    /// Every reminder that isn't completed, in no particular order. A first-time permission request
    /// waits for `gate` (see `ensureAccess`).
    func openReminders(timeZone: TimeZone, askAfter gate: CommitGate? = nil) async throws -> [ReminderItem] {
        try await ensureAccess(.reminder, askAfter: gate)
        let store = self.store
        let predicate = store.predicateForIncompleteReminders(withDueDateStarting: nil, ending: nil, calendars: nil)
        return await withCheckedContinuation { (continuation: CheckedContinuation<[ReminderItem], Never>) in
            _ = store.fetchReminders(matching: predicate) { reminders in
                // EventKit objects stay in this callback; only value copies leave it.
                let items = (reminders ?? []).map { ReminderItem($0, timeZone: timeZone) }
                continuation.resume(returning: items)
            }
        }
    }

    /// Adds a reminder. A due time also gets an alarm at that time. A due day without a time gets
    /// no alarm (see below). Returns the reminder, and whether the list the user named was found
    /// (otherwise it went to the default list).
    func addReminder(_ draft: ReminderDraft, timeZone: TimeZone) async throws -> (item: ReminderItem, listFound: Bool) {
        try await ensureAccess(.reminder)
        try Task.checkCancellation()

        let lists = store.calendars(for: .reminder).filter(\.allowsContentModifications)
        var named: EKCalendar?
        if let name = draft.listName?.trimmed.lowercased(), !name.isEmpty {
            named = lists.first { $0.title.trimmed.lowercased() == name }
        }
        guard let list = named ?? store.defaultCalendarForNewReminders() ?? lists.first else {
            throw EventStoreError.noReminderList
        }

        let reminder = EKReminder(eventStore: store)
        reminder.title = draft.title
        reminder.calendar = list
        if let notes = draft.notes {
            reminder.notes = notes
        }
        if let due = draft.due {
            var calendar = Calendar(identifier: .gregorian)
            calendar.timeZone = timeZone
            if draft.dueHasTime {
                var components = calendar.dateComponents([.year, .month, .day, .hour, .minute], from: due)
                components.timeZone = timeZone
                reminder.dueDateComponents = components
                reminder.addAlarm(EKAlarm(absoluteDate: due))
            } else {
                // No alarm on purpose: an absolute alarm at the stored start of the day would go off
                // at 00:00. Saved like this, the reminder is the same as one made in the Reminders
                // app without a time, which alerts at the user's "Today Notification" time.
                reminder.dueDateComponents = calendar.dateComponents([.year, .month, .day], from: due)
            }
        }

        do {
            try store.save(reminder, commit: true)
        } catch {
            throw EventStoreError.saveFailed(error.localizedDescription)
        }
        let item = ReminderItem(reminder, timeZone: timeZone)
        return (item: item, listFound: named != nil || draft.listName == nil)
    }

    /// Marks a reminder as done. `id` is normally a `calendarItemIdentifier`; the exact title of
    /// exactly one open reminder is accepted too. Returns the reminder, and whether it was still open.
    func completeReminder(id: String, timeZone: TimeZone) async throws -> (item: ReminderItem, wasOpen: Bool) {
        try await ensureAccess(.reminder)

        var found = store.calendarItem(withIdentifier: id) as? EKReminder
        if found == nil {
            let wanted = id.trimmed.lowercased()
            let matches = try await openReminders(timeZone: timeZone).filter { $0.title.lowercased() == wanted }
            if matches.count == 1 {
                found = store.calendarItem(withIdentifier: matches[0].id) as? EKReminder
            }
        }
        guard let reminder = found else {
            throw EventStoreError.reminderNotFound(id)
        }
        guard !reminder.isCompleted else {
            return (item: ReminderItem(reminder, timeZone: timeZone), wasOpen: false)
        }

        try Task.checkCancellation()
        reminder.isCompleted = true
        do {
            try store.save(reminder, commit: true)
        } catch {
            throw EventStoreError.saveFailed(error.localizedDescription)
        }
        return (item: ReminderItem(reminder, timeZone: timeZone), wasOpen: true)
    }

    // MARK: Events

    /// The events of all calendars that overlap `start..<end`, by start time. A first-time
    /// permission request waits for `gate` (see `ensureAccess`).
    func events(from start: Date, to end: Date, askAfter gate: CommitGate? = nil) async throws -> [EventItem] {
        try await ensureAccess(.event, askAfter: gate)
        let predicate = store.predicateForEvents(withStart: start, end: end, calendars: nil)
        return store.events(matching: predicate)
            .map { EventItem($0) }
            .sorted { ($0.start, $0.title) < ($1.start, $1.title) }
    }

    /// Adds an event to the default calendar, for this occurrence only and without attendees.
    func addEvent(_ draft: EventDraft) async throws -> EventItem {
        try await ensureAccess(.event, addOnly: true)
        try Task.checkCancellation()

        guard let calendar = store.defaultCalendarForNewEvents else {
            throw EventStoreError.noCalendar
        }
        let event = EKEvent(eventStore: store)
        event.title = draft.title
        event.isAllDay = draft.isAllDay
        event.startDate = draft.start
        event.endDate = draft.end
        event.calendar = calendar
        if let location = draft.location {
            event.location = location
        }
        if let notes = draft.notes {
            event.notes = notes
        }

        do {
            try store.save(event, span: .thisEvent, commit: true)
        } catch {
            throw EventStoreError.saveFailed(error.localizedDescription)
        }
        return EventItem(event)
    }
}
