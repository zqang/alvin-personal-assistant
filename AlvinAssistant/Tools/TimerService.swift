import Foundation
import UserNotifications

/// A running timer: a pending local notification whose identifier starts with `timer.`.
struct TimerInfo: Equatable, Sendable {
    /// The notification identifier, `timer.<uuid>`.
    var id: String
    var label: String?
    /// When it goes off.
    var ends: Date
    /// How long it was set for.
    var seconds: Int
}

/// Why a timer tool couldn't act. The descriptions are written for the model.
enum TimerServiceError: LocalizedError, Equatable {
    case notificationsOff
    case scheduleFailed(String)

    var errorDescription: String? {
        switch self {
        case .notificationsOff:
            return "Notifications are off for this app, so a timer couldn't alert the user. They can turn them on in \(DeviceToolSupport.settingsPath("Notifications"))."
        case .scheduleFailed(let message):
            return "Couldn't start the timer: \(message)"
        }
    }
}

/// Timers as local notifications: each one is a `UNTimeIntervalNotificationTrigger` with the
/// identifier `timer.<uuid>` and its end time in `userInfo["ends"]`. They go off with the app
/// closed; `TimerNotificationDelegate` shows them while it is open.
struct TimerService: Sendable {
    static let shared = TimerService()

    static let identifierPrefix = "timer."

    private var center: UNUserNotificationCenter { UNUserNotificationCenter.current() }

    // MARK: Access

    /// Whether timers can alert the user.
    func access() async -> DeviceToolAccess {
        let settings = await center.notificationSettings()
        switch settings.authorizationStatus {
        case .notDetermined:
            return .notDetermined
        case .denied:
            return .denied
        case .authorized, .provisional, .ephemeral:
            return .granted
        default:
            return .unknown
        }
    }

    /// Asks to show alerts with sound if the user hasn't been asked yet, and returns the outcome.
    func requestAccess() async -> DeviceToolAccess {
        if await access() == .notDetermined {
            _ = try? await center.requestAuthorization(options: [.alert, .sound])
        }
        return await access()
    }

    /// Returns when timers can alert the user, asking the first time; throws when they can't.
    func ensureAccess() async throws {
        switch await access() {
        case .granted:
            return
        case .notDetermined:
            let granted = (try? await center.requestAuthorization(options: [.alert, .sound])) ?? false
            guard granted else { throw TimerServiceError.notificationsOff }
        default:
            throw TimerServiceError.notificationsOff
        }
    }

    // MARK: Timers

    /// Starts a timer of `seconds` (at least 1).
    func start(seconds: Int, label: String?) async throws -> TimerInfo {
        try await ensureAccess()
        try Task.checkCancellation()

        let id = Self.identifierPrefix + UUID().uuidString
        let length = DeviceToolSupport.durationLabel(seconds: seconds)
        let content = UNMutableNotificationContent()
        content.title = label ?? "Timer"
        content.body = "Your \(length) timer is done."
        content.sound = UNNotificationSound.default
        let ends = Date().addingTimeInterval(TimeInterval(seconds))
        content.userInfo = [
            "ends": ends.timeIntervalSince1970,
            "seconds": seconds,
            "label": label ?? "",
        ]
        let trigger = UNTimeIntervalNotificationTrigger(timeInterval: TimeInterval(max(1, seconds)), repeats: false)
        let request = UNNotificationRequest(identifier: id, content: content, trigger: trigger)
        do {
            try await center.add(request)
        } catch {
            throw TimerServiceError.scheduleFailed(error.localizedDescription)
        }
        return TimerInfo(id: id, label: label, ends: ends, seconds: seconds)
    }

    /// The timers that haven't gone off yet, soonest first.
    func running() async -> [TimerInfo] {
        let requests = await center.pendingNotificationRequests()
        return requests
            .compactMap { Self.timer(from: $0) }
            .sorted { $0.ends < $1.ends }
    }

    /// Stops the timers with these identifiers.
    func cancel(ids: [String]) {
        center.removePendingNotificationRequests(withIdentifiers: ids)
    }

    /// The timer `request` schedules, or nil when it isn't one of ours.
    private static func timer(from request: UNNotificationRequest) -> TimerInfo? {
        guard request.identifier.hasPrefix(identifierPrefix) else { return nil }
        let info = request.content.userInfo
        let ends: Date
        if let seconds = info["ends"] as? Double {
            ends = Date(timeIntervalSince1970: seconds)
        } else if let next = (request.trigger as? UNTimeIntervalNotificationTrigger)?.nextTriggerDate() {
            ends = next
        } else {
            return nil
        }
        let label = (info["label"] as? String)?.trimmed
        let length = (info["seconds"] as? Int)
            ?? (request.trigger as? UNTimeIntervalNotificationTrigger).map { Int($0.timeInterval) }
            ?? 0
        return TimerInfo(id: request.identifier, label: (label?.isEmpty ?? true) ? nil : label, ends: ends, seconds: length)
    }
}
