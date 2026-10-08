import Foundation
import UserNotifications

/// Shows timer notifications while the app is open, with their sound; without a delegate, iOS
/// drops notifications that arrive in the foreground. Set it as the notification center's
/// delegate at launch (`AppSetup` does).
final class TimerNotificationDelegate: NSObject, UNUserNotificationCenterDelegate, @unchecked Sendable {
    static let shared = TimerNotificationDelegate()

    override private init() {
        super.init()
    }

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        completionHandler([.banner, .sound, .list])
    }
}
