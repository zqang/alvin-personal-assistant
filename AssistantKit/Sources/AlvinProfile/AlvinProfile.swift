import AssistantKit
import Foundation

/// This app's domain: a personal assistant on the iPhone whose tools are reminders, the calendar,
/// timers and the clock (`AlvinAssistant/Tools`). The app selects it once
/// (`AssistantProfile.app` in `AppSetup.swift`); another app writes its own profile in its own
/// project and never links this module.
///
/// Every string and table here is what the packages hard-coded before they took a profile, byte
/// for byte (`ProfileGoldenTests`): Claude's prompt cache, the prewarmed on-device prefix and the
/// phone's 12/12 tool benchmark all depend on these exact bytes.
public enum AlvinProfile {
    public static let profile = AssistantProfile(
        persona: { name in
            "You are a warm, capable personal assistant\(name.isEmpty ? "" : " for \(name)"), running as an iPhone app."
        },
        localToolHint: "You run on the user's iPhone. Use your tools for reminders, calendar, timers and the current time.",
        localHandoffHint: "If a request needs the internet (news, weather, prices, scores) or deep expertise, call handoff_to_cloud before saying anything.",
        handoffDescription: handoffDescription,
        intents: IntentClassifier(actionPhrases: actionPhrases, actionVerbs: ["schedule"]),
        claims: SuccessClaim(
            nouns: ["reminder", "timer", "alarm", "event", "meeting"],
            verbs: ["remind"],
            phrases: ["会提醒你", "会提醒您", "到时提醒你", "到时提醒您"]
        ),
        // Woof looked up the time to place "next Monday", then said the meeting was in the
        // calendar: after the time, an action is still due.
        contextLookups: ["get_current_time"],
        callGuard: TimerCheck.guard
    )

    /// Words that ask for a reminder, an event or a timer, or about them.
    static let actionPhrases = [
        "remind me", "reminder", "to-do", "todo", "calendar", "schedule", "meeting", "appointment",
        "what's on", "am i free", "timer", "alarm", "wake me",
        "提醒", "日程", "日历", "会议", "开会", "安排", "待办", "计时", "闹钟", "定时",
    ]

    /// More explicit than the drafter lab's first description: with that, asked how long a timer
    /// had left while it could only start timers, the on-device model started a one-second timer
    /// instead of handing off.
    static let handoffDescription = "Hand this request to the more capable cloud assistant, which sees the whole conversation. Call this first, instead of answering or calling another tool, when the request needs the internet (news, weather, prices, scores, opening hours), asks about something none of your other tools can read or do (for example cancelling a timer, when your tools can only start and list timers), or needs long, careful reasoning. Never call another tool as a stand-in for a request it doesn't fit."
}
