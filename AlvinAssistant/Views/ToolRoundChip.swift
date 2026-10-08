import AssistantKit
import SwiftUI
import UIKit

/// What a reply did with tools: one line per call that has a summary, e.g.
/// "Reminder: Call mum · today 17:00", and a warning line for an action that didn't happen.
struct ToolRoundChip: View {
    let rounds: [ToolRound]

    var body: some View {
        let lines = Self.lines(for: rounds)
        if !lines.isEmpty {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(lines.indices, id: \.self) { index in
                    let line = lines[index]
                    Label(line.text, systemImage: line.symbol)
                        .foregroundStyle(line.failed ? Color.orange : Color.secondary)
                }
            }
            .font(.footnote)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .accessibilityElement(children: .combine)
        }
    }

    struct Line: Equatable {
        var text: String
        var symbol: String
        var failed: Bool
    }

    /// The lines for `rounds`, in call order. A call without a summary gets a line only when it
    /// was an action that failed; failed reads and the cloud handoff stay silent.
    static func lines(for rounds: [ToolRound]) -> [Line] {
        var lines: [Line] = []
        for round in rounds {
            for call in round.calls {
                if let summary = call.summary?.trimmed, !summary.isEmpty {
                    lines.append(Line(text: summary, symbol: call.isError ? "exclamationmark.triangle" : symbol(for: call.name), failed: call.isError))
                } else if call.isError, let failure = failureText[call.name] {
                    lines.append(Line(text: failure, symbol: "exclamationmark.triangle", failed: true))
                }
            }
        }
        return lines
    }

    private static let failureText: [String: String] = [
        CreateReminderTool.name: "Reminder not added",
        CompleteReminderTool.name: "Reminder not marked done",
        CreateEventTool.name: "Event not added",
        SetTimerTool.name: "Timer not started",
        CancelTimerTool.name: "Timer not cancelled",
    ]

    private static func symbol(for name: String) -> String {
        switch name {
        case CreateReminderTool.name, ListRemindersTool.name, CompleteReminderTool.name:
            return "checklist"
        case CreateEventTool.name, ListEventsTool.name:
            return "calendar"
        case SetTimerTool.name, ListTimersTool.name, CancelTimerTool.name:
            return "timer"
        case CurrentTimeTool.name:
            return "clock"
        default:
            return "wrench.and.screwdriver"
        }
    }
}
