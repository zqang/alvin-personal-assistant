import AlvinProfile
import XCTest
@testable import AssistantKit

final class SuccessClaimTests: XCTestCase {
    /// This app's claims: the shared tables plus its reminder, timer and event nouns.
    private let claims = AlvinProfile.profile.claims

    func testClaims() {
        for text in [
            // Measured on the phone: Woof answered two benchmark requests without a tool call.
            "已经为你设置好了明天的提醒。",
            "好的，会议已加入您的日程。",
            "好的，我为你设置了一个二十分钟的计时器。",
            "明天早上八点我会提醒你去取快递。",
            "好的，我会在明天早上八点提醒你去取快递。",
            "没问题，已帮你记下了",
            "我已将会议添加到您的日历中。",
            "计时器已启动，二十分钟。",
            "I've set a timer for ten minutes.",
            "I have just added it to your calendar.",
            "Okay, I'll remind you at 5.",
            "Reminder set for tomorrow at 8 a.m.",
            "Done! Anything else?",
            "You're all set.",
            "Added buy milk to your to-do list.",
            "Your dentist appointment is scheduled for Friday at 3 pm.",
            "I\u{2019}ve set it for 8, is that okay?",
        ] {
            XCTAssertTrue(claims.claimsDone(text), "\"\(text)\"")
        }
    }

    func testNonClaims() {
        for text in [
            "",
            "要我帮你设置提醒吗？",
            "需要我帮你设置好吗",
            "你想几点提醒？",
            "明天是星期四。",
            "I can set a reminder for you.",
            "Want me to add it to your calendar?",
            "Would you like me to set a timer?",
            "What time should I set it for?",
            "Sorry, I can't set alarms here.",
            "This set of reminders looks long.",
            "Ten minutes is 600 seconds.",
            "The capital of Peru is Lima.",
            SuccessClaim.actionFailedMessage(for: "Remind me at 5"),
            SuccessClaim.actionFailedMessage(for: "提醒我"),
        ] {
            XCTAssertFalse(claims.claimsDone(text), "\"\(text)\"")
        }
    }

    func testActionFailedMessageFollowsTheUserLanguage() {
        XCTAssertEqual(SuccessClaim.actionFailedMessage(for: "提醒我明天早上八点去取快递。"), "抱歉，这个操作我没能完成。你可以再说一次，或者换个说法。")
        XCTAssertEqual(SuccessClaim.actionFailedMessage(for: "Remind me at 5 to call mum."), "Sorry, I couldn't do that. Please try again or rephrase.")
        XCTAssertEqual(SuccessClaim.actionFailedMessage(for: "Add 牛奶 to my list"), "抱歉，这个操作我没能完成。你可以再说一次，或者换个说法。")
    }

    func testTablesAreLowercaseAndUnique() {
        for table in [claims.phrases, SuccessClaim.chinesePhrases, claims.allParticiples, SuccessClaim.promisedVerbs] {
            XCTAssertEqual(Set(table).count, table.count)
            for phrase in table {
                XCTAssertEqual(phrase, IntentClassifier.normalize(phrase))
            }
        }
    }
}
