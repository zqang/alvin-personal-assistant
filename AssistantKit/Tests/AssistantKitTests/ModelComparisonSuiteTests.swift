import XCTest
@testable import AssistantKit

final class ModelComparisonSuiteTests: XCTestCase {
    private func prompt(_ id: String) throws -> ModelComparisonSuite.Prompt {
        try XCTUnwrap(ModelComparisonSuite.prompts.first { $0.id == id })
    }

    func testSuiteIsBalancedAndAllReferenceAnswersPass() {
        let prompts = ModelComparisonSuite.prompts
        XCTAssertEqual(prompts.count, 24)
        XCTAssertEqual(Set(prompts.map(\.id)).count, prompts.count)
        XCTAssertEqual(Dictionary(grouping: prompts, by: \.category).values.map(\.count).sorted(), [4, 4, 4, 4, 4, 4])
        XCTAssertEqual(prompts.filter { $0.language == "en" }.count, 12)
        XCTAssertEqual(prompts.filter { $0.language == "zh" }.count, 12)
        for item in prompts {
            XCTAssertTrue(ModelComparisonSuite.evaluate(item, reply: item.expected).passed, item.id)
            XCTAssertTrue(ModelComparisonSuite.evaluate(item, reply: " \n\(item.expected)\n ").passed, item.id)
            XCTAssertFalse(ModelComparisonSuite.evaluate(item, reply: "").passed, item.id)
        }
    }

    func testExactAnswerRejectsCorrectSubstringInsideWrongOrVerboseReply() throws {
        let arithmetic = try prompt("arithmetic-en-1")
        for answer in ["143", "43 pencils", "The answer is 43.", "43\n44", "43.", "\"43\"", "<think>3 * 12 + 7</think>43"] {
            XCTAssertFalse(ModelComparisonSuite.evaluate(arithmetic, reply: answer).passed, answer)
        }
        let instruction = try prompt("instructions-zh-2")
        for answer in ["stop", "Stop", "STOP.", "GO STOP"] {
            XCTAssertFalse(ModelComparisonSuite.evaluate(instruction, reply: answer).passed, answer)
        }
    }

    func testChineseAnswerMustMatchEntireRequiredOutput() throws {
        let item = try prompt("reading-zh-1")
        XCTAssertTrue(ModelComparisonSuite.evaluate(item, reply: "未知").passed)
        for answer in ["未知。", "“未知”", "不知道", "13", "未知，因为资料没有给出年龄"] {
            XCTAssertFalse(ModelComparisonSuite.evaluate(item, reply: answer).passed, answer)
        }
    }

    func testJSONObjectAllowsKeyOrderAndWhitespaceButRejectsWrongTypesAndExtraKeys() throws {
        let item = try prompt("json-en-1")
        XCTAssertTrue(ModelComparisonSuite.evaluate(item, reply: "{\n \"repeats\": false, \"title\": \"Call Mei\", \"at\": \"2026-10-14T09:15\"\n}").passed)
        for answer in [
            #"{"title":"Call Mei","at":"2026-10-14T09:15","repeats":0}"#,
            #"{"title":"Call Mei","at":"2026-10-14T09:15","repeats":"false"}"#,
            #"{"title":"Call Mei","at":"2026-10-14T09:15","repeats":true}"#,
            #"{"title":"Call Mei","at":"2026-10-14T09:15"}"#,
            #"{"title":"Call Mei","at":"2026-10-14T09:15","repeats":false,"extra":null}"#,
            #"{"title":"call mei","at":"2026-10-14T09:15","repeats":false}"#,
            #"{"title":"Call Mei","at":"2026-10-13T09:15","repeats":false}"#,
        ] {
            XCTAssertFalse(ModelComparisonSuite.evaluate(item, reply: answer).passed, answer)
        }
    }

    func testTimerRequiresTotalSecondsAsANumber() throws {
        let item = try prompt("json-zh-1")
        XCTAssertTrue(ModelComparisonSuite.evaluate(item, reply: #"{"label":"泡茶","seconds":150}"#).passed)
        for answer in [
            #"{"label":"泡茶","seconds":"150"}"#,
            #"{"label":"泡茶","seconds":2.5}"#,
            #"{"label":"泡茶","seconds":120}"#,
            #"{"label":"泡茶","minutes":2,"seconds":30}"#,
        ] {
            XCTAssertFalse(ModelComparisonSuite.evaluate(item, reply: answer).passed, answer)
        }
    }

    func testJSONRejectsMarkdownExplanationsAndTrailingDocuments() throws {
        let item = try prompt("json-en-1")
        for answer in [
            "```json\n\(item.expected)\n```",
            "Here is the JSON: \(item.expected)",
            "\(item.expected)\nDone.",
            "\(item.expected)\n{}",
        ] {
            XCTAssertFalse(ModelComparisonSuite.evaluate(item, reply: answer).passed, answer)
        }
    }

    func testJSONArrayRequiresCorrectOrderAndElementTypes() throws {
        let item = try prompt("json-en-2")
        XCTAssertTrue(ModelComparisonSuite.evaluate(item, reply: "[ 3, 8 ]").passed)
        for answer in ["[8,3]", "[3,5,8]", "[3]", "[\"3\",\"8\"]", "{\"ids\":[3,8]}"] {
            XCTAssertFalse(ModelComparisonSuite.evaluate(item, reply: answer).passed, answer)
        }
    }

    func testJSONDistinguishesNullFromMissingStringOrInventedValue() throws {
        let item = try prompt("json-zh-2")
        XCTAssertTrue(ModelComparisonSuite.evaluate(item, reply: #"{"phone":null,"name":"李明"}"#).passed)
        for answer in [
            #"{"name":"李明"}"#,
            #"{"name":"李明","phone":"null"}"#,
            #"{"name":"李明","phone":"未知"}"#,
            #"{"name":"李明","phone":"123"}"#,
        ] {
            XCTAssertFalse(ModelComparisonSuite.evaluate(item, reply: answer).passed, answer)
        }
    }
}
