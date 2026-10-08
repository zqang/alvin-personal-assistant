import XCTest
@testable import AssistantKit

final class ToolInputValidatorTests: XCTestCase {
    // Schemas in the shape of §5.3's tools (scripts/lab_prompts.json).
    private let setTimer: JSONValue = [
        "type": "object",
        "properties": [
            "seconds": ["type": "integer", "description": "From 1 to 86400."],
            "label": ["type": "string"],
        ],
        "required": ["seconds"],
        "additionalProperties": false,
    ]
    private let listReminders: JSONValue = [
        "type": "object",
        "properties": [
            "scope": ["type": "string", "enum": ["today", "upcoming", "overdue", "all"]],
        ],
        "required": ["scope"],
        "additionalProperties": false,
    ]
    private let createEvent: JSONValue = [
        "type": "object",
        "properties": [
            "title": ["type": "string"],
            "start": ["type": "string"],
            "end": ["type": "string"],
            "duration_minutes": ["type": "integer"],
            "all_day": ["type": "boolean"],
        ],
        "required": ["title", "start"],
        "additionalProperties": false,
    ]
    private let nested: JSONValue = [
        "type": "object",
        "properties": [
            "filter": [
                "type": "object",
                "properties": ["limit": ["type": "integer"]],
                "required": ["limit"],
                "additionalProperties": false,
            ],
            "tags": ["type": "array", "items": ["type": "string"]],
            "note": ["type": ["string", "null"]],
            "ratio": ["type": "number"],
        ],
        "required": .array([]),
        "additionalProperties": false,
    ]

    private func validate(_ input: JSONValue?, raw: String? = nil, schema: JSONValue, lenient: Bool = false) -> Result<[String: JSONValue], ToolInputError> {
        ToolInputValidator.validate(PendingToolCall(id: "c1", name: "tool", input: input, rawInput: raw), schema: schema, lenient: lenient)
    }

    private func assertAccepts(
        _ input: JSONValue?,
        schema: JSONValue,
        lenient: Bool = false,
        as expected: [String: JSONValue],
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(validate(input, schema: schema, lenient: lenient), .success(expected), file: file, line: line)
    }

    private func assertRejects(
        _ input: JSONValue?,
        raw: String? = nil,
        schema: JSONValue,
        lenient: Bool = false,
        with expected: ToolInputError,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(validate(input, raw: raw, schema: schema, lenient: lenient), .failure(expected), file: file, line: line)
    }

    // MARK: - Shape

    func testUnparsedInputIsInvalidJSONInBothModes() {
        for lenient in [false, true] {
            assertRejects(nil, raw: #"{"seconds": 6"#, schema: setTimer, lenient: lenient, with: .invalidJSON(#"{"seconds": 6"#))
            assertRejects(nil, schema: setTimer, lenient: lenient, with: .invalidJSON(""))
        }
    }

    func testInputMustBeAnObject() {
        assertRejects([1, 2], schema: setTimer, with: .notAnObject)
        assertRejects("x", schema: setTimer, with: .notAnObject)
        assertRejects(.null, schema: setTimer, with: .notAnObject)
        assertRejects(#"{"seconds":5}"#, schema: setTimer, with: .notAnObject)
        assertRejects(5, schema: setTimer, lenient: true, with: .notAnObject)
        assertRejects("not json", schema: setTimer, lenient: true, with: .notAnObject)
    }

    func testLenientModeUnwrapsObjectsGivenAsStringsAndNullAsEmpty() {
        assertAccepts(#"{"seconds":5}"#, schema: setTimer, lenient: true, as: ["seconds": 5])
        assertAccepts(.null, schema: FakeTool.emptySchema, lenient: true, as: [:])
        assertRejects(.null, schema: setTimer, lenient: true, with: .missing("seconds"))
    }

    func testValidInputPassesUnchanged() {
        for lenient in [false, true] {
            assertAccepts(["seconds": 600, "label": "Pasta"], schema: setTimer, lenient: lenient, as: ["seconds": 600, "label": "Pasta"])
            assertAccepts(["scope": "overdue"], schema: listReminders, lenient: lenient, as: ["scope": "overdue"])
            assertAccepts(
                ["title": "Dentist", "start": "2026-10-09T15:00", "all_day": false, "duration_minutes": 30],
                schema: createEvent,
                lenient: lenient,
                as: ["title": "Dentist", "start": "2026-10-09T15:00", "all_day": false, "duration_minutes": 30]
            )
            assertAccepts([:], schema: FakeTool.emptySchema, lenient: lenient, as: [:])
        }
    }

    // MARK: - Required and unexpected fields

    func testMissingRequiredFields() {
        assertRejects([:], schema: setTimer, with: .missing("seconds"))
        assertRejects(["label": "Pasta"], schema: setTimer, lenient: true, with: .missing("seconds"))
        // The first missing field in the schema's order.
        assertRejects(["end": "2026-10-09T16:00"], schema: createEvent, with: .missing("title"))
        assertRejects(["title": "Dentist"], schema: createEvent, with: .missing("start"))
    }

    func testUnexpectedFieldsAreRejectedInBothModes() {
        for lenient in [false, true] {
            assertRejects(["seconds": 5, "minutes": 1], schema: setTimer, lenient: lenient, with: .unexpected("minutes"))
            assertRejects(["scope": "all", "time": "today"], schema: listReminders, lenient: lenient, with: .unexpected("time"))
        }
    }

    func testOpenSchemasKeepExtraFields() {
        let open: JSONValue = ["type": "object", "properties": ["a": ["type": "integer"]]]
        assertAccepts(["a": 1, "b": "x"], schema: open, as: ["a": 1, "b": "x"])
        assertAccepts(["anything": [1, 2]], schema: .null, as: ["anything": [1, 2]])
        let typedExtras: JSONValue = ["type": "object", "additionalProperties": ["type": "integer"]]
        assertAccepts(["n": "7"], schema: typedExtras, lenient: true, as: ["n": 7])
        assertRejects(["n": "7"], schema: typedExtras, with: .wrongType(field: "n", expected: "integer"))
    }

    // MARK: - Types

    func testIntegers() {
        assertAccepts(["seconds": 5.0], schema: setTimer, as: ["seconds": 5])
        assertRejects(["seconds": 5.5], schema: setTimer, with: .wrongType(field: "seconds", expected: "integer"))
        assertRejects(["seconds": "5"], schema: setTimer, with: .wrongType(field: "seconds", expected: "integer"))
        assertRejects(["seconds": true], schema: setTimer, with: .wrongType(field: "seconds", expected: "integer"))
        assertAccepts(["seconds": "5"], schema: setTimer, lenient: true, as: ["seconds": 5])
        assertAccepts(["seconds": " 600 "], schema: setTimer, lenient: true, as: ["seconds": 600])
        assertAccepts(["seconds": "5.0"], schema: setTimer, lenient: true, as: ["seconds": 5])
        assertRejects(["seconds": "5.5"], schema: setTimer, lenient: true, with: .wrongType(field: "seconds", expected: "integer"))
        assertRejects(["seconds": "ten"], schema: setTimer, lenient: true, with: .wrongType(field: "seconds", expected: "integer"))
    }

    func testBooleans() {
        let base: [String: JSONValue] = ["title": "Trip", "start": "2026-10-09"]
        func input(_ value: JSONValue) -> JSONValue { .object(base.merging(["all_day": value]) { $1 }) }
        assertRejects(input("true"), schema: createEvent, with: .wrongType(field: "all_day", expected: "boolean"))
        assertRejects(input(1), schema: createEvent, with: .wrongType(field: "all_day", expected: "boolean"))
        assertAccepts(input("true"), schema: createEvent, lenient: true, as: base.merging(["all_day": true]) { $1 })
        assertAccepts(input(" FALSE "), schema: createEvent, lenient: true, as: base.merging(["all_day": false]) { $1 })
        assertRejects(input("yes"), schema: createEvent, lenient: true, with: .wrongType(field: "all_day", expected: "boolean"))
    }

    func testStrings() {
        assertRejects(["seconds": 5, "label": 7], schema: setTimer, with: .wrongType(field: "label", expected: "string"))
        assertAccepts(["seconds": 5, "label": 7], schema: setTimer, lenient: true, as: ["seconds": 5, "label": "7"])
        assertAccepts(["seconds": 5, "label": 2.5], schema: setTimer, lenient: true, as: ["seconds": 5, "label": "2.5"])
        assertRejects(["seconds": 5, "label": ["a"]], schema: setTimer, lenient: true, with: .wrongType(field: "label", expected: "string"))
    }

    func testNumbers() {
        assertAccepts(["ratio": 2], schema: nested, as: ["ratio": 2])
        assertAccepts(["ratio": 2.5], schema: nested, as: ["ratio": 2.5])
        assertRejects(["ratio": "2.5"], schema: nested, with: .wrongType(field: "ratio", expected: "number"))
        assertAccepts(["ratio": "2.5"], schema: nested, lenient: true, as: ["ratio": 2.5])
        assertAccepts(["ratio": "3"], schema: nested, lenient: true, as: ["ratio": 3])
    }

    func testNulls() {
        // Strict: null is a wrong type unless the schema allows it.
        assertRejects(["seconds": 5, "label": .null], schema: setTimer, with: .wrongType(field: "label", expected: "string"))
        assertAccepts(["note": .null], schema: nested, as: ["note": .null])
        assertRejects(["note": 3], schema: nested, with: .wrongType(field: "note", expected: "string or null"))
        // Lenient: a null optional field is left out; a null required field is missing.
        assertAccepts(["seconds": 5, "label": .null], schema: setTimer, lenient: true, as: ["seconds": 5])
        assertRejects(["seconds": .null], schema: setTimer, lenient: true, with: .missing("seconds"))
        assertAccepts(["note": .null], schema: nested, lenient: true, as: ["note": .null])
    }

    // MARK: - Enums

    func testEnumValues() {
        assertRejects(["scope": "weekly"], schema: listReminders, with: .notAllowed(field: "scope", value: "weekly"))
        assertRejects(["scope": "Today"], schema: listReminders, with: .notAllowed(field: "scope", value: "Today"))
        assertAccepts(["scope": " Today "], schema: listReminders, lenient: true, as: ["scope": "today"])
        assertRejects(["scope": "weekly"], schema: listReminders, lenient: true, with: .notAllowed(field: "scope", value: "weekly"))
        let numeric: JSONValue = ["type": "object", "properties": ["level": ["enum": [1, 2, 3]]]]
        assertAccepts(["level": 2], schema: numeric, as: ["level": 2])
        assertRejects(["level": 4], schema: numeric, with: .notAllowed(field: "level", value: "4"))
    }

    // MARK: - Nesting and order

    func testNestedFieldsReportTheirPath() {
        assertAccepts(["filter": ["limit": "3"], "tags": ["a", 2]], schema: nested, lenient: true, as: ["filter": ["limit": 3], "tags": ["a", "2"]])
        assertRejects(["filter": [:]], schema: nested, with: .missing("filter.limit"))
        assertRejects(["filter": ["limit": 1, "x": 2]], schema: nested, with: .unexpected("filter.x"))
        assertRejects(["filter": ["limit": "3"]], schema: nested, with: .wrongType(field: "filter.limit", expected: "integer"))
        assertRejects(["tags": ["a", 2]], schema: nested, with: .wrongType(field: "tags[1]", expected: "string"))
        assertRejects(["filter": 3], schema: nested, with: .wrongType(field: "filter", expected: "object"))
    }

    func testSeveralProblemsGiveTheSameErrorEveryTime() {
        // Missing required fields come first, then the given fields in name order.
        let input: JSONValue = ["zeta": 1, "alpha": 2, "duration_minutes": "x"]
        for _ in 0..<5 {
            assertRejects(input, schema: createEvent, with: .missing("title"))
        }
        let present: JSONValue = ["title": "T", "start": "S", "zeta": 1, "alpha": 2]
        assertRejects(present, schema: createEvent, with: .unexpected("alpha"))
        let mistyped: JSONValue = ["title": 1, "start": 2]
        assertRejects(mistyped, schema: createEvent, with: .wrongType(field: "start", expected: "string"))
    }

    // MARK: - Error content

    func testErrorContent() {
        XCTAssertEqual(ToolInputValidator.errorContent(.invalidJSON(#"{"a":"#)), ["INVALID_JSON": #"{"a":"#])
        XCTAssertEqual(ToolInputValidator.errorContent(.missing("seconds")), ["error": #"Missing required field "seconds"."#])
        XCTAssertEqual(ToolInputValidator.errorContent(.wrongType(field: "seconds", expected: "integer")), ["error": #"Field "seconds" must be an integer."#])
        XCTAssertEqual(
            ToolInputValidator.errorContent(.wrongType(field: "note", expected: "string or null")),
            ["error": #"Field "note" must be a string or null."#]
        )
        XCTAssertEqual(ToolInputValidator.errorContent(.notAnObject), ["error": "The input must be a JSON object."])
        let others: [ToolInputError] = [.notAllowed(field: "scope", value: "weekly"), .unexpected("minutes")]
        for error in others {
            let content = ToolInputValidator.errorContent(error)
            XCTAssertEqual(content.objectValue?.keys.sorted(), ["error"])
            XCTAssertFalse(content["error"]?.stringValue?.isEmpty ?? true)
        }
        XCTAssertTrue(ToolInputValidator.errorContent(.notAllowed(field: "scope", value: "weekly"))["error"]?.stringValue?.contains("weekly") == true)
        XCTAssertTrue(ToolInputValidator.errorContent(.unexpected("minutes"))["error"]?.stringValue?.contains("minutes") == true)
    }
}
