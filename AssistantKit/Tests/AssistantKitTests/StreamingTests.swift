import XCTest
@testable import AssistantKit

final class JSONValueTests: XCTestCase {
    func testRoundTripsNestedValues() throws {
        let value: JSONValue = ["name": "Claude", "count": 3, "ratio": 0.5, "ok": true, "items": [1, "two"], "none": .null]
        XCTAssertEqual(try JSONValue.parse(value.serialized()), value)
    }

    func testAccessors() throws {
        let value = try JSONValue.parse(#"{"a": {"b": [1, 2.0, "x"]}, "flag": false}"#)
        XCTAssertEqual(value["a"]?["b"]?.arrayValue?.count, 3)
        XCTAssertEqual(value["a"]?["b"]?.arrayValue?[1].intValue, 2)
        XCTAssertEqual(value["a"]?["b"]?.arrayValue?[2].stringValue, "x")
        XCTAssertEqual(value["flag"]?.boolValue, false)
        XCTAssertNil(value["missing"])
    }

    func testSerializationSortsKeys() throws {
        let value: JSONValue = ["b": 1, "a": 2]
        XCTAssertEqual(String(decoding: try value.serialized(), as: UTF8.self), #"{"a":2,"b":1}"#)
    }
}

final class LineAndSSETests: XCTestCase {
    func testLineSplitterHandlesCRLFAndBlankLines() {
        var splitter = LineSplitter()
        let lines = splitter.append(contentsOf: Array("event: a\r\ndata: 1\r\n\r\ndata: 2".utf8))
        XCTAssertEqual(lines, ["event: a", "data: 1", ""])
        XCTAssertEqual(splitter.finish(), "data: 2")
        XCTAssertNil(splitter.finish())
    }

    func testLineSplitterKeepsMultibyteCharacters() {
        var splitter = LineSplitter()
        XCTAssertEqual(splitter.append(contentsOf: Array("你好 👋\n".utf8)), ["你好 👋"])
    }

    func testSSEParserDispatchesEachDataLineWithItsEventName() {
        var parser = SSEParser()
        XCTAssertNil(parser.consume(line: "event: message_start"))
        XCTAssertEqual(parser.consume(line: #"data: {"x":1}"#), SSEParser.Event(name: "message_start", data: #"{"x":1}"#))
        XCTAssertNil(parser.consume(line: ""))
        XCTAssertNil(parser.consume(line: ": keep-alive"))
        XCTAssertEqual(parser.consume(line: "data:[DONE]"), SSEParser.Event(name: nil, data: "[DONE]"))
    }

    func testSSEParserSpecModeJoinsDataLines() {
        var parser = SSEParser(dispatchesEachDataLine: false)
        XCTAssertNil(parser.consume(line: "event: note"))
        XCTAssertNil(parser.consume(line: "data: first"))
        XCTAssertNil(parser.consume(line: "data: second"))
        XCTAssertEqual(parser.consume(line: ""), SSEParser.Event(name: "note", data: "first\nsecond"))
        XCTAssertNil(parser.finish())
    }

    func testRetryPolicy() {
        XCTAssertTrue(RetryPolicy.isRetryable(HTTPStatusError(statusCode: 529, body: "")))
        XCTAssertTrue(RetryPolicy.isRetryable(HTTPStatusError(statusCode: 429, body: "")))
        XCTAssertFalse(RetryPolicy.isRetryable(HTTPStatusError(statusCode: 400, body: "")))
        XCTAssertFalse(RetryPolicy.isRetryable(HTTPStatusError(statusCode: 401, body: "")))
        XCTAssertEqual(RetryPolicy.delayNanoseconds(attempt: 1, retryAfter: nil), 2_000_000_000)
        XCTAssertEqual(RetryPolicy.delayNanoseconds(attempt: 0, retryAfter: 30), 10_000_000_000)
    }
}
