import XCTest
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
@testable import AssistantKit

/// Records each request and how its body stream ended, answering with `response`.
private final class RecordingTransport: HTTPStreamingTransport, @unchecked Sendable {
    private let response: MockTransport.Response
    private let lock = NSLock()
    private var recorded: [URLRequest] = []
    private var finishedCount = 0

    init(_ response: MockTransport.Response = .lines(["{\"data\":[]}"])) {
        self.response = response
    }

    var requests: [URLRequest] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    /// Body streams that were read to their end.
    var finishedStreams: Int {
        lock.lock()
        defer { lock.unlock() }
        return finishedCount
    }

    func lines(for request: URLRequest) async throws -> AsyncThrowingStream<String, Error> {
        lock.withTestLock { recorded.append(request) }
        return try MockTransport.stream(response) { [self] termination in
            guard case .finished = termination else { return }
            lock.withTestLock { finishedCount += 1 }
        }
    }
}

final class ConnectionPrewarmerTests: XCTestCase {
    private func modelsRequest(_ base: String = "https://api.anthropic.com") -> URLRequest {
        var request = URLRequest(url: URL(string: base + "/v1/models?limit=1")!)
        request.httpMethod = "GET"
        request.setValue("sk-test", forHTTPHeaderField: "x-api-key")
        request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        return request
    }

    func testSendsTheRequestUnchangedAndDrainsIt() async {
        let transport = RecordingTransport(.lines(["{\"data\":[", "{\"id\":\"model-1\"}", "]}"]))
        let prewarmer = ConnectionPrewarmer(transport: transport)
        await prewarmer.prewarm(modelsRequest())

        XCTAssertEqual(transport.requests.count, 1)
        let sent = transport.requests[0]
        XCTAssertEqual(sent.httpMethod, "GET")
        XCTAssertEqual(sent.url?.absoluteString, "https://api.anthropic.com/v1/models?limit=1")
        XCTAssertEqual(sent.value(forHTTPHeaderField: "x-api-key"), "sk-test")
        XCTAssertEqual(sent.value(forHTTPHeaderField: "anthropic-version"), "2023-06-01")
        XCTAssertNil(sent.httpBody)
        XCTAssertEqual(transport.finishedStreams, 1, "the response is read to its end")
        XCTAssertEqual(prewarmer.minimumInterval, 30)
    }

    func testRateLimitsPerOrigin() async {
        let transport = RecordingTransport()
        let clock = ManualClock(now: 100)
        let prewarmer = ConnectionPrewarmer(transport: transport, minimumInterval: 30) { clock.now }

        await prewarmer.prewarm(modelsRequest())
        clock.advance(by: 29)
        await prewarmer.prewarm(modelsRequest())
        XCTAssertEqual(transport.requests.count, 1, "a second prewarm within 30 s is skipped")

        // Another origin has its own allowance.
        await prewarmer.prewarm(modelsRequest("https://proxy.example.com"))
        XCTAssertEqual(transport.requests.count, 2)

        clock.advance(by: 1)
        await prewarmer.prewarm(modelsRequest())
        XCTAssertEqual(transport.requests.count, 3, "30 s after the last one, it goes again")
        await prewarmer.prewarm(modelsRequest())
        XCTAssertEqual(transport.requests.count, 3)
        XCTAssertEqual(transport.requests.map { $0.url?.host }, ["api.anthropic.com", "proxy.example.com", "api.anthropic.com"])
    }

    func testConcurrentCallsSendOnce() async {
        let transport = RecordingTransport()
        let prewarmer = ConnectionPrewarmer(transport: transport, minimumInterval: 30) { 0 }
        let request = modelsRequest()
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<5 {
                group.addTask { await prewarmer.prewarm(request) }
            }
        }
        XCTAssertEqual(transport.requests.count, 1)
    }

    func testErrorsAreIgnoredAndStillCount() async {
        let refused = RecordingTransport(.failure(HTTPStatusError(statusCode: 401, body: "{}")))
        let clock = ManualClock()
        let prewarmer = ConnectionPrewarmer(transport: refused, minimumInterval: 30) { clock.now }
        await prewarmer.prewarm(modelsRequest())
        await prewarmer.prewarm(modelsRequest())
        XCTAssertEqual(refused.requests.count, 1)

        let offline = RecordingTransport(.failure(URLError(.notConnectedToInternet)))
        await ConnectionPrewarmer(transport: offline).prewarm(modelsRequest())
        XCTAssertEqual(offline.requests.count, 1)
    }

    func testAZeroIntervalNeverSkips() async {
        let transport = RecordingTransport()
        let prewarmer = ConnectionPrewarmer(transport: transport, minimumInterval: 0) { 5 }
        await prewarmer.prewarm(modelsRequest())
        await prewarmer.prewarm(modelsRequest())
        XCTAssertEqual(transport.requests.count, 2)
    }

    func testOrigin() {
        XCTAssertEqual(ConnectionPrewarmer.origin(of: modelsRequest()), "https://api.anthropic.com:443")
        XCTAssertEqual(ConnectionPrewarmer.origin(of: modelsRequest("HTTPS://API.Anthropic.com:443")), "https://api.anthropic.com:443")
        XCTAssertEqual(ConnectionPrewarmer.origin(of: modelsRequest("http://localhost:8080")), "http://localhost:8080")
        XCTAssertEqual(ConnectionPrewarmer.origin(of: modelsRequest("http://example.com")), "http://example.com:80")
    }
}
