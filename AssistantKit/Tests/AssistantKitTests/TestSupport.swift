import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
@testable import AssistantKit

/// A transport that replays scripted responses and records the requests it receives.
final class MockTransport: HTTPStreamingTransport, @unchecked Sendable {
    enum Response {
        case lines([String])
        case failure(Error)
    }

    private let lock = NSLock()
    private var responses: [Response]
    private var recorded: [URLRequest] = []

    init(_ responses: [Response]) {
        self.responses = responses
    }

    var requests: [URLRequest] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    func lines(for request: URLRequest) async throws -> AsyncThrowingStream<String, Error> {
        switch nextResponse(for: request) {
        case .failure(let error):
            throw error
        case .lines(let lines):
            return AsyncThrowingStream { continuation in
                for line in lines { continuation.yield(line) }
                continuation.finish()
            }
        }
    }

    private func nextResponse(for request: URLRequest) -> Response {
        lock.lock()
        defer { lock.unlock() }
        recorded.append(request)
        return responses.isEmpty ? .failure(URLError(.resourceUnavailable)) : responses.removeFirst()
    }

    /// Formats event payloads as an Anthropic-style SSE body.
    static func sse(_ events: [String]) -> [String] {
        events.flatMap { event -> [String] in
            let type = (try? JSONValue.parse(event))?["type"]?.stringValue ?? "message"
            return ["event: \(type)", "data: \(event)", ""]
        }
    }
}

func collect(_ stream: AsyncThrowingStream<ReplyEvent, Error>) async throws -> [ReplyEvent] {
    var events: [ReplyEvent] = []
    for try await event in stream {
        events.append(event)
    }
    return events
}
