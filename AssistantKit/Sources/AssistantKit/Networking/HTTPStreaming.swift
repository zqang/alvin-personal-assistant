import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// A non-2xx HTTP response, with its body read in full.
public struct HTTPStatusError: Error, Equatable, Sendable {
    public let statusCode: Int
    public let body: String
    public let retryAfter: TimeInterval?

    public init(statusCode: Int, body: String, retryAfter: TimeInterval? = nil) {
        self.statusCode = statusCode
        self.body = body
        self.retryAfter = retryAfter
    }
}

/// Opens streaming HTTP requests. The app uses `URLSessionStreamingTransport`; tests script responses.
public protocol HTTPStreamingTransport: Sendable {
    /// Sends `request` and returns the response body as lines (LF or CRLF, empty lines kept).
    /// Throws `HTTPStatusError` when the server answers with a non-2xx status.
    func lines(for request: URLRequest) async throws -> AsyncThrowingStream<String, Error>
}

/// Splits a byte stream into lines, accepting LF and CRLF terminators and keeping empty lines,
/// which Server-Sent Events use to end each event.
public struct LineSplitter: Sendable {
    private var buffer: [UInt8] = []

    public init() {}

    public mutating func append(_ byte: UInt8) -> String? {
        guard byte == 0x0A else {
            buffer.append(byte)
            return nil
        }
        if buffer.last == 0x0D { buffer.removeLast() }
        let line = String(decoding: buffer, as: UTF8.self)
        buffer.removeAll(keepingCapacity: true)
        return line
    }

    public mutating func append<Bytes: Sequence>(contentsOf bytes: Bytes) -> [String] where Bytes.Element == UInt8 {
        var lines: [String] = []
        for byte in bytes {
            if let line = append(byte) { lines.append(line) }
        }
        return lines
    }

    /// Returns the unterminated final line, if any.
    public mutating func finish() -> String? {
        guard !buffer.isEmpty else { return nil }
        if buffer.last == 0x0D { buffer.removeLast() }
        let line = String(decoding: buffer, as: UTF8.self)
        buffer.removeAll()
        return line
    }
}

/// Server-Sent Events parser for the subset the model APIs use: `event:` and `data:` fields and comments.
public struct SSEParser: Sendable {
    public struct Event: Equatable, Sendable {
        public var name: String?
        public var data: String

        public init(name: String?, data: String) {
            self.name = name
            self.data = data
        }
    }

    /// When true, every `data:` line is delivered as its own event instead of waiting for the blank
    /// line that ends it. Neither API sends multi-line data, and this tolerates servers that omit blank lines.
    public var dispatchesEachDataLine: Bool

    private var eventName: String?
    private var dataLines: [String] = []

    public init(dispatchesEachDataLine: Bool = true) {
        self.dispatchesEachDataLine = dispatchesEachDataLine
    }

    public mutating func consume(line: String) -> Event? {
        if line.isEmpty {
            let event = pendingEvent()
            eventName = nil
            return event
        }
        if line.hasPrefix(":") { return nil }

        let field: Substring
        var value: Substring
        if let colon = line.firstIndex(of: ":") {
            field = line[..<colon]
            value = line[line.index(after: colon)...]
            if value.hasPrefix(" ") { value = value.dropFirst() }
        } else {
            field = Substring(line)
            value = ""
        }

        switch field {
        case "event":
            eventName = String(value)
        case "data":
            dataLines.append(String(value))
            if dispatchesEachDataLine { return pendingEvent() }
        default:
            break
        }
        return nil
    }

    /// Delivers an event left open when the stream ended without a trailing blank line.
    public mutating func finish() -> Event? {
        pendingEvent()
    }

    private mutating func pendingEvent() -> Event? {
        guard !dataLines.isEmpty else { return nil }
        let event = Event(name: eventName, data: dataLines.joined(separator: "\n"))
        dataLines.removeAll()
        return event
    }
}

enum RetryPolicy {
    static func isRetryable(status: Int) -> Bool {
        status == 408 || status == 409 || status == 429 || (500...599).contains(status)
    }

    static func isRetryable(_ error: Error) -> Bool {
        if let status = error as? HTTPStatusError { return isRetryable(status: status.statusCode) }
        guard let urlError = error as? URLError else { return false }
        switch urlError.code {
        case .timedOut, .networkConnectionLost, .cannotConnectToHost, .dnsLookupFailed:
            return true
        default:
            return false
        }
    }

    /// Exponential backoff (1s, 2s, 4s...), or the server's `retry-after` when it is short.
    static func delayNanoseconds(attempt: Int, retryAfter: TimeInterval?) -> UInt64 {
        let backoff = pow(2.0, Double(attempt))
        let seconds = retryAfter.map { min(max($0, 0.5), 10) } ?? backoff
        return UInt64(seconds * 1_000_000_000)
    }

    /// Opens `request`, retrying transient failures before any of the body has been read.
    static func open(
        _ request: URLRequest,
        with transport: HTTPStreamingTransport,
        maxRetries: Int = 2
    ) async throws -> AsyncThrowingStream<String, Error> {
        var attempt = 0
        while true {
            do {
                return try await transport.lines(for: request)
            } catch {
                guard attempt < maxRetries, isRetryable(error) else { throw error }
                let retryAfter = (error as? HTTPStatusError)?.retryAfter
                try await Task.sleep(nanoseconds: delayNanoseconds(attempt: attempt, retryAfter: retryAfter))
                attempt += 1
            }
        }
    }
}

#if canImport(Darwin)
/// Streams response bodies with `URLSession.bytes(for:)`.
public struct URLSessionStreamingTransport: HTTPStreamingTransport {
    private let session: URLSession

    public init(session: URLSession = .shared) {
        self.session = session
    }

    public func lines(for request: URLRequest) async throws -> AsyncThrowingStream<String, Error> {
        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        guard (200..<300).contains(http.statusCode) else {
            var body = Data()
            for try await byte in bytes {
                body.append(byte)
                if body.count >= 64 * 1024 { break }
            }
            let retryAfter = http.value(forHTTPHeaderField: "retry-after").flatMap { TimeInterval($0) }
            throw HTTPStatusError(
                statusCode: http.statusCode,
                body: String(decoding: body, as: UTF8.self),
                retryAfter: retryAfter
            )
        }
        return AsyncThrowingStream { continuation in
            let task = Task {
                var splitter = LineSplitter()
                do {
                    for try await byte in bytes {
                        if let line = splitter.append(byte) { continuation.yield(line) }
                    }
                    if let line = splitter.finish() { continuation.yield(line) }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
#endif
