import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Opens the connection to an API before the request that needs it, so the DNS lookup and TLS
/// handshake are done by the time the user finishes speaking.
///
/// The app sends a cheap request (`GET {baseURL}/v1/models?limit=1` with the API headers) through
/// the same transport, and therefore the same `URLSession`, that the reply will use, so the reply
/// reuses the warmed HTTP/2 connection. The response is read to the end and thrown away; errors are
/// ignored. Each origin is warmed at most once per `minimumInterval`, counted from the start of the
/// previous prewarm, whatever its outcome, on a clock that keeps running while the device sleeps.
public actor ConnectionPrewarmer {
    public nonisolated let minimumInterval: TimeInterval
    private let transport: HTTPStreamingTransport
    private let clock: @Sendable () -> TimeInterval
    private var lastStarted: [String: TimeInterval] = [:]

    public init(transport: HTTPStreamingTransport, minimumInterval: TimeInterval = 30) {
        self.init(transport: transport, minimumInterval: minimumInterval, clock: Self.continuousSeconds())
    }

    /// `clock` returns monotonic seconds; tests pass a manual clock.
    init(transport: HTTPStreamingTransport, minimumInterval: TimeInterval, clock: @escaping @Sendable () -> TimeInterval) {
        self.transport = transport
        self.minimumInterval = minimumInterval
        self.clock = clock
    }

    /// Sends `request` unchanged and drains the response, unless its origin was warmed less than
    /// `minimumInterval` ago. Never throws.
    public func prewarm(_ request: URLRequest) async {
        let origin = Self.origin(of: request)
        let now = clock()
        if let last = lastStarted[origin], now - last < minimumInterval { return }
        lastStarted[origin] = now
        do {
            for try await _ in try await transport.lines(for: request) {}
        } catch {
            // A failed prewarm costs nothing; the real request reports its own errors.
        }
    }

    /// Seconds on `ContinuousClock`, which keeps counting while the device sleeps. A connection goes
    /// cold during sleep, so time spent locked must count toward the interval; `systemUptime` stops
    /// then, and would skip the prewarm when the user unlocks the phone and starts speaking.
    static func continuousSeconds() -> @Sendable () -> TimeInterval {
        let clock = ContinuousClock()
        let origin = clock.now
        return {
            let elapsed = origin.duration(to: clock.now).components
            return Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
        }
    }

    /// `scheme://host:port`, lowercased: the unit a connection is reused for.
    static func origin(of request: URLRequest) -> String {
        guard let url = request.url else { return "" }
        let scheme = url.scheme?.lowercased() ?? ""
        let host = url.host?.lowercased() ?? ""
        let port = url.port.map(String.init) ?? (scheme == "http" ? "80" : "443")
        return "\(scheme)://\(host):\(port)"
    }
}
