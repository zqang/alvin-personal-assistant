import Foundation
import Network
import Observation

/// Whether the iPhone has a network path, for routing (offline requests go to the on-device
/// model), setup hints and the connection prewarm.
///
/// It is optimistic: until the first path update arrives it reports online, so a launch never
/// starts out routing everything on device. A path that is up but leads nowhere (a captive portal,
/// dead Wi-Fi) still counts as online; the orchestrator's first-event watchdog covers that.
@MainActor
@Observable
final class Connectivity {
    static let shared = Connectivity()

    /// The current path is satisfied (or no path update has arrived yet).
    private(set) var isOnline = true
    /// The path uses cellular data or a personal hotspot.
    private(set) var isExpensive = false
    /// The path is in Low Data Mode.
    private(set) var isConstrained = false
    /// Whether a path update has arrived since launch.
    private(set) var hasPathUpdate = false

    private let monitor = NWPathMonitor()

    private init() {
        // Called on the monitor's queue, so it must not inherit this initializer's main-actor
        // isolation.
        monitor.pathUpdateHandler = { @Sendable path in
            let online = path.status == .satisfied
            let expensive = path.isExpensive
            let constrained = path.isConstrained
            Task { @MainActor in
                Connectivity.shared.update(online: online, expensive: expensive, constrained: constrained)
            }
        }
        monitor.start(queue: DispatchQueue(label: "alvin.connectivity", qos: .utility))
    }

    private func update(online: Bool, expensive: Bool, constrained: Bool) {
        // Assigning an equal value would still notify every view that reads it.
        if isOnline != online { isOnline = online }
        if isExpensive != expensive { isExpensive = expensive }
        if isConstrained != constrained { isConstrained = constrained }
        if !hasPathUpdate { hasPathUpdate = true }
    }
}
