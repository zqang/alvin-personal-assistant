import Foundation

/// Where a reply is generated.
public enum ReplyEngine: String, Codable, Sendable {
    /// The Claude API.
    case cloud
    /// The model on the iPhone.
    case local
}

/// How much work a reply gets.
public enum ReplyMode: String, Codable, Sendable {
    case standard
    /// Deep mode: more effort or several workers, with a longer deadline.
    case deep
}

/// Why a route was chosen.
public enum RouteReason: String, Codable, Sendable {
    /// The user picked a provider and automatic routing is off.
    case userChoice
    case noCloudKey
    case offline
    case deepRequested
    case deepAutomatic
    /// Deep mode was wanted but today's allowance is used up.
    case deepBudgetExhausted
    /// The request needs current information (news, weather, prices).
    case freshFacts
    /// The request acts on the device (reminders, calendar, timers).
    case deviceAction
    case complex
    /// The user prefers on-device replies.
    case localFirst
    /// Small talk in voice mode, answered on device for speed.
    case latencyFirst
    /// The cloud has been slow to start replying lately.
    case cloudSlow
    case cloudDefault
    /// The cloud couldn't be reached, so the on-device model answers.
    case networkFallback
    /// The on-device model handed the request to the cloud.
    case escalated
}

/// The engine and mode chosen for one reply, and where it may fall back to.
public struct RouteDecision: Equatable, Sendable {
    public var engine: ReplyEngine
    public var mode: ReplyMode
    public var reason: RouteReason
    /// The engine to retry on if this one fails before anything was committed.
    public var fallback: ReplyEngine?

    public init(engine: ReplyEngine, mode: ReplyMode = .standard, reason: RouteReason, fallback: ReplyEngine? = nil) {
        self.engine = engine
        self.mode = mode
        self.reason = reason
        self.fallback = fallback
    }
}
