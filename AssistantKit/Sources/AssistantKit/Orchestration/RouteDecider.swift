import Foundation

/// Which engines the router may choose from.
public enum RoutePolicy: Equatable, Sendable {
    /// Always the cloud (deep mode when asked). The on-device model is only a fallback, and only when
    /// it is downloaded.
    case cloudOnly
    /// Cloud or on-device per request. `preferLocal` is the "Prefer on-device" setting.
    case automatic(preferLocal: Bool)
}

/// Everything the router knows about the moment, gathered by the app on the main actor.
public struct RouteSignals: Equatable, Sendable {
    public var policy: RoutePolicy
    /// The device has a network path. Captive portals still count as online; the orchestrator's
    /// watchdog covers them.
    public var isOnline: Bool
    /// An Anthropic API key is present.
    public var cloudConfigured: Bool
    /// The on-device model's weights are downloaded.
    public var localAvailable: Bool
    /// The on-device model is loaded and can answer at once.
    public var localReady: Bool
    /// The request was spoken in a voice session.
    public var inputIsVoice: Bool
    /// The user asked for deep mode for this turn ("Think deeper").
    public var deepRequested: Bool
    public var deepMode: DeepModeSetting
    /// Today's deep-mode allowance isn't used up.
    public var deepBudgetLeft: Bool
    /// The "Answer small talk on device" setting.
    public var fastLocalSmallTalk: Bool
    /// The typical time to the first text of a standard cloud reply, for rule 8. Leave it nil when the
    /// estimate is old; `setExpectations(from:now:)` does that.
    public var expectedCloudFirstText: TimeInterval?
    /// The typical time to the first text of an on-device reply, from `LatencyEstimator`.
    public var expectedLocalFirstText: TimeInterval?

    public init(
        policy: RoutePolicy = .automatic(preferLocal: false),
        isOnline: Bool = true,
        cloudConfigured: Bool = true,
        localAvailable: Bool = false,
        localReady: Bool = false,
        inputIsVoice: Bool = false,
        deepRequested: Bool = false,
        deepMode: DeepModeSetting = .onRequest,
        deepBudgetLeft: Bool = true,
        fastLocalSmallTalk: Bool = true,
        expectedCloudFirstText: TimeInterval? = nil,
        expectedLocalFirstText: TimeInterval? = nil
    ) {
        self.policy = policy
        self.isOnline = isOnline
        self.cloudConfigured = cloudConfigured
        self.localAvailable = localAvailable
        self.localReady = localReady
        self.inputIsVoice = inputIsVoice
        self.deepRequested = deepRequested
        self.deepMode = deepMode
        self.deepBudgetLeft = deepBudgetLeft
        self.fastLocalSmallTalk = fastLocalSmallTalk
        self.expectedCloudFirstText = expectedCloudFirstText
        self.expectedLocalFirstText = expectedLocalFirstText
    }

    /// Fills `expectedCloudFirstText` and `expectedLocalFirstText` from the estimator, for standard
    /// replies. The cloud's estimate counts only while its newest sample is at most
    /// `RouteDecider.cloudSlowMaximumAge` old: while rule 8 keeps simple requests on device, the
    /// cloud gets few new samples, so an old "slow" verdict has to lapse rather than hold. The series
    /// itself keeps its history.
    public mutating func setExpectations(from latency: LatencyEstimator, now: Date = Date()) {
        expectedCloudFirstText = latency.expectedFirstText(
            engine: .cloud, mode: .standard, recordedWithin: RouteDecider.cloudSlowMaximumAge, now: now
        )
        expectedLocalFirstText = latency.expectedFirstText(engine: .local, mode: .standard)
    }
}

/// Picks the engine and mode for one request. A pure function of the utterance and the signals.
///
/// With `.automatic`, the first matching rule wins:
///
/// | # | Condition | Decision (fallback) |
/// |---|---|---|
/// | 1 | No cloud key | local, `noCloudKey` |
/// | 2 | Offline | local, `offline` |
/// | 3 | Deep wanted, budget left | cloud deep (local if downloaded) |
/// | 4 | fresh facts | cloud (local if downloaded) |
/// | 5 | device action ∧ prefer local ∧ local ready | local, `deviceAction` (cloud, by handoff) |
/// | 6 | device action | cloud (local if downloaded) |
/// | 7 | voice ∧ small talk ∧ fast local small talk ∧ local ready ∧ ¬complex | local, `latencyFirst` (cloud) |
/// | 8 | ¬complex ∧ local ready ∧ recent cloud first text > 3.5 s ∧ local not expected slower | local, `cloudSlow` (cloud) |
/// | 9 | complex | cloud |
/// | 10 | prefer local ∧ local ready | local, `localFirst` (cloud) |
/// | 11 | otherwise | cloud, `cloudDefault` (local if downloaded) |
///
/// *Deep wanted* = online ∧ cloud key ∧ (deep requested ∨ (explicit depth ∧ deep mode ≠ off) ∨
/// (deep mode automatic ∧ complex ∧ ¬voice)). Voice never goes deep automatically. When deep is
/// wanted but today's budget is used up, rules 4–11 pick the engine in standard mode and the reason
/// becomes `deepBudgetExhausted`, so the app can say why.
///
/// `.cloudOnly` always picks the cloud: deep when wanted and budget is left, otherwise standard
/// (`deepBudgetExhausted` or `userChoice`), with a local fallback only when the model is downloaded.
public struct RouteDecider: Sendable {
    /// Rule 8 sends simple requests on device when the cloud's typical time to first text exceeds this.
    public static let cloudSlowThreshold: TimeInterval = 3.5
    /// Rule 8 trusts a cloud estimate whose newest sample is at most this old (15 minutes); see
    /// `RouteSignals.setExpectations(from:now:)`.
    public static let cloudSlowMaximumAge: TimeInterval = 15 * 60

    public let classifier: IntentClassifier

    public init(classifier: IntentClassifier = .init()) {
        self.classifier = classifier
    }

    public func decide(_ utterance: String, signals: RouteSignals) -> RouteDecision {
        decide(intents: classifier.classify(utterance), signals: signals)
    }

    /// The decision for already classified intents.
    public func decide(intents: IntentClassifier.Intents, signals: RouteSignals) -> RouteDecision {
        let localFallback: ReplyEngine? = signals.localAvailable ? .local : nil
        let deep = Self.deepReason(intents: intents, signals: signals)

        switch signals.policy {
        case .cloudOnly:
            if let deep {
                return signals.deepBudgetLeft
                    ? RouteDecision(engine: .cloud, mode: .deep, reason: deep, fallback: localFallback)
                    : RouteDecision(engine: .cloud, reason: .deepBudgetExhausted, fallback: localFallback)
            }
            return RouteDecision(engine: .cloud, reason: .userChoice, fallback: localFallback)

        case .automatic(let preferLocal):
            if !signals.cloudConfigured {
                return RouteDecision(engine: .local, reason: .noCloudKey)
            }
            if !signals.isOnline {
                return RouteDecision(engine: .local, reason: .offline)
            }
            if let deep {
                if signals.deepBudgetLeft {
                    return RouteDecision(engine: .cloud, mode: .deep, reason: deep, fallback: localFallback)
                }
                var standard = standardRoute(intents: intents, signals: signals, preferLocal: preferLocal)
                standard.reason = .deepBudgetExhausted
                return standard
            }
            return standardRoute(intents: intents, signals: signals, preferLocal: preferLocal)
        }
    }

    /// Why deep mode is wanted (`deepRequested` or `deepAutomatic`), or nil when it isn't.
    public static func deepReason(intents: IntentClassifier.Intents, signals: RouteSignals) -> RouteReason? {
        guard signals.isOnline, signals.cloudConfigured else { return nil }
        if signals.deepRequested || (intents.contains(.explicitDepth) && signals.deepMode != .off) {
            return .deepRequested
        }
        if signals.deepMode == .automatic, intents.contains(.complex), !signals.inputIsVoice {
            return .deepAutomatic
        }
        return nil
    }

    /// Rules 4–11: online, with a cloud key, in standard mode.
    private func standardRoute(intents: IntentClassifier.Intents, signals: RouteSignals, preferLocal: Bool) -> RouteDecision {
        let localFallback: ReplyEngine? = signals.localAvailable ? .local : nil
        let complex = intents.contains(.complex)

        if intents.contains(.freshFacts) {
            return RouteDecision(engine: .cloud, reason: .freshFacts, fallback: localFallback)
        }
        if intents.contains(.deviceAction) {
            if preferLocal, signals.localReady {
                return RouteDecision(engine: .local, reason: .deviceAction, fallback: .cloud)
            }
            return RouteDecision(engine: .cloud, reason: .deviceAction, fallback: localFallback)
        }
        if signals.inputIsVoice, intents.contains(.smallTalk), signals.fastLocalSmallTalk, signals.localReady, !complex {
            return RouteDecision(engine: .local, reason: .latencyFirst, fallback: .cloud)
        }
        if !complex, signals.localReady, let cloud = signals.expectedCloudFirstText, cloud > Self.cloudSlowThreshold,
           signals.expectedLocalFirstText.map({ $0 < cloud }) ?? true {
            return RouteDecision(engine: .local, reason: .cloudSlow, fallback: .cloud)
        }
        if complex {
            return RouteDecision(engine: .cloud, reason: .complex)
        }
        if preferLocal, signals.localReady {
            return RouteDecision(engine: .local, reason: .localFirst, fallback: .cloud)
        }
        return RouteDecision(engine: .cloud, reason: .cloudDefault, fallback: localFallback)
    }
}
