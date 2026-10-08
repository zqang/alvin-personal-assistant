import XCTest
@testable import AssistantKit

final class RouteDeciderTests: XCTestCase {
    private let decider = RouteDecider()

    /// Automatic routing, online, with a key and a downloaded model that is loaded.
    private func signals(
        preferLocal: Bool = false,
        isOnline: Bool = true,
        cloudConfigured: Bool = true,
        localAvailable: Bool = true,
        localReady: Bool = true,
        voice: Bool = false,
        deepRequested: Bool = false,
        deepMode: DeepModeSetting = .onRequest,
        deepBudgetLeft: Bool = true,
        fastLocalSmallTalk: Bool = true,
        cloudFirstText: TimeInterval? = nil,
        localFirstText: TimeInterval? = nil
    ) -> RouteSignals {
        RouteSignals(
            policy: .automatic(preferLocal: preferLocal),
            isOnline: isOnline,
            cloudConfigured: cloudConfigured,
            localAvailable: localAvailable,
            localReady: localReady,
            inputIsVoice: voice,
            deepRequested: deepRequested,
            deepMode: deepMode,
            deepBudgetLeft: deepBudgetLeft,
            fastLocalSmallTalk: fastLocalSmallTalk,
            expectedCloudFirstText: cloudFirstText,
            expectedLocalFirstText: localFirstText
        )
    }

    /// A standard-mode decision.
    private func route(_ engine: ReplyEngine, _ reason: RouteReason, fallback: ReplyEngine? = nil) -> RouteDecision {
        RouteDecision(engine: engine, mode: .standard, reason: reason, fallback: fallback)
    }

    /// A deep-mode cloud decision.
    private func deep(_ reason: RouteReason, fallback: ReplyEngine? = nil) -> RouteDecision {
        RouteDecision(engine: .cloud, mode: .deep, reason: reason, fallback: fallback)
    }

    // MARK: The table, row by row

    func testRow1NoCloudKeyGoesLocal() {
        let expected = route(.local, .noCloudKey)
        XCTAssertEqual(decider.decide("What's the weather?", signals: signals(cloudConfigured: false)), expected)
        XCTAssertEqual(decider.decide("Think hard about this", signals: signals(cloudConfigured: false, deepRequested: true)), expected)
        XCTAssertEqual(decider.decide("Hi", signals: signals(isOnline: false, cloudConfigured: false, localAvailable: false, localReady: false)), expected)
    }

    func testRow2OfflineGoesLocal() {
        let expected = route(.local, .offline)
        XCTAssertEqual(decider.decide("Latest news", signals: signals(isOnline: false)), expected)
        XCTAssertEqual(decider.decide("Compare these", signals: signals(isOnline: false, deepRequested: true)), expected)
        XCTAssertEqual(decider.decide("天气怎么样", signals: signals(isOnline: false, localReady: false)), expected)
    }

    func testRow3DeepWantedWithBudget() {
        XCTAssertEqual(decider.decide("Tell me about Rome", signals: signals(deepRequested: true)), deep(.deepRequested, fallback: .local))
        XCTAssertEqual(decider.decide("Think hard about my options", signals: signals()), deep(.deepRequested, fallback: .local))
        XCTAssertEqual(decider.decide("请认真想一下", signals: signals(localAvailable: false, localReady: false)), deep(.deepRequested))
        XCTAssertEqual(decider.decide("Compare these two laptops", signals: signals(deepMode: .automatic)), deep(.deepAutomatic, fallback: .local))
        // Explicit depth is ignored when deep mode is off; an explicit request still counts.
        XCTAssertEqual(decider.decide("Think hard about my options", signals: signals(deepMode: .off)), route(.cloud, .cloudDefault, fallback: .local))
        XCTAssertEqual(decider.decide("Tell me about Rome", signals: signals(deepRequested: true, deepMode: .off)), deep(.deepRequested, fallback: .local))
        // Complex requests go deep automatically only with the automatic setting.
        XCTAssertEqual(decider.decide("Compare these two laptops", signals: signals()), route(.cloud, .complex))
    }

    func testRow4FreshFacts() {
        XCTAssertEqual(decider.decide("What's the weather in Paris", signals: signals(preferLocal: true)), route(.cloud, .freshFacts, fallback: .local))
        XCTAssertEqual(decider.decide("今天有什么新闻", signals: signals(localAvailable: false, localReady: false)), route(.cloud, .freshFacts))
        // Fresh facts win over small talk, even by voice.
        XCTAssertEqual(decider.decide("Hi, any news?", signals: signals(voice: true)), route(.cloud, .freshFacts, fallback: .local))
    }

    func testRow5DeviceActionOnDeviceWhenPreferredAndReady() {
        XCTAssertEqual(decider.decide("Set a timer for ten minutes", signals: signals(preferLocal: true)), route(.local, .deviceAction, fallback: .cloud))
        XCTAssertEqual(decider.decide("明天早上提醒我开会", signals: signals(preferLocal: true, voice: true)), route(.local, .deviceAction, fallback: .cloud))
    }

    func testRow6DeviceActionInTheCloudOtherwise() {
        XCTAssertEqual(decider.decide("Set a timer for ten minutes", signals: signals()), route(.cloud, .deviceAction, fallback: .local))
        XCTAssertEqual(decider.decide("Set a timer for ten minutes", signals: signals(preferLocal: true, localReady: false)), route(.cloud, .deviceAction, fallback: .local))
        XCTAssertEqual(decider.decide("What's on my calendar", signals: signals(localAvailable: false, localReady: false)), route(.cloud, .deviceAction))
    }

    func testRow7VoiceSmallTalkOnDevice() {
        XCTAssertEqual(decider.decide("Thanks!", signals: signals(voice: true)), route(.local, .latencyFirst, fallback: .cloud))
        XCTAssertEqual(decider.decide("What time is it right now?", signals: signals(voice: true)), route(.local, .latencyFirst, fallback: .cloud))
        XCTAssertEqual(decider.decide("早上好", signals: signals(voice: true)), route(.local, .latencyFirst, fallback: .cloud))
        XCTAssertEqual(decider.decide("12 x 4", signals: signals(voice: true)), route(.local, .latencyFirst, fallback: .cloud))
        // Each condition is needed.
        XCTAssertEqual(decider.decide("Thanks!", signals: signals(voice: true, fastLocalSmallTalk: false)), route(.cloud, .cloudDefault, fallback: .local))
        XCTAssertEqual(decider.decide("Thanks!", signals: signals(localReady: false, voice: true)), route(.cloud, .cloudDefault, fallback: .local))
        XCTAssertEqual(decider.decide("Hi! Why? How?", signals: signals(voice: true)), route(.cloud, .complex))
    }

    func testTypedSmallTalkNeverGoesLocalThroughSmallTalk() {
        for utterance in ["Thanks!", "Hi", "你好", "What time is it", "2 + 2"] {
            let decision = decider.decide(utterance, signals: signals(voice: false))
            XCTAssertEqual(decision, route(.cloud, .cloudDefault, fallback: .local), utterance)
            XCTAssertNotEqual(decision.reason, .latencyFirst)
        }
    }

    func testRow8SlowCloudSendsSimpleRequestsOnDevice() {
        XCTAssertEqual(decider.decide("Tell me a joke", signals: signals(cloudFirstText: 4.2)), route(.local, .cloudSlow, fallback: .cloud))
        XCTAssertEqual(decider.decide("Tell me a joke", signals: signals(cloudFirstText: 4.2, localFirstText: 1.1)), route(.local, .cloudSlow, fallback: .cloud))
        // Not when the cloud is fast enough, unknown, local isn't ready, or local is even slower.
        XCTAssertEqual(decider.decide("Tell me a joke", signals: signals(cloudFirstText: 3.5)), route(.cloud, .cloudDefault, fallback: .local))
        XCTAssertEqual(decider.decide("Tell me a joke", signals: signals(cloudFirstText: nil)), route(.cloud, .cloudDefault, fallback: .local))
        XCTAssertEqual(decider.decide("Tell me a joke", signals: signals(localReady: false, cloudFirstText: 6)), route(.cloud, .cloudDefault, fallback: .local))
        XCTAssertEqual(decider.decide("Tell me a joke", signals: signals(cloudFirstText: 4, localFirstText: 5)), route(.cloud, .cloudDefault, fallback: .local))
        // Never for complex requests, and fresh facts and device actions come first.
        XCTAssertEqual(decider.decide("Compare these two laptops", signals: signals(cloudFirstText: 9)), route(.cloud, .complex))
        XCTAssertEqual(decider.decide("Latest scores", signals: signals(cloudFirstText: 9)), route(.cloud, .freshFacts, fallback: .local))
        XCTAssertEqual(decider.decide("Set an alarm", signals: signals(cloudFirstText: 9)), route(.cloud, .deviceAction, fallback: .local))
    }

    func testRow9ComplexGoesToTheCloudWithoutFallback() {
        XCTAssertEqual(decider.decide("Help me plan a trip to Japan", signals: signals(preferLocal: true)), route(.cloud, .complex))
        XCTAssertEqual(decider.decide("优缺点是什么", signals: signals(preferLocal: true, voice: true)), route(.cloud, .complex))
    }

    func testRow10PreferLocal() {
        XCTAssertEqual(decider.decide("Tell me a joke", signals: signals(preferLocal: true)), route(.local, .localFirst, fallback: .cloud))
        XCTAssertEqual(decider.decide("Tell me a joke", signals: signals(preferLocal: true, localReady: false)), route(.cloud, .cloudDefault, fallback: .local))
        // Typed small talk with a preference for on-device goes local, but as localFirst.
        XCTAssertEqual(decider.decide("Thanks", signals: signals(preferLocal: true)), route(.local, .localFirst, fallback: .cloud))
    }

    func testRow11CloudDefault() {
        XCTAssertEqual(decider.decide("Tell me a joke", signals: signals()), route(.cloud, .cloudDefault, fallback: .local))
        XCTAssertEqual(decider.decide("Tell me a joke", signals: signals(localAvailable: false, localReady: false)), route(.cloud, .cloudDefault))
        XCTAssertEqual(decider.decide("给我讲个故事", signals: signals(localReady: false)), route(.cloud, .cloudDefault, fallback: .local))
    }

    // MARK: Deep mode rules

    func testVoiceNeverGoesDeepAutomatically() {
        let decision = decider.decide("Compare these two laptops", signals: signals(voice: true, deepMode: .automatic))
        XCTAssertEqual(decision, route(.cloud, .complex))
        XCTAssertEqual(decision.mode, .standard)
        // An explicit request by voice still goes deep.
        XCTAssertEqual(decider.decide("Think hard about this", signals: signals(voice: true)), deep(.deepRequested, fallback: .local))
    }

    func testExhaustedBudgetRoutesStandard() {
        XCTAssertEqual(decider.decide("Tell me about Rome", signals: signals(deepRequested: true, deepBudgetLeft: false)), route(.cloud, .deepBudgetExhausted, fallback: .local))
        XCTAssertEqual(decider.decide("Think hard: compare these plans", signals: signals(deepBudgetLeft: false)), route(.cloud, .deepBudgetExhausted))
        XCTAssertEqual(decider.decide("Think hard about a joke", signals: signals(preferLocal: true, deepBudgetLeft: false)), route(.local, .deepBudgetExhausted, fallback: .cloud))
        // A budget only matters when deep is wanted.
        XCTAssertEqual(decider.decide("Tell me a joke", signals: signals(deepBudgetLeft: false)), route(.cloud, .cloudDefault, fallback: .local))
    }

    // MARK: Cloud only

    func testCloudOnlyNeverRoutesLocal() {
        var cloudOnly = signals(preferLocal: true, voice: true, cloudFirstText: 9)
        cloudOnly.policy = .cloudOnly
        for utterance in ["Thanks!", "Set a timer", "Tell me a joke", "Compare these", "Latest news", "你好"] {
            XCTAssertEqual(decider.decide(utterance, signals: cloudOnly), route(.cloud, .userChoice, fallback: .local), utterance)
        }
        cloudOnly.isOnline = false
        XCTAssertEqual(decider.decide("Hi", signals: cloudOnly), route(.cloud, .userChoice, fallback: .local))
        cloudOnly.cloudConfigured = false
        XCTAssertEqual(decider.decide("Hi", signals: cloudOnly), route(.cloud, .userChoice, fallback: .local))
    }

    func testCloudOnlyFallsBackLocallyOnlyWhenDownloaded() {
        var cloudOnly = signals(localAvailable: false, localReady: false)
        cloudOnly.policy = .cloudOnly
        XCTAssertEqual(decider.decide("Hi", signals: cloudOnly), route(.cloud, .userChoice))
        cloudOnly.deepRequested = true
        XCTAssertEqual(decider.decide("Hi", signals: cloudOnly), deep(.deepRequested))
    }

    func testCloudOnlyDeepMode() {
        var cloudOnly = signals(deepMode: .automatic)
        cloudOnly.policy = .cloudOnly
        XCTAssertEqual(decider.decide("Think carefully about this", signals: cloudOnly), deep(.deepRequested, fallback: .local))
        XCTAssertEqual(decider.decide("Compare these two laptops", signals: cloudOnly), deep(.deepAutomatic, fallback: .local))
        cloudOnly.inputIsVoice = true
        XCTAssertEqual(decider.decide("Compare these two laptops", signals: cloudOnly), route(.cloud, .userChoice, fallback: .local))
        cloudOnly.inputIsVoice = false
        cloudOnly.deepBudgetLeft = false
        XCTAssertEqual(decider.decide("Think carefully about this", signals: cloudOnly), route(.cloud, .deepBudgetExhausted, fallback: .local))
        cloudOnly.deepBudgetLeft = true
        cloudOnly.isOnline = false
        XCTAssertEqual(decider.decide("Think carefully about this", signals: cloudOnly), route(.cloud, .userChoice, fallback: .local))
    }

    func testDeepReason() {
        typealias Intents = IntentClassifier.Intents
        XCTAssertNil(RouteDecider.deepReason(intents: [.complex], signals: signals()))
        XCTAssertEqual(RouteDecider.deepReason(intents: [.complex], signals: signals(deepMode: .automatic)), .deepAutomatic)
        XCTAssertNil(RouteDecider.deepReason(intents: [.complex], signals: signals(voice: true, deepMode: .automatic)))
        XCTAssertEqual(RouteDecider.deepReason(intents: [.explicitDepth, .complex], signals: signals(deepMode: .automatic)), .deepRequested)
        XCTAssertNil(RouteDecider.deepReason(intents: [.explicitDepth], signals: signals(isOnline: false)))
        XCTAssertNil(RouteDecider.deepReason(intents: [], signals: signals(cloudConfigured: false, deepRequested: true)))
        XCTAssertEqual(decider.decide(intents: [.explicitDepth], signals: signals()), deep(.deepRequested, fallback: .local))
    }

    func testDefaultSignalsMatchTheDefaultSettings() {
        let defaults = RouteSignals()
        let settings = AssistantSettings()
        XCTAssertEqual(defaults.policy, .automatic(preferLocal: settings.preferOnDevice))
        XCTAssertEqual(defaults.deepMode, settings.deepMode)
        XCTAssertEqual(defaults.fastLocalSmallTalk, settings.fastLocalSmallTalk)
        XCTAssertTrue(defaults.isOnline)
        XCTAssertFalse(defaults.localReady)
    }
}
