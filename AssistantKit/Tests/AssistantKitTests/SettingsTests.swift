import XCTest
@testable import AssistantKit

final class SettingsTests: XCTestCase {
    private func decode(_ json: String) throws -> AssistantSettings {
        try JSONDecoder().decode(AssistantSettings.self, from: Data(json.utf8))
    }

    func testSettingsFromAnOlderBuildGetEveryNewDefault() throws {
        let settings = try decode(#"{"provider":"anthropic"}"#)
        XCTAssertEqual(settings, AssistantSettings())

        XCTAssertEqual(settings.localEngineMode, .automatic)
        XCTAssertEqual(settings.localSpeculation, .automatic)
        XCTAssertTrue(settings.localPrefixCache)
        XCTAssertFalse(settings.localFastKernels)
        XCTAssertTrue(settings.localMTP)
        XCTAssertEqual(settings.routingMode, .single)
        XCTAssertFalse(settings.preferOnDevice)
        XCTAssertTrue(settings.fastLocalSmallTalk)
        XCTAssertTrue(settings.deviceToolsEnabled)
        XCTAssertEqual(settings.disabledTools, [])
        XCTAssertEqual(settings.deepMode, .onRequest)
        XCTAssertEqual(settings.deepStrategy, .single)
        XCTAssertEqual(settings.deepDailyLimit, 10)
        XCTAssertEqual(settings.deepWorkerModel, "")
        XCTAssertTrue(settings.spokenCues)
        XCTAssertEqual(settings.fillerDelay, 1.8)
        XCTAssertFalse(settings.turnChime)
        XCTAssertTrue(settings.bargeInDucking)
        XCTAssertEqual(settings.earlyReplyStart, .automatic)

        XCTAssertEqual(settings.provider, .anthropic)
        XCTAssertTrue(settings.localSpeculativeDecoding)
    }

    func testUnknownOrMistypedValuesFallBackToDefaults() throws {
        let json = """
        {"provider":"onDevice","localEngineMode":"turbo","localSpeculation":"always","routingMode":"smart",
         "deepMode":"sometimes","deepStrategy":"swarm","earlyReplyStart":"eager","disabledTools":"set_timer",
         "deepDailyLimit":"ten","fillerDelay":"slow","localPrefixCache":"yes","deepWorkerModel":7,
         "preferOnDevice":true,"turnChime":true}
        """
        let settings = try decode(json)
        let defaults = AssistantSettings()
        XCTAssertEqual(settings.localEngineMode, defaults.localEngineMode)
        XCTAssertEqual(settings.localSpeculation, defaults.localSpeculation)
        XCTAssertEqual(settings.routingMode, defaults.routingMode)
        XCTAssertEqual(settings.deepMode, defaults.deepMode)
        XCTAssertEqual(settings.deepStrategy, defaults.deepStrategy)
        XCTAssertEqual(settings.earlyReplyStart, defaults.earlyReplyStart)
        XCTAssertEqual(settings.disabledTools, defaults.disabledTools)
        XCTAssertEqual(settings.deepDailyLimit, defaults.deepDailyLimit)
        XCTAssertEqual(settings.fillerDelay, defaults.fillerDelay)
        XCTAssertEqual(settings.localPrefixCache, defaults.localPrefixCache)
        XCTAssertEqual(settings.deepWorkerModel, defaults.deepWorkerModel)

        XCTAssertEqual(settings.provider, .onDevice, "valid values next to bad ones still load")
        XCTAssertTrue(settings.preferOnDevice)
        XCTAssertTrue(settings.turnChime)
    }

    func testEveryNewSettingRoundTrips() throws {
        var settings = AssistantSettings()
        settings.localEngineMode = .stock
        settings.localSpeculation = .toolsOnly
        settings.localPrefixCache = false
        settings.localFastKernels = true
        settings.localMTP = false
        settings.routingMode = .automatic
        settings.preferOnDevice = true
        settings.fastLocalSmallTalk = false
        settings.deviceToolsEnabled = false
        settings.disabledTools = ["set_timer", "create_event"]
        settings.deepMode = .automatic
        settings.deepStrategy = .parallel
        settings.deepDailyLimit = 3
        settings.deepWorkerModel = "claude-sonnet-5"
        settings.spokenCues = false
        settings.fillerDelay = 2.5
        settings.turnChime = true
        settings.bargeInDucking = false
        settings.earlyReplyStart = .onDeviceOnly
        XCTAssertNotEqual(settings, AssistantSettings())

        let data = try JSONEncoder().encode(settings)
        XCTAssertEqual(try JSONDecoder().decode(AssistantSettings.self, from: data), settings)

        let json = try JSONValue.parse(data)
        XCTAssertEqual(json["localEngineMode"]?.stringValue, "stock")
        XCTAssertEqual(json["earlyReplyStart"]?.stringValue, "onDeviceOnly")
        XCTAssertEqual(json["disabledTools"], ["set_timer", "create_event"])
    }

    func testPickerEnumsListEveryCase() {
        XCTAssertEqual(LocalEngineMode.allCases.map(\.id), ["automatic", "alvin", "stock"])
        XCTAssertEqual(LocalSpeculationMode.allCases.map(\.id), ["off", "toolsOnly", "automatic"])
        XCTAssertEqual(RoutingMode.allCases.map(\.id), ["single", "automatic"])
        XCTAssertEqual(DeepModeSetting.allCases.map(\.id), ["off", "onRequest", "automatic"])
        XCTAssertEqual(DeepStrategy.allCases.map(\.id), ["single", "parallel"])
        XCTAssertEqual(EarlyStartSetting.allCases.map(\.id), ["off", "onDeviceOnly", "automatic"])
        XCTAssertEqual(AssistantSettings.Provider.allCases.count, 3, "routing is a separate setting, not a provider")
    }
}
