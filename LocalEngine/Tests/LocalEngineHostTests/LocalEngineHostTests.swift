import AssistantKit
@testable import LocalEngineHost
import XCTest

/// The host as another app sets it up (Roam): its own profile and tools, none of this app's other
/// models. Nothing here loads a model or touches the network.
@MainActor
final class LocalEngineHostTests: XCTestCase {
    func testTheDefaultsNeedNoAppTies() {
        var settings = AssistantSettings()
        settings.userName = "Ann"
        let prompt = LocalEngineHost.prewarmPrompt(settings)
        let neutral = AssistantProfile()
        let base = PromptBuilder.systemPrompt(userName: "Ann", customInstructions: settings.customInstructions, profile: neutral)
        XCTAssertEqual(prompt.system, PromptBuilder.localSystemPrompt(base: base, handoffAvailable: false, profile: neutral))
        XCTAssertEqual(prompt.tools, [])
        XCTAssertTrue(LocalEngineHost.isOnline())
        XCTAssertEqual(LocalEngineHost.memoryAdvice, "")
        XCTAssertEqual(LocalEngineHost.localize("Loading the on-device model"), "Loading the on-device model")
    }

    func testThePrewarmRendersTheHostsProfileAndTools() {
        let savedProfile = LocalEngineHost.profile
        let savedProvider = LocalEngineHost.systemProvider
        defer {
            LocalEngineHost.profile = savedProfile
            LocalEngineHost.systemProvider = savedProvider
        }
        let roam = AssistantProfile(persona: { _ in "You are Roam." }, localToolHint: "Use your tools to read the open trip's plan.")
        let tools = [ToolDefinition(name: "trip_plan", description: "The open trip's plan.", inputSchema: ["type": "object"]), roam.handoffTool.definition]
        LocalEngineHost.profile = roam
        LocalEngineHost.systemProvider = { _ in ("You are Roam.", tools) }

        let prompt = LocalEngineHost.prewarmPrompt(AssistantSettings())
        XCTAssertEqual(prompt.tools.map(\.name), ["trip_plan", HandoffTool.name])
        // Exactly what an engine reply with a handoff sends, so the prefix made resident is reused.
        XCTAssertEqual(prompt.system, PromptBuilder.localSystemPrompt(base: "You are Roam.", handoffAvailable: true, profile: roam))
        XCTAssertTrue(prompt.system.hasSuffix(roam.localToolHint + " " + roam.localHandoffHint))
    }

    /// Also: what the reply shows goes through `localize`, so it can be in the app's language.
    func testAReplyForAModelOutsideTheCatalogFailsWithoutLoading() async {
        let savedBeforeLoad = LocalEngineHost.beforeLoad
        let savedLocalize = LocalEngineHost.localize
        defer {
            LocalEngineHost.beforeLoad = savedBeforeLoad
            LocalEngineHost.localize = savedLocalize
        }
        var loads = 0
        LocalEngineHost.beforeLoad = { loads += 1 }
        LocalEngineHost.localize = { "[zh] " + $0 }
        var settings = AssistantSettings()
        settings.localModelID = "nobody/no-such-model"
        let provider = LocalProvider(settings: settings, tools: nil, handoffAvailable: false, commitGate: nil)

        var events: [AssistantEvent] = []
        do {
            for try await event in provider.streamEvents(system: "", turns: [ChatTurn(role: .user, text: "Hi")]) {
                events.append(event)
            }
            XCTFail("Expected the reply to fail")
        } catch {
            XCTAssertEqual(error as? AssistantError, .missingConfiguration("[zh] The on-device model isn't loaded. Try again."))
        }
        XCTAssertEqual(events, [.cue(.loadingModel), .reply(.activity("[zh] Loading the on-device model"))])
        XCTAssertEqual(loads, 0)
        XCTAssertEqual(LocalEngineHost.shared.status, .off)
    }
}
