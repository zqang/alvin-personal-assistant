import Foundation

/// User preferences (everything except API keys, which live in the Keychain).
public struct AssistantSettings: Codable, Equatable, Sendable {
    public enum Provider: String, Codable, CaseIterable, Identifiable, Sendable {
        case anthropic
        case openAICompatible

        public var id: String { rawValue }
    }

    public enum VoiceEngine: String, Codable, CaseIterable, Identifiable, Sendable {
        /// On-device `AVSpeechSynthesizer`: free and instant.
        case apple
        /// OpenAI text-to-speech: more natural, needs an OpenAI key.
        case openAI

        public var id: String { rawValue }
    }

    // Model
    public var provider: Provider = .anthropic
    public var claudeModel: String = ClaudeModelCatalog.defaultModelID
    public var effort: String = "low"
    public var webSearchEnabled = true
    public var compatibleBaseURL = CompatibleServices.presets[0].baseURL
    public var compatibleModel = ""

    // Personalization
    public var userName = ""
    public var customInstructions = ""

    // Voice
    /// Speech recognition locale, e.g. "en-US" or "zh-CN".
    public var speechLocale = "en-US"
    public var voiceEngine: VoiceEngine = .apple
    /// Empty means the best installed voice for the language.
    public var appleVoiceIdentifier = ""
    /// Multiplier on the default speaking rate.
    public var speechRate = 1.0
    public var openAIVoice = "coral"
    public var openAISpeechModel = "gpt-4o-mini-tts"
    /// Seconds of silence that end the user's turn.
    public var endOfTurnDelay = 0.9
    public var voiceInterruptions = true
    public var echoCancellation = true

    public init() {}

    private enum CodingKeys: String, CodingKey {
        case provider, claudeModel, effort, webSearchEnabled, compatibleBaseURL, compatibleModel
        case userName, customInstructions
        case speechLocale, voiceEngine, appleVoiceIdentifier, speechRate, openAIVoice, openAISpeechModel
        case endOfTurnDelay, voiceInterruptions, echoCancellation
    }

    /// Missing or unreadable keys keep their defaults, so settings saved by older builds still load.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        func value<T: Decodable>(_ key: CodingKeys, _ fallback: T) -> T {
            (try? container.decodeIfPresent(T.self, forKey: key)) ?? fallback
        }
        let defaults = AssistantSettings()
        provider = value(.provider, defaults.provider)
        claudeModel = value(.claudeModel, defaults.claudeModel)
        effort = value(.effort, defaults.effort)
        webSearchEnabled = value(.webSearchEnabled, defaults.webSearchEnabled)
        compatibleBaseURL = value(.compatibleBaseURL, defaults.compatibleBaseURL)
        compatibleModel = value(.compatibleModel, defaults.compatibleModel)
        userName = value(.userName, defaults.userName)
        customInstructions = value(.customInstructions, defaults.customInstructions)
        speechLocale = value(.speechLocale, defaults.speechLocale)
        voiceEngine = value(.voiceEngine, defaults.voiceEngine)
        appleVoiceIdentifier = value(.appleVoiceIdentifier, defaults.appleVoiceIdentifier)
        speechRate = value(.speechRate, defaults.speechRate)
        openAIVoice = value(.openAIVoice, defaults.openAIVoice)
        openAISpeechModel = value(.openAISpeechModel, defaults.openAISpeechModel)
        endOfTurnDelay = value(.endOfTurnDelay, defaults.endOfTurnDelay)
        voiceInterruptions = value(.voiceInterruptions, defaults.voiceInterruptions)
        echoCancellation = value(.echoCancellation, defaults.echoCancellation)
    }
}
