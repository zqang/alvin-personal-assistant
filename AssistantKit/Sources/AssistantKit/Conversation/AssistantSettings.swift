import Foundation

/// Which generator answers on-device replies.
public enum LocalEngineMode: String, Codable, CaseIterable, Identifiable, Sendable {
    /// The Alvin engine once it has passed its on-device self-test for this model and build;
    /// MLX's stock chat session until then.
    case automatic
    /// Always the Alvin engine.
    case alvin
    /// Always MLX's stock chat session.
    case stock

    public var id: String { rawValue }
}

/// When the on-device engine checks drafted tokens in blocks (speculative decoding).
public enum LocalSpeculationMode: String, Codable, CaseIterable, Identifiable, Sendable {
    case off
    /// Only inside tool calls, or when the reply is copying a long stretch of the prompt.
    case toolsOnly
    /// Whenever the measured cost curve says it pays.
    case automatic

    public var id: String { rawValue }
}

/// Whether each request goes to the chosen provider or is routed per request.
public enum RoutingMode: String, Codable, CaseIterable, Identifiable, Sendable {
    /// Always the provider picked in Settings.
    case single
    /// Cloud or on-device per request, by connectivity, intent and latency.
    case automatic

    public var id: String { rawValue }
}

/// When replies use deep mode (more thinking, or several workers).
public enum DeepModeSetting: String, Codable, CaseIterable, Identifiable, Sendable {
    case off
    /// When the user asks for it.
    case onRequest
    /// Also for typed requests that look complex. Voice never goes deep automatically.
    case automatic

    public var id: String { rawValue }
}

/// How deep mode spends its extra effort.
public enum DeepStrategy: String, Codable, CaseIterable, Identifiable, Sendable {
    /// One call at high effort.
    case single
    /// A researcher, a reasoner and a critic, then a merged answer.
    case parallel

    public var id: String { rawValue }
}

/// Whether a voice reply may start before the user's turn is final.
public enum EarlyStartSetting: String, Codable, CaseIterable, Identifiable, Sendable {
    case off
    /// Only when the on-device model answers.
    case onDeviceOnly
    case automatic

    public var id: String { rawValue }
}

/// User preferences (everything except API keys, which live in the Keychain).
public struct AssistantSettings: Codable, Equatable, Sendable {
    public enum Provider: String, Codable, CaseIterable, Identifiable, Sendable {
        case anthropic
        case openAICompatible
        /// A model running on the iPhone through MLX.
        case onDevice

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
    public var localModelID = LocalModelCatalog.defaultModelID
    /// Speeds up models that have a draft model by checking several drafted tokens per step.
    public var localSpeculativeDecoding = true

    // On-device engine
    public var localEngineMode: LocalEngineMode = .automatic
    public var localSpeculation: LocalSpeculationMode = .automatic
    /// Keeps the processed system prompt on disk so a new session starts without re-reading it.
    public var localPrefixCache = true
    /// Custom GPU kernels for block verification; used only where they measure faster.
    public var localFastKernels = false
    /// The model's own multi-token-prediction drafter, when its weights include one.
    public var localMTP = true

    // Routing
    public var routingMode: RoutingMode = .single
    public var preferOnDevice = false
    /// With automatic routing, short spoken small talk is answered on device.
    public var fastLocalSmallTalk = true

    // Tools
    public var deviceToolsEnabled = true
    /// Names of tools the user turned off.
    public var disabledTools: [String] = []

    // Deep mode
    public var deepMode: DeepModeSetting = .onRequest
    public var deepStrategy: DeepStrategy = .single
    /// Deep replies allowed per day.
    public var deepDailyLimit = 10
    /// Model for deep-mode workers; empty means the reply model.
    public var deepWorkerModel = ""

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
    /// Re-transcribes each finished voice turn on-device with Qwen3-ASR before it is sent.
    public var qwenListening = false
    /// Short spoken phrases ("Let me check.") while a voice reply is being prepared.
    public var spokenCues = true
    /// Seconds without a word before the "One moment." filler plays.
    public var fillerDelay = 1.8
    /// A short tone when the assistant has taken the user's turn.
    public var turnChime = false
    /// Lowers the assistant's voice while the user may be interrupting.
    public var bargeInDucking = true
    public var earlyReplyStart: EarlyStartSetting = .automatic

    public init() {}

    /// Qwen3-ASR listening is on and covers the recognition language.
    public var usesQwenListening: Bool {
        qwenListening && FinalTranscript.qwenLanguage(forLocale: speechLocale) != nil
    }

    private enum CodingKeys: String, CodingKey {
        case provider, claudeModel, effort, webSearchEnabled, compatibleBaseURL, compatibleModel
        case localModelID, localSpeculativeDecoding
        case localEngineMode, localSpeculation, localPrefixCache, localFastKernels, localMTP
        case routingMode, preferOnDevice, fastLocalSmallTalk
        case deviceToolsEnabled, disabledTools
        case deepMode, deepStrategy, deepDailyLimit, deepWorkerModel
        case userName, customInstructions
        case speechLocale, voiceEngine, appleVoiceIdentifier, speechRate, openAIVoice, openAISpeechModel
        case endOfTurnDelay, voiceInterruptions, echoCancellation, qwenListening
        case spokenCues, fillerDelay, turnChime, bargeInDucking, earlyReplyStart
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
        localModelID = value(.localModelID, defaults.localModelID)
        localSpeculativeDecoding = value(.localSpeculativeDecoding, defaults.localSpeculativeDecoding)
        localEngineMode = value(.localEngineMode, defaults.localEngineMode)
        localSpeculation = value(.localSpeculation, defaults.localSpeculation)
        localPrefixCache = value(.localPrefixCache, defaults.localPrefixCache)
        localFastKernels = value(.localFastKernels, defaults.localFastKernels)
        localMTP = value(.localMTP, defaults.localMTP)
        routingMode = value(.routingMode, defaults.routingMode)
        preferOnDevice = value(.preferOnDevice, defaults.preferOnDevice)
        fastLocalSmallTalk = value(.fastLocalSmallTalk, defaults.fastLocalSmallTalk)
        deviceToolsEnabled = value(.deviceToolsEnabled, defaults.deviceToolsEnabled)
        disabledTools = value(.disabledTools, defaults.disabledTools)
        deepMode = value(.deepMode, defaults.deepMode)
        deepStrategy = value(.deepStrategy, defaults.deepStrategy)
        deepDailyLimit = value(.deepDailyLimit, defaults.deepDailyLimit)
        deepWorkerModel = value(.deepWorkerModel, defaults.deepWorkerModel)
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
        qwenListening = value(.qwenListening, defaults.qwenListening)
        spokenCues = value(.spokenCues, defaults.spokenCues)
        fillerDelay = value(.fillerDelay, defaults.fillerDelay)
        turnChime = value(.turnChime, defaults.turnChime)
        bargeInDucking = value(.bargeInDucking, defaults.bargeInDucking)
        earlyReplyStart = value(.earlyReplyStart, defaults.earlyReplyStart)
    }
}
