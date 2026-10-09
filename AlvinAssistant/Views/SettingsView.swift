import AssistantKit
import AVFoundation
import Speech
import SwiftUI
import UIKit

struct SettingsView: View {
    @Bindable var store: SettingsStore
    @Environment(\.dismiss) private var dismiss

    @State private var anthropicKey = ""
    @State private var compatibleKey = ""
    @State private var openAIKey = ""
    @State private var previewSynthesizer = AVSpeechSynthesizer()

    private static let speechLocales: [String] = SFSpeechRecognizer.supportedLocales()
        .map { $0.identifier.replacingOccurrences(of: "_", with: "-") }
        .sorted { languageName($0) < languageName($1) }

    var body: some View {
        NavigationStack {
            Form {
                aboutYouSection
                modelSection
                assistantSections
                voiceSection
                VoiceLatencySection(store: store)
                conversationSection
                infoSection
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .onAppear {
            anthropicKey = store.secret(.anthropic)
            compatibleKey = store.secret(.compatible)
            openAIKey = store.secret(.openAI)
        }
        .onChange(of: store.settings) {
            store.save()
            // A different model, draft model or kernel setting unloads the model; the engine's
            // other options change on the running engine.
            LocalModelHost.shared.apply(store.settings)
        }
        .onChange(of: anthropicKey) { store.setSecret(anthropicKey, for: .anthropic) }
        .onChange(of: compatibleKey) { store.setSecret(compatibleKey, for: .compatible) }
        .onChange(of: openAIKey) { store.setSecret(openAIKey, for: .openAI) }
        .onChange(of: store.settings.usesQwenListening) {
            if store.settings.usesQwenListening { QwenListener.shared.load() } else { QwenListener.shared.unload(stopDownload: true) }
        }
    }

    // MARK: - Sections

    private var aboutYouSection: some View {
        Section {
            TextField("Your name", text: $store.settings.userName)
                .textContentType(.givenName)
            TextField("Anything the assistant should know about you, or how it should reply", text: $store.settings.customInstructions, axis: .vertical)
                .lineLimit(2...6)
        } header: {
            Text("About you")
        } footer: {
            Text("For example: “I live in Singapore and speak English and Mandarin. Keep answers short.”")
        }
    }

    private var modelSection: some View {
        Section {
            Picker("Provider", selection: $store.settings.provider) {
                Text("Claude").tag(AssistantSettings.Provider.anthropic)
                Text("OpenAI-compatible").tag(AssistantSettings.Provider.openAICompatible)
                Text("On this iPhone").tag(AssistantSettings.Provider.onDevice)
            }

            switch store.settings.provider {
            case .anthropic:
                SecureField("Anthropic API key", text: $anthropicKey)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                Picker("Model", selection: $store.settings.claudeModel) {
                    ForEach(ClaudeModelCatalog.options) { option in
                        Text(option.displayName).tag(option.id)
                    }
                    if !ClaudeModelCatalog.options.contains(where: { $0.id == store.settings.claudeModel }) {
                        Text(store.settings.claudeModel).tag(store.settings.claudeModel)
                    }
                }
                LabeledContent("Model ID") {
                    TextField(ClaudeModelCatalog.defaultModelID, text: $store.settings.claudeModel)
                        .multilineTextAlignment(.trailing)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }
                Picker("Thinking effort", selection: $store.settings.effort) {
                    Text("Low").tag("low")
                    Text("Medium").tag("medium")
                    Text("High").tag("high")
                }
                Toggle("Web search", isOn: $store.settings.webSearchEnabled)
            case .openAICompatible:
                Menu("Fill in a preset") {
                    ForEach(CompatibleServices.presets) { preset in
                        Button(preset.name) {
                            store.settings.compatibleBaseURL = preset.baseURL
                        }
                    }
                }
                TextField("Base URL", text: $store.settings.compatibleBaseURL)
                    .keyboardType(.URL)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                TextField(compatibleModelHint, text: $store.settings.compatibleModel)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                SecureField("API key", text: $compatibleKey)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
            case .onDevice:
                onDeviceRows
            }
        } header: {
            Text("Model")
        } footer: {
            switch store.settings.provider {
            case .anthropic:
                Text("Get a key at console.anthropic.com. Low effort answers fastest, which suits voice. Keys are stored in the iOS Keychain on this device.")
            case .openAICompatible:
                Text("Works with any service that offers an OpenAI-style chat completions API, such as Doubao on Volcengine Ark, DeepSeek, or OpenAI. Web search is available with Claude only.")
            case .onDevice:
                Text("Replies are generated on this iPhone and work offline once the model is downloaded (Wi-Fi only). There's no web search, and answers are simpler than Claude's. Best on iPhone 15 Pro or newer.")
            }
        }
    }

    /// The on-device model to download and load. Speculative decoding, the self-test and the
    /// benchmark are in the on-device engine section.
    @ViewBuilder
    private var onDeviceRows: some View {
        let option = LocalModelCatalog.option(for: store.settings.localModelID)
        Picker("Model", selection: $store.settings.localModelID) {
            ForEach(LocalModelCatalog.options) { option in
                Text(option.displayName).tag(option.id)
            }
        }
        if let option {
            Text(option.note)
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        Text(localStatus)
            .font(.footnote)
            .foregroundStyle(.secondary)
        if LocalModelHost.shared.status == .off || isLocalFailure {
            Button("Download and load now") { LocalModelHost.shared.prepare(store.settings) }
        }
    }

    /// Routing, the on-device model and engine, tools and deep mode, for the providers that have
    /// them.
    @ViewBuilder
    private var assistantSections: some View {
        if store.settings.provider == .anthropic {
            RoutingSettingsSection(store: store)
        }
        if routesAutomatically {
            onDeviceModelSection
        }
        if usesOnDeviceModel {
            OnDeviceEngineSection(store: store)
        }
        if store.settings.provider != .openAICompatible {
            ToolsSettingsSection(store: store)
        }
        if store.settings.provider == .anthropic {
            DeepSettingsSection(store: store)
        }
    }

    /// The on-device model for automatic routing, which answers when Claude can't or shouldn't.
    private var onDeviceModelSection: some View {
        Section {
            onDeviceRows
        } header: {
            Text("On-device model")
        } footer: {
            Text("Answers when you're offline, and the requests routing sends to this iPhone while it's loaded. It downloads over Wi-Fi only; until it's downloaded, Claude answers everything.")
        }
    }

    /// Automatic routing is on, with Claude as the provider.
    private var routesAutomatically: Bool {
        store.settings.provider == .anthropic && store.settings.routingMode == .automatic
    }

    /// The on-device model answers some or all replies.
    private var usesOnDeviceModel: Bool {
        store.settings.provider == .onDevice || routesAutomatically
    }

    private var voiceSection: some View {
        Section {
            Picker("I speak", selection: $store.settings.speechLocale) {
                ForEach(Self.speechLocales, id: \.self) { identifier in
                    Text(Self.languageName(identifier)).tag(identifier)
                }
            }
            .pickerStyle(.navigationLink)

            Toggle("Qwen3-ASR listening", isOn: $store.settings.qwenListening)
            if store.settings.qwenListening {
                Text(qwenStatus)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Picker("Voice engine", selection: $store.settings.voiceEngine) {
                Text("Built-in").tag(AssistantSettings.VoiceEngine.apple)
                Text("OpenAI").tag(AssistantSettings.VoiceEngine.openAI)
            }
            .pickerStyle(.segmented)

            switch store.settings.voiceEngine {
            case .apple:
                Picker("Voice", selection: $store.settings.appleVoiceIdentifier) {
                    Text("Automatic (best installed)").tag("")
                    ForEach(VoiceCatalog.voices(forLanguage: store.settings.speechLocale), id: \.identifier) { voice in
                        Text("\(voice.name) · \(VoiceCatalog.qualityLabel(voice))").tag(voice.identifier)
                    }
                }
                .pickerStyle(.navigationLink)
                VStack(alignment: .leading) {
                    Text("Speaking rate: \(store.settings.speechRate, format: .number.precision(.fractionLength(2)))×")
                    Slider(value: $store.settings.speechRate, in: 0.7...1.4, step: 0.05)
                }
                Button("Preview voice") { previewVoice() }
            case .openAI:
                SecureField("OpenAI API key", text: $openAIKey)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                Picker("Voice", selection: $store.settings.openAIVoice) {
                    ForEach(OpenAISpeech.voices, id: \.self) { voice in
                        Text(voice.capitalized).tag(voice)
                    }
                }
                LabeledContent("Model") {
                    TextField("gpt-4o-mini-tts", text: $store.settings.openAISpeechModel)
                        .multilineTextAlignment(.trailing)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }
            }
        } header: {
            Text("Voice")
        } footer: {
            switch store.settings.voiceEngine {
            case .apple:
                Text("For the most natural sound, download an Enhanced or Premium voice in Settings › Accessibility › Read & Speak › Voices.")
            case .openAI:
                Text("OpenAI voices sound more natural and cost a little per minute of speech. If a request fails, the built-in voice takes over.")
            }
        }
    }

    private var conversationSection: some View {
        Section {
            VStack(alignment: .leading) {
                Text("Wait after I stop talking: \(store.settings.endOfTurnDelay, format: .number.precision(.fractionLength(1))) s")
                Slider(value: $store.settings.endOfTurnDelay, in: 0.5...2.0, step: 0.1)
            }
            Toggle("Interrupt by talking", isOn: $store.settings.voiceInterruptions)
            Toggle("Echo cancellation", isOn: $store.settings.echoCancellation)
        } header: {
            Text("Conversation")
        } footer: {
            Text("A longer wait lets you pause mid-thought. Echo cancellation keeps the assistant from hearing itself on the speaker; turn it off only if your voice sounds muffled with headphones.")
        }
    }

    private var infoSection: some View {
        Section {
            LabeledContent("Version", value: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0")
        } footer: {
            Text("Speech is recognized on your iPhone. Your messages go to the model provider you choose above.")
        }
    }

    // MARK: - Helpers

    private var qwenStatus: String {
        guard store.settings.usesQwenListening else {
            return "Qwen3-ASR is used for English and Chinese (China mainland) only."
        }
        switch QwenListener.shared.status {
        case .off:
            return "Transcribes each finished turn again on this iPhone for better accuracy, while the app is open. Loads when you start voice mode."
        case .loading(let progress):
            return progress < 1 ? "Downloading about 1 GB over Wi-Fi… \(Int(progress * 100))%. Keep the app open." : "Loading…"
        case .ready:
            return "Ready."
        case .failed(let message):
            return message
        }
    }

    private var isLocalFailure: Bool {
        if case .failed = LocalModelHost.shared.status { return true }
        return false
    }

    private var localStatus: String {
        let host = LocalModelHost.shared
        switch host.status {
        case .off:
            return host.isDownloaded(store.settings.localModelID)
                ? "Downloaded. Loads when it's first needed."
                : "Downloads and loads the first time you use it."
        case .loading(let progress):
            return progress < 1 ? "Downloading over Wi-Fi… \(Int(progress * 100))%. Keep the app open." : "Loading…"
        case .ready:
            return "Ready."
        case .failed(let message):
            return message
        }
    }

    private var compatibleModelHint: String {
        CompatibleServices.presets.first { $0.baseURL == store.settings.compatibleBaseURL }?.modelHint ?? "Model ID"
    }

    private func previewVoice() {
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio)
        try? AVAudioSession.sharedInstance().setActive(true)
        let settings = store.settings
        let utterance = AVSpeechUtterance(string: Self.previewSentence(for: settings.speechLocale, name: settings.userName))
        utterance.voice = settings.appleVoiceIdentifier.isEmpty
            ? VoiceCatalog.bestVoice(forLanguage: settings.speechLocale)
            : AVSpeechSynthesisVoice(identifier: settings.appleVoiceIdentifier)
        utterance.rate = Float(settings.speechRate) * AVSpeechUtteranceDefaultSpeechRate
        previewSynthesizer.stopSpeaking(at: .immediate)
        previewSynthesizer.speak(utterance)
    }

    private static func previewSentence(for locale: String, name: String) -> String {
        let name = name.trimmed
        switch VoiceCatalog.baseLanguage(locale) {
        case "zh": return name.isEmpty ? "你好！有什么可以帮你的吗？" : "\(name)，你好！有什么可以帮你的吗？"
        case "ja": return "こんにちは！何かお手伝いできることはありますか？"
        case "ko": return "안녕하세요! 무엇을 도와드릴까요?"
        case "ms": return "Hai! Ada apa yang boleh saya bantu?"
        default: return name.isEmpty ? "Hi! What can I help you with today?" : "Hi \(name)! What can I help you with today?"
        }
    }

    private static func languageName(_ identifier: String) -> String {
        Locale.current.localizedString(forIdentifier: identifier) ?? identifier
    }
}
