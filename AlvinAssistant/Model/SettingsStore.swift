import AssistantKit
import Foundation
import Observation
import Security
import Speech

enum SecretKey: String, CaseIterable {
    case anthropic = "anthropic-api-key"
    case compatible = "compatible-api-key"
    case openAI = "openai-api-key"
}

/// Preferences (UserDefaults) and API keys (Keychain), plus what the app learns as it runs: the
/// deep-mode replies used today and how long each engine takes to its first text.
@MainActor
@Observable
final class SettingsStore {
    /// Bind to this directly; call `save()` after changes.
    var settings: AssistantSettings
    private(set) var secrets: [SecretKey: String]
    /// Deep-mode replies started today (`recordDeepRun()`), against `settings.deepDailyLimit`.
    private(set) var deepUsage: DeepBudget
    /// Times to first text per engine and mode (`recordFirstText(engine:mode:seconds:)`), for
    /// routing.
    private(set) var latency: LatencyEstimator

    private let defaults: UserDefaults
    private static let settingsKey = "assistant.settings"
    private static let deepUsageKey = "assistant.deepUsage"
    private static let latencyKey = "assistant.latency"

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        if let data = defaults.data(forKey: Self.settingsKey),
           let saved = try? JSONDecoder().decode(AssistantSettings.self, from: data) {
            settings = saved
        } else {
            var initial = AssistantSettings()
            initial.speechLocale = Self.defaultSpeechLocale()
            settings = initial
        }
        var loaded: [SecretKey: String] = [:]
        for key in SecretKey.allCases {
            loaded[key] = Keychain.read(account: key.rawValue) ?? ""
        }
        secrets = loaded
        deepUsage = Self.load(DeepBudget.self, key: Self.deepUsageKey, from: defaults) ?? DeepBudget()
        latency = Self.load(LatencyEstimator.self, key: Self.latencyKey, from: defaults) ?? LatencyEstimator()
    }

    func save() {
        if let data = try? JSONEncoder().encode(settings) {
            defaults.set(data, forKey: Self.settingsKey)
        }
    }

    // MARK: Deep mode and latency

    /// Counts one deep-mode reply for today (in the current time zone).
    func recordDeepRun(now: Date = Date()) {
        deepUsage.record(now: now, timeZone: .current)
        persist(deepUsage, key: Self.deepUsageKey)
    }

    /// Deep-mode replies still allowed today.
    func deepRunsLeft(now: Date = Date()) -> Int {
        deepUsage.remaining(limit: settings.deepDailyLimit, now: now, timeZone: .current)
    }

    /// Deep-mode replies started today.
    func deepRunsToday(now: Date = Date()) -> Int {
        deepUsage.usedToday(now: now, timeZone: .current)
    }

    /// Adds one measured time from a reply's start (or its fallback's) to its first text.
    func recordFirstText(engine: ReplyEngine, mode: ReplyMode, seconds: TimeInterval, now: Date = Date()) {
        latency.record(engine: engine, mode: mode, firstText: seconds, at: now)
        persist(latency, key: Self.latencyKey)
    }

    private func persist<Value: Encodable>(_ value: Value, key: String) {
        if let data = try? JSONEncoder().encode(value) {
            defaults.set(data, forKey: key)
        }
    }

    private static func load<Value: Decodable>(_ type: Value.Type, key: String, from defaults: UserDefaults) -> Value? {
        guard let data = defaults.data(forKey: key) else { return nil }
        return try? JSONDecoder().decode(type, from: data)
    }

    func secret(_ key: SecretKey) -> String {
        secrets[key] ?? ""
    }

    func setSecret(_ value: String, for key: SecretKey) {
        let trimmed = value.trimmed
        guard trimmed != secret(key) else { return }
        secrets[key] = trimmed
        if trimmed.isEmpty {
            Keychain.delete(account: key.rawValue)
        } else {
            Keychain.save(trimmed, account: key.rawValue)
        }
    }

    /// The device language if speech recognition supports it, else US English.
    private static func defaultSpeechLocale() -> String {
        let supported = Set(SFSpeechRecognizer.supportedLocales().map { $0.identifier.replacingOccurrences(of: "_", with: "-") })
        for language in Locale.preferredLanguages {
            if supported.contains(language) { return language }
            if language.hasPrefix("zh-Hans") { return "zh-CN" }
            if language.hasPrefix("zh-Hant") { return language.hasSuffix("HK") ? "zh-HK" : "zh-TW" }
        }
        return "en-US"
    }
}

/// Minimal Keychain wrapper for generic passwords.
enum Keychain {
    private static let service = Bundle.main.bundleIdentifier ?? "AlvinAssistant"

    static func read(account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func save(_ value: String, account: String) {
        let item: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let attributes: [String: Any] = [
            kSecValueData as String: Data(value.utf8),
            // Readable while locked after first unlock, so voice chat keeps working with the screen off.
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly,
        ]
        let status = SecItemUpdate(item as CFDictionary, attributes as CFDictionary)
        if status == errSecItemNotFound {
            SecItemAdd(item.merging(attributes) { _, new in new } as CFDictionary, nil)
        }
    }

    static func delete(account: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
    }
}
