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

/// Preferences (UserDefaults) and API keys (Keychain).
@MainActor
@Observable
final class SettingsStore {
    /// Bind to this directly; call `save()` after changes.
    var settings: AssistantSettings
    private(set) var secrets: [SecretKey: String]

    private let defaults: UserDefaults
    private static let settingsKey = "assistant.settings"

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
    }

    func save() {
        if let data = try? JSONEncoder().encode(settings) {
            defaults.set(data, forKey: Self.settingsKey)
        }
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
