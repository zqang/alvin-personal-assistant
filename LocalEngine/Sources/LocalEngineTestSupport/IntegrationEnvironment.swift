import Foundation
import LocalEngine
#if canImport(XCTest)
import XCTest
#endif

/// Switches for the integration suites, read from the environment. xcodebuild passes
/// `TEST_RUNNER_<NAME>` to the test process as `<NAME>`; both spellings are accepted.
public enum IntegrationEnvironment {
    /// `LOCAL_ENGINE_INTEGRATION=1`: run the suites that load real models.
    public static var enabled: Bool { flag("LOCAL_ENGINE_INTEGRATION") }
    /// `LOCAL_ENGINE_WOOF=1`: also run them on Underdog Woof 4B.
    public static var woof: Bool { flag("LOCAL_ENGINE_WOOF") }
    /// `LOCAL_ENGINE_OFFLINE=1`: use only models already in the Hugging Face cache.
    public static var offline: Bool { flag("LOCAL_ENGINE_OFFLINE") }
    /// `LOCAL_ENGINE_REPORT`: the Markdown file `EngineReport` appends to.
    public static var reportPath: String? { value("LOCAL_ENGINE_REPORT") }

    public static let woofRepo = "ConwayResearch/Underdog-Woof-4B-1.1"

    public struct Disabled: Error, CustomStringConvertible {
        public let description: String
    }

    /// Skips the calling test unless the integration suites are enabled.
    public static func requireEnabled() throws {
        guard enabled else {
            let reason = "Integration tests are off; set LOCAL_ENGINE_INTEGRATION=1 to run them."
            #if canImport(XCTest)
            throw XCTSkip(reason)
            #else
            throw Disabled(description: reason)
            #endif
        }
    }

    /// The snapshot directory of `spec` (`repo` or `repo@revision`) in the Hugging Face cache that
    /// `HF_HUB_CACHE` / `HF_HOME` point at. Without an explicit revision, the pin from
    /// `LocalEngine/ci-models.txt` is used, else `main`. Offline, nothing is downloaded.
    public static func snapshot(_ spec: String) async throws -> URL {
        let (repo, explicitRevision) = split(spec)
        let revision = explicitRevision ?? pinnedRevision(for: repo) ?? "main"
        return try await HubDownloader.usingEnvironmentCache().snapshot(id: repo, revision: revision, localFilesOnly: offline)
    }

    /// The revision `ci-models.txt` pins `repo` to, if any.
    public static func pinnedRevision(for repo: String) -> String? {
        for spec in ciModelSpecs() {
            let (name, revision) = split(spec)
            if name == repo { return revision }
        }
        return nil
    }

    /// The lines of `LocalEngine/ci-models.txt`, without comments and blank lines. Empty when the
    /// file isn't next to the sources (for example in an installed build).
    public static func ciModelSpecs() -> [String] {
        let file = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // LocalEngineTestSupport
            .deletingLastPathComponent()  // Sources
            .deletingLastPathComponent()  // LocalEngine
            .appendingPathComponent("ci-models.txt")
        guard let text = try? String(contentsOf: file, encoding: .utf8) else { return [] }
        return text.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
    }

    private static func split(_ spec: String) -> (repo: String, revision: String?) {
        guard let at = spec.lastIndex(of: "@") else { return (spec, nil) }
        let revision = String(spec[spec.index(after: at)...])
        return (String(spec[..<at]), revision.isEmpty ? nil : revision)
    }

    private static func value(_ name: String) -> String? {
        let environment = ProcessInfo.processInfo.environment
        guard let found = environment[name] ?? environment["TEST_RUNNER_" + name], !found.isEmpty else {
            return nil
        }
        return found
    }

    private static func flag(_ name: String) -> Bool {
        value(name) == "1"
    }
}
