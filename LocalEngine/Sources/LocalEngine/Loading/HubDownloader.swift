import AssistantKit
import Foundation
import HuggingFace
import MLXLMCommon

/// Fetches model files through swift-huggingface, preferring files already on the device so a
/// downloaded model loads offline.
///
/// Copied from the app's `LocalModelHost.swift`; the app keeps its private copy until the engine
/// is wired in.
public struct HubDownloader: MLXLMCommon.Downloader {
    /// The files a model needs: weights, configuration, tokenizer and chat template.
    public static let patterns = ["*.safetensors", "*.json", "*.jinja"]

    public let client: HubClient

    public init(client: HubClient) {
        self.client = client
    }

    /// A downloader whose cache follows `HF_HUB_CACHE` / `HF_HOME` (the Python `huggingface_hub`
    /// layout), as the CI prefetch step fills it. For tests and tools, not the app.
    public static func usingEnvironmentCache(session: URLSession = .shared) -> HubDownloader {
        HubDownloader(client: HubClient(session: session, cache: HubCache(location: .environment)))
    }

    public func download(id: String, revision: String?, matching patterns: [String], useLatest: Bool, progressHandler: @Sendable @escaping (Progress) -> Void) async throws -> URL {
        let repo = try Self.repo(id)
        let revision = revision ?? "main"
        if !useLatest, let cached = try? await client.downloadSnapshot(of: repo, revision: revision, matching: patterns, localFilesOnly: true) {
            return cached
        }
        return try await client.downloadSnapshot(of: repo, revision: revision, matching: patterns, progressHandler: { progress in
            progressHandler(progress)
        })
    }

    public func download(id: String, progress: @escaping @MainActor @Sendable (Double) -> Void) async throws -> URL {
        try await download(id: id, revision: nil, matching: Self.patterns, useLatest: false) { value in
            let fraction = value.fractionCompleted
            Task { @MainActor in progress(fraction) }
        }
    }

    /// The local snapshot directory of `id` at `revision`. With `localFilesOnly` nothing is
    /// fetched and a model missing from the cache throws.
    public func snapshot(id: String, revision: String = "main", localFilesOnly: Bool) async throws -> URL {
        let repo = try Self.repo(id)
        return try await client.downloadSnapshot(of: repo, revision: revision, matching: Self.patterns, localFilesOnly: localFilesOnly)
    }

    private static func repo(_ id: String) throws -> Repo.ID {
        guard let repo = Repo.ID(rawValue: id) else {
            throw AssistantError.missingConfiguration("“\(id)” isn't a valid model ID.")
        }
        return repo
    }
}
