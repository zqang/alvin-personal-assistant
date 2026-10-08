import AssistantKit
import Foundation
import MLX
import MLXLMCommon

/// The system prefix saved between launches (plan §4.8), so a cold start skips its prefill.
///
/// - One `<prefixKey>.safetensors` per prefix, written with `savePromptCache`: attention layers as
///   `KVCacheSimple`s holding the first p positions, recurrent layers as `MambaCache`s with the
///   `systemEnd` checkpoint's state. Metadata: prefix key, p, token hash, format version and
///   model id.
/// - Written to a temporary file, then renamed. At most `capacity` files (least recently used
///   go first); `pruneOtherModels` removes other models' files.
/// - `load` refuses a file whose key, length, token hash, format version, layer count or layer
///   classes don't match.
///
/// An index (`prefix-index.json`) records each file's model and last use. Not thread-safe:
/// engine queue only.
public final class PrefixCacheStore {
    public static let capacity = 3
    static let indexName = "prefix-index.json"

    public let directory: URL

    struct Entry: Codable, Equatable {
        var key: String
        var modelID: String
        var lastUsed: Date
    }

    public init(directory: URL) {
        self.directory = directory
    }

    public func url(for key: String) -> URL {
        directory.appendingPathComponent("\(key).safetensors")
    }

    public func contains(key: String) -> Bool {
        FileManager.default.fileExists(atPath: url(for: key).path)
    }

    /// The keys on disk, most recently used first.
    public var keys: [String] {
        readIndex().sorted { $0.lastUsed > $1.lastUsed }.map(\.key).filter(contains)
    }

    // MARK: Saving

    /// Saves the prefix `tokens` (`position` of them) of a live cache.
    ///
    /// - Parameters:
    ///   - snapshot: the `systemEnd` checkpoint, taken at `position`.
    ///   - kvState: each attention layer's `state` (`[keys, values]`, holding at least
    ///     `position` positions), by layer index.
    ///   - layout: the cache layout.
    public func save(
        snapshot: CacheSnapshot, position: Int, kvState: [Int: [MLXArray]], layout: CacheLayout,
        key: String, tokens: [Int], modelID: String
    ) throws {
        precondition(snapshot.position == position && tokens.count == position, "The snapshot, position and tokens disagree.")
        precondition(snapshot.slots.count == layout.recurrent.count, "The snapshot doesn't match the layout.")
        let count = layout.attention.count + layout.recurrent.count
        var layers = [KVCache?](repeating: nil, count: count)
        for index in layout.attention {
            guard let state = kvState[index], state.count == 2, state[0].dim(2) >= position else {
                throw PrefixCacheError.missingState(index)
            }
            let layer = KVCacheSimple()
            layer.state = [state[0][.ellipsis, ..<position, 0...], state[1][.ellipsis, ..<position, 0...]]
            layers[index] = layer
        }
        for (index, slot) in zip(layout.recurrent, snapshot.slots) {
            guard slot.0 != nil, slot.1 != nil else { throw PrefixCacheError.missingState(index) }
            let layer = MambaCache()
            layer[0] = slot.0
            layer[1] = slot.1
            layers[index] = layer
        }
        let cache = layers.compactMap { $0 }
        guard cache.count == count else { throw PrefixCacheError.missingState(-1) }

        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        Self.excludeFromBackup(directory)
        let temporary = directory.appendingPathComponent("\(key).\(UUID().uuidString).tmp.safetensors")
        let metadata = [
            "prefixKey": key,
            "position": String(position),
            "tokenHash": PrefixKey.tokenHash(tokens),
            "formatVersion": String(EngineInfo.formatVersion),
            "modelID": modelID,
        ]
        do {
            try savePromptCache(url: temporary, cache: cache, metadata: metadata)
            let destination = url(for: key)
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.moveItem(at: temporary, to: destination)
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }

        var entries = readIndex().filter { $0.key != key }
        entries.append(Entry(key: key, modelID: modelID, lastUsed: Date()))
        entries.sort { $0.lastUsed > $1.lastUsed }
        for evicted in entries.dropFirst(Self.capacity) {
            try? FileManager.default.removeItem(at: url(for: evicted.key))
        }
        writeIndex(Array(entries.prefix(Self.capacity)))
    }

    // MARK: Loading

    /// The saved prefix for `key`, if it holds exactly `expectedTokens` in `layout`; nil
    /// otherwise (a refused file is deleted).
    public func load(key: String, expectedTokens: [Int], layout: CacheLayout) -> [KVCache]? {
        let file = url(for: key)
        guard FileManager.default.fileExists(atPath: file.path) else { return nil }
        guard let loaded = try? loadPromptCache(url: file) else {
            discard(key)
            return nil
        }
        let (cache, metadata) = loaded
        let position = expectedTokens.count
        guard metadata["prefixKey"] == key,
              metadata["position"] == String(position),
              metadata["tokenHash"] == PrefixKey.tokenHash(expectedTokens),
              metadata["formatVersion"] == String(EngineInfo.formatVersion),
              CacheLayout.detect(cache) == layout,
              cache.count == layout.attention.count + layout.recurrent.count,
              layout.attention.allSatisfy({ cache[$0].offset == position })
        else {
            discard(key)
            return nil
        }
        touch(key)
        return cache
    }

    // MARK: Housekeeping

    /// Deletes the files of models other than `modelID`, and files the index doesn't know.
    public func pruneOtherModels(modelID: String) {
        let entries = readIndex()
        let kept = entries.filter { $0.modelID == modelID }
        for entry in entries where entry.modelID != modelID {
            try? FileManager.default.removeItem(at: url(for: entry.key))
        }
        let known = Set(kept.map { "\($0.key).safetensors" })
        if let files = try? FileManager.default.contentsOfDirectory(atPath: directory.path) {
            for file in files where file.hasSuffix(".safetensors") && !known.contains(file) {
                try? FileManager.default.removeItem(at: directory.appendingPathComponent(file))
            }
        }
        writeIndex(kept)
    }

    /// Deletes every saved prefix.
    public func removeAll() {
        for entry in readIndex() {
            try? FileManager.default.removeItem(at: url(for: entry.key))
        }
        writeIndex([])
    }

    private func discard(_ key: String) {
        try? FileManager.default.removeItem(at: url(for: key))
        writeIndex(readIndex().filter { $0.key != key })
    }

    private func touch(_ key: String) {
        var entries = readIndex()
        if let index = entries.firstIndex(where: { $0.key == key }) {
            entries[index].lastUsed = Date()
            writeIndex(entries)
        }
    }

    private var indexURL: URL {
        directory.appendingPathComponent(Self.indexName)
    }

    func readIndex() -> [Entry] {
        guard let data = try? Data(contentsOf: indexURL),
              let entries = try? JSONDecoder().decode([Entry].self, from: data)
        else { return [] }
        return entries
    }

    private func writeIndex(_ entries: [Entry]) {
        guard let data = try? JSONEncoder().encode(entries) else { return }
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try? data.write(to: indexURL, options: .atomic)
    }

    static func excludeFromBackup(_ url: URL) {
        #if canImport(Darwin)
        var url = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? url.setResourceValues(values)
        #endif
    }
}

public enum PrefixCacheError: Error, Equatable {
    /// A layer's state was missing or shorter than the prefix (`-1`: the layer count is off).
    case missingState(Int)
}
