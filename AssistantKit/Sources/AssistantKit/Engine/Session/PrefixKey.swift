import Foundation

/// Identifies a system prefix: the cached tokens are valid only for the same model snapshot, engine
/// format, system prompt, tools and chat-template context. Used to match the live cache to a request
/// and to name the prefix saved on disk.
public enum PrefixKey {
    /// FNV-1a-64, as 16 lowercase hex digits, of the fields' UTF-8 bytes joined with U+001F.
    /// - Parameters:
    ///   - revision: the snapshot directory name of the downloaded weights.
    ///   - toolsJSON: the tools as rendered, canonically (see `canonicalToolsJSON`).
    ///   - context: the template context, canonically (see `canonicalContext`).
    public static func make(modelID: String, revision: String, system: String, toolsJSON: String, context: String, formatVersion: Int) -> String {
        let fields = [modelID, revision, system, toolsJSON, context, String(formatVersion)]
        return hex(fnv1a64(Array(fields.joined(separator: "\u{1F}").utf8)))
    }

    /// FNV-1a-64 of the tokens, each as 8 little-endian bytes, as 16 lowercase hex digits.
    public static func tokenHash<C: Collection>(_ tokens: C) -> String where C.Element == Int {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(tokens.count * 8)
        for token in tokens {
            withUnsafeBytes(of: Int64(token).littleEndian) { bytes.append(contentsOf: $0) }
        }
        return hex(fnv1a64(bytes))
    }

    /// Tools as compact JSON with sorted keys, in the order given (the order they render in):
    /// `[{"description":…,"input_schema":…,"name":…}, …]`.
    public static func canonicalToolsJSON(_ tools: [ToolDefinition]) -> String {
        let value = JSONValue.array(tools.map { tool in
            .object(["name": .string(tool.name), "description": .string(tool.description), "input_schema": tool.inputSchema])
        })
        guard let data = try? value.serialized() else { return "[]" }
        return String(decoding: data, as: UTF8.self)
    }

    /// A template context such as `["enable_thinking": false]` as `key=value` pairs sorted by key,
    /// joined with commas.
    public static func canonicalContext(_ context: [String: Bool]) -> String {
        context.keys.sorted().map { "\($0)=\(context[$0] == true)" }.joined(separator: ",")
    }

    static func fnv1a64(_ bytes: [UInt8]) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in bytes {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return hash
    }

    static func hex(_ value: UInt64) -> String {
        let digits = String(value, radix: 16)
        return String(repeating: "0", count: max(0, 16 - digits.count)) + digits
    }
}
