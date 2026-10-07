import Foundation

/// The Markdown report CI prints between `=== BEGIN ENGINE REPORT ===` and
/// `=== END ENGINE REPORT ===`. Tests append measurements and facts to it; without a report path
/// the text only goes to standard output.
public enum EngineReport {
    private static let lock = NSLock()

    /// Appends `markdown` (followed by a newline) to the file at `LOCAL_ENGINE_REPORT`, creating
    /// it if needed, and echoes it to standard output.
    public static func append(_ markdown: String) {
        let text = markdown.hasSuffix("\n") ? markdown : markdown + "\n"
        print(text, terminator: "")
        guard let path = IntegrationEnvironment.reportPath else { return }
        lock.lock()
        defer { lock.unlock() }
        let url = URL(fileURLWithPath: path)
        let data = Data(text.utf8)
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            if FileManager.default.fileExists(atPath: path) {
                let handle = try FileHandle(forWritingTo: url)
                defer { try? handle.close() }
                try handle.seekToEnd()
                try handle.write(contentsOf: data)
            } else {
                try data.write(to: url)
            }
        } catch {
            print("EngineReport: couldn't write \(path): \(error)")
        }
    }

    /// Appends a Markdown table.
    public static func appendTable(title: String, header: [String], rows: [[String]]) {
        var lines = ["", "### \(title)", "", "| " + header.joined(separator: " | ") + " |",
                     "|" + String(repeating: "---|", count: header.count)]
        for row in rows {
            lines.append("| " + row.joined(separator: " | ") + " |")
        }
        append(lines.joined(separator: "\n"))
    }
}
