import Foundation

/// Turns model output into text a speech synthesizer can read naturally: strips Markdown,
/// links, and emoji, which TTS voices otherwise read out symbol by symbol.
public enum SpeechTextCleaner {
    public static func clean(_ text: String) -> String {
        var result = removingEmoji(text)
        for (pattern, template) in rules {
            result = pattern.stringByReplacingMatches(
                in: result,
                range: NSRange(result.startIndex..., in: result),
                withTemplate: template
            )
        }
        return result.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines.union(CharacterSet(charactersIn: ",")))
    }

    private static let rules: [(NSRegularExpression, String)] = [
        (regex("```[A-Za-z0-9_+-]*"), " "),
        (regex("!\\[([^\\]]*)\\]\\([^)]*\\)"), "$1"),
        (regex("\\[([^\\]]+)\\]\\([^)]*\\)"), "$1"),
        (regex("https?://[^\\s)\\]]+"), ""),
        (regex("^[ \\t]*#{1,6}[ \\t]*", lines: true), ""),
        (regex("^[ \\t]*>[ \\t]?", lines: true), ""),
        (regex("^[ \\t]*(?:[-*+•]|\\d{1,3}[.)])[ \\t]+", lines: true), ""),
        (regex("\\*\\*|__|\\*|`|~~"), ""),
        (regex("-{3,}|={3,}"), " "),
        (regex("[ \\t]*\\|[ \\t]*"), ", "),
        (regex("(?:,\\s*){2,}"), ", "),
        (regex("\\s+"), " "),
    ]

    private static func regex(_ pattern: String, lines: Bool = false) -> NSRegularExpression {
        // The patterns are constants, so failing to compile one is a programming error.
        try! NSRegularExpression(pattern: pattern, options: lines ? [.anchorsMatchLines] : [])
    }

    private static func removingEmoji(_ text: String) -> String {
        text.filter { character in
            !character.unicodeScalars.contains { scalar in
                scalar.properties.isEmojiPresentation
                    || (scalar.properties.isEmoji && scalar.value >= 0x2190)
                    || scalar.value == 0xFE0F
                    || scalar.value == 0x200D
            }
        }
    }
}
