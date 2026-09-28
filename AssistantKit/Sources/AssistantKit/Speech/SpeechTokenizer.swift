import Foundation

/// Tokenizes transcripts for comparison: lowercased words for alphabetic scripts,
/// one token per character for Chinese, Japanese, and Korean.
enum SpeechTokenizer {
    static func isCJK(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x3040...0x30FF,   // Hiragana, Katakana
             0x3400...0x4DBF,   // CJK Extension A
             0x4E00...0x9FFF,   // CJK Unified Ideographs
             0xAC00...0xD7AF,   // Hangul syllables
             0xF900...0xFAFF,   // CJK Compatibility Ideographs
             0x20000...0x2FA1F: // CJK Extensions B-F and supplements
            return true
        default:
            return false
        }
    }

    static func isCJK(_ character: Character) -> Bool {
        character.unicodeScalars.first.map { isCJK($0) } ?? false
    }

    static func isCJKToken(_ token: String) -> Bool {
        token.unicodeScalars.count == 1 && isCJK(token.unicodeScalars.first!)
    }

    static func tokens(_ text: String) -> [String] {
        var tokens: [String] = []
        var word = String.UnicodeScalarView()
        func flushWord() {
            if !word.isEmpty {
                tokens.append(String(word))
                word = String.UnicodeScalarView()
            }
        }
        for scalar in text.lowercased().unicodeScalars {
            if isCJK(scalar) {
                flushWord()
                tokens.append(String(scalar))
            } else if CharacterSet.alphanumerics.contains(scalar) || scalar == "'" {
                word.append(scalar)
            } else {
                flushWord()
            }
        }
        flushWord()
        return tokens
    }

    /// Units for overlap comparison: whole words, and overlapping pairs of adjacent CJK characters
    /// (single characters are too common to tell echo from new speech).
    static func matchingUnits(_ text: String) -> [String] {
        var units: [String] = []
        var run: [String] = []
        func flushRun() {
            if run.count == 1 {
                units.append(run[0])
            } else if run.count > 1 {
                for index in 0..<(run.count - 1) {
                    units.append(run[index] + run[index + 1])
                }
            }
            run.removeAll()
        }
        for token in tokens(text) {
            if isCJKToken(token) {
                run.append(token)
            } else {
                flushRun()
                units.append(token)
            }
        }
        flushRun()
        return units
    }
}
