import Foundation

/// A short spoken phrase that tells a voice user the assistant is busy. It is not part of the
/// answer: it is never shown, stored or sent back to a model.
public enum SpokenCue: String, Codable, Equatable, Sendable {
    /// Before a web search.
    case lookingUp
    /// Before a read-only tool, e.g. checking the calendar.
    case checking
    /// A filler while nothing has been heard for a while.
    case working
    /// Deep mode started.
    case deepThinking
    /// Deep mode is taking a while.
    case stillThinking
    /// The on-device model handed the request to the cloud assistant.
    case handingOff
    /// The on-device model is still loading.
    case loadingModel

    /// The phrase for `cue` in Chinese when `localeIdentifier` starts with "zh", otherwise in English.
    public static func phrase(_ cue: SpokenCue, localeIdentifier: String) -> String {
        let chinese = localeIdentifier.lowercased().hasPrefix("zh")
        switch cue {
        case .lookingUp: return chinese ? "我查一下。" : "Let me look that up."
        case .checking: return chinese ? "我看看。" : "Let me check."
        case .working: return chinese ? "稍等。" : "One moment."
        case .deepThinking: return chinese ? "让我好好想想。" : "Let me think that through properly."
        case .stillThinking: return chinese ? "还在想，马上就好。" : "Still thinking, almost there."
        case .handingOff: return chinese ? "我上网查一下。" : "Let me check online."
        case .loadingModel: return chinese ? "稍等，正在准备。" : "One moment, getting ready."
        }
    }
}
