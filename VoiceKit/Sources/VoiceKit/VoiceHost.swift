import AssistantKit
import Foundation

// What an app gives voice mode. It makes a `VoiceSession(conversation:pipeline:host:)` from its
// conversation, its `ReplyPipeline` (which produces the replies: routing, engines and tools, read
// from `ReplyRequest.messages`) and a host, then shows `VoiceModeView(session:)`.

/// A message voice mode writes: a spoken user turn, or the reply to one.
@MainActor
public protocol VoiceMessage: AnyObject {
    var text: String { get set }
    var createdAt: Date { get set }
    var status: StoredMessage.Status { get set }
    var errorText: String? { get set }
    /// Client tool rounds the reply ran, in order.
    var toolRounds: [ToolRound] { get set }
    /// The photo the user took for this turn, as a JPEG.
    var photoData: Data? { get set }
    /// What the photo reader read in `photoData`; nil until it has.
    var photoNote: String? { get set }
    /// The message as a reply reads it.
    var stored: StoredMessage { get }
}

/// The conversation voice mode talks in.
@MainActor
public protocol VoiceConversation: AnyObject {
    /// Every message, oldest first.
    var voiceMessages: [any VoiceMessage] { get }
    /// Adds a message at the end and returns it.
    func appendMessage(role: ChatRole, text: String, isVoice: Bool, status: StoredMessage.Status) -> any VoiceMessage
    func remove(_ message: any VoiceMessage)
    /// The user turn that read `old` now reads `new` (a second transcription, or the words an early
    /// reply heard): a title made from `old` should follow it.
    func userTurnRevised(from old: String, to new: String)
    /// Marks the conversation updated and persists it, after each reply and when the session stops.
    func save()
}

/// The app around voice mode: its settings, and hooks for what only it has. Each hook has a
/// default that goes without it, so a host need only give `settings`.
@MainActor
public protocol VoiceSessionHost: AnyObject {
    /// Voice mode reads the settings and never changes them.
    var settings: AssistantSettings { get }
    /// For OpenAI voices (`settings.voiceEngine`); empty (the default) for none, and the built-in
    /// voice speaks.
    var openAIKey: String { get }
    /// A session is starting, with the microphone and speech recognition allowed: get ready what
    /// its replies and `refineTurn` use, such as on-device models.
    func voiceSessionWillStart(_ settings: AssistantSettings)
    /// The session announced by `voiceSessionWillStart` stopped or failed: free what it alone
    /// needed. Called exactly once for each `voiceSessionWillStart`.
    func voiceSessionDidStop()
    /// The longest clip of a turn `refineTurn` reads, in 16 kHz mono samples; voice mode records
    /// twice that per request so a full clip always fits. Read when a session starts. 0 records
    /// nothing, and `refineTurn` is never called. Above 0, replies likely to run on device don't
    /// start early: the second transcription and the reply model would compete for the GPU.
    var refineClipSamples: Int { get }
    /// A second transcription of a committed turn from its audio, in `locale`; nil keeps the live
    /// transcript. Voice mode waits for it before the reply starts.
    func refineTurn(_ samples: [Float], locale: String) async -> String?
    /// Reads a photo (JPEG data) into streamed text pieces, the note that goes into the turn's
    /// context. Nil while photos can't be read: the camera is hidden, and a photo already in the
    /// conversation without a note gets the unreadable note, as when a read fails.
    var photoReader: ((Data) -> AsyncThrowingStream<String, Error>)? { get }
}

public extension VoiceSessionHost {
    var openAIKey: String { "" }
    func voiceSessionWillStart(_ settings: AssistantSettings) {}
    func voiceSessionDidStop() {}
    var refineClipSamples: Int { 0 }
    func refineTurn(_ samples: [Float], locale: String) async -> String? { nil }
    var photoReader: ((Data) -> AsyncThrowingStream<String, Error>)? { nil }
}
