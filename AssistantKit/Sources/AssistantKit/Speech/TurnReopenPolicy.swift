import Foundation

/// Decides whether a committed voice turn may still be taken back.
///
/// When the user speaks again before any of the reply has been heard, they usually weren't
/// finished: the voice session takes their turn back (the committed message and the reply go)
/// and listens on. That is safe only while the reply can't have acted. Once it starts a tool call
/// or reports a tool round, the turn stands and the new words start the next turn: the tool may
/// already have acted, and its record, kept with the reply, needs the turn that asked for it.
/// Without that turn the model wouldn't see the action, and asking again would repeat it.
///
/// Two calls don't count: one whose name isn't known yet (no call runs before its name is read),
/// and `handoff_to_cloud`, which only passes the request on.
///
/// Create one per reply and pass it each of the reply's events.
public struct TurnReopenPolicy: Equatable, Sendable {
    /// Whether the turn may still be taken back.
    public private(set) var canReopen = true

    public init() {}

    public mutating func received(_ event: AssistantEvent) {
        switch event {
        case .toolRound:
            canReopen = false
        case .progress(.toolCallStarted(let name?)) where name != HandoffTool.name:
            canReopen = false
        case .reply, .cue, .routed, .progress:
            break
        }
    }
}
