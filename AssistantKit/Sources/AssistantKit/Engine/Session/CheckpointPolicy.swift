import Foundation

/// A ledger position the engine can rewind a hybrid model's recurrent state to without re-feeding
/// from the start.
public enum CheckpointMark: String, CaseIterable, Codable, Sendable {
    /// End of the system prefix. Always kept.
    case systemEnd
    /// The `<|im_start|>` of the newest user turn: replacing a tentative or edited user turn.
    case lastUserStart
    /// Right after the newest generation prompt: replacing an interrupted reply.
    case replyStart
}

/// Which checkpoints fit the memory budget.
///
/// A checkpoint is a set of references to the recurrent layers' state (about 49 MiB for Woof 4B);
/// marks at the same position share one. `systemEnd` is always kept; over the budget,
/// `replyStart` is dropped first, then `lastUserStart`. Pure-attention models need no checkpoints
/// (they trim), so their marks cost nothing.
public enum CheckpointPolicy {
    public static let defaultBudgetBytes = 160 << 20

    /// The marks to keep, with their positions.
    public static func marksToKeep(_ marks: [CheckpointMark: Int], bytesPerMark: Int, budgetBytes: Int = defaultBudgetBytes) -> [CheckpointMark: Int] {
        var kept: [CheckpointMark: Int] = [:]
        for mark in CheckpointMark.allCases {
            guard let position = marks[mark] else { continue }
            var candidate = kept
            candidate[mark] = position
            if mark == .systemEnd || bytes(of: candidate, bytesPerMark: bytesPerMark) <= budgetBytes {
                kept = candidate
            }
        }
        return kept
    }

    /// Memory held by `marks`: one checkpoint per distinct position.
    public static func bytes(of marks: [CheckpointMark: Int], bytesPerMark: Int) -> Int {
        Set(marks.values).count * max(0, bytesPerMark)
    }
}
