import AssistantKit
import Foundation
import MLX
import MLXLMCommon

/// The rewind checkpoints of a live session: named ledger positions (`CheckpointMark`) and, for
/// hybrid models, the recurrent state at each (plan §4.4, invariant L2).
///
/// Marks at the same position share one snapshot. Pure-attention models store marks with empty
/// snapshots: rewinding them only trims. Not thread-safe: engine queue only.
public final class CheckpointStore {
    public private(set) var marks: [CheckpointMark: Int] = [:]
    private var snapshots: [Int: CacheSnapshot] = [:]

    public init() {}

    /// Records `snapshot` under `mark`, replacing the mark's previous position.
    public func record(_ mark: CheckpointMark, _ snapshot: CacheSnapshot) {
        marks[mark] = snapshot.position
        snapshots[snapshot.position] = snapshot
        prune()
    }

    /// The snapshot `mark` points at, if any.
    public func snapshot(for mark: CheckpointMark) -> CacheSnapshot? {
        marks[mark].flatMap { snapshots[$0] }
    }

    /// The checkpoint with the largest position ≤ `position`.
    public func deepest(atOrBelow position: Int) -> CacheSnapshot? {
        snapshots.keys.filter { $0 <= position }.max().flatMap { snapshots[$0] }
    }

    /// Drops every mark above `position` (a rewind to `position` makes them stale).
    public func drop(above position: Int) {
        marks = marks.filter { $0.value <= position }
        prune()
    }

    /// Drops every mark, or every mark but `systemEnd`.
    public func removeAll(keepSystem: Bool = false) {
        marks = keepSystem ? marks.filter { $0.key == .systemEnd } : [:]
        prune()
    }

    /// Keeps only `kept` (from `CheckpointPolicy.marksToKeep`).
    public func keep(_ kept: [CheckpointMark: Int]) {
        marks = marks.filter { kept[$0.key] == $0.value }
        prune()
    }

    /// Bytes the snapshots hold.
    public var bytes: Int {
        snapshots.values.reduce(0) { $0 + $1.bytes }
    }

    /// Positions that have a snapshot, ascending.
    public var positions: [Int] {
        snapshots.keys.sorted()
    }

    /// Forgets snapshots no mark points at.
    private func prune() {
        let referenced = Set(marks.values)
        snapshots = snapshots.filter { referenced.contains($0.key) }
    }
}
