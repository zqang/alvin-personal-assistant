import AssistantKit
import Foundation
import MLX
import MLXLMCommon

/// Thrown when the GPU stopped being allowed in the middle of a prefill, or of the re-feed of a
/// rewind before it. The ledger is still exact (it holds every chunk that was fed).
struct PrefillInterrupted: Error {}

/// Feeds prompt tokens into a live session (plan §4.6).
///
/// - Chunks of at most `chunk` tokens. Intermediate chunks compute no logits (`.none`) and are
///   evaluated asynchronously, so the CPU builds the next chunk's graph while the GPU runs.
/// - Chunks also split at checkpoint positions; there the cache is evaluated and snapshotted.
/// - The final chunk computes only the last row's logits (`.last`).
enum Prefill {
    /// Feeds `tokens` after the end of the ledger and checkpoints at `marks` (absolute ledger
    /// positions from the current end through the end of `tokens`). Returns the `[1, V]` logits
    /// predicting the token after the last one when `wantLogits`, else nil. Checks `isAllowed`
    /// before every chunk.
    static func run(
        _ session: LiveSession, tokens: [Int], chunk: Int, marks: [(CheckpointMark, Int)] = [],
        wantLogits: Bool, isAllowed: () -> Bool
    ) throws -> MLXArray? {
        let start = session.ledger.count
        let end = start + tokens.count
        let marks = marks.filter { $0.1 >= start && $0.1 <= end }
        func checkpoints(at position: Int) {
            for (label, markPosition) in marks where markPosition == position {
                session.checkpoint(label)
            }
        }

        checkpoints(at: start)
        let splits = Set(marks.map(\.1)).filter { $0 > start }
        var position = start
        var logits: MLXArray?
        while position < end {
            guard isAllowed() else { throw PrefillInterrupted() }
            let nextSplit = splits.filter { $0 > position }.min() ?? end
            let limit = min(position + max(1, chunk), nextSplit, end)
            let piece = Array(tokens[(position - start) ..< (limit - start)])
            let isLast = limit == end
            if isLast && wantLogits {
                logits = session.feed(piece, rows: .last).logits
            } else {
                session.feed(piece, rows: .none)
            }
            if splits.contains(limit) {
                eval(session.target.cache)
                checkpoints(at: limit)
            } else if !isLast {
                asyncEval(session.target.cache)
            }
            position = limit
        }
        return logits
    }
}
