import MLX
import MLXLMCommon

/// Which cache layers hold attention keys and values (`KVCacheSimple`) and which hold a
/// recurrent state (`MambaCache`: slot 0 the conv state, slot 1 the gated-delta state).
public struct CacheLayout: Equatable, Sendable {
    public let attention: [Int]
    public let recurrent: [Int]

    public init(attention: [Int], recurrent: [Int]) {
        self.attention = attention
        self.recurrent = recurrent
    }

    /// Whether some layers keep a recurrent state, which can't be trimmed and must be
    /// snapshotted instead.
    public var isHybrid: Bool { !recurrent.isEmpty }

    /// Nil if any layer is something other than exactly `KVCacheSimple` or `MambaCache`
    /// (rotating, quantized and chunked caches move or rewrite their contents, so trimming and
    /// snapshots wouldn't be exact).
    public static func detect(_ cache: [KVCache]) -> CacheLayout? {
        var attention: [Int] = []
        var recurrent: [Int] = []
        for (index, layer) in cache.enumerated() {
            let kind = type(of: layer)
            if kind == KVCacheSimple.self {
                attention.append(index)
            } else if kind == MambaCache.self {
                recurrent.append(index)
            } else {
                return nil
            }
        }
        return CacheLayout(attention: attention, recurrent: recurrent)
    }
}

/// The recurrent state of every recurrent layer at a position: references to the slot arrays,
/// which forwards replace rather than mutate (F6), so holding them is enough.
public struct CacheSnapshot {
    /// The number of tokens in the cache when the snapshot was taken (from the caller's ledger).
    public let position: Int
    /// Bytes held by the referenced slots.
    public let bytes: Int
    /// `(cache[0], cache[1])` per recurrent layer, in `CacheLayout.recurrent` order.
    let slots: [(MLXArray?, MLXArray?)]
}

/// Exact cache rewinds without `ArraysCache.state` or `metaState` setters (F10), and without
/// reading `MambaCache.offset`.
public enum EngineCacheOps {
    /// References the recurrent slots and evaluates them, so the snapshot doesn't keep the graph
    /// that produced them alive.
    public static func snapshot(_ cache: [KVCache], layout: CacheLayout, position: Int) -> CacheSnapshot {
        var slots: [(MLXArray?, MLXArray?)] = []
        var arrays: [MLXArray] = []
        for index in layout.recurrent {
            let layer = recurrentLayer(cache, index)
            let slot = (layer[0], layer[1])
            slots.append(slot)
            if let conv = slot.0 { arrays.append(conv) }
            if let state = slot.1 { arrays.append(state) }
        }
        if !arrays.isEmpty {
            eval(arrays)
        }
        return CacheSnapshot(position: position, bytes: arrays.reduce(0) { $0 + $1.nbytes }, slots: slots)
    }

    /// Moves the cache from `currentPosition` back to the snapshot's position: attention layers
    /// are trimmed by the difference and the recurrent slots are set back through the
    /// subscripts. Rewinding forward (a snapshot past the current position) is a programming
    /// error.
    public static func restore(_ cache: [KVCache], layout: CacheLayout, to snapshot: CacheSnapshot, currentPosition: Int) {
        precondition(currentPosition >= snapshot.position, "Can't restore position \(snapshot.position) from position \(currentPosition).")
        precondition(snapshot.slots.count == layout.recurrent.count, "The snapshot was taken with a different cache layout.")
        trimAttention(cache, layout: layout, by: currentPosition - snapshot.position)
        for (index, slot) in zip(layout.recurrent, snapshot.slots) {
            let layer = recurrentLayer(cache, index)
            layer[0] = slot.0
            layer[1] = slot.1
        }
    }

    /// Drops the newest `n` tokens from every attention layer. Recurrent layers are untouched.
    public static func trimAttention(_ cache: [KVCache], layout: CacheLayout, by n: Int) {
        guard n > 0 else { return }
        for index in layout.attention {
            cache[index].trim(n)
        }
    }

    /// Bytes currently held by the recurrent slots: what one snapshot of this cache costs.
    public static func recurrentBytes(_ cache: [KVCache], layout: CacheLayout) -> Int {
        layout.recurrent.reduce(0) { total, index in
            let layer = recurrentLayer(cache, index)
            return total + (layer[0]?.nbytes ?? 0) + (layer[1]?.nbytes ?? 0)
        }
    }

    private static func recurrentLayer(_ cache: [KVCache], _ index: Int) -> ArraysCache {
        guard let layer = cache[index] as? ArraysCache else {
            preconditionFailure("Cache layer \(index) is not a recurrent cache.")
        }
        return layer
    }
}
