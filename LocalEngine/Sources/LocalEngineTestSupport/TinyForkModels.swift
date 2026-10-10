import Foundation
import LocalEngine
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Tiny random-weight models built from the engine's `HybridQwen35` fork, from the same JSON as
/// the stock tiny models in `TinyModels`, and the weight copy that makes a fork compute exactly
/// what a stock model computes.
public enum TinyForkModels {
    public static func hybridTextConfiguration() throws -> HybridQwen35TextConfiguration {
        try JSONDecoder.json5().decode(HybridQwen35TextConfiguration.self, from: Data(TinyModels.hybridTextConfigJSON.utf8))
    }

    public static func hybridWrapperConfiguration() throws -> HybridQwen35Configuration {
        try JSONDecoder.json5().decode(HybridQwen35Configuration.self, from: Data(TinyModels.hybridWrapperConfigJSON.utf8))
    }

    /// The fork's text model (`HybridQwen35TextModel`) with weights drawn from `seed`.
    public static func makeForkHybrid(seed: UInt64 = 0) throws -> HybridQwen35TextModel {
        let configuration = try hybridTextConfiguration()
        return build(seed: seed) { HybridQwen35TextModel(configuration) }
    }

    /// The fork in wrapper form (`HybridQwen35Model`) with weights drawn from `seed`.
    public static func makeForkHybridWrapper(seed: UInt64 = 0) throws -> HybridQwen35Model {
        let configuration = try hybridWrapperConfiguration()
        return build(seed: seed) { HybridQwen35Model(configuration) }
    }

    /// Replaces every parameter of `fork` with the stock model's. The module trees must match
    /// key for key and shape for shape (quantize both identically first, if at all), or this
    /// throws.
    public static func copyWeights(from stock: Module, to fork: Module) throws {
        try fork.update(parameters: stock.parameters(), verify: .all)
        eval(fork)
    }

    /// A fork text model with `stock`'s weights.
    public static func fork(of stock: Qwen35TextModel) throws -> HybridQwen35TextModel {
        let model = try makeForkHybrid(seed: 0)
        try copyWeights(from: stock, to: model)
        return model
    }

    /// A fork wrapper model with `stock`'s weights.
    public static func fork(of stock: Qwen35Model) throws -> HybridQwen35Model {
        let model = try makeForkHybridWrapper(seed: 0)
        try copyWeights(from: stock, to: model)
        return model
    }

    /// One batch-1 `engineForward` of `tokens`, appending to `cache`. Returns float32 logits
    /// `[R, V]` (nil for `.none`) and, when `wantHidden`, the hidden states `[S, H]`.
    public static func engineLogits(
        _ model: any HybridQwen35Forwarding, _ tokens: [Int], cache: [KVCache], rows: LogitRows = .all,
        capture: GDNCaptureSink? = nil, wantHidden: Bool = false
    ) -> (logits: MLXArray?, hidden: MLXArray?) {
        let input = MLXArray(tokens.map { Int32($0) })[.newAxis]
        let (logits, hidden) = model.engineForward(input, cache: cache, rows: rows, capture: capture, wantHidden: wantHidden)
        return (logits.map { $0[0].asType(.float32) }, hidden.map { $0[0] })
    }

    private static func build<M: Module>(seed: UInt64, _ make: () -> M) -> M {
        MLXRandom.seed(seed)
        let model = make()
        eval(model)
        return model
    }
}
