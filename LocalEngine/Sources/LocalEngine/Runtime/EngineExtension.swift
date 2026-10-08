import Foundation

/// A capability added to the engine after load: drafters (MTP, a draft model) or kernels.
///
/// Extensions are prepared once by `InferenceEngine.warmUp()`, on the engine queue. One whose
/// `prepare` throws is disabled for the engine's lifetime (and logged); the engine keeps working
/// without it.
public protocol EngineExtension: AnyObject {
    var name: String { get }
    /// On the engine queue after load; throw = disabled (logged).
    func prepare(_ engine: InferenceEngine) throws
    /// Drafters for one reply. The engine resets them to the ledger once the prompt is in the
    /// cache and hands them to the generator through `GeneratorContext.drafters`.
    func drafters(for request: EngineRequest, engine: InferenceEngine) -> [any Drafter]
}
