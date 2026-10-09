import AssistantKit
import Foundation
import MLX
import MLXLMCommon

/// Makes each reply's token generator: a `SpeculativeLoop` with the built-in drafters, or the
/// plain `DecodeLoop` when speculation is off or the target can't roll a round back (plan §4.7,
/// WP30 instruction 1).
///
/// The drafters, in the order the loop asks them:
/// 1. `ExemplarDrafter`: tool-call skeletons for the request's tools (only inside a tool call);
/// 2. `PromptLookupDrafter`: n-grams of the live session;
/// 3. `SuffixDrafter`: past replies and tool results, saved at `corpusURL` (in memory only when
///    nil);
/// 4. the drafters of the engine's extensions (`GeneratorContext.drafters`);
/// 5. `DraftModelDrafter` over `draftModel`, when it is a pure-attention model with the
///    target's vocabulary size, stop tokens and tokenization.
///
/// The draft policy, the prompt-lookup index, the corpus and the draft model's cache live as long
/// as the factory, so acceptance statistics, backoff and indexes carry over from reply to reply.
/// Use one factory per engine.
public struct SpeculativeGeneratorFactory: GeneratorFactory {
    public let mode: LocalSpeculationMode
    public let curve: CostCurve
    public let maxDraft: Int
    public let corpusURL: URL?
    public let draftModel: LoadedModel?
    let resources: SpeculationResources

    public init(mode: LocalSpeculationMode, curve: CostCurve, maxDraft: Int = 4, corpusURL: URL?, draftModel: LoadedModel? = nil) {
        self.mode = mode
        self.curve = curve
        self.maxDraft = maxDraft
        self.corpusURL = corpusURL
        self.draftModel = draftModel
        self.resources = SpeculationResources(policy: SharedDraftPolicy(curve: curve, maxDraft: maxDraft))
    }

    /// The draft policy the replies share.
    public var policy: SharedDraftPolicy { resources.policy }

    /// Why the draft model isn't drafting, once a reply has checked it; nil while it drafts (or
    /// hasn't been checked, or there is none).
    public var draftModelProblem: String? { resources.draftModelProblem }

    public func makeGenerator(_ context: GeneratorContext) -> TokenGenerator {
        guard mode != .off, context.session.target.supportsRollback else {
            return DecodeLoop(context)
        }
        let drafters = resources.drafters(for: context, corpusURL: corpusURL, draftModel: draftModel)
        return SpeculativeLoop(context, drafters: drafters, policy: resources.policy, options: SpeculativeLoop.Options(mode: mode))
    }
}

/// What a `SpeculativeGeneratorFactory` keeps between replies. Its replies run on one engine
/// queue; the lock only guards against a factory shared by mistake.
final class SpeculationResources: @unchecked Sendable {
    let policy: SharedDraftPolicy
    private let lock = NSLock()
    private weak var session: LiveSession?
    private var promptLookup: PromptLookupDrafter?
    private var corpus: SuffixDrafter?
    private var draftDrafter: DraftModelDrafter?
    private var draftChecked = false
    private var problem: String?

    init(policy: SharedDraftPolicy) {
        self.policy = policy
    }

    var draftModelProblem: String? {
        lock.lock()
        defer { lock.unlock() }
        return problem
    }

    /// The drafters for one generation, in order, reset to the session's ledger (the extension
    /// drafters were reset by the engine).
    func drafters(for context: GeneratorContext, corpusURL: URL?, draftModel: LoadedModel?) -> [any Drafter] {
        lock.lock()
        defer { lock.unlock() }

        if session !== context.session {
            // Another engine (or a reloaded one): its indexes and caches start over.
            session = context.session
            promptLookup = nil
            draftDrafter?.startOver()
            draftDrafter = nil
            draftChecked = false
            problem = nil
        }
        let promptLookup = self.promptLookup ?? PromptLookupDrafter(renderer: context.renderer)
        self.promptLookup = promptLookup
        let corpus = self.corpus ?? SuffixDrafter(url: corpusURL, renderer: context.renderer, stopTokens: context.stopTokens)
        self.corpus = corpus
        if !draftChecked, let draftModel {
            draftChecked = true
            do {
                let drafter = try DraftModelDrafter(draft: draftModel)
                if let found = drafter.compatibilityProblem(
                    targetVocabularySize: context.session.target.vocabularySize, stopTokens: context.stopTokens,
                    renderer: context.renderer)
                {
                    problem = found.description
                } else {
                    draftDrafter = drafter
                }
            } catch {
                problem = "\(error)"
            }
            if let problem {
                print("LocalEngine: the draft model \(draftModel.id) won't draft: \(problem)")
            }
        }

        let exemplar = ExemplarDrafter(tools: context.request.tools, format: context.toolCallFormat, renderer: context.renderer)
        var builtIns: [any Drafter] = []
        if !exemplar.isEmpty { builtIns.append(exemplar) }
        builtIns.append(promptLookup)
        builtIns.append(corpus)
        let ledger = context.session.ledger
        for drafter in builtIns {
            drafter.reset(ledger: ledger, request: context.request)
        }
        draftDrafter?.reset(ledger: ledger, request: context.request)

        var drafters = builtIns + context.drafters
        if let draftDrafter { drafters.append(draftDrafter) }
        return drafters
    }
}
