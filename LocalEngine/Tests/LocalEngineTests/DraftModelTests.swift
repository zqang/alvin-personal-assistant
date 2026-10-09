import AssistantKit
import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

/// `DraftModelDrafter`: a tiny Qwen3 target with a tiny Qwen3 draft of the same vocabulary
/// decodes exactly like plain decoding, whether the draft is always right (the target's own
/// weights), unrelated (other weights) or anything between; its cache follows rejections and
/// full acceptances; its cost is calibrated at the first round.
final class DraftModelTests: XCTestCase {
    override func setUpWithError() throws {
        try MetalAvailability.require()
    }

    private func draft(_ model: any LanguageModel, id: String) throws -> LoadedModel {
        try EngineTestHarness.loadedModel(model, id: id, modelType: "qwen3")
    }

    func testDraftModelDecodingEqualsPlain() throws {
        let target = try EngineTestHarness.makeModel(.qwen3, seed: 500)
        let unrelated = try EngineTestHarness.makeModel(.qwen3, seed: 501)
        var tally = NearTieTally(tolerance: NearTieTally.float32Tolerance)
        var rows: [[String]] = []
        for (name, draftModel) in [("same weights", target), ("other weights", unrelated)] {
            for promptSeed in UInt64(502) ... 503 {
                let prompt = TinyModels.tokens(12, seed: promptSeed)
                let reference = try SpecHarness.plain(model: target, prompt: prompt, maxTokens: 40, stops: [])
                let margins = TeacherForcing.run(model: target, prompt: prompt, continuation: reference.tokens).map(\.margin)
                for forced in [1, 3, 4, nil] as [Int?] {
                    let label = "\(name) prompt \(promptSeed) K=\(forced.map(String.init) ?? "policy")"
                    let drafter = try DraftModelDrafter(draft: draft(draftModel, id: "draft-\(name)"))
                    XCTAssertNil(drafter.compatibilityProblem(
                        targetVocabularySize: TinyModels.vocabularySize, stopTokens: drafter.draft.stopTokenIDs, renderer: FakeChatMLTokenizer()))
                    let run = try SpecHarness.speculative(
                        model: target, prompt: prompt, maxTokens: 40, stops: [], drafters: [drafter],
                        options: SpeculativeLoop.Options(forcedDraftLength: forced))
                    tally.record(label, reference: reference.tokens, candidate: run.tokens, referenceMargins: margins)
                    try SpecHarness.assertExact(run, prompt: prompt, maxTokens: 40, stops: [], label: label)
                    XCTAssertGreaterThan(run.loop.rounds.count, 0, "\(label): the draft model drafted")
                    XCTAssertTrue(run.loop.rounds.allSatisfy { $0.source == .draftModel })
                    if name == "same weights" {
                        let accepted = run.loop.rounds.reduce(0) { $0 + $1.accepted }
                        let proposed = run.loop.rounds.reduce(0) { $0 + $1.proposed }
                        XCTAssertGreaterThanOrEqual(Double(accepted), 0.9 * Double(proposed), "\(label): its own weights are nearly always right")
                    }
                    // The draft cache holds a prefix of what it was last asked about.
                    let fed = drafter.fed
                    XCTAssertTrue(fed.count <= run.session.ledger.count + 8)
                    XCTAssertTrue(drafter.calibrated, "\(label): calibrated at the first round")
                    XCTAssertGreaterThanOrEqual(drafter.costPerToken, 0.01)
                    XCTAssertLessThanOrEqual(drafter.costPerToken, 4)
                    if forced == nil {
                        rows.append([label, "\(run.loop.rounds.count)", String(format: "%.2f", run.loop.speculation?.meanTokensPerRound ?? 0),
                                     String(format: "%.3f", drafter.costPerToken)])
                    }
                }
            }
        }
        tally.assertPassed()
        EngineReport.appendTable(
            title: "Draft model (tiny Qwen3 draft for tiny Qwen3), policy-driven", header: ["Case", "Rounds", "Tokens/round", "Cost/token"],
            rows: rows)
    }

    /// After a rejection the draft cache is trimmed back to the accepted tokens; after a full
    /// acceptance the last draft and the bonus are fed together: either way it then matches the
    /// context it is asked about, and its drafts equal a fresh draft model's.
    func testDraftCacheFollowsTheContext() throws {
        let model = try EngineTestHarness.makeModel(.qwen3, seed: 510)
        let drafter = try DraftModelDrafter(draft: draft(model, id: "draft-cache"))
        let prompt = TinyModels.tokens(10, seed: 511)
        let request = EngineRequest(system: "", turns: [])
        drafter.reset(ledger: prompt, request: request)

        let first = try XCTUnwrap(drafter.propose(context: prompt[...], maxTokens: 4))
        XCTAssertEqual(first.tokens.count, 4)
        XCTAssertEqual(drafter.fed, prompt + Array(first.tokens.prefix(3)), "every draft but the last is fed")

        // A rejection at depth 2: the context keeps 1 draft and a correction.
        let rejected = prompt + [first.tokens[0], (first.tokens[1] + 1) % TinyModels.vocabularySize]
        let second = try XCTUnwrap(drafter.propose(context: rejected[...], maxTokens: 3))
        XCTAssertEqual(drafter.fed, rejected + Array(second.tokens.prefix(2)))
        XCTAssertEqual(second.tokens, freshDraft(model, context: rejected, count: 3))

        // A full acceptance: the context holds every draft and a bonus.
        let accepted = rejected + second.tokens + [7]
        let third = try XCTUnwrap(drafter.propose(context: accepted[...], maxTokens: 2))
        XCTAssertEqual(drafter.fed, accepted + Array(third.tokens.prefix(1)))
        XCTAssertEqual(third.tokens, freshDraft(model, context: accepted, count: 2))

        // Another conversation entirely.
        let other = TinyModels.tokens(7, seed: 512)
        let fourth = try XCTUnwrap(drafter.propose(context: other[...], maxTokens: 3))
        XCTAssertEqual(fourth.tokens, freshDraft(model, context: other, count: 3))
    }

    /// Greedy drafts from a fresh cache, one token at a time.
    private func freshDraft(_ model: any LanguageModel, context: [Int], count: Int) -> [Int] {
        let cache = model.newCache(parameters: nil)
        var logits = TinyModels.logits(model, context, cache: cache)
        var tokens: [Int] = []
        while tokens.count < count {
            let token = argMax(logits[logits.dim(0) - 1], axis: -1).item(Int.self)
            tokens.append(token)
            logits = TinyModels.logits(model, [token], cache: cache)
        }
        return tokens
    }

    func testCompatibilityChecks() throws {
        let qwen3 = try EngineTestHarness.makeModel(.qwen3, seed: 520)
        let loaded = try draft(qwen3, id: "draft-compat")
        let drafter = try DraftModelDrafter(draft: loaded)
        let tokenizer = FakeChatMLTokenizer()
        XCTAssertNil(drafter.compatibilityProblem(targetVocabularySize: 128, stopTokens: loaded.stopTokenIDs, renderer: tokenizer))
        XCTAssertEqual(
            drafter.compatibilityProblem(targetVocabularySize: 256, stopTokens: loaded.stopTokenIDs, renderer: tokenizer),
            .vocabularyMismatch(target: 256, draft: 128))
        let otherStops = loaded.stopTokenIDs.union([99])
        XCTAssertEqual(
            drafter.compatibilityProblem(targetVocabularySize: 128, stopTokens: otherStops, renderer: tokenizer),
            .stopTokensMismatch(target: otherStops, draft: loaded.stopTokenIDs))

        // A hybrid can't be a draft: its recurrent state can't be trimmed.
        let hybrid = try EngineTestHarness.makeModel(.hybrid, seed: 521)
        XCTAssertThrowsError(try DraftModelDrafter(draft: EngineTestHarness.loadedModel(hybrid, id: "draft-hybrid", modelType: "qwen3_5_text"))) {
            XCTAssertEqual($0 as? DraftModelDrafter.Problem, .notPureAttention)
        }
    }

    /// Through `SpeculativeGeneratorFactory`: a compatible draft model drafts and the replies
    /// equal a plain engine's; an incompatible one is reported and left out.
    func testFactoryUsesACompatibleDraftModel() async throws {
        let configuration = EngineTestHarness.testConfiguration(maxTokens: 32)
        let plain = try EngineTestHarness.makeEngine(tiny: .qwen3, seed: 530, configuration: configuration)
        let target = try EngineTestHarness.makeEngine(tiny: .qwen3, seed: 530, configuration: configuration)
        // The draft is a copy of the target (same seed): always right.
        let draftModel = try EngineTestHarness.makeModel(.qwen3, seed: 530)
        let factory = SpeculativeGeneratorFactory(
            mode: .automatic, curve: .stockMLXDefault, corpusURL: nil, draftModel: try draft(draftModel, id: "draft-factory"))
        await target.updateConfiguration { $0.generatorFactory = factory }

        let request = EngineRequest(system: "You are Alvin.", turns: [ChatTurn(role: .user, text: "Tell me something.")])
        let expected = try await EngineTestHarness.collect(plain.reply(request))
        let actual = try await EngineTestHarness.collect(target.reply(request))
        XCTAssertEqual(EngineTestHarness.text(actual), EngineTestHarness.text(expected))
        XCTAssertNil(factory.draftModelProblem)
        let stats = try XCTUnwrap(EngineTestHarness.finish(actual)?.stats.speculation)
        XCTAssertGreaterThan(stats.drafted[DraftSource.draftModel.rawValue] ?? 0, 0)
        XCTAssertGreaterThan(stats.accepted[DraftSource.draftModel.rawValue] ?? 0, 0)
        let report = try await target.withSession { $0.assertConsistent() }
        XCTAssertTrue(report.isConsistent(), "\(report)")

        // A hybrid draft is refused; the engine still replies (without it).
        let hybrid = try EngineTestHarness.makeModel(.hybrid, seed: 531)
        let refusing = SpeculativeGeneratorFactory(
            mode: .automatic, curve: .stockMLXDefault, corpusURL: nil,
            draftModel: try EngineTestHarness.loadedModel(hybrid, id: "draft-hybrid-factory", modelType: "qwen3_5_text"))
        let other = try EngineTestHarness.makeEngine(tiny: .qwen3, seed: 530, configuration: configuration)
        await other.updateConfiguration { $0.generatorFactory = refusing }
        let fallback = try await EngineTestHarness.collect(other.reply(request))
        XCTAssertEqual(EngineTestHarness.text(fallback), EngineTestHarness.text(expected))
        XCTAssertNotNil(refusing.draftModelProblem)
    }
}
