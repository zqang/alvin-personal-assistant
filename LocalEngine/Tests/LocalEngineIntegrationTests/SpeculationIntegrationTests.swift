import AssistantKit
import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

/// Speculative decoding on real models (WP30): prompt lookup on Qwen3.5-0.8B (the hybrid fork)
/// at K = 4 and K = 8, the policy-driven factory, and the Qwen3-0.6B draft model for
/// Qwen3-1.7B. Every speculative reply must pass the teacher-forced check (each token the greedy
/// choice of a fresh pass over its context, up to near-ties); acceptance per depth, tokens per
/// round and wall time against plain decoding go into the report, with the M1 cost curve.
final class SpeculationIntegrationTests: XCTestCase {
    private static let system = "You are a careful copy editor. Return only the corrected text."
    private static let editPrompt = """
        Fix the typos in this paragraph and return the whole paragraph, otherwise unchanged:

        The engine keeps the conversation resident in memory, so a continued chat only reads the new \
        tokens. When the reply repeats the prompt, the drafter guesess the next tokens from earlier \
        text and the model checks several of them in one pass. Accepted tokens are emited at once; a \
        rejected one is replaced by the models own choice, so the anwser is exactly what plain \
        decoding would have produced.
        """
    private static let maxTokens = 160

    override func setUpWithError() throws {
        try IntegrationEnvironment.requireEnabled()
        try MetalAvailability.require()
    }

    func testPromptLookupOnQwen35_0_8B() async throws {
        let repo = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
        let engine = try await Self.loadEngine(repo)
        XCTAssertTrue(engine.info.forked, "\(repo) should run on the fork")
        let curve = try await CostProbe.measure(engine: engine)
        Self.reportCurve(curve, repo: repo)

        let request = EngineRequest(system: Self.system, turns: [ChatTurn(role: .user, text: Self.editPrompt)], maxTokens: Self.maxTokens, greedy: true)
        let plain = try await Self.generate(engine, request: request, factory: nil)
        var rows: [[String]] = [Self.row("plain", plain, plain: plain)]
        var tally = NearTieTally(tolerance: NearTieTally.quantizedTolerance)
        for k in [4, 8] {
            let factory = RecordingFactory(options: SpeculativeLoop.Options(forcedDraftLength: k)) { context in
                [PromptLookupDrafter(renderer: context.renderer)]
            }
            let run = try await Self.generate(engine, request: request, factory: factory)
            try Self.teacherForcedCheck(engine, run, label: "\(repo) prompt lookup K=\(k)", tally: &tally)
            XCTAssertGreaterThan(run.rounds.count, 0, "an edit of the prompt must find prompt-lookup matches")
            rows.append(Self.row("prompt lookup, K = \(k) (forced)", run, plain: plain))
        }
        let automatic = SpeculativeGeneratorFactory(mode: .automatic, curve: curve, corpusURL: nil)
        let run = try await Self.generate(engine, request: request, factory: automatic)
        try Self.teacherForcedCheck(engine, run, label: "\(repo) automatic", tally: &tally)
        rows.append(Self.row("factory, automatic (probed curve)", run, plain: plain))

        EngineReport.appendTable(
            title: "Speculation, \(repo), \"edit this paragraph\" (greedy, \(Self.maxTokens) tokens max)",
            header: ["Generator", "Tokens", "Rounds", "Tokens/round", "Acceptance by depth", "Decode s", "Speed-up"], rows: rows)
        EngineReport.append("- Teacher-forced check (\(repo)): \(tally.summary)")
        tally.assertPassed()
    }

    func testDraftModelOnQwen3_1_7B() async throws {
        let repo = "mlx-community/Qwen3-1.7B-4bit"
        let draftRepo = "mlx-community/Qwen3-0.6B-4bit"
        let engine = try await Self.loadEngine(repo)
        let draftDirectory = try await IntegrationEnvironment.snapshot(draftRepo)
        let draft = try await ModelLoader.load(directory: draftDirectory, id: draftRepo)
        let curve = try await CostProbe.measure(engine: engine)
        Self.reportCurve(curve, repo: repo)

        let probe = try DraftModelDrafter(draft: draft)
        let problem = probe.compatibilityProblem(
            targetVocabularySize: engine.info.vocabularySize, stopTokens: engine.info.stopTokenIDs, renderer: engine.loaded.renderer)
        XCTAssertNil(problem, "\(draftRepo) should draft for \(repo)")

        let prompts = [
            Self.editPrompt,
            "Explain in a short paragraph why the sky is blue.",
        ]
        var rows: [[String]] = []
        var tally = NearTieTally(tolerance: NearTieTally.quantizedTolerance)
        for (index, prompt) in prompts.enumerated() {
            let request = EngineRequest(system: "You are a helpful assistant.", turns: [ChatTurn(role: .user, text: prompt)], maxTokens: 128, greedy: true)
            let plain = try await Self.generate(engine, request: request, factory: nil)
            rows.append(Self.row("prompt \(index + 1): plain", plain, plain: plain))
            let drafter = try DraftModelDrafter(draft: draft)
            let forced = RecordingFactory(options: SpeculativeLoop.Options(forcedDraftLength: 3)) { _ in [drafter] }
            let run = try await Self.generate(engine, request: request, factory: forced)
            try Self.teacherForcedCheck(engine, run, label: "\(repo) draft K=3 prompt \(index + 1)", tally: &tally)
            XCTAssertGreaterThan(run.rounds.count, 0)
            rows.append(Self.row("prompt \(index + 1): draft model, K = 3 (forced)", run, plain: plain))

            let automatic = SpeculativeGeneratorFactory(mode: .automatic, curve: curve, corpusURL: nil, draftModel: draft)
            let auto = try await Self.generate(engine, request: request, factory: automatic)
            try Self.teacherForcedCheck(engine, auto, label: "\(repo) automatic prompt \(index + 1)", tally: &tally)
            XCTAssertNil(automatic.draftModelProblem)
            rows.append(Self.row("prompt \(index + 1): factory, automatic + draft model", auto, plain: plain))
        }
        EngineReport.appendTable(
            title: "Speculation, \(repo) with the \(draftRepo) draft (greedy)",
            header: ["Generator", "Tokens", "Rounds", "Tokens/round", "Acceptance by depth", "Decode s", "Speed-up"], rows: rows)
        EngineReport.append("- Teacher-forced check (\(repo) + draft): \(tally.summary)")
        tally.assertPassed()
    }

    // MARK: Helpers

    struct Run {
        /// The reply's tokens as fed (a final stop token included).
        let context: [Int]
        let generated: [Int]
        let seconds: Double
        let rounds: [SpeculativeLoop.Round]
        let stats: LocalGenerationStats
    }

    static func loadEngine(_ repo: String) async throws -> InferenceEngine {
        let directory = try await IntegrationEnvironment.snapshot(repo)
        var configuration = EngineConfiguration()
        configuration.temperature = 0
        let engine = try await InferenceEngine.load(directory: directory, modelID: repo, configuration: configuration)
        try await engine.warmUp()
        return engine
    }

    /// One reply from a fresh session with `factory` (nil: plain), timed from the first token to
    /// the last.
    static func generate(_ engine: InferenceEngine, request: EngineRequest, factory: (any GeneratorFactory)?) async throws -> Run {
        await engine.updateConfiguration { $0.generatorFactory = factory }
        await engine.invalidateSession()
        let events = try await EngineTestHarness.collect(engine.reply(request))
        let stats = try XCTUnwrap(EngineTestHarness.finish(events)?.stats)
        let (context, generated) = try await engine.withSession { session -> ([Int], [Int]) in
            let replyStart = session.snapshot?.turns.last(where: { $0.turn.role == .user })?.replyStart ?? session.ledger.count
            return (Array(session.ledger[..<replyStart]), Array(session.ledger[replyStart...]))
        }
        let rounds = (factory as? RecordingFactory)?.last?.rounds ?? []
        let seconds = stats.generateTime - (stats.phases?.firstToken ?? 0)
        return Run(context: context, generated: generated, seconds: seconds, rounds: rounds, stats: stats)
    }

    /// Each generated token must be the greedy choice of a fresh pass over its context, up to the
    /// first near-tie.
    static func teacherForcedCheck(_ engine: InferenceEngine, _ run: Run, label: String, tally: inout NearTieTally) throws {
        guard !run.generated.isEmpty else {
            XCTFail("\(label): nothing generated")
            return
        }
        let forced = TeacherForcing.run(model: engine.loaded.model, prompt: run.context, continuation: run.generated)
        tally.record(label, reference: forced.map(\.argmax), candidate: run.generated, referenceMargins: forced.map(\.margin))
    }

    static func row(_ name: String, _ run: Run, plain: Run) -> [String] {
        let speculation = run.stats.speculation
        let depths = SpeculativeLoop.acceptanceByDepth(run.rounds).sorted { $0.key < $1.key }
            .map { "\($0.key): \(String(format: "%.0f%%", ($0.value.rate ?? 0) * 100)) of \($0.value.observed)" }
            .joined(separator: ", ")
        func rate(_ run: Run) -> Double {
            run.seconds > 0 ? Double(run.stats.generatedTokens) / run.seconds : 0
        }
        let speedUp = rate(plain) > 0 ? rate(run) / rate(plain) : 0
        return [
            name, "\(run.stats.generatedTokens)", "\(speculation?.rounds ?? 0)",
            speculation?.meanTokensPerRound.map { String(format: "%.2f", $0) } ?? "–",
            depths.isEmpty ? "–" : depths,
            String(format: "%.2f", run.seconds),
            String(format: "%.2f×", speedUp),
        ]
    }

    static func reportCurve(_ curve: CostCurve, repo: String) {
        let widths = curve.seconds.keys.sorted()
        EngineReport.appendTable(
            title: "Cost curve (CostProbe on this GPU), \(repo)",
            header: ["Rows S"] + widths.map(String.init),
            rows: [
                ["forward + eval ms"] + widths.map { String(format: "%.2f", (curve.seconds[$0] ?? 0) * 1000) },
                ["c(S) / c(1)"] + widths.map { String(format: "%.2f", curve.relative($0)) },
                ["stock default"] + widths.map { String(format: "%.2f", CostCurve.stockMLXDefault.relative($0)) },
            ])
    }
}

/// Makes `SpeculativeLoop`s with fixed drafters and options and keeps the last one, so a test can
/// read its rounds.
final class RecordingFactory: GeneratorFactory, @unchecked Sendable {
    private let makeDrafters: (GeneratorContext) -> [any Drafter]
    private let options: SpeculativeLoop.Options
    private let policy = SharedDraftPolicy(curve: .stockMLXDefault)
    private let lock = NSLock()
    private var latest: SpeculativeLoop?

    init(options: SpeculativeLoop.Options, drafters: @escaping (GeneratorContext) -> [any Drafter]) {
        self.options = options
        self.makeDrafters = drafters
    }

    var last: SpeculativeLoop? {
        lock.lock()
        defer { lock.unlock() }
        return latest
    }

    func makeGenerator(_ context: GeneratorContext) -> TokenGenerator {
        let drafters = makeDrafters(context)
        for drafter in drafters {
            drafter.reset(ledger: context.session.ledger, request: context.request)
        }
        let loop = SpeculativeLoop(context, drafters: drafters, policy: policy, options: options)
        lock.lock()
        latest = loop
        lock.unlock()
        return loop
    }
}
