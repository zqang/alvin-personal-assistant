import AssistantKit
import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

/// Speculative decoding is lossless (plan §6.4, WP30): on the tiny float32 models (hybrid on the
/// fork, Qwen3 on the stock code), for every kind of drafter and draft length, the tokens equal
/// plain decoding (greedy, up to near-ties), with a position-keyed seed at T = 0.7 they are
/// identical, and without a seed the first tokens follow the same distribution (chi-square).
final class SpeculativeLosslessTests: XCTestCase {
    override func setUpWithError() throws {
        try MetalAvailability.require()
    }

    /// nil is the real prompt-lookup drafter.
    private static let drafterKinds: [SequenceDrafter.Corruption?] = [.oracle, .alwaysWrong, .wrongEveryThird, nil, .adversarialStop]
    private static let draftLengths = [1, 2, 4, 8]

    /// Greedy on the sharpened models: every drafter × K ∈ {1, 2, 4, 8} equals plain decoding
    /// (up to near-ties); the ledger, the statistics and the cache stay exact. (Random tiny models
    /// settle on one repeated token when decoded greedily; the seeded test below covers varied
    /// output and stop tokens.)
    func testGreedySpeculationEqualsPlain() throws {
        var rows: [[String]] = []
        for tiny in EngineTestHarness.Tiny.allCases {
            let model = try EngineTestHarness.makeModel(tiny, seed: 300)
            let prompt = TinyModels.tokens(14, seed: 301)
            let result = try runMatrix(
                "\(tiny) greedy", model: model, prompt: prompt, sampler: FastSampler(temperature: 0),
                scenarios: [(.length, [FakeChatMLTokenizer.imEnd])], exact: false)
            rows.append([tiny.rawValue, "greedy, sharpened", result.summary, "\(result.rounds)", "\(result.accepted)/\(result.drafted)"])
        }
        EngineReport.appendTable(
            title: "Speculative == plain (tiny, 5 drafters × K ∈ {1,2,4,8})",
            header: ["Model", "Sampling", "Comparison", "Rounds", "Accepted/drafted"], rows: rows)
    }

    /// Position-keyed sampling at T = 0.7 / top-p 0.8 / top-k 20 on the unsharpened models
    /// (varied output): a position always draws the same way, so for every drafter × K ∈
    /// {1, 2, 4, 8}, to the length limit and to a stop token, the speculative sequence is
    /// identical to plain decoding.
    func testSeededSpeculationIsIdentical() throws {
        let sampler = FastSampler(temperature: 0.7, topP: 0.8, topK: 20, seed: 4242)
        var rows: [[String]] = []
        for tiny in EngineTestHarness.Tiny.allCases {
            let model = try EngineTestHarness.makeModel(tiny, seed: 320, sharpen: false)
            let prompt = TinyModels.tokens(12, seed: 321)
            let free = try SpecHarness.plain(model: model, prompt: prompt, maxTokens: 48, stops: [], sampler: sampler)
            XCTAssertGreaterThan(Set(free.tokens).count, 8, "\(tiny): sampled output should vary: \(free.tokens)")
            let stop = SpecHarness.stopToken(in: free.tokens)
            let result = try runMatrix(
                "\(tiny) seeded", model: model, prompt: prompt, sampler: sampler,
                scenarios: [(.length, []), (.stop, [stop])], exact: true)
            rows.append([tiny.rawValue, "T=0.7 seeded, unsharpened", result.summary, "\(result.rounds)", "\(result.accepted)/\(result.drafted)"])
        }
        EngineReport.appendTable(
            title: "Speculative == plain (tiny, seeded sampling, 5 drafters × K ∈ {1,2,4,8} × length/stop)",
            header: ["Model", "Sampling", "Comparison", "Rounds", "Accepted/drafted"], rows: rows)
    }

    private struct MatrixResult {
        var summary = ""
        var rounds = 0
        var accepted = 0
        var drafted = 0
    }

    /// Runs every drafter kind × draft length for each scenario against the plain run with the
    /// same sampler. `exact` requires identical tokens; otherwise the near-tie rule applies
    /// (greedy, plan §6.4).
    private func runMatrix(
        _ name: String, model: any LanguageModel, prompt: [Int], sampler: FastSampler,
        scenarios: [(SpecHarness.Scenario, Set<Int>)], exact: Bool
    ) throws -> MatrixResult {
        var result = MatrixResult()
        var tally = NearTieTally(tolerance: NearTieTally.float32Tolerance)
        for (scenario, stops) in scenarios {
            let maxTokens = scenario == .length ? 40 : 48
            let reference = try SpecHarness.plain(model: model, prompt: prompt, maxTokens: maxTokens, stops: stops, sampler: sampler)
            if scenario == .stop {
                XCTAssertEqual(reference.reason, .stop, "\(name): the stop scenario must end at its stop token")
            }
            let margins = sampler.isGreedy
                ? TeacherForcing.run(model: model, prompt: prompt, continuation: reference.tokens).map(\.margin)
                : Array(repeating: Float.infinity, count: reference.tokens.count + 1)
            let sequence = prompt + reference.tokens + (reference.reason == .stop ? Array(stops.prefix(1)) : [])
            for kind in Self.drafterKinds {
                for k in Self.draftLengths {
                    let label = "\(name) \(scenario) \(kind.map { "\($0)" } ?? "ngram") K=\(k)"
                    let drafter: any Drafter
                    if let kind {
                        drafter = SequenceDrafter(sequence: sequence, corruption: kind, stop: stops.first)
                    } else {
                        drafter = PromptLookupDrafter(renderer: FakeChatMLTokenizer())
                    }
                    let run = try SpecHarness.speculative(
                        model: model, prompt: prompt, maxTokens: maxTokens, stops: stops, drafters: [drafter],
                        options: SpeculativeLoop.Options(forcedDraftLength: k), sampler: sampler)
                    let outcome = tally.record(label, reference: reference.tokens, candidate: run.tokens, referenceMargins: margins)
                    if exact {
                        XCTAssertEqual(run.tokens, reference.tokens, label)
                    }
                    try SpecHarness.assertExact(run, prompt: prompt, maxTokens: maxTokens, stops: stops, label: label)
                    if outcome == .identical {
                        XCTAssertEqual(run.reason, reference.reason, label)
                        XCTAssertEqual(run.session.ledger, reference.session.ledger, "\(label): the same tokens in the cache")
                        if kind == .oracle {
                            XCTAssertTrue(run.loop.rounds.allSatisfy { $0.accepted == $0.proposed }, "\(label): the oracle is always right")
                        }
                        if kind == .alwaysWrong {
                            XCTAssertTrue(run.loop.rounds.allSatisfy { $0.accepted == 0 }, "\(label): every draft is wrong")
                        }
                    }
                    if kind != nil && reference.tokens.count >= 3 {
                        XCTAssertGreaterThan(run.loop.rounds.count, 0, "\(label): the sequence drafter always proposes")
                    }
                    XCTAssertTrue(run.loop.rounds.allSatisfy { $0.proposed <= k }, "\(label): K respected")
                    result.rounds += run.loop.rounds.count
                    result.accepted += run.loop.rounds.reduce(0) { $0 + $1.accepted }
                    result.drafted += run.loop.rounds.reduce(0) { $0 + $1.proposed }
                }
            }
        }
        tally.assertPassed()
        result.summary = tally.summary
        return result
    }

    /// The remaining-length cap: a round never verifies more than `remaining − 1` drafts, and the
    /// reply ends exactly at `maxTokens`.
    func testMaxTokensIsRespected() throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let model = try EngineTestHarness.makeModel(tiny, seed: 310)
            let prompt = TinyModels.tokens(10, seed: 311)
            for maxTokens in [1, 2, 3, 5, 9] {
                let reference = try SpecHarness.plain(model: model, prompt: prompt, maxTokens: maxTokens, stops: [])
                let sequence = prompt + reference.tokens + TinyModels.tokens(16, seed: 312)
                let run = try SpecHarness.speculative(
                    model: model, prompt: prompt, maxTokens: maxTokens, stops: [],
                    drafters: [SequenceDrafter(sequence: sequence, corruption: .oracle, stop: nil)],
                    options: SpeculativeLoop.Options(forcedDraftLength: 8))
                let label = "\(tiny) maxTokens \(maxTokens)"
                XCTAssertEqual(run.tokens, reference.tokens, label)
                XCTAssertEqual(run.reason, .length, label)
                XCTAssertEqual(run.session.ledger, prompt + run.tokens, "\(label): every emitted token is in the cache once")
                try SpecHarness.assertExact(run, prompt: prompt, maxTokens: maxTokens, stops: [], label: label)
            }
        }
    }

    /// Without a seed, the first three tokens of 2,000 plain and 2,000 speculative replies (a
    /// fixed draft, forced to 2 drafts) follow the same distribution: chi-square test of
    /// homogeneity, p > 0.001.
    func testUnseededDistributionMatchesPlain() throws {
        // Unsharpened, so the first token has a spread-out distribution (top-1 about 0.7).
        let model = try TinyForkModels.makeForkHybrid(seed: 330)
        let prompt = TinyModels.tokens(10, seed: 331)
        let session = LiveSession(target: SpecHarness.target(for: model))
        let firstLogits = session.feed(prompt, rows: .last).logits!
        eval(firstLogits)
        let greedy = try SpecHarness.plain(model: model, prompt: prompt, maxTokens: 4, stops: [])
        let drafter = SequenceDrafter(sequence: prompt + greedy.tokens, corruption: .oracle, stop: nil, fallback: Array(greedy.tokens.prefix(2)))
        let runs = 2_000

        func sample(speculative: Bool) throws -> [[Int]: Int] {
            let sampler = FastSampler(temperature: 0.7, topP: 0.8, topK: 20)
            var counts: [[Int]: Int] = [:]
            for _ in 0 ..< runs {
                let tokens = try session.withScratch { () throws -> [Int] in
                    let context = SpecHarness.context(session: session, firstLogits: firstLogits, sampler: sampler, stops: [], maxTokens: 3)
                    let generator: TokenGenerator = speculative
                        ? SpeculativeLoop(context, drafters: [drafter], policy: SharedDraftPolicy(curve: .stockMLXDefault),
                                          options: SpeculativeLoop.Options(forcedDraftLength: 2))
                        : DecodeLoop(context)
                    return try SpecHarness.drain(generator).tokens
                }
                counts[tokens, default: 0] += 1
            }
            return counts
        }

        let plain = try sample(speculative: false)
        let speculative = try sample(speculative: true)
        let (statistic, degrees) = Self.homogeneity(plain, speculative, total: runs)
        let pValue = degrees > 0 ? FastSamplerTests.chiSquarePValue(statistic, degreesOfFreedom: degrees) : 1
        XCTAssertGreaterThan(pValue, 0.001, "chi-square \(statistic) on \(degrees) df")
        XCTAssertGreaterThan(plain.count, 1, "the distribution should not be degenerate")
        EngineReport.append(
            "- Unseeded first-3-token chi-square (hybrid, 2×\(runs) runs, \(plain.count) plain / \(speculative.count) speculative outcomes): "
                + "\(String(format: "%.2f", statistic)) on \(degrees) df, p = \(String(format: "%.3f", pValue))")
    }

    /// Two samples of equal size: chi-square statistic of homogeneity over the outcomes, with
    /// outcomes seen fewer than 10 times in total pooled.
    static func homogeneity(_ a: [[Int]: Int], _ b: [[Int]: Int], total: Int) -> (statistic: Double, degrees: Int) {
        var cells: [(Int, Int)] = []
        var pooled = (0, 0)
        for key in Set(a.keys).union(b.keys) {
            let x = a[key] ?? 0, y = b[key] ?? 0
            if x + y < 10 {
                pooled.0 += x
                pooled.1 += y
            } else {
                cells.append((x, y))
            }
        }
        if pooled.0 + pooled.1 > 0 { cells.append(pooled) }
        guard cells.count > 1 else { return (0, 0) }
        var statistic = 0.0
        for (x, y) in cells {
            let expected = Double(x + y) / 2
            statistic += pow(Double(x) - expected, 2) / expected + pow(Double(y) - expected, 2) / expected
        }
        _ = total
        return (statistic, cells.count - 1)
    }

    // MARK: Policy-driven behaviour

    /// Without forcing, the policy speculates on a mostly reliable drafter and the output still
    /// equals plain decoding (seeded sampling).
    func testPolicyDrivenSpeculationEqualsPlain() throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let model = try EngineTestHarness.makeModel(tiny, seed: 340, sharpen: false)
            let prompt = TinyModels.tokens(12, seed: 341)
            let sampler = FastSampler(temperature: 0.7, topP: 0.8, topK: 20, seed: 99)
            let reference = try SpecHarness.plain(model: model, prompt: prompt, maxTokens: 40, stops: [], sampler: sampler)
            let drafter = SequenceDrafter(sequence: prompt + reference.tokens, corruption: .wrongEveryThird, stop: nil)
            let run = try SpecHarness.speculative(model: model, prompt: prompt, maxTokens: 40, stops: [], drafters: [drafter], sampler: sampler)
            XCTAssertEqual(run.tokens, reference.tokens, "\(tiny)")
            XCTAssertGreaterThan(run.loop.rounds.count, 0, "\(tiny): a mostly right drafter should speculate")
            XCTAssertTrue(run.loop.rounds.allSatisfy { $0.proposed <= 4 }, "\(tiny): Kmax = 4 on the stock curve")
            let stats = try XCTUnwrap(run.loop.speculation)
            XCTAssertEqual(stats.plainTokens + stats.roundTokens, run.tokens.count, "\(tiny)")
            try SpecHarness.assertExact(run, prompt: prompt, maxTokens: 40, stops: [], label: "\(tiny) policy")
        }
    }

    /// `toolsOnly` drafts short matches only inside a tool call; critical heat and Low Power
    /// Mode turn speculation off.
    func testModesAndDeviceStateGateDrafting() throws {
        let model = try EngineTestHarness.makeModel(.qwen3, seed: 350)
        let prompt = TinyModels.tokens(12, seed: 351)
        let reference = try SpecHarness.plain(model: model, prompt: prompt, maxTokens: 24, stops: [])
        let sequence = prompt + reference.tokens

        func run(matchLength: Int, mode: LocalSpeculationMode, insideToolCall: Bool, state: (ThermalLevel, Bool)) throws -> SpecHarness.Run {
            let drafter = SequenceDrafter(sequence: sequence, corruption: .oracle, stop: nil, source: .corpus, matchLength: matchLength)
            let options = SpeculativeLoop.Options(mode: mode, deviceState: { (state.0, state.1) })
            return try SpecHarness.speculative(
                model: model, prompt: prompt, maxTokens: 24, stops: [], drafters: [drafter], options: options,
                insideToolCall: insideToolCall)
        }

        XCTAssertEqual(try run(matchLength: 2, mode: .toolsOnly, insideToolCall: false, state: (.nominal, false)).loop.rounds.count, 0)
        XCTAssertGreaterThan(try run(matchLength: 2, mode: .toolsOnly, insideToolCall: true, state: (.nominal, false)).loop.rounds.count, 0)
        XCTAssertGreaterThan(try run(matchLength: 3, mode: .toolsOnly, insideToolCall: false, state: (.nominal, false)).loop.rounds.count, 0)
        XCTAssertGreaterThan(try run(matchLength: 4, mode: .automatic, insideToolCall: false, state: (.serious, false)).loop.rounds.count, 0)
        XCTAssertEqual(try run(matchLength: 4, mode: .automatic, insideToolCall: false, state: (.critical, false)).loop.rounds.count, 0)
        XCTAssertEqual(try run(matchLength: 4, mode: .automatic, insideToolCall: false, state: (.nominal, true)).loop.rounds.count, 0)
        for state in [(ThermalLevel.critical, false), (.nominal, true)] {
            XCTAssertEqual(try run(matchLength: 4, mode: .automatic, insideToolCall: false, state: state).tokens, reference.tokens)
        }
    }

    /// Special tokens are never drafted outside exemplar structure; a drafted stop token ends
    /// the draft.
    func testSpecialTokensAreNotDrafted() throws {
        let tokenizer = FakeChatMLTokenizer()
        let stop = FakeChatMLTokenizer.imEnd
        let blocked = SpeculativeLoop.blockedDraftTokens(renderer: tokenizer, stopTokens: [stop, FakeChatMLTokenizer.endOfText])
        XCTAssertTrue(blocked.isSuperset(of: [FakeChatMLTokenizer.imStart, FakeChatMLTokenizer.endOfText, FakeChatMLTokenizer.toolCallStart, FakeChatMLTokenizer.thinkStart]))
        XCTAssertFalse(blocked.contains(stop), "<|im_end|> may be drafted")
        XCTAssertTrue(SpeculativeLoop.specialTokens(renderer: tokenizer, stopTokens: [99]).isSuperset(of: [1, 2, 3, 4, 5, 6, 7, 8, 9, 99]))

        let model = try EngineTestHarness.makeModel(.qwen3, seed: 360)
        let prompt = TinyModels.tokens(12, seed: 361)
        let reference = try SpecHarness.plain(model: model, prompt: prompt, maxTokens: 20, stops: [])
        // Every draft is [true, true, <tool_call>, …]: at most two tokens may be verified.
        let drafter = SequenceDrafter(sequence: prompt + reference.tokens, corruption: .insert(FakeChatMLTokenizer.toolCallStart, at: 2), stop: nil)
        let run = try SpecHarness.speculative(
            model: model, prompt: prompt, maxTokens: 20, stops: [], drafters: [drafter], options: SpeculativeLoop.Options(forcedDraftLength: 4))
        XCTAssertEqual(run.tokens, reference.tokens)
        XCTAssertGreaterThan(run.loop.rounds.count, 0)
        XCTAssertTrue(run.loop.rounds.allSatisfy { $0.proposed <= 2 }, "drafts are cut before <tool_call>")

        // The exemplar source may draft structure.
        let exemplar = SequenceDrafter(
            sequence: prompt + reference.tokens, corruption: .insert(FakeChatMLTokenizer.toolCallStart, at: 2), stop: nil, source: .exemplar)
        let structural = try SpecHarness.speculative(
            model: model, prompt: prompt, maxTokens: 20, stops: [], drafters: [exemplar], options: SpeculativeLoop.Options(forcedDraftLength: 4),
            insideToolCall: true)
        XCTAssertEqual(structural.tokens, reference.tokens)
        XCTAssertTrue(structural.loop.rounds.contains { $0.proposed > 2 }, "exemplar drafts keep special tokens")
    }

    // MARK: Through the engine

    /// The factory's speculative replies equal a plain engine's, turn after turn, and report
    /// their speculation statistics; the session stays consistent.
    func testEngineRepliesWithTheFactoryEqualPlain() async throws {
        for tiny in EngineTestHarness.Tiny.allCases {
            let plain = try EngineTestHarness.makeEngine(tiny: tiny, seed: 370, configuration: EngineTestHarness.testConfiguration(maxTokens: 40))
            var configuration = EngineTestHarness.testConfiguration(maxTokens: 40)
            let factory = SpeculativeGeneratorFactory(mode: .automatic, curve: .stockMLXDefault, corpusURL: nil)
            configuration.generatorFactory = factory
            let speculative = try EngineTestHarness.makeEngine(tiny: tiny, seed: 370, configuration: configuration)
            var turns: [ChatTurn] = []
            var rounds = 0
            for (index, text) in ["Say hello hello hello hello.", "Repeat: abc abc abc abc abc abc.", "Once more, abc abc abc abc."].enumerated() {
                turns.append(ChatTurn(role: .user, text: text))
                let request = EngineRequest(system: "You are Alvin.", turns: turns)
                let expected = try await EngineTestHarness.collect(plain.reply(request))
                let actual = try await EngineTestHarness.collect(speculative.reply(request))
                XCTAssertEqual(EngineTestHarness.text(actual), EngineTestHarness.text(expected), "\(tiny) turn \(index + 1)")
                XCTAssertEqual(EngineTestHarness.finish(actual)?.reason, EngineTestHarness.finish(expected)?.reason, "\(tiny) turn \(index + 1)")
                let stats = try XCTUnwrap(EngineTestHarness.finish(actual)?.stats.speculation, "\(tiny) turn \(index + 1)")
                rounds += stats.rounds
                XCTAssertEqual(stats.plainTokens + stats.roundTokens, EngineTestHarness.finish(actual)?.stats.generatedTokens)
                let (expectedLedger, actualLedger) = (
                    try await plain.withSession { $0.ledger }, try await speculative.withSession { $0.ledger })
                XCTAssertEqual(actualLedger, expectedLedger, "\(tiny) turn \(index + 1)")
                let report = try await speculative.withSession { $0.assertConsistent() }
                XCTAssertTrue(report.isConsistent(), "\(tiny) turn \(index + 1): \(report)")
                turns.append(ChatTurn(role: .assistant, text: EngineTestHarness.text(expected)))
            }
            EngineReport.append("- Engine with SpeculativeGeneratorFactory (\(tiny)): 3 turns equal plain, \(rounds) rounds")
        }
    }

    /// A scripted tool call: the exemplar drafter drafts its structure, and the events equal a
    /// plain engine's.
    func testExemplarDraftsAScriptedToolCall() async throws {
        let tokenizer = FakeChatMLTokenizer()
        let tools = [
            ToolDefinition(
                name: "set_timer", description: "Starts a timer.",
                inputSchema: ["type": "object", "properties": ["seconds": ["type": "integer"]], "required": ["seconds"]]),
        ]
        let call = "<tool_call>\n{\"name\": \"set_timer\", \"arguments\": {\"seconds\": 600}}\n</tool_call>"
        let scripts = [tokenizer.encodeRaw("Sure. " + call), tokenizer.encodeRaw("Timer set.")]
        let request = EngineRequest(system: "You are Alvin.", tools: tools, turns: [ChatTurn(role: .user, text: "Ten minute timer")])

        let (plain, _) = try EngineTestHarness.makeScriptedEngine(tiny: .qwen3, scripts: scripts)
        var configuration = EngineTestHarness.testConfiguration()
        configuration.generatorFactory = SpeculativeGeneratorFactory(mode: .toolsOnly, curve: .stockMLXDefault, corpusURL: nil)
        let (speculative, _) = try EngineTestHarness.makeScriptedEngine(tiny: .qwen3, scripts: scripts, configuration: configuration)

        let expected = try await EngineTestHarness.collect(plain.reply(request))
        let actual = try await EngineTestHarness.collect(speculative.reply(request))
        XCTAssertEqual(EngineTestHarness.kinds(actual), EngineTestHarness.kinds(expected))
        XCTAssertEqual(EngineTestHarness.text(actual), "Sure. ")
        XCTAssertEqual(EngineTestHarness.toolCalls(actual).map(\.name), ["set_timer"])
        XCTAssertEqual(EngineTestHarness.toolCalls(actual).first?.input, ["seconds": 600])
        let stats = try XCTUnwrap(EngineTestHarness.finish(actual)?.stats.speculation)
        XCTAssertGreaterThan(stats.accepted[DraftSource.exemplar.rawValue] ?? 0, 0, "the skeleton was drafted and accepted")

        let round = ToolRound(calls: EngineTestHarness.toolCalls(actual).map { ToolCallRecord(call: $0, output: .ok(["ok": true], summary: "10 min")) })
        let second = try await EngineTestHarness.collect(speculative.continueReply(after: round))
        XCTAssertEqual(EngineTestHarness.text(second), "Timer set.")
        let report = try await speculative.withSession { $0.assertConsistent() }
        XCTAssertTrue(report.isConsistent(), "\(report)")
    }

    /// `.off` and a target that can't roll back get the plain `DecodeLoop`.
    func testFactoryFallsBackToPlainDecoding() throws {
        let model = try EngineTestHarness.makeModel(.qwen3, seed: 380)
        let session = LiveSession(target: StockTarget(model: model))
        let logits = session.feed(TinyModels.tokens(6, seed: 381), rows: .last).logits!
        let context = SpecHarness.context(session: session, firstLogits: logits, sampler: FastSampler(temperature: 0), stops: [], maxTokens: 4)
        let off = SpeculativeGeneratorFactory(mode: .off, curve: .stockMLXDefault, corpusURL: nil)
        XCTAssertTrue(off.makeGenerator(context) is DecodeLoop)
        let on = SpeculativeGeneratorFactory(mode: .automatic, curve: .stockMLXDefault, corpusURL: nil)
        XCTAssertTrue(on.makeGenerator(context) is SpeculativeLoop)

        // The stock hybrid can't roll a round back.
        let stockHybrid = try TinyModels.makeStockHybrid(seed: 382)
        let hybridSession = LiveSession(target: StockTarget(model: stockHybrid))
        XCTAssertFalse(hybridSession.target.supportsRollback)
        let hybridLogits = hybridSession.feed(TinyModels.tokens(6, seed: 383), rows: .last).logits!
        let hybridContext = SpecHarness.context(session: hybridSession, firstLogits: hybridLogits, sampler: FastSampler(temperature: 0), stops: [], maxTokens: 4)
        XCTAssertTrue(on.makeGenerator(hybridContext) is DecodeLoop)
    }
}

// MARK: - Shared helpers for the speculation tests

/// Builds sessions, contexts and loops over the tiny models, and checks a run's invariants.
enum SpecHarness {
    /// How the reference run ends.
    enum Scenario: CustomStringConvertible {
        /// No stop token in the output: the length limit ends it.
        case length
        /// A token the free run produces partway ends it.
        case stop

        var description: String { self == .length ? "length" : "stop" }
    }

    /// A stop token for a run whose free (stop-less) output is `free`: the first token that
    /// appears for the first time at position 10 or later, else the latest first appearance.
    static func stopToken(in free: [Int]) -> Int {
        var firstSeen: [Int: Int] = [:]
        for (index, token) in free.enumerated() where firstSeen[token] == nil {
            firstSeen[token] = index
        }
        if let late = firstSeen.filter({ $0.value >= 10 }).min(by: { $0.value < $1.value }) {
            return late.key
        }
        return firstSeen.max(by: { $0.value < $1.value })?.key ?? 0
    }

    struct Run {
        let tokens: [Int]
        let reason: EngineFinish.Reason
        let session: LiveSession
        let loop: SpeculativeLoop
        let confidence: ConfidenceRecorder
        /// What the session fed the target to verify, and what it kept.
        let recorder: RecordingTarget
    }

    struct PlainRun {
        let tokens: [Int]
        let reason: EngineFinish.Reason
        let session: LiveSession
    }

    static func target(for model: any LanguageModel) -> any TargetModel {
        if let fork = model as? any HybridQwen35Forwarding {
            return HybridTarget(model: fork)
        }
        return StockTarget(model: model)
    }

    static func context(
        session: LiveSession, firstLogits: MLXArray, sampler: FastSampler, stops: Set<Int>, maxTokens: Int,
        insideToolCall: Bool = false, confidence: ConfidenceRecorder = ConfidenceRecorder()
    ) -> GeneratorContext {
        GeneratorContext(
            session: session, sampler: sampler, firstLogits: firstLogits, stopTokens: stops, maxTokens: maxTokens,
            request: EngineRequest(system: "", turns: []), drafters: [], insideToolCall: { insideToolCall }, isAllowed: { true },
            renderer: FakeChatMLTokenizer(), toolCallFormat: .json, confidence: confidence)
    }

    /// Steps `generator` to its end and flushes it.
    static func drain(_ generator: TokenGenerator) throws -> (tokens: [Int], reason: EngineFinish.Reason) {
        var tokens: [Int] = []
        while true {
            let (emitted, finished) = try generator.step()
            tokens += emitted
            if let finished {
                try generator.flush()
                return (tokens, finished)
            }
        }
    }

    static func plain(
        model: any LanguageModel, prompt: [Int], maxTokens: Int, stops: Set<Int>, sampler: FastSampler = FastSampler(temperature: 0)
    ) throws -> PlainRun {
        let session = LiveSession(target: target(for: model))
        let logits = session.feed(prompt, rows: .last).logits!
        let loop = DecodeLoop(context(session: session, firstLogits: logits, sampler: sampler, stops: stops, maxTokens: maxTokens))
        let (tokens, reason) = try drain(loop)
        return PlainRun(tokens: tokens, reason: reason, session: session)
    }

    static func speculative(
        model: any LanguageModel, prompt: [Int], maxTokens: Int, stops: Set<Int>, drafters: [any Drafter],
        options: SpeculativeLoop.Options = SpeculativeLoop.Options(), sampler: FastSampler = FastSampler(temperature: 0),
        insideToolCall: Bool = false, policy: SharedDraftPolicy = SharedDraftPolicy(curve: .stockMLXDefault)
    ) throws -> Run {
        let recorder = RecordingTarget(target(for: model))
        let session = LiveSession(target: recorder)
        let logits = session.feed(prompt, rows: .last).logits!
        let request = EngineRequest(system: "", turns: [])
        for drafter in drafters {
            drafter.reset(ledger: session.ledger, request: request)
        }
        let confidence = ConfidenceRecorder()
        let loop = SpeculativeLoop(
            context(session: session, firstLogits: logits, sampler: sampler, stops: stops, maxTokens: maxTokens,
                    insideToolCall: insideToolCall, confidence: confidence),
            drafters: drafters, policy: policy, options: options)
        let (tokens, reason) = try drain(loop)
        return Run(tokens: tokens, reason: reason, session: session, loop: loop, confidence: confidence, recorder: recorder)
    }

    /// The invariants every run keeps, whatever it drafted.
    static func assertExact(_ run: Run, prompt: [Int], maxTokens: Int, stops: Set<Int>, label: String,
                            file: StaticString = #filePath, line: UInt = #line) throws {
        XCTAssertLessThanOrEqual(run.tokens.count, maxTokens, "\(label): maxTokens", file: file, line: line)
        XCTAssertFalse(run.tokens.contains { stops.contains($0) }, "\(label): stop tokens are never emitted", file: file, line: line)
        // Every emitted token is in the cache exactly once, in order; a stop ends it.
        var expected = prompt + run.tokens
        if run.reason == .stop, let last = run.session.ledger.last, stops.contains(last) {
            expected.append(last)
        }
        XCTAssertEqual(run.session.ledger, expected, "\(label): the ledger is the prompt and the emitted tokens", file: file, line: line)
        if run.reason == .stop {
            XCTAssertTrue(stops.contains(run.session.ledger.last ?? -1), "\(label): the stop token is fed", file: file, line: line)
        }
        XCTAssertEqual(run.session.pendingCount, 0, file: file, line: line)

        // The counts, checked against what the target was really given rather than against the
        // loop's own bookkeeping: each round is one verify forward of `[y] + drafts` (proposed =
        // the drafts fed) and one commit (accepted = the drafts kept, rejected = the rows rolled
        // back), and what it kept is in the ledger at that position.
        let rounds = run.loop.rounds
        let verifies = run.recorder.verifies
        let commits = run.recorder.commits
        XCTAssertEqual(verifies.count, rounds.count, "\(label): one verify forward per round", file: file, line: line)
        XCTAssertEqual(commits.count, rounds.count, "\(label): one commit per round", file: file, line: line)
        var fed = 0, kept = 0, rolledBack = 0
        for (index, (verify, commit)) in zip(verifies, commits).enumerated() {
            let roundLabel = "\(label) round \(index)"
            XCTAssertEqual(commit.verified, verify.tokens.count, "\(roundLabel): commits the rows it verified", file: file, line: line)
            XCTAssertTrue(commit.keep >= 1 && commit.keep <= commit.verified, "\(roundLabel): keeps y and at most every row", file: file, line: line)
            fed += verify.tokens.count - 1
            kept += commit.keep - 1
            rolledBack += commit.verified - commit.keep
            if let position = verify.position {
                let end = position + commit.keep
                XCTAssertTrue(
                    end <= run.session.ledger.count && Array(run.session.ledger[position ..< end]) == Array(verify.tokens.prefix(commit.keep)),
                    "\(roundLabel): the kept rows are what the ledger holds there", file: file, line: line)
            }
            if index < rounds.count {
                let round = rounds[index]
                XCTAssertEqual(round.proposed, verify.tokens.count - 1, "\(roundLabel): proposed = the drafts fed", file: file, line: line)
                XCTAssertEqual(round.accepted, commit.keep - 1, "\(roundLabel): accepted = the drafts kept", file: file, line: line)
                XCTAssertEqual(round.rejected, commit.verified - commit.keep, "\(roundLabel): rejected = the rows rolled back", file: file, line: line)
                XCTAssertEqual(round.kept, commit.keep, file: file, line: line)
                XCTAssertLessThanOrEqual(round.emitted, round.accepted + 1, file: file, line: line)
            }
        }
        let stats = try XCTUnwrap(run.loop.speculation, file: file, line: line)
        XCTAssertEqual(stats.rounds, verifies.count, file: file, line: line)
        XCTAssertEqual(stats.totalDrafted, fed, "\(label): drafted = the drafts fed", file: file, line: line)
        XCTAssertEqual(stats.totalAccepted, kept, "\(label): accepted = the drafts kept", file: file, line: line)
        XCTAssertEqual(
            stats.totalAccepted + rolledBack, fed,
            "\(label): accepted + rejected == proposed (accepted as reported, rejected as rolled back, proposed as fed)",
            file: file, line: line)
        XCTAssertEqual(stats.plainTokens + stats.roundTokens, run.tokens.count, "\(label): every token counted once", file: file, line: line)
        let report = run.session.assertConsistent()
        XCTAssertTrue(report.isConsistent(), "\(label): \(report)", file: file, line: line)
    }
}

/// A target that records what the session asks of it: every verify forward (`.all` with a
/// rollback capture: a speculative round's `[y] + drafts`) with the cache position it started
/// at, and every commit. The speculation tests derive the round counts from these.
final class RecordingTarget: TargetModel {
    struct Verify {
        let tokens: [Int]
        /// The attention cache's offset before the forward (nil without attention layers).
        let position: Int?
    }

    struct Commit {
        let keep: Int
        let verified: Int
    }

    let inner: any TargetModel
    private(set) var verifies: [Verify] = []
    private(set) var commits: [Commit] = []

    init(_ inner: any TargetModel) {
        self.inner = inner
    }

    var model: any LanguageModel { inner.model }
    var cache: [KVCache] { inner.cache }
    var layout: CacheLayout { inner.layout }
    var vocabularySize: Int { inner.vocabularySize }
    var supportsRollback: Bool { inner.supportsRollback }
    var supportsRowSelection: Bool { inner.supportsRowSelection }

    func forward(_ tokens: MLXArray, rows: LogitRows, captureForRollback: Bool, wantHidden: Bool) -> ForwardResult {
        if rows == .all && captureForRollback {
            let values = tokens.asType(.int32).asArray(Int32.self).map { Int($0) }
            let position = inner.layout.attention.first.map { inner.cache[$0].offset }
            verifies.append(Verify(tokens: values, position: position))
        }
        return inner.forward(tokens, rows: rows, captureForRollback: captureForRollback, wantHidden: wantHidden)
    }

    func commit(_ capture: (any RoundCapture)?, keep: Int, of verified: Int) {
        commits.append(Commit(keep: keep, verified: verified))
        inner.commit(capture, keep: keep, of: verified)
    }

    func resetCache() {
        inner.resetCache()
    }

    func adopt(_ cache: [KVCache]) throws {
        try inner.adopt(cache)
    }
}

/// A drafter that knows the reference sequence (prompt + plain continuation) and proposes its
/// continuation after any context that is a prefix of it, corrupted on purpose.
final class SequenceDrafter: Drafter {
    enum Corruption: Equatable, CustomStringConvertible {
        /// The perfect oracle.
        case oracle
        /// Every drafted token is wrong.
        case alwaysWrong
        /// Every third position (by absolute position) is wrong.
        case wrongEveryThird
        /// A stop token replaces the draft at depth 1 or 2 (by position), except where the
        /// reference really stops.
        case adversarialStop
        /// `token` inserted at depth `at` (0-based).
        case insert(Int, at: Int)

        var description: String {
            switch self {
            case .oracle: return "oracle"
            case .alwaysWrong: return "alwaysWrong"
            case .wrongEveryThird: return "wrongEveryThird"
            case .adversarialStop: return "adversarialStop"
            case .insert(let token, let at): return "insert(\(token)@\(at))"
            }
        }
    }

    let sequence: [Int]
    let corruption: Corruption
    let stop: Int?
    /// Proposed when the context has left the sequence (nil: no proposal).
    let fallback: [Int]?
    let source: DraftSource
    let matchLength: Int
    var costPerToken: Double { 0 }
    var wantsHidden: Bool { false }
    private(set) var observations: [RoundObservation] = []

    init(sequence: [Int], corruption: Corruption, stop: Int?, fallback: [Int]? = nil, source: DraftSource = .mtp, matchLength: Int = 4) {
        self.sequence = sequence
        self.corruption = corruption
        self.stop = stop
        self.fallback = fallback
        self.source = source
        self.matchLength = matchLength
    }

    func reset(ledger: [Int], request: EngineRequest) {}

    func propose(context: ArraySlice<Int>, maxTokens: Int) -> DraftProposal? {
        guard maxTokens > 0 else { return nil }
        let start = context.count
        let onSequence = start <= sequence.count && context.elementsEqual(sequence[..<start])
        var tokens: [Int]
        if onSequence {
            tokens = Array(sequence[start...].prefix(maxTokens))
            if tokens.isEmpty {
                // Past the reference's end: anything goes.
                tokens = [((sequence.last ?? 0) + 1) % TinyModels.vocabularySize]
            }
        } else if let fallback {
            tokens = Array(fallback.prefix(maxTokens))
        } else {
            return nil
        }
        switch corruption {
        case .oracle:
            break
        case .alwaysWrong:
            tokens = tokens.map { ($0 + 1) % TinyModels.vocabularySize }
        case .wrongEveryThird:
            for index in tokens.indices where (start + index) % 3 == 2 {
                tokens[index] = (tokens[index] + 1) % TinyModels.vocabularySize
            }
        case .adversarialStop:
            if let stop {
                let depth = 1 + start % 2
                if depth < tokens.count, tokens[depth] != stop {
                    tokens[depth] = stop
                }
            }
        case .insert(let token, let at):
            if at <= tokens.count {
                tokens.insert(token, at: at)
                tokens = Array(tokens.prefix(maxTokens))
            }
        }
        return tokens.isEmpty ? nil : DraftProposal(tokens: tokens, source: source, matchLength: matchLength)
    }

    func observe(_ round: RoundObservation) {
        observations.append(round)
    }
}
