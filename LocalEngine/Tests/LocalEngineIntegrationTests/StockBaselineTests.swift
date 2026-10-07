import Foundation
import LocalEngine
import LocalEngineTestSupport
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

/// Today's stock path on real models: the system-prompt duplication the engine fixes (F1), the
/// tokenizer bridge the engine renders with, and the stock decode speed it must keep up with.
final class StockBaselineTests: XCTestCase {
    private static let qwen3 = "mlx-community/Qwen3-0.6B-4bit"
    private static let qwen35 = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
    private static let chatContext: [String: any Sendable] = ["enable_thinking": false]

    /// About the size of the app's real system prompt, so the duplication is unmistakable.
    private static let system = """
        You are Alvin, a personal voice assistant on the user's iPhone. Answer in one or two short \
        sentences unless asked for more. Be warm and direct. Use plain words, no lists or markdown, \
        because your answer is read aloud. If you are unsure, say so briefly. Never invent facts \
        about the user. Prefer metric units and the 24-hour clock unless the user uses others.
        """

    override func setUpWithError() throws {
        try IntegrationEnvironment.requireEnabled()
        try MetalAvailability.require()
    }

    private func load(_ repo: String) async throws -> LoadedModel {
        let directory = try await IntegrationEnvironment.snapshot(repo)
        return try await ModelLoader.load(directory: directory, id: repo)
    }

    /// F1: `ChatSession(instructions:)` re-sends the system prompt on every turn, so turn 2
    /// prefills it again; with `history: [.system(S)]` it is sent once.
    func testChatSessionInstructionsRepeatTheSystemPromptEveryTurn() async throws {
        let loaded = try await load(Self.qwen3)
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0)

        let withInstructions = ChatSession(
            loaded.container, instructions: Self.system, generateParameters: parameters,
            additionalContext: Self.chatContext)
        let instructionsTurn1 = try await promptTokens(withInstructions, "Hi there.")
        let instructionsTurn2 = try await promptTokens(withInstructions, "What can you do?")

        let withHistory = ChatSession(
            loaded.container, instructions: nil, history: [.system(Self.system)],
            generateParameters: parameters, additionalContext: Self.chatContext)
        let historyTurn1 = try await promptTokens(withHistory, "Hi there.")
        let historyTurn2 = try await promptTokens(withHistory, "What can you do?")

        let systemTokens = loaded.renderer.encodeRaw(Self.system).count
        EngineReport.appendTable(
            title: "F1: ChatSession prompt tokens per turn (Qwen3-0.6B-4bit, system prompt \(systemTokens) tokens)",
            header: ["Session", "Turn 1", "Turn 2"],
            rows: [
                ["instructions: S", "\(instructionsTurn1)", "\(instructionsTurn2)"],
                ["instructions: nil, history: [.system(S)]", "\(historyTurn1)", "\(historyTurn2)"],
            ])

        XCTAssertEqual(instructionsTurn1, historyTurn1, "turn 1 renders the same messages either way")
        XCTAssertGreaterThanOrEqual(instructionsTurn2 - historyTurn2, systemTokens, "turn 2 with instructions should prefill S again")
        XCTAssertLessThan(historyTurn2, systemTokens, "turn 2 with history should not include S")
    }

    /// The bridge renders without the generation prompt (F13), and the stop set holds `<|im_end|>`.
    func testTokenizerBridgeRendersWithoutGenerationPrompt() async throws {
        let loaded = try await load(Self.qwen3)
        XCTAssertTrue(loaded.renderer is TokenizerBridge)
        let system: [String: any Sendable] = ["role": "system", "content": Self.system]
        let user: [String: any Sendable] = ["role": "user", "content": "Hi there."]

        let systemOnly = try loaded.renderer.renderTokens(messages: [system], tools: nil, context: Self.chatContext, addGenerationPrompt: false)
        let withUser = try loaded.renderer.renderTokens(messages: [system, user], tools: nil, context: Self.chatContext, addGenerationPrompt: false)
        let withPrompt = try loaded.renderer.renderTokens(messages: [system, user], tools: nil, context: Self.chatContext, addGenerationPrompt: true)
        XCTAssertEqual(Array(withUser.prefix(systemOnly.count)), systemOnly)
        XCTAssertEqual(Array(withPrompt.prefix(withUser.count)), withUser)
        XCTAssertGreaterThan(withPrompt.count, withUser.count)

        let imStart = try XCTUnwrap(loaded.renderer.tokenID("<|im_start|>"))
        let imEnd = try XCTUnwrap(loaded.renderer.tokenID("<|im_end|>"))
        // The system prefix is everything before the last <|im_start|> (plan §4.5).
        XCTAssertEqual(withUser.lastIndex(of: imStart), systemOnly.count)
        XCTAssertTrue(loaded.stopTokenIDs.contains(imEnd))
        XCTAssertNil(loaded.renderer.tokenID("<|not a token|>"))
        XCTAssertTrue(loaded.renderer.isChatML)

        let generationPrompt = loaded.renderer.decodeRaw(Array(withPrompt.suffix(withPrompt.count - withUser.count)))
        let toolCallFormat = loaded.configuration.toolCallFormat.map { String(describing: $0) } ?? "default"
        EngineReport.append("""

            ### Tokenizer bridge (Qwen3-0.6B-4bit)

            - System-only render: \(systemOnly.count) tokens; with the user turn: \(withUser.count); with the generation prompt: \(withPrompt.count)
            - Generation prompt: `\(generationPrompt.debugDescription)`
            - Stop tokens: \(loaded.stopTokenIDs.sorted()); tool-call format: \(toolCallFormat)
            """)
    }

    /// Stock `TokenIterator` greedy decode speed: the baseline the engine's plain decode must
    /// keep (WP20 requires at least 0.8×).
    func testStockTokenIteratorThroughput() async throws {
        var rows: [[String]] = []
        for repo in [Self.qwen3, Self.qwen35] {
            let loaded = try await load(repo)
            let user: [String: any Sendable] = ["role": "user", "content": "Tell me a short story about a robot who learns to paint."]
            let prompt = try loaded.renderer.renderTokens(messages: [user], tools: nil, context: Self.chatContext, addGenerationPrompt: true)

            _ = try decode(loaded, prompt: prompt, count: 8)  // warm-up: compiles the kernels
            let (generated, seconds) = try decode(loaded, prompt: prompt, count: 64)
            let tokensPerSecond = Double(generated) / seconds
            rows.append([repo, loaded.modelType, "\(prompt.count)", "\(generated)", String(format: "%.1f", tokensPerSecond)])
            XCTAssertEqual(generated, 64)
            XCTAssertGreaterThan(tokensPerSecond, 0)
        }
        EngineReport.appendTable(
            title: "Stock TokenIterator greedy decode (64 tokens after the first; prefill not counted)",
            header: ["Model", "model_type", "Prompt tokens", "Generated", "tok/s"],
            rows: rows)
    }

    /// `ModelLoader` builds the model with the given type registry when it knows the
    /// `model_type`, falls back to the stock factory when that model code throws
    /// `EngineModelError.unsupported`, and ignores a registry that doesn't know the type.
    func testModelLoaderUsesTheGivenRegistryOrFallsBack() async throws {
        let directory = try await IntegrationEnvironment.snapshot(Self.qwen3)

        let accepted = CallCounter()
        let accepting = ModelTypeRegistry<LanguageModel>(creators: [
            "qwen3": { data in
                accepted.increment()
                return Qwen3Model(try JSONDecoder.json5().decode(Qwen3Configuration.self, from: data))
            },
        ])
        let viaRegistry = try await ModelLoader.load(directory: directory, id: Self.qwen3, typeRegistry: accepting)
        XCTAssertEqual(accepted.count, 1)
        XCTAssertTrue(viaRegistry.model is Qwen3Model)
        XCTAssertEqual(viaRegistry.modelType, "qwen3")

        let declined = CallCounter()
        let declining = ModelTypeRegistry<LanguageModel>(creators: [
            "qwen3": { _ in
                declined.increment()
                throw EngineModelError.unsupported("declined by the test")
            },
        ])
        let viaFallback = try await ModelLoader.load(directory: directory, id: Self.qwen3, typeRegistry: declining)
        XCTAssertEqual(declined.count, 1)
        XCTAssertTrue(viaFallback.model is Qwen3Model)

        let unrelated = CallCounter()
        let other = ModelTypeRegistry<LanguageModel>(creators: [
            "qwen3_5": { _ in
                unrelated.increment()
                throw EngineModelError.unsupported("not this model type")
            },
        ])
        let viaStock = try await ModelLoader.load(directory: directory, id: Self.qwen3, typeRegistry: other)
        XCTAssertEqual(unrelated.count, 0)
        XCTAssertTrue(viaStock.model is Qwen3Model)
        XCTAssertEqual(viaStock.stopTokenIDs, viaRegistry.stopTokenIDs)
    }

    // MARK: Helpers

    private func promptTokens(_ session: ChatSession, _ prompt: String) async throws -> Int {
        var count: Int?
        for try await item in session.streamDetails(to: prompt) {
            if case .info(let info) = item {
                count = info.promptTokenCount
            }
        }
        return try XCTUnwrap(count, "no completion info for “\(prompt)”")
    }

    /// Prefills `prompt` on a fresh cache and takes the first token, which waits for the prefill,
    /// then times `count` more greedy tokens. Stop tokens don't end the run.
    private func decode(_ loaded: LoadedModel, prompt: [Int], count: Int) throws -> (generated: Int, seconds: Double) {
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt)), model: loaded.model, cache: nil,
            parameters: GenerateParameters(maxTokens: count + 1, temperature: 0))
        guard iterator.next() != nil else { return (0, 0) }
        let start = Date()
        var generated = 0
        while iterator.next() != nil {
            generated += 1
        }
        return (generated, Date().timeIntervalSince(start))
    }
}

/// Counts calls from model-creator closures, which may run on any thread.
private final class CallCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    func increment() {
        lock.lock()
        defer { lock.unlock() }
        value += 1
    }
}
