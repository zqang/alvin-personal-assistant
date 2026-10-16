import AssistantKit
import XCTest
import Darwin
@testable import MiniCPM

final class MiniCPMSessionTests: XCTestCase {
    func testRuntimeConfigurationReadsExistingReports() throws {
        let legacy = Data(#"{"name":"balanced","checkpointLayerInterval":4,"draftMaximum":3,"expertCacheSize":48,"directGreedy":true}"#.utf8)
        XCTAssertEqual(try JSONDecoder().decode(MiniCPMRuntimeConfiguration.self, from: legacy), .balanced)
    }

    func testReal35BRepeatedBalancedUnloadRetiresMemory() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["MINICPM_BENCH_35B_MODEL"],
              let expertsPath = env["MINICPM_BENCH_35B_EXPERTS"],
              let draftPath = env["MINICPM_BENCH_35B_DRAFT"] else {
            throw XCTSkip("Set the real 35B artifact paths to check repeated model retirement.")
        }
        func footprint() -> UInt64 {
            var info = task_vm_info_data_t()
            var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
            let status = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                    task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
                }
            }
            return status == KERN_SUCCESS ? info.phys_footprint : 0
        }
        let initial = footprint()
        XCTAssertGreaterThan(initial, 0)
        for cycle in 1...2 {
            let session = MiniCPMSession()
            try await session.load(modelURL: URL(fileURLWithPath: modelPath), projectorURL: nil,
                expertsURL: URL(fileURLWithPath: expertsPath), draftURL: URL(fileURLWithPath: draftPath), configuration: .balanced)
            _ = try await session.run(prompt: "Write a numbered list of three practical uses for an on-device personal assistant.", maxTokens: 64)
            let active = footprint()
            await session.unload()
            let immediate = footprint()
            var retired = immediate
            for _ in 0..<100 {
                if retired < initial + 512 * 1024 * 1024 { break }
                try await Task.sleep(nanoseconds: 100_000_000)
                retired = footprint()
            }
            print("PROFILE_MEMORY cycle=\(cycle) initial=\(initial) active=\(active) immediate=\(immediate) retired=\(retired)")
            XCTAssertGreaterThan(active, initial + 512 * 1024 * 1024)
            XCTAssertLessThan(retired, initial + 512 * 1024 * 1024, "An unloaded model must not retain gigabytes of Metal resources")
        }
    }

    func testCancellationBeforeLoadStaysLatched() async throws {
        let session = MiniCPMSession()
        session.cancel()
        XCTAssertTrue(session.waitUntilIdle(timeout: 0))
        do {
            try await session.load(modelURL: URL(fileURLWithPath: "/missing-model.gguf"),
                                   projectorURL: URL(fileURLWithPath: "/missing-projector.gguf"))
            XCTFail("Cancelled sessions must not load")
        } catch is CancellationError {} catch { XCTFail("Expected cancellation, got \(error)") }
        await session.unload()
    }

    func testUnloadedSessionRejectsRun() async {
        do {
            _ = try await MiniCPMSession().run(prompt: "Hello")
            XCTFail("An unloaded session must not run")
        } catch MiniCPMError.notLoaded {} catch { XCTFail("Unexpected error: \(error)") }
    }

    func testDraftCannotSilentlyRunWithoutLES() async throws {
        let model = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try Data([1]).write(to: model)
        defer { try? FileManager.default.removeItem(at: model) }
        do {
            try await MiniCPMSession().load(modelURL: model, projectorURL: nil, draftURL: model)
            XCTFail("A requested DSpark draft must require LES before entering native loading")
        } catch MiniCPMError.invalidInput(let message) {
            XCTAssertTrue(message.contains("LES"))
        } catch { XCTFail("Unexpected error: \(error)") }
    }

    func testReal35BDSparkMatchesGreedyTarget() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["MINICPM_BENCH_35B_MODEL"],
              let expertsPath = env["MINICPM_BENCH_35B_EXPERTS"],
              let draftPath = env["MINICPM_BENCH_35B_DRAFT"] else {
            throw XCTSkip("Set MINICPM_BENCH_35B_MODEL/EXPERTS/DRAFT to compare real LES and DSpark output.")
        }
        let prompts = ["Return only the number: 17 plus 26.",
            "请用一句中文解释为什么手机上的离线助手能保护隐私。",
            "Write a numbered list of three practical uses for an on-device personal assistant."]
        var baseline: [MiniCPMBenchmarkResult] = []
        let target = MiniCPMSession()
        try await target.load(modelURL: URL(fileURLWithPath: modelPath), projectorURL: nil,
                              expertsURL: URL(fileURLWithPath: expertsPath))
        for prompt in prompts { baseline.append(try await target.run(prompt: prompt, maxTokens: 64)) }
        await target.unload()

        let speculative = MiniCPMSession()
        try await speculative.load(modelURL: URL(fileURLWithPath: modelPath), projectorURL: nil,
            expertsURL: URL(fileURLWithPath: expertsPath), draftURL: URL(fileURLWithPath: draftPath))
        var drafted = 0
        var accepted = 0
        for (index, prompt) in prompts.enumerated() {
            let actual = try await speculative.run(prompt: prompt, maxTokens: 64)
            XCTAssertTrue(actual.lesEnabled)
            XCTAssertTrue(actual.dsparkEnabled)
            XCTAssertEqual(actual.text, baseline[index].text, "Greedy target verification must preserve output for prompt \(index)")
            XCTAssertEqual(actual.generatedTokens, baseline[index].generatedTokens)
            XCTAssertEqual(actual.finishReason, baseline[index].finishReason)
            drafted += actual.draftTokens
            accepted += actual.acceptedDraftTokens
        }
        XCTAssertGreaterThan(drafted, 0, "DSpark must actually propose draft tokens")
        XCTAssertGreaterThan(accepted, 0, "At least one draft token should be accepted by the target")
        for limit in [1, 2] {
            let bounded = try await speculative.run(prompt: prompts[2], maxTokens: limit)
            XCTAssertEqual(bounded.generatedTokens, limit)
            XCTAssertEqual(bounded.finishReason, "tokenLimit")
            XCTAssertEqual(bounded.draftTokens, 0, "Tiny budgets should not verify unused draft rows")
        }
        // Exercise a cancellation after both target and draft contexts have run.
        let active = Task { try await speculative.run(prompt: String(repeating: prompts[2] + " ", count: 20), maxTokens: 128) }
        var observedWork = false
        for _ in 0..<200 {
            if !speculative.waitUntilIdle(timeout: 0) { observedWork = true; break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertTrue(observedWork)
        speculative.cancel()
        XCTAssertTrue(speculative.waitUntilIdle(timeout: 3), "Both engines must retire on cancellation")
        do { _ = try await active.value; XCTFail("Expected cancellation") }
        catch is CancellationError {} catch { XCTFail("Unexpected cancellation error: \(error)") }
        await speculative.unload()
    }

    func testReal35BOptimizedProfilesPreserveGreedyOutputAndCancellation() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["MINICPM_BENCH_35B_MODEL"],
              let expertsPath = env["MINICPM_BENCH_35B_EXPERTS"],
              let draftPath = env["MINICPM_BENCH_35B_DRAFT"] else {
            throw XCTSkip("Set the real 35B artifact paths to validate optimization profiles.")
        }
        let prompts = ["Return only the number: 17 plus 26.",
            "请用一句中文解释为什么手机上的离线助手能保护隐私。",
            "Write a numbered list of three practical uses for an on-device personal assistant."]
        var baseline: [MiniCPMBenchmarkResult] = []
        for profile in [MiniCPMRuntimeConfiguration.baseline, .balanced, .checkpoints8, .draft2] {
            let session = MiniCPMSession()
            try await session.load(modelURL: URL(fileURLWithPath: modelPath), projectorURL: nil,
                expertsURL: URL(fileURLWithPath: expertsPath), draftURL: URL(fileURLWithPath: draftPath), configuration: profile)
            for (index, prompt) in prompts.enumerated() {
                let actual = try await session.run(prompt: prompt, maxTokens: 64)
                XCTAssertEqual(actual.runtimeConfiguration, profile)
                if profile == .baseline { baseline.append(actual) }
                else {
                    XCTAssertEqual(actual.text, baseline[index].text, "Profile \(profile.name) changed greedy text for prompt \(index)")
                    XCTAssertEqual(actual.generatedTokens, baseline[index].generatedTokens)
                    XCTAssertEqual(actual.finishReason, baseline[index].finishReason)
                }
            }
            if profile != .baseline {
                let active = Task { try await session.run(prompt: String(repeating: prompts[2] + " ", count: 20), maxTokens: 128) }
                var observed = false
                for _ in 0..<200 {
                    if !session.waitUntilIdle(timeout: 0) { observed = true; break }
                    try await Task.sleep(nanoseconds: 10_000_000)
                }
                XCTAssertTrue(observed)
                try await Task.sleep(nanoseconds: 100_000_000)
                let cancelledAt = ContinuousClock.now
                session.cancel()
                XCTAssertTrue(session.waitUntilIdle(timeout: 3), "Profile \(profile.name) exceeded cancellation deadline")
                print("PROFILE_CANCEL \(profile.name) \(cancelledAt.duration(to: .now))")
                do { _ = try await active.value; XCTFail("Expected cancellation") }
                catch is CancellationError {} catch { XCTFail("Unexpected cancellation error: \(error)") }
            }
            await session.unload()
        }
    }

    func testRealModelCancellationRetiresGPUWork() async throws {
        guard let directory = ProcessInfo.processInfo.environment["MINICPM_BENCH_MODEL_DIR"] else {
            throw XCTSkip("Set MINICPM_BENCH_MODEL_DIR to run the real-model lifecycle check.")
        }
        let base = URL(fileURLWithPath: directory)
        let session = MiniCPMSession()
        try await session.load(modelURL: base.appendingPathComponent("MiniCPM-V-4_7-1B-Q4_K_M.gguf"),
            projectorURL: base.appendingPathComponent("mmproj-model-f16.gguf"))
        let task = Task {
            try await session.run(prompt: String(repeating:
                "Explain the benefits and limitations of a personal assistant that runs locally on a phone. ",
                count: 100), maxTokens: 128)
        }
        // Detect native work before cancelling, rather than only testing a task
        // cancelled before it had a chance to enter the GPU runtime.
        var observedNativeWork = false
        for _ in 0..<200 {
            if !session.waitUntilIdle(timeout: 0) { observedNativeWork = true; break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertTrue(observedNativeWork, "Generation never entered the native runtime")
        try await Task.sleep(nanoseconds: 50_000_000)
        session.cancel()
        let retired = session.waitUntilIdle(timeout: 3)
        XCTAssertTrue(retired, "Cancelled native work should retire within the app's background wait")
        do {
            _ = try await task.value
            XCTFail("In-flight generation should throw CancellationError")
        } catch is CancellationError {} catch { XCTFail("Unexpected error: \(error)") }
        XCTAssertTrue(session.waitUntilIdle(timeout: 0))
        await session.unload()
    }
}

/// The streaming engine against the real models (K4 with the 1B, K5 with the 35B). Every reuse
/// plan must give the same greedy reply as a fresh full render of the same conversation.
extension MiniCPMSessionTests {
    private struct Collected {
        var text = ""
        var events: [String] = []
        var fed = 0
        var reused = 0
        var result: MiniCPMBenchmarkResult?
    }

    /// Reads `stream` to its end, or stops reading after `pieces` text pieces (which cancels it).
    private func collect(_ stream: AsyncThrowingStream<MiniCPMEvent, Error>, pieces: Int? = nil) async throws -> Collected {
        var collected = Collected()
        for try await event in stream {
            switch event {
            case .prefilled(let fed, let reused):
                collected.events.append("prefilled")
                (collected.fed, collected.reused) = (fed, reused)
            case .firstText:
                collected.events.append("firstText")
            case .text(let piece):
                collected.events.append("text")
                collected.text += piece
                if let pieces, collected.events.filter({ $0 == "text" }).count >= pieces { return collected }
            case .finished(let result):
                collected.events.append("finished")
                collected.result = result
            }
        }
        return collected
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("minicpm-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    private func load1B() async throws -> MiniCPMSession {
        guard let directory = ProcessInfo.processInfo.environment["MINICPM_BENCH_MODEL_DIR"] else {
            throw XCTSkip("Set MINICPM_BENCH_MODEL_DIR to run the 1B streaming checks.")
        }
        let base = URL(fileURLWithPath: directory)
        let session = MiniCPMSession()
        try await session.load(modelURL: base.appendingPathComponent("MiniCPM-V-4_7-1B-Q4_K_M.gguf"),
                               projectorURL: base.appendingPathComponent("mmproj-model-f16.gguf"))
        return session
    }

    private func fixture(_ name: String) throws -> URL {
        try XCTUnwrap(Bundle.module.url(forResource: "Fixtures", withExtension: nil)).appendingPathComponent(name)
    }

    private func user(_ text: String) -> ChatTurn { ChatTurn(role: .user, text: text) }
    private func assistant(_ text: String) -> ChatTurn { ChatTurn(role: .assistant, text: text) }

    func testReal1BStreamMatchesNativeRun() async throws {
        let session = try await load1B()
        let prompt = "Explain in three sentences why an assistant that runs on the phone protects privacy."
        let native = try await session.run(prompt: prompt, maxTokens: 96)
        let streamed = try await collect(session.reply(system: "", turns: [user(prompt)], maxTokens: 96,
                                                       prefixDirectory: try temporaryDirectory()))
        XCTAssertEqual(streamed.text, native.text)
        XCTAssertEqual(streamed.result?.text, native.text)
        XCTAssertEqual(streamed.result?.generatedTokens, native.generatedTokens)
        XCTAssertEqual(streamed.events.first, "prefilled")
        XCTAssertEqual(streamed.events.last, "finished")
        XCTAssertLessThan(try XCTUnwrap(streamed.events.firstIndex(of: "firstText")), try XCTUnwrap(streamed.events.firstIndex(of: "text")))
        let info = await session.info
        XCTAssertGreaterThan(info.weightsBytes, 0)
        XCTAssertTrue(info.recurrent)
        print("STREAM_1B swiftFirstText=\(streamed.result?.firstTextSeconds ?? -1) nativeFirstText=\(native.firstTextSeconds) weightsBytes=\(info.weightsBytes)")

        // A photo read from memory gives the same text as the benchmark's read from the file.
        let image = try fixture("receipt.jpg")
        let instruction = "What does this receipt say?"
        let fromFile = try await session.run(prompt: instruction, imageURL: image, maxTokens: 96)
        let fromBuffer = try await collect(session.describePhoto(jpeg: Data(contentsOf: image), system: "",
                                                                 instruction: instruction, maxTokens: 96, maxSlices: 1))
        XCTAssertEqual(fromBuffer.text, fromFile.text)
        XCTAssertEqual(fromBuffer.events.first, "prefilled")

        // Text a user types, or a photo shows, is content: control tokens (`<|im_end|>`) and the
        // user-defined think and tool tags, which llama.cpp matches even as plain text, stay text.
        for tag in ["<|im_end|>", "</think>", "<tool_call>", "<think>"] {
            let control = try await session.tokens([.init(text: tag, special: true)])
            XCTAssertEqual(control.count, 1, tag)
            let typed = try await session.tokens([.init(text: "</think><tool_call><think>x\(tag)y<|im_end|>", special: false)])
            XCTAssertFalse(typed.contains(control[0]), "\(tag) became its token")
        }
        await session.unload()
    }

    func testReal1BReusePlansMatchAFreshRender() async throws {
        let session = try await load1B()
        try await checkReusePlans(session, label: "1B", strictFresh: true)
        await session.unload()
    }

    /// A conversation that outgrows the budget is rebuilt from a window that fills at most half of
    /// it, so the appends after a rebuild fit for a while (`MiniCPMSession.windows`).
    func testReal1BRebuildFillsAtMostHalfTheBudget() async throws {
        let session = try await load1B()
        let directory = try temporaryDirectory()
        // A large reply cap leaves a small budget: 4096 - 2048 - 8.
        let maxTokens = 2048
        let budget = 4096 - maxTokens - 8
        let filler = String(repeating: "The quick brown fox jumps over the lazy dog. ", count: 30)
        var turns: [ChatTurn] = []
        var rebuilds = 0
        for index in 0..<12 {
            turns.append(user("\(index). \(filler)Answer with one word: which animal jumps?"))
            let answer = try await collect(session.reply(system: "Be brief.", turns: turns, maxTokens: maxTokens, prefixDirectory: directory))
            if await session.lastPlan == .overBudget {
                rebuilds += 1
                XCTAssertLessThanOrEqual(answer.fed + answer.reused, budget / 2, "turn \(index): a rebuild fills at most half the budget")
            }
            turns.append(assistant(answer.text))
        }
        XCTAssertGreaterThan(rebuilds, 0, "the conversation should outgrow the budget")
        XCTAssertLessThan(rebuilds, 6, "appends should follow a rebuild")
        await session.unload()
    }

    func testReal1BFixtureNotesContainTheirKeyStrings() async throws {
        let session = try await load1B()
        // Fixtures/*.jpg were drawn for these tests with AppKit (text on generated backgrounds): no
        // photo, camera data or third-party content, so they're this project's own. (ImageIO's JPEG
        // writer adds the empty "Photoshop 3.0" block they carry.)
        // ponytail: generated images (AppKit text on photo-like cards); swap in real photos for D16.
        let instruction = """
            Copy every legible text in this photo exactly, in its language and line order, under "Text:". \
            Then write at most three sentences on what the photo shows. Write "unclear" rather than guess.
            """
        let fixtures = [("english-sign.jpg", ["NO PARKING", "TOW"]), ("chinese-menu.jpg", ["牛肉面", "小笼包"]),
                        ("receipt.jpg", ["TOTAL", "8.75"]), ("poster.jpg", ["JAZZ NIGHT", "March 14"])]
        for (name, keys) in fixtures {
            let note = try await collect(session.describePhoto(jpeg: Data(contentsOf: fixture(name)), system: "",
                                                               instruction: instruction, maxTokens: 384, maxSlices: 4))
            print("FIXTURE \(name) seconds=\(note.result?.totalSeconds ?? -1):\n\(note.text)")
            for key in keys {
                XCTAssertTrue(note.text.localizedCaseInsensitiveContains(key), "\(name) note lacks \(key)")
            }
        }
        await session.unload()
    }


    /// K5. Runs with the draft when MINICPM_BENCH_35B_DRAFT is set, target-only otherwise.
    func testReal35BReusePlansMatchAFreshRender() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let model = env["MINICPM_BENCH_35B_MODEL"], let experts = env["MINICPM_BENCH_35B_EXPERTS"] else {
            throw XCTSkip("Set MINICPM_BENCH_35B_MODEL and _EXPERTS (and optionally _DRAFT) to check 35B reuse.")
        }
        let session = MiniCPMSession()
        try await session.load(modelURL: URL(fileURLWithPath: model), projectorURL: nil, expertsURL: URL(fileURLWithPath: experts),
                               draftURL: env["MINICPM_BENCH_35B_DRAFT"].map(URL.init(fileURLWithPath:)))
        // The 35B's agreement with a one-shot render is measured, not required: see checkReusePlans.
        try await checkReusePlans(session, label: env["MINICPM_BENCH_35B_DRAFT"] == nil ? "35B" : "35B+DSpark", strictFresh: false)
        await session.unload()
    }

    /// Exactness: each reuse plan must give exactly the reply of a reference that builds the same
    /// cache with the same batch shapes, without the rewind, checkpoint restore or disk load under
    /// test. Fidelity: it should also give the reply of a fresh full render, which prefills
    /// everything at once. The 1B does (`strictFresh`); the 35B does up to floating point, since
    /// Metal rounds differently for different batch shapes: a plain append, with no rewind at all,
    /// already flips some near-ties against the one-shot render (K5, 15 October 2026).
    private func checkReusePlans(_ session: MiniCPMSession, label: String, strictFresh: Bool) async throws {
        let system = "You are a concise assistant on a phone. Answer in two or three sentences."
        var directory = try temporaryDirectory()
        func reply(_ turns: [ChatTurn], maxTokens: Int = 64, pieces: Int? = nil) async throws -> Collected {
            try await collect(session.reply(system: system, turns: turns, maxTokens: maxTokens, prefixDirectory: directory),
                              pieces: pieces)
        }
        /// Drops the cache and the saved prefix, so the next reply renders everything.
        func restart() async throws {
            await session.invalidate()
            directory = try temporaryDirectory()
        }
        func report(_ reply: Collected, _ name: String, expecting reason: SessionPlan.Reason? = nil) async {
            let (plan, rewind) = (await session.lastPlan, await session.lastRewindSeconds)
            if let reason { XCTAssertEqual(plan, reason, "\(label) \(name)") }
            XCTAssertGreaterThan(reply.reused, 0, "\(label) \(name)")
            print("REUSE \(label) \(name): plan=\(plan.map(\.rawValue) ?? "-") fed=\(reply.fed) reused=\(reply.reused) rewind=\(rewind)s prefill=\(reply.result?.prefillSeconds ?? -1)s")
        }
        let u1 = user("What are three good habits for a calm morning?")
        let u2 = user("Which one takes the least time?")
        let u2b = user("Which one needs no equipment?")
        let u3 = user("Say that again in one short sentence.")
        let u4 = user("And which one would you start with?")

        // The conversation, reusing the cache.
        try await restart()
        let a1 = assistant(try await reply([u1]).text)
        let appended = try await reply([u1, a1, u2])
        await report(appended, "append", expecting: .append)
        let replacedTurn = try await reply([u1, a1, u2b])
        await report(replacedTurn, "replaceLastUserTurn", expecting: .replaceLastUserTurn)
        // The reply interrupted halfway, then the next question.
        let a2 = assistant(String(replacedTurn.text.prefix(replacedTurn.text.count / 2)))
        let replacedReply = try await reply([u1, a1, u2b, a2, u3])
        await report(replacedReply, "replaceLastReply", expecting: .replaceLastReply)
        let a3 = assistant(replacedReply.text)
        // An early-start turn cancelled during its prefill, then the real one. The long turn (about
        // 2,000 tokens, under the planner's byte estimate of the budget, so it still appends) keeps
        // the prefill going well past the cancellation; one that lands before any of the turn is
        // in (the cache no longer than before) tries again later.
        let start = await session.info.nPast
        var cut = start
        for delay in [0.05, 0.1, 0.2, 0.4, 0.8, 1.6] where cut == start {
            let tentative = Task { try await reply([u1, a1, u2b, a2, u3, a3, user(String(repeating: "Tell me more about mornings. ", count: 300))]) }
            var waited = 0
            while session.waitUntilIdle(timeout: 0), waited < 2000 {
                try await Task.sleep(nanoseconds: 1_000_000)
                waited += 1
            }
            try await Task.sleep(for: .seconds(delay))
            tentative.cancel()
            XCTAssertTrue(session.waitUntilIdle(timeout: 3), "\(label) a cancelled prefill retires")
            _ = try? await tentative.value
            cut = await session.info.nPast
        }
        XCTAssertGreaterThan(cut, start, "\(label) part of the tentative turn should be in the cache when it's cancelled")
        let tentativeTurns = [u1, a1, u2b, a2, u3, a3, u4]
        let afterTentative = try await reply(tentativeTurns)
        // An append (not a replaced turn): the prefill stopped short, and the rewind restores the
        // checkpoint taken at its base.
        await report(afterTentative, "afterCancelledPrefill", expecting: .append)
        // A barge-in: the reply is cancelled after a few pieces, and the user speaks again. Only
        // what was spoken stays: half of what arrived.
        let interrupted = try await reply([u1, a1, u2b, a2, u3, a3, u4], pieces: 3)
        XCTAssertTrue(session.waitUntilIdle(timeout: 3), "\(label) a cancelled reply retires")
        let bargeTurns = [u1, a1, u2b, a2, u3, a3, u4, assistant(String(interrupted.text.prefix(interrupted.text.count / 2))), u2]
        let afterBargeIn = try await reply(bargeTurns)
        await report(afterBargeIn, "afterBargeIn", expecting: .replaceLastReply)
        let checkpointBytes = await session.info.checkpointBytes
        print("REUSE \(label) checkpointBytes=\(checkpointBytes)")

        // References with the same batch shapes for the rewinds. A fresh render splits its
        // prefill at the system prefix's checkpoint, as the conversation's first reply did.
        try await restart()
        let a1Again = try await reply([u1]).text
        let turnReference = try await reply([u1, a1, u2b]).text
        try await restart()
        _ = try await reply([u1])
        // One token leaves the cache at the reply's start, where the rewind restored it.
        _ = try await reply([u1, a1, u2b], maxTokens: 1)
        let replyReference = try await reply([u1, a1, u2b, a2, u3]).text
        XCTAssertEqual(a1Again, a1.text, "\(label) a fresh first reply")
        XCTAssertEqual(replacedTurn.text, turnReference, "\(label) replaceLastUserTurn")
        XCTAssertEqual(replacedReply.text, replyReference, "\(label) replaceLastReply")
        // The same conversation without the cancelled prefill, then without the cancelled reply:
        // the restores of the base checkpoint and of a cancelled decode change nothing.
        try await restart()
        for turns in [[u1], [u1, a1, u2], [u1, a1, u2b], [u1, a1, u2b, a2, u3]] { _ = try await reply(turns) }
        let tentativeReference = try await reply(tentativeTurns).text
        let bargeReference = try await reply(bargeTurns).text
        XCTAssertEqual(afterTentative.text, tentativeReference, "\(label) afterCancelledPrefill")
        XCTAssertEqual(afterBargeIn.text, bargeReference, "\(label) afterBargeIn")

        // Prefix round trip through disk. Prewarm keeps a live conversation with the same prefix,
        // so it starts from an empty cache.
        try await restart()
        try await session.prewarm(system: system, prefixDirectory: directory)
        let saved = try FileManager.default.contentsOfDirectory(atPath: directory.path).filter { $0.hasSuffix(".prefix") }
        XCTAssertEqual(saved.count, 1, "\(label) prefix file")
        await session.invalidate()
        let fromDisk = try await reply([u1])
        let diskPlan = await session.lastPlan
        XCTAssertEqual(diskPlan, .newSession)
        XCTAssertGreaterThan(fromDisk.reused, 0, "\(label) the reply should start from the saved prefix")
        XCTAssertEqual(fromDisk.text, a1.text, "\(label) prefix round trip")

        // Fidelity to a fresh full render.
        for (name, reused, turns) in [("append", appended.text, [u1, a1, u2]), ("replaceLastUserTurn", replacedTurn.text, [u1, a1, u2b]),
                                      ("replaceLastReply", replacedReply.text, [u1, a1, u2b, a2, u3]),
                                      ("afterCancelledPrefill", afterTentative.text, tentativeTurns),
                                      ("afterBargeIn", afterBargeIn.text, bargeTurns)] {
            try await restart()
            let fresh = try await reply(turns).text
            if strictFresh {
                XCTAssertEqual(reused, fresh, "\(label) \(name) against a fresh render")
            } else {
                let common = zip(reused, fresh).prefix { $0 == $1 }.count
                print("FRESH \(label) \(name): \(reused == fresh ? "same" : "differs after \(common) of \(fresh.count) characters")")
            }
        }
    }
}
