import XCTest
import Darwin
@testable import MiniCPMBenchmark

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
