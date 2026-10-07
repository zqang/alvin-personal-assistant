import Foundation
import LocalEngineTestSupport
import MLX
import MLXLMCommon
import XCTest

/// Proves the CI machine can run MLX on its Metal GPU before anything else is trusted. When it
/// can't, every MLX test skips through `MetalAvailability.require()` and the job stays a
/// compile-only check.
final class GPUCanaryTests: XCTestCase {
    func testMetalDeviceRunsMLX() throws {
        let status = MetalAvailability.status
        var lines = ["", "### GPU canary", "", "- Metal device: \(status.deviceName ?? "none")", "- MLX on the GPU: \(status.detail)"]
        if status.usable {
            let info = GPU.deviceInfo()
            lines.append("- Architecture: \(info.architecture)")
            lines.append("- Memory: \(info.memorySize / 1_048_576) MiB; recommended working set: \(info.maxRecommendedWorkingSetSize / 1_048_576) MiB")
        }
        EngineReport.append(lines.joined(separator: "\n"))
        try MetalAvailability.require()
        XCTAssertNotNil(status.deviceName)
        XCTAssertEqual(Device.defaultDevice().deviceType, .gpu)
    }

    func testSmallOperationsMatchHandComputedValues() throws {
        try MetalAvailability.require()
        let a = MLXArray([1, 2, 3, 4, 5, 6] as [Float], [2, 3])
        let b = MLXArray([1, 0, 0, 1, 1, 1] as [Float], [3, 2])
        // [[1, 2, 3], [4, 5, 6]] × [[1, 0], [0, 1], [1, 1]] = [[4, 5], [10, 11]]
        XCTAssertEqual(matmul(a, b).asArray(Float.self), [4, 5, 10, 11])
        XCTAssertEqual(softmax(MLXArray([0, 0] as [Float])).asArray(Float.self), [0.5, 0.5])
        XCTAssertEqual(a.sum(axis: 1).asArray(Float.self), [6, 15])
    }

    func testGatedDeltaKernelRuns() throws {
        try MetalAvailability.require()
        let keys = MLXRandom.split(key: MLXRandom.key(11), into: 5)
        let q = MLXRandom.normal([1, 3, 2, 32], key: keys[0]).asType(.bfloat16)
        let k = MLXRandom.normal([1, 3, 2, 32], key: keys[1]).asType(.bfloat16)
        let v = MLXRandom.normal([1, 3, 4, 32], key: keys[2]).asType(.bfloat16)
        let a = MLXRandom.normal([1, 3, 4], key: keys[3]).asType(.bfloat16)
        let b = MLXRandom.normal([1, 3, 4], key: keys[4]).asType(.bfloat16)
        let (y, state) = gatedDeltaUpdate(
            q: q, k: k, v: v, a: a, b: b,
            aLog: MLXArray.zeros([4]), dtBias: MLXArray.ones([4])
        )
        eval(y, state)
        XCTAssertEqual(y.shape, [1, 3, 4, 32])
        XCTAssertEqual(state.shape, [1, 4, 32, 32])
        XCTAssertEqual(state.dtype, .float32)
        XCTAssertTrue(MLX.abs(y.asType(.float32)).max().item(Float.self).isFinite)
        XCTAssertTrue(MLX.abs(state).max().item(Float.self).isFinite)
        XCTAssertGreaterThan(MLX.abs(state).max().item(Float.self), 0)
    }
}
