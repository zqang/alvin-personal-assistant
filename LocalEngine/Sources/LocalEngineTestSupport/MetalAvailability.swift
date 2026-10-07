import Foundation
import MLX
#if canImport(Metal)
import Metal
#endif
#if canImport(XCTest)
import XCTest
#endif

/// Whether this machine can run MLX on a Metal GPU. Checked once per test process: a Metal
/// device must exist and a small MLX computation must give the right answer.
public enum MetalAvailability {
    public struct Status: Sendable {
        /// The Metal device's name, such as "Apple M1 (Virtual)"; nil without a device.
        public let deviceName: String?
        /// Whether MLX computed correctly on it.
        public let usable: Bool
        /// "ok", or why MLX can't be used.
        public let detail: String
    }

    public struct Unavailable: Error, CustomStringConvertible {
        public let description: String
    }

    public static let status: Status = probe()

    public static var isAvailable: Bool { status.usable }

    /// Skips the calling test (`XCTSkip`) when there is no usable Metal device.
    public static func require() throws {
        guard status.usable else {
            let reason = "No usable Metal GPU for MLX: \(status.detail)"
            #if canImport(XCTest)
            throw XCTSkip(reason)
            #else
            throw Unavailable(description: reason)
            #endif
        }
    }

    private static func probe() -> Status {
        #if canImport(Metal)
        guard let device = MTLCreateSystemDefaultDevice() else {
            return Status(deviceName: nil, usable: false, detail: "no Metal device")
        }
        let name = device.name
        do {
            // [[1, 2], [3, 4]] squared is [[7, 10], [15, 22]], which sums to 54.
            let sum: Float = try withError { () throws -> Float in
                let a = MLXArray([1, 2, 3, 4] as [Float], [2, 2])
                let total = matmul(a, a).sum()
                try checkedEval(total)
                return total.item(Float.self)
            }
            guard sum == 54 else {
                return Status(deviceName: name, usable: false, detail: "MLX computed \(sum) instead of 54")
            }
            return Status(deviceName: name, usable: true, detail: "ok")
        } catch {
            return Status(deviceName: name, usable: false, detail: "MLX failed: \(error)")
        }
        #else
        return Status(deviceName: nil, usable: false, detail: "Metal is not available on this platform")
        #endif
    }
}
