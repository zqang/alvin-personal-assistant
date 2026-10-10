import Foundation
import MLX
import MLXNN

/// Checks the small-M kernel against MLX's own `quantizedMM` on this device (plan WP41
/// instruction 3). `FastKernelsExtension` swaps the kernel in only after a pass; any MLX error or
/// mismatch keeps it off.
///
/// For each shape it draws random packed 4-bit weights, scales and biases, then compares
/// `FastQuantizedLinear` with the stock product for every row count (1 runs the stock path, 2–9
/// the kernel) and every dtype asked for. Random packed words are drawn directly (not quantized
/// from float weights), so the largest shape needs about 15 MB instead of about 100 MB.
///
/// Tolerance: `allClose` with a relative tolerance of 1e-3 for float32. For float16 and bfloat16
/// it adds two of the dtype's rounding units (2⁻¹⁰, 2⁻⁷): both products round their float32 sums
/// to the dtype, and MLX's own matrix kernel (6 rows and more) also rounds each dequantized weight
/// to it, so the stock result itself is only that close to the exact one. The absolute tolerance
/// is the same fraction of the reference's largest magnitude, for outputs near zero. A wrong
/// index, nibble or group gives errors of the order of the outputs themselves.
public enum KernelSelfTest {
    /// One weight shape: K inputs to N outputs, with or without a bias.
    public struct Shape: Sendable, Equatable, CustomStringConvertible {
        public var inputDimensions: Int
        public var outputDimensions: Int
        public var bias: Bool

        public init(_ inputDimensions: Int, _ outputDimensions: Int, bias: Bool = false) {
            self.inputDimensions = inputDimensions
            self.outputDimensions = outputDimensions
            self.bias = bias
        }

        public var description: String {
            "\(inputDimensions)→\(outputDimensions)\(bias ? " +bias" : "")"
        }
    }

    /// Woof 4B's MLP and attention widths, plus a tiny shape: the smallest K, an odd number of
    /// 4-row tiles (a partial threadgroup) and a bias.
    public static let defaultShapes: [Shape] = [
        Shape(2560, 9216),
        Shape(9216, 2560),
        Shape(2560, 8192),
        Shape(512, 12, bias: true),
    ]

    public struct Options: Sendable {
        public var shapes: [Shape]
        /// Rows of x; 1 (and anything above 9) runs on the stock path.
        public var rows: ClosedRange<Int>
        /// The dtypes of x, the scales and the biases.
        public var dtypes: [DType]
        public var seed: UInt64
        /// Tests: corrupts every product that ran on the kernel, which the check must catch.
        var debugBreakKernel = false

        public init(
            shapes: [Shape] = KernelSelfTest.defaultShapes, rows: ClosedRange<Int> = 1 ... 9,
            dtypes: [DType] = [.float32], seed: UInt64 = 41
        ) {
            self.shapes = shapes
            self.rows = rows
            self.dtypes = dtypes
            self.seed = seed
        }
    }

    public struct Result: Sendable, Equatable {
        public var passed: Bool
        /// Products compared.
        public var cases: Int
        /// Products of them that ran on the kernel.
        public var kernelCases: Int
        /// One line per mismatch or error.
        public var failures: [String]
        /// The largest |fast − stock| relative to the reference's largest magnitude.
        public var worstRelativeError: Double
        public var seconds: Double
        public var detail: String
    }

    /// Runs the check on MLX's default device, which must be the GPU. `isAllowed` is checked
    /// before every product; when it turns false the run stops and fails ("stopped").
    public static func run(options: Options = Options(), isAllowed: () -> Bool = { true }) -> Result {
        let started = ProcessInfo.processInfo.systemUptime
        var cases = 0
        var kernelCases = 0
        var failures: [String] = []
        var worst = 0.0

        func result(_ detail: String? = nil) -> Result {
            let seconds = ProcessInfo.processInfo.systemUptime - started
            let passed = detail == nil && failures.isEmpty && kernelCases > 0
            var text = detail ?? ""
            if text.isEmpty {
                if !failures.isEmpty {
                    text = "\(failures.count) of \(cases) products differ: \(failures.prefix(3).joined(separator: "; "))"
                } else if kernelCases == 0 {
                    text = "no product ran on the kernel"
                } else {
                    text = "\(cases) products match (\(kernelCases) on the kernel), worst error \(String(format: "%.2g", worst))"
                }
            }
            return Result(
                passed: passed, cases: cases, kernelCases: kernelCases, failures: failures, worstRelativeError: worst,
                seconds: seconds, detail: text)
        }

        guard SmallMQuantizedMatmul.isAvailable else {
            return result("custom Metal kernels don't exist on this platform")
        }
        guard SmallMQuantizedMatmul.canRunOnDefaultDevice else {
            return result("MLX's default device is not the GPU")
        }

        do {
            try withError {
                for (index, shape) in options.shapes.enumerated() {
                    let k = shape.inputDimensions
                    let n = shape.outputDimensions
                    precondition(
                        k > 0 && k % SmallMQuantizedMatmul.groupSize == 0 && n > 0,
                        "Self-test shape \(shape) must have K a multiple of \(SmallMQuantizedMatmul.groupSize).")
                    let keys = MLXRandom.split(key: MLXRandom.key(options.seed &+ UInt64(index)), into: 6)
                    let words = [n, k / 8]
                    let groups = [n, k / SmallMQuantizedMatmul.groupSize]
                    // Two 16-bit halves: MLX draws integers through float32, exact only below 2²⁴.
                    let high = MLXRandom.randInt(UInt32(0) ..< UInt32(1 << 16), words, key: keys[0])
                    let low = MLXRandom.randInt(UInt32(0) ..< UInt32(1 << 16), words, key: keys[1])
                    let weight = (high << UInt32(16)) | low
                    let scales32 = MLXRandom.uniform(Float(0.002) ..< Float(0.02), groups, key: keys[2])
                    let biases32 = scales32 * Float(-8) + MLXRandom.uniform(Float(-0.01) ..< Float(0.01), groups, key: keys[3])
                    let bias32: MLXArray? = shape.bias ? MLXRandom.normal([n], key: keys[4]) : nil
                    let x32 = MLXRandom.normal([1, max(options.rows.upperBound, 1), k], key: keys[5])
                    try checkedEval(weight, scales32, biases32, x32)
                    if let bias32 {
                        try checkedEval(bias32)
                    }

                    for dtype in options.dtypes {
                        let scales = scales32.asType(dtype)
                        let biases = biases32.asType(dtype)
                        let bias = bias32?.asType(dtype)
                        let stock = QuantizedLinear(
                            weight: weight, bias: bias, scales: scales, biases: biases,
                            groupSize: SmallMQuantizedMatmul.groupSize, bits: SmallMQuantizedMatmul.bits, mode: .affine)
                        let fast = FastQuantizedLinear(stock)
                        let tolerance = Self.tolerance(for: dtype)

                        for rows in options.rows {
                            guard isAllowed() else { throw Stopped() }
                            let x = x32[0..., ..<rows, 0...].asType(dtype)
                            let onKernel = fast.usesKernel(for: x)
                            var output = fast(x)
                            if onKernel && options.debugBreakKernel {
                                output = output + 1
                            }
                            var reference = quantizedMM(
                                x, weight, scales: scales, biases: biases, transpose: true,
                                groupSize: SmallMQuantizedMatmul.groupSize, bits: SmallMQuantizedMatmul.bits, mode: .affine)
                            if let bias {
                                reference = reference + bias
                            }
                            let candidate = output.asType(.float32)
                            let expected = reference.asType(.float32)
                            let scale = abs(expected).max()
                            let error = abs(candidate - expected).max()
                            try checkedEval(candidate, expected, scale, error)

                            let scaleValue = Double(scale.item(Float.self))
                            let errorValue = Double(error.item(Float.self))
                            let close = candidate.shape == expected.shape
                                && allClose(candidate, expected, rtol: tolerance, atol: tolerance * scaleValue).item(Bool.self)
                            cases += 1
                            if onKernel { kernelCases += 1 }
                            let relative = scaleValue > 0 ? errorValue / scaleValue : errorValue
                            if !(relative <= worst) {
                                worst = relative.isNaN ? .infinity : relative
                            }
                            if !close {
                                failures.append(
                                    "\(shape) \(dtype) M=\(rows)\(onKernel ? "" : " (stock)"): max |Δ| \(String(format: "%.3g", errorValue)) of \(String(format: "%.3g", scaleValue))")
                            }
                        }
                    }
                    // Next to a loaded model, keep only one shape's buffers at a time.
                    Memory.clearCache()
                }
            }
        } catch is Stopped {
            Memory.clearCache()
            return result("stopped: the GPU is no longer allowed")
        } catch {
            Memory.clearCache()
            return result("MLX error: \(error)")
        }
        Memory.clearCache()
        return result()
    }

    /// The relative tolerance for outputs of `dtype`: 1e-3 plus two rounding units of the dtype
    /// (one for the output rounding, one for MLX's matrix kernel, which dequantizes the weights in
    /// the dtype rather than in float32 for 6 rows or more).
    static func tolerance(for dtype: DType) -> Double {
        switch dtype {
        case .float16: return 1e-3 + 2.0 / 1024
        case .bfloat16: return 1e-3 + 2.0 / 128
        default: return 1e-3
        }
    }

    private struct Stopped: Error {}
}
