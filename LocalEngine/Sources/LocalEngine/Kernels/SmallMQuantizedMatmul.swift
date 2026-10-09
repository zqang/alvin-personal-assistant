import Foundation
import MLX

/// A custom Metal kernel for quantized matrix products with few rows (plan WP41, §3 "8 rows for
/// ~1.5× one row"): `y[M, N] = x[M, K] · dequant(W)ᵀ` for 2 ≤ M ≤ 9, affine 4-bit weights with
/// group size 64.
///
/// Stock MLX runs such products as M separate matrix-vector passes (`qmv`), so every row of x
/// reads all the weights again; a verify pass over 8 rows costs about 3× a single row on stock
/// kernels. Here each simdgroup owns 4 output rows and reads their packed weights **once**:
/// - lanes stride K over the packed `uint32` words (2 words, 16 values, per lane per 512-value
///   block; the 16 values share one quantization group),
/// - each 4-bit nibble is dequantized with its group's scale and bias,
/// - the dequantized values are FMA'd against all M rows of x into `float acc[M][4]`,
/// - `simd_sum` reduces the lanes at the end.
///
/// The template arguments are M, K and the input dtype, so every distinct source has its own
/// kernel name (MLX rebuilds a kernel whose source changes under the same name; see
/// `TinyModels.hybridTextConfigJSON`). Every input has at least 8 elements, so none of them
/// moves to `constant` memory.
///
/// Off by default: `FastKernelsExtension` swaps `FastQuantizedLinear` in only when the app asks
/// for it, `KernelSelfTest` passes and the cost probe measures a real gain.
public enum SmallMQuantizedMatmul {
    /// Bits per weight the kernel reads.
    public static let bits = 4
    /// Weights per quantization group.
    public static let groupSize = 64
    /// The row counts of x the kernel handles.
    public static let rows: ClosedRange<Int> = 2 ... 9
    /// K must be a multiple of this: 32 lanes × 2 packed words × 8 values.
    public static let blockSize = 512
    /// Output rows one simdgroup computes; N must be a multiple of this.
    public static let rowsPerSimdgroup = 4
    /// Simdgroups per threadgroup.
    static let simdgroupsPerThreadgroup = 2
    /// The dtypes of x (and of the scales and biases, which must match it).
    public static let supportedTypes: [DType] = [.float32, .float16, .bfloat16]

    /// Whether this platform has custom Metal kernels at all (MLX traps on any other platform).
    public static var isAvailable: Bool {
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        return true
        #else
        return false
        #endif
    }

    /// Whether the kernel can run on MLX's current default device: custom kernels run only on the
    /// GPU.
    public static var canRunOnDefaultDevice: Bool {
        isAvailable && Device.defaultDevice().deviceType == .gpu
    }

    /// Whether a product of `rows` rows of x (`dtype`) with an `[outputDimensions,
    /// inputDimensions]` weight is a shape the kernel handles. Says nothing about the weight's
    /// format or the device; see `FastQuantizedLinear.isEligible(_:)`.
    public static func supports(rows: Int, inputDimensions: Int, outputDimensions: Int, dtype: DType) -> Bool {
        Self.rows.contains(rows)
            && inputDimensions > 0 && inputDimensions % blockSize == 0
            && outputDimensions > 0 && outputDimensions % rowsPerSimdgroup == 0
            && supportedTypes.contains(dtype)
    }

    /// `x · dequant(weight)ᵀ`, lazily.
    ///
    /// - `x`: `[M, K]` with M in `rows`;
    /// - `weight`: `[N, K / 8]` `uint32`, 4-bit affine, group size 64, N a multiple of 4;
    /// - `scales`, `biases`: `[N, K / 64]`, the same dtype as x.
    ///
    /// Returns `[M, N]` in x's dtype. Only valid where `canRunOnDefaultDevice` is true.
    public static func multiply(_ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray) -> MLXArray {
        precondition(isAvailable, "Custom Metal kernels don't exist on this platform.")
        precondition(x.ndim == 2 && weight.ndim == 2, "x must be [M, K] and the weight [N, K / 8].")
        let m = x.dim(0)
        let k = x.dim(1)
        let n = weight.dim(0)
        precondition(
            supports(rows: m, inputDimensions: k, outputDimensions: n, dtype: x.dtype),
            "The small-M kernel doesn't handle \(m) × \(k) → \(n) in \(x.dtype).")
        precondition(weight.dtype == .uint32 && weight.dim(1) * 32 / bits == k, "The weight isn't 4-bit packed for K = \(k).")
        precondition(scales.shape == [n, k / groupSize] && biases.shape == [n, k / groupSize], "Scales and biases must be [N, K / 64].")
        precondition(scales.dtype == x.dtype && biases.dtype == x.dtype, "Scales and biases must be \(x.dtype) like x.")
        let outputs = kernel(
            [x, weight, scales, biases],
            template: [("M", m), ("K", k), ("T", x.dtype)],
            grid: (32, n / rowsPerSimdgroup, 1),
            threadGroup: (32, simdgroupsPerThreadgroup, 1),
            outputShapes: [[m, n]],
            outputDTypes: [x.dtype])
        return outputs[0]
    }

    /// The kernel object (compiled per template instance on first use). Created only where custom
    /// kernels exist.
    private static let kernel = MLXFast.metalKernel(
        name: "alvin_small_m_qmm_affine_q4_g64",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: source)

    /// The kernel body. MLX generates the signature: `x`, `w`, `scales`, `biases`, the `w_shape`
    /// buffer (because the body names it), the output `y`, and the thread-position attributes the
    /// body names. The grid is `(32, N / 4, 1)` threads in threadgroups of `(32, 2, 1)`: each
    /// row of 32 threads is one simdgroup, and its y position is its 4-row output tile.
    static let source = """
        constexpr int ROWS = 4;
        constexpr int WORDS = 2;
        constexpr int VALUES = WORDS * 8;
        constexpr int BLOCK = 32 * VALUES;
        constexpr int KW = K / 8;
        constexpr int KG = K / 64;

        const int n_out = w_shape[0];
        const int lane = int(thread_index_in_simdgroup);
        const int row0 = int(thread_position_in_grid.y) * ROWS;

        float acc[M][ROWS];
        for (int m = 0; m < M; ++m) {
          for (int r = 0; r < ROWS; ++r) {
            acc[m][r] = 0.0f;
          }
        }

        const device uint32_t* w_lane = w + row0 * KW + lane * WORDS;
        const device T* s_rows = scales + row0 * KG;
        const device T* b_rows = biases + row0 * KG;
        const device T* x_lane = x + lane * VALUES;

        for (int kb = 0; kb < K; kb += BLOCK) {
          const int g_idx = (kb + lane * VALUES) / 64;
          float s_val[ROWS];
          float b_val[ROWS];
          for (int r = 0; r < ROWS; ++r) {
            s_val[r] = static_cast<float>(s_rows[r * KG + g_idx]);
            b_val[r] = static_cast<float>(b_rows[r * KG + g_idx]);
          }
          for (int j = 0; j < WORDS; ++j) {
            float w_val[ROWS][8];
            for (int r = 0; r < ROWS; ++r) {
              const uint32_t packed = w_lane[r * KW + kb / 8 + j];
              for (int i = 0; i < 8; ++i) {
                const float q = static_cast<float>((packed >> (4 * i)) & 0xfu);
                w_val[r][i] = fma(q, s_val[r], b_val[r]);
              }
            }
            for (int m = 0; m < M; ++m) {
              const device T* x_row = x_lane + m * K + kb + j * 8;
              for (int i = 0; i < 8; ++i) {
                const float xv = static_cast<float>(x_row[i]);
                for (int r = 0; r < ROWS; ++r) {
                  acc[m][r] = fma(xv, w_val[r][i], acc[m][r]);
                }
              }
            }
          }
        }

        for (int m = 0; m < M; ++m) {
          for (int r = 0; r < ROWS; ++r) {
            const float total = simd_sum(acc[m][r]);
            if (lane == 0) {
              y[m * n_out + row0 + r] = static_cast<T>(total);
            }
          }
        }
        """
}
