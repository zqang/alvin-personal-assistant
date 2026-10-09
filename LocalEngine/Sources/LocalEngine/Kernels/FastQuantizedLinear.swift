import Foundation
import MLX
import MLXNN

/// A `QuantizedLinear` that runs products with 2–9 rows (a speculative verify pass) on the
/// small-M kernel (plan WP41 instruction 2) and everything else on the stock path.
///
/// It reuses the original layer's `weight`, `scales`, `biases` and `bias` arrays (no copies), so
/// it can replace that layer in place through `Module.update(modules:)` and be swapped back the
/// same way. The kernel is used only when:
/// - the layer is 4-bit affine with group size 64 and has biases (`isEligible`);
/// - x has 2–9 rows, K is a multiple of 512 and N a multiple of 4;
/// - x, the scales and the biases share one float dtype;
/// - MLX's default device is the GPU;
/// - its `kernelSwitch` is on (`FastKernelsExtension.kernelsActive`).
///
/// Otherwise it calls `super`, which is bitwise the stock layer.
public final class FastQuantizedLinear: QuantizedLinear {
    /// Read at every call: off makes the layer run the stock product. Shared by the layers one
    /// `FastKernelsExtension` puts in, so flipping it never touches the model's modules.
    let kernelSwitch: FastKernelSwitch

    /// A fast layer over `layer`'s arrays, with its own switch (on).
    public convenience init(_ layer: QuantizedLinear) {
        self.init(layer, kernelSwitch: FastKernelSwitch())
    }

    /// A fast layer over `layer`'s arrays that uses the kernel while `kernelSwitch` is on.
    init(_ layer: QuantizedLinear, kernelSwitch: FastKernelSwitch) {
        self.kernelSwitch = kernelSwitch
        super.init(
            weight: layer.weight, bias: layer.bias, scales: layer.scales, biases: layer.biases,
            groupSize: layer.groupSize, bits: layer.bits, mode: layer.mode)
        // Like the stock layer, which is frozen when it is made.
        freeze()
    }

    /// Whether `layer`'s format and shape can ever use the kernel (for some row count and
    /// device): 4-bit affine, group size 64, biases present, a 2-D packed `uint32` weight with
    /// K a multiple of 512 and N a multiple of 4, and scales and biases of one supported float
    /// dtype.
    public static func isEligible(_ layer: QuantizedLinear) -> Bool {
        guard layer.bits == SmallMQuantizedMatmul.bits, layer.groupSize == SmallMQuantizedMatmul.groupSize,
              layer.mode == .affine, let biases = layer.biases
        else { return false }
        let weight = layer.weight
        guard weight.ndim == 2, weight.dtype == .uint32 else { return false }
        let n = weight.dim(0)
        let k = weight.dim(1) * 32 / layer.bits
        let groups = k / layer.groupSize
        guard layer.scales.shape == [n, groups], biases.shape == [n, groups], biases.dtype == layer.scales.dtype else {
            return false
        }
        return SmallMQuantizedMatmul.supports(
            rows: SmallMQuantizedMatmul.rows.lowerBound, inputDimensions: k, outputDimensions: n, dtype: layer.scales.dtype)
    }

    /// Whether this call of `x` runs on the kernel.
    func usesKernel(for x: MLXArray) -> Bool {
        guard kernelSwitch.isOn, x.ndim >= 1, let biases else { return false }
        let k = x.dim(-1)
        guard k > 0, weight.ndim == 2, k == weight.dim(1) * 32 / bits else { return false }
        let rows = x.size / k
        guard SmallMQuantizedMatmul.rows.contains(rows),
              bits == SmallMQuantizedMatmul.bits, groupSize == SmallMQuantizedMatmul.groupSize, mode == .affine,
              x.dtype == scales.dtype, x.dtype == biases.dtype,
              SmallMQuantizedMatmul.supports(rows: rows, inputDimensions: k, outputDimensions: weight.dim(0), dtype: x.dtype)
        else { return false }
        return SmallMQuantizedMatmul.canRunOnDefaultDevice
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard usesKernel(for: x), let biases else {
            return super.callAsFunction(x)
        }
        let k = x.dim(-1)
        let n = weight.dim(0)
        let rows = x.size / k
        var y = SmallMQuantizedMatmul.multiply(x.reshaped(rows, k), weight: weight, scales: scales, biases: biases)
        y = y.reshaped(Array(x.shape.dropLast()) + [n])
        if let bias {
            y = y + bias
        }
        return y
    }
}

/// An on/off switch that fast layers read at every call; safe to flip from any thread at any
/// time. A forward being built while it flips may run some layers each way, and both are correct.
final class FastKernelSwitch: @unchecked Sendable {
    private let lock = NSLock()
    private var on: Bool

    init(isOn: Bool = true) {
        on = isOn
    }

    var isOn: Bool {
        get {
            lock.lock()
            defer { lock.unlock() }
            return on
        }
        set {
            lock.lock()
            on = newValue
            lock.unlock()
        }
    }
}
