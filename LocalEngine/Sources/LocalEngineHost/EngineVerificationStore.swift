import AssistantKit
import Foundation
import LocalEngine

/// What the on-device checks found, kept between launches as JSON in UserDefaults (plan §4.10,
/// WP40 instruction 5):
///
/// - **self-test results** (`EngineSelfTest.Result`), per model id, snapshot directory, app build
///   and engine format version, so a new build, a new model revision or a new cache layout tests
///   again before `.automatic` trusts the engine;
/// - **cost curves** measured by `CostProbe`, per model, device (`utsname` machine), OS major
///   version and engine format version, and whether the model ran the fast kernels;
/// - **fast-kernel gains**: how much cheaper an 8-row forward was with the fast kernels than
///   without (`1 − c(8) with ÷ c(8) without`), per model, device, OS and format version.
///
/// Each kind keeps its `capacity` most recently saved entries.
struct EngineVerificationStore {
    /// Identifies a self-test result.
    struct SelfTestKey: Equatable, Sendable {
        var modelID: String
        /// The model's snapshot directory name (its revision).
        var snapshot: String
        var appBuild: String = EngineVerificationStore.appBuild
        var formatVersion: Int = EngineInfo.formatVersion

        var rawValue: String {
            [modelID, snapshot, appBuild, String(formatVersion)].joined(separator: "|")
        }
    }

    /// Identifies a measurement that depends on the model and the hardware, not on the build.
    struct DeviceKey: Equatable, Sendable {
        var modelID: String
        /// Whether the model ran the fast kernels (a cost curve measured with them is another
        /// curve).
        var fastKernels = false
        var machine: String = EngineVerificationStore.machine
        var osMajor: Int = EngineVerificationStore.osMajor
        var formatVersion: Int = EngineInfo.formatVersion

        var rawValue: String {
            var parts = [modelID, machine, String(osMajor), String(formatVersion)]
            if fastKernels { parts.append("fast-kernels") }
            return parts.joined(separator: "|")
        }
    }

    /// Entries kept per kind.
    static let capacity = 16

    static let selfTestsKey = "assistant.engine.selfTests"
    static let costCurvesKey = "assistant.engine.costCurves"
    static let kernelGainsKey = "assistant.engine.kernelGains"

    let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    // MARK: Self-test

    func selfTest(for key: SelfTestKey) -> EngineSelfTest.Result? {
        value(EngineSelfTest.Result.self, in: Self.selfTestsKey, key: key.rawValue)
    }

    func setSelfTest(_ result: EngineSelfTest.Result, for key: SelfTestKey) {
        set(result, in: Self.selfTestsKey, key: key.rawValue)
    }

    // MARK: Cost curve

    func costCurve(for key: DeviceKey) -> CostCurve? {
        value(CostCurve.self, in: Self.costCurvesKey, key: key.rawValue)
    }

    func setCostCurve(_ curve: CostCurve, for key: DeviceKey) {
        set(curve, in: Self.costCurvesKey, key: key.rawValue)
    }

    // MARK: Fast kernels

    func kernelGain(for key: DeviceKey) -> Double? {
        value(Double.self, in: Self.kernelGainsKey, key: key.rawValue)
    }

    /// Stores `gain` (`1 − c(8) with ÷ c(8) without`; below 0 when the kernels were slower).
    func setKernelGain(_ gain: Double, for key: DeviceKey) {
        guard gain.isFinite else { return }
        set(gain, in: Self.kernelGainsKey, key: key.rawValue)
    }

    /// Forgets every stored result and measurement.
    func removeAll() {
        for group in [Self.selfTestsKey, Self.costCurvesKey, Self.kernelGainsKey] {
            defaults.removeObject(forKey: group)
        }
    }

    // MARK: Storage

    private struct Entry<Value: Codable>: Codable {
        var value: Value
        var saved: Date
    }

    private func value<Value: Codable>(_ type: Value.Type, in group: String, key: String) -> Value? {
        entries(Value.self, in: group)[key]?.value
    }

    private func set<Value: Codable>(_ value: Value, in group: String, key: String) {
        var all = entries(Value.self, in: group)
        all[key] = Entry(value: value, saved: Date())
        if all.count > Self.capacity {
            let oldest = all.sorted { $0.value.saved < $1.value.saved }.prefix(all.count - Self.capacity)
            for entry in oldest {
                all[entry.key] = nil
            }
        }
        guard let data = try? JSONEncoder().encode(all) else { return }
        defaults.set(data, forKey: group)
    }

    private func entries<Value: Codable>(_ type: Value.Type, in group: String) -> [String: Entry<Value>] {
        guard let data = defaults.data(forKey: group),
              let decoded = try? JSONDecoder().decode([String: Entry<Value>].self, from: data)
        else { return [:] }
        return decoded
    }

    // MARK: This app and device

    /// The app's version and build, plus the executable's modification time, so every install of
    /// a new build counts as a new build even when the version numbers stay the same.
    static let appBuild: String = {
        let info = Bundle.main.infoDictionary
        let version = info?["CFBundleShortVersionString"] as? String ?? "?"
        let build = info?["CFBundleVersion"] as? String ?? "?"
        var stamp = ""
        if let executable = Bundle.main.executableURL,
           let modified = (try? executable.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
        {
            stamp = " \(Int(modified.timeIntervalSince1970))"
        }
        return "\(version) (\(build))\(stamp)"
    }()

    /// The hardware identifier, e.g. "iPhone17,1".
    static let machine: String = {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { bytes in
            String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
        }
    }()

    static let osMajor = ProcessInfo.processInfo.operatingSystemVersion.majorVersion
}
