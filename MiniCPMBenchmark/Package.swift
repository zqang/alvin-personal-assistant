// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MiniCPMBenchmark",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "MiniCPMBenchmark", targets: ["MiniCPMBenchmark"]),
        .executable(name: "minicpm-bench", targets: ["MiniCPMBenchCLI"]),
    ],
    targets: [
        // Produced by scripts/prepare_minicpm.sh from the pinned upstream source.
        .binaryTarget(name: "llama", path: "Frameworks/llama.xcframework"),
        .target(name: "CMiniCPM", dependencies: ["llama"], publicHeadersPath: "include"),
        .target(name: "MiniCPMBenchmark", dependencies: ["CMiniCPM"]),
        .executableTarget(name: "MiniCPMBenchCLI", dependencies: ["MiniCPMBenchmark"]),
        .testTarget(name: "MiniCPMBenchmarkTests", dependencies: ["MiniCPMBenchmark"]),
    ],
    cxxLanguageStandard: .cxx17
)
