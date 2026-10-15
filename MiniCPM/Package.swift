// swift-tools-version: 5.9
import PackageDescription

// MiniCPM-V 4.7 on device through the pinned llama.cpp-omni runtime: the 1B photo reader, the
// 35B-A3B accurate mode, their model files, and the benchmark CLI. Apple platforms only.
let package = Package(
    name: "MiniCPM",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "MiniCPM", targets: ["MiniCPM"]),
        .executable(name: "minicpm-bench", targets: ["MiniCPMBenchCLI"]),
    ],
    dependencies: [.package(path: "../AssistantKit")],
    targets: [
        // Built by scripts/prepare_minicpm.sh. Replaced by the release asset's url and checksum
        // once it is published, so a clean clone needs no CMake.
        .binaryTarget(name: "llama", path: "Frameworks/llama.xcframework"),
        .target(name: "CMiniCPM", dependencies: ["llama"], publicHeadersPath: "include"),
        .target(
            name: "MiniCPM",
            dependencies: ["CMiniCPM", .product(name: "AssistantKit", package: "AssistantKit")],
            resources: [.copy("MiniCPM35BManifest.json")]
        ),
        .executableTarget(
            name: "MiniCPMBenchCLI",
            dependencies: ["MiniCPM", .product(name: "AssistantKit", package: "AssistantKit")]
        ),
        .testTarget(
            name: "MiniCPMTests",
            dependencies: ["MiniCPM", .product(name: "AssistantKit", package: "AssistantKit")],
            resources: [.copy("Fixtures")]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
