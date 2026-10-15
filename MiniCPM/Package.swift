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
        // Built by scripts/prepare_minicpm.sh at c5b08ab (see the release's PROVENANCE.txt), so a
        // clean clone needs no CMake. To try a local build, use
        // path: "Frameworks/llama.xcframework" instead (docs/minicpm-evaluation.md §9.6).
        .binaryTarget(
            name: "llama",
            url: "https://github.com/zqang/alvin-personal-assistant/releases/download/minicpm-runtime-96bbf61-p60c9974a-se2234816/llama.xcframework.zip",
            checksum: "22244ae63e34299ac7d459afda63ff953235cf3df748488809920754ceb2619f"
        ),
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
