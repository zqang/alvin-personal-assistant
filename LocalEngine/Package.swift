// swift-tools-version: 5.9
import PackageDescription

// The on-device reply engine ("Alvin engine"). It is built and tested on the CI GPU by
// .github/workflows/local-engine.yml and is not linked into the app yet. The dependency URLs
// are byte-identical to AlvinAssistant.xcodeproj, and the exact pins equal the app's resolution.
let package = Package(
    name: "LocalEngine",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "LocalEngine", targets: ["LocalEngine"]),
        .library(name: "LocalEngineTestSupport", targets: ["LocalEngineTestSupport"]),
    ],
    dependencies: [
        .package(path: "../AssistantKit"),
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", exact: "3.31.4"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", exact: "1.3.4"),
        .package(url: "https://github.com/huggingface/swift-huggingface", exact: "0.11.0"),
    ],
    targets: [
        .target(name: "LocalEngine", dependencies: [
            .product(name: "AssistantKit", package: "AssistantKit"),
            .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXLLM", package: "mlx-swift-lm"), .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            .product(name: "Tokenizers", package: "swift-transformers"),
            .product(name: "HuggingFace", package: "swift-huggingface"),
        ]),
        .target(name: "LocalEngineTestSupport", dependencies: ["LocalEngine",
            .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXLLM", package: "mlx-swift-lm"), .product(name: "MLXLMCommon", package: "mlx-swift-lm")]),
        .testTarget(name: "LocalEngineTests", dependencies: ["LocalEngine", "LocalEngineTestSupport"]),
        .testTarget(name: "LocalEngineIntegrationTests", dependencies: ["LocalEngine", "LocalEngineTestSupport"]),
    ]
)
