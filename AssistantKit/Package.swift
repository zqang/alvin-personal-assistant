// swift-tools-version: 5.9
import PackageDescription

// Platform-independent core of the assistant: model clients, streaming parsers,
// and the conversation/turn-taking logic. It has no UIKit/AVFoundation dependency,
// so `swift test` runs it on macOS and Linux.
let package = Package(
    name: "AssistantKit",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "AssistantKit", targets: ["AssistantKit"]),
    ],
    targets: [
        .target(name: "AssistantKit"),
        .testTarget(name: "AssistantKitTests", dependencies: ["AssistantKit"]),
    ]
)
