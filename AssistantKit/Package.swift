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
        .library(name: "AlvinProfile", targets: ["AlvinProfile"]),
    ],
    targets: [
        .target(name: "AssistantKit"),
        // This app's own domain (AssistantProfile), which another app replaces with its own and
        // never links. ponytail: it sits in this package only because AssistantKitTests and
        // LocalEngine's test targets use it and the app has no test target. Once a second app
        // shares AssistantKit, give it its own package with those tests, give LocalEngine's tests
        // a small fixture profile, and drop the AlvinProfile product from LocalEngine/Package.swift.
        .target(name: "AlvinProfile", dependencies: ["AssistantKit"]),
        .testTarget(name: "AssistantKitTests", dependencies: ["AssistantKit", "AlvinProfile"]),
    ]
)
