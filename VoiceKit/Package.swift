// swift-tools-version: 5.9
import PackageDescription

// Hands-free voice mode as a package an app embeds: listening and turn taking, spoken replies with
// barge-in, cues, early start and the camera. The app brings its conversation, its `ReplyPipeline`
// and a `VoiceSessionHost` (VoiceHost.swift). Its strings are in English and Simplified Chinese.
let package = Package(
    name: "VoiceKit",
    defaultLocalization: "en",
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "VoiceKit", targets: ["VoiceKit"]),
    ],
    dependencies: [.package(path: "../AssistantKit")],
    targets: [
        .target(
            name: "VoiceKit",
            dependencies: [.product(name: "AssistantKit", package: "AssistantKit")],
            resources: [.process("Localizable.xcstrings")]
        ),
        .testTarget(
            name: "VoiceKitTests",
            dependencies: ["VoiceKit", .product(name: "AssistantKit", package: "AssistantKit")]
        ),
    ]
)
