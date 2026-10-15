import Foundation
import MiniCPMBenchmark

@main
struct MiniCPMBenchCLI {
    struct Report: Encodable {
        let model: String
        let runtimeRevision = "96bbf61710d44421ea54e63f5194f9b17c20d877"
        let image: String?
        let prompt: String
        let loadSeconds: Double
        let tokensPerSecond: Double
        let result: MiniCPMBenchmarkResult
    }

    static func main() async {
        let usage = "Usage: minicpm-bench MODEL.gguf MMPROJ.gguf|- [IMAGE.png] [--experts MANIFEST.json] [--draft DSPARK.gguf] [--prompt TEXT | --prompts-json FILE] [--max-tokens N] [--profile NAME] [--warmup N] [--repeat N]"
        var arguments = Array(CommandLine.arguments.dropFirst())
        if arguments.isEmpty || arguments.contains("--help") {
            print(usage)
            return
        }
        let session = MiniCPMSession()
        do {
            guard arguments.count >= 2 else { throw MiniCPMError.invalidInput(usage) }
            let model = URL(fileURLWithPath: arguments.removeFirst())
            let projectorPath = arguments.removeFirst()
            let projector = projectorPath == "-" ? nil : URL(fileURLWithPath: projectorPath)
            var experts: URL?
            var draft: URL?
            var image: URL?
            var prompt: String?
            var maxTokens = 128
            var configuration = MiniCPMRuntimeConfiguration.baseline
            var promptsFile: URL?
            var repetitions = 1
            var warmupCount = 0
            if let first = arguments.first, !first.hasPrefix("--") {
                image = URL(fileURLWithPath: arguments.removeFirst())
            }
            while !arguments.isEmpty {
                let flag = arguments.removeFirst()
                guard let value = arguments.first else { throw MiniCPMError.invalidInput(usage) }
                arguments.removeFirst()
                switch flag {
                case "--experts": experts = URL(fileURLWithPath: value)
                case "--draft": draft = URL(fileURLWithPath: value)
                case "--prompt": prompt = value
                case "--profile":
                    guard let profile = MiniCPMRuntimeConfiguration.profiles.first(where: { $0.name == value }) else {
                        throw MiniCPMError.invalidInput("Unknown runtime profile: \(value)")
                    }
                    configuration = profile
                case "--prompts-json": promptsFile = URL(fileURLWithPath: value)
                case "--repeat":
                    guard let parsed = Int(value), (1...100).contains(parsed) else { throw MiniCPMError.invalidInput("Invalid repeat count") }
                    repetitions = parsed
                case "--warmup":
                    guard let parsed = Int(value), (0...10).contains(parsed) else { throw MiniCPMError.invalidInput("Invalid warmup count") }
                    warmupCount = parsed
                case "--max-tokens":
                    guard let parsed = Int(value) else { throw MiniCPMError.invalidInput("Invalid token limit") }
                    maxTokens = parsed
                default: throw MiniCPMError.invalidInput("Unknown option: \(flag)\n\(usage)")
                }
            }
            let instruction = prompt ?? (image == nil
                ? "In one sentence, explain what an on-device personal assistant does."
                : "Describe what is visible in this image. Be concise and precise.")
            let start = ContinuousClock.now
            try await session.load(modelURL: model, projectorURL: projector, expertsURL: experts, draftURL: draft, configuration: configuration)
            let elapsed = start.duration(to: .now).components
            let loadSeconds = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
            let prompts = try promptsFile.map { try JSONDecoder().decode([String].self, from: Data(contentsOf: $0)) } ?? [instruction]
            guard !prompts.isEmpty else { throw MiniCPMError.invalidInput("Prompt list cannot be empty") }
            for _ in 0..<warmupCount { _ = try await session.run(prompt: "Reply with exactly OK.", maxTokens: 8) }
            var reports: [Report] = []
            for _ in 0..<repetitions {
                for instruction in prompts {
                    let result = try await session.run(prompt: instruction, imageURL: image, maxTokens: maxTokens)
                    reports.append(Report(model: model.lastPathComponent, image: image?.lastPathComponent,
                        prompt: instruction, loadSeconds: loadSeconds, tokensPerSecond: result.tokensPerSecond, result: result))
                }
            }
            await session.unload()
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try reports.count == 1 && promptsFile == nil ? encoder.encode(reports[0]) : encoder.encode(reports)
            print(String(decoding: data, as: UTF8.self))
        } catch {
            await session.unload()
            FileHandle.standardError.write(Data("MiniCPM benchmark failed: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }
}
