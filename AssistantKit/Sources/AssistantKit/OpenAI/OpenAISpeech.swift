import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

public struct OpenAISpeechConfiguration: Equatable, Sendable {
    public var apiKey: String
    public var model: String
    public var voice: String
    public var baseURL: URL

    public init(
        apiKey: String,
        model: String = "gpt-4o-mini-tts",
        voice: String = "coral",
        baseURL: URL = URL(string: "https://api.openai.com/v1")!
    ) {
        self.apiKey = apiKey
        self.model = model
        self.voice = voice
        self.baseURL = baseURL
    }
}

/// Requests for OpenAI text-to-speech, streamed back as raw 24 kHz mono 16-bit PCM.
public enum OpenAISpeech {
    public static let voices = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"]
    public static let sampleRate: Double = 24_000
    static let styleInstructions = "Speak in a warm, natural, conversational tone, like a friendly assistant on a phone call."

    public static func request(for text: String, configuration: OpenAISpeechConfiguration) throws -> URLRequest {
        var request = URLRequest(url: configuration.baseURL.appendingPathComponent("audio/speech"))
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.setValue("Bearer \(configuration.apiKey)", forHTTPHeaderField: "authorization")
        var body: [String: JSONValue] = [
            "model": .string(configuration.model),
            "voice": .string(configuration.voice),
            "input": .string(text),
            "response_format": .string("pcm"),
        ]
        // Only the gpt-4o TTS models take style instructions.
        if configuration.model.hasPrefix("gpt-4o") {
            body["instructions"] = .string(styleInstructions)
        }
        request.httpBody = try JSONValue.object(body).serialized()
        return request
    }
}

/// Converts little-endian 16-bit PCM bytes, arriving in chunks of any size, to Float samples.
public struct PCM16Decoder: Sendable {
    private var leftover: UInt8?

    public init() {}

    public mutating func decode(_ data: Data) -> [Float] {
        var samples: [Float] = []
        samples.reserveCapacity(data.count / 2 + 1)
        var iterator = data.makeIterator()
        if let low = leftover {
            guard let high = iterator.next() else { return [] }
            samples.append(Self.sample(low: low, high: high))
            leftover = nil
        }
        while let low = iterator.next() {
            guard let high = iterator.next() else {
                leftover = low
                break
            }
            samples.append(Self.sample(low: low, high: high))
        }
        return samples
    }

    static func sample(low: UInt8, high: UInt8) -> Float {
        Float(Int16(bitPattern: UInt16(low) | (UInt16(high) << 8))) / 32768
    }
}
