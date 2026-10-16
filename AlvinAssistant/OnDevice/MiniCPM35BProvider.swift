import AssistantKit
import Foundation
import MiniCPM
import UIKit

/// "Most accurate on device" (plan decision 19): MiniCPM-V 4.7 35B answers, with no tools or live
/// information. When it fails or is stopped before showing any text (not downloaded, not enough
/// memory, a load or runtime error, the on-device model taking the GPU), `fallback`, the on-device
/// model, answers instead, with its tools, and `onFallback` runs.
///
/// Events: `.cue(.loadingModel)` and an activity while the 35B loads, the progress marks, the
/// text, then `.reply(.finished)`.
struct MiniCPM35BProvider: AssistantProvider {
    let fallback: LocalProvider
    /// How the activity names `fallback`, e.g. "Underdog Woof 4B".
    let fallbackName: String
    let inputIsVoice: Bool
    let onFallback: @MainActor @Sendable () -> Void

    /// The app's system prompt, plus what the 35B can't do here.
    static func system(base: String) -> String {
        base + "\n\nYou have no tools or live information here; say so briefly if asked to act or to look something up."
    }

    func streamEvents(system: String, turns: [ChatTurn]) -> AsyncThrowingStream<AssistantEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { @MainActor in
                do {
                    try await relay(system: system, turns: turns, into: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    @MainActor
    private func relay(system: String, turns: [ChatTurn], into continuation: AsyncThrowingStream<AssistantEvent, Error>.Continuation) async throws {
        let host = MiniCPMHost.shared
        if !host.isLoaded(.accurate) {
            continuation.yield(.cue(.loadingModel))
            continuation.yield(.reply(.activity("Loading MiniCPM-V 35B")))
        }
        continuation.yield(.progress(.responseStarted))
        // Spoken replies are capped shorter than typed ones (decision 12).
        let reply = host.replyAccurate(system: Self.system(base: system), turns: turns, maxTokens: inputIsVoice ? 256 : 768)
        var shown = false
        do {
            for try await event in reply {
                switch event {
                case .prefilled(let fed, let reused):
                    continuation.yield(.reply(.activity(nil)))
                    continuation.yield(.progress(.prefillDone(prefilledTokens: fed, reusedTokens: reused)))
                case .firstText:
                    continuation.yield(.progress(.firstToken))
                case .text(let text):
                    shown = true
                    continuation.yield(.reply(.text(text)))
                case .finished(let result):
                    continuation.yield(.reply(.finished(result.finishReason == "tokenLimit" ? .truncated : .completed)))
                }
            }
        } catch {
            // Stopped by the caller.
            try Task.checkCancellation()
            // Stopped by the host: the app left the screen, or another model took the GPU (an
            // on-device model load, too little memory), which the fallback can work around.
            let stopped = error is CancellationError
            if stopped, UIApplication.shared.applicationState != .active {
                throw AssistantError.stream(type: "background", message: "the app left the screen.")
            }
            guard !shown else {
                throw stopped ? AssistantError.stream(type: "stopped", message: "another on-device model needed the memory.") : error
            }
            continuation.yield(.reply(.activity(stopped ? "Using \(fallbackName)" : "Using \(fallbackName): \(error.localizedDescription)")))
            onFallback()
            for try await event in fallback.streamEvents(system: system, turns: turns) {
                continuation.yield(event)
            }
        }
    }
}
