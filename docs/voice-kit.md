# VoiceKit: hands-free voice mode in another app

VoiceKit (`VoiceKit/`) is this app's voice mode as a Swift package. It listens, decides when the user has
finished, speaks the reply sentence by sentence, stops when the user talks over it, plays short spoken cues,
starts a reply early, and opens the camera. The app embedding it provides three things:

- the conversation it talks in;
- the replies, through a `ReplyPipeline`;
- its settings, plus a few optional hooks.

This guide shows how an app plugs in. Section 7 covers Roam, the travel app that embeds VoiceKit next. It
shows how Roam's tools, its hybrid AI and its two languages fit in, and lists the work items R1–R8 that make
up the Roam round.

## Contents

1. [What's in the package](#1-whats-in-the-package)
2. [Adding it to an app](#2-adding-it-to-an-app)
3. [The host](#3-the-host)
4. [Opening voice mode](#4-opening-voice-mode)
5. [The reply pipeline](#5-the-reply-pipeline)
6. [Languages](#6-languages)
7. [Roam](#7-roam)
8. [Tests and CI](#8-tests-and-ci)

## 1. What's in the package

| Public type | What it is |
|---|---|
| `VoiceSession(conversation:pipeline:host:)` | The voice conversation: microphone, recognition, turn taking, early start, spoken replies, barge-in, cues, camera, latency trace |
| `VoiceModeView(session:)` | The full-screen voice screen. It starts the session when it appears and stops it when it goes away. |
| `VoiceConversation`, `VoiceMessage`, `VoiceSessionHost` | What the app provides (`VoiceHost.swift`, section 3) |
| `ReplyPipeline`, `ReplyRequest`, `ReplyOutcome` | How replies are asked for (section 5), and how a finished or stopped reply is recorded |
| `PendingPhoto`, `PhotoThumbnail`, `CameraFeedView` | A photo attached to the next turn and read while the user talks, and the views that show and take one |
| `OrbView`, `OrbMood`, `VoiceCatalog`, `LatencyLog.shared` | Pieces the app's own screens reuse: the orb, the installed voices, and the per-turn latency log (stored under the key `assistant.latencyTraces`) |

VoiceKit depends only on `AssistantKit`. It builds for iOS 17 or later in Swift 5 language mode. It doesn't
include a reply model, storage or a settings screen: the app provides those.

## 2. Adding it to an app

**In this repository**, the Xcode project adds `VoiceKit` as a local package, next to `AssistantKit`, and the
app links its product.

**From another repository**, SwiftPM can't depend on a package in a subdirectory of a remote repository.
Instead, the other app adds this repository as a git submodule pinned to a tag and adds the packages it needs
as local packages. Section 7.1 has Roam's exact lines.

**Info.plist.** The app needs:

- `NSMicrophoneUsageDescription` and `NSSpeechRecognitionUsageDescription`;
- `NSCameraUsageDescription`, in any app that links VoiceKit. The camera code is in every binary, and App
  Store processing flags one that references the camera without the purpose string (ITMS-90683). The
  camera only appears when the host reads photos (`photoReader`, section 3);
- the `audio` background mode, if voice mode should keep talking when the screen locks.

**Orientation.** The voice screen and the camera are laid out for portrait. In landscape, an iPhone is too
short for the orb, the captions and the controls, and photos would come out rotated. An app that supports
landscape presents voice mode in portrait only (section 7.5 shows Roam's way).

## 3. The host

These types are the smallest complete host: plain message and conversation classes, a reply pipeline that
echoes the user, and a host that takes every default. They are copied from
`VoiceKit/Tests/VoiceKitTests/VoiceKitTests.swift`, where the tests compile and run them.

```swift
import AssistantKit
import Foundation
import VoiceKit

@MainActor
final class Message: VoiceMessage {
    let role: ChatRole
    var text: String
    var createdAt = Date()
    let isVoice: Bool
    var status: StoredMessage.Status
    var errorText: String?
    var toolRounds: [ToolRound] = []
    var photoData: Data?
    var photoNote: String?

    init(role: ChatRole, text: String, isVoice: Bool, status: StoredMessage.Status) {
        self.role = role
        self.text = text
        self.isVoice = isVoice
        self.status = status
    }

    var stored: StoredMessage {
        StoredMessage(role: role, text: text, createdAt: createdAt, isVoice: isVoice, status: status, toolRounds: toolRounds, photoNote: photoNote)
    }
}

@MainActor
final class Chat: VoiceConversation {
    var title = ""
    private(set) var messages: [Message] = []

    var voiceMessages: [any VoiceMessage] { messages }

    func appendMessage(role: ChatRole, text: String, isVoice: Bool, status: StoredMessage.Status) -> any VoiceMessage {
        let message = Message(role: role, text: text, isVoice: isVoice, status: status)
        messages.append(message)
        if title.isEmpty, role == .user { title = text }
        return message
    }

    func remove(_ message: any VoiceMessage) {
        messages.removeAll { $0 === message }
    }

    func userTurnRevised(from old: String, to new: String) {
        if title == old { title = new }
    }

    func save() {}
}

/// Answers every turn by repeating it.
@MainActor
final class Pipeline: ReplyPipeline {
    /// What `missingSetup()` reports.
    var problem: String?
    private(set) var prewarms = 0

    var supportsEarlyStart: Bool { false }

    func stream(_ request: ReplyRequest) -> AsyncThrowingStream<AssistantEvent, Error> {
        let heard = request.messages.last { $0.role == .user }?.text ?? ""
        return AsyncThrowingStream { continuation in
            continuation.yield(.reply(.text("You said: \(heard)")))
            continuation.yield(.reply(.finished(.completed)))
            continuation.finish()
        }
    }

    func missingSetup() -> String? { problem }

    func prewarm(inputIsVoice: Bool) { prewarms += 1 }

    func displayName(for decision: RouteDecision?) -> String { "Echo" }
}

@MainActor
final class Host: VoiceSessionHost {
    var settings = AssistantSettings()
}
```

What each part means:

- **`VoiceMessage`.** Voice mode writes the spoken user turns and the replies through these properties.
  `stored` is the message as a reply reads it. A SwiftData model conforms with no new members if it already
  has them, as this app's `ChatMessage` does (`AlvinAssistant/App/VoiceHosting.swift`).
- **`VoiceConversation`.**
  - `voiceMessages` lists every message, oldest first.
  - `appendMessage` adds a message at the end and returns it, and `remove` deletes one.
  - `userTurnRevised(from:to:)` reports that a second transcription, or the words an early reply heard,
    replaced a user turn's text. A title made from the old text should follow the new one.
  - `save()` runs after each reply and when the session stops.
- **`VoiceSessionHost`.** Only `settings` is required. Voice mode reads it when a session starts and again at
  each turn, and never changes it.

  Each hook has a default that goes without it:

  | Hook | What it does | Default |
  |---|---|---|
  | `openAIKey` | The key for OpenAI voices (`settings.voiceEngine`) | empty: the built-in voice speaks |
  | `voiceSessionWillStart(_:)` | Gets ready what the session's replies use, such as an on-device model. It runs once the microphone and speech recognition are allowed. | does nothing |
  | `voiceSessionDidStop()` | Frees it again, when the session stops or fails after starting. It runs exactly once for each `voiceSessionWillStart`, so a start that fails before then calls neither. | does nothing |
  | `refineClipSamples`, `refineTurn(_:locale:)` | A second transcription of each committed turn from its audio (this app uses Qwen3-ASR) | 0 and nil: no audio is kept and no second transcription runs |
  | `photoReader` | Reads a photo into the note that goes into the turn's context | nil: the camera is hidden |

**The settings voice mode reads** (`AssistantSettings`). The defaults suit a new app:

- **Speech:** `speechLocale` (recognition, the built-in voice and the cues), `voiceEngine`,
  `appleVoiceIdentifier`, `speechRate`, `openAIVoice`, `openAISpeechModel`.
- **Turn taking:** `endOfTurnDelay`, `voiceInterruptions`, `echoCancellation`, `bargeInDucking`,
  `turnChime`, `spokenCues`, `fillerDelay`.
- **Early start:** `earlyReplyStart`, plus `provider`, `routingMode` and `preferOnDevice`, which say whether
  replies are likely to run on device. A host that refines turns (`refineClipSamples` above 0) gets no early
  start for those replies, because the second transcription and the reply model would compete for the GPU.
- **"Think deeper":** shown only when `provider` is `.anthropic` and `deepMode` isn't `.off`.

`provider`, `routingMode` and `preferOnDevice` describe this app's routing. A host whose pipeline routes each
request itself leaves them at their defaults. It also sets `deepMode` to `.off` unless its pipeline honours
`ReplyRequest.deep`, so "Think deeper" stays hidden.

## 4. Opening voice mode

```swift
@State private var voice: VoiceSession?

Button("Talk", systemImage: "waveform") {
    voice = VoiceSession(conversation: chat, pipeline: pipeline, host: host)
}
.fullScreenCover(item: $voice) { VoiceModeView(session: $0) }
```

Every start makes a new session. Before it touches the microphone, `start()` checks
`pipeline.missingSetup()`, then asks for microphone and speech-recognition access. If any of these fails, the
screen shows the message, and nothing is recorded or added to the conversation.

## 5. The reply pipeline

`ReplyPipeline` is the seam between voice mode and whatever answers. Voice mode calls it in two ways:

- `stream(ReplyRequest(messages:pendingUser:commitGate:inputIsVoice:deep:))` for each reply;
- `prewarm(inputIsVoice:)` when the user starts talking.

`missingSetup()` is called before a session starts. `displayName(for:)` names the model that answered.

**Events.** Voice mode handles the `AssistantEvent`s like this:

| Event | What voice mode does |
|---|---|
| `.reply(.text)` | Adds it to the reply and speaks it sentence by sentence |
| `.reply(.activity)` | Shows the line under the orb, for example "Checking the plan" |
| `.reply(.finished(stop))` | Records it with `ReplyOutcome.apply`. A refusal is cleared and answered with a spoken apology. |
| `.toolRound` | Stores it with the reply. A reply that ran tools stays in the history even if it was cut off before it said anything. |
| `.cue` | Plays the cue when the cue policy allows it. A cue is never stored. |
| `.routed`, `.progress` | Used for the model name and the latency trace |
| a thrown error | Marks the reply failed and shows the error's `localizedDescription`. It then says "Sorry, something went wrong." |

**Early start.** When `supportsEarlyStart` is true and `earlyReplyStart` allows it, voice mode can start a
reply while the user's turn is still open. That request carries:

- `pendingUser`, the transcript so far, which isn't in `messages` yet;
- a `CommitGate`.

Voice mode buffers the reply's events without playing them, then does one of two things:

- **Adopts** the reply if the committed turn asks the same thing. It opens the gate and plays what it buffered.
- **Cancels** it otherwise. It cancels the gate and starts a fresh reply.

A pipeline that sets `supportsEarlyStart` to true must hold every side effect until the gate opens:

- Client tools that change something get the gate through `ToolContext(commitGate:)`. `ToolRunner` waits on
  it, so they act only once the turn is final.
- A request that leaves a durable record somewhere waits for the gate before it is sent at all
  (`CommitGatedProvider`, section 7.4).

A pipeline that can't hold its side effects keeps `supportsEarlyStart` false, and every reply then starts
when the turn is final.

## 6. Languages

**What's on screen** comes from `VoiceKit/Sources/VoiceKit/Localizable.xcstrings`, in English and
Simplified Chinese, looked up with `bundle: .module`. iOS resolves a package's strings in the app's own
language, picked from the localizations the app declares (`CFBundleLocalizations` or its `.lproj` folders):

- **This app** is English-only (`knownRegions` is `en`), so the voice screen stays English even on a Chinese
  iPhone.
- **An app that declares `zh-Hans`** gets the Chinese strings when iOS runs it in Chinese. That depends on the
  iPhone's language or the app's language in iOS Settings.

The SwiftUI `\.locale` environment doesn't change these strings, and neither does the `locale:` argument
of `String(localized:)`. Both only affect formatting. A check on this Mac confirmed it: with the language
set to `zh-Hans` and a bundle that has a `zh-Hans.lproj`:

- an executable that declares only `en` got the English string, even with `locale: zh-Hans`;
- one that declares `en` and `zh-Hans` got the Chinese string.

**What's spoken** follows `settings.speechLocale`:

- speech recognition;
- the built-in voice's language;
- the spoken cues (`SpokenCue.phrase`, Chinese for any `zh…` locale);
- the default question for a photo sent without words.

Two spoken apologies are still English only: "Sorry, something went wrong." and "Sorry, I can't help with that
one." Work item R8 fixes that.

**A new string** goes into the catalog with its Chinese translation. `testEveryCatalogKeyHasChinese` fails
for any key without one.

## 7. Roam

Roam is a SwiftUI app for iOS 18 that is built with XcodeGen. It has an Assistant tab backed by its own
backend (`POST /v1/assistant`). Its language setting is its own, `store.language` (`en` or `zh-Hans`),
separate from the iPhone's.

Roam needs voice mode that:

- answers quick app actions on device, and works offline;
- sends open questions to the backend;
- never sends the backend a request the user didn't finish;
- works in both of its languages.

The code in this section is a sketch for the Roam round (section 7.8), and nothing here compiles it. It
was checked once, when this guide was written, against this branch's VoiceKit and AssistantKit for an iOS 18
simulator. That check used stand-ins for:

- Roam's `RoamStore`, `RoamRecord`, `AIResult` and `JSONValue`, using the names and signatures in
  `ios/Roam/Core`;
- two types the round adds or makes public: `LocalEngineHost` and `LocalProvider` (R1), and
  `CommitGatedProvider` (R2).

R3–R5 write the real code in Roam and test it there.

### 7.1 Packages (R6)

From the root of the Roam repository:

```sh
git submodule add https://github.com/zqang/alvin-personal-assistant.git ios/Vendor/personal-assistant
git -C ios/Vendor/personal-assistant checkout <tag>
```

In `ios/project.yml` (paths are relative to `ios/`):

```yaml
packages:
  VoiceKit:
    path: Vendor/personal-assistant/VoiceKit
  AssistantKit:
    path: Vendor/personal-assistant/AssistantKit
```

Both the `Roam` and `RoamPreview` targets need these dependencies:

```yaml
- package: VoiceKit
- package: AssistantKit
```

VoiceKit finds AssistantKit through its own `../AssistantKit` path inside the submodule. After R1, the
on-device engine needs one more package entry. This repository needs no change for any of this.

The Info.plist already has the microphone, speech, camera and `audio` background entries (section 2). Its
`zh-Hans.lproj/InfoPlist.strings` translates them.

**One name clash.** Roam has its own `JSONValue`, and inside Roam that name means Roam's type. Tool code
therefore writes AssistantKit's as `AssistantKit.JSONValue` (section 7.3).

### 7.2 Conversation and host (R5)

**The conversation.** Roam's voice conversation starts as the sample's in-memory `Message` and `Chat` from
section 3, used as they are. The backend already keeps the questions it answers, in the Assistant tab's
history.

**The host** sets two things:

- the speech language, from Roam's own setting;
- deep mode off, so "Think deeper" stays hidden.

```swift
/// Roam's host: speech follows Roam's own language setting.
@MainActor
final class RoamVoiceHost: VoiceSessionHost {
    let store: RoamStore

    init(store: RoamStore) {
        self.store = store
    }

    var settings: AssistantSettings {
        var settings = AssistantSettings()
        settings.speechLocale = store.language == "zh-Hans" ? "zh-CN" : "en-US"
        // Roam has no deep mode, so "Think deeper" stays hidden.
        settings.deepMode = .off
        return settings
    }
}
```

It has no OpenAI key, so the built-in voice speaks.

It reads no photos. Roam could get photo reading by adding the MiniCPM package and returning
`MiniCPMHost.shared.readPhoto` from `photoReader` once the reader is ready, as this app's
`VoiceHosting.swift` does. That isn't part of R1–R8.

### 7.3 Tools and profile (R4)

Roam's actions are `AssistantTool`s, and its words are an `AssistantProfile`. The profile tells the router
which requests are app actions, and the tools carry those actions out on device.

**What each tool declares:**

- **`effect`.** `.readOnly` tools run at once. `.sideEffect` tools wait for the commit gate: `ToolRunner`
  holds them until it opens.
- **`presentation`.** The activity line shown under the orb and the cue that's spoken. The registry is built
  for each reply, so both are in Roam's language at that moment.
- **The summary.** The line returned with `.ok(…, summary:)` is stored with the reply's tool round, so the
  history shows what was done.

```swift
extension AssistantProfile {
    static let roam = AssistantProfile(
        persona: { name in "You are Roam, a travel assistant\(name.isEmpty ? "" : " for \(name)") in a trip-planning app." },
        localToolHint: "Use your tools to read the open trip's plan and to save ideas to it.",
        intents: IntentClassifier(
            actionPhrases: ["my plan", "itinerary", "what's next", "save", "remember", "行程", "接下来", "保存", "记下"],
            actionVerbs: ["save", "remember"],
            // In Roam these ask about the trip, not the web.
            freshFactsPhrases: IntentClassifier.freshFactsPhrases.filter { !["today's", "flight", "今天的"].contains($0) }
        ),
        claims: SuccessClaim(nouns: ["idea", "trip", "灵感", "行程"])
    )
}

/// Roam's on-device tools. Built for each reply, so their lines on screen are in Roam's language
/// of the moment.
@MainActor
enum RoamTools {
    static func registry(store: RoamStore, tripID: String, handoff: Bool) -> ToolRegistry {
        let chinese = store.language == "zh-Hans"
        var tools: [any AssistantTool] = [
            TripPlanTool(store: store, tripID: tripID, chinese: chinese),
            SaveIdeaTool(store: store, tripID: tripID, chinese: chinese),
        ]
        if handoff { tools.append(AssistantProfile.roam.handoffTool) }
        return ToolRegistry(tools)
    }
}

/// Reads the open trip's plan. Read-only, so it runs at once, even in a reply started early.
struct TripPlanTool: AssistantTool {
    let store: RoamStore
    let tripID: String
    let chinese: Bool

    var definition: ToolDefinition {
        ToolDefinition(
            name: "trip_plan",
            description: "The open trip's activities in order, with their start times (ISO 8601 with the local offset) and places.",
            inputSchema: ["type": "object", "properties": [:], "additionalProperties": false]
        )
    }

    var effect: ToolEffect { .readOnly }
    var presentation: ToolPresentation {
        ToolPresentation(activity: chinese ? "正在查看行程" : "Checking the plan", cue: .checking)
    }

    func run(_ input: [String: AssistantKit.JSONValue], context: ToolContext) async throws -> ToolOutput {
        let activities = await store.records(kind: "activity", tripID: tripID)
        let plan = activities.sorted { $0.text("startsAt") < $1.text("startsAt") }.prefix(30).map { activity -> AssistantKit.JSONValue in
            ["title": .string(activity.text("title")), "startsAt": .string(activity.text("startsAt")), "location": .string(activity.text("location"))]
        }
        return .ok(["activities": .array(Array(plan))])
    }
}

/// Saves an idea to the open trip. A side effect: `ToolRunner` runs it only once the reply's commit
/// gate opens, so a reply started early never saves anything for a turn the user didn't finish.
struct SaveIdeaTool: AssistantTool {
    let store: RoamStore
    let tripID: String
    let chinese: Bool

    var definition: ToolDefinition {
        ToolDefinition(
            name: "save_idea",
            description: "Saves a place or plan to the open trip's ideas. Only when the user asks to save or remember something.",
            inputSchema: [
                "type": "object",
                "properties": [
                    "title": ["type": "string", "description": "A short name, such as the place."],
                    "notes": ["type": "string", "description": "Anything else the user said about it."],
                ],
                "required": ["title"],
                "additionalProperties": false,
            ]
        )
    }

    var effect: ToolEffect { .sideEffect }
    var presentation: ToolPresentation {
        ToolPresentation(activity: chinese ? "正在保存灵感" : "Saving the idea", cue: .working)
    }

    func run(_ input: [String: AssistantKit.JSONValue], context: ToolContext) async throws -> ToolOutput {
        let title = input["title"]?.stringValue ?? ""
        let notes = input["notes"]?.stringValue ?? ""
        try await store.save(kind: "idea", tripID: tripID, data: ["title": .string(title), "notes": .string(notes)])
        return .ok(["saved": true], summary: (chinese ? "灵感：" : "Idea: ") + title)
    }
}
```

The profile goes wherever a profile is asked for. See the README's
[Reuse in another app](../README.md#reuse-in-another-app).

### 7.4 The backend as the cloud engine (R3)

`RoamAssistantProvider` is an `AssistantProvider` used as `Orchestrator.Engines.cloud`. For each reply:

- **It sends one request.** It sends `POST /v1/assistant` once, under a new request ID, through Roam's durable
  request. The answer then also appears in the Assistant tab. It sends only the newest user turn: the
  backend adds the trip and its own history, and doesn't use `system`.
- **It polls only to see the request arrive.** It asks `GET /v1/assistant/:id` every 0.5 s until the backend
  knows the request (it answers 202 while it works on it). At that point it yields
  `.progress(.responseStarted)` once. The backend answers the POST only after inference, which can take up to
  45 s on the home machine. Without this mark, the orchestrator's first-event watchdog (8 s by default) would
  also answer on device while the backend is still working.
- **It ends with one answer.** It yields one `.reply(.text)` and then `.reply(.finished(.completed))`.
- **A failed result is final.** The backend's failed results are terminal, so the provider throws them with
  the backend's text, which voice mode shows.
- **It never retries.** After an uncertain failure, such as a timeout or a dropped connection, the stream
  fails and the POST is never sent again. The backend contract forbids repeating an inference request on an
  uncertain failure. If the failure counts as a connectivity failure and nothing was committed yet, the
  orchestrator may answer on device instead. Either way, Roam's durable request keeps the ID, so the
  Assistant tab can collect the answer later with a GET.

The pipeline wraps the provider in `CommitGatedProvider` whenever the request has a gate. A reply that
started early then sends nothing to the backend until the turn is final, and nothing at all if the turn is
taken back. The wait for the gate counts against the watchdog. With the default end-of-turn delay of 0.9 s,
that wait is usually under a second: an early start fires after 0.25–0.6 s of unchanged transcript.

```swift
/// Roam's backend as the cloud engine: one POST /v1/assistant per reply, never repeated. It sends
/// the newest user turn; the backend adds the trip and its own history, and ignores `system`.
struct RoamAssistantProvider: AssistantProvider {
    /// POST /v1/assistant with this request ID and question, through Roam's durable request, so
    /// the answer is also kept in the Assistant tab. Returns the final result.
    let send: @Sendable (_ requestID: String, _ question: String) async throws -> AIResult
    /// GET /v1/assistant/:requestID; nil while the backend doesn't have the request (404).
    let check: @Sendable (_ requestID: String) async throws -> AIResult?

    func streamEvents(system: String, turns: [ChatTurn]) -> AsyncThrowingStream<AssistantEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let id = UUID().uuidString
                    let question = turns.last(where: { $0.role == .user })?.text ?? ""
                    let result = try await withThrowingTaskGroup(of: AIResult?.self) { group in
                        group.addTask { try await send(id, question) }
                        group.addTask {
                            // Polls until the backend has the request. `.responseStarted` then keeps
                            // the orchestrator's watchdog from also answering on device.
                            while (try? await check(id)) == nil {
                                try await Task.sleep(for: .milliseconds(500))
                            }
                            continuation.yield(.progress(.responseStarted))
                            return nil
                        }
                        defer { group.cancelAll() }
                        for try await case let result? in group {
                            return result
                        }
                        throw CancellationError()
                    }
                    // A failed result is terminal: voice mode shows its text and doesn't ask again.
                    guard result.status == "completed" else { throw RoamAssistantFailed(text: result.text) }
                    continuation.yield(.reply(.text(result.text)))
                    continuation.yield(.reply(.finished(.completed)))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

struct RoamAssistantFailed: LocalizedError {
    let text: String
    var errorDescription: String? { text }
}
```

### 7.5 The pipeline: quick actions on device, open questions on the backend (R5)

`RoamReplyPipeline` uses AssistantKit's router and orchestrator, the same ones this app's
`OrchestratedReplyPipeline` uses:

- **The route.** `RouteDecider` runs with `.automatic(preferLocal:)`, which is true only for requests the
  profile classifies as app actions:
  - an app action goes on device once the model is loaded, with the backend as the fallback (before that, it
    goes to the backend);
  - small talk in voice goes on device once the model is loaded;
  - offline, or without an account, everything goes on device;
  - everything else goes to the backend. When the backend can't be reached, the reply falls back to the
    device, except for complex questions (long or multi-part ones, comparisons and plans).
- **The on-device side.** The model gets Roam's tools and `handoff_to_cloud`, so it can pass a question it
  can't answer to the backend.
- **Deep mode.** Roam has none.

```swift
/// Roam's replies: its own actions on device, open questions on the backend.
@MainActor
final class RoamReplyPipeline: ReplyPipeline {
    let store: RoamStore
    let tripID: String

    init(store: RoamStore, tripID: String) {
        self.store = store
        self.tripID = tripID
    }

    var supportsEarlyStart: Bool { true }

    private var signedIn: Bool { store.session != nil && !store.isDemo }

    func stream(_ request: ReplyRequest) -> AsyncThrowingStream<AssistantEvent, Error> {
        let turns = PromptBuilder.turns(from: request.messages + [request.pendingUser].compactMap { $0 })
        let system = PromptBuilder.systemPrompt(userName: store.session?.user.name ?? "", customInstructions: "", profile: .roam)
        let intents = AssistantProfile.roam.intents.classify(turns.last(where: { $0.role == .user })?.text ?? "")
        let local = LocalEngineHost.shared
        let signals = RouteSignals(
            // Roam's own actions prefer the device; everything else prefers the backend.
            policy: .automatic(preferLocal: intents.contains(.deviceAction)),
            isOnline: !store.isOffline,
            cloudConfigured: signedIn,
            localAvailable: local.isDownloaded,
            localReady: local.status == .ready,
            inputIsVoice: request.inputIsVoice,
            deepMode: .off
        )
        let decision = RouteDecider(classifier: AssistantProfile.roam.intents).decide(intents: intents, signals: signals)

        let store = store
        let tripID = tripID
        let remote = RoamAssistantProvider(
            send: { id, question in try await store.askAssistant(tripID: tripID, mode: "chat", message: question, requestID: id) },
            check: { id in
                do {
                    return try await store.api.get("/v1/assistant/\(id)", as: AIResult.self)
                } catch let error as APIError where error.status == 404 {
                    return nil
                }
            }
        )
        let gate = request.commitGate
        // A reply started early reaches the backend only once its turn is final.
        let backend: any AssistantProvider = if let gate { CommitGatedProvider(base: remote, gate: gate) } else { remote }
        let tools = ToolRunner(registry: RoamTools.registry(store: store, tripID: tripID, handoff: signedIn), lenientInput: true)
        let engines = Orchestrator.Engines(
            cloud: signedIn ? backend : nil,
            local: local.isDownloaded ? LocalProvider(settings: AssistantSettings(), tools: tools, handoffAvailable: signedIn, commitGate: gate) : nil
        )
        return Orchestrator(decision: decision, engines: engines).streamEvents(system: system, turns: turns)
    }

    func missingSetup() -> String? {
        if signedIn || LocalEngineHost.shared.isDownloaded { return nil }
        return store.l("Sign in, or download the on-device model, to talk to Roam.", "请登录或下载设备端模型，再与 Roam 对话。")
    }

    func prewarm(inputIsVoice: Bool) {
        LocalEngineHost.shared.prewarm()
    }

    func displayName(for decision: RouteDecision?) -> String {
        decision?.engine == .local ? store.l("On this iPhone", "本机") : store.l("Roam assistant", "Roam 助手")
    }
}

struct TripVoiceButton: View {
    let store: RoamStore
    let tripID: String
    @State private var voice: VoiceSession?

    var body: some View {
        Button(store.l("Talk", "语音对话"), systemImage: "waveform") {
            voice = VoiceSession(conversation: Chat(), pipeline: RoamReplyPipeline(store: store, tripID: tripID), host: RoamVoiceHost(store: store))
        }
        .fullScreenCover(item: $voice) { VoiceModeView(session: $0) }
    }
}
```

**Orientation.** Roam supports portrait and both landscape orientations, but voice mode is portrait only
(section 2). While the cover is up, an app delegate (`@UIApplicationDelegateAdaptor`) returns `.portrait`
from `application(_:supportedInterfaceOrientationsFor:)`. If the iPhone is in landscape when the cover opens,
Roam also asks the window scene for portrait with `requestGeometryUpdate(.iOS(interfaceOrientations: .portrait))`.

### 7.6 Bilingual UI

Roam's language reaches each part of voice mode as follows:

| Part of voice mode | Where its language comes from |
|---|---|
| Recognition, the built-in voice, spoken cues | `settings.speechLocale`, which the host derives from `store.language` (section 7.2). It's read when a session starts. |
| Tool activity lines and stored summaries | `RoamTools.registry`, built for each reply from `store.language` |
| `missingSetup()` and `displayName(for:)` | `store.l(…)` in the pipeline |
| The voice screen's own text (VoiceKit's catalog) | Today: the iPhone's language for Roam, as section 6 explains, which may differ from `store.language`. After R8: `settings.speechLocale`, as long as the app declares that language. |
| The two spoken apologies | English today; after R8, `settings.speechLocale` |

The voice screen's own text is the one gap. Roam's in-app language changes at run time without a relaunch,
and nothing in a package's string lookup sees it. R8 closes the gap inside VoiceKit:

- VoiceKit's lookups use the catalog's localization that matches `settings.speechLocale`, chosen with
  `Bundle.preferredLocalizations(from:forPreferences:)` from among the languages the app declares.
- This app declares only English, so it stays English. Roam follows `store.language`.
- No new host member is needed.

### 7.7 Navigation audio (R7)

Voice mode and Roam's turn-by-turn navigation both use the shared audio session:

- **Voice mode** sets `.playAndRecord`. With echo cancellation it also uses the `.voiceChat` mode, which
  plays everything through the voice-processing path. When it stops, it deactivates the session with
  `.notifyOthersOnDeactivation`.
- **Navigation** speaks its prompts through an `AVSpeechSynthesizer`.

In R7, Roam's host pauses the spoken prompts in `voiceSessionWillStart` and resumes them in
`voiceSessionDidStop`. Guidance and location updates keep running in between, and `NavigationSpeech` already
has `setEnabled(_:)`.

Two things need a check on a device:

- a prompt that is already playing when voice mode starts;
- a route change, such as CarPlay or headphones. Voice mode answers a route change by rebuilding its audio
  graph and handing the turn back to the user.

### 7.8 The Roam round

| | Where | Work | Done when |
|---|---|---|---|
| **R1** `LocalEngineHost` package | this repository | Move `LocalModelHost` and `LocalProvider` from the app into a package. The app's ties become hooks, like the existing `systemProvider` and `isOnline`: the device tools in the prewarm prompt, MiniCPM-V and Qwen3-ASR. | This app behaves as before, and Roam can answer on device |
| **R2** public `CommitGatedProvider` | this repository | Move the private struct in `OrchestratedReplyPipeline.swift` into AssistantKit as public, with a test | AssistantKit tests cover it; the app uses the public one |
| **R3** Roam provider | Roam | `RoamAssistantProvider` (section 7.4). `RoamStore.askAssistant` takes the request ID instead of making its own. | A test with a stub API: one POST per reply; `.responseStarted` on the first 202; no POST after an uncertain failure; nothing sent before the gate opens |
| **R4** Roam tools and profile | Roam | `RoamTools` and `AssistantProfile.roam` (section 7.3) | Tool tests against a demo store; the profile classifies the English and Chinese action phrases |
| **R5** Roam pipeline | Roam | `RoamReplyPipeline`, `RoamVoiceHost`, the conversation and a voice button that keeps voice mode in portrait (sections 7.2 and 7.5) | Routing tests for each row of section 7.5; turning the iPhone during voice mode leaves the screen in portrait |
| **R6** submodule and XcodeGen | Roam | Section 7.1 | `xcodegen generate` and the Roam and RoamPreview builds pass. VoiceKit builds for Roam's iOS 18. |
| **R7** navigation audio | Roam | Section 7.7 | On a device: voice mode during guidance, then guidance after it |
| **R8** localized spoken fallbacks and the in-app language | this repository | Speak the two apologies in `settings.speechLocale`, the way `SpokenCue.phrase` does. Resolve the voice screen's strings for that locale (section 7.6). | VoiceKit tests in both languages. Launched with `-AppleLanguages "(zh-Hans)"`, this app's voice screen stays English. |

**Order.** R1, R2 and R8 come first, here, under a new tag. Then R6, R4, R3, R5 and R7, in Roam.

## 8. Tests and CI

```sh
cd VoiceKit && xcodebuild test -scheme VoiceKit -destination 'platform=iOS Simulator,name=<an iPhone simulator>'
```

The tests check these things:

- a host that takes every default compiles and gets them;
- a session whose pipeline reports missing setup fails before it touches the audio session, and calls
  neither lifecycle hook;
- `ReplyOutcome` records refused, finished and stopped replies;
- every catalog string has Simplified Chinese;
- `PendingPhoto` (`PendingPhotoTests.swift`):
  - a photo comes out upright, at most 1344 px, as JPEG and without GPS, and data that isn't an image is
    refused;
  - a failed or empty read gives the unreadable note, and a cancelled or host-stopped read starts over;
  - `noteUnread` reads again only the photos whose read was cut short, and throws `PhotoReadStopped` only for
    the attached photo;
  - the default question is in Simplified Chinese, Traditional Chinese or English.

`.github/workflows/ios.yml` runs the same command on the first available iPhone simulator.

On a device, voice mode hosted through VoiceKit behaved the same as before the move (check D26 in
[the architecture doc's §11](architecture-plan-on-device-and-orchestration.md#11-device-checklist), passed by
hand on an iPhone 17 Pro Max on 16 October 2026).
