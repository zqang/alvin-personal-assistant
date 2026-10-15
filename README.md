# Alvin's Personal Assistant

An iPhone voice assistant in the spirit of ChatGPT Voice and Doubao (豆包): tap the waveform, just talk, and it
talks back. There's no push-to-talk. It notices when you've finished speaking, answers out loud sentence by
sentence, and stops as soon as you start talking over it. Text chat and conversation history are built in, and
conversations started by voice show up as transcripts in the chat.

Replies come from **Claude** by default (with live web search). You can also plug in any OpenAI-compatible
service, such as **Doubao on Volcengine Ark**, DeepSeek, or OpenAI.

## What it does

- **Hands-free voice mode.** A full-screen animated orb that reacts to your voice and the assistant's voice.
  Live captions, mute, and "tap the orb to send now or to interrupt".
- **Natural turn-taking.**
  - The end of your turn is detected from recognizer silence plus an adaptive noise-floor voice detector.
  - It waits longer after words like "and…" or "然后…" and less after a finished question.
  - If you keep talking before the answer starts, it takes your turn back instead of answering half a
    sentence. Once the answer has used a tool (to set a timer, say), your words start the next turn instead.
- **Interrupt by talking (barge-in).** Speech is played through the same audio engine as the microphone, so
  Apple's echo cancellation removes it. A second filter ignores words that match what the assistant is saying,
  so it doesn't interrupt itself.
- **Low latency.**
  - Speech recognition runs on device.
  - The reply streams in, and speech starts after the first sentence (or clause) instead of waiting for the
    whole answer.
  - Claude runs at low effort, with a prompt tuned for fast first words.
- **Voices.** Built-in iOS voices (it picks the best installed Enhanced or Premium voice, and switches voice
  when a sentence is in another language), or OpenAI voices for a more natural sound.
- **Chinese and English.** Choose the recognition language in Settings; the assistant replies in the language
  you speak.
- **Sharper listening (optional).** Turn on **Qwen3-ASR listening** and each finished turn is transcribed again
  on the iPhone by [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) before it's sent, for English and
  Chinese (China mainland). Apple's recognizer still drives the live captions and turn-taking.
- **On-device replies (optional).** Choose **Provider › On this iPhone** and replies come from a model running
  on the phone with MLX, offline once downloaded:
  - [Underdog Woof 4B](https://huggingface.co/ConwayResearch/Underdog-Woof-4B-1.1)
  - Underdog Woof 2B
  - Qwen3.5 2B
  - Qwen3 4B with a 0.6B draft model for speculative decoding

  The model keeps its cache between turns, so each reply only reads the new message. **Settings › On-device
  engine › Benchmark** measures it on your phone. A mode in which the much larger MiniCPM-V 35B answers,
  **Most accurate on device**, is built but hidden in this version, because it was too slow in voice mode
  ([below](#most-accurate-on-device)).
- **Photos (optional).** Attach a photo in the chat, or take one in voice mode, and ask about it. A small
  vision model reads it on the iPhone, and only the text it reads goes to the model that answers
  ([Photo reading](#photo-reading)).
- **Web search.** "What's the weather tomorrow?" and "Any news on…?" work, because Claude searches the web
  server-side.
- **Private by default.** Speech is transcribed on the iPhone. API keys live in the iOS Keychain. There's no
  backend.

## On-device engine and new settings

On-device replies run on the **Alvin engine** (`LocalEngine/`), a Swift inference engine on MLX built for
Woof and Qwen. It keeps the conversation's tokens in the model's cache and feeds only what changed, with exact
rewind points for the hybrid Qwen3.5 layers; it keeps the processed system prompt on disk, so a new session
starts warm; it checks drafted tokens in blocks (lossless speculative decoding, which pays most in tool calls
and replies that repeat the prompt); and it lets the model use the device tools. The first time a model loads,
a self-test of a few seconds checks the engine on your iPhone. With **Settings › On-device engine › Engine ›
Automatic** (the default) the engine answers only once that test has passed, and MLX's stock session answers
until then; the same section sets speculative decoding (off, tool calls only, automatic) and the system prompt
on disk, measures the phone's speed and opens a benchmark that compares the two engines. The other new
settings are **Routing** (off by default: automatic routing sends each request to Claude or to the iPhone by
connection, content and recent speed, with "Prefer on-device" and "Answer small talk on device"), **Tools**
(reminders, calendar and timers, for Claude and for the on-device model), **Deep thinking** ("Think deeper" in
the chat and in voice mode: one careful answer, or three perspectives merged into one, with a daily limit) and
**Voice responsiveness** (start answering early, short spoken cues such as "Let me check.", a turn chime,
lowering the voice when you talk over it, and a latency report). The design, the CI checks and what still
needs testing on a device are in
[docs/architecture-plan-on-device-and-orchestration.md](docs/architecture-plan-on-device-and-orchestration.md).

## Requirements

- A Mac with **Xcode 26.4 or later** and the Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`),
  which the on-device Qwen3-ASR model (MLX) needs to build. For the simulator, an Apple silicon Mac: the
  simulator build is arm64 only.
- An iPhone on **iOS 17 or later**. A free Apple ID is enough to run it on your own phone.
- An **Anthropic API key** from [console.anthropic.com](https://console.anthropic.com), or a key for an
  OpenAI-compatible service

## Run it on your iPhone

1. Clone the repo and open `AlvinAssistant.xcodeproj` in Xcode. Xcode fetches the Swift packages, including
   the prebuilt MiniCPM-V runtime (a checksum-pinned `llama.xcframework` from this repository's GitHub
   releases), so there's nothing to build first and no CMake to install.
2. Select the **AlvinAssistant** target, open **Signing & Capabilities**, and choose your **Team**. If Xcode
   says the bundle identifier is taken, change `com.alvinang.personalassistant` to something unique.
3. Plug in your iPhone (or pair it over Wi-Fi), select it as the run destination, and press **Run** (⌘R). The
   first time, trust the developer profile on the phone under **Settings › General › VPN & Device
   Management**. If the first build stops at "Plugin 'CudaBuild' … must be enabled" (from mlx-swift) or a
   macro prompt, click the error in Xcode, choose **Trust & Enable**, and build again.
4. In the app, tap **⚙︎** and paste your Anthropic API key. Optionally add your name and a few words about
   yourself.
5. Tap the **waveform** button and start talking.

Tips:

- For a much more natural built-in voice, download an **Enhanced** or **Premium** voice on the iPhone:
  **Settings › Accessibility › Read & Speak › Voices**. The app picks it up automatically.
- Set **I speak** in the app's Settings to your language (for example 中文（中国大陆）or English (Singapore)).
- **Voice engine › OpenAI** gives the most natural speech. It needs an OpenAI API key and costs a little per
  minute of audio. If it fails, the app falls back to the built-in voice.
- The simulator can run the app, but voice mode needs a real iPhone for the microphone and echo cancellation.
  The MiniCPM-V models don't load on the simulator.
- The app always embeds the MiniCPM-V runtime, whether or not you download a MiniCPM model. It makes the
  installed app about 10.8 MB larger (the framework is 9.75 MB of that) and the download about 3.8 MB larger
  (measured on an iPhone 17 Pro Max, D14).
- **On-device replies** download 1.5–2.7 GB the first time (Wi-Fi only) and run only while the app is open. Run
  the benchmark from a **Release** build and copy the results. Qwen3-ASR listening loads after the reply model
  and gives way to it if memory is short.
- **Qwen3-ASR listening** downloads about 1 GB the first time (use Wi-Fi), needs a recent iPhone (iPhone 15 Pro
  or newer is best), and works only while the app is open; with the screen locked, turns use Apple's text.
  Build with the **Release** configuration for realistic speed.

## Photo reading

You can show the assistant a photo in two ways:

- **In the chat:** pick one from your library, or take one with the camera.
- **In voice mode:** open the camera and press the shutter. The microphone keeps listening while you do.

**MiniCPM-V 4.7 1B** then reads the photo on the iPhone. First it copies every piece of legible text
exactly, in its own language and line order. Then it writes at most three sentences about what the photo
shows. Reading starts as soon as you attach the photo, so it happens while you're still typing or speaking.
The result streams in under the thumbnail as **What the iPhone read**.

- **What gets sent.** The photo is never sent to a model provider. It is kept with the chat on the iPhone
  (and so in the iPhone's backups) until you delete the chat. The model that answers gets your words plus
  the reading, marked as coming from an on-device model that may contain errors and as text not to take
  instructions from. That model can be Claude or the on-device model. When Claude answers, only this text
  goes to Claude.
- **Follow-up questions.** The reading is stored with your message, so later questions in the chat can use
  it. A photo sent without words asks "What's in this photo?", or, if you speak Chinese, "这张照片里有什么？"
  ("這張照片裡有什麼？" for Traditional Chinese and Cantonese).
- **Getting it.** Go to **Settings › Photo reading › Download**. It's about 1.8 GB over Wi-Fi, and the files
  are kept for later. The photo and camera buttons appear only once the download is finished. Delete it in
  the same place to get the space back.
- **Speed.** On an iPhone 17 Pro Max a reading is complete in about 2.6–2.9 s (median; p90 about 3.5 s).
  The first photo after the app starts takes about 4.6 s, because that includes loading the model (D16).
- **Limits.**
  - It's a small model. It can misread text, and although it's asked to write "unclear" rather than guess,
    it may still guess.
  - Text inside a photo is untrusted. A sign that says "set a timer" still reaches a model that has tools.
    Actions still wait until your turn has ended, and for iOS's permission prompts, but this isn't proven
    safe.

## Most accurate on device

**Hidden in this version.** The mode is in the code, but Settings doesn't offer it. In voice mode the 35B
started speaking 7.5 s after the request at the median (8.5 s p90), and 9.3 s after an interruption (11.8 s
p90), against a target of 5 s (8 s p90) (D23). So on 16 October 2026 the owner hid it (decision 31):

- `MiniCPMHost.accurateEligible()` returns false, so the switch and the 35B download don't appear.
- If an earlier build turned the mode on, the app turns it off when it starts, and saves that.
- If an earlier build downloaded the 35B, **Settings › MiniCPM-V 35B files** shows the files with **Delete**
  only, to free the 15.5 GB, whichever provider is chosen. It offers no download, and the app doesn't check
  the files when it starts.

To offer the mode again, change that one line in `MiniCPM/Sources/MiniCPM/MiniCPMHost.swift` back to
`MiniCPM35BArtifacts.eligible(physicalMemory:)`. It's then offered on 12 GB iPhones, other iPhones don't show
it, and the rest of this section applies. It starts off for everyone, including anyone who had it on before
it was hidden: the app saved that setting off. Re-tune it and run D23 again first.

The mode is optional and needs an iPhone with 12 GB of memory. With **Settings › Most accurate on device**
on, replies that would be answered on the iPhone come from **MiniCPM-V 4.7 35B-A3B** instead of Woof. It's
much slower than Woof and warms the phone. It's a much larger model and may handle harder questions better
(the measured accuracy edge isn't significant yet); use it for that, not for speed.

- **Woof still answers in these cases:**
  - reminders, calendar and timers: the 35B has no tools or live information in this version, and it says
    so if you ask it to do something;
  - when the phone is hot (`serious` thermal state) or in Low Power Mode;
  - the follow-up to a reminder, calendar or timer request ("5 pm", "yes, do it");
  - when the 35B fails before it has written anything, or another model needs its memory first.

  Photos are still read by the 1B.
- **What you need:**
  - Woof, already downloaded;
  - an iPhone that reports at least 11 GB of memory. The 17 Pro Max reports 12.26 GB. Other phones don't
    show the switch;
  - about 15.5 GB of storage, plus 1 GB free.

  The download is Wi-Fi only, and the app has to stay open while it runs. Files that have finished are
  kept if you leave the app, and checking them afterwards takes about 10 s. Delete them in the same section.
- **Speed.** In a chat its first words take about 5 s for a short message, longer for a long one. In voice
  mode it starts speaking about 7–9 s after you finish, and up to about 12 s after you interrupt it. Then it
  writes about 6 tokens a second. The very first answer, and the first after you change your name or
  instructions, can take about 25 s.
- **Engine.** It runs a second, prebuilt runtime, llama.cpp-omni. The model keeps the conversation in its
  cache between turns and keeps the system prompt on disk, so each reply reads only the new text. After a
  photo it reads the whole conversation again, which took 28–114 s in testing, because the photo reader needs
  its memory. A conversation that outgrows the model's 4,096-token context is rebuilt from its latest turns
  every few replies, and that reply starts much later (not measured yet).

These are the settings the mode runs with, measured on an iPhone 17 Pro Max on 15–16 October 2026 (D22 and
D23). Times are taken where the app receives the text; [the evaluation](docs/minicpm-evaluation.md) has the
details.

| | Woof 4B | MiniCPM-V 35B-A3B |
|---|---|---|
| 24 small bilingual tasks | 22/24 | 24/24 in every run. The gap is two arithmetic items and is **not significant** (p = 0.5) |
| First text, short benchmark prompt | 0.16 s | 2.30 s (p90 2.44 s) |
| First text, prompt of about 500 / 1,000 / 2,000 tokens | 1.1 / 2.2 / 4.2 s | 20–24 / 40–47 / 83–94 s: about 40 ms per prompt token |
| First text, next turn of a conversation | 0.26 s | 4.9 s |
| Decode speed | about 23–24 tok/s | 5.9 tok/s. OpenBMB quotes 13+ tok/s, and the gap is unexplained |
| First audio in voice mode | 1.5 s after you stop speaking ([D8](docs/architecture-plan-on-device-and-orchestration.md#11-device-checklist)) | 7.5 s after the request (p90 8.5 s), about 8.6 s after you stop speaking; 9.3 s after a barge-in (p90 11.8 s) |
| Memory | 2.6–2.9 GB, weights included | 2.78 GB, **plus** 1.65 GB of memory-mapped weights that this figure leaves out (4.41 GB resident) |
| Heat and battery | — | Every benchmark run ended at `serious`. Battery use is not measured yet (D24) |

The 35B runs without the DSpark draft model: on the phone the draft made the first text slower, didn't
speed up decoding by the 10% required, and pushed resident memory to 5.2–6.0 GB. **In voice mode the 35B
misses the target** of first audio within 5 s (p50), which is why this version hides it. The runtime keeps a
small Metal patch that adds autorelease pools. It made no measurable difference on the iPhone (D20 A/B) and
does no harm, so the published runtime stays as it is; the patch goes only when the runtime is rebuilt for
another reason.

The full record is in [docs/minicpm-evaluation.md](docs/minicpm-evaluation.md): every run, the memory
accounting, the device checks that are still pending, and how to run the benchmarks yourself.

## Use Doubao, DeepSeek, or OpenAI instead of Claude

In Settings, choose **Provider › OpenAI-compatible**, tap **Fill in a preset**, then enter the model ID and the
API key:

| Service | Base URL | Model |
|---|---|---|
| Doubao (Volcengine Ark) | `https://ark.cn-beijing.volces.com/api/v3` | Your model or endpoint ID from the Ark console |
| DeepSeek | `https://api.deepseek.com/v1` | e.g. `deepseek-chat` |
| OpenAI | `https://api.openai.com/v1` | e.g. `gpt-5-mini` |

Web search is only available with Claude.

## How it works

```
 mic ─▶ AVAudioEngine (voice processing: echo cancellation, noise suppression)
          │
          ├─▶ SFSpeechRecognizer (on-device, live partial transcripts)
          │        │
          │        ├─▶ TurnDetector ──▶ "the user is done" ──▶ Claude Messages API (streaming)
          │        └─▶ BargeInDetector ──▶ "the user is interrupting" ──▶ stop speaking
          │                                                                  │
          │                         SentenceChunker ◀── streamed text ◀──────┘
          │                               │
          │                  SpeechTextCleaner (strip Markdown, links, emoji)
          │                               │
          │            AVSpeechSynthesizer.write / OpenAI TTS (PCM stream)
          │                               │
          └──────────── AVAudioPlayerNode ◀┘ ─▶ speaker
```

The code is split in five:

- **`AssistantKit/`** is a Swift package with everything that isn't iOS-specific, all covered by unit tests:
  - the Claude streaming client with its tool loop, and the OpenAI-compatible client
  - SSE parsing
  - the sentence chunker and the speech text cleaner
  - turn detection and barge-in echo filtering
  - prompt and history building
  - the settings model
  - routing, tools, deep mode and voice-latency logic, and the on-device engine's session planning and
    drafting
- **`LocalEngine/`** is the on-device inference engine (MLX), tested on a Mac GPU in CI.
- **`MiniCPM/`** runs MiniCPM-V 4.7 on Apple devices through a prebuilt, pinned llama.cpp-omni runtime. It
  holds:
  - the 1B photo reader;
  - the 35B behind "Most accurate on device" (hidden in this version);
  - their downloads;
  - a benchmark command-line tool.
- **`VoiceKit/`** is hands-free voice mode as a package another app can embed:
  - the audio engine, speech recognition, and speech playback
  - `VoiceSession`, the state machine that ties them together, and the voice screen
  - its strings, in English and Simplified Chinese
- **`AlvinAssistant/`** is the iOS app:
  - SwiftUI views
  - SwiftData storage
  - the reply pipeline, the tools and the on-device models
  - how it hosts voice mode (`App/VoiceHosting.swift`)

AssistantKit and LocalEngine know nothing about reminders, calendars or timers. This app's domain lives in one
value, `AssistantProfile.app` (`AlvinAssistant/App/AppSetup.swift`), which comes from the `AlvinProfile` module
(`AssistantKit/Sources/AlvinProfile/`). It holds:

- the persona;
- the on-device prompt lines and the hand-off description;
- the words that ask for an action, in English and Chinese;
- what a reply claiming an action sounds like;
- the timer check.

### Reuse in another app

Add the `AssistantKit` (and, for on-device replies, `LocalEngine`) packages, write your tools as `AssistantTool`
types, and describe your domain in one profile in your own project. Everything left out keeps a neutral default.

```swift
extension AssistantProfile {
    static let app = AssistantProfile(
        persona: { name in "You are a friendly travel guide\(name.isEmpty ? "" : " for \(name)") in a trip-planning app." },
        localToolHint: "Use your tools to open trips, add places to a day, and show the app's features.",
        intents: IntentClassifier(
            actionPhrases: ["my trip", "itinerary", "add to day", "flight", "行程"],
            // Flights are this app's own action, not a web search.
            freshFactsPhrases: IntentClassifier.freshFactsPhrases.filter { $0 != "flight" }
        ),
        claims: SuccessClaim(nouns: ["trip", "itinerary"])
    )
}
```

Pass `.app` wherever a profile is asked for: `PromptBuilder.systemPrompt` and `localSystemPrompt`,
`RouteDecider(classifier: AssistantProfile.app.intents)`, `AssistantProfile.app.handoffTool`, and `LocalToolLoop(…, profile: .app)`.
The rules after the persona (spoken and typed replies, brevity, the tool cue) are shared `PromptBuilder` text, not
part of the profile.

To measure your tools on a phone, define a tool-call scenario with stand-in results, as
`AlvinProfile/ToolBenchmark.swift` does, and list it after the shared ones (`LocalBenchmark.scenarios + [yours]`). The
screen that plays it (`LocalBenchmarkRunner` and `BenchmarkToolExecutor` in `AlvinAssistant/OnDevice/LocalBenchmarkView.swift`)
is app code tied to `LocalModelHost`, so copy and adapt it.

For hands-free voice mode, add the `VoiceKit` package too. Give it your conversation, your reply pipeline and a
small host; everything else is optional. [The VoiceKit guide](docs/voice-kit.md) shows how, with a complete
host that the package's tests compile. It also shows how Roam, a travel app, plugs in: its tools, on-device
answers for its own actions, its backend for open questions, and its two languages.

### Claude API details

Swift has no official Anthropic SDK, so `ClaudeProvider` calls the Messages API over HTTP with streaming.

- **Model:** the catalog's newest Opus model by default (`ClaudeModelCatalog.defaultModelID`), with
  `output_config.effort: "low"` for fast spoken replies. Change the model and effort in Settings; the Sonnet
  and Haiku models are faster and cheaper.
- **Web search:** the server-side `web_search` tool. When a long search pauses the turn (`pause_turn`), the
  client resumes it automatically.
- **Refusal fallbacks are on** for the models that support them (`ClaudeModelCapabilities`), with
  `fallbacks: "default"` and the `server-side-fallback-2026-07-01` beta header. If Claude's safety classifiers
  decline a request, the API retries it on a fallback model instead of failing. Remove `fallbacks` in
  `ClaudeRequest.swift` if you'd rather not use it.
- **Prompt caching:** the system prompt is static, and each turn's local time is stored with the message rather
  than injected fresh. Every request's history is therefore an exact prefix of the next, and the conversation
  stays cached.

## Tests and CI

```bash
swift test --package-path AssistantKit
swift test --package-path MiniCPM      # the model tests skip unless model files are given
python3 -m unittest discover -s scripts -p 'test_*.py'
cd VoiceKit && xcodebuild test -scheme VoiceKit \
  -destination 'platform=iOS Simulator,name=iPhone 17'    # voice mode and attached photos; iOS only, so on a simulator
```

`.github/workflows/ios.yml` runs on every push and pull request, except ones that change only docs or
scripts. It runs these tests and builds the app with Xcode, and it also runs AssistantKit's tests on Linux. `.github/workflows/local-engine.yml` tests the on-device engine on a
Mac GPU, and `model-facts.yml` and `drafter-lab.yml` measure the models; see
[the CI section of the architecture doc](docs/architecture-plan-on-device-and-orchestration.md#10-ci-workflows-and-how-to-read-them).
The MiniCPM benchmarks and the slower model tests are described in
[the MiniCPM-V evaluation](docs/minicpm-evaluation.md#9-running-the-benchmarks).

## Security note

This is set up as a personal app: your own API key is stored in the Keychain on your own phone. Don't ship a
build with a key inside to other people. For a public app, put a small server in front of the model API and
have the app authenticate to that instead.

## Ideas for next steps

- **Speech-to-speech mode** (the OpenAI Realtime API or Doubao's realtime voice model) for the lowest possible
  latency and more expressive voices
- **Send the photo itself to Claude** when the on-device reading isn't enough (Claude supports images).
  Today only the text the iPhone read is sent.
- **More personal tools:** contacts and messages, next to the reminders, calendar and timer tools
- **iOS 26 `SpeechAnalyzer`** for faster, more accurate on-device recognition
- **Shortcuts, the Action button, or a widget** to jump straight into voice mode
