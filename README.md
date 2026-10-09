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
  engine › Benchmark** measures it on your phone.
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
  which the on-device Qwen3-ASR model (MLX) needs to build
- An iPhone on **iOS 17 or later**. A free Apple ID is enough to run it on your own phone.
- An **Anthropic API key** from [console.anthropic.com](https://console.anthropic.com), or a key for an
  OpenAI-compatible service

## Run it on your iPhone

1. Clone the repo and open `AlvinAssistant.xcodeproj` in Xcode.
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
- **On-device replies** download 1.5–2.7 GB the first time (Wi-Fi only) and run only while the app is open. Run
  the benchmark from a **Release** build and copy the results. Qwen3-ASR listening loads after the reply model
  and gives way to it if memory is short.
- **Qwen3-ASR listening** downloads about 1 GB the first time (use Wi-Fi), needs a recent iPhone (iPhone 15 Pro
  or newer is best), and works only while the app is open; with the screen locked, turns use Apple's text.
  Build with the **Release** configuration for realistic speed.

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

The code is split in three:

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
- **`AlvinAssistant/`** is the iOS app:
  - SwiftUI views
  - SwiftData storage
  - the audio engine, speech recognition, and speech playback
  - `VoiceSession`, the state machine that ties them together

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
```

`.github/workflows/ios.yml` runs these tests and builds the app with Xcode on every push.
`.github/workflows/local-engine.yml` tests the on-device engine on a Mac GPU, and `model-facts.yml` and
`drafter-lab.yml` measure the models; see
[the CI section of the architecture doc](docs/architecture-plan-on-device-and-orchestration.md#10-ci-workflows-and-how-to-read-them).

## Security note

This is set up as a personal app: your own API key is stored in the Keychain on your own phone. Don't ship a
build with a key inside to other people. For a public app, put a small server in front of the model API and
have the app authenticate to that instead.

## Ideas for next steps

- **Speech-to-speech mode** (the OpenAI Realtime API or Doubao's realtime voice model) for the lowest possible
  latency and more expressive voices
- **Camera mode:** show the assistant what you're looking at (Claude supports images)
- **More personal tools:** contacts and messages, next to the reminders, calendar and timer tools
- **iOS 26 `SpeechAnalyzer`** for faster, more accurate on-device recognition
- **Shortcuts, the Action button, or a widget** to jump straight into voice mode
