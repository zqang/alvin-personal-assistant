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

1. Clone the repo, run `scripts/prepare_minicpm.sh ios` (requires CMake 3.28+), then open
   `AlvinAssistant.xcodeproj` in Xcode. This builds the native runtime for the experimental MiniCPM benchmark.
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

## MiniCPM-V experiment

**Settings › On-device engine › Advanced settings › MiniCPM-V benchmark** runs MiniCPM-V 4.7 1B
locally with its Q4_K_M language model and F16 vision projector. The first run downloads about 1.8 GB
over Wi-Fi, verifies both files against pinned SHA-256 hashes, and keeps them for later runs. It measures
either the eight English/Chinese benchmark prompts or three repetitions of a fixed image-reading task.
The results include the generated answers, load time, time to first text, prefill time including vision,
decode throughput, thermal state, and sampled process memory. Copy or share the report from the screen.
Use **Delete MiniCPM download** on that screen to reclaim its storage.

This uses a separate, pinned llama.cpp-omni runtime. Your selected chat model stays the same. Existing
MLX models are unloaded before each benchmark, and MiniCPM is unloaded afterwards. Each prompt starts
fresh, with thinking off, greedy decoding, one image overview slice, and a maximum of 128 output tokens.
The first prompt includes cold kernel work. These timings are not directly equivalent to the normal
benchmark's continued conversation, system prompt, tool loop, or MLX memory accounting.
GPU layer checkpoints allow cancellation; their synchronization overhead is included in the results.

Use a Release build on a real iPhone and keep the app open. Leaving the app or tapping Stop cancels
the run. The official demo recommends at least 4 GB device RAM; this benchmark also checks available
app memory before loading. The simulator is not offered as a performance measurement device.

For a connected development iPhone, launch the installed app with `--minicpm-benchmark all` to run
the text and image scenarios consecutively (`text` or `image` runs just that scenario):

```bash
xcrun devicectl device process launch --device YOUR_DEVICE_ID --terminate-existing --console \
  com.alvinang.personalassistant --minicpm-benchmark all
```

The app starts once it is active and saves each completed prompt. It writes raw timings, answers,
device details, and completion status to `Documents/MiniCPMBenchmark/latest.json`, with JSON and
text reports retained in a separate folder for each run. To retrieve them:

```bash
xcrun devicectl device copy from --device YOUR_DEVICE_ID \
  --domain-type appDataContainer --domain-identifier com.alvinang.personalassistant \
  --source Documents/MiniCPMBenchmark --destination build/minicpm-results/iphone
```

Use your app's bundle identifier if you changed it for signing. A cancelled or failed run retains
its partial results and does not automatically restart when the app returns to the foreground.

To compare MiniCPM against the model currently selected in the app, launch with
`--compare-local-models` instead. Both model downloads must already be present. This runs the
current model, MiniCPM, MiniCPM, then the current model, with an excluded warm-up and eight
independent English/Chinese performance prompts per block. It also scores 24 fixed bilingual
tasks once per model. Answers must satisfy both the content and the requested output format;
this small diagnostic is not a standardized accuracy benchmark.

The comparison uses greedy decoding, thinking off, a 128-token output limit, and no tools,
speculation, or conversation/prefix reuse. It therefore measures fresh model requests rather
than the full voice/tool assistant. It records wall-clock response time and visible characters
per second alongside each backend's differently defined token metrics. Reports are saved under
`Documents/ModelComparison/`, including `latest.json`, and can be copied with the same device
command above using that source directory. Keep Low Power Mode off and the app in the foreground.

Runtime source and model revisions are pinned in `scripts/prepare_minicpm.sh` and
`AssistantKit/Sources/AssistantKit/OnDevice/MiniCPMArtifacts.swift`. Generated native binaries and
model files are not committed. See [the official mobile implementation](https://github.com/OpenBMB/MiniCPM-V-Apps)
for the runtime and model provenance. Build all Apple Silicon slices with `scripts/prepare_minicpm.sh`
when running the package's macOS command-line benchmark as well as the app.

On an Apple Silicon Mac, after preparing all slices:

```bash
python3 scripts/download_minicpm.py
swift run --package-path MiniCPMBenchmark -c release minicpm-bench \
  build/minicpm-models/MiniCPM-V-4_7-1B-Q4_K_M.gguf \
  build/minicpm-models/mmproj-model-f16.gguf
```

Add an image path after the two model paths to exercise vision. `--prompt "..."` and
`--max-tokens 128` override the input and limit. The CLI writes a JSON report to stdout and
native runtime logs to stderr. Mac measurements do not predict iPhone speed.

### MiniCPM-V 35B-A3B with LES and DSpark

The 35B comparison runs the same 24 scored bilingual tasks and eight performance prompts against
the model selected in Settings. It uses the published `mix_q3gud_im` LES split and a Q8 DSpark
draft converted from the official checkpoint. The complete file set occupies about **17.1 GB**,
in addition to the current model. Allow extra Mac storage for the source checkpoint and conversion.
Preparation needs an Apple Silicon Mac, the Xcode setup above, **CMake 3.28+**, and **Python 3.11+**.

From the repository root, build the native runtime and download the pinned target files:

```bash
scripts/prepare_minicpm.sh all
python3 scripts/download_minicpm_35b.py --target-only
python3 -m venv build/minicpm-dspark-venv
build/minicpm-dspark-venv/bin/python -m pip install numpy==2.4.4 tqdm==4.67.3 pyyaml==6.0.3

mkdir -p build/minicpm-dspark-source
MINICPM_DSPARK_SOURCE='https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-DSpark/resolve/5facb75c251ffd524524a544c1f5fd97ba0d6c5c'
for file in config.json model.safetensors; do
  curl --fail --location --retry 3 "$MINICPM_DSPARK_SOURCE/$file" \
    -o "build/minicpm-dspark-source/$file"
done

"${CMAKE:-cmake}" --build build/minicpm/macos-arm64 --config Release --target llama-quantize
build/minicpm-dspark-venv/bin/python scripts/convert_minicpm_dspark.py
python3 scripts/download_minicpm_35b.py --verify-only
```

Run `export CMAKE=/absolute/path/to/cmake` first if CMake is not on your PATH. The converter verifies the
checkpoint, configuration, and paired target hashes, copies the target tokenizer, and maps all
62 DSpark tensors to the pinned runtime's DFlash/Markov implementation. It preserves normalization
and bias vectors in F32 and uses the runtime's Q8 quantizer for matrices; the single-row confidence
head remains F16. Its `.provenance.json` sidecar records source hashes, revisions, and conversion
settings. The app and downloader share `MiniCPM35BManifest.json`, including the exact converted
draft hash. A missing or mismatched draft stops the run instead of silently disabling DSpark.

Build and install the **Release** app with your signing team, select and download Woof in Settings,
then copy the prepared files to the connected development iPhone. Replace `YOUR_DEVICE_ID` and,
if necessary, the bundle identifier:

```bash
xcrun devicectl device copy to --device YOUR_DEVICE_ID \
  --domain-type appDataContainer --domain-identifier com.alvinang.personalassistant \
  --source build/minicpm-35b-models \
  --destination 'Library/Application Support/MiniCPM35BBenchmark'
xcrun devicectl device process launch --device YOUR_DEVICE_ID --terminate-existing --console \
  com.alvinang.personalassistant --compare-local-models-35b
xcrun devicectl device copy from --device YOUR_DEVICE_ID \
  --domain-type appDataContainer --domain-identifier com.alvinang.personalassistant \
  --source Documents/ModelComparison35B --destination build/minicpm-results/iphone-35b
```

Wait for the benchmark to finish before copying its reports. Keep Low Power Mode off, leave the
phone unlocked, and keep Assistant in the foreground. `Documents/ModelComparison35B/latest.json`
records progress and retains completed prompts if the run is interrupted.

This suite measures **text only** and does not load the vision projector. LES retains the model's
top-8 expert routing, with a 256-slot shared bank, 32 cached experts per layer, and lookahead
prefetch depth 4. DSpark proposes up to three tokens per round; the target verifies the exact
greedy prefix, and reports include proposed and accepted draft counts. Thinking is off, the output
limit is 128 tokens, and tools and prompt reuse are disabled. Woof uses the comparison's existing
non-speculative path. These settings and the locally converted draft must be reported with any
measurements; results from the upstream demo are not measurements of this integration.

Runtime tuning keeps that original configuration available as `baseline`. Add
`--minicpm-profile balanced` to the 35B comparison launch to use 48 cached experts per layer,
four-layer cancellation checkpoints, and allocation-free target argmax sampling. `checkpoints4`
keeps the original 32-expert cache; `draft2` uses the balanced settings with two draft tokens.
The weights, quantization, top-8 routing, and exact target verification are unchanged. Exported
rows record the actual runtime configuration. Larger caches consume more RAM and must be
measured on the target phone; a profile name does not imply that it is faster.

For a short profile-selection run, append
`--minicpm-pilot baseline,checkpoints4,balanced,draft2,baseline` instead. This reloads each profile,
warms it up, and runs frozen performance prompts 1, 4, and 5 at the same 128-token limit.
Pilot reports are separate under `Documents/MiniCPM35BProfilePilot`; they do not replace the
full 80-row comparison. All blocks now reclaim model memory before loading and wait up to
120 seconds for nominal thermal state. Preparation time and thermal state at load are recorded
outside the measured load/reply times. Apply this policy equally to Woof and MiniCPM, and account
for it when comparing with older runs that did not cool between blocks.

`io1` and `io8` vary storage-reader threads from the balanced profile's four; `threads4` varies
CPU threads from six to four. Storage-thread trials require separate app launches because the
upstream runtime reads this setting once per process. Mixing different I/O-thread profiles in
one pilot fails explicitly. These candidates also retain the same weights and exact verification.

The pinned runtime has a local `minicpm-metal-worker-autorelease-v1` patch that adds autorelease
pools to Metal encoding and residency workers. The Swift bridge also drains temporary native
objects at each call boundary. This prevents temporary worker objects from delaying model memory
release across benchmark blocks. The preparation script applies the patch reproducibly and records
its hash in the framework provenance; comparison exports include the patch ID and hash. The patch
does not change model operations, weights, or precision.

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
