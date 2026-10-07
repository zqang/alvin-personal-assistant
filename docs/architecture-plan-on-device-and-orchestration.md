# Plan: on-device replies and multi-agent orchestration

Status: Phases 0 and 1 are built (on-device provider and benchmark; see `AlvinAssistant/OnDevice/`). The rest is a proposal.

This plan borrows two ideas:

- **Underdog / Husky** (Conway Research): a small local model with an inference engine tuned for that one
  model, plus small specialist models run by a harness loop.
- **Meta Muse Spark**: parallel agents that propose, refine, and merge an answer, and sub-agents a request can
  be handed off to.

It maps both onto this app's existing code.

## 1. What the reference systems do

All of this is from public reporting. Conway's and Meta's own pages could not be read from the build
environment, so check the details before relying on them.

### Underdog

| Piece | What it is | Can we use it? |
|---|---|---|
| **Woof 4B 1.1** | Apache-2.0 open weights based on Qwen3.5, 4-bit MLX, under 2.5 GB, tuned for browser use and tool calls. A 2B version also exists. | **Yes.** It loads with mlx-swift-lm, which the app already pulls in. |
| **Woof selector + text filler** | A 4B selector picks the next action and its target. A 4B or 0.8B text filler writes the field value as JSON. A harness runs the action and loops. | **The pattern, yes.** It is a good model for local tool use. |
| **Husky** | An inference engine built for one model, Woof. It uses Metal kernels shaped for that model and verifies blocks of 8 tokens per step. "Flash" mode adds a draft model trained on Woof for speculative decoding. Conway reports 1.02–3.9× MLX's speed without Flash and up to 4.5× with it (730 tok/s on an M5 Max). | **Not now.** It is benchmarked on Mac, it only runs Woof, and I found no Swift or iOS SDK. Its license and source availability are unconfirmed. Reported iPhone support comes from second-hand posts. Revisit when the Underdog iPhone app ships. |

### Muse Spark

- **Contemplating mode:** several agents reason in parallel, propose answers, refine them, and the results are
  merged. Meta says this improves quality "at comparable latency".
- **Sub-agent handoff:** in the 1.1 agent architecture, a request can be split into sub-tasks that are handed
  off to parallel sub-agents.
- **Availability:** closed weights. The Meta Model API is a US-only preview at $1.25 / $4.25 per million
  input / output tokens. **Don't add it as a provider** unless the API becomes available where the app is
  used. We can copy the orchestration pattern with Claude instead.

## 2. Target architecture

```
 Voice I/O (unchanged: AVAudioEngine, SFSpeechRecognizer, TurnDetector, BargeIn, Speaker)
                                   │ committed user turn
                                   ▼
                     ┌───────── Orchestrator (ChatProvider) ─────────┐
                     │  RouteDecider: offline? tool intent? depth?   │
                     └──┬──────────────┬──────────────┬──────────────┘
                        │ fast         │ tools         │ deep
                        ▼              ▼               ▼
              LocalLLMProvider     AgentLoop       Deliberation
              (MLX, Woof/Qwen,     (client tools:  (N parallel Claude
               on device)           EventKit,       workers + 1 merger,
                                    contacts,       Muse-style)
                                    timers)
                        └──────────────┴───────────────┘
                                       │ ReplyEvent stream
                                       ▼
                         SentenceChunker → TTS → speaker
                     (cloud default: ClaudeProvider, unchanged)
```

The key design choice: the **Orchestrator is itself a `ChatProvider`**. `VoiceSession`, `ChatController`, and
`SentenceChunker` don't change, because they still consume one `AsyncThrowingStream<ReplyEvent, Error>`
(`AssistantKit/Sources/AssistantKit/Conversation/ChatTypes.swift:42`).

### 2.1 Routing (`RouteDecider`, in AssistantKit and unit-tested)

Start with heuristics. A model is not needed on day one.

1. **No network** → local.
2. **Tool intent** ("remind me", "what's on my calendar", "set a timer") → AgentLoop.
3. **Needs fresh facts** (weather, news, prices) → Claude with `web_search`, as today.
4. **Explicit depth** ("think hard about…", a Settings toggle, or a long multi-part question) → Deliberation.
5. **Otherwise** → the user's chosen default: Claude today, or local once it is good enough.

Later, the local model can classify the route itself. This is the Woof selector idea: a constrained JSON
output such as `{"route":"local|tools|cloud|deep"}`.

### 2.2 Local replies (`LocalLLMProvider`)

- Lives in the **app target**, next to `AlvinAssistant/Voice/QwenListener.swift`, because the MLX packages are
  linked there. It uses `MLXLLM` / `MLXLMCommon` from mlx-swift-lm, which already resolves transitively.
- Candidate models, to be chosen by measurement:
  - Woof 2B or 4B (`ConwayResearch/Underdog-Woof-*`)
  - a plain Qwen3.5 2B/4B instruct model, in case Woof's browser-use tuning hurts casual conversation
- Download, caching, and memory checks reuse QwenListener's approach:
  - a HubCache in Application Support that is excluded from backup
  - Wi-Fi only
  - an `os_proc_available_memory()` gate
  - unload when a memory warning arrives or the app goes to the background
- **Husky's speed techniques, applied with what we have ("Husky-lite").** The mlx-swift-lm version we already
  resolve (3.31.4) provides the following:
  - `ChatSession` keeps the KV cache between turns and can `saveCache(to:)`. Our history is append-only, so
    each turn only prefills the new turn.
  - `SpeculativeDecodingConfig` runs a draft model with a memory policy (`.recommendedWorkingSet`). This is the
    same verify-a-block-per-step idea as Husky. **It does not work for Woof or Qwen3.5.** They are hybrid
    models (Gated DeltaNet layers), and their recurrent state can't be rolled back after a rejected draft. The
    library refuses with "Speculative decoding requires trimmable KV caches." Making it work means snapshotting
    and replaying that state, which is Husky-level engine work. The app therefore offers speculative decoding
    on Qwen3 4B, which uses plain attention, with Qwen3 0.6B as the draft.
  - `ChatSession.tools` / `toolDispatch` handle local tool calls.

  The library does not have **prompt-lookup drafting** (proposing tokens copied from the prompt), which is
  where Husky reports its biggest gains. We can add it on top of the same verification step; it pays off for
  tool-call JSON, not for casual chat.

  Skip the rest of Husky:
  - **Model-specific Metal kernels:** specialist work, with gains measured only on Mac.
  - **Training our own "Flash" drafter:** needs GPU training plus a new drafter model type.

  Note that a voice reply is limited by time to first token, not tokens/s, because speech consumes only a few
  tokens per second. Measure both in Phase 0.

### 2.3 One GPU and memory owner (`OnDeviceModels`)

Today `QwenListener` has the only GPU guard (`gpuAllowed` and `inFlight`). Two models need one coordinator:

- **A single serial GPU queue and one background/foreground guard** shared by ASR and the LLM.
- **A memory budget:** if both models don't fit, keep the LLM loaded and give up the Qwen3-ASR refinement
  pass, which already falls back to Apple's transcript.
- **Order of work:** the ASR refinement (≤2 s) finishes before LLM generation starts.

### 2.4 Client tools (`AgentLoop`, in AssistantKit)

This is required before any local or cloud tool use beyond `web_search`:

- **`ChatTurn`:** it is text-only today. Add optional tool-call and tool-result content.
- **`ReplyEvent`:** add `.toolCall(id:name:input:)`. Keep `.activity` for spoken status such as
  "Checking your calendar".
- **`Tool` protocol:** `name`, a JSON-schema `inputSchema`, and `run(input) async throws -> String`. EventKit,
  Contacts, and timer tools live in the app and are registered with the loop.
- **`ClaudeProvider`:** send client tools, parse `tool_use`, and loop with `tool_result`. Model the loop on
  the existing `pause_turn` continuation (`ClaudeProvider.swift:65-75`).
- **The local model** uses the same loop through Qwen's tool-call template, which Woof was trained for.

### 2.5 Deep mode (`Deliberation`, Muse-style)

- Fan out N (2–3) parallel `ClaudeProvider` calls with different angles. For example: one with `web_search`,
  one reasoning at higher `effort`, and one acting as a critic.
- Then one merger call streams the final spoken answer.
- **Voice latency:** while the workers run, emit `.activity("Thinking it through…")` and an optional short
  spoken acknowledgement from the local model. That way the user hears something within about 1 s.
- **Cost guard:** only use deep mode when asked for, or when the RouteDecider is confident it's needed.
  Haiku/Sonnet workers with an Opus merger keep the cost down.

## 3. Phases

| Phase | Deliverable | Exit criteria |
|---|---|---|
| 0. Spike (built: Settings › Benchmark) | Load Woof 4B and a Qwen3.5 instruct model with MLXLLM on an iPhone 15 Pro and 17 Pro; script about 30 voice-style prompts in English and Chinese | Time to first token, tokens/s, peak memory with ASR loaded, and a quality rating recorded in this doc |
| 1. Local provider (built) | `Provider.onDevice`, `LocalProvider` + `LocalModelHost`, GPU and memory hand-off with Qwen3-ASR, Settings UI and download flow | Offline voice turns work end to end; unit tests for the new settings decoding |
| 2. Orchestrator | `Orchestrator` + heuristic `RouteDecider` in AssistantKit, with tests; offline fallback to local | Routing table tested; no regression in Claude latency |
| 3. Tools | `ChatTurn`/`ReplyEvent` tool support, `AgentLoop`, Claude client tools, EventKit reminders/calendar | "Remind me at 5 to call mum" works by voice, through both Claude and local |
| 4. Deep mode | `Deliberation` with parallel workers and a merger | Better answers on a hand-picked hard set, with the first audio within about 1.5 s |
| 5. Revisit Husky | Check whether Husky ships for iOS with a usable license | Go/no-go on swapping the Woof runtime |

## 4. Risks

- **Memory:** iOS limits how much memory one app can use. The ASR model and a 4B LLM together may not fit
  without the increased-memory entitlement.
- **Quality:** a 2–4B model will be clearly weaker than Claude at open conversation. Keep Claude as the
  default and local as the fallback until Phase 0 says otherwise.
- **Heat and battery:** long local generations on the phone get hot. Keep local replies short (the voice
  prompt already asks for brevity).
- **Benchmarks:** Conway's numbers are its own and measured on Mac. Our Phase 0 measurements decide.
