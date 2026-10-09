# On-device engine, routing, tools and deep mode

How AlvinAssistant answers on the iPhone with its own inference engine (the "Alvin engine"), routes each
request between Claude and the phone, calls device tools, thinks harder on request, and answers voice turns
sooner. This document started as a plan; it now describes what is built and what is left.

**Status (2026-10-09, integration commit `ab1f341`).** Every phase below is built and merged, except the MTP
drafter (skipped), the device validation and the deferred items. The pure logic is unit-tested in
AssistantKit, on macOS and on Linux. The engine is tested on the Metal GPU of a GitHub macOS runner, with tiny
random models and with real checkpoints up to Underdog Woof 4B. What only an iPhone can settle is in the
[device checklist](#11-device-checklist). Facts measured in CI are in [Model facts](model-facts.md) and
[Drafter lab](drafter-lab.md).

The defaults keep today's behaviour where a device has not yet proved the new path:

- routing stays **single** (the chosen provider answers everything, as before) until the latency report says
  otherwise;
- the engine mode is **automatic**: the Alvin engine answers only after its on-device self-test has passed
  for this model and build, and MLX's stock `ChatSession` answers until then;
- the fast GPU kernels are **off**.

## Contents

1. [Phases](#1-phases)
2. [Reference systems](#2-reference-systems)
3. [Architecture](#3-architecture)
4. [On-device engine](#4-on-device-engine)
5. [Routing](#5-routing)
6. [Tools](#6-tools)
7. [Deep mode](#7-deep-mode)
8. [Voice latency](#8-voice-latency)
9. [Settings](#9-settings)
10. [CI workflows and how to read them](#10-ci-workflows-and-how-to-read-them)
11. [Device checklist](#11-device-checklist)
12. [Skipped and deferred](#12-skipped-and-deferred)
13. [Risks](#13-risks)

## 1. Phases

The work packages (WPs) ran in waves; packages in one wave owned disjoint files. "Verified in CI" names the
GitHub Actions runs whose logs hold the numbers quoted in this document.

| Phase | Deliverable | WPs | Status |
|---|---|---|---|
| 0. Spike | Settings › Benchmark: time to first token, tokens/s and memory on a phone | — | Built earlier; now the scenario benchmark of phase 9 |
| 1. Local provider | `Provider.onDevice`, `LocalProvider` and `LocalModelHost`, the GPU and memory hand-off with Qwen3-ASR, download flow | — | Built earlier |
| 2. Contracts, CI lab, engine skeleton | `AssistantEvent`, tool and route types, `CommitGate`, the new settings, the `ReplyPipeline` seam; the model-facts and drafter-lab workflows; the `LocalEngine` package with a GPU canary | WP00, WP01, WP02 | Built. Verified in CI on `ca75f08`: iOS 37705247621, Local engine 37705247676, Model facts 37705247655, Drafter lab 37705247648 |
| 3. Engine core | session planning (AssistantKit); the `HybridQwen35` fork with exact rollback; the engine runtime (session reuse, checkpoints, pipelined decode, prefix on disk) | WP10, WP16, WP20 | Built. Verified in CI: Local engine 37861143970 (`c16ec26`) |
| 4. Speculation and self-test | drafters, draft policy, acceptance rule (AssistantKit); speculative loop, cost probe, self-test (LocalEngine) | WP11, WP30 | Built. Verified in CI: Local engine 37867824542 (`51b0eaf`); Woof 4B in 37875179123 (`d3779c3`) |
| 5. Tools | tool runtime; Claude client tools and agent loop; reminders, calendar and timer tools; the on-device tool loop | WP12, WP13, WP22, WP31 | Built. Woof 4B tool calls verified in CI (37875179123); permission flows need a device (D9) |
| 6. Routing | intent classifier, router, orchestrator with fallbacks, deep-mode budget, latency estimator, connection prewarm | WP14, WP50 | Built. `routingMode` stays `.single` by default until D13 |
| 7. Deep mode | one high-effort call, or a researcher, a reasoner and a critic plus a merger | WP21, WP50 | Built |
| 8. Voice latency | early reply start, spoken cues, ducking, latency traces | WP15, WP23 | Built. Tuning needs a device (D7, D8) |
| 9. App integration | the engine in the app with its self-test gate and fallback, the engine settings, benchmark scenarios, the orchestrated reply pipeline | WP40, WP50 | Built |
| 10. Kernels and MTP | small-M quantized matmul kernel; MTP drafter | WP41, WP42 | Kernel built and **off by default** (slower than stock on the CI GPU). MTP **skipped**: no catalog checkpoint keeps `mtp.*` weights |
| 11. Device validation | the [device checklist](#11-device-checklist) | — | Not started; needs an iPhone |
| 12. Later | Swift DFlash port, mlx-swift-lm 3.32.x upgrade, background session save, two-step tool selector | — | Deferred; see [§12](#12-skipped-and-deferred) |

## 2. Reference systems

Two public systems shaped this work. Their pages could not be read from the build environment, so the
details come from public reporting.

- **Underdog / Husky** (Conway Research). Woof 4B 1.1 is an Apache-2.0, Qwen3.5-based, 4-bit MLX model
  (2.39 GB) tuned for tool calls; Husky is an inference engine built for that one model, with model-shaped
  Metal kernels, block verification of 8 tokens per step and a trained draft model ("Flash"). Husky has no
  iOS SDK and no published license, so instead of waiting for it the app now has its own engine that applies
  the same techniques to Woof (table below). Woof 4B is the default on-device model. Woof 2B also exists,
  but it is a Llama model with a different tokenizer, not a Qwen3.5 hybrid ([Model facts](model-facts.md)).
- **Meta Muse Spark.** Parallel agents propose, refine and merge an answer. Its API is a US-only preview, so
  it is not a provider; the pattern is copied with Claude as deep mode's parallel strategy
  ([§7](#7-deep-mode)).

What became of Husky's public claims:

| Husky feature | Status in the app |
|---|---|
| Conversation state stays resident | **Built.** Token-exact session reuse with rewind checkpoints (WP10, WP20). The stock path also no longer re-sends the system prompt every turn (WP40). |
| ~35 ms to the first word on a continued chat | **The techniques are built:** delta-only prefill with logits for the last row only, a resident engine queue, the system prefix kept on disk, prewarm when the mic or composer opens, and early reply start. On the CI GPU a continued Woof 4B turn reached its first token in 349–577 ms (789 ms for the first turn); phone numbers are D1. |
| Next read queued before this one ends | **Built.** Pipelined decode loop. |
| Block verification ("eight tokens at once") | **Built**, lossless, with exact hybrid rollback (capture/replay). Up to 4 drafts per round on stock kernels; 8 only with the fast kernels on and a cost curve measured with them that allows it (D3). |
| Cheap wide verifies (small-M quantized matmul) | **Built, gated, off.** Correct, but 11–23 % slower than stock at 8 rows on the CI GPU. Enabled only after "Test fast kernels" measures a ≥ 25 % gain on the phone. |
| Big gains when the output repeats the prompt | **Built.** Prompt lookup, a cross-session suffix corpus and tool-call skeletons. |
| "Flash" trained drafter | **Partly.** A Qwen3-0.6B draft model for Qwen3 4B. MTP skipped (no weights). DFlash for Woof measured 5.0 tokens per round in the lab but was not faster or verified lossless in dflash-mlx, so its Swift port is a follow-up ([§12](#12-skipped-and-deferred)). |
| Engine built for Woof | **Built.** The `HybridQwen35` fork serves Woof 4B and Qwen3.5; other models run through the stock adapter. |
| Model-shaped kernels, weight repacking, compiled decode | **Skipped.** Specialist work with gains measured only on Mac. |
| "Same answers" | **Built, further:** exact rejection sampling at the app's T 0.7 / top-p 0.8 / top-k 20, proven statistically in CI. |
| Selector + text-filler harness | **Not needed.** Woof 4B called 12/12 tools correctly in the lab. Native tool calls with a plausibility guard instead. |
| iOS SDK, Underdog 27B, ternary models | **Skipped.** No SDK; not phone-sized. |

## 3. Architecture

```
 VoiceSession (voice)            ChatController (typed)
        │  ReplyRequest: conversation, pending user turn (early start), commit gate, voice?, deep?
        ▼
 OrchestratedReplyPipeline (app) ── signals ──► RouteDecider (AssistantKit) ──► RouteDecision
        │
        ▼
 Orchestrator (AssistantKit): yields .routed first; one fallback, only before the reply commits
   ├─ cloud: ClaudeProvider + ToolRunner(device tools)                  ┐
   ├─ deep:  Deliberation (one high-effort call, or 3 workers + merger) │  AssistantEvent stream:
   └─ local: LocalProvider → LocalModelHost                             │  .reply(ReplyEvent), .cue,
               ├─ Alvin engine: LocalToolLoop → InferenceEngine         │  .toolRound, .routed,
               └─ stock: MLX ChatSession over the same model (no tools) ┘  .progress
        │
        ▼
 VoiceSession: CuePolicy → CuePlayer; SentenceChunker → Speaker     ChatController: message, ToolRoundChip
```

- **Events.** Replies travel as `AssistantEvent`. `.reply` wraps the unchanged `ReplyEvent` (text, activity,
  finished); `.cue` is speech that is not part of the answer and is never stored; `.toolRound` is a finished
  round of client tools, which consumers persist; `.routed` names the engine (again after a fallback);
  `.progress` carries latency marks. The OpenAI-compatible provider still produces `ReplyEvent`s and is
  wrapped by `LegacyProviderAdapter`.
- **The seam.** `ReplyPipelines.make` builds the pipeline the chat and voice screens use. `AppSetup.install`
  sets it to `OrchestratedReplyPipeline` at launch; `LegacyReplyPipeline` keeps the old single-provider path.
- **Code layout.**
  - `AssistantKit/`: Foundation-only Swift, tested on macOS and Linux. Events, settings, prompt building,
    session planning and drafting logic for the engine, tools runtime, Claude client and agent loop, router,
    orchestrator, deep mode, voice latency logic.
  - `LocalEngine/`: the Alvin engine, a Swift package on mlx-swift-lm 3.31.4, mlx-swift 0.31.6,
    swift-transformers 1.3.4 and swift-huggingface 0.11.0 (exact pins), linked into the app.
  - `AlvinAssistant/`: the app. Device tools, the model host, settings screens, voice session, pipeline.

## 4. On-device engine

### 4.1 Components

| Part | Where | What it does |
|---|---|---|
| `ModelLoader` | `LocalEngine/Loading` | Loads a snapshot. Qwen3.5 (non-MoE) loads through a private `LLMModelFactory` whose type registry maps `qwen3_5` and `qwen3_5_text` to the fork, so the global registry is never touched. One loaded model serves the engine and the stock `ChatSession` fallback, so a fallback needs no reload. |
| `HybridQwen35` fork | `LocalEngine/Models` | An MIT-attributed copy of the 3.31.4 Qwen3.5 text model with the same module keys, plus row-selective logits and capture/replay of the recurrent (Gated DeltaNet) layers. |
| `HybridTarget`, `StockTarget` | `LocalEngine/Runtime` | One interface over the fork and over stock models (forward with selected logit rows, commit a verified round, reset, adopt a cache). |
| `SessionPlanner`, `TurnDelta`, `CheckpointPolicy`, `PrefixKey` | `AssistantKit/Engine/Session` | Pure planning of what the cache can keep and what must be fed. |
| `LiveSession`, `CheckpointStore`, `SessionRuntime`, `TurnRenderer`, `Prefill` | `LocalEngine/Runtime` | The token ledger, rewind checkpoints, rendering of turn deltas, chunked prefill. |
| `FastSampler`, `DecodeLoop`, `TextStreamer` | `LocalEngine/Runtime` | Sampling, pipelined decode, tool-call and thinking filtering of the visible text. |
| `PrefixCacheStore` | `LocalEngine/Runtime` | The system prefix on disk. |
| `InferenceEngine` | `LocalEngine/Runtime` | The public API: load, warm up, prewarm, reply, continue after tools, invalidate. All MLX work runs on its serial queue. |
| Drafting core | `AssistantKit/Engine/Drafting` | `NGramIndex`, `SuffixCorpus`, `ExemplarSeeds`, `CostCurve`, `DraftPolicy`, `AcceptanceRule`. |
| `SpeculativeLoop`, drafters, `CostProbe`, `EngineSelfTest` | `LocalEngine/Speculation`, `LocalEngine/Drafters` | Speculative decoding, the on-device cost curve, the self-test. |
| `LocalToolLoop` | `LocalEngine/Tools` | Native tool calls on device, with handoff to the cloud. |
| Small-M kernel | `LocalEngine/Kernels` | Optional `MLXFast.metalKernel` for 2–9-row 4-bit matmuls, behind a self-test and a measured gain. |
| `LocalModelHost`, `EngineSetup`, `EngineVerificationStore` | `AlvinAssistant/OnDevice` | Download, load, warm-up, self-test and cost probe at load, mode choice per reply, fallback to stock. |

### 4.2 Loading

1. Decode `config.json` as `BaseConfiguration` (JSON5).
2. Qwen3.5 dense models go through the fork's registry; everything else through `LLMModelFactory.shared`.
3. Wrap the loaded context in a `ModelContainer` for the stock path.
4. The stop set is the configured EOS ids ∪ the tokenizer's EOS ∪ `<|im_end|>` and `<|endoftext|>`. For
   Woof 4B that is `[248044, 248046]`. There are no stop strings: a text-level stop would leave tokens in the
   cache that the ledger does not know about.
5. Weights load on the CPU inside a detached task; `warmUp()` then runs small GPU forwards in the foreground.

### 4.3 The `HybridQwen35` fork

- **Sanitize** uses the 3.32.3 rule: norm weights shift only for an unsanitized `conv1d`. It drops `mtp.*`
  and vision weights. (The stock 3.31.4 rule shifts whenever any `mtp.*` key exists; no catalog checkpoint
  keeps one, so today's stock path is not affected.)
- **`engineForward(rows:)`**: `.last` slices the hidden state to the last row *before* the head, `.none`
  skips the head, `.all` returns every row; `wantHidden` returns the post-norm hidden states.
- **Capture.** During a verify forward each recurrent layer records exactly what it passed to
  `gatedDeltaUpdate` (conv input, q, k, v, a, b, `A_log`, `dt_bias`) and its state before the call. These
  are references, not copies.
- **Commit (keep m of S rows).** Attention caches trim S − m tokens. Each recurrent layer takes the conv
  window ending at row m and replays `gatedDeltaUpdate` over the first m rows from the saved state, which is
  bit-identical to the verify pass's state after m steps. Rollback costs about one small kernel call, not a
  forward pass.

### 4.4 Session reuse

- **Ledger.** `LiveSession` owns the cache and the exact list of tokens in it. Invariants: the cache holds
  exactly the ledger (plus a pending token inside a pipelined step); every forward appends; every commit or
  rewind truncates.
- **History is kept as generated.** Past replies stay in the cache exactly as the model produced them,
  including the empty `<think>\n\n</think>\n\n` of the generation prompt, which a fresh template render would
  drop. Only the new delta is fed. Whether this costs answer quality is D6.
- **Deltas.** A continuation is rendered with sentinel turns and cut at the first `<|im_end|>` after the
  sentinel reply, so its tokens equal that segment of a real render (checked exact for every catalog
  template in [Model facts](model-facts.md)). Before appending, the longest overlap (up to 4 tokens) with the
  ledger's tail is dropped, which handles "EOS fed or not" uniformly.
- **Checkpoints** (hybrid models only; pure-attention models just trim): `systemEnd` (always kept),
  `lastUserStart` and `replyStart`. Each is a reference snapshot of the recurrent slots, about 49 MiB for
  Woof 4B, under a 160 MiB budget; `replyStart` is dropped first. Rewinding to any position restores the
  deepest checkpoint at or below it and re-feeds the gap cache-only.
- **Planner.** `SessionPlanner.plan` is pure; the first matching rule wins:

| # | Condition | Keeps | Feeds | Reason |
|---|---|---|---|---|
| 0 | The request does not end with a user turn | — | nothing (no plan) | — |
| 1 | No live session | the prefix on disk if present, else nothing | system prefix (when empty) + the window | `newSession` |
| 2 | The prefix key changed (model, revision, system text, tools, chat context, format version) | as rule 1 | as rule 1 | `prefixChanged` |
| 3 | Every cached turn matches and the request adds turns | the whole ledger | the new turns | `append` (rule 7 if over budget) |
| 4 | Only the last cached reply differs (interrupted or edited), without tool rounds | up to `replyStart` | the new reply text + the rest | `replaceLastReply` |
| 5 | Only the last cached user turn differs (a tentative turn replaced) | up to `lastUserStart` | the new user turn | `replaceLastUserTurn` |
| 6 | Any other divergence, including another conversation | up to `systemEnd` | the window | `diverged` |
| 7 | Over 6,144 tokens | as rule 6 | as rule 6 | `overBudget` |

  The window is the last 12 turns, starting on a user turn.

### 4.5 Prefill, sampling, decode, streaming

- **Prefill** in chunks of up to 512 tokens, split at checkpoint positions; only the final chunk computes
  logits, for its last row. `StockTarget` (no row selection) prefills all but the last 16 tokens cache-only.
- **`FastSampler`**: float32 log-softmax, top-k 20 by partition, then the nucleus (top-p 0.8) over those
  20, then temperature 0.7. It keeps the same token set as MLX's stock filter chain without sorting the
  whole 248k vocabulary each token. Temperature 0 is argmax. It also returns the top-1 probability for
  confidence telemetry.
- **`DecodeLoop`** (the default generator) is pipelined: the next token is sampled lazily while the
  previous one syncs. It stops on a stop token (already fed, so the ledger stays exact), the length limit,
  cancellation, or the app revoking the GPU.
- **`TextStreamer`**: streaming detokenizer → `ToolCallProcessor` in the model's format (xml-function for
  Qwen3.5 and Woof, JSON for Qwen3) → thinking filter → visible text. It reports the tool-call start as soon
  as `<tool_call>` is sampled, and the name once parsed, which lets the tool loop abort early for a handoff.

### 4.6 Speculative decoding

- **Round.** Input `[y] + K drafts`; one forward over K + 1 rows with capture; one sampler call; **one** host
  sync; `AcceptanceRule` keeps the leading drafts that equal the target's own samples and adds the target's
  next sample; commit (lazy). With deterministic drafts this is exact rejection sampling at any temperature,
  so speculation never changes the output distribution.
- **Drafters**, asked in this order: tool-call skeletons while inside a tool call; prompt lookup (n-grams
  2–4 over the whole ledger: system, tools, history, tool results, the current reply); a suffix corpus of
  past replies and tool results (64k tokens, persisted per model in `Application Support/EngineCache`); the
  Qwen3-0.6B draft model for Qwen3 4B (when "Draft model" is on). Special tokens are never drafted outside
  the skeletons.
- **`DraftPolicy`** picks K to maximize expected tokens per unit of cost, `E(K) / (c(K+1) + K·draftCost +
  0.04)`, with acceptance rates kept per source, match length and depth. It speculates only when that beats
  plain decoding by 5 %. Prompt-lookup matches shorter than 3 tokens never speculate (the lab measured 43 %
  first-draft acceptance for length 2, about 90 % for 3 and more). After 3 rounds without an accepted draft a
  source backs off for 16 plain tokens, doubling up to 256. A `serious` thermal state triples the margin;
  `critical` or Low Power Mode turns speculation off. At most 4 drafts per round; 8 only while the fast
  kernels are on and the curve measured with them gives c(9)/c(1) ≤ 2. The decision never uses wall time.
- **Cost curve.** The stock default is `{1: 1.0, 2: 1.05, 3: 1.35, 4: 1.63, 5: 1.95, 6: 2.3, 8: 3.07,
  9: 3.4}`. `CostProbe` measures the phone's own curve (S = 1, 2, 3, 4, 6, 8 on a scratch rewind of the live
  session, about a second) once per model, device, OS major version and engine format version.
- **Modes** (`localSpeculation`): `off`; `toolsOnly` (inside tool calls, or prompt-lookup matches of 3+
  tokens); `automatic` (the default, gated by the policy above).

### 4.7 Prefix on disk and prewarm

- After the first reply of a session, off the critical path and only while the app is in the foreground, the
  system prefix (tools + system prompt, about 1,650 tokens for Woof with the device tools) is saved with
  `savePromptCache` to `Application Support/EngineCache/<prefixKey>.safetensors`, excluded from backup,
  written to a temporary file and renamed. At most 3 files are kept; files of other models are deleted. A load
  checks the layer count, the cache classes, the key and a hash of the prefix tokens.
- **Prewarm** makes `systemEnd` resident: live, else loaded from disk, else prefilled and saved. It runs when
  voice listening starts, on the first words of each voice turn, and when the chat composer gets focus, for
  the on-device provider or for automatic routing with a downloaded model. CI showed why it matters: Woof
  tool requests that each started a fresh session, with about 1,650 tokens of tool schemas to prefill, took
  4.9–18 s to their first token.

### 4.8 Tools on device

- `LocalToolLoop` runs the engine, executes the model's tool calls with `ToolRunner` in lenient mode (Woof
  writes numbers and booleans as strings: `{"seconds": "600"}`), feeds the results back with
  `continueReply(after:)`, and repeats, for at most 3 rounds.
- **`handoff_to_cloud`** is intercepted, never run. Online, with nothing shown yet, generation stops as soon
  as the model has written the function name and the loop throws `ReplyHandoff`; the orchestrator then asks
  Claude ([§5](#5-routing)). Offline, the model gets a tool result saying the cloud is unreachable. After
  visible text it gets "Handoff unavailable now; finish your answer."
- **Plausibility guard.** In the lab, asked how long a timer had left, Woof started a one-second timer. A
  `set_timer` call is now refused only when the user stated no amount the guard can read (a digit in any
  script, an English number word, "a"/"an" before a unit of time, or a Chinese numeral before a unit of
  time) and either the words ask about a running timer ("how long is left", "还剩多久") or `seconds` is under
  5. A stated amount overrides the running-timer check ("how long is left on my 10 minute timer" runs), and
  every other call runs, so "pon un temporizador de diez minutos" with `seconds` 600 sets the timer though
  the guard cannot read Spanish. A refused call is handed off when it still can be, else answered with an
  error that tells the model to ask the user.

### 4.9 Self-test, fallback, telemetry

- **`EngineSelfTest`** (greedy, at most 48 tokens per prompt; 2.1 s on Qwen3.5-0.8B and 8.4 s on Woof 4B on
  the CI GPU):
  1. session reuse: a reused second turn's first-token logits match a fresh rebuild's (same argmax, or a
     near-tie with a top-2 margin under 0.5);
  2. speculation: 32 tokens plain versus forced K = 4 prompt lookup are equal up to the first near-tie;
  3. rollback: after a forced all-wrong round, the next logits equal those without the round.
- Results are stored per model id, snapshot, app build and engine format version. In `.automatic` mode the
  engine answers only if the stored result passed. When a load finds no stored result for this model and
  build, the host runs the test right after warm-up ("Checking the on-device engine…"); stock answers until
  it passes. With speculation on, the host also measures the cost curve once per model and device.
- **Fallback.** An engine error before the first token answers that reply once with the stock session and
  resets the engine's session. The stock path now passes the system prompt as history instead of as
  `ChatSession(instructions:)`, which re-sends it every turn (measured in CI on Qwen3-0.6B: 102 prompt tokens
  on turn 2 before, 17 after).
- **Telemetry** in `LocalGenerationStats`: phase times (plan, render, rewind, prefill, first token), prefilled
  and reused tokens, the plan reason, speculation (rounds, drafted and accepted per source, tokens per round),
  token confidence (mean and p10 of the top-1 probability), the engine name and peak memory.

### 4.10 Threading and the GPU guard

- All MLX work runs on one serial `DispatchQueue` owned by `InferenceEngine`; async APIs bridge to it.
- `EngineHooks` carry the app's existing guard: `beginGPU` (GPU allowed, then `inFlight.enter()`), `endGPU`,
  and `isAllowed`, checked every decode step and prefill chunk. On `willResignActive` the app stops new GPU
  work; on `didEnterBackground` it unloads the model and waits up to 3 s for work in flight, as before.
- `QwenListener.finishPasses()` is awaited before every engine GPU job, so the Qwen3-ASR pass and the reply
  model never share the GPU.

### 4.11 Measured in CI

All on the GitHub macOS runner's virtual GPU ("Apple Paravirtual device", 7 GiB). Speeds are not phone
numbers; exactness and acceptance do not depend on the device.

- **Fork parity:** fork versus stock on Qwen3.5-0.8B, 2B and Woof 4B: 64 greedy tokens identical on every
  prompt, first-step logits max |Δ| = 0.0.
- **Session reuse** (time to first token): Qwen3.5-0.8B 632 ms on turn 1 → 116–157 ms on turns 2–5; Woof 4B
  789 ms → 349–577 ms (17–28 tokens prefilled, 76–180 reused).
- **Throughput,** engine versus stock plain decode: Qwen3.5-0.8B 0.95×, Qwen3-0.6B 1.06×. Woof 4B decodes at
  about 19–20 tok/s there.
- **Lossless speculation** on tiny models, 5 drafters × K ∈ {1, 2, 4, 8}: greedy 800 positions and seeded
  T = 0.7 1,000 positions with 0 near-ties; first-3-token chi-square p = 0.907; worst relative rollback error
  2–3e-7 at every accepted length. The self-test catches a deliberately broken rollback.
- **Speed-ups:** Qwen3.5-0.8B "edit this paragraph" 1.32× with the automatic policy (1.44× forced K = 4);
  Qwen3-1.7B with the 0.6B draft 1.40× and 0.91× on two prompts. The lab projects Woof 4B at about 1.0× on
  chat, 1.46× on copy-heavy replies and 1.64× on tool calls ([Drafter lab](drafter-lab.md)).
- **Cost curve,** c(S)/c(1) for S = 2, 3, 4, 6, 8: Qwen3-1.7B 1.14, 1.43, 1.85, 2.77, 3.58; Qwen3.5-0.8B
  1.01, 1.41, 1.62, 2.24, 3.03.
- **Woof 4B tools** in the engine: `create_reminder` parsed, run and with the expected arguments 3/3
  (English and Chinese).
- **Small-M kernel:** correct (worst error 0.0075), but 8 rows cost more than stock (Woof 0.90×,
  Qwen3.5-0.8B 0.82×); only 2-row products were faster (1.09–1.41×).

## 5. Routing

`routingMode` is separate from the provider setting.

**Single** (`.single`, the default) keeps one provider, as before; only tools and deep mode are new:

- **Claude:** every reply goes to Claude, with the device tools, and deep when wanted (definition below) and
  budget is left. There is no on-device fallback.
- **On this iPhone:** every reply is local, with the on-device tools and no handoff. Tools need the Alvin
  engine; while MLX's stock session answers, there are none.
- **OpenAI-compatible:** unchanged, no tools.

**Automatic** (`.automatic`, with Claude as the provider) decides per request with `RouteDecider`, a pure
function of the utterance and live signals: connectivity, whether a key exists, whether the model is
downloaded and loaded ("local ready"), voice or typed, the deep request, the deep-mode setting and budget,
"Answer small talk on device", and recent times to first text. The intent classifier is rule-based (English
and Chinese phrase tables): *fresh facts* (news, weather, prices, scores, search), *device action*
(reminders, calendar, timers, alarms), *explicit depth* ("think hard about…"), *complex* (more than 45 words or
80 CJK characters, two or more questions, or a comparison or plan) and *small talk* (short greetings, thanks,
the time, a sum).

*Deep wanted* = online ∧ key ∧ deep mode ≠ off ∧ ("Think deeper" ∨ explicit depth ∨ (deep mode automatic ∧
complex ∧ typed)). The same definition applies with a single Claude provider. The first matching rule wins:

| # | Condition | Decision | Fallback |
|---|---|---|---|
| 1 | No cloud key | local, `noCloudKey` | — |
| 2 | Offline | local, `offline` | — |
| 3 | Deep wanted and budget left | cloud, deep | local if downloaded |
| 4 | Fresh facts | cloud | local if downloaded |
| 5 | Device action ∧ prefer on-device ∧ local ready | local, `deviceAction` | cloud, by handoff |
| 6 | Device action | cloud | local if downloaded |
| 7 | Voice ∧ small talk ∧ "Answer small talk on device" ∧ local ready ∧ not complex | local, `latencyFirst` | cloud |
| 8 | Not complex ∧ local ready ∧ Claude's recent time to first text > 3.5 s ∧ local not expected slower | local, `cloudSlow` | cloud |
| 9 | Complex | cloud | — |
| 10 | Prefer on-device ∧ local ready | local, `localFirst` | cloud |
| 11 | Otherwise | cloud, `cloudDefault` | local if downloaded |

Details that differ from a plain reading of the table:

- When deep is wanted but today's budget is used up, rules 4–11 pick a standard reply and the reason
  becomes `deepBudgetExhausted`. Voice never goes deep automatically.
- Rule 8 trusts Claude's estimate only while its newest sample is under 15 minutes old, so an old "slow"
  verdict lapses instead of keeping requests on the phone. Replies whose first text followed a tool call, a
  web search or a model load give no sample.
- A device action counts as "local ready" only when the Alvin engine will answer and the device tools are
  on: MLX's stock session has no tools.

**Orchestrator** (`Orchestrator`, AssistantKit):

- It yields `.routed` first. A reply is *committed* once it has shown text, finished a tool round or
  finished.
- Before commit, and only then, it falls back once: on a connectivity error from Claude (to local, reason
  `networkFallback`); on `ReplyHandoff` from the phone (to Claude, reason `escalated`, after the
  `.handingOff` cue "Let me check online."); or when Claude stays silent past the first-event watchdog (3 s
  for voice when the local model is loaded, 8 s otherwise; only when the fallback is local). Deep mode keeps
  its own deadline instead.
- A 401 never falls back, and nothing falls back after a tool round: that would run its tools twice.

**Tools are not a route.** The first version of this document sent "tool intent" to a separate agent loop. Now
every engine carries its own tools: Claude's agent loop runs the device tools, the on-device model calls
its subset natively, deep mode's merger has the full set. The router only picks the engine; the
`deviceAction` intent merely biases that choice (rules 5 and 6).

**Tool rounds are persisted.** Each finished round arrives as `.toolRound` and is stored on the reply
(`ChatMessage.toolRoundsData`, JSON of `[ToolRound]`), and `ToolRoundChip` shows one line per call, such as
"Reminder: Call mum · today 17:00". `PromptBuilder` keeps a reply in the history if it completed, was
interrupted, **or ran tool rounds**, so the next request replays them:

- to Claude, as `tool_use` / `tool_result` blocks (ids the local model generated are remapped
  deterministically to `toolu_…`, so the cached prefix stays byte-identical);
- to the engine, as an assistant turn with `tool_calls` plus `tool` messages, which the session planner
  compares like any other turn;
- to the OpenAI-compatible service, as the rounds' summary lines when the reply has no text.

A failed reply that ran tools stays as an interrupted reply without its text, and "Retry" is refused when a
round contained a side-effect tool, because the retried request would act a second time.

## 6. Tools

Schemas are shared by the lab (`scripts/lab_prompts.json`) and the app (`AlvinAssistant/Tools`). Every
object has `additionalProperties: false` and a `required` list; local times are ISO strings without an
offset, such as `2026-10-07T17:00`.

| Tool | Effect | Properties (required in **bold**) | On device |
|---|---|---|---|
| `create_reminder` | side effect | **title**, due, notes, list | yes |
| `list_reminders` | read | **scope** (today, upcoming, overdue, all) | yes |
| `complete_reminder` | side effect | **id** | no |
| `list_events` | read | **start**, **end** | yes |
| `create_event` | side effect | **title**, **start**, end, duration_minutes, location, notes, all_day | yes |
| `set_timer` | side effect | **seconds** (1–86,400, checked by the app), label | yes |
| `list_timers` | read | — | no |
| `cancel_timer` | side effect | **id** | no |
| `get_current_time` | read | — | yes |
| `handoff_to_cloud` | intercepted | **reason** | on device only, with automatic routing and a key |

- Tools are sorted by name and keep their definitions when turned off (a disabled tool answers "The user
  turned this off in Settings."), so the per-category switches (Reminders, Calendar, Timers) never change
  the tool list or the cached prompt prefix. Claude keeps every definition even with the master switch
  of Settings › Tools ("Reminders, calendar and timers", `deviceToolsEnabled`) off. The on-device prompt does
  not: it then drops its device tools and keeps only `handoff_to_cloud` (or nothing), so the engine's prefix
  key changes and its next reply starts a new session (`prefixChanged`, [§4.4](#44-session-reuse)).
- Read-only calls run concurrently; side-effect calls run one at a time in the model's order, each first
  waiting on the request's `CommitGate`, so a reply started early acts only once the user's turn is final.
  Each call has a 10 s timeout and a 2,000-character result.
- Read-only tools cue "Let me check."; side-effect tools give no cue; Claude's web search cues "Let me look
  that up."
- Claude: tools are sent with `strict: true` (and `eager_input_streaming` only for `api.anthropic.com`);
  `tool_choice` stays automatic; all results of a round go back in one message; at most 6 rounds per reply;
  `max_tokens` or a refusal never runs that turn's tools.
- Reminders and calendar use EventKit full access, asked only when a tool first runs after the commit gate
  opens; timers are local notifications that also show while the app is open.

## 7. Deep mode

- **Asking for it.** Unless deep mode is off: "Think deeper" in the chat composer or in voice mode, or an
  explicit phrase ("think hard about…"); with deep mode on **Automatic**, also typed requests that look
  complex. Voice never goes deep on its own. Deep mode is Claude-only, needs a connection and a key, and is
  limited per day.
- **Single strategy** (default, "One careful answer"): one same-model call with per-message effort `high`
  and the full tools. The cue "Let me think that through properly." and the activity "Thinking it through"
  come before any request is sent.
- **Parallel strategy** ("Three perspectives"): a researcher (effort `low`, read-only tools, its activity
  lines shown), a reasoner (`high`, no tools) and a critic (`medium`, no tools) run at once. Every worker
  declares the same tools so all requests share one cached prefix; side-effect tools never run in a worker.
  Their briefs go in a trailing mid-conversation system message where the model supports it, else in an
  `<instructions>` block. A streamed merger then answers with the full tools and the workers' notes as an
  `<analyst_notes>` block at the end of the user's last message. The briefs ask for conclusions, never a
  write-up of the reasoning. A refused worker is dropped; a truncated or timed-out one keeps its partial
  notes; with no notes at all, a plain call answers.
- **Limits.** A deadline of 25 s for voice and 60 s typed: workers still running are cut off, "Still
  thinking, almost there." plays at half the deadline if nothing has been said, and a call the user waits on
  that hears nothing from the server by then fails as a timeout, so the orchestrator can fall back to the
  phone. Merger output is capped at 4,000 tokens for voice and 16,000 typed; workers at 12,000. A daily limit
  (10 by default) counts deep replies; past it, replies are standard. A deep reply started early waits for
  the commit gate, so a discarded tentative turn never spends deep work or budget. A cheaper worker model can
  be chosen, at the cost of the shared cache.

## 8. Voice latency

- **Early start.** After 0.35 s of a stable transcript (0.25 s after a sentence end), with at least 2 words or
  3 CJK characters, a tentative reply starts. Its events are buffered silently: nothing is spoken, shown or
  done. When the turn commits (after the Qwen3-ASR pass when that is on), `TurnText.sameRequest` decides
  between adopting the stream (the commit gate opens and the buffered events replay) and cancelling it and
  starting fresh. A tuner moves the delay within 0.25–0.6 s from the discard rate of the last 20 turns. Early
  start is off for the local route while Qwen3-ASR listening is on (both need the GPU), for deep replies, and
  with the legacy pipeline; "On-device only" limits it to on-device replies, so it costs no extra cloud
  requests.
- **Cues.** At most one short spoken cue before the answer is heard ("Let me check.", "Let me look that up.",
  "One moment.", in English or Chinese); deep mode may add "Still thinking, almost there.". Never for typed
  input, never after any text. The filler "One moment." plays after `fillerDelay` (1.8 s), or at 0.6 s when
  the expected time to first text exceeds 2.5 s. Cues are pre-rendered to audio in the session's voice and
  are never stored in the conversation.
- **Ducking.** While the assistant speaks, at least 150 ms of input 18 dB over the noise floor (after echo
  cancellation) lowers playback to 0.3; it comes back after 600 ms without a confirmed interruption.
- **Traces.** Every voice turn records a `TurnLatencyTrace` (speech ended, early start, committed, routed,
  response started, first token, first text, first cue, first audio, finished or interrupted). The last 50
  are kept; Settings › Voice responsiveness › Latency report shows p50 and p90 per engine.
- **Connection prewarm.** When the user starts speaking or opens the composer, at most every 30 s, the app
  sends `GET {baseURL}/v1/models?limit=1` with the API headers through the same transport replies use, so
  the first request finds a warm TLS connection. The on-device model is prewarmed at the same moments
  ([§4.7](#47-prefix-on-disk-and-prewarm)).

## 9. Settings

| Setting | In Settings | Default | Notes |
|---|---|---|---|
| `localEngineMode` | On-device engine › Engine | Automatic | Automatic: the Alvin engine once its self-test passed, else MLX stock. Also "Alvin engine" and "MLX stock". |
| `localSpeculation` | On-device engine › Speculative decoding | Automatic | Off, Tool calls only, Automatic. |
| `localPrefixCache` | On-device engine › Keep the system prompt on disk | on | |
| `localSpeculativeDecoding` | On-device engine › Draft model | on | Shown for models with a draft model (Qwen3 4B). |
| `localMTP` | On-device engine › Multi-token prediction | on | Shown only for checkpoints with MTP weights; none today. |
| `localFastKernels` | On-device engine › Fast GPU kernels | off | Can be switched on once "Test fast kernels" measured a ≥ 25 % lower cost at 8 rows on this iPhone. |
| `routingMode` | Routing › Automatic routing | off (single) | |
| `preferOnDevice` | Routing › Prefer on-device | off | |
| `fastLocalSmallTalk` | Routing › Answer small talk on device | on | |
| `deviceToolsEnabled`, `disabledTools` | Tools | on; none disabled | Reminders, Calendar, Timers, each with its iOS permission state. |
| `deepMode` | Deep thinking | When I ask | Off, When I ask, Automatic. |
| `deepStrategy` | Deep thinking › Strategy | One careful answer | Or "Three perspectives" (about four requests). |
| `deepDailyLimit` | Deep thinking › Daily limit | 10 | "Used today" shows the count. |
| `deepWorkerModel` | Deep thinking › Worker model | same as the chat model | Parallel strategy only. |
| `earlyReplyStart` | Voice responsiveness › Start answering early | Automatic | Off, On-device only, Automatic. |
| `spokenCues` | Voice responsiveness › Spoken cues | on | |
| `fillerDelay` | Voice responsiveness › Say "One moment" after | 1.8 s | 1.0–4.0 s. |
| `turnChime` | Voice responsiveness › Chime when I finish speaking | off | |
| `bargeInDucking` | Voice responsiveness › Lower the voice when I talk over it | on | |

Routing and Deep thinking appear with Claude as the provider; On-device engine with "On this iPhone" or
automatic routing; Tools unless the provider is OpenAI-compatible. Settings saved by older builds load with
these defaults. The On-device engine section also shows the self-test result, the measured speed, the last
reply's drafting and the engine's session, with "Run self-test", "Measure speed", "Test fast kernels" and the
Benchmark (scenarios: continued chat, barge-in, cold start with and without the disk prefix, a 30-turn chat,
copy-heavy prompts, tool prompts; "Compare engines" plays one on the Alvin engine and then on MLX stock over
the same loaded model).

## 10. CI workflows and how to read them

| Workflow | Runs on | Trigger | Proves |
|---|---|---|---|
| `ios.yml` | macos-26; ubuntu-24.04 (`swift:6.2-noble`) | every push and pull request, except changes only to `docs/**`, `scripts/**`, `README.md` and the two lab workflows; by hand | AssistantKit `swift test`; the iOS app build (which compiles LocalEngine into the app); AssistantKit on Linux (informational, `continue-on-error`) |
| `local-engine.yml` | macos-26 (Metal GPU), 75 min | pushes touching `LocalEngine/**`, `AssistantKit/**` or the workflow; nightly at 04:41 UTC; by hand (input `woof`); `[ci woof]` in the head commit message adds Woof 4B | unit tests on tiny random models (parity, replay exactness, speculative = plain, session reuse = fresh prefill, sampler equivalence, prefix round trip, kernels) with Metal API validation on; an iOS compile of LocalEngine; integration tests on Qwen3-0.6B/1.7B and Qwen3.5-0.8B/2B (pinned in `LocalEngine/ci-models.txt`), plus Woof 4B when asked |
| `model-facts.yml` | ubuntu-24.04, 30 min | pushes touching `scripts/model_facts.py`, `scripts/lab_models.txt`, `scripts/lab_prompts.json` or the workflow; Mondays at 05:17 UTC; by hand | checkpoint layout, `mtp.*` presence, conv1d layout, EOS ids, quantization and chat-template behaviour, from the Hub without downloading weights |
| `drafter-lab.yml` | macos-26 (Metal GPU), 90 min | every push starts a small gate job; the lab runs only when its script, prompts, model list or workflow changed, when the head commit message contains `[ci lab]`, or by hand | Woof greedy outputs, drafting acceptance simulation, tool-call accuracy, template comparison, and DFlash acceptance (optional step) |

- A newer push to the same ref cancels an older `ios`, `Local engine` or `Model facts` run; drafter-lab runs
  queue instead. "By hand" (`workflow_dispatch`) works only once a workflow exists on the default branch.
  Docs-only commits do not run `ios.yml`; check the last code commit's run instead.

**Reading the results.** Every number a job measures is printed into its log between markers, because
artifacts cannot be downloaded from the sandbox the agents work in:

| Workflow | Step | Block | Contents |
|---|---|---|---|
| Local engine | Build for testing | `=== BEGIN PACKAGE RESOLVED ===` | the resolved package pins (`LocalEngine/Package.resolved` is copied from it) |
| Local engine | Prefetch models (retried), only when the model cache missed | `=== BEGIN MODEL SNAPSHOTS ===` | repo, snapshot SHA and size of each test model (the pins in `ci-models.txt`). The cache key is a hash of `ci-models.txt` plus "core" or "woof", so the step runs, and the block appears, only on the first core or Woof run after that file changes, or after the cache was evicted; most runs restore the cache and print no snapshot block |
| Local engine | Engine report | `=== BEGIN ENGINE REPORT ===` | the Markdown report the tests write: GPU device, parity, time to first token and reuse, throughput, speculation and cost curves, self-test, tool calls, kernel timings |
| Local engine | Failure summary (on failure) | `=== SUMMARY …`, `=== FAILED TESTS …`, `=== CRASH REPORTS ===` | the xcresult summary, failed test cases with messages, recent crash reports |
| Model facts | Print facts | `=== BEGIN MODEL FACTS ===`, `=== BEGIN MODEL FACTS SUMMARY ===` | one JSON object per repo; the Markdown tables of [Model facts](model-facts.md) |
| Drafter lab | Print lab | `=== BEGIN DRAFTER LAB ===`, `=== BEGIN DRAFTER LAB SUMMARY ===` | the lab JSON; the Markdown tables of [Drafter lab](drafter-lab.md) |
| Drafter lab | Print DFlash | `=== BEGIN DFLASH ===`, `=== BEGIN DFLASH SUMMARY ===` | DFlash rounds, tokens per round and acceptance |

Each `=== BEGIN` block ends at the next `=== END` line. The failure-summary sections (`=== SUMMARY`,
`=== FAILED TESTS`, `=== CRASH REPORTS`, then `=== CRASH <file>` per report) have no END marker: each runs
until the next `===` header or the end of the step.

- **In a browser:** open the run in the Actions tab, open the job, expand the step and search for
  `=== BEGIN`. A failed Local engine run also uploads `local-engine-xcresult` (kept 7 days) for Xcode.
- **From an agent sandbox** (no artifacts, no `gh run view --log`): list runs with
  `gh run list --branch <branch> --limit 5`, list jobs with
  `gh api repos/zqang/alvin-personal-assistant/actions/runs/<run_id>/jobs --jq '.jobs[] | {id,name,conclusion}'`,
  and read a job's log with the GitHub MCP tool `get_job_logs` (`job_id`, `return_content: true`,
  `tail_lines: 400`), or all failed jobs of a run with `run_id` and `failed_only: true`.

**On a Mac:**

```
swift test --package-path AssistantKit
cd LocalEngine
xcodebuild build-for-testing -scheme LocalEngine-Package -destination 'platform=macOS,arch=arm64' -configuration Release ENABLE_TESTABILITY=YES -skipMacroValidation -skipPackagePluginValidation
xcodebuild test-without-building -scheme LocalEngine-Package -destination 'platform=macOS,arch=arm64' -configuration Release -skip-testing:LocalEngineIntegrationTests
TEST_RUNNER_LOCAL_ENGINE_INTEGRATION=1 xcodebuild test-without-building -scheme LocalEngine-Package -destination 'platform=macOS,arch=arm64' -configuration Release -only-testing:LocalEngineIntegrationTests
```

Add `TEST_RUNNER_LOCAL_ENGINE_WOOF=1` for the Woof 4B suites (about 2.5 GB). The lab scripts' own commands
are in [Model facts](model-facts.md#running-it) and [Drafter lab](drafter-lab.md#running-it).

**What the engine tests call exact.**

- *Bitwise* (`max|Δ| == 0`): fork versus stock logits; replayed recurrent state versus the masked-kernel
  reference; a prefix loaded from disk versus the live cache; a restored checkpoint versus the state that
  never advanced.
- *allClose* (1e-4 to 1e-5, float32 tiny models): a reused session versus a fresh prefill of the same ledger;
  the logits after a commit versus a fresh run over the accepted prefix.
- *Token equality with a near-tie exemption* (speculative versus plain, engine versus `TokenIterator`): at the
  first mismatch the plain run's top-1 minus top-2 margin must be under 1e-3 on tiny models (0.5 logits on
  real 4-bit models, where differently shaped matmuls round differently); comparison then stops for that
  prompt. More than 2 near-ties that also exceed 2 % of the compared positions fail the test.
- *Statistical:* the acceptance rule passes a chi-square test over 200k simulated rounds; seeded T = 0.7 runs
  are identical, and unseeded first-token frequencies over 2,000 runs pass a chi-square test.

## 11. Device checklist

What only a run on an iPhone can settle. Run the benchmark from a **Release** build.

| # | Unknown | How to settle it | What it decides |
|---|---|---|---|
| D1 | Time to first token and tok/s of Woof 4B (and the smaller models) on target iPhones, engine versus stock: continued chat, barge-in, cold start with and without the disk prefix, long chat | Benchmark scenarios with "Compare engines" | Whether `.automatic` stays the engine default; which model is the default |
| D2 | Self-test pass rate on iPhone GPUs (A17, A18; on iOS 26.2+ the A19 runs MLX's neural-accelerator matmul and attention kernels) | Automatic at load; On-device engine › Run self-test | Engine availability per device |
| D3 | The iPhone's cost curve c(S) | On-device engine › Measure speed (stored) | 4 or 8 drafts per round (8 needs the fast kernels on as well); whether the fast kernels are worth enabling (they need a ≥ 25 % lower c(8): Test fast kernels) |
| D4 | Memory headroom: Woof plus Qwen3-ASR plus 3 checkpoints (about 150 MB) plus the prefix cache; jetsam limits; the increased-memory entitlement | Benchmark peak memory; Instruments | The checkpoint budget; whether to request the entitlement (a signing decision) |
| D5 | Thermals over a 10-minute voice session; how often the speculation gate backs off | Benchmark long chat plus the thermal state | Policy margins |
| D6 | Answer quality with as-generated history (empty think blocks kept) | Side-by-side benchmark transcripts | Whether to add an "exact re-render" policy |
| D7 | Early-start discard rate and its cloud cost; how often Qwen3-ASR changes the committed text | Latency report; traces' `adoptedEarlyStart` | The `earlyReplyStart` default and the tuner's bounds |
| D8 | Echo-cancellation residue causing false ducks; cue timing and voice consistency; barge-in latency for local, cloud and deep | Manual voice sessions | Ducking margins; cue defaults |
| D9 | Reminders, calendar and notification permission flows; the timer sound while the audio session plays and records | Tools section and voice requests | Whether voice mode must announce finished timers itself |
| D10 | Woof tool-call accuracy and time to the tool call on the phone (the lab measured 12/12) | Benchmark tool scenario | Whether the deferred two-step selector/filler is needed (below 85 %) |
| D11 | Foreground/background transitions with engine work in flight (risk of a GPU abort) | Lock, unlock and switch apps during replies and prefix saves | Whether background session save can ever be enabled |
| D12 | Whether Claude accepts remapped on-device `tool_use` ids in history, and `strict` together with `eager_input_streaming` | The first cloud turn after an on-device tool round (also testable on a Mac with an API key) | Keep or remove the id remap; the eager streaming flag |
| D13 | Whether to turn on automatic routing by default | Latency report p50/p90 per engine plus subjective quality | The `routingMode` default |

**Already settled in CI** (no device needed): `mtp.*` presence (none: WP42 skipped, and today's app does not
load Woof as garbage); template behaviour and EOS ids; quantization layout; prompt-lookup acceptance and tool
accuracy ([Drafter lab](drafter-lab.md)); fork parity; rollback exactness; lossless speculation; session reuse
exactness; that the macOS runner's GPU runs MLX ([Model facts](model-facts.md)). DFlash on Woof was measured
too: 5.03 tokens per round, but not faster or lossless in dflash-mlx ([Drafter lab](drafter-lab.md#dflash)).

## 12. Skipped and deferred

| Item | Decision | Reason |
|---|---|---|
| MTP drafter (WP42) | Skipped | No catalog checkpoint keeps `mtp.*` weights, although some configs declare a layer. |
| Swift DFlash port | Deferred (threshold met) | The lab measured 5.03 tokens and 4.03 accepted draft tokens per round on Woof (run 37786174712), above the 2.5 threshold, but dflash-mlx was slower than plain greedy on the CI GPU and not lossless against it. A port needs hidden-state taps from several layers and 0.3–1 GB more memory, and must prove lossless and faster on an iPhone. |
| Two-step tool selector and filler | Deferred | Woof called 12/12 tools correctly; revisit only if D10 shows < 85 % on the phone. |
| Fast kernels on by default | Deferred | Slower than stock at 8 rows on the CI GPU; enabled per device after a measured ≥ 25 % gain. |
| Model-shaped kernels, weight repacking, compiled decode | Skipped | Months of specialist work; upstream's compiled decode gave +1.7–3.5 % on a dense 4B. Revisit through a 3.32.x upgrade. |
| mlx-swift-lm 3.32.x upgrade | Deferred | Its MTP rewinds 1 token and is greedy-only; the fork already carries the sanitize fix. |
| Tree verification, lookahead/Jacobi decoding, EAGLE/Medusa | Skipped | Need tree-aware recurrent kernels or training; small gains on Mac; extra heat. |
| KV-cache quantization | Skipped | Only 8 of Woof's 32 layers hold KV; quality loss at 4 bits; replacing cache objects breaks checkpoints. |
| Per-step recurrent state capture, rollback by re-forwarding, carry-forward rollback | Skipped | 240–430 MB per round, a full extra forward, or expensive extra verify rows; capture/replay is exact and costs about 2 ms. |
| Prefill during speech | Skipped | Early start covers the same window more simply. |
| Background session save | Deferred | GPU work while backgrounding needs device testing (D11). |
| Hedged local and cloud requests | Skipped | Double cost, heat, inconsistent answers. |
| Token-confidence escalation | Telemetry only | Recorded per reply; a policy needs calibration on a device. |

## 13. Risks

- **Memory.** iOS limits one app's memory. Woof 4B, Qwen3-ASR and the checkpoints together may not fit without
  the increased-memory entitlement (D4). Qwen3-ASR already gives way to the reply model when memory is short.
- **Quality.** A 2–4B model is clearly weaker than Claude at open conversation. Claude stays the default, and
  automatic routing keeps complex, fresh-fact and deep requests in the cloud.
- **Heat and battery.** Long local generations get hot. Speculation backs off when the device is hot or in Low
  Power Mode, and the voice prompt asks for brevity.
- **GPU differences.** CI's virtual M1-class GPU is not an iPhone GPU. The self-test at load and the cost probe
  exist so that a device where the engine misbehaves or speculation does not pay falls back on its own.
- **API assumptions.** Replayed on-device tool ids and eager tool-input streaming are unverified against the
  live API (D12); both have a simple fallback (remove the remap or the flag).
