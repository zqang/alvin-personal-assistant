# MiniCPM-V evaluation

This document records what was measured with MiniCPM-V 4.7 on the iPhone, what those numbers support, and
how to run the benchmarks again. The result files live under `build/minicpm-results/` on the owner's Mac.
`build/` is gitignored, so this document is the record. Each figure below names the run it comes from.

**Device.** All runs used an iPhone 17 Pro Max (`iPhone18,2`, A19 Pro, iOS 26.6.1) with a Release build,
on 13–14 October 2026. The phone reports 12,262,113,280 bytes of physical memory. With no model loaded,
`os_proc_available_memory()` gave the app about 6.43 GB (the device probe, with the increased-memory
entitlement).

**What came of it.**

- **Woof 4B stays the default on-device model.**
- **MiniCPM-V 4.7 1B reads photos** for every engine; see the README's
  [Photo reading](../README.md#photo-reading).
- **MiniCPM-V 4.7 35B-A3B is an opt-in mode, "Most accurate on device"**; see the README's
  [Most accurate on device](../README.md#most-accurate-on-device). It isn't settled yet: the device checks
  D14–D26 in [the architecture doc's §11](architecture-plan-on-device-and-orchestration.md#11-device-checklist)
  can still hide it (decision 31 of the plan).

Anything marked **pending** needs a run on the iPhone (owner gate G2). This draft leaves those values open
until that run.

## Contents

1. [What the numbers support](#1-what-the-numbers-support)
2. [Runs](#2-runs)
3. [35B: first text and prompt length](#3-35b-first-text-and-prompt-length)
4. [Memory](#4-memory)
5. [Packaging](#5-packaging)
6. [Downloads](#6-downloads)
7. [Photo flow](#7-photo-flow)
8. [35B production profile and voice latency](#8-35b-production-profile-and-voice-latency)
9. [Running the benchmarks](#9-running-the-benchmarks)

## 1. What the numbers support

The numbers Codex reported match its result files. Several of the conclusions drawn from them don't hold.
In this table, *native* means a time taken inside the C bridge. *Swift* means the time at which the app's
Swift code received the text.

| Claim as first reported | What the data supports | Run (folder) |
|---|---|---|
| 1B: 40 ms to first text, 66 tok/s, 1.68 GB | Correct, as a native measurement | `1972C31D` (`iphone-normal`) |
| Woof 22/24 vs 1B 16/24 on the scored tasks | Holds. Exact McNemar p ≈ 0.03: Woof alone got 6 items right, the 1B alone got none. | `E71292B4` (`comparison-final`) |
| 1B first text 44 ms vs Woof 282 ms | The 44 ms is native, and that bridge returned only finished replies. The Swift caller got the 1B's text at about 0.48 s, its median full reply. Woof's 282 ms is a Swift time. | `E71292B4` |
| 1B memory 1.53 GB vs Woof 2.81 GB | The 1B's `phys_footprint` leaves out its 767 MiB of memory-mapped weights (`comparison-final-console.log:577-578`). Even with them added, it uses less than Woof. | `E71292B4` |
| 1B loading peak 2.41 GB | That is memory Woof left behind. The 1B's own block peaked at 1.53 GB. | `E71292B4` |
| 35B first text 2.349 s | Holds only for 19–22-token prompts. In the same run, 40 tokens took 3.42 s and 99 tokens took 4.97 s ([§3](#3-35b-first-text-and-prompt-length)). | `7C253525` (`comparison-35b-final`) |
| Woof 0.365 s and 65.3 visible characters/s | These figures mix thermal states. Woof's first block, which started at `nominal`, measured 0.259 s and 95.47 characters/s. | `7C253525` |
| 35B memory 3.11 GB vs Woof 2.81 GB | Not comparable. The 35B figure leaves out about 2.2 GiB of mapped weights, and Woof's figure includes its weights ([§4](#4-memory)). | `7C253525` |
| 35B 24/24 vs Woof 22/24 | Not significant. The whole gap is two arithmetic items (McNemar p = 0.5). | `7C253525` |
| 35B decodes at 5.58 tok/s | Measured: the median native decode rate over 16 performance rows. OpenBMB quotes 13+ tok/s, and nobody has explained the gap. The run used a DSpark draft that was converted locally, because the official file returned 404. | `7C253525` |
| Woof peak 5.11 GB, "not established" | Woof's last block loaded on top of 2,594 MB that the 35B hadn't released (`35b-comparison-reclaim-console.log:3609`). Woof's first block peaked at 2.80 GB, and its other seven replies in the last block at 2.73 GB. | `7C253525` |
| "About 17% faster after optimization" | Only a 3-prompt pilot supports this: `balanced` took 44.37 s in total, `baseline` 53.57 s. Its first text (2.31 s) was no better than the original 2.349 s, and peak memory rose from 2.70 to 3.65 GB. This project doesn't make that claim. | `75E359AD` |
| The Metal autorelease patch makes model memory release on time | Unproven on the phone. The patched runtime was built after the pilots and never installed. After the last unpatched block, 2,554 MB was still held (`35b-optimization-pool-pilot-console.log:5038`). D20 decides whether the patch stays. | `75E359AD` (its last block) |

Some things were never measured: the `balanced` profile on the 24-task suite, battery drain, and any 35B
prompt longer than 99 tokens. In 21 of the 35B's 40 prompts, the phone was already at `serious` thermal
state when the prompt started.

## 2. Runs

Run IDs are the first 8 characters of the folder names under `build/minicpm-results/`.

| Run | Started (UTC) | What | Outcome |
|---|---|---|---|
| `05C26CCA` | 13 Oct 13:01 | 1B, text and image | Low Power Mode was on, so it's superseded by `1972C31D`. `iphone/latest.json` still points here. |
| `1972C31D` | 13 Oct 13:03 | 1B: 8 text prompts and 3 image repetitions | Text: load 0.99 s, first text 40 ms, 66.17 tok/s, peak 1.68 GB. Image: first text 0.32 s, peak 1.53 GB. All native. |
| `DC4841AB` | 13 Oct 13:13 | Woof vs 1B | Cancelled after 1 row |
| `960137D6` | 13 Oct 13:37 | Woof vs 1B, ABBA | Completed, but report rendering inflated its timings, so `E71292B4` supersedes them. The accuracy scores were identical in both runs. |
| `E71292B4` | 13 Oct 13:43 | Woof vs 1B, ABBA, 80 rows | Accuracy 22/24 vs 16/24; first text 282 ms (Swift) vs 44 ms (native); peak 2.81 vs 1.53 GB |
| `F0231B5E` | 13 Oct 14:14 | Woof vs 35B | Cancelled |
| `423A5805`, `89D9000C` | 13 Oct 14:20, 14:28 | Woof vs 35B | Failed: not enough app memory to load the 35B after Woof (4,096 MB available; the check at the time needed about 5,000 MB) |
| `7C253525` | 13 Oct 14:33 | Woof vs 35B (`baseline` profile, DSpark on), ABBA, 80 rows | Accuracy 22/24 vs 24/24; first text 0.365 vs 2.349 s; decode 5.58 tok/s; peak 3.11 GB without mapped weights |
| `5A3A919E` | 14 Oct 08:03 | 35B profile pilot | Failed: the previous model's memory wasn't released within 10 s |
| `C9BC6753` | 14 Oct 08:10 | Pilot `io1` | First text 3.63 s, total 56.6 s, peak 3.63 GB |
| `75E359AD` | 14 Oct 08:14 | Pilot `balanced`, `draft2`, `baseline` (3 prompts each) | Totals 44.37 / 44.66 / 53.57 s; peaks 3.65 / 3.52 / 2.70 GB; every reply identical to `7C253525` |
| `B99B8196` | 14 Oct 08:24 | Pilot `io8` | First text 2.71 s, total 46.82 s, peak 3.64 GB |
| `threads4` | 14 Oct | Pilot | No data: CoreDevice reported "The connection was invalidated" (`35b-optimization-threads4-console.log:12`) |

`optimization-status.json`, `pilot-progress.json`, `comparison/latest.json` and `iphone/latest.json` are
stale pointers. Use the run folders above instead.

The 1B image test at one slice found the shapes and bars in the fixture but left out the text it was asked
for. That is one reason photo reading defaults to 4 slices. D16 chooses between 1 and 4.

## 3. 35B: first text and prompt length

The bridge in that run cleared the model's memory on every call, so each prompt was prefilled from
scratch. Here are the native first-text times by prompt length in `7C253525`:

| Prompt tokens | Rows | First text |
|---|---|---|
| 19–22 | 16 | 2.12–2.63 s (the 2.349 s median) |
| 40 | 1 | 3.42 s |
| 47–52 | 10 | 3.57–3.81 s |
| 56–70 | 12 | 3.85–4.38 s |
| 99 | 1 | 4.97 s |

A least-squares fit over all 40 rows gives about 40 ms per extra prompt token, on top of about 1.6 s. At
that rate a 1,000-token prompt would take about 40 s to reach its first word. That figure is an
extrapolation, not a measurement. A system prompt plus a few turns of history easily reaches that length;
Woof's system prefix with its tool schemas is about 1,650 tokens. For this reason the production path reuses
the 35B's session between turns, rewinds to checkpoints, and keeps the system prefix on disk
([architecture doc §4.12](architecture-plan-on-device-and-orchestration.md#412-minicpm-v-on-device)).

**Pending (A8 benchmarks, then D22 and D23).** First text with prompts of about 500, 1,000 and 2,000
tokens, and in a 5-turn reuse scenario, for both engines. Woof will run with its production speculation and
prefix cache. The first text will be timed in Swift, with the native value kept as `nativeFirstText`.

## 4. Memory

`phys_footprint`, the figure every earlier report used, leaves out llama.cpp's memory-mapped weights. Woof's
figure includes its weights, because MLX allocates them. The rows below are taken from llama.cpp's load log
plus arithmetic, not from a memory dump.

| Model | `phys_footprint` peak | Mapped model buffers (load log) | Source |
|---|---|---|---|
| Woof 4B | 2.73–2.81 GB (weights included) | — | `E71292B4`, `7C253525` |
| MiniCPM-V 1B | 1.53 GB during replies | 767 MiB (136 CPU + 630 Metal) | `comparison-final-console.log:577-578` |
| MiniCPM-V 35B-A3B, `baseline` | 3.11 GB | about 2.2 GiB: target 1,848 MiB plus DSpark draft 394 MiB | `35b-comparison-reclaim-console.log:1074-1075, 1582` |
| MiniCPM-V 35B-A3B, `balanced` | 3.65 GB | as above | `75E359AD` |

The 35B's load log also reports a 2,186 MiB mapped MoE expert bank (`:1076`). It is not yet known whether
`phys_footprint` counts it. The production benchmarks report `phys_footprint`, `resident_size` and the
runtime's `weights_bytes` side by side, so this question gets answered on the phone.

**Pending (D18).** No jetsam over 20 turns each of Woof + Qwen3-ASR + the 1B, and of the 35B + Qwen3-ASR +
the 1B. Record the peak `phys_footprint`, `resident_size` and `weights_bytes` here. The results replace the
estimated constants: 4.3 GB of available memory to load the 35B, and the 600 MB margin before Woof is
unloaded.

| Combination, 20 turns | Peak `phys_footprint` | `resident_size` | `weights_bytes` | Jetsam |
|---|---|---|---|---|
| Woof + Qwen3-ASR + 1B | pending | pending | pending | pending |
| 35B + Qwen3-ASR + 1B | pending | pending | pending | pending |

**Pending (D20).** The Metal autorelease patch. Run 3 load/unload cycles of 35B → Woof → 1B, and check that
the footprint returns to within 300 MB of baseline and that Woof loads. If the patch has no effect, it gets
dropped.

## 5. Packaging

The app links the `MiniCPM` package. That package's `llama` target is a prebuilt `llama.xcframework`:
llama.cpp-omni `96bbf61`, plus the local Metal patch. It is published as a checksum-pinned release asset
([§9.6](#96-building-the-runtime)), so a clean clone needs no CMake.

Each `llama.framework` slice (`ios-arm64`, `ios-arm64-simulator`, `macos-arm64`) carries two things:

- **`LICENSES/`:**
  - llama.cpp's `LICENSE`;
  - everything in the runtime's `licenses/`, including nlohmann/json's `LICENSE-jsonhpp` and cpp-httplib's
    `LICENSE-httplib`;
  - the header licence blocks of stb_image, miniaudio and llamafile's sgemm;
  - minja's and YaRN's copyright lines with the MIT terms, since their sources name the licence without
    carrying it.
- **`PrivacyInfo.xcprivacy`:** C617.1 (`stat`), the only required-reason API that `nm -u` shows. The build
  fails if that changes. The app's own `PrivacyInfo.xcprivacy` keeps 35F9.1 for its own use.

The app shows the same notices under Settings › Acknowledgements.

**Weight licences (owner check G1, before any TestFlight build).** Read through the Hugging Face API on
15 October 2026:

| Repository | Revision | Declared licence |
|---|---|---|
| openbmb/MiniCPM-V-4.7-1B (model card) | b0918c7 | `apache-2.0`; the card names no other model licence |
| openbmb/MiniCPM-V-4.7-35B-A3B (model card) | f0e2552 | `apache-2.0`; the card names no other model licence |
| openbmb/MiniCPM-V-4.7-35B-A3B-DSpark | 5facb75 (pinned) | `apache-2.0` |
| tc-mb/MiniCPM-V-Apps-gguf (the GGUF files) | 8716e52 (pinned) | none declared |

The model cards aren't pinned by the app, and the GGUF host is a personal repository with no licence of
its own. The owner confirms the terms before distribution; Acknowledgements follows the result.

**Pending (D14).** Install, then archive and validate. Check that the app launches, that the size grows by
about the framework (each slice is about 9.6–9.8 MB unzipped), that the archive's privacy report lists
`llama.framework`'s reasons, that the camera prompt text is right, and that Acknowledgements opens.

| Check | Result |
|---|---|
| App size change on the device | pending |
| Privacy report lists `llama.framework` | pending |
| Camera prompt text | pending |

## 6. Downloads

Both model sets download in the app. Photo reading (the 1B) is 1.78 GB: the model is 672 MB and the vision
projector 1,109 MB. Most accurate on device (the 35B) is 15.50 GB: the model is 1.66 GB and the experts
13.84 GB. How downloads resume, verify, use space and are deleted is described in the
[architecture doc §4.12](architecture-plan-on-device-and-orchestration.md#412-minicpm-v-on-device). In
short, each file is hashed once, and after that a load checks only its size and modification time.

**Pending (D15, D21).**

| Check | 1B (D15) | 35B (D21) |
|---|---|---|
| Cancel, then leave mid-file: finished files kept, download resumes on return | pending | — |
| First verification time | pending | pending (≤ N min, recorded) |
| Later loads check in ≤ 3 s | pending | pending |
| Existing folder reused without a download; projector removed, 1.12 GB freed | — | pending |
| Refused below the missing bytes + 1 GB free | — | pending |
| `isExcludedFromBackup` set | pending | pending |
| Delete frees the bytes shown | pending | pending (about 15.5 GB) |

## 7. Photo flow

**Pending (D16).** Run the 4 test fixtures (English sign, Chinese menu, receipt, poster) plus 4 real scenes,
at 1 and at 4 slices. The setting passes if at least 90% of the key strings are present and the note is
complete within 6 s at p50. The default becomes the smallest slice count that passes. The chosen value
goes into `MiniCPMHost`, and the speed into the README.

| Slices | Key strings present | Note complete, p50 | Note complete, p90 |
|---|---|---|---|
| 1 | pending | pending | pending |
| 4 | pending | pending | pending |

**Pending (D17).** Chat questions about a photo answered by Woof and by Claude, with only the note text sent;
a poster becoming a calendar event through the permission prompt; in voice mode, the shutter working while
the app listens, echo cancellation and barge-in still working, a barge-in during a read cancelling it
without a reload, and the next turn carrying the photo.

## 8. 35B production profile and voice latency

`MiniCPMSession.production` stays **target-only** until D22. D22 is an ABBA run with three arms
(target-only, `baseline` and `balanced`) covering the 24-task suite, the 8 performance prompts, the
500/1,000/2,000-token prompts and the 5-turn reuse scenario. It adopts a DSpark profile only if all three of
these hold:

- no item of the 24-task suite gets worse;
- the median first text and tok/s, both measured in Swift, are each at least 10% better;
- peak resident memory is at most 4.5 GB.

| Arm | 24-task score | First text p50 (Swift) | tok/s p50 | Peak resident |
|---|---|---|---|---|
| target-only | pending | pending | pending | pending |
| `baseline` | pending | pending | pending | pending |
| `balanced` | pending | pending | pending | pending |

**Pending (D23).** Voice latency to first audio, p50 and p90, for these cases: a cold start from the disk
prefix, a cold start without one (the first reply then also writes the prefix file, about 73 MB, before it
answers), an append, a barge-in followed by `replaceLastReply`, and a replaced tentative turn. Also the
checkpoint size and the time a rewind takes. The targets for append and barge-in are p50 ≤ 5 s and
p90 ≤ 8 s; for a cold start, ≤ 10 s.

A conversation that outgrows the context budget (4,096 tokens minus the reply cap) is rebuilt from its
latest turns, which are read again. A rebuild fills at most half the budget, so it is followed by several
appends before the next one; each rebuild is a stall of roughly 1,000–1,900 tokens at 26–53 ms each.
Record it in voice (256-token cap) and typed (768) conversations that run past the budget, and put the
measured stall and how often it comes in the settings copy.

| Case | First audio p50 | p90 |
|---|---|---|
| Cold start from the disk prefix | pending | pending |
| Cold start without a prefix file (it is written first) | pending | pending |
| Append | pending | pending |
| Barge-in, then `replaceLastReply` | pending | pending |
| Replaced tentative turn | pending | pending |
| Rebuild past the budget, voice | pending | pending (and every how many turns) |
| Rebuild past the budget, typed | pending | pending (and every how many turns) |

**Pending (D24).** A 10-minute 35B voice conversation, then Low Power Mode. Record the battery % per
10 minutes and the time until `.serious`. Check that Woof takes over within one turn.

**Pending (D25).** In airplane mode, "Remind me at 5" and "5点提醒我", then an open question. Woof should
create the reminder, and the 35B should answer next, with its prefix restored. Record the swap time.

## 9. Running the benchmarks

Apart from the reuse scenario, these benchmarks measure fresh model requests for comparison. They don't
measure the assistant as you use it. Run them from a **Release** build on a real iPhone. Keep the app in the foreground, the phone unlocked
and Low Power Mode off. The simulator is not a measurement device, and Mac numbers don't predict iPhone
speed.

Replace `YOUR_DEVICE_ID` in the commands below. If you changed the bundle identifier for signing, use
yours.

### 9.1 1B benchmark in the app

**Settings › On-device engine › Advanced settings › MiniCPM-V benchmark** runs MiniCPM-V 4.7 1B. It uses the
Q4_K_M language model and the F16 vision projector. It measures either the eight English and Chinese
benchmark prompts, or three repetitions of a fixed image-reading task. Each prompt starts fresh, with
thinking off, greedy decoding, one image overview slice and at most 128 output tokens. The first prompt
includes cold kernel work. MLX models are unloaded before the run, and MiniCPM is unloaded after it.

To run the text and image scenarios back to back from a Mac, launch the app with `--minicpm-benchmark all`.
Pass `text` or `image` instead of `all` to run just one of them.

```bash
xcrun devicectl device process launch --device YOUR_DEVICE_ID --terminate-existing --console \
  com.alvinang.personalassistant --minicpm-benchmark all
xcrun devicectl device copy from --device YOUR_DEVICE_ID \
  --domain-type appDataContainer --domain-identifier com.alvinang.personalassistant \
  --source Documents/MiniCPMBenchmark --destination build/minicpm-results/iphone
```

Each run gets its own folder of JSON and text reports, and `latest.json` points to the newest. A run that
was cancelled or failed keeps its partial results, and it doesn't restart on its own when the app comes back
to the foreground.

### 9.2 Woof versus the 1B

Launch with `--compare-local-models`. Both models must already be downloaded. The run goes current model,
MiniCPM, MiniCPM, current model. Each block excludes one warm-up and then runs eight English and Chinese
performance prompts. Each model also answers the 24 fixed bilingual tasks once. An answer passes only if
both its content and its format are right. This is a small diagnostic, not a standard accuracy benchmark.
Reports go to `Documents/ModelComparison/`; copy them with the command in §9.1, using that folder as the
source.

### 9.3 Woof versus the 35B

The 35B suite runs the same 24 tasks and 8 performance prompts against the model selected in Settings. It
measures text only and doesn't load a vision projector. LES keeps the model's top-8 expert routing, with a
256-slot shared bank, 32 or 48 cached experts per layer (depending on the profile), and lookahead prefetch
depth 4.

The 35B files come either from the in-app download (Settings › Most accurate on device) or from the Mac:

```bash
python3 scripts/download_minicpm_35b.py --target-only
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

The DSpark draft that the profile arms need comes from §9.7 and is copied the same way. Wait for the run to
finish before you copy its reports. `latest.json` is written as soon as the run starts
(status `running`) and again when it ends, so an interrupted run keeps the prompts it completed.

Here is what the reports contain:

- **Memory:** `phys_footprint`, `resident_size` and the runtime's `weights_bytes`.
- **First text:** the time the Swift stream received it, with the native time kept as `nativeFirstText`.
- **Prompt rows:** rows of about 500, 1,000 and 2,000 tokens, and a 5-turn reuse scenario for both engines.
  Woof runs with its production speculation and prefix cache.

### 9.4 Profiles and pilots

Without `--minicpm-profile`, the 35B comparison uses the production profile (`MiniCPMSession.production`),
which is target-only until D22. These profiles are available:

| Profile | What it changes |
|---|---|
| `baseline` | The original configuration: 32 cached experts per layer and up to 3 DSpark drafts |
| `balanced` | 48 cached experts, four-layer cancellation checkpoints and allocation-free argmax |
| `checkpoints4` | Four-layer checkpoints with the original 32-expert cache |
| `draft2` | The `balanced` settings with 2 draft tokens |
| `io1`, `io8` | 1 or 8 storage-reader threads instead of 4. Each needs its own app launch, because the runtime reads this setting once per process. |
| `threads4` | 4 CPU threads instead of 6 |

The target-only arm runs the same weights without the draft. No profile changes the weights,
quantization, top-8 routing or exact greedy verification. A profile's name doesn't make it faster; only a
measurement on the phone shows that.

For a short profile pilot, add `--minicpm-pilot baseline,checkpoints4,balanced,draft2,baseline` instead.
Each named profile reloads the model, warms up, and then runs performance prompts 1, 4 and 5. Pilot reports
go to `Documents/MiniCPM35BProfilePilot` and don't replace the full comparison. Every block first frees the
previous model's memory, then waits up to 120 s for a `nominal` thermal state. Older runs didn't cool down
between blocks.

### 9.5 On a Mac

These commands build the MiniCPM package against the prebuilt runtime, so they need no CMake.

```bash
swift test --package-path MiniCPM                     # model tests skip without model files
python3 scripts/download_minicpm.py                   # 1B into build/minicpm-models
MINICPM_BENCH_MODEL_DIR=build/minicpm-models swift test -c release --package-path MiniCPM
swift run -c release --package-path MiniCPM minicpm-bench \
  build/minicpm-models/MiniCPM-V-4_7-1B-Q4_K_M.gguf build/minicpm-models/mmproj-model-f16.gguf
```

To read an image, add its path after the two model paths. `--prompt "..."`, `--max-tokens N`,
`--profile NAME`, `--warmup N` and `--repeat N` change the run. The CLI writes a JSON report to stdout and
the runtime's logs to stderr.

The 35B tests are slow. Run them once with the draft and once without it, by leaving
`MINICPM_BENCH_35B_DRAFT` unset:

```bash
M35=build/minicpm-35b-models
MINICPM_BENCH_35B_MODEL="$M35/MiniCPM-V-4_7-35B-A3B-mix_q3gud_im-les.gguf" \
MINICPM_BENCH_35B_EXPERTS="$M35/experts/manifest.json" \
MINICPM_BENCH_35B_DRAFT="$M35/MiniCPMV47-DSpark-Q8_0.gguf" \
  swift test -c release --package-path MiniCPM --filter 35B
```

### 9.6 Building the runtime

Only the owner needs to do this, when the pinned runtime or its patch changes. Nobody needs it just to
build the app.

`scripts/prepare_minicpm.sh all` builds the three slices on an Apple Silicon Mac with Xcode. Here is what it
does:

- **CMake.** It uses `$CMAKE` if that is set. Otherwise it installs the hash-pinned CMake wheel from
  `scripts/cmake-requirements.txt` into a project-local Python virtual environment, with
  `pip install --require-hashes`.
- **Source.** It downloads llama.cpp-omni `96bbf61` and checks the archive against its pinned SHA-256. It
  always extracts into a fresh directory, then applies `minicpm-metal-worker-autorelease-v1`.
- **Output.** It writes `MiniCPM/Frameworks/llama.xcframework`, then zips it with `ditto` into
  `build/minicpm/llama.xcframework.zip` and prints `swift package compute-checksum` for that zip. It also
  writes `build/minicpm/PROVENANCE.txt`.
- **Licences.** The build fails if any licence is missing from a slice.

To publish the result, the owner first commits the build script and its inputs, rebuilds on a clean tree
(so `PROVENANCE.txt` names that commit rather than a `-dirty` one), pushes the commit, and uploads both
files to a release named as the script prints it:
`minicpm-runtime-96bbf61-p<first 8 hex of the patch SHA-256>-s<first 8 hex of the script's SHA-256>`,
with `--target <that commit>`. That release's URL and the checksum go into the `binaryTarget` in
`MiniCPM/Package.swift`, and its runtime part (`96bbf61…-p<patch>`) stays in
`MiniCPMRuntimeConfiguration.runtimeRevision`, which keys the saved prefixes. Anything that changes the
asset changes the tag. Never re-upload under an existing tag (`--clobber`): older commits pin its
checksum. Deleting the release asset breaks package resolution for every clone. To try a local build before
publishing it, point that `binaryTarget` at `path: "Frameworks/llama.xcframework"` for the time being;
otherwise the package ignores `MiniCPM/Frameworks`.

The Metal patch adds `@autoreleasepool` blocks to the Metal encoding and residency workers. It changes no
model operation, weight or precision, and the Swift bridge also drains temporary native objects at each
call. Whether the patch makes any difference on the phone is unproven ([§4](#4-memory), D20).

### 9.7 Converting the DSpark draft

Only the D22 profile arms need the draft. Converting it needs Python 3.11 or later (macOS's own
`/usr/bin/python3` is 3.9, so name a newer one), plus the macOS slice from §9.6 (for `llama-quantize`).

```bash
MINICPM_INSTALL=0 scripts/prepare_minicpm.sh macos   # leaves MiniCPM/Frameworks as it is
python3 scripts/download_minicpm_35b.py --target-only
python3.11 -m venv build/minicpm-dspark-venv
build/minicpm-dspark-venv/bin/python -m pip install numpy==2.4.4 tqdm==4.67.3 pyyaml==6.0.3

mkdir -p build/minicpm-dspark-source
MINICPM_DSPARK_SOURCE='https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-DSpark/resolve/5facb75c251ffd524524a544c1f5fd97ba0d6c5c'
for file in config.json model.safetensors; do
  curl --fail --location --retry 3 "$MINICPM_DSPARK_SOURCE/$file" \
    -o "build/minicpm-dspark-source/$file"
done

CMAKE="${CMAKE:-build/cmake-venv/bin/cmake}"   # what prepare_minicpm.sh used
"$CMAKE" --build build/minicpm/macos-arm64 --config Release --target llama-quantize
build/minicpm-dspark-venv/bin/python scripts/convert_minicpm_dspark.py
python3 scripts/download_minicpm_35b.py --verify-only
```

`CMAKE` defaults to the pinned wheel's CMake, which `prepare_minicpm.sh` uses unless `CMAKE` was set for it.
The converter does the following:

- It verifies the checkpoint, the configuration and the paired target's hashes, and copies the target's
  tokenizer.
- It maps all 62 DSpark tensors onto the pinned runtime's DFlash/Markov implementation. Normalization and
  bias vectors stay F32, matrices go through the runtime's Q8 quantizer, and the single-row confidence head
  stays F16.
- It writes a `.provenance.json` sidecar with the source hashes, the revisions and the conversion settings.

`MiniCPM35BManifest.json` pins the converted draft's hash. Report the converted draft and the profile with
any measurement; numbers from the upstream demo aren't measurements of this integration.

Runtime and model revisions are pinned in `scripts/prepare_minicpm.sh`,
`MiniCPM/Sources/MiniCPM/MiniCPMArtifacts.swift` and `MiniCPM/Sources/MiniCPM/MiniCPM35BManifest.json`.
For provenance, see [OpenBMB's mobile implementation](https://github.com/OpenBMB/MiniCPM-V-Apps).
