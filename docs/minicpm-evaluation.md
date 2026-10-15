# MiniCPM-V evaluation

This document records what was measured with MiniCPM-V 4.7 on the iPhone, what those numbers support, and
how to run the benchmarks again. The result files live under `build/minicpm-results/` on the owner's Mac;
the device gate's are in `build/minicpm-results/device-gate-2026-10-15/<check>/`. `build/` is gitignored,
so this document is the record. Each figure below names the run or check it comes from.

**Device.** All runs used an iPhone 17 Pro Max (`iPhone18,2`, A19 Pro, iOS 26.6.1) with a Release build:
Codex's runs on 13–14 October 2026, and the device gate (D14–D23) on 15–16 October 2026 with the build of
`d0387f0`. The phone reports 12,262,113,280 bytes of physical memory. With no model loaded,
`os_proc_available_memory()` gave the app about 6.43 GB, and the app's limit (footprint plus available)
was 6 GiB (the device probe, with the increased-memory entitlement).

**What came of it.**

- **Woof 4B stays the default on-device model.**
- **MiniCPM-V 4.7 1B reads photos** for every engine; see the README's
  [Photo reading](../README.md#photo-reading).
- **MiniCPM-V 4.7 35B-A3B is an opt-in mode, "Most accurate on device"**; see the README's
  [Most accurate on device](../README.md#most-accurate-on-device). It runs target-only: D22 found that
  neither DSpark profile earns its place ([§8](#8-35b-production-profile-and-voice-latency)). **It failed
  D23's voice-latency line**: first audio after an append took 7.5 s at p50 and a barge-in 9.3 s, against
  p50 ≤ 5 s. Under decision 31 of the plan, the owner either re-tunes it and runs D23 again, or merges with
  the mode hidden (`MiniCPMHost.accurateEligible` returns false).

The device gate's results per check are in
[the architecture doc's §11](architecture-plan-on-device-and-orchestration.md#11-device-checklist). Anything
still marked **pending** needs somebody holding the iPhone: Acknowledgements on the phone (D14), D15, the
real photos of D16, D17, locking the phone (D19), Delete (D21), the live-voice parts of D23, D24, D25, and
D26 (for the VoiceKit PR).

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
| 35B decodes at 5.58 tok/s | Measured: the median native decode rate over 16 performance rows. OpenBMB quotes 13+ tok/s, and nobody has explained the gap. The run used a DSpark draft that was converted locally, because the official file returned 404. D22 measured the production profile (target-only) at 5.87 tok/s in Swift. | `7C253525`; D22 |
| Woof peak 5.11 GB, "not established" | Woof's last block loaded on top of 2,594 MB that the 35B hadn't released (`35b-comparison-reclaim-console.log:3609`). Woof's first block peaked at 2.80 GB, and its other seven replies in the last block at 2.73 GB. | `7C253525` |
| "About 17% faster after optimization" | Only a 3-prompt pilot supports this: `balanced` took 44.37 s in total, `baseline` 53.57 s. Its first text (2.31 s) was no better than the original 2.349 s, and peak memory rose from 2.70 to 3.65 GB. This project doesn't make that claim. | `75E359AD` |
| The Metal autorelease patch makes model memory release on time | Unproven when claimed: the patched runtime was built after the pilots and never installed. After the last unpatched block, 2,554 MB was still held (`35b-optimization-pool-pilot-console.log:5038`). D20 then found **no effect** ([§4](#4-memory)). | `75E359AD` (its last block); D20 |

In 21 of the 35B's 40 prompts in `7C253525`, the phone was already at `serious` thermal state when the
prompt started. The device gate has since measured the `balanced` profile on the 24-task suite and 35B
prompts of 500 to 2,000 tokens (D22). Battery drain is still unmeasured (D24).

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
for. That was one reason photo reading defaulted to 4 slices, and D16 kept it ([§7](#7-photo-flow)).

**Device gate, 15–16 October 2026** (times +08). Every check ran the Release build of `d0387f0` (installed
app SHA-256 `a2c7f781…`), on the phone above, with Low Power Mode off and the thermal state recorded at
each start. D16, D18, D19, D20 and D23 needed measurements the app doesn't expose, so each used a small
test harness hooked into the device probe (`--benchmark-device-probe`); it was built into the same build,
never committed, and the clean build was reinstalled afterwards. Its source is kept in the check's folder.
No model file changed in any check, and nothing was downloaded.

| Check | When | What ran |
|---|---|---|
| D14 | 15 Oct 19:11–21:20 | Install and 2 launches; Release archives of this branch and of `main`; the archives' privacy manifests; Acknowledgements in a simulator UI test |
| D21 | 15 Oct 21:25–21:39 | Probes of eligibility, status, backup flags and files; a timed re-hash of the 35B and 1B files |
| D16 | 15 Oct 21:45–21:57 | `D16PhotoGate`: 2 runs of the 4 fixtures at 1 and 4 slices, plus stress variants and the product path |
| D18 | 15 Oct 22:16 – 16 Oct 00:15 | `D18MemoryGate`: 20 turns each in modes A (Woof), B (35B) and C (forced co-residence) |
| D20 | 16 Oct 00:30–01:01 | `D20ReleaseCycles`: 6 runs of 3 cycles, alternating the shipped runtime and the same runtime without the patch |
| D22 | 16 Oct 01:09–03:34 | `--compare-local-models-35b`, one process per arm in ABCCBA order: `1E1EA672` and `CEFA3E06` (target-only), `AD83B50C` and `A2C7B565` (`baseline`), `E12206C6` and `EC1DA291` (`balanced`) |
| D23 | 16 Oct 03:48–05:19 | `D23VoiceLatency`: 6 scripted 10-request conversations plus 2 cold starts without a prefix file |
| D19 (automated) | 16 Oct 05:28–06:24 | `D19Background`: 30 app switches through Settings during reads, replies and prefix saves |

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

**Measured (D22).** First text timed in Swift, in the first 35B block of each of D22's six runs (all three
profiles; target-only alone in brackets), against Woof's block in the same runs, with its production
speculation and prefix cache. Every long answer was correct.

| Prompt tokens | 35B first text | Woof first text |
|---|---|---|
| 515 | 20.1–23.6 s (target-only 20.1, 23.6) | 1.10–1.29 s |
| 1,035 | 39.7–46.6 s (39.7, 44.9) | 2.12–2.24 s |
| 2,073 | 82.8–93.8 s (86.3, 87.7) | 4.10–4.23 s |

So the extrapolation holds: about 40 ms per prompt token, whatever the profile. In the 5-turn reuse scenario
the first turn (407 tokens, no saved prefix) took 19.1–23.2 s to first text; turns 2–5 fed only 58–65 new
tokens and took 4.6–5.9 s (target-only p50 4.88 s, n=8). Woof took 0.91–1.00 s and 0.26 s.

Short feeds cost more per token. In the app's own path (D23), first text = 0.29 s + 78.7 ms per fed token
(fit over 54 warm requests that fed 52–87 tokens). A spoken turn of 3–4 words still feeds about 50 tokens:
the chat scaffolding and the per-turn `<context>` time tag are about 47 of them, roughly 3.7 s.

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

The device gate measured all three figures side by side. `weights_bytes` is the runtime's mapped weights
(`llama_model_size`), which `phys_footprint` leaves out and `resident_size` includes.

| Model, as the app runs it | Peak `phys_footprint` | Peak `resident_size` | `weights_bytes` | Source |
|---|---|---|---|---|
| 35B, production (target-only) | 2.78 GB | 4.41 GB | 1,651,756,800 B | D22 |
| 35B, DSpark `baseline` / `balanced` | 3.27 / 4.13 GB | 5.20 / 5.99 GB | 2,065 MB (with the draft) | D22 |
| 1B, reading a photo | 1.54 GB | 2.24 GB | 661,048,048 B | D16 |

The 35B's 13.8 GB of LES experts aren't mapped: they are read into a cache, and that cache is in
`phys_footprint`. The 35B's recurrent-state checkpoint is 65,864,428 B (D18, D23).

**D18: memory over 20 turns.** One turn is what voice mode does: a 1B photo note, a Qwen3-ASR pass over a
spoken clip, then the reply. Modes A and B use the app's own hosts with the constants of `d0387f0`; mode C
is an experiment that keeps all three models loaded.

| Combination, 20 turns | Peak `phys_footprint` | Peak `resident_size` | Lowest available | Memory warnings | Jetsam |
|---|---|---|---|---|---|
| A: Woof + Qwen3-ASR + 1B | 5.13 GB (5.25 GB at a warning) | 5.61 GB | 1.32 GB (1.19 GB at a warning) | 33 | none; 20/20 replies, 18/20 ASR passes |
| B: 35B + Qwen3-ASR + 1B | 4.12 GB (4.14 GB at a warning) | 5.17 GB | 2.33 GB | 17 | none; 15/20 35B replies |
| C: all three kept loaded | 5.43 GB | 6.06 GB | 1.01 GB | 62 | not the app; one system `vm-pageshortage` event (8.31 GB wired) killed about 80 background daemons |

What each model adds to `phys_footprint`:

- **Woof 4B:** +2.52 GB after load and prewarm (peak +2.90 GB); a reload plus a reply beside ASR peaks at
  +3.18 to 3.27 GB.
- **Qwen3-ASR:** +0.93 to 1.14 GB to load (median 1.02 GB, n=8); each pass +0.12 to 0.33 GB (n=20).
- **1B:** a load plus a read peaks at +1.26 to 1.34 GB (n=3), +1.38 GB beside the 35B and ASR; a read on the
  loaded model adds 0.015–0.065 GB.
- **35B, production:** +0.73 GB after load and prewarm; the first reply fills the expert cache, for about
  2.73 GB in use. A reload plus a reply peaks at +2.12 to 2.69 GB (median 2.56 GB, n=12); later replies add
  at most 0.053 GB. Decision 21's 3.65 GB was the `balanced` profile with the draft, about 0.9 GB more.

iOS sends the app a memory warning once its footprint passes about 5.04–5.09 GB, which leaves about
1.36–1.41 GB available. With the 35B and the 1B loading together, warnings also came at 1.92–4.14 GB, from
system-wide pressure (GPU-wired memory). The app's handlers then unload every idle model. So with the
constants of `d0387f0` (4.3 GB to load the 35B, 2.2 GB for the 1B, 600 MB to spare), every turn crossed the
warning line:

- in mode A the 1B loaded beside Woof and ASR (2.36–2.77 GB was available, and it needed 2.2 GB), and the
  warning then unloaded all three: Woof reloaded for every reply (3.3–9.6 s) and ASR for every turn;
- in mode B each photo's warning unloaded the 35B, so its next reply read the whole conversation again:
  28–114 s, median 70.7 s, against a median of 22 s in mode C, where it stayed loaded;
- 3 of mode B's 35B replies were refused at 4.01–4.05 GB available, because the 1B's memory was still
  being released; 2 more were cancelled mid-prefill at `serious`, for a reason not found.

**The constants now** (`MiniCPMHost`), derived from these numbers:

- `runBytes`, what a run on a loaded model leaves free: 600 MB → **1.75 GB**, the 1.41 GB warning distance
  plus a 0.33 GB ASR pass.
- `loadBytes`, what a model needs free to load: its peak plus `runBytes`. For the 35B that is 2.75 + 1.75 =
  **4.5 GB** (was 3.7 + 0.6 = 4.3 GB); for the 1B, 1.45 + 1.75 = **3.2 GB** (was 2.2 GB).
- Before refusing a load, the host waits up to 5 s for memory that is still being released (D18's refusals,
  Woof's 0.32 s, D19's 3.5–5 s).

The intended effect: with Woof and ASR loaded, a photo evicts Woof first instead of setting off the warning
that unloads all three, and in 35B mode it evicts the idle 35B, whose next reply still reads the
conversation again (keeping both loaded, as mode C did, isn't viable). **These effects are predicted from
the measurements, not re-measured.** Run modes A and B again with the D18 harness (kept in
`build/minicpm-results/device-gate-2026-10-15/D18/harness/`) before relying on them.
One risk is known: a 35B unloaded for the second time in a process can stay counted for 2–90 s (D20), longer
than the 5 s wait, and a 1B load that needs 3.2 GB may then be refused (the photo's read stops, and the user
is asked to try again).

**D20: the Metal autorelease patch.** 3 load/unload cycles of 35B → Woof → 1B, 30 or 90 s watched after each
unload, in 6 alternating runs: 3 with the shipped runtime and 3 with the same runtime built without the
patch (18 cycles). No memory warning, jetsam or crash. The idle baseline was 23.3–23.5 MB.

- **Pass.** Woof loaded 18/18 times (1.66–6.18 s; the first load in a process takes about 6 s). Every unload
  came back within 300 MB of baseline, and after each full cycle the footprint was +70 to +184 MB (median
  +148 MB), with no growth from cycle to cycle.
- **Woof and 1B.** Woof's unload returned with 1.7–2.7 GB still counted, which was gone 0.10–0.32 s later; the
  1B's was gone at once.
- **35B.** Its first unload in a process was immediate (+69 to +72 MB, 6/6). Every later one left +2.46 to
  2.50 GB counted (12/12) for 2.4 to more than 90 s: with the patch 4.7, 13.4, 28.5, 81.4 s and two still held
  at 30 and 90 s; without it 2.4, 67.5, 67.6 s, two still held at 30 and 90 s, and one released during the
  next Woof load. Medians 29.3 s and 48.8 s (Mann-Whitney U = 16, n = 6 and 6, p = 0.82): **no difference**.
  malloc's own count fell from about 3.4 GB in use to 20–35 MB at every 35B unload, and
  `malloc_zone_pressure_relief` returned 0 B. So the hold looks like the system reclaiming large freed
  allocations late, not Metal; that is an inference. While it lasts, `os_proc_available_memory()` reads
  3.9–4.0 GB instead of 6.4 GB, and a 35B load in that window is refused. Woof's load on top of 2.36 GB still
  held peaked at 4.20 GB with no warning, and the held memory went away during it.
- **Codex's runs before the patch** show the same pattern: 61–103 MB right after 7 of 10 releases (every first
  release in a process), and 2,554–3,466 MB after the other 3.

So the patch is to be dropped. Dropping it changes the runtime asset (a new tag and checksum), which is the
owner's release step (gate G1); until then the shipped runtime keeps it, with a `ponytail:` note at
`MiniCPMRuntimeConfiguration.runtimeRevision`.

**D19, automated part: app switches.** 30 switches to Settings and back (about 10 s away), during 1B reads
(7), 35B replies (10) and the 35B's prefix prefill or save (13), with the app suspended (24) or kept running
by audio, as voice mode does (6). The same process came back every time, with no GPU error, crash, jetsam or
memory warning. Interrupted GPU work stopped within 0–95 ms with `CancellationError`, a prefix save already
writing finished within 13 ms and left a valid file, a stopped prefill left no file, and reads, replies and
the prefix all worked on return. After two or more 35B unloads in a process, the first 35B reply after a
return was refused in 3 of 7 suspended trials, at 4.11 GB available; 4.3 GB was back 3.5 and 5.0 s later.
Locking the phone was not tested; that stays manual.

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

**D14: install and archive.** The Release build installed and launched normally twice, each time alive at
5, 15 and 30 s, with no crash or jetsam report. The size change comes from Release archives (`xcodebuild
archive`) of this branch and of `main` (`88a7eb4`).

| Check | Result |
|---|---|
| App size change | The archived app grows from 30.02 MB to 40.85 MB uncompressed: **+10.83 MB**, of which `llama.framework` is 9.75 MB (90 %). The rest is the app's own code (+1.05 MB), the MiniCPM resource bundle (+17 kB) and the acknowledgements (+16 kB). App Thinning for this phone (`iPhone18,2`): 13.5 MB compressed instead of 9.7 MB (**+3.8 MB** to download), 40.7 MB instead of 29.8 MB installed. |
| Privacy report lists `llama.framework` | Yes. The archive holds three manifests: the app's, `llama.framework`'s (`NSPrivacyAccessedAPICategoryFileTimestamp`, C617.1; no tracking, no collected data) and swift-crypto's. `nm -u` on `llama` shows `_stat` as its only required-reason API. The app's manifest is byte-identical to `main`'s: its new imports (file dates and free space) fall in categories it already declares. |
| Camera prompt text | `NSCameraUsageDescription` reads "Take a photo for the assistant to read on this iPhone. The photo is never sent to a model provider; it stays in the chat on this iPhone until you delete the chat." in `Config/Info.plist`, the built app and the archive; `main` has none. The iOS alert itself wasn't shown (nobody could tap the phone). |
| Acknowledgements opens | In a simulator UI test of the same commit: Settings › Acknowledgements opens and shows the llama.cpp-omni and MiniCPM-V 4.7 35B-A3B notices. The bundled file is identical on the phone. **Pending on the phone** (manual). |
| Validate | Not possible: the signing team is a free Personal Team, which can't make a distribution build. Xcode Organizer's privacy report PDF has no command-line form, so the manifests were read from the archive instead. |

Plan decision 6 had `llama.framework` declare C617.1 and 35F9.1. It declares C617.1 only, which is what
`nm -u` supports (`clock_gettime` and `sysctlbyname` aren't on Apple's list); 35F9.1 stays in the app's
manifest.

## 6. Downloads

Both model sets download in the app. Photo reading (the 1B) is 1.78 GB: the model is 672 MB and the vision
projector 1,109 MB. Most accurate on device (the 35B) is 15.50 GB: the model is 1.66 GB and the experts
13.84 GB. How downloads resume, verify, use space and are deleted is described in the
[architecture doc §4.12](architecture-plan-on-device-and-orchestration.md#412-minicpm-v-on-device). In
short, each file is hashed once, and after that a load checks only its size and modification time.

D15 (the 1B's download) needs somebody to cancel and leave the app mid-file, so it is still to do. D21 ran
on the files Codex had already copied to the phone; nothing was downloaded. The first verification had run
when the app first prepared the 35B on 15 October, so its time comes from a read-only re-hash that copies
the app's hashing loop (size, GGUF magic, SHA-256 in 1 MiB reads).

| Check | 1B (D15) | 35B (D21) |
|---|---|---|
| Offered | — | Yes: `physicalMemory` 12,262,113,280, and `accurateEligible()` is true. The switch shows in a simulator run of Settings, with "about 15.5 GB" |
| Cancel, then leave mid-file: finished files kept, download resumes on return | pending (manual) | — |
| First verification time | 1.30 s for 1.78 GB (re-hash) | **about 10 s**: 10.2 and 9.5 s for 43 files, 15.93 GB (1.6–1.7 GB/s). The app's own first check also ends with writing `.verified.json`, and on a first launch it runs beside the 1B's, so expect a little more |
| Later loads check in ≤ 3 s | Yes: `ready` 0.19 s after process start (D21) | Yes: the same 0.19 s |
| Existing folder reused without a download | — | Yes: all 43 files at their pinned sizes and SHA-256s, no staging files, nothing changed between the start and end of the gate |
| Projector removed | — | Yes: the app deleted `mmproj-model-f16.gguf` (1,119,575,840 B) during its first 35B preparation; free storage rose by 1.13 GB |
| Refused below the missing bytes + 1 GB free | — | Not exercised (it needs a download); covered by `MiniCPMArtifactsTests` |
| `isExcludedFromBackup` set | Yes (2/2 files and the folder) | Yes (43/43 files and the folder) |
| Delete frees the bytes shown | pending (manual) | pending (manual). What the button shows (`bytesOnDisk`) was 15.93 GB here, more than the 15.5 GB download, because this phone also holds the 425 MB DSpark draft; the 73 MB system prefix saved later adds to it |

## 7. Photo flow

**D16: photo reading.** A setting passes if at least 90% of the key strings are present and the note is
complete within 6 s at p50; the default is the smallest slice count that passes. Each image was prepared as
the app does it (`PendingPhoto(imageData:)`) and read with the app's reader prompt and 384-token cap. Times
run from the start of the read to the end of the note, as Swift receives it.

The 4 test fixtures (English sign, Chinese menu, receipt, poster), two runs pooled, n=32 per row (the
second run warmed from `nominal` to `fair`):

| Slices | Key strings present | Note complete, p50 | Note complete, p90 |
|---|---|---|---|
| 1 | 100% (and 100% of all the text on the cards) | 1.40 s | 1.73 s |
| 4 | 100% (and 100%) | 2.60 s | 3.00 s |

- **Through the app's path** (`MiniCPMHost.readPhoto`, 4 slices): p50 2.94 s, p90 3.52 s (n=16). The first
  photo after launch adds the 1B's load: about 2.0 s plus a 2.6 s read, about 4.6 s in all.
- Every note ended on its own (53–76 tokens) and was identical across repeats. The prompt was 123 tokens at
  1 slice and 384 at 4; its prefill took 0.31 s and 1.36 s.
- **Smaller text.** The fixtures are cards with very large text, so both settings hit the ceiling. With the
  same cards placed at a half or a third of a 12 MP frame's long edge (a stand-in for photos taken from
  further away; not part of the pass line), 1 slice found 87.5% and 62.5% of the key strings and misread
  confidently, never writing "unclear" (a receipt line read as "Latte - $1.59", "小碗包" for 小笼包). 4 slices
  stayed at 100%, at p50 2.9 s.

**Decision: 4 slices stay the default.** On the fixtures alone the rule would pick 1, but the smaller-text
variants fail it at 1 slice already at half size, and 4 slices cost only about 1.2 s more, far under 6 s.
**Pending (manual):** the 4 real scenes (a sign, a menu, a receipt and a poster at hand-held distance),
photographed in the app.

**Pending (D17, manual).** Chat questions about a photo answered by Woof and by Claude, with only the note text sent;
a poster becoming a calendar event through the permission prompt; in voice mode, the shutter working while
the app listens, echo cancellation and barge-in still working, a barge-in during a read cancelling it
without a reload, and the next turn carrying the photo.

## 8. 35B production profile and voice latency

`MiniCPMSession.production` was to stay **target-only** until D22. D22 is an ABBA run with three arms
(target-only, `baseline` and `balanced`) covering the 24-task suite, the 8 performance prompts, the
500/1,000/2,000-token prompts and the 5-turn reuse scenario. It adopts a DSpark profile only if all three of
these hold:

- no item of the 24-task suite gets worse;
- the median first text and tok/s, both measured in Swift, are each at least 10% better;
- peak resident memory is at most 4.5 GB.

**D22.** Each arm ran in its own process, in the order target-only, `baseline`, `balanced`, `balanced`,
`baseline`, target-only (the in-app ABBA couldn't finish on the phone: see [§9.3](#93-woof-versus-the-35b)).
The numbers come from each run's first 35B block, after one Woof block: per arm 16 performance rows, 48
scored rows (24 × 2), 6 long-prompt rows and 10 reuse rows. Swift decode is (generated tokens − 1) / (Swift
total − Swift first text), for replies of at least 16 tokens.

| Arm | 24-task score | First text p50 (p90), Swift | Swift decode p50 | With the reuse rows | Peak resident | Peak `phys_footprint` | Load |
|---|---|---|---|---|---|---|---|
| target-only | 48/48 | 2.30 s (2.44) | 5.87 tok/s | 5.61 tok/s | 4.41 GB | 2.78 GB | 1.7 s |
| `baseline` | 48/48 | 2.38 s (2.53) | 5.63 tok/s | 4.61 tok/s | 5.20 GB | 3.27 GB | 3.1 s |
| `balanced` | 48/48 | 2.40 s (2.57) | 6.27 tok/s | 4.93 tok/s | 5.99 GB | 4.13 GB | 3.1–3.3 s |

- **Decision 13: target-only stays.** `baseline` fails two conditions: first text 3.5% slower and decode 4.1%
  lower, and 5.20 GB resident. `balanced` fails two: first text 4.2% slower, decode +6.9% (under the +10%
  bar, and −12.2% with the reuse rows), and 5.99 GB resident. No item got worse in either. So
  `MiniCPMSession.production` stays `(.baseline, useDraft: false)`, the draft isn't hosted (gate G1b doesn't
  happen), and its manifest entry keeps no `sourceURL`: a fresh install never downloads it.
- DSpark accepted 1,270 of 2,932 draft tokens (43.3%). Its replies are **not** the same as target-only's:
  68 of 80 rows match, and the same 12 differ under both profiles (one performance prompt, and the reuse
  conversation from its first turn on). The two target-only runs match each other on 40/40.
- 5-turn reuse in total: target-only 148 s, `baseline` 201 s, `balanced` 187 s (p50).
- Even target-only peaks at 4.41 GB resident, 0.09 GB under the 4.5 GB line.
- **Heat.** Runs 2–6 started at `fair` after 505–513 s of cooling, the 35B blocks loaded at `fair` (runs 1–2)
  or `serious` (3–6), and every run ended at `serious`. Each arm has rows at both states, and their medians
  barely differ. Woof's control block was steady throughout: first text 0.158–0.167 s, 97–102 characters/s,
  22/24.

**D23: voice latency to first audio. It fails its line.** The targets were p50 ≤ 5 s and p90 ≤ 8 s for an
append and a barge-in, and ≤ 10 s for a cold start. A harness drove `MiniCPMHost.prepareAccurate` and
`replyAccurate` with the app's system prompt (366 tokens) and voice cap (256 tokens) into the app's
`Speaker` (its audio engine not started), in 6
scripted conversations of 10 requests plus 2 cold starts without a prefix file; every request started at
`nominal` or `fair`. Times run from the request to the first TTS buffer of the first sentence (where
`VoiceSession` marks first audio), nearest-rank percentiles.

| Case | First audio p50 | p90 | First text p50 | n |
|---|---|---|---|---|
| Cold start from the disk prefix (the request at the same moment as voice mode's prewarm) | 7.05 s | 8.93 s | 5.21 s | 6 |
| Cold start without a prefix file (the prewarm prefills and saves it first) | 22.75 s | 27.21 s | 21.72 s | 2 |
| Append | **7.53 s** | **8.47 s** | 4.83 s | 30 |
| Barge-in, then `replaceLastReply` | **9.26 s** | **11.75 s** | 6.95 s | 12 |
| Replaced tentative turn | 7.86 s | 11.32 s | — | 12 |
| Rebuild past the budget, voice and typed | pending | pending | | |

- **Where the time goes.** First text is 0.29 s plus 78.7 ms per fed token (§3), and every spoken turn feeds
  about 47 tokens of scaffolding and time tag before the user's words. Then the first sentence has to finish:
  2.6 s at p50 after the first text (3.6 s p90); the 4 first sentences of 100–140 characters took 4.7–7.0 s,
  and they make the p90. TTS adds 0.01–0.03 s once warm. Even speaking the first text at once would leave an
  append at about 4.8 s p50, 5.2 s p90.
- **Speech end to first audio.** Without early start add the 0.9 s end-of-turn delay and the Qwen3-ASR pass
  (about 0.2 s): about 8.6 s p50 for an append and 10.4 s for a barge-in.
- **By thermal state.** Appends: 7.59 / 8.07 s at `nominal` (n=11), 7.49 / 8.80 s at `fair` (n=19).
  Barge-ins: 7.74 / 9.52 s at `nominal` (n=5), 10.58 / 13.17 s at `fair` (n=7). A replaced tentative turn
  cancelled 0.5 s after its first text: 7.95 / 11.52 s; cancelled mid-prefill at 1.0 s: 7.24 / 8.72 s.
- **Checkpoints work as designed.** A checkpoint is 65,864,428 B (62.8 MiB; the three marks 197.6 MB, within
  the 256 MiB budget) and takes 1.8–6.7 ms. Rewinds took 6.2 ms p50 for `replaceLastReply` (13.0 ms max),
  2.3 ms for `replaceLastUserTurn` and 2.3–6.1 ms back to the base after a cancelled prefill. The cost is the
  prefill of the new tokens, not the rewind.
- Decode in Swift: 5.69 tok/s p50. Peak `phys_footprint` 2.76 GB, resident 4.44 GB.
- **Not measured** without somebody talking to the phone: live end-of-speech detection, Qwen3-ASR on real
  audio, the audible output latency of the running audio engine, and the in-app latency log. A conversation
  long enough to be rebuilt (about 3,800 tokens) wasn't reached either; the longest reached 2,300.

Under decision 31 the owner chooses: re-tune (for example a shorter turn scaffold or speaking a first
clause sooner) and run D23 again, or merge with the mode hidden. The Settings copy now gives the measured
times.

**Pending (D24, manual).** A 10-minute 35B voice conversation, then Low Power Mode. Record the battery % per
10 minutes and the time until `.serious`. Check that Woof takes over within one turn.

**Pending (D25, manual).** In airplane mode, "Remind me at 5" and "5点提醒我", then an open question. Woof should
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

Before each block the comparison waits for the previous model's memory to be released, outside every
timing. After the 35B, the phone often keeps about 2.5 GB counted for 2–90 s or more (§4, D20). With the
10 s wait of `d0387f0`, D22's runs stopped after their first 35B block, which is why it ran one arm per
launch with `--minicpm-profile`; the wait is now up to 300 s.

Here is what the reports contain:

- **Memory:** `phys_footprint`, `resident_size` and the runtime's `weights_bytes`.
- **First text:** the time the Swift stream received it, with the native time kept as `nativeFirstText`.
- **Prompt rows:** rows of about 500, 1,000 and 2,000 tokens, and a 5-turn reuse scenario for both engines.
  Woof runs with its production speculation and prefix cache.

### 9.4 Profiles and pilots

Without `--minicpm-profile`, the 35B comparison uses the production profile (`MiniCPMSession.production`),
which is target-only (D22 kept it). These profiles are available:

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
previous model's memory (waiting up to 300 s for it), then waits up to 120 s for a `nominal` thermal state. Older runs didn't cool down
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
call. On the phone it made no difference ([§4](#4-memory), D20), so the next runtime release drops it. That
release needs a new tag and checksum; until the owner publishes one, the pinned asset keeps the patch.

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
