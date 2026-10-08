# Drafter lab

Results from **Drafter lab run 37705247648** (job 113077978407, commit `ca75f08`, 2026-10-07 23:59 UTC to
2026-10-08 00:18 UTC), copied from the `=== BEGIN DRAFTER LAB SUMMARY ===` and `=== BEGIN DFLASH SUMMARY ===`
blocks of its log. Environment: macos-26 runner, Metal device "Apple Paravirtual device" (`air64_v27`, 7 GiB,
recommended working set 4.67 GiB), Python 3.12.10, mlx 0.32.3, mlx-lm 0.32.0, transformers 5.19.0.

The lab measures, on the CI Mac's GPU, how much speculative drafting and native tool calls would give on Woof.
It uses Python mlx-lm, which loads Qwen3.5 checkpoints with the corrected sanitize rule, so its numbers describe
the model, not today's app. The GPU is virtualized: absolute speeds are not phone numbers, but acceptance,
accuracy and the template results do not depend on the device.

## Go/no-go facts

| Question | Rule | Result |
|---|---|---|
| Does mlx-lm load Woof 4B? | If not, the fallback model's numbers stand in | **Yes**, in 37.0 s. No fallback was needed; all numbers below are Woof 4B's. |
| Prompt-lookup acceptance | Sets WP11/WP30 defaults (Kmax, `toolsOnly`, match-length priors) | Prompt lookup alone: 1.23x pooled (K=2). By match length (K=4): length 2 accepts the first draft 43 % of the time (1.12 accepted drafts per round), length 3 94 % (2.34), length 4+ 90 % (2.53). Per set, see [Speculation policy](#what-this-means-for-the-speculation-policy). |
| Woof single-shot tool accuracy (`tools`, arguments) | Below 85 % → build the deferred selector/filler (plan section 3) | **12/12 (100 %)** names and arguments with all ten schemas. On-device subset (`tools_local`): names 11/12, arguments 10/12. **The selector/filler harness is not needed.** |
| DFlash tokens per round, and accepted draft tokens per round | At least 2.5 → schedule the deferred Swift DFlash port | **Not measured yet.** All 6 DFlash runs exited at draft load (dflash-mlx 0.1.8 could not read the draft's config, see [DFlash](#dflash)). `scripts/drafter_lab.py` now patches the config; the decision waits for the next lab run. |
| As-generated history versus re-render | Size of the plan 4.5 quality caveat (empty think block kept) | For both models, all 20 past replies differ from their re-render by the 4-token empty think block. Apart from that, 19 of 20 (Woof 4B) and 17 of 20 (Qwen3.5-0.8B) are token-identical; the rest re-tokenize differently. See [Template comparison](#template-comparison). |

## Greedy outputs

Woof 4B (`ConwayResearch/Underdog-Woof-4B-1.1`, primary). Tool-call format xml_function; stop ids
[248044, 248046]; load 37.0 s.

| Set | Prompts | Errors | Prompt tokens | Output tokens | Stopped by EOS | Decode tok/s | Prefill tok/s | Teacher-forced argmax ≠ greedy |
|---|---|---|---|---|---|---|---|---|
| chat | 12 | 0 | 356 | 34.8 | 12 | 16.6 | 105 | 7 |
| copy | 8 | 0 | 397 | 61.9 | 8 | 17.3 | 74 | 1 |
| tools | 12 | 0 | 1982 | 43.0 | 12 | 17.7 | 147 | 2 |
| tools_local | 12 | 0 | 1723 | 43.7 | 12 | 18.3 | 161 | 0 |

Qwen3.5-0.8B (`mlx-community/Qwen3.5-0.8B-MLX-4bit`, comparison). Load 7.8 s.

| Set | Prompts | Errors | Prompt tokens | Output tokens | Stopped by EOS | Decode tok/s | Prefill tok/s | Teacher-forced argmax ≠ greedy |
|---|---|---|---|---|---|---|---|---|
| chat | 12 | 0 | 356 | 63.7 | 10 | 79.8 | 760 | 12 |
| copy | 8 | 0 | 397 | 43.2 | 8 | 83.6 | 817 | 4 |
| tools | 12 | 0 | 1982 | 26.1 | 12 | 71.0 | 931 | 6 |
| tools_local | 12 | 0 | 1723 | 22.9 | 12 | 80.8 | 941 | 11 |

Woof 4B decodes at about **17 tok/s** (16.6 to 18.3) and prefills at **74 to 161 tok/s** on this paravirtual
GPU; mlx-lm runs Qwen3.5-0.8B at about 80 tok/s on the same machine. For comparison, the Swift stock
`TokenIterator` decodes Qwen3.5-0.8B at 95.9 tok/s and Qwen3-0.6B at 128.8 tok/s on the same runner type
(Local engine run 37705247676, see [model facts](model-facts.md#swift-engine-baseline-local-engine-run-37705247676)).
All Woof replies ended with EOS; replies are short (35 to 62 tokens on average).

## Drafting (Woof 4B)

Speedups are projections: tokens emitted ÷ cost, with the stock cost curve of plan section 4.7. They are not
measured wall-clock gains; the real gain depends on the device's cost curve (WP30's cost probe).

### chat

| Drafter | Positions with a proposal | Speedup T=0 at K=1/2/3/4/6/8 | Best T=0 | Best T=0.7 | Tokens per drafted round K=4 (T=0 / T=0.7) | Acceptance by depth K=4, T=0 | Acceptance by depth K=4, T=0.7 |
|---|---|---|---|---|---|---|---|
| prompt_lookup | 0.05 | 1.00 / 0.99 / 0.99 / 0.98 / 0.96 / 0.94 | K=1: 1.00 | K=1: 1.00 | 1.19 / 1.20 | 0.19 0.00 0.00 0.00 | 0.20 0.00 0.00 0.00 |
| prompt_lookup_n2 | 0.05 | 1.00 / 0.99 / 0.99 / 0.98 / 0.96 / 0.94 | K=1: 1.00 | K=1: 1.00 | 1.19 / 1.20 | 0.19 0.00 0.00 0.00 | 0.20 0.00 0.00 0.00 |
| prompt_lookup_n3 | 0.01 | 1.00 / 1.00 / 1.00 / 0.99 / 0.99 / 0.99 | K=1: 1.00 | K=1: 1.00 | 1.00 / 1.00 | 0.00 0.00 0.00 0.00 | 0.00 0.00 0.00 0.00 |
| prompt_lookup_n4 | 0.00 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | K=1: 1.00 | K=1: 1.00 | – / – | – – – – | – – – – |
| suffix_corpus | 0.02 | 1.01 / 1.00 / 1.00 / 0.99 / 0.98 / 0.97 | K=1: 1.01 | K=1: 1.01 | 1.43 / 1.43 | 0.43 0.00 0.00 0.00 | 0.43 0.00 0.00 0.00 |
| exemplar | 0.00 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | K=1: 1.00 | K=1: 1.00 | – / – | – – – – | – – – – |
| combined | 0.07 | 1.01 / 1.00 / 0.99 / 0.97 / 0.95 / 0.93 | K=1: 1.01 | K=1: 1.01 | 1.29 / 1.29 | 0.29 0.00 0.00 0.00 | 0.29 0.00 0.00 0.00 |

### copy

| Drafter | Positions with a proposal | Speedup T=0 at K=1/2/3/4/6/8 | Best T=0 | Best T=0.7 | Tokens per drafted round K=4 (T=0 / T=0.7) | Acceptance by depth K=4, T=0 | Acceptance by depth K=4, T=0.7 |
|---|---|---|---|---|---|---|---|
| prompt_lookup | 0.63 | 1.33 / 1.39 / 1.38 / 1.33 / 1.22 / 1.13 | K=2: 1.39 | K=2: 1.39 | 3.18 / 3.17 | 0.77 0.65 0.42 0.35 | 0.77 0.65 0.42 0.35 |
| prompt_lookup_n2 | 0.63 | 1.31 / 1.38 / 1.37 / 1.31 / 1.20 / 1.13 | K=2: 1.38 | K=2: 1.38 | 3.09 / 3.08 | 0.75 0.61 0.42 0.35 | 0.74 0.61 0.41 0.34 |
| prompt_lookup_n3 | 0.53 | 1.27 / 1.31 / 1.31 / 1.27 / 1.21 / 1.14 | K=3: 1.31 | K=2: 1.31 | 3.26 / 3.24 | 0.80 0.61 0.47 0.42 | 0.80 0.61 0.46 0.41 |
| prompt_lookup_n4 | 0.44 | 1.22 / 1.25 / 1.23 / 1.22 / 1.16 / 1.11 | K=2: 1.25 | K=2: 1.25 | 3.22 / 3.20 | 0.71 0.63 0.49 0.42 | 0.71 0.63 0.48 0.41 |
| suffix_corpus | 0.23 | 1.08 / 1.09 / 1.07 / 1.05 / 1.00 / 0.96 | K=2: 1.09 | K=2: 1.09 | 2.43 / 2.43 | 0.55 0.43 0.23 0.21 | 0.56 0.43 0.23 0.21 |
| exemplar | 0.00 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | K=1: 1.00 | K=1: 1.00 | – / – | – – – – | – – – – |
| combined | 0.69 | 1.37 / 1.46 / 1.45 / 1.40 / 1.28 / 1.18 | K=2: 1.46 | K=2: 1.46 | 3.31 / 3.29 | 0.80 0.69 0.45 0.38 | 0.80 0.68 0.45 0.38 |

### tools (all ten schemas)

| Drafter | Positions with a proposal | Speedup T=0 at K=1/2/3/4/6/8 | Best T=0 | Best T=0.7 | Tokens per drafted round K=4 (T=0 / T=0.7) | Acceptance by depth K=4, T=0 | Acceptance by depth K=4, T=0.7 |
|---|---|---|---|---|---|---|---|
| prompt_lookup | 0.76 | 1.34 / 1.33 / 1.29 / 1.21 / 1.02 / 0.89 | K=1: 1.34 | K=1: 1.34 | 2.51 / 2.50 | 0.58 0.46 0.30 0.18 | 0.58 0.46 0.29 0.18 |
| prompt_lookup_n2 | 0.76 | 1.22 / 1.22 / 1.22 / 1.08 / 0.93 / 0.82 | K=1: 1.22 | K=1: 1.22 | 2.11 / 2.10 | 0.43 0.34 0.26 0.11 | 0.42 0.34 0.25 0.11 |
| prompt_lookup_n3 | 0.58 | 1.29 / 1.37 / 1.32 / 1.29 / 1.16 / 1.05 | K=2: 1.37 | K=2: 1.37 | 3.11 / 3.08 | 0.90 0.64 0.32 0.29 | 0.90 0.63 0.31 0.27 |
| prompt_lookup_n4 | 0.47 | 1.23 / 1.23 / 1.20 / 1.18 / 1.07 / 0.99 | K=1: 1.23 | K=1: 1.23 | 2.79 / 2.75 | 0.82 0.40 0.33 0.29 | 0.81 0.39 0.31 0.28 |
| suffix_corpus | 0.70 | 1.40 / 1.47 / 1.45 / 1.42 / 1.25 / 1.15 | K=2: 1.47 | K=2: 1.47 | 3.21 / 3.17 | 0.76 0.65 0.49 0.38 | 0.75 0.64 0.48 0.37 |
| exemplar | 0.62 | 1.39 / 1.42 / 1.52 / 1.54 / 1.46 / 1.36 | K=4: 1.54 | K=4: 1.53 | 4.06 / 4.03 | 0.92 0.88 0.86 0.61 | 0.91 0.87 0.85 0.60 |
| combined | 0.92 | 1.52 / 1.60 / 1.63 / 1.64 / 1.41 / 1.24 | K=4: 1.64 | K=4: 1.63 | 3.21 / 3.20 | 0.80 0.66 0.51 0.38 | 0.79 0.66 0.51 0.37 |

### tools_local (on-device subset)

| Drafter | Positions with a proposal | Speedup T=0 at K=1/2/3/4/6/8 | Best T=0 | Best T=0.7 | Tokens per drafted round K=4 (T=0 / T=0.7) | Acceptance by depth K=4, T=0 | Acceptance by depth K=4, T=0.7 |
|---|---|---|---|---|---|---|---|
| prompt_lookup | 0.75 | 1.34 / 1.33 / 1.30 / 1.22 / 1.02 / 0.89 | K=1: 1.34 | K=1: 1.34 | 2.54 / 2.54 | 0.60 0.47 0.30 0.18 | 0.60 0.47 0.30 0.18 |
| prompt_lookup_n2 | 0.75 | 1.22 / 1.22 / 1.21 / 1.09 / 0.92 / 0.81 | K=1: 1.22 | K=1: 1.22 | 2.14 / 2.14 | 0.46 0.35 0.26 0.10 | 0.45 0.35 0.26 0.10 |
| prompt_lookup_n3 | 0.58 | 1.28 / 1.38 / 1.32 / 1.28 / 1.14 / 1.03 | K=2: 1.38 | K=2: 1.37 | 3.09 / 3.08 | 0.91 0.64 0.30 0.26 | 0.91 0.64 0.30 0.25 |
| prompt_lookup_n4 | 0.47 | 1.23 / 1.23 / 1.18 / 1.17 / 1.06 / 0.97 | K=1: 1.23 | K=1: 1.23 | 2.75 / 2.75 | 0.84 0.37 0.29 0.28 | 0.82 0.38 0.29 0.28 |
| suffix_corpus | 0.85 | 1.57 / 1.68 / 1.67 / 1.61 / 1.42 / 1.26 | K=2: 1.68 | K=2: 1.68 | 3.37 / 3.32 | 0.81 0.67 0.52 0.45 | 0.80 0.66 0.50 0.43 |
| exemplar | 0.63 | 1.39 / 1.43 / 1.53 / 1.54 / 1.45 / 1.37 | K=4: 1.54 | K=4: 1.53 | 4.05 / 4.00 | 0.91 0.88 0.86 0.59 | 0.89 0.87 0.85 0.59 |
| combined | 0.93 | 1.52 / 1.62 / 1.62 / 1.66 / 1.42 / 1.29 | K=4: 1.66 | K=4: 1.65 | 3.23 / 3.22 | 0.80 0.68 0.52 0.38 | 0.79 0.68 0.52 0.38 |

### all (chat, copy, tools; each prompt once)

| Drafter | Positions with a proposal | Speedup T=0 at K=1/2/3/4/6/8 | Best T=0 | Best T=0.7 | Tokens per drafted round K=4 (T=0 / T=0.7) | Acceptance by depth K=4, T=0 | Acceptance by depth K=4, T=0.7 |
|---|---|---|---|---|---|---|---|
| prompt_lookup | 0.51 | 1.22 / 1.23 / 1.21 / 1.16 / 1.06 / 0.98 | K=2: 1.23 | K=2: 1.23 | 2.68 / 2.66 | 0.63 0.50 0.33 0.24 | 0.63 0.50 0.33 0.23 |
| prompt_lookup_n2 | 0.51 | 1.17 / 1.19 / 1.18 / 1.11 / 1.02 / 0.94 | K=2: 1.19 | K=2: 1.19 | 2.39 / 2.38 | 0.52 0.42 0.30 0.19 | 0.52 0.42 0.30 0.18 |
| prompt_lookup_n3 | 0.39 | 1.18 / 1.22 / 1.20 / 1.18 / 1.12 / 1.06 | K=2: 1.22 | K=2: 1.22 | 3.14 / 3.12 | 0.84 0.61 0.38 0.34 | 0.84 0.61 0.37 0.33 |
| prompt_lookup_n4 | 0.32 | 1.15 / 1.16 / 1.14 / 1.13 / 1.08 / 1.03 | K=2: 1.16 | K=2: 1.16 | 2.98 / 2.94 | 0.77 0.50 0.40 0.35 | 0.77 0.49 0.39 0.34 |
| suffix_corpus | 0.34 | 1.15 / 1.17 / 1.16 / 1.14 / 1.07 / 1.02 | K=2: 1.17 | K=2: 1.17 | 2.92 / 2.90 | 0.69 0.56 0.40 0.32 | 0.68 0.56 0.39 0.31 |
| exemplar | 0.23 | 1.11 / 1.12 / 1.14 / 1.15 / 1.13 / 1.11 | K=4: 1.15 | K=4: 1.14 | 4.06 / 4.03 | 0.92 0.88 0.86 0.61 | 0.91 0.87 0.85 0.60 |
| combined | 0.59 | 1.28 / 1.32 / 1.32 / 1.30 / 1.20 / 1.11 | K=2: 1.32 | K=2: 1.32 | 3.10 / 3.09 | 0.76 0.62 0.45 0.35 | 0.76 0.62 0.45 0.35 |

### Prompt lookup by match length (chat, copy, tools; K=4)

| Match length | Rounds | Mean accepted T=0 | First draft accepted T=0 | Mean accepted T=0.7 | First draft accepted T=0.7 |
|---|---|---|---|---|---|
| 2 | 162 | 1.12 | 0.43 | 1.11 | 0.43 |
| 3 | 32 | 2.34 | 0.94 | 2.29 | 0.94 |
| 4+ | 81 | 2.53 | 0.90 | 2.52 | 0.90 |

Qwen3.5-0.8B, for comparison, pooled over chat, copy and tools:

| Drafter | Positions with a proposal | Speedup T=0 at K=1/2/3/4/6/8 | Best T=0 | Best T=0.7 | Tokens per drafted round K=4 (T=0 / T=0.7) | Acceptance by depth K=4, T=0 | Acceptance by depth K=4, T=0.7 |
|---|---|---|---|---|---|---|---|
| prompt_lookup | 0.43 | 1.20 / 1.21 / 1.20 / 1.18 / 1.10 / 1.06 | K=2: 1.21 | K=2: 1.20 | 2.94 / 2.86 | 0.61 0.52 0.46 0.41 | 0.59 0.50 0.43 0.38 |
| prompt_lookup_n2 | 0.43 | 1.20 / 1.21 / 1.20 / 1.18 / 1.10 / 1.06 | K=2: 1.21 | K=2: 1.20 | 2.92 / 2.84 | 0.61 0.51 0.45 0.40 | 0.59 0.50 0.42 0.38 |
| prompt_lookup_n3 | 0.34 | 1.17 / 1.19 / 1.20 / 1.20 / 1.18 / 1.16 | K=3: 1.20 | K=3: 1.19 | 3.67 / 3.59 | 0.81 0.72 0.66 0.62 | 0.81 0.69 0.63 0.59 |
| prompt_lookup_n4 | 0.30 | 1.16 / 1.18 / 1.19 / 1.19 / 1.18 / 1.17 | K=3: 1.19 | K=3: 1.18 | 4.05 / 3.92 | 0.88 0.81 0.76 0.72 | 0.85 0.78 0.73 0.67 |
| suffix_corpus | 0.22 | 1.06 / 1.06 / 1.03 / 0.99 / 0.93 / 0.88 | K=1: 1.06 | K=1: 1.06 | 1.87 / 1.81 | 0.54 0.26 0.04 0.03 | 0.53 0.24 0.02 0.02 |
| exemplar | 0.01 | 1.00 / 1.00 / 1.01 / 1.01 / 1.01 / 1.01 | K=3: 1.01 | K=3: 1.01 | 4.67 / 4.67 | 1.00 1.00 1.00 1.00 | 1.00 1.00 1.00 1.00 |
| combined | 0.49 | 1.22 / 1.22 / 1.20 / 1.17 / 1.08 / 1.03 | K=1: 1.22 | K=1: 1.21 | 2.73 / 2.63 | 0.56 0.46 0.41 0.36 | 0.54 0.43 0.37 0.33 |

| Match length | Rounds | Mean accepted T=0 | First draft accepted T=0 | Mean accepted T=0.7 | First draft accepted T=0.7 |
|---|---|---|---|---|---|
| 2 | 133 | 1.03 | 0.42 | 0.94 | 0.40 |
| 3 | 5 | 0.80 | 0.20 | 0.82 | 0.23 |
| 4+ | 76 | 3.62 | 0.97 | 3.55 | 0.96 |

### What this means for the speculation policy

- **Chat: about 1.0x.** No drafter helps (best 1.01x at K=1). Prompt lookup proposes at only 5 % of
  positions, and its first draft is accepted 19 % of the time. Voice chat replies are short and do not copy
  the prompt, so drafting on plain chat turns should stay off, or cost nothing when no long match exists.
- **Copy: 1.46x** with the combined drafters at K=2 (prompt lookup alone 1.39x at K=2). Acceptance falls
  with depth (0.80, 0.69, 0.45, 0.38 at K=4), so K=2 to 3 is the sweet spot.
- **Tools: 1.64x** with the combined drafters at K=4 (1.66x on `tools_local`). The exemplar tool skeletons
  carry it: 0.92, 0.88, 0.86, 0.61 acceptance by depth and 4.06 tokens per drafted round at K=4.
- **Pooled: 1.32x** at K=2 to 3 with the combined drafters.
- **K of 6 or 8 is never the best K** for Woof under the stock cost curve, so Kmax = 4 unless the device's
  cost curve is flatter.
- **Match length matters.** Length-2 prompt-lookup matches accept the first draft only 43 % of the time
  (1.12 accepted drafts per round); length 3 and longer accept it about 90 % of the time (2.3 to 2.5 accepted
  drafts per round). Draft only from matches of 3 or more tokens, or cap length-2 matches at K=1.
- **T=0.7 changes almost nothing:** the best T=0.7 speedups are within 0.01 of T=0 (and they are a lower
  bound), so the app's sampling settings do not erode acceptance.
- Taken together: speculation pays where the reply copies its input or emits a tool call. A tools-only default
  captures the largest gain (1.64x); enabling prompt lookup on any turn whose draft comes from a match of 3 or
  more tokens adds the copy gain (1.46x) and costs chat almost nothing, since chat rarely has such matches.

## Tool calls (Woof 4B)

### tools (all ten schemas)

Name accuracy 1.00, arguments 1.00 (1.00 when the name was right), parsed 12/12.

| Prompt | Text | Expected | Called | Name | Arguments | Arguments given | Output tokens |
|---|---|---|---|---|---|---|---|
| tool-01 | Remind me at 5 to call mum. | create_reminder | create_reminder | yes | yes | {"title": "Call mum", "due": "2026-10-07T17:00"} | 53 |
| tool-02 | Set a timer for ten minutes for the pasta. | set_timer | set_timer | yes | yes | {"seconds": "600", "label": "Pasta"} | 39 |
| tool-03 | What's on my calendar tomorrow? | list_events | list_events | yes | yes | {"start": "2026-10-08T00:00", "end": "2026-10-09T00:00"} | 65 |
| tool-04 | Put a dentist appointment in my calendar for Friday at 3pm. | create_event | create_event | yes | yes | {"title": "Dentist appointment", "start": "2026-10-09T15:00"} | 53 |
| tool-05 | Which of my reminders are due today? | list_reminders | list_reminders | yes | yes | {"scope": "today"} | 26 |
| tool-06 | Add buy milk to my to-do list. | create_reminder | create_reminder | yes | yes | {"title": "Buy milk"} | 26 |
| tool-07 | 提醒我明天早上八点去取快递。 | create_reminder | create_reminder | yes | yes | {"title": "去取快递", "due": "2026-10-08T08:00"} | 54 |
| tool-08 | 帮我设一个二十分钟的计时器。 | set_timer | set_timer | yes | yes | {"seconds": "1200"} | 28 |
| tool-09 | 我今天下午有什么安排？ | list_events | list_events | yes | yes | {"start": "2026-10-07T16:00", "end": "2026-10-07T24:00"} | 65 |
| tool-10 | 下周一上午十点和王经理开会，帮我加到日历里。 | create_event | create_event | yes | yes | {"title": "和王经理开会", "start": "2026-10-12T10:00", "duration_minutes": "60"} | 66 |
| tool-11 | 有哪些提醒已经过期了？ | list_reminders | list_reminders | yes | yes | {"scope": "overdue"} | 27 |
| tool-12 | 我的计时器还剩多少时间？ | list_timers | list_timers | yes | yes | {} | 14 |

### tools_local (on-device subset)

Name accuracy 0.92, arguments 0.83 (0.91 when the name was right), parsed 12/12. 1 prompt(s) ask for a tool outside the on-device subset and expect handoff_to_cloud instead.

| Prompt | Text | Expected | Called | Name | Arguments | Arguments given | Output tokens |
|---|---|---|---|---|---|---|---|
| tool-01 | Remind me at 5 to call mum. | create_reminder | create_reminder | yes | no | {"title": "Call mum", "due": "2026-10-08T17:00"} | 53 |
| tool-02 | Set a timer for ten minutes for the pasta. | set_timer | set_timer | yes | yes | {"seconds": "600", "label": "Pasta"} | 39 |
| tool-03 | What's on my calendar tomorrow? | list_events | list_events | yes | yes | {"start": "2026-10-08T00:00", "end": "2026-10-09T00:00"} | 65 |
| tool-04 | Put a dentist appointment in my calendar for Friday at 3pm. | create_event | create_event | yes | yes | {"title": "Dentist", "start": "2026-10-09T15:00"} | 52 |
| tool-05 | Which of my reminders are due today? | list_reminders | list_reminders | yes | yes | {"scope": "today"} | 26 |
| tool-06 | Add buy milk to my to-do list. | create_reminder | create_reminder | yes | yes | {"title": "Buy milk"} | 26 |
| tool-07 | 提醒我明天早上八点去取快递。 | create_reminder | create_reminder | yes | yes | {"title": "去取快递", "due": "2026-10-08T08:00"} | 54 |
| tool-08 | 帮我设一个二十分钟的计时器。 | set_timer | set_timer | yes | yes | {"seconds": "1200", "label": "计时器"} | 40 |
| tool-09 | 我今天下午有什么安排？ | list_events | list_events | yes | yes | {"start": "2026-10-07T16:00", "end": "2026-10-07T24:00"} | 65 |
| tool-10 | 下周一上午十点和王经理开会，帮我加到日历里。 | create_event | create_event | yes | yes | {"title": "和王经理开会", "start": "2026-10-12T10:00"} | 52 |
| tool-11 | 有哪些提醒已经过期了？ | list_reminders | list_reminders | yes | yes | {"scope": "overdue"} | 27 |
| tool-12 | 我的计时器还剩多少时间？ | handoff_to_cloud (not list_timers) | set_timer | no | no | {"seconds": "1"} | 25 |

- **The selector/filler harness is not needed:** Woof 4B called the right tool with the right arguments for
  12/12 prompts with the full tool set, well above the 85 % bar.
- **On the on-device subset, Woof did not hand off.** tool-12 ("how much time is left on my timer") needs
  `list_timers`, which the subset lacks. Woof called `set_timer` with `{"seconds": "1"}` instead of
  `handoff_to_cloud`. The engine must not rely on the model to hand off for tools outside the subset, and a
  side-effecting call that does not fit the request should not run unchecked. tool-01 also scored wrong on
  `tools_local`: "remind me at 5", said at 16:05, got a due time of tomorrow 17:00 instead of today.
- **Numbers arrive as strings.** The xml-function format carries every parameter as text, and Woof emits
  `"seconds": "600"`, `"seconds": "1200"`, `"duration_minutes": "60"`. Clients must coerce arguments to the
  JSON-schema types (integer, number, boolean) before running a tool.
- Qwen3.5-0.8B is not usable for native tool calls with this prompt: it parsed 1/12 calls on `tools` (only
  `list_timers`) and 0/12 on `tools_local`.

## Template comparison

Woof 4B: 20 past replies re-rendered: canonical equals as-generated for 0; the template drops the generation prompt's empty think block for 20; reply tokens unchanged (ignoring the think block) for 19. Other differences: {"re-tokenized differently": 1}.

- chat-01: as generated "<think>\n\n</think>\n\nHello! I can help you with almost"…, canonical "Hello! I can help you with almost anything, from planning"…; first difference at token 0: ['<think>', 'ĊĊ', '</think>', 'ĊĊ', 'Hello', '!', 'ĠI', 'Ġcan'] vs ['Hello', '!', 'ĠI', 'Ġcan', 'Ġhelp', 'Ġyou', 'Ġwith', 'Ġalmost']
- chat-02: as generated "<think>\n\n</think>\n\nTry to keep your bedroom cool, around"…, canonical "Try to keep your bedroom cool, around 18 to"…; first difference at token 0: ['<think>', 'ĊĊ', '</think>', 'ĊĊ', 'Try', 'Ġto', 'Ġkeep', 'Ġyour'] vs ['Try', 'Ġto', 'Ġkeep', 'Ġyour', 'Ġbedroom', 'Ġcool', ',', 'Ġaround']

Qwen3.5-0.8B: 20 past replies re-rendered: canonical equals as-generated for 0; the template drops the generation prompt's empty think block for 20; reply tokens unchanged (ignoring the think block) for 17. Other differences: {"re-tokenized differently": 3}.

The qwen3_5 template drops the generation prompt's empty `<think>\n\n</think>\n\n` (4 tokens) from past
replies. A session that keeps as-generated history in its cache therefore holds 4 extra tokens per past reply
compared with a fresh render, and occasionally a reply re-tokenizes differently. This is the plan 4.5 caveat;
its effect on answer quality is device item D6.

## DFlash

Run 37705247648, dflash-mlx 0.1.8, target Woof 4B (snapshot `5decb824793e`), draft `z-lab/Qwen3.5-4B-DFlash`.
The step is `continue-on-error`, so the job stayed green.

| Prompt | Tokens | Rounds | Tokens per round | Accepted per round | Acceptance | Copy-spec rounds / tokens | tok/s | vs mlx-lm greedy | Same text | Error |
|---|---|---|---|---|---|---|---|---|---|---|
| chat-01 | – | – | – | – | – | – | – | – | – | exit 1 |
| chat-05 | – | – | – | – | – | – | – | – | – | exit 1 |
| copy-01 | – | – | – | – | – | – | – | – | – | exit 1 |
| copy-03 | – | – | – | – | – | – | – | – | – | exit 1 |
| copy-06 | – | – | – | – | – | – | – | – | – | exit 1 |
| tool-01 | – | – | – | – | – | – | – | – | – | exit 1 |

Every run failed while loading the draft:
`TypeError: DFlashDraftModelArgs.__init__() missing 2 required positional arguments: 'rope_theta' and 'block_size'`
(`dflash_mlx/model.py` `DFlashDraftModelArgs.from_dict`, reached through `runtime/loading.py`
`load_draft_bundle` → `mlx_lm.utils.load_model`). dflash-mlx 0.1.8 expects both keys at the top level of the
draft's config.json; z-lab/Qwen3.5-4B-DFlash keeps them as `rope_parameters.rope_theta` (10000000) and
`dflash_config.block_size` (16).

**Fix (in `scripts/drafter_lab.py`):** `--dflash` now downloads the draft snapshot and, when keys are
missing, writes a patched config.json into a temporary directory that symlinks every other snapshot file. It
copies `rope_parameters.rope_theta` to `rope_theta` and `dflash_config.block_size` to `block_size`, keeps every
other key, and passes that directory as `--draft`. It would also map a non-default rope type to `rope_scaling`,
and turn sliding layers into full attention if `sliding_window` were null (the transformers meaning of a null
window), since dflash-mlx rejects sliding layers without a positive window. The patch is recorded in
`dflash.json` (`draft_config_patch`) and in the summary. The wrapper's error capture and the plain-CLI fallback
are unchanged.

**Status:** DFlash tokens per round, and with them the Swift DFlash port decision, wait for the next run of the
lab (push a change to the lab files, or a commit whose message contains `[ci lab]`).

## Method

`scripts/drafter_lab.py` runs Woof 4B (`ConwayResearch/Underdog-Woof-4B-1.1`). If mlx-lm cannot load it, the
error is recorded and `mlx-community/Qwen3.5-2B-4bit` takes its place. `mlx-community/Qwen3.5-0.8B-MLX-4bit`
runs too, for comparison.

1. **Prompts** (`scripts/lab_prompts.json`): 12 voice-style chat prompts (the 8 of `LocalBenchmark.prompts`
   plus 4), 8 copy-heavy prompts (read back, fix typos, to JSON, polite rewrite; English and Chinese), and
   12 tool prompts with an expected call. Every prompt is `[system, user]` with the app's system prompt and
   `PromptBuilder`'s `<context>` tag, rendered with the generation prompt and `enable_thinking=False`. Tool
   prompts run twice: with all ten tool schemas of plan section 5.3 (`tools`) and with the on-device subset
   (`tools_local`). In `tools_local`, a prompt whose tool is outside the subset (tool-12, `list_timers`)
   expects `handoff_to_cloud` instead, with any reason.
2. **Greedy outputs**, at most 200 tokens each.
3. **Teacher forcing.** One chunked pass per prompt over its greedy output keeps, for each position, only
   the top 20 of the app's sampling distribution (top-p 0.8 on the full distribution, intersected with top-k
   20, then temperature 0.7), never full-vocabulary logits for every position.
4. **Acceptance simulation** for K ∈ {1, 2, 3, 4, 6, 8}, along the greedy output:
   - drafters: prompt lookup (n-grams 2 to 4, and each n alone), a suffix corpus built from the previous
     prompts' outputs, exemplar tool-call skeletons in the model's format, and their combination;
   - T = 0: a round accepts the drafts that match the greedy output;
   - T = 0.7: a round is credited with the expected number of accepted drafts, the running product of the
     drafted tokens' sampling probabilities. Drafts after the first one that leaves the greedy path have no
     recorded distribution and count as rejected, so these numbers are a lower bound;
   - projected speedup = tokens emitted ÷ cost, with the default cost curve of plan section 4.7
     (`{1: 1.0, 2: 1.05, 3: 1.35, 4: 1.63, 5: 1.95, 6: 2.3, 8: 3.07, 9: 3.4}`, interpolated);
   - the pooled `all` figures cover chat, copy and tools, each prompt once. `tools_local` replays the tool
     prompts, so it is reported on its own and left out of the pool.
5. **Tool-call accuracy.** The first `<tool_call>` block (xml-function or JSON) is scored against the
   prompt's `expect`: the name must match, and each listed argument must match (dates by day or by minute,
   integers by value, text by its words). "Arguments" accuracy means the name and every key argument are
   right.
6. **Template comparison.** Each past reply is re-rendered by the chat template and compared token by token
   with the tokens the model actually saw (the generation prompt plus the generated reply).
7. **DFlash** (optional step): `dflash generate` from `dflash-mlx`, with the Woof snapshot as target and
   `z-lab/Qwen3.5-4B-DFlash` as draft (config patched as above), on 6 prompts (greedy). The script reads the
   engine's summary event: tokens per verify round and accepted draft tokens per round. The two differ by one:
   each round also emits the target's own token. dflash-mlx also drafts by copying from the prompt; those
   rounds are counted in its totals and reported separately.

## Running it

- CI: the `Drafter lab` workflow starts on every push, but its macOS job runs only when
  `scripts/drafter_lab.py`, `scripts/lab_prompts.json`, `scripts/lab_models.txt` or the workflow changed, when
  the head commit message contains `[ci lab]`, or when started by hand. The job's time limit is 90 minutes.
  For an ordinary push, "changed" means the diff between the old and the new tip. For a new branch or a
  force push, it means the files of the commits that push introduced (the event payload's `distinct`
  commits), so a branch created from, or rebased onto, the integration branch does not start the lab for
  lab changes it only carries.
- Locally on an Apple silicon Mac:
  ```
  python3 -m venv .venv && .venv/bin/pip install mlx mlx-lm "huggingface_hub[hf_xet]"
  .venv/bin/python -I scripts/drafter_lab.py --prompts scripts/lab_prompts.json --out lab.json
  .venv/bin/pip install dflash-mlx
  .venv/bin/python -I scripts/drafter_lab.py --dflash --prompts scripts/lab_prompts.json --out dflash.json --baseline lab.json
  ```
  `--limit N` runs only N prompts per set. Each run writes its JSON and a Markdown summary next to it.
