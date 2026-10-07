# Drafter lab

Status: the workflow and script are in place. The results below are **pending the first green run** of
`.github/workflows/drafter-lab.yml`; until then every result cell says *pending*. Copy the tables from the
`=== BEGIN DRAFTER LAB SUMMARY ===` and `=== BEGIN DFLASH SUMMARY ===` blocks of that run's log.

The lab answers, on the CI Mac's GPU and without a phone, how much speculative drafting and native tool calls
would give on Woof. It uses Python mlx-lm, which loads Qwen3.5 checkpoints with the corrected sanitize rule,
so its numbers describe the model, not today's app.

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
   `z-lab/Qwen3.5-4B-DFlash` as draft, on 6 prompts (greedy). The script reads the engine's summary event:
   tokens per verify round and accepted draft tokens per round. The two differ by one: each round also
   emits the target's own token. dflash-mlx also drafts by copying from the prompt; those rounds are counted
   in its totals and reported separately.

## Decisions these numbers settle

| Question | Rule | Result |
|---|---|---|
| Does mlx-lm load Woof 4B? | If not, the fallback model's numbers stand in | *pending* |
| Prompt-lookup acceptance | Sets WP11/WP30 defaults (Kmax, `toolsOnly`, match-length priors) | *pending* |
| Woof single-shot tool accuracy (`tools`, arguments) | Below 85 % → build the deferred selector/filler (plan section 3) | *pending* |
| DFlash tokens per round, and accepted draft tokens per round | At least 2.5 → schedule the deferred Swift DFlash port. Plan section 3 says "accepted tokens per round", WP01 item 8 says "tokens per round"; both readings are reported, and if they disagree the orchestrator decides | *pending* |
| As-generated history versus re-render | Size of the plan 4.5 quality caveat (empty think block kept) | *pending* |

## Greedy outputs

| Model | Set | Prompts | Errors | Prompt tokens | Output tokens | Stopped by EOS | Decode tok/s | Prefill tok/s | Teacher-forced argmax ≠ greedy |
|---|---|---|---|---|---|---|---|---|---|
| *pending* | | | | | | | | | |

## Drafting (primary model; chat, copy and tools, each prompt once)

| Drafter | Positions with a proposal | Speedup T=0 at K=1/2/3/4/6/8 | Best T=0 | Best T=0.7 | Tokens per drafted round K=4 (T=0 / T=0.7) | Acceptance by depth K=4, T=0 | Acceptance by depth K=4, T=0.7 |
|---|---|---|---|---|---|---|---|
| prompt_lookup | *pending* | | | | | | |
| suffix_corpus | *pending* | | | | | | |
| exemplar | *pending* | | | | | | |
| combined | *pending* | | | | | | |

Per-set tables (chat, copy, tools, tools_local) and the prompt-lookup breakdown by match length are in the
log summary.

## Tool calls (primary model)

| Set | Name accuracy | Arguments accuracy | Arguments when the name was right | Parsed |
|---|---|---|---|---|
| tools | *pending* | | | |
| tools_local | *pending* | | | |

## DFlash

| Prompt | Tokens | Rounds | Tokens per round | Accepted per round | Acceptance | Copy-spec rounds / tokens | tok/s | vs mlx-lm greedy | Same text |
|---|---|---|---|---|---|---|---|---|---|
| *pending* | | | | | | | | | |

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
