# Model facts

Status: the workflow and script are in place. The results below are **pending the first green run** of
`.github/workflows/model-facts.yml`; until then every result cell says *pending*. Copy the tables from the
`=== BEGIN MODEL FACTS SUMMARY ===` block of that run's log.

## What is measured

`scripts/model_facts.py` reads the facts later work packages depend on for every repo in
`scripts/lab_models.txt`. It downloads no weights:

- **Hub:** whether the repo exists, its resolved revision, and its file sizes.
- **Safetensors headers** (HTTP range reads through `HfApi().get_safetensors_metadata`): tensor count and
  bytes, dtypes, `mtp.*` tensors, vision tensors, and the layout of the first `conv1d.weight`
  (`[channels, kernel, 1]` means already converted to MLX, "sanitized").
- **`config.json`:** model type, architectures, `text_config`, layer counts and types, the linear-attention
  dimensions, KV heads, `head_dim`, `mtp_num_hidden_layers`, `tie_word_embeddings`, and the quantization
  settings with their per-layer overrides.
- **EOS ids** from `config.json` and `generation_config.json`, the tokenizer's EOS token, and the ids of
  `<|im_end|>`, `<|endoftext|>`, `<tool_call>` and `<think>`. The stop set follows plan section 4.2:
  configured EOS ids (generation_config overrides config) ∪ the tokenizer's EOS ∪ `<|im_end|>` and
  `<|endoftext|>`.
- **Chat template checks**, rendered with `enable_thinking=False` and the app's tool schemas
  (`scripts/lab_prompts.json`):
  - (a) `[system, user]` with the generation prompt;
  - (b) `[system, user, assistant, user]`;
  - (c) (b) with tools;
  - (d) a tool round (assistant with `tool_calls`, then a `tool` message);
  - (e) `[system]` alone, without the generation prompt: does it raise?
  - (f) the sentinel delta of plan section 4.5: render with placeholder turns, cut at the first
    `<|im_end|>` after the placeholder reply, and compare text and tokens with the tail of the real
    render. Checked plain, with tools, and for a tool round;
  - (g) whether past replies render with a think block, and whether an as-generated reply keeps the
    generation prompt's empty `<think>\n\n</think>\n\n`;
  - (h) the system prefix: the tokens of `[system, user]` before the last `<|im_start|>` equal the encoded
    system segment, start render (a), and are followed by `<|im_start|>`.

## Decisions these facts settle

| Fact | Decides | Result |
|---|---|---|
| Any catalog checkpoint keeps `mtp.*` tensors | Whether WP42 (MTP drafter) is built | *pending* |
| A checkpoint keeps `mtp.*` **and** its conv1d is already sanitized | Whether today's app (mlx-swift-lm 3.31.4) loads it as garbage (plan F7) | *pending* |
| EOS ids and the stop set per model | The engine's stop set (WP02, WP20) | *pending* |
| (f) sentinel delta exact | Whether delta-only turn rendering is exact (WP10, WP20) | *pending* |
| (h) system prefix | Whether the `systemEnd` checkpoint and the persisted prefix work (WP20) | *pending* |
| (e) `[system]` alone raises | Whether the system prefix must come from `[system, user]` (it does, by design) | *pending* |
| (g) think block | How far as-generated history differs from a re-render (quality caveat, plan 4.5) | *pending* |
| Woof 2B exists, and its size | Whether the catalog can offer it | *pending* |

## Checkpoints

| Repo | Exists | Revision | Files GB | Tensors | Tensor GB | `mtp.*` tensors | conv1d sanitized | Vision tensors | Quantization | Garbage on 3.31.4 (F7) |
|---|---|---|---|---|---|---|---|---|---|---|
| ConwayResearch/Underdog-Woof-4B-1.1 | *pending* | | | | | | | | | |
| ConwayResearch/Underdog-Woof-2B-1.1 | *pending* | | | | | | | | | |
| mlx-community/Qwen3.5-2B-4bit | *pending* | | | | | | | | | |
| mlx-community/Qwen3.5-0.8B-MLX-4bit | *pending* | | | | | | | | | |
| mlx-community/Qwen3-4B-4bit | *pending* | | | | | | | | | |
| mlx-community/Qwen3-0.6B-4bit | *pending* | | | | | | | | | |
| mlx-community/Qwen3-1.7B-4bit | *pending* | | | | | | | | | |
| z-lab/Qwen3.5-4B-DFlash | *pending* | | | | | | | | | |

## Architecture

| Repo | model_type | text_config | Layers | Full-attention interval | Hidden | Vocab | Tied | linear k heads / v heads / k dim / v dim | KV heads | head_dim | MTP layers |
|---|---|---|---|---|---|---|---|---|---|---|---|
| *pending* | | | | | | | | | | | |

## Tokens

| Repo | EOS (config) | EOS (generation_config) | Tokenizer EOS | `<\|im_end\|>` | `<\|endoftext\|>` | `<tool_call>` | `<think>` | Stop set | Vocab hash |
|---|---|---|---|---|---|---|---|---|---|
| *pending* | | | | | | | | | |

## Chat template

| Repo | Tool-call format | (c) tools rendered | (c) tools before system text | (d) tool result role | (e) `[system]` alone raises | (f) sentinel delta exact | (g) history reply keeps a think block | (g) as-generated think block kept | (h) system prefix ok | Generation prompt |
|---|---|---|---|---|---|---|---|---|---|---|
| *pending* | | | | | | | | | | |

The full JSON (one object per repo, with the rendered texts) is printed between `=== BEGIN MODEL FACTS ===`
and `=== END MODEL FACTS ===`. Errors are recorded per repo and per section, so a gated or missing repo never
hides the others.

## Running it

- CI: pushes that touch `scripts/model_facts.py`, `scripts/lab_models.txt`, `scripts/lab_prompts.json` or the
  workflow; every Monday at 05:17 UTC; or by hand once the workflow is on the default branch.
- Locally: `pip install "huggingface_hub>=0.25" "transformers>=4.56" jinja2 tokenizers`, then
  `python -I scripts/model_facts.py --models scripts/lab_models.txt --out facts.json`. It writes `facts.json`
  and the summary `facts.md`. Set `HF_TOKEN` for gated repos.
