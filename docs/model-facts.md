# Model facts

Results from **Model facts run 37705247655** (job 113077953977, commit `ca75f08`, 2026-10-07 23:59 UTC;
huggingface_hub 1.33.0, transformers 5.19.0, tokenizers 0.23.2), copied from the
`=== BEGIN MODEL FACTS SUMMARY ===` block of its log. The Swift baseline at the end comes from
**Local engine run 37705247676** (job 113077954381). The iOS workflow (run 37705247621) and every other job on
that commit were green.

## Go/no-go facts

| Fact | Decides | Result |
|---|---|---|
| Any catalog checkpoint keeps `mtp.*` tensors | Whether WP42 (MTP drafter) is built | **No.** All 8 repos have 0 `mtp.*` tensors. Woof 4B, Qwen3.5-2B and Qwen3.5-0.8B still declare `mtp_num_hidden_layers: 1` in config.json, but the weights were stripped. **WP42 is skipped.** |
| A checkpoint keeps `mtp.*` **and** its conv1d is already sanitized | Whether today's app (mlx-swift-lm 3.31.4) loads it as garbage (plan F7) | **No.** Woof 4B's conv1d is sanitized (`[8192, 4, 1]`, all 24 alike), but with no `mtp.*` key the 3.31.4 norm shift never fires. Woof 4B is not garbage-loaded today. |
| EOS ids and the stop set per model | The engine's stop set (WP02, WP20) | Woof 4B and Qwen3.5: `[248044, 248046]` (`<\|endoftext\|>`, `<\|im_end\|>`). Qwen3: `[151643, 151645]`. Woof 2B: `[1, 130073]` (`</s>`, `<\|im_end\|>`). See [Tokens](#tokens). |
| (f) sentinel delta exact | Whether delta-only turn rendering is exact (WP10, WP20) | **Yes** for every model with a template, in all three variants (plain, with tools, tool round). Woof 4B: 56-token delta plain and with tools, 60 for a tool round; text and tokens match. |
| (h) system prefix | Whether the `systemEnd` checkpoint and the persisted prefix work (WP20) | **Yes** for every model. Woof 4B: 20 tokens without tools, 1646 with the 10 tool schemas. |
| (e) `[system]` alone raises | Whether the system prefix must come from `[system, user]` | **Yes** for the qwen3_5 templates (Woof 4B, Qwen3.5): `TemplateError: No user query found in messages.` No for Qwen3 and Woof 2B. The engine renders the prefix from `[system, user]`, as designed. |
| (g) think block | How far as-generated history differs from a re-render (plan 4.5) | Qwen3 and qwen3_5 templates **drop** the think block from past replies, so the generation prompt's empty `<think>\n\n</think>\n\n` that the model saw is not in a re-render. Woof 2B keeps it. Measured effect in the [drafter lab](drafter-lab.md#template-comparison). |
| Woof 2B exists, and its size | Whether the catalog can offer it | **Yes**, 1.43 GB (4-bit g64), but it is a **llama** model (`LlamaForCausalLM`, 42 layers, vocabulary 130560, untied embeddings), not a Qwen3.5 hybrid. Its tokenizer differs from Woof 4B's (vocabulary hash `1043a49e871843d1` vs `5660eab8ed1d73c3`), so **it cannot draft for Woof 4B**, and it would load through the stock llama path, not the `HybridQwen35` fork. |

Template quirks worth knowing:

- **qwen3_5 renders tools before the system text.** With tools, the system turn starts with the `# Tools` block
  and the app's system text comes last. A cached system prefix is therefore keyed on the tool set first:
  changing the tool set invalidates the whole prefix, changing only the system text invalidates its tail.
  Qwen3 and Woof 2B append the tools after the system text.
- Tool results render as a `user` turn with `<tool_response>…</tool_response>` in every template.
- Tool-call formats: qwen3_5 uses xml-function (`<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>…`),
  Qwen3 uses JSON inside `<tool_call>`, and Woof 2B uses `<function name="…"><param name="…">…</param></function>`
  (the script reports it as "unknown").
- In Woof 4B's tokenizer `<tool_call>` (248058) and `<think>` (248068) are added tokens but not special
  tokens; `<|im_start|>` (248045), `<|im_end|>` and `<|endoftext|>` are special.
- Every chat template's generation prompt with `enable_thinking=False` is `<|im_start|>assistant\n<think>\n\n</think>\n\n`
  (Woof 4B ids `[248045, 74455, 198, 248068, 271, 248069, 271]`).
- z-lab/Qwen3.5-4B-DFlash ships no chat template and only a stub tokenizer (length 1), so its token rows
  below are not meaningful. Its config keeps `rope_theta` (10000000) inside `rope_parameters` and
  `block_size` (16) inside `dflash_config` (also `mask_token_id` 248077, `target_layer_ids`
  `[1, 5, 9, 13, 17, 21, 25, 29]`, `num_target_layers` 32, 5 sliding-attention layers and 1 full). It has
  no `lm_head`; it projects through the target's. dflash-mlx 0.1.8 needs `rope_theta` and `block_size` at
  the top level, which is why the first DFlash run failed (see the drafter lab).

## Checkpoints

| Repo | Exists | Revision | Files GB | Tensors | Tensor GB | `mtp.*` tensors | conv1d sanitized | Vision tensors | Quantization | Garbage on 3.31.4 (F7) |
|---|---|---|---|---|---|---|---|---|---|---|
| ConwayResearch/Underdog-Woof-4B-1.1 | yes | 5decb824793e | 2.39 | 924 | 2.37 | 0 | yes | 0 | 4-bit g64 | no |
| ConwayResearch/Underdog-Woof-2B-1.1 | yes | 73c8b5ca3abf | 1.43 | 973 | 1.42 | 0 | – | 0 | 4-bit g64 | – |
| mlx-community/Qwen3.5-2B-4bit | yes | 674aaa7240b9 | 1.75 | 991 | 1.72 | 0 | yes | 297 | 4-bit g64 | no |
| mlx-community/Qwen3.5-0.8B-MLX-4bit | yes | 5d894f8cc4ef | 0.65 | 847 | 0.63 | 0 | yes | 153 | 4-bit g64 | no |
| mlx-community/Qwen3-4B-4bit | yes | 4dcb3d101c2a | 2.28 | 904 | 2.26 | 0 | – | 0 | 4-bit g64 | – |
| mlx-community/Qwen3-0.6B-4bit | yes | 73e3e38d9813 | 0.35 | 704 | 0.34 | 0 | – | 0 | 4-bit g64 | – |
| mlx-community/Qwen3-1.7B-4bit | yes | 3b1b1768f8f8 | 0.98 | 704 | 0.97 | 0 | – | 0 | 4-bit g64 | – |
| z-lab/Qwen3.5-4B-DFlash | yes | 9a1996ccf887 | 1.27 | 69 | 1.27 | 0 | – | 0 | – | – |

Text-only weight sizes (without vision tensors): Qwen3.5-2B 1.06 GB, Qwen3.5-0.8B 0.42 GB. No quantization
config has per-layer overrides. The DFlash draft is BF16.

## Architecture

| Repo | model_type | text_config | Layers | Full-attention interval | Hidden | Vocab | Tied | linear k heads / v heads / k dim / v dim | KV heads | head_dim | MTP layers |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ConwayResearch/Underdog-Woof-4B-1.1 | qwen3_5 | yes | 32 | 4 | 2560 | 248320 | yes | 16/32/128/128 | 4 | 256 | 1 |
| ConwayResearch/Underdog-Woof-2B-1.1 | llama | no | 42 | – | 2048 | 130560 | no | –/–/–/– | 2 | 128 | – |
| mlx-community/Qwen3.5-2B-4bit | qwen3_5 | yes | 24 | 4 | 2048 | 248320 | yes | 16/16/128/128 | 2 | 256 | 1 |
| mlx-community/Qwen3.5-0.8B-MLX-4bit | qwen3_5 | yes | 24 | 4 | 1024 | 248320 | yes | 16/16/128/128 | 2 | 256 | 1 |
| mlx-community/Qwen3-4B-4bit | qwen3 | no | 36 | – | 2560 | 151936 | yes | –/–/–/– | 8 | 128 | – |
| mlx-community/Qwen3-0.6B-4bit | qwen3 | no | 28 | – | 1024 | 151936 | yes | –/–/–/– | 8 | 128 | – |
| mlx-community/Qwen3-1.7B-4bit | qwen3 | no | 28 | – | 2048 | 151936 | yes | –/–/–/– | 8 | 128 | – |
| z-lab/Qwen3.5-4B-DFlash | qwen3 | no | 6 | – | 2560 | 248320 | yes | –/–/–/– | 8 | 128 | – |

"MTP layers" is `mtp_num_hidden_layers` from config.json; the weights for it are absent everywhere.

## Tokens

| Repo | EOS (config) | EOS (generation_config) | Tokenizer EOS | `<\|im_end\|>` | `<\|endoftext\|>` | `<tool_call>` | `<think>` | Stop set | Vocab hash |
|---|---|---|---|---|---|---|---|---|---|
| ConwayResearch/Underdog-Woof-4B-1.1 | [248044, 248046] | [248044, 248046] | <\|im_end\|> (248046) | 248046 | 248044 | 248058 | 248068 | [248044, 248046] | 5660eab8ed1d73c3 |
| ConwayResearch/Underdog-Woof-2B-1.1 | [1, 130073] | [1, 130073] | </s> (1) | 130073 | None | 2 | 8 | [1, 130073] | 1043a49e871843d1 |
| mlx-community/Qwen3.5-2B-4bit | 248044 | no file | <\|im_end\|> (248046) | 248046 | 248044 | 248058 | 248068 | [248044, 248046] | 5660eab8ed1d73c3 |
| mlx-community/Qwen3.5-0.8B-MLX-4bit | 248044 | no file | <\|im_end\|> (248046) | 248046 | 248044 | 248058 | 248068 | [248044, 248046] | 5660eab8ed1d73c3 |
| mlx-community/Qwen3-4B-4bit | 151645 | no file | <\|im_end\|> (151645) | 151645 | 151643 | 151657 | 151667 | [151643, 151645] | 8a14c88d4bebfca7 |
| mlx-community/Qwen3-0.6B-4bit | 151645 | no file | <\|im_end\|> (151645) | 151645 | 151643 | 151657 | 151667 | [151643, 151645] | 8a14c88d4bebfca7 |
| mlx-community/Qwen3-1.7B-4bit | 151645 | no file | <\|im_end\|> (151645) | 151645 | 151643 | 151657 | 151667 | [151643, 151645] | 8a14c88d4bebfca7 |
| z-lab/Qwen3.5-4B-DFlash | 248044 | no file | <\|endoftext\|> (0) | None | 0 | None | None | [0, 248044] | ccb3ac1f03b5fba7 |

Woof 4B and the Qwen3.5 checkpoints share one tokenizer (same hash), so Qwen3.5-0.8B or -2B could draft for
Woof 4B; Woof 2B could not.

## Chat template

| Repo | Tool-call format | (c) tools rendered | (c) tools before system text | (d) tool result role | (e) `[system]` alone raises | (f) sentinel delta exact | (g) history reply keeps a think block | (g) as-generated think block kept | (h) system prefix ok | Generation prompt |
|---|---|---|---|---|---|---|---|---|---|---|
| ConwayResearch/Underdog-Woof-4B-1.1 | xml_function | 10/10 | yes | user | yes | yes | no | no | yes | `"<\|im_start\|>assistant\n<think>\n\n</think>\n\n"` |
| ConwayResearch/Underdog-Woof-2B-1.1 | unknown | 10/10 | no | user | no | yes | yes | yes | yes | `"<\|im_start\|>assistant\n<think>\n\n</think>\n\n"` |
| mlx-community/Qwen3.5-2B-4bit | xml_function | 10/10 | yes | user | yes | yes | no | no | yes | `"<\|im_start\|>assistant\n<think>\n\n</think>\n\n"` |
| mlx-community/Qwen3.5-0.8B-MLX-4bit | xml_function | 10/10 | yes | user | yes | yes | no | no | yes | `"<\|im_start\|>assistant\n<think>\n\n</think>\n\n"` |
| mlx-community/Qwen3-4B-4bit | json | 10/10 | no | user | no | yes | no | no | yes | `"<\|im_start\|>assistant\n<think>\n\n</think>\n\n"` |
| mlx-community/Qwen3-0.6B-4bit | json | 10/10 | no | user | no | yes | no | no | yes | `"<\|im_start\|>assistant\n<think>\n\n</think>\n\n"` |
| mlx-community/Qwen3-1.7B-4bit | json | 10/10 | no | user | no | yes | no | no | yes | `"<\|im_start\|>assistant\n<think>\n\n</think>\n\n"` |
| z-lab/Qwen3.5-4B-DFlash | None | – | – | – | – | – | – | – | – | – |

The full JSON (one object per repo, with the rendered texts) is printed between `=== BEGIN MODEL FACTS ===`
and `=== END MODEL FACTS ===`. Errors are recorded per repo and per section, so a gated or missing repo never
hides the others. Apart from the expected (e) template errors, run 37705247655 recorded none.

## Swift engine baseline (Local engine run 37705247676)

From the `=== BEGIN ENGINE REPORT ===` block. The runner's GPU is virtualized, so absolute speeds are not
phone numbers.

- **GPU canary:** Metal device "Apple Paravirtual device", architecture `air64_v27`, 7168 MiB, recommended
  working set 4778 MiB; MLX on the GPU ok. So the job ran the unit tests (tiny models) and the integration
  tests on the GPU rather than compile-only, and both steps passed, as did the iOS compile of LocalEngine.
  xcodebuild ran with `-quiet`, so the log has no per-test counts. Woof was not loaded in Swift
  (`WOOF=false` on this run).
- **F1 confirmed** (Qwen3-0.6B-4bit, 80-token system prompt): `ChatSession(instructions: S)` prefills 100
  tokens on turn 1 and **102** on turn 2, because it re-sends the system prompt every turn;
  `instructions: nil, history: [.system(S)]` prefills 100 and then **17**.
- **Stock `TokenIterator` greedy decode** (25 prompt tokens, 64 generated, prefill not counted):
  Qwen3-0.6B-4bit **128.8 tok/s**, Qwen3.5-0.8B-MLX-4bit **95.9 tok/s**.
- **Tokenizer bridge** (Qwen3-0.6B-4bit): system-only render 85 tokens, with the user turn 93, with the
  generation prompt 100; stop tokens `[151643, 151645]`; tool-call format default.
- **Pins:** `LocalEngine/Package.resolved` is copied from the run's `PACKAGE RESOLVED` block, and
  `LocalEngine/ci-models.txt` pins each test model to the snapshot that run downloaded.

## What is measured

`scripts/model_facts.py` reads, for every repo in `scripts/lab_models.txt` and without downloading weights:

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
  (`scripts/lab_prompts.json`): (a) `[system, user]` with the generation prompt; (b)
  `[system, user, assistant, user]`; (c) (b) with tools; (d) a tool round; (e) `[system]` alone without the
  generation prompt; (f) the sentinel delta of plan section 4.5, plain, with tools and for a tool round; (g)
  whether past replies keep a think block; (h) whether the tokens before the last `<|im_start|>` of (a) equal
  the encoded system segment.

## Running it

- CI: pushes that touch `scripts/model_facts.py`, `scripts/lab_models.txt`, `scripts/lab_prompts.json` or the
  workflow; every Monday at 05:17 UTC; or by hand once the workflow is on the default branch.
- Locally: `pip install "huggingface_hub>=0.25" "transformers>=4.56" jinja2 tokenizers`, then
  `python -I scripts/model_facts.py --models scripts/lab_models.txt --out facts.json`. It writes `facts.json`
  and the summary `facts.md`. Set `HF_TOKEN` for gated repos.
