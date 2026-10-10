#!/usr/bin/env python3
"""Record the facts about the lab models that later work packages depend on.

    python -I scripts/model_facts.py --models scripts/lab_models.txt --out facts.json

For each Hugging Face repo in the list it gathers, without downloading any weights:
- whether the repo exists and its resolved revision;
- the safetensors layout from the file headers (HTTP range reads): tensor bytes and count,
  `mtp.*` tensors, the first conv1d layout, vision tensors, dtypes;
- config.json fields (layers, linear-attention dimensions, quantization, MTP layers);
- EOS ids from config.json, generation_config.json and the tokenizer, and the ids of the
  chat-control tokens;
- how the chat template renders the turns the engine needs (checks a to h, see `template_checks`).

Each repo, and each section within a repo, records its own error instead of stopping the run.
facts.json holds one object per repo. A Markdown summary of the key tables is written next to
it (facts.md by default) for docs/model-facts.md.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from collections import Counter


def _hf_token():
    # The workflow passes secrets.HF_TOKEN, which is empty when the secret isn't set.
    token = os.environ.get("HF_TOKEN")
    if not token:
        os.environ.pop("HF_TOKEN", None)
        return None
    return token


HF_TOKEN = _hf_token()

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402  (after the token cleanup)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Tokens whose ids the engine needs (stop set, drafting exclusions, tool-call detection).
CONTROL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<tool_call>",
    "</tool_call>",
    "<think>",
    "</think>",
    "<tool_response>",
    "</tool_response>",
]

# Chat used by the template checks. The <context> tag matches PromptBuilder.
CONTEXT = "<context>time: Wednesday 7 October 2026, 16:05 Asia/Singapore; input: spoken</context>\n\n"
SYSTEM = "You are a warm, capable personal assistant, running as an iPhone app."
USER_1 = CONTEXT + "Hi! What can you help me with?"
ASSISTANT_1 = "I can answer questions, set timers and reminders, and check your calendar."
USER_2 = CONTEXT + "Remind me at 5 to call mum."
TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "create_reminder", "arguments": {"title": "Call mum", "due": "2026-10-07T17:00"}},
}
TOOL_RESULT = '{"id": "r-1", "status": "created", "title": "Call mum", "due": "2026-10-07T17:00"}'
ASSISTANT_TOOL_TEXT = "Sure."
# Placeholder turns for the sentinel render (plan section 4.5).
SENTINEL_USER = "SENTINEL-USER-7F3A"
SENTINEL_ASSISTANT = "SENTINEL-ASSISTANT-7F3A"
AS_GENERATED_THINK = "<think>\n\n</think>\n\n"


# ---------------------------------------------------------------------------
# Helpers


def retry(fn, attempts=3, delay=5.0):
    """Retries transient Hub failures; missing or gated repos and files fail at once."""
    permanent = {
        "RepositoryNotFoundError",
        "GatedRepoError",
        "EntryNotFoundError",
        "RemoteEntryNotFoundError",
        "RevisionNotFoundError",
        "NotASafetensorsRepoError",
        "DisabledRepoError",
    }
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as error:  # noqa: BLE001
            if type(error).__name__ in permanent or attempt == attempts:
                raise
            time.sleep(delay * attempt)


def describe(error):
    text = str(error).strip().splitlines()
    return f"{type(error).__name__}: {text[0] if text else ''}"[:500]


def section(entry, name, fn):
    """Runs one section of a repo's facts, recording its error instead of raising."""
    try:
        value = fn()
        if value is not None:
            entry[name] = value
        return value
    except Exception as error:  # noqa: BLE001
        entry.setdefault("errors", {})[name] = describe(error)
        entry.setdefault("tracebacks", {})[name] = traceback.format_exc(limit=4)[-2000:]
        return None


def compact_json(value, indent=0, width=160):
    """JSON with every value that fits in `width` characters on one line, so the CI log stays readable."""
    flat = json.dumps(value, ensure_ascii=False)
    if not isinstance(value, (dict, list)) or not value or len(flat) + indent <= width:
        return flat
    pad = " " * (indent + 1)
    if isinstance(value, dict):
        items = [f"{pad}{json.dumps(str(k), ensure_ascii=False)}: {compact_json(v, indent + 1, width)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    items = [pad + compact_json(v, indent + 1, width) for v in value]
    return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"


def short_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def token_pieces(tokenizer, ids):
    return [tokenizer.convert_ids_to_tokens(i) for i in ids]


def load_tool_schemas(path):
    with open(path, encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts["tool_schemas"]


def template_tools(schemas):
    """The tool list a Hugging Face chat template expects (WP20's JSONBridge.templateTools does the same)."""
    return [
        {
            "type": "function",
            "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]},
        }
        for t in schemas
    ]


# ---------------------------------------------------------------------------
# Hub and safetensors


def hub_facts(api, repo):
    info = retry(lambda: api.model_info(repo, files_metadata=True, token=HF_TOKEN))
    files = []
    for sibling in info.siblings or []:
        files.append({"name": sibling.rfilename, "size": sibling.size})
    total = sum(f["size"] or 0 for f in files)
    weights = sum(f["size"] or 0 for f in files if f["name"].endswith(".safetensors"))
    card = {}
    card_data = getattr(info, "card_data", None)
    if card_data is not None:
        data = card_data.to_dict() if hasattr(card_data, "to_dict") else dict(card_data)
        for key in ("base_model", "license", "pipeline_tag", "library_name"):
            if data.get(key) is not None:
                card[key] = data[key]
    return {
        "exists": True,
        "revision": info.sha,
        "last_modified": info.last_modified.isoformat() if getattr(info, "last_modified", None) else None,
        "gated": getattr(info, "gated", None),
        "private": getattr(info, "private", None),
        "library_name": getattr(info, "library_name", None),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "tags": list(getattr(info, "tags", None) or [])[:30],
        "card": card,
        "total_file_bytes": total,
        "safetensors_file_bytes": weights,
        "files": files,
    }


def is_vision_tensor(name):
    return (
        name.startswith("vision_tower")
        or name.startswith("model.visual")
        or ".visual." in name
        or "vision_tower" in name
        or name.startswith("visual.")
    )


def analyze_tensors(tensors):
    """tensors: {name: (dtype, shape, nbytes)}. Pure, so it can be tested on a local file."""
    names = sorted(tensors)
    dtype_counts = Counter()
    dtype_bytes = Counter()
    total = 0
    for name in names:
        dtype, _, nbytes = tensors[name]
        dtype_counts[dtype] += 1
        dtype_bytes[dtype] += nbytes
        total += nbytes
    mtp = [n for n in names if "mtp." in n]
    vision = [n for n in names if is_vision_tensor(n)]
    conv = [n for n in names if n.endswith("conv1d.weight")]
    conv_info = None
    if conv:
        shape = list(tensors[conv[0]][1])
        conv_info = {
            "name": conv[0],
            "shape": shape,
            # MLX layout is [channels, kernel, 1]; the PyTorch layout [channels, 1, kernel] still
            # needs sanitizing (and, in 3.31.4 and 3.32.3 alike, a +1 shift of the norm weights).
            "sanitized": bool(shape) and shape[-1] == 1,
            "count": len(conv),
            "all_same_layout": len({tuple(tensors[n][1])[-1] == 1 for n in conv}) == 1,
        }
    prefixes = Counter(".".join(n.split(".")[:2]) for n in names)
    scales = [n for n in names if n.endswith(".scales")]
    embed = [n for n in names if n.endswith("embed_tokens.weight")]
    return {
        "tensor_count": len(names),
        "tensor_bytes": total,
        "dtypes": {k: {"tensors": dtype_counts[k], "bytes": dtype_bytes[k]} for k in sorted(dtype_counts)},
        "mtp_tensor_count": len(mtp),
        "mtp_tensor_bytes": sum(tensors[n][2] for n in mtp),
        "mtp_examples": mtp[:8],
        "conv1d": conv_info,
        "vision_tensor_count": len(vision),
        "vision_tensor_bytes": sum(tensors[n][2] for n in vision),
        "vision_examples": vision[:4],
        "quantized_tensor_count": len(scales),
        "has_lm_head": any(n.endswith("lm_head.weight") for n in names),
        "embed_tokens": [{"name": n, "dtype": tensors[n][0], "shape": list(tensors[n][1])} for n in embed[:2]],
        "top_prefixes": dict(prefixes.most_common(12)),
    }


def safetensors_facts(api, repo, revision):
    meta = retry(lambda: api.get_safetensors_metadata(repo, revision=revision, token=HF_TOKEN))
    tensors = {}
    for file_meta in meta.files_metadata.values():
        for name, info in file_meta.tensors.items():
            begin, end = info.data_offsets
            tensors[name] = (info.dtype, list(info.shape), int(end) - int(begin))
    facts = analyze_tensors(tensors)
    facts["sharded"] = bool(meta.sharded)
    facts["files"] = sorted(meta.files_metadata)
    return facts


# ---------------------------------------------------------------------------
# Config files

LINEAR_KEYS = ["linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"]
TEXT_KEYS = [
    "model_type",
    "num_hidden_layers",
    "full_attention_interval",
    "hidden_size",
    "intermediate_size",
    "vocab_size",
    "tie_word_embeddings",
    *LINEAR_KEYS,
    "linear_conv_kernel_dim",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "mtp_num_hidden_layers",
    "mtp_use_dedicated_embeddings",
    "max_position_embeddings",
    "num_experts",
    "eos_token_id",
    "bos_token_id",
]


def download_json(repo, filename, revision):
    path = retry(lambda: hf_hub_download(repo, filename, revision=revision, token=HF_TOKEN))
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summarize_quantization(quant):
    if not isinstance(quant, dict):
        return quant
    base = {k: v for k, v in quant.items() if not isinstance(v, (dict, bool))}
    overrides = {k: v for k, v in quant.items() if isinstance(v, (dict, bool))}
    distinct = Counter(json.dumps(v, sort_keys=True) for v in overrides.values())
    return {
        "base": base,
        "override_count": len(overrides),
        "override_settings": dict(distinct.most_common(8)),
        "override_examples": dict(list(overrides.items())[:8]),
    }


def summarize_layer_types(layer_types):
    if not isinstance(layer_types, list):
        return None
    counts = Counter(layer_types)
    full = [i for i, t in enumerate(layer_types) if "full" in str(t)]
    return {"counts": dict(counts), "full_attention_layers": full}


def config_facts(repo, revision, present):
    if "config.json" not in present:
        raise FileNotFoundError("no config.json in the repo")
    config = download_json(repo, "config.json", revision)
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else None
    source = text or config
    facts = {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "has_text_config": text is not None,
        "text": {k: source.get(k) for k in TEXT_KEYS if k in source},
        "root_tie_word_embeddings": config.get("tie_word_embeddings"),
        "root_eos_token_id": config.get("eos_token_id"),
        "layer_types": source.get("layer_types"),
        "layer_types_summary": summarize_layer_types(source.get("layer_types")),
        "rope_parameters": source.get("rope_parameters") or source.get("rope_scaling"),
        "root_keys": sorted(config),
    }
    quant = config.get("quantization", config.get("quantization_config"))
    if quant is not None:
        facts["quantization"] = summarize_quantization(quant)
    # Draft models (DFlash) keep their settings in their own keys.
    extra = {}
    for key, value in config.items():
        lowered = key.lower()
        if any(word in lowered for word in ("dflash", "block", "target", "draft", "mask")):
            if isinstance(value, (int, float, str, bool)) or (isinstance(value, list) and len(value) <= 64):
                extra[key] = value
            elif isinstance(value, dict):
                extra[key] = {k: v for k, v in value.items() if isinstance(v, (int, float, str, bool, list))}
    if extra:
        facts["draft_settings"] = extra
    return facts


def generation_config_facts(repo, revision, present):
    if "generation_config.json" not in present:
        return {"present": False}
    gen = download_json(repo, "generation_config.json", revision)
    keys = ["eos_token_id", "bos_token_id", "pad_token_id", "do_sample", "temperature", "top_p", "top_k", "repetition_penalty"]
    return {"present": True, **{k: gen.get(k) for k in keys if k in gen}}


# ---------------------------------------------------------------------------
# Tokenizer and template


def load_tokenizer(repo, revision):
    from transformers import AutoTokenizer

    return retry(lambda: AutoTokenizer.from_pretrained(repo, revision=revision, token=HF_TOKEN))


def chat_template_text(tokenizer):
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):
        template = template.get("default") or next(iter(template.values()), None)
    return template


def infer_tool_format(template):
    """Mirrors what the engine relies on: Qwen3.5 templates use <function=...> (xml-function)."""
    if not template:
        return None
    if "<function=" in template:
        return "xml_function"
    if "<tool_call>" in template:
        return "json"
    return "unknown"


def tokenizer_facts(tokenizer):
    vocab = tokenizer.get_vocab()
    added = getattr(tokenizer, "added_tokens_decoder", {}) or {}
    controls = {}
    for token in CONTROL_TOKENS:
        token_id = vocab.get(token)
        entry = {"id": token_id}
        if token_id is not None and token_id in added:
            entry["special"] = bool(getattr(added[token_id], "special", False))
        controls[token] = entry
    template = chat_template_text(tokenizer)
    vocab_items = sorted(vocab.items(), key=lambda kv: (kv[1], kv[0]))
    return {
        "class": type(tokenizer).__name__,
        "length": len(tokenizer),
        "vocab_hash": short_hash(json.dumps(vocab_items, ensure_ascii=False)),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token": tokenizer.bos_token,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "control_tokens": controls,
        "chat_template": {
            "present": bool(template),
            "sha256_16": short_hash(template) if template else None,
            "length": len(template) if template else 0,
            "tool_call_format": infer_tool_format(template),
            "mentions_enable_thinking": bool(template) and "enable_thinking" in template,
        },
    }


def render(tokenizer, messages, *, generation_prompt, tools=None):
    kwargs = {"add_generation_prompt": generation_prompt, "tokenize": False, "enable_thinking": False}
    if tools is not None:
        kwargs["tools"] = tools
    text = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(text, str):
        raise TypeError(f"apply_chat_template returned {type(text).__name__}, not text")
    return text


def encode(tokenizer, text):
    return list(tokenizer.encode(text, add_special_tokens=False))


def last_index(values, target):
    for i in range(len(values) - 1, -1, -1):
        if values[i] == target:
            return i
    return -1


def clip(text, limit=1200):
    if text is None or len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f" …[{len(text) - limit} chars]… " + text[-half:]


def sentinel_check(tokenizer, sentinel_messages, real_messages, sentinel_marker, *, tools=None):
    """Plan section 4.5: the delta after a reply is the sentinel render from the first <|im_end|> after
    the placeholder reply. It must equal the tail of the real conversation's render, as text and as tokens."""
    sentinel = render(tokenizer, sentinel_messages, generation_prompt=True, tools=tools)
    canonical = render(tokenizer, real_messages, generation_prompt=True, tools=tools)
    marker = sentinel.find(sentinel_marker)
    if marker < 0:
        return {"error": "the placeholder reply is missing from the sentinel render"}
    start = sentinel.find("<|im_end|>", marker)
    if start < 0:
        return {"error": "no <|im_end|> after the placeholder reply"}
    delta = sentinel[start:]
    delta_tokens = encode(tokenizer, delta)
    canonical_tokens = encode(tokenizer, canonical)
    tail = canonical_tokens[-len(delta_tokens):] if delta_tokens else []
    return {
        "delta_text": delta,
        "delta_tokens": len(delta_tokens),
        "text_match": canonical.endswith(delta),
        "token_match": bool(delta_tokens) and tail == delta_tokens,
    }


def template_checks(tokenizer, tools):
    """Checks (a) to (h) of WP01. Each records its own error."""
    out = {}
    im_start = tokenizer.get_vocab().get("<|im_start|>")

    def check(name, fn):
        try:
            out[name] = fn()
        except Exception as error:  # noqa: BLE001
            out[name] = {"error": describe(error)}

    a_messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER_1}]
    b_messages = a_messages + [{"role": "assistant", "content": ASSISTANT_1}, {"role": "user", "content": USER_2}]

    def a():
        text = render(tokenizer, a_messages, generation_prompt=True)
        tokens = encode(tokenizer, text)
        cut = text.rfind("<|im_start|>")
        generation_prompt = text[cut:] if cut >= 0 else None
        prompt_tokens = encode(tokenizer, generation_prompt) if generation_prompt else []
        return {
            "text": text,
            "tokens": len(tokens),
            "generation_prompt": generation_prompt,
            "generation_prompt_ids": prompt_tokens,
            "generation_prompt_pieces": token_pieces(tokenizer, prompt_tokens),
        }

    def b():
        text = render(tokenizer, b_messages, generation_prompt=True)
        return {"text": text, "tokens": len(encode(tokenizer, text))}

    def c():
        text = render(tokenizer, b_messages, generation_prompt=True, tools=tools)
        tokens = encode(tokenizer, text)
        tools_at = text.find("<tools>")
        system_at = text.find(SYSTEM)
        names = [t["function"]["name"] for t in tools]
        return {
            "tokens": len(tokens),
            "chars": len(text),
            "head": text[:400],
            "tail": text[-400:],
            "tools_block_before_system_text": tools_at >= 0 and system_at >= 0 and tools_at < system_at,
            "tools_rendered": sum(1 for name in names if name in text),
            "tools_given": len(names),
        }

    def d():
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_2},
            {"role": "assistant", "content": "", "tool_calls": [TOOL_CALL]},
            {"role": "tool", "tool_call_id": "call_1", "name": "create_reminder", "content": TOOL_RESULT},
        ]
        text = render(tokenizer, messages, generation_prompt=True, tools=tools)
        cut = text.find("<|im_start|>user\n" + USER_2)
        tail = text[cut:] if cut >= 0 else text[-1200:]
        reply_at = tail.find("<|im_start|>assistant")
        reply_end = tail.find("<|im_end|>", reply_at) if reply_at >= 0 else -1
        reply = tail[reply_at:reply_end] if reply_at >= 0 and reply_end >= 0 else None
        return {
            "text_after_system": tail,
            "tokens": len(encode(tokenizer, text)),
            "tool_result_role_rendered_as": "user" if "<|im_start|>user\n<tool_response>" in tail else (
                "tool" if "<|im_start|>tool" in tail else "other"),
            "assistant_tool_turn_has_think_block": reply is not None and "<think>" in reply,
        }

    def e():
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM}], add_generation_prompt=False, tokenize=False, enable_thinking=False
            )
            return {"raises": False, "text": text}
        except Exception as error:  # noqa: BLE001
            return {"raises": True, "error": describe(error)}

    def f():
        plain = sentinel_check(
            tokenizer,
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": SENTINEL_USER},
                {"role": "assistant", "content": SENTINEL_ASSISTANT},
                {"role": "user", "content": USER_2},
            ],
            b_messages,
            SENTINEL_ASSISTANT,
        )
        with_tools = sentinel_check(
            tokenizer,
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": SENTINEL_USER},
                {"role": "assistant", "content": SENTINEL_ASSISTANT},
                {"role": "user", "content": USER_2},
            ],
            b_messages,
            SENTINEL_ASSISTANT,
            tools=tools,
        )
        tool_round = sentinel_check(
            tokenizer,
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": SENTINEL_USER},
                {"role": "assistant", "content": SENTINEL_ASSISTANT, "tool_calls": [TOOL_CALL]},
                {"role": "tool", "tool_call_id": "call_1", "name": "create_reminder", "content": TOOL_RESULT},
            ],
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_2},
                {"role": "assistant", "content": ASSISTANT_TOOL_TEXT, "tool_calls": [TOOL_CALL]},
                {"role": "tool", "tool_call_id": "call_1", "name": "create_reminder", "content": TOOL_RESULT},
            ],
            SENTINEL_ASSISTANT,
            tools=tools,
        )
        return {"plain": plain, "with_tools": with_tools, "tool_round": tool_round}

    def g():
        b_text = render(tokenizer, b_messages, generation_prompt=True)
        start = b_text.find("<|im_start|>assistant")
        end = b_text.find("<|im_end|>", start) if start >= 0 else -1
        history_reply = b_text[start:end] if start >= 0 and end >= 0 else None
        as_generated = list(b_messages)
        as_generated[2] = {"role": "assistant", "content": AS_GENERATED_THINK + ASSISTANT_1}
        g_text = render(tokenizer, as_generated, generation_prompt=True)
        start = g_text.find("<|im_start|>assistant")
        end = g_text.find("<|im_end|>", start) if start >= 0 else -1
        kept = g_text[start:end] if start >= 0 and end >= 0 else None
        return {
            "history_reply_rendered": history_reply,
            "history_reply_has_think_block": history_reply is not None and "<think>" in history_reply,
            "as_generated_reply_rendered": kept,
            "as_generated_think_block_kept": kept is not None and "<think>" in kept,
        }

    def h():
        # Plan section 4.5: the system prefix is the tokens of [system, user] (no generation prompt)
        # before the last <|im_start|>. It must be exactly the system segment, and (a) must start with
        # it, followed by the <|im_start|> of the user turn.
        result = {}
        for label, tool_list in (("without_tools", None), ("with_tools", tools)):
            prefix_text = render(
                tokenizer,
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": SENTINEL_USER}],
                generation_prompt=False,
                tools=tool_list,
            )
            prefix_tokens = encode(tokenizer, prefix_text)
            cut = last_index(prefix_tokens, im_start)
            prefix = prefix_tokens[:cut] if cut > 0 else []
            system_segment = prefix_text[: prefix_text.rfind("<|im_start|>")]
            target = encode(tokenizer, render(tokenizer, a_messages, generation_prompt=True, tools=tool_list))
            result[label] = {
                "prefix_tokens": len(prefix),
                "prefix_equals_encoded_system_segment": bool(prefix) and encode(tokenizer, system_segment) == prefix,
                "prefix_is_prefix_of_a": bool(prefix) and target[: len(prefix)] == prefix,
                "next_token_is_im_start": bool(prefix) and len(target) > len(prefix) and target[len(prefix)] == im_start,
                "system_segment_tail": system_segment[-80:],
            }
        return result

    for name, fn in (("a", a), ("b", b), ("c", c), ("d", d), ("e", e), ("f", f), ("g", g), ("h", h)):
        check(name, fn)
    return out


# ---------------------------------------------------------------------------
# Verdicts


def as_id_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, int)]
    return [value] if isinstance(value, int) else []


def verdicts(entry):
    out = {}
    st = entry.get("safetensors") or {}
    if st:
        conv = st.get("conv1d") or {}
        has_mtp = st.get("mtp_tensor_count", 0) > 0
        out["has_mtp_tensors"] = has_mtp
        if conv:
            # 3.31.4's Qwen35.sanitize shifts the norms whenever any mtp.* key exists; on an already
            # converted (sanitized) checkpoint that is a second shift, which gives garbage (plan F7).
            out["loads_as_garbage_on_mlx_swift_lm_3_31_4"] = bool(has_mtp and conv.get("sanitized"))
        out["weight_gb"] = round(st.get("tensor_bytes", 0) / 1e9, 3)
        out["weight_gb_without_mtp_and_vision"] = round(
            (st.get("tensor_bytes", 0) - st.get("mtp_tensor_bytes", 0) - st.get("vision_tensor_bytes", 0)) / 1e9, 3
        )
    config = entry.get("config") or {}
    gen = entry.get("generation_config") or {}
    tok = entry.get("tokenizer") or {}
    if config or tok:
        # Plan section 4.2: configuration EOS ids (generation_config overrides config) ∪ the tokenizer's
        # EOS ∪ <|im_end|> and <|endoftext|>.
        configured = as_id_list(gen.get("eos_token_id")) if gen.get("eos_token_id") is not None else (
            as_id_list((config.get("text") or {}).get("eos_token_id")) or as_id_list(config.get("root_eos_token_id"))
        )
        stop = set(configured)
        if tok.get("eos_token_id") is not None:
            stop.add(tok["eos_token_id"])
        for token in ("<|im_end|>", "<|endoftext|>"):
            token_id = ((tok.get("control_tokens") or {}).get(token) or {}).get("id")
            if token_id is not None:
                stop.add(token_id)
        out["configured_eos_ids"] = configured
        out["stop_token_ids"] = sorted(stop)
    template = entry.get("template") or {}
    if template and "skipped" not in template:
        f = template.get("f") or {}
        out["sentinel_delta_exact"] = all(
            bool((f.get(k) or {}).get("text_match")) and bool((f.get(k) or {}).get("token_match"))
            for k in ("plain", "with_tools", "tool_round")
        )
        out["system_alone_raises"] = (template.get("e") or {}).get("raises")
        out["history_drops_think_block"] = (template.get("g") or {}).get("history_reply_has_think_block") is False
        h = template.get("h") or {}
        out["system_prefix_ok"] = all(
            bool((h.get(k) or {}).get(check))
            for k in ("without_tools", "with_tools")
            for check in ("prefix_equals_encoded_system_segment", "prefix_is_prefix_of_a", "next_token_is_im_start")
        )
    return out


# ---------------------------------------------------------------------------
# One repo


def repo_facts(api, repo, tools):
    entry = {"repo": repo}
    hub = section(entry, "hub", lambda: hub_facts(api, repo))
    if hub is None:
        error = (entry.get("errors") or {}).get("hub", "")
        if error.startswith("RepositoryNotFoundError"):
            entry["hub"] = {"exists": False}
            return entry
        entry["hub"] = {"exists": None}  # unknown: gated, private or a network failure
    revision = (entry.get("hub") or {}).get("revision")
    present = {f["name"] for f in (entry.get("hub") or {}).get("files", [])}
    if not present:
        # model_info failed for another reason (gated, network): try the files anyway.
        present = {"config.json", "generation_config.json"}
    section(entry, "safetensors", lambda: safetensors_facts(api, repo, revision))
    section(entry, "config", lambda: config_facts(repo, revision, present))
    section(entry, "generation_config", lambda: generation_config_facts(repo, revision, present))
    tokenizer = section(entry, "tokenizer_load", lambda: load_tokenizer(repo, revision))
    if tokenizer is not None:
        entry.pop("tokenizer_load", None)
        section(entry, "tokenizer", lambda: tokenizer_facts(tokenizer))
        if chat_template_text(tokenizer):
            section(entry, "template", lambda: template_checks(tokenizer, tools))
        else:
            entry["template"] = {"skipped": "no chat template"}
    entry["verdicts"] = verdicts(entry)
    return entry


# ---------------------------------------------------------------------------
# Markdown summary


def gb(value):
    return "–" if value is None else f"{value / 1e9:.2f}"


def yes_no(value):
    if value is None:
        return "–"
    return "yes" if value else "no"


def table(headers, rows):
    """A GitHub Markdown table. Pipes inside cells (as in <|im_end|>) are escaped."""

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(cell(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return lines


def markdown_summary(result):
    repos = result["repos"]
    lines = ["# Model facts summary", "", f"Generated {result['generated_at']} with {json.dumps(result['versions'])}.", ""]

    rows = []
    for e in repos:
        hub = e.get("hub") or {}
        st = e.get("safetensors") or {}
        conv = st.get("conv1d") or {}
        quant = (e.get("config") or {}).get("quantization") or {}
        base = quant.get("base") if isinstance(quant, dict) else None
        qtext = "–"
        if base:
            qtext = f"{base.get('bits')}-bit g{base.get('group_size')}"
            if quant.get("override_count"):
                qtext += f", {quant['override_count']} overrides"
        v = e.get("verdicts") or {}
        rows.append([
            e["repo"], yes_no(hub.get("exists")), (hub.get("revision") or "–")[:12], gb(hub.get("total_file_bytes")),
            st.get("tensor_count", "–"), gb(st.get("tensor_bytes")), st.get("mtp_tensor_count", "–"),
            yes_no(conv.get("sanitized")) if conv else "–", st.get("vision_tensor_count", "–"), qtext,
            yes_no(v.get("loads_as_garbage_on_mlx_swift_lm_3_31_4")),
        ])
    lines += ["## Checkpoints", ""] + table(
        ["Repo", "Exists", "Revision", "Files GB", "Tensors", "Tensor GB", "`mtp.*` tensors", "conv1d sanitized",
         "Vision tensors", "Quantization", "Garbage on 3.31.4 (F7)"], rows)

    rows = []
    for e in repos:
        c = e.get("config") or {}
        t = c.get("text") or {}
        tied = t.get("tie_word_embeddings", c.get("root_tie_word_embeddings"))
        rows.append([
            e["repo"], c.get("model_type", "–"), yes_no(c.get("has_text_config")) if c else "–", t.get("num_hidden_layers", "–"),
            t.get("full_attention_interval", "–"), t.get("hidden_size", "–"), t.get("vocab_size", "–"), yes_no(tied),
            "/".join(str(t.get(k, "–")) for k in LINEAR_KEYS), t.get("num_key_value_heads", "–"), t.get("head_dim", "–"),
            t.get("mtp_num_hidden_layers", "–"),
        ])
    lines += ["", "## Architecture", ""] + table(
        ["Repo", "model_type", "text_config", "Layers", "Full-attention interval", "Hidden", "Vocab", "Tied",
         "linear k heads / v heads / k dim / v dim", "KV heads", "head_dim", "MTP layers"], rows)

    rows = []
    for e in repos:
        c = e.get("config") or {}
        g = e.get("generation_config") or {}
        tk = e.get("tokenizer") or {}
        ct = tk.get("control_tokens") or {}
        v = e.get("verdicts") or {}

        def tid(name):
            return (ct.get(name) or {}).get("id", "–")

        cfg_eos = (c.get("text") or {}).get("eos_token_id", c.get("root_eos_token_id")) if c else "–"
        gen_eos = g.get("eos_token_id", "–") if g.get("present") else ("no file" if g else "–")
        rows.append([
            e["repo"], cfg_eos, gen_eos, f"{tk.get('eos_token', '–')} ({tk.get('eos_token_id', '–')})" if tk else "–",
            tid("<|im_end|>"), tid("<|endoftext|>"), tid("<tool_call>"), tid("<think>"), v.get("stop_token_ids", "–"),
            tk.get("vocab_hash", "–"),
        ])
    lines += ["", "## Tokens", ""] + table(
        ["Repo", "EOS (config)", "EOS (generation_config)", "Tokenizer EOS", "`<|im_end|>`", "`<|endoftext|>`",
         "`<tool_call>`", "`<think>`", "Stop set", "Vocab hash"], rows)

    rows = []
    for e in repos:
        tk = e.get("tokenizer") or {}
        tp = e.get("template") or {}
        v = e.get("verdicts") or {}
        g = tp.get("g") or {}
        checked = bool(tp) and "skipped" not in tp
        gen_prompt = (tp.get("a") or {}).get("generation_prompt")
        c = tp.get("c") or {}
        d = tp.get("d") or {}
        rows.append([
            e["repo"], (tk.get("chat_template") or {}).get("tool_call_format", "–"),
            f"{c['tools_rendered']}/{c['tools_given']}" if "tools_rendered" in c else "–",
            yes_no(c.get("tools_block_before_system_text")), d.get("tool_result_role_rendered_as", "–"),
            yes_no(v.get("system_alone_raises")),
            yes_no(v.get("sentinel_delta_exact")) if checked else "–", yes_no(g.get("history_reply_has_think_block")),
            yes_no(g.get("as_generated_think_block_kept")), yes_no(v.get("system_prefix_ok")) if checked else "–",
            f"`{json.dumps(gen_prompt)}`" if gen_prompt else "–",
        ])
    lines += ["", "## Chat template", ""] + table(
        ["Repo", "Tool-call format", "(c) tools rendered", "(c) tools before system text", "(d) tool result role",
         "(e) `[system]` alone raises", "(f) sentinel delta exact",
         "(g) history reply keeps a think block", "(g) as-generated think block kept", "(h) system prefix ok",
         "Generation prompt"], rows)

    errors = [[e["repo"], k, m] for e in repos for k, m in (e.get("errors") or {}).items()]
    if errors:
        lines += ["", "## Errors", ""] + table(["Repo", "Section", "Error"], errors)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------


def versions():
    out = {"python": platform.python_version()}
    for name in ("huggingface_hub", "transformers", "tokenizers", "jinja2"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "?")
        except Exception as error:  # noqa: BLE001
            out[name] = f"unavailable ({describe(error)})"
    return out


def read_models(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", required=True, help="File with one Hugging Face repo per line.")
    parser.add_argument("--out", required=True, help="Where to write the JSON facts.")
    parser.add_argument("--summary", help="Where to write the Markdown summary (default: --out with .md).")
    parser.add_argument("--prompts", default=os.path.join(SCRIPT_DIR, "lab_prompts.json"), help="lab_prompts.json, for the tool schemas.")
    args = parser.parse_args(argv)

    tools = template_tools(load_tool_schemas(args.prompts))
    api = HfApi()
    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "token": "set" if HF_TOKEN else "none",
        "versions": versions(),
        "repos": [],
    }
    for repo in read_models(args.models):
        print(f"[model-facts] {repo}", file=sys.stderr, flush=True)
        try:
            entry = repo_facts(api, repo, tools)
        except Exception as error:  # noqa: BLE001  (a bug in this script must not lose the other repos)
            entry = {"repo": repo, "errors": {"repo": describe(error)}, "tracebacks": {"repo": traceback.format_exc(limit=6)}}
        for name, message in (entry.get("errors") or {}).items():
            print(f"[model-facts]   {name}: {message}", file=sys.stderr, flush=True)
        result["repos"].append(entry)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(compact_json(result))
        f.write("\n")
    summary = args.summary or os.path.splitext(args.out)[0] + ".md"
    with open(summary, "w", encoding="utf-8") as f:
        f.write(markdown_summary(result))
    print(f"[model-facts] wrote {args.out} and {summary}", file=sys.stderr)
    # A run that could read no repo at all (Hub down, network blocked) must not look green.
    if not any((e.get("hub") or {}).get("exists") for e in result["repos"]):
        print("[model-facts] no repo could be read from the Hub", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
