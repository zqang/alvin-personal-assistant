#!/usr/bin/env python3
"""Drafter lab: what speculative drafting and native tool calls would give on Woof, measured with mlx-lm.

    python -I scripts/drafter_lab.py --prompts scripts/lab_prompts.json --out lab.json
    python -I scripts/drafter_lab.py --dflash --prompts scripts/lab_prompts.json --out dflash.json --baseline lab.json

Default mode, for Woof 4B (or Qwen3.5-2B if mlx-lm can't load Woof) and Qwen3.5-0.8B:
1. Greedy outputs (max 200 tokens, enable_thinking=False) for every chat, copy and tool prompt. Tool
   prompts run twice: with every tool schema ("tools") and with the on-device subset ("tools_local").
2. One chunked teacher-forced pass per prompt over its greedy output. It keeps, per position, only
   the top-20 of the app's sampling distribution (top-p 0.8 within top-k 20, at temperature 0.7),
   never full-vocabulary logits for all positions.
3. Acceptance simulation for the drafters (prompt lookup, suffix corpus, exemplar tool skeletons,
   and their combination) at K in {1, 2, 3, 4, 6, 8}. Rounds follow the greedy output. At T=0 a round
   accepts the drafts that match it; at T=0.7 a round is credited with its expected accepted count,
   the running product of the drafted tokens' probabilities. Drafts past the first one that leaves
   the greedy path have no recorded distribution and count as rejected, so T=0.7 is a lower bound.
   The projected speedup uses the default cost curve of plan section 4.7. The pooled "all" figures
   cover chat, copy and tools, each prompt once (tools_local replays the tool prompts).
4. Tool-call accuracy: name and key-argument matches against each prompt's `expect`. In tools_local, a
   prompt whose tool is outside the on-device subset expects handoff_to_cloud instead.
5. As-generated versus canonical tokens around a past reply (what the template re-renders).

--dflash mode runs `dflash generate` (dflash-mlx) with z-lab/Qwen3.5-4B-DFlash against the Woof
snapshot on 6 prompts, greedy, and records tokens per round, accepted draft tokens per round and
acceptance, with both readings of the Swift port threshold.

lab.json holds everything; a Markdown summary of the key tables is written next to it.
"""

import argparse
import gc
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import unicodedata


def _hf_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        os.environ.pop("HF_TOKEN", None)
        return None
    return token


HF_TOKEN = _hf_token()

PRIMARY = "ConwayResearch/Underdog-Woof-4B-1.1"
FALLBACK = "mlx-community/Qwen3.5-2B-4bit"
COMPARE = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
DFLASH_DRAFT = "z-lab/Qwen3.5-4B-DFlash"
DFLASH_PROMPTS = ["chat-01", "chat-05", "copy-01", "copy-03", "copy-06", "tool-01"]
# Go/no-go for the deferred Swift DFlash port (plan section 3 and WP01 item 8; see run_dflash).
PORT_THRESHOLD = 2.5

KS = [1, 2, 3, 4, 6, 8]
MAX_DRAFT = max(KS)
# Plan section 4.7: stock MLX on M-series, relative cost of one forward over S rows.
COST_CURVE = {1: 1.0, 2: 1.05, 3: 1.35, 4: 1.63, 5: 1.95, 6: 2.3, 8: 3.07, 9: 3.4}
# The app's sampling parameters (LocalModelHost.parameters).
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20

SETS = ["chat", "copy", "tools", "tools_local"]
# The sets pooled into the "all" drafting aggregate: every prompt once. tools_local replays the tools
# prompts with fewer schemas, so pooling it too would count each tool prompt twice.
ALL_SETS = ["chat", "copy", "tools"]
DRAFTERS = [
    "prompt_lookup",
    "prompt_lookup_n2",
    "prompt_lookup_n3",
    "prompt_lookup_n4",
    "suffix_corpus",
    "exemplar",
    "combined",
]


def log(message):
    print(f"[drafter-lab] {message}", file=sys.stderr, flush=True)


def compact_json(value, indent=0, width=160, flat_depth=7):
    """JSON with every value that fits in `width` characters, or sits `flat_depth` levels deep, on one
    line, so the CI log stays readable (one line per drafter and K, per tool prompt)."""
    flat = json.dumps(value, ensure_ascii=False)
    if not isinstance(value, (dict, list)) or not value or len(flat) + indent <= width or indent >= flat_depth:
        return flat
    pad = " " * (indent + 1)
    if isinstance(value, dict):
        items = [f"{pad}{json.dumps(str(k), ensure_ascii=False)}: {compact_json(v, indent + 1, width)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    items = [pad + compact_json(v, indent + 1, width) for v in value]
    return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as f:
        f.write(compact_json(value))
        f.write("\n")


def describe(error):
    text = str(error).strip().splitlines()
    return f"{type(error).__name__}: {text[0] if text else ''}"[:500]


# ---------------------------------------------------------------------------
# Cost curve


def relative_cost(rows, curve=COST_CURVE):
    """c(S) relative to c(1); linear interpolation, extrapolated from the last segment (as WP11's CostCurve)."""
    points = sorted(curve.items())
    base = curve.get(1, points[0][1])
    if rows <= points[0][0]:
        return points[0][1] / base
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if rows <= x1:
            return (y0 + (y1 - y0) * (rows - x0) / (x1 - x0)) / base
    (x0, y0), (x1, y1) = points[-2], points[-1]
    return (y1 + (y1 - y0) * (rows - x1) / (x1 - x0)) / base


# ---------------------------------------------------------------------------
# Drafters. Each proposal is (tokens, source, match_length).


def prompt_lookup_proposals(prompt, trajectory, min_n, max_n, max_tokens, excluded):
    """Proposals at every position of the trajectory, from the ledger prompt + trajectory[:pos].

    WP11's NGramIndex: the longest suffix n-gram (max_n down to min_n) with an earlier occurrence;
    the most recent occurrence wins; the continuation stops at an excluded token or the current end.
    """
    ledger = list(prompt)
    index = {n: {} for n in range(min_n, max_n + 1)}
    proposals = []

    def add_ngrams_ending_before_last(length):
        # Make every n-gram that ends at ledger[length - 2] findable (starts < length - n).
        for n in index:
            start = length - 1 - n
            if start >= 0:
                index[n][tuple(ledger[start:start + n])] = start

    for m in range(1, len(ledger)):
        add_ngrams_ending_before_last(m)
    for pos in range(len(trajectory)):
        m = len(ledger)
        add_ngrams_ending_before_last(m)
        proposal = None
        for n in range(max_n, min_n - 1, -1):
            if m < n + 1:
                continue
            start = index[n].get(tuple(ledger[m - n:m]))
            if start is None:
                continue
            tokens = []
            j = start + n
            while len(tokens) < max_tokens and j < m and ledger[j] not in excluded:
                tokens.append(ledger[j])
                j += 1
            if tokens:
                proposal = (tokens, "prompt_lookup", n)
            break
        proposals.append(proposal)
        ledger.append(trajectory[pos])
    return proposals


def prompt_lookup_reference(ledger, min_n, max_n, max_tokens, excluded):
    """Brute-force version of one prompt-lookup proposal, for the self-test."""
    m = len(ledger)
    for n in range(max_n, min_n - 1, -1):
        if m < n + 1:
            continue
        suffix = ledger[m - n:]
        for start in range(m - n - 1, -1, -1):
            if ledger[start:start + n] == suffix:
                tokens = []
                j = start + n
                while len(tokens) < max_tokens and j < m and ledger[j] not in excluded:
                    tokens.append(ledger[j])
                    j += 1
                return (tokens, "prompt_lookup", n) if tokens else None
    return None


class SequenceMatcher:
    """Longest context match against stored sequences, continued token by token with the most frequent
    next token (ties go to the most recent occurrence). Used for the suffix corpus (WP11 SuffixCorpus)
    and for exemplar tool-call skeletons (WP11 ExemplarSeeds)."""

    def __init__(self, source, min_match=2, max_match=8, single_ok=()):
        self.source = source
        self.min_match = min_match
        self.max_match = max_match
        self.single_ok = set(single_ok)  # tokens that may match on their own (e.g. <tool_call>)
        self.docs = []
        self.tags = []
        self.index = {}
        self.tokens = 0

    def add(self, doc, tag=None):
        doc = list(doc)
        if not doc:
            return
        d = len(self.docs)
        self.docs.append(doc)
        self.tags.append(tag)
        self.tokens += len(doc)
        for n in range(self.min_match, self.max_match + 1):
            for start in range(len(doc) - n + 1):
                self.index.setdefault(tuple(doc[start:start + n]), []).append((d, start + n))

    def propose(self, context, max_tokens, excluded, skip_tag=None):
        for n in range(min(self.max_match, len(context)), self.min_match - 1, -1):
            if n == 1 and self.min_match == 1 and context[-1] not in self.single_ok:
                continue
            occurrences = self.index.get(tuple(context[-n:]))
            if skip_tag is not None and occurrences:
                occurrences = [(d, p) for d, p in occurrences if self.tags[d] != skip_tag]
            if not occurrences:
                continue
            candidates = list(occurrences)
            tokens = []
            while len(tokens) < max_tokens:
                counts = {}
                for d, p in candidates:
                    doc = self.docs[d]
                    if p < len(doc) and doc[p] not in excluded:
                        count, recent = counts.get(doc[p], (0, (-1, -1)))
                        counts[doc[p]] = (count + 1, max(recent, (d, p)))
                if not counts:
                    break
                best = max(counts.items(), key=lambda kv: kv[1])[0]
                tokens.append(best)
                candidates = [(d, p + 1) for d, p in candidates if p < len(self.docs[d]) and self.docs[d][p] == best]
            if tokens:
                return tokens, self.source, n
        return None

    def proposals(self, prompt, trajectory, max_tokens, excluded, skip_tag=None):
        ledger = list(prompt)
        out = []
        for token in trajectory:
            out.append(self.propose(ledger, max_tokens, excluded, skip_tag))
            ledger.append(token)
        return out


def exemplar_texts(schemas, tool_format):
    """One tool-call skeleton per tool in the model's format, plus the joints between arguments.

    Fragments end before a value and never with a space, which BPE would merge into the value's
    first token (` "Call`); JSON joints come with and without the closing quote of a string value."""
    texts = []
    for schema in schemas:
        name = schema["name"]
        params = list((schema.get("input_schema") or {}).get("properties") or {})
        required = list((schema.get("input_schema") or {}).get("required") or [])
        first = required[0] if required else (params[0] if params else None)
        if tool_format == "json":
            if first:
                texts.append('<tool_call>\n{"name": "' + name + '", "arguments": {"' + first + '":')
            else:
                texts.append('<tool_call>\n{"name": "' + name + '", "arguments": {}}\n</tool_call>')
            texts += ['", "' + p + '":' for p in params] + [', "' + p + '":' for p in params]
        else:
            if first:
                texts.append(f"<tool_call>\n<function={name}>\n<parameter={first}>\n")
            else:
                texts.append(f"<tool_call>\n<function={name}>\n</function>\n</tool_call>")
            texts += [f"\n</parameter>\n<parameter={p}>\n" for p in params]
    if tool_format == "json":
        texts += ['"}}\n</tool_call>', "}}\n</tool_call>"]
    else:
        texts.append("\n</parameter>\n</function>\n</tool_call>")
    seen = set()
    return [t for t in texts if not (t in seen or seen.add(t))]


def combine(*proposal_lists):
    """First non-empty proposal per position, in the given drafter order."""
    out = []
    for options in zip(*proposal_lists):
        out.append(next((p for p in options if p), None))
    return out


# ---------------------------------------------------------------------------
# Round simulation


def processed_probability(dist, token):
    """dist = (ids, q): the kept top-k entries of the app's sampling distribution."""
    ids, q = dist
    for i, candidate in enumerate(ids):
        if candidate == token:
            return q[i]
    return 0.0


def simulate(trajectory, proposals, dists, k, curve=COST_CURVE):
    """Sequential rounds along the greedy trajectory with at most k drafts per round.

    Returns raw sums; `finish_stats` turns them into rates. dists[i] is the distribution that
    predicts trajectory[i] (None disables the T=0.7 numbers)."""
    n_tokens = len(trajectory)
    s = {
        "positions": n_tokens,
        "rounds": 0,
        "spec_rounds": 0,
        "tokens": 0,
        "spec_tokens_t0": 0,
        "expected_tokens_t07": 0.0,
        "spec_expected_t07": 0.0,
        "cost": 0.0,
        "drafted": 0,
        "accepted_t0": 0,
        "proposed_at": [0] * k,
        "accepted_at_t0": [0] * k,
        "expected_at_t07": [0.0] * k,
        "by_match": {},
    }
    pos = 0
    while pos < n_tokens:
        proposal = proposals[pos]
        drafts = proposal[0][: min(k, n_tokens - pos - 1)] if proposal else []
        s["rounds"] += 1
        if not drafts:
            s["tokens"] += 1
            s["expected_tokens_t07"] += 1.0
            s["cost"] += relative_cost(1, curve)
            pos += 1
            continue
        accepted = 0
        while accepted < len(drafts) and drafts[accepted] == trajectory[pos + accepted]:
            accepted += 1
        expected = 0.0
        running = 1.0
        on_path = True
        for i, token in enumerate(drafts):
            s["proposed_at"][i] += 1
            if i < accepted:
                s["accepted_at_t0"][i] += 1
            if dists is not None and on_path:
                running *= processed_probability(dists[pos + i], token)
                expected += running
                s["expected_at_t07"][i] += running
                # Beyond a draft that leaves the greedy path there is no recorded distribution.
                on_path = token == trajectory[pos + i]
        s["spec_rounds"] += 1
        s["drafted"] += len(drafts)
        s["accepted_t0"] += accepted
        s["tokens"] += accepted + 1
        s["spec_tokens_t0"] += accepted + 1
        s["expected_tokens_t07"] += 1.0 + expected
        s["spec_expected_t07"] += 1.0 + expected
        s["cost"] += relative_cost(len(drafts) + 1, curve)
        bucket = str(min(proposal[2], 4)) + ("+" if proposal[2] >= 4 else "")
        b = s["by_match"].setdefault(bucket, {"rounds": 0, "accepted_t0": 0, "first_t0": 0, "expected_t07": 0.0, "first_t07": 0.0})
        b["rounds"] += 1
        b["accepted_t0"] += accepted
        b["first_t0"] += 1 if accepted >= 1 else 0
        if dists is not None:
            b["expected_t07"] += expected
            b["first_t07"] += processed_probability(dists[pos], drafts[0])
        pos += accepted + 1
    return s


def merge_sums(a, b):
    if a is None:
        return json.loads(json.dumps(b))
    for key, value in b.items():
        if key == "by_match":
            for bucket, values in value.items():
                target = a["by_match"].setdefault(bucket, {k: 0 for k in values})
                for k2, v2 in values.items():
                    target[k2] += v2
        elif isinstance(value, list):
            a[key] = [x + y for x, y in zip(a[key], value)]
        else:
            a[key] += value
    return a


def ratio(a, b, digits=3):
    return round(a / b, digits) if b else None


def finish_stats(s, with_t07=True):
    out = {
        "positions": s["positions"],
        "rounds": s["rounds"],
        "spec_rounds": s["spec_rounds"],
        "spec_round_rate": ratio(s["spec_rounds"], s["rounds"]),
        "tokens_per_round_t0": ratio(s["tokens"], s["rounds"]),
        "tokens_per_spec_round_t0": ratio(s["spec_tokens_t0"], s["spec_rounds"]),
        "draft_acceptance_t0": ratio(s["accepted_t0"], s["drafted"]),
        "acceptance_by_depth_t0": [ratio(a, p) for a, p in zip(s["accepted_at_t0"], s["proposed_at"])],
        "speedup_t0": ratio(s["tokens"], s["cost"]),
    }
    if with_t07:
        out.update({
            "tokens_per_round_t07": ratio(s["expected_tokens_t07"], s["rounds"]),
            "tokens_per_spec_round_t07": ratio(s["spec_expected_t07"], s["spec_rounds"]),
            "acceptance_by_depth_t07": [ratio(e, p) for e, p in zip(s["expected_at_t07"], s["proposed_at"])],
            "speedup_t07": ratio(s["expected_tokens_t07"], s["cost"]),
        })
    out["by_match_length"] = {
        bucket: {
            "rounds": v["rounds"],
            "mean_accepted_t0": ratio(v["accepted_t0"], v["rounds"]),
            "first_draft_accepted_t0": ratio(v["first_t0"], v["rounds"]),
            **({"mean_accepted_t07": ratio(v["expected_t07"], v["rounds"]),
                "first_draft_accepted_t07": ratio(v["first_t07"], v["rounds"])} if with_t07 else {}),
        }
        for bucket, v in sorted(s["by_match"].items())
    }
    return out


# ---------------------------------------------------------------------------
# Tool calls


def _strip_value(value):
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value.strip()


def parse_xml_function(body):
    m = re.search(r"<function=([^>\n]+)>", body)
    if not m:
        return None
    args = {}
    for p in re.finditer(r"<parameter=([^>\n]+)>(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", body[m.end():], re.S):
        args[p.group(1).strip()] = _strip_value(p.group(2))
    return {"name": m.group(1).strip(), "arguments": args}


def parse_json_call(body):
    body = body.strip()
    start = body.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(body[start:])
    except ValueError:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("name"), str):
        return None
    arguments = value.get("arguments", value.get("parameters", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = {"_raw": arguments}
    return {"name": value["name"], "arguments": arguments if isinstance(arguments, dict) else {"_raw": arguments}}


def parse_tool_calls(text):
    """Every <tool_call> block in the text (xml-function or JSON), and bare <function=...> blocks."""
    calls = []
    blocks = list(re.finditer(r"<tool_call>(.*?)(?:</tool_call>|(?=<tool_call>)|$)", text, re.S))
    for block in blocks:
        body = block.group(1)
        call = parse_xml_function(body) if "<function=" in body else parse_json_call(body)
        if call:
            call["wrapped"] = True
            calls.append(call)
    if not blocks:
        for block in re.finditer(r"<function=[^>\n]+>.*?(?:</function>|$)", text, re.S):
            call = parse_xml_function(block.group(0))
            if call:
                call["wrapped"] = False
                calls.append(call)
    return calls


DATE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ](\d{1,2}):(\d{2}))?")


def normalize_text(value):
    value = unicodedata.normalize("NFKC", str(value)).lower()
    return "".join(" " if unicodedata.category(ch).startswith(("P", "S")) else ch for ch in value)


def match_value(expected, actual):
    if isinstance(expected, list):
        return any(match_value(e, actual) for e in expected)
    if actual is None:
        return False
    if isinstance(expected, bool):
        return str(actual).strip().lower() == str(expected).lower()
    if isinstance(expected, int):
        try:
            number = float(str(actual).strip())
        except ValueError:
            return False
        return number == expected
    expected = str(expected)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", expected):
        m = DATE.search(str(actual))
        return bool(m) and tuple(int(x) for x in m.groups()[:3]) == tuple(int(x) for x in expected.split("-"))
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", expected):
        m = DATE.search(str(actual))
        if not m or m.group(4) is None:
            return False
        want = DATE.fullmatch(expected).groups()
        return tuple(int(x) for x in m.groups()) == tuple(int(x) for x in want)
    haystack = normalize_text(actual)
    return all(word in haystack for word in normalize_text(expected).split())


def score_tool_call(expect, text):
    """expect["name"] None means no tool call is expected."""
    calls = parse_tool_calls(text)
    first = calls[0] if calls else None
    name_ok = first is None if expect["name"] is None else bool(first) and first["name"] == expect["name"]
    arg_results = {}
    if first:
        for key, value in (expect.get("args") or {}).items():
            arg_results[key] = match_value(value, (first.get("arguments") or {}).get(key))
    args_ok = name_ok and (first is None or all(arg_results.values()))
    return {
        "calls": calls,
        "parsed": bool(calls),
        "wrapped": bool(first) and first.get("wrapped", False),
        "name_ok": name_ok,
        "args": arg_results,
        "args_ok": args_ok,
        "text_before_call": text[: text.find("<tool_call>")].strip()[:200] if "<tool_call>" in text else None,
    }


# ---------------------------------------------------------------------------
# Prompts and rendering


def load_prompts(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def template_tools(schemas):
    return [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
        for t in schemas
    ]


def local_expect(prompts, expect):
    """The expected call when only the on-device subset is offered (tools_local).

    A prompt whose tool is outside the subset cannot be served on the device, so it expects
    handoff_to_cloud (any reason), or no call when the subset has no handoff tool.
    """
    local = set(prompts["local_tools"])
    if expect["name"] in local:
        return expect
    return {"name": "handoff_to_cloud" if "handoff_to_cloud" in local else None, "args": {}}


def lab_items(prompts, limit=0):
    """Every prompt run, in order: chat, copy, tools (every schema), tools_local (on-device subset)."""
    items = []
    for set_name in SETS:
        source = prompts["tools" if set_name.startswith("tools") else set_name]
        for item in source[: limit or None]:
            entry = {"set": set_name, **item}
            if set_name == "tools_local" and "expect" in item:
                entry["expect"] = local_expect(prompts, item["expect"])
                if entry["expect"] is not item["expect"]:
                    entry["expect_original"] = item["expect"]
            items.append(entry)
    return items


def schemas_for(prompts, set_name):
    if set_name == "tools":
        return prompts["tool_schemas"]
    if set_name == "tools_local":
        local = set(prompts["local_tools"])
        return [t for t in prompts["tool_schemas"] if t["name"] in local]
    return None


def messages_for(prompts, item):
    content = prompts["user_template"].format(time=prompts["context_time"], input=item.get("input", "spoken"), text=item["text"])
    return [{"role": "system", "content": prompts["system"]}, {"role": "user", "content": content}]


def hf_tokenizer(tokenizer):
    """The Hugging Face tokenizer inside mlx-lm's TokenizerWrapper. A plain transformers tokenizer is
    returned as is (its own `_tokenizer` is the Rust tokenizer, which has no chat template)."""
    inner = getattr(tokenizer, "_tokenizer", None)
    return inner if inner is not None and hasattr(inner, "apply_chat_template") else tokenizer


def render(tokenizer, messages, schemas=None, generation_prompt=True):
    kwargs = {"add_generation_prompt": generation_prompt, "tokenize": False, "enable_thinking": False}
    if schemas is not None:
        kwargs["tools"] = template_tools(schemas)
    text = hf_tokenizer(tokenizer).apply_chat_template(messages, **kwargs)
    if not isinstance(text, str):
        raise TypeError(f"apply_chat_template returned {type(text).__name__}")
    return text


def encode(tokenizer, text):
    return list(hf_tokenizer(tokenizer).encode(text, add_special_tokens=False))


def chat_template_text(tokenizer):
    template = getattr(hf_tokenizer(tokenizer), "chat_template", None)
    if isinstance(template, dict):
        template = template.get("default") or next(iter(template.values()), None)
    return template or ""


def infer_tool_format(template):
    if "<function=" in template:
        return "xml_function"
    if "<tool_call>" in template:
        return "json"
    return "unknown"


# ---------------------------------------------------------------------------
# MLX: loading, greedy generation, teacher forcing


def mlx_imports():
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    try:
        from mlx_lm.generate import generation_stream, wired_limit
    except ImportError:  # older or newer mlx-lm
        generation_stream, wired_limit = None, None
    return mx, load, generate_step, make_prompt_cache, generation_stream, wired_limit


def greedy(model, prompt_ids, stop_ids, max_tokens):
    mx, _, generate_step, _, stream, wired_limit = mlx_imports()
    import contextlib

    context = wired_limit(model, [stream]) if wired_limit and stream is not None else contextlib.nullcontext()
    tokens = []
    stop_token = None
    start = time.perf_counter()
    first = None
    with context:
        for token, _ in generate_step(mx.array(prompt_ids), model, max_tokens=max_tokens):
            if first is None:
                first = time.perf_counter()
            if token in stop_ids:
                stop_token = token
                break
            tokens.append(int(token))
    end = time.perf_counter()
    generated = len(tokens) + (1 if stop_token is not None else 0)
    decode_time = end - first if first else 0.0
    return {
        "tokens": tokens,
        "stop_token": stop_token,
        "first_token_seconds": round(first - start, 4) if first else None,
        "prefill_tokens_per_second": round(len(prompt_ids) / (first - start), 1) if first and first > start else None,
        "decode_tokens_per_second": round((generated - 1) / decode_time, 1) if generated > 1 and decode_time > 0 else None,
    }


def processed_top_k(mx, logits, top_k=TOP_K, top_p=TOP_P, temperature=TEMPERATURE):
    """The app's TopPSampler over float32 logits [L, V]: log-softmax, nucleus top_p on the full
    distribution (entries whose exclusive cumulative probability is below top_p), intersected with
    the top_k, then softmax(logprob / T). Returns the kept ids in descending order, their sampling
    probabilities (zero for entries outside the nucleus) and the top-1 probability at T=1."""
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    k = min(top_k, logits.shape[-1])
    part = mx.argpartition(-logprobs, kth=k - 1, axis=-1)[:, :k]
    values = mx.take_along_axis(logprobs, part, axis=-1)
    order = mx.argsort(-values, axis=-1)
    ids = mx.take_along_axis(part, order, axis=-1)
    values = mx.take_along_axis(values, order, axis=-1)
    probs = mx.exp(values)
    keep = (mx.cumsum(probs, axis=-1) - probs) < top_p
    weights = mx.where(keep, mx.exp((values - values[:, :1]) / temperature), 0.0)
    q = weights / mx.sum(weights, axis=-1, keepdims=True)
    return ids, q, probs[:, 0]


def teacher_forced(model, prompt_ids, trajectory, chunk=64, prefill_chunk=512):
    """Distributions predicting each trajectory token, from one cache: prefill prompt[:-1] cache-only,
    then feed [prompt[-1]] + trajectory[:-1] in chunks and keep only the processed top-k per row."""
    mx, _, _, make_prompt_cache, _, _ = mlx_imports()
    if not trajectory:
        return [], 0
    cache = make_prompt_cache(model)
    head = prompt_ids[:-1]
    for i in range(0, len(head), prefill_chunk):
        model(mx.array(head[i:i + prefill_chunk])[None], cache=cache)
        mx.eval([c.state for c in cache])
    inputs = [prompt_ids[-1]] + list(trajectory[:-1])
    dists = []
    argmax_mismatch = 0
    for i in range(0, len(inputs), chunk):
        logits = model(mx.array(inputs[i:i + chunk])[None], cache=cache)[0].astype(mx.float32)
        ids, q, top1 = processed_top_k(mx, logits)
        mx.eval(ids, q, top1)
        mx.eval([c.state for c in cache])
        for row_ids, row_q in zip(ids.tolist(), q.tolist()):
            if row_ids[0] != trajectory[len(dists)]:
                argmax_mismatch += 1
            dists.append((row_ids, row_q))
    return dists, argmax_mismatch


# ---------------------------------------------------------------------------
# One model


def special_ids(tokenizer):
    vocab = hf_tokenizer(tokenizer).get_vocab()
    names = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<tool_call>", "</tool_call>", "<think>", "</think>"]
    return {name: vocab.get(name) for name in names}


def added_token_ids(tokenizer):
    added = getattr(hf_tokenizer(tokenizer), "added_tokens_decoder", {}) or {}
    return set(added)


def drafting_for_model(records, prompts, tokenizer, tool_format, excluded, single_ok):
    """Acceptance simulation per set and drafter; the corpus grows with each prompt's output."""
    exemplar_cache = {}
    corpus = SequenceMatcher("suffix_corpus", min_match=2, max_match=8)
    sums = {}
    proposal_positions = {}
    for record in records:
        if record.get("error") or not record.get("trajectory"):
            continue
        prompt, trajectory, dists = record["prompt_ids"], record["trajectory"], record.get("dists")
        set_name = record["set"]
        schemas = schemas_for(prompts, set_name) or prompts["tool_schemas"]
        key = tuple(s["name"] for s in schemas)
        if key not in exemplar_cache:
            matcher = SequenceMatcher("exemplar", min_match=1, max_match=8, single_ok=single_ok)
            for text in exemplar_texts(schemas, tool_format):
                matcher.add(encode(tokenizer, text))
            exemplar_cache[key] = matcher
        per_drafter = {
            "prompt_lookup": prompt_lookup_proposals(prompt, trajectory, 2, 4, MAX_DRAFT, excluded),
            "prompt_lookup_n2": prompt_lookup_proposals(prompt, trajectory, 2, 2, MAX_DRAFT, excluded),
            "prompt_lookup_n3": prompt_lookup_proposals(prompt, trajectory, 3, 3, MAX_DRAFT, excluded),
            "prompt_lookup_n4": prompt_lookup_proposals(prompt, trajectory, 4, 4, MAX_DRAFT, excluded),
            # tools_local repeats the tools prompts: never draft from an earlier answer to the same prompt.
            "suffix_corpus": corpus.proposals(prompt, trajectory, MAX_DRAFT, excluded, skip_tag=record["id"]),
            "exemplar": exemplar_cache[key].proposals(prompt, trajectory, MAX_DRAFT, excluded),
        }
        per_drafter["combined"] = combine(per_drafter["prompt_lookup"], per_drafter["exemplar"], per_drafter["suffix_corpus"])
        for drafter, proposals in per_drafter.items():
            counts = proposal_positions.setdefault(set_name, {}).setdefault(drafter, [0, 0])
            counts[0] += sum(1 for p in proposals if p)
            counts[1] += len(proposals)
            for k in KS:
                slot = sums.setdefault(set_name, {}).setdefault(drafter, {})
                slot[k] = merge_sums(slot.get(k), simulate(trajectory, proposals, dists, k))
        corpus.add(record["output_ids"], tag=record["id"])

    out = {}
    with_t07 = all(r.get("dists") is not None for r in records if r.get("trajectory"))
    sets = list(sums)
    pooled = [s for s in sets if s in ALL_SETS]
    for set_name in sets + ["all"]:
        out[set_name] = {}
        for drafter in DRAFTERS:
            if set_name == "all":
                merged = {}
                positions = [0, 0]
                for s in pooled:
                    if drafter not in sums[s]:
                        continue
                    for k, v in sums[s][drafter].items():
                        merged[k] = merge_sums(merged.get(k), v)
                    positions = [a + b for a, b in zip(positions, proposal_positions[s][drafter])]
                by_k = merged
            else:
                by_k = sums[set_name].get(drafter, {})
                positions = proposal_positions[set_name].get(drafter, [0, 0])
            if not by_k:
                continue
            finished = {str(k): finish_stats(v, with_t07) for k, v in sorted(by_k.items())}
            best_t0 = max(finished.items(), key=lambda kv: kv[1]["speedup_t0"] or 0)
            entry = {
                "proposal_rate": ratio(positions[0], positions[1]),
                "best_k_t0": int(best_t0[0]),
                "best_speedup_t0": best_t0[1]["speedup_t0"],
                "by_k": finished,
            }
            if with_t07:
                best_t07 = max(finished.items(), key=lambda kv: kv[1]["speedup_t07"] or 0)
                entry["best_k_t07"] = int(best_t07[0])
                entry["best_speedup_t07"] = best_t07[1]["speedup_t07"]
            out[set_name][drafter] = entry
    return out


def template_comparison(records, prompts, tokenizer, ids):
    """As-generated versus canonical tokens of a past reply."""
    hf = hf_tokenizer(tokenizer)
    im_start, im_end = ids.get("<|im_start|>"), ids.get("<|im_end|>")
    header = encode(tokenizer, "<|im_start|>assistant\n")
    think = encode(tokenizer, "<think>\n\n</think>\n\n")
    chat = [r for r in records if r["set"] in ("chat", "copy") and not r.get("error") and r.get("output_ids") is not None]
    differing, identical = [], []
    stable = equal_as_generated = compared = think_dropped = 0
    reasons = {}
    for index, record in enumerate(chat):
        follow = chat[(index + 1) % len(chat)]
        messages = messages_for(prompts, record) + [
            {"role": "assistant", "content": hf.decode(record["output_ids"])},
            messages_for(prompts, follow)[1],
        ]
        canonical = encode(tokenizer, render(tokenizer, messages))
        starts = [i for i, t in enumerate(canonical) if t == im_start]
        if len(starts) < 3 or canonical[starts[2]:starts[2] + len(header)] != header:
            reasons["unexpected layout"] = reasons.get("unexpected layout", 0) + 1
            continue
        begin = starts[2] + len(header)
        end = canonical.index(im_end, begin) if im_end in canonical[begin:] else len(canonical)
        canonical_reply = canonical[begin:end]
        prompt_ids = record["prompt_ids"]
        last = max(i for i, t in enumerate(prompt_ids) if t == im_start)
        if prompt_ids[last:last + len(header)] != header:
            reasons["unexpected layout"] = reasons.get("unexpected layout", 0) + 1
            continue
        as_generated = prompt_ids[last + len(header):] + record["output_ids"]
        compared += 1
        is_stable = canonical_reply[-len(record["output_ids"]):] == record["output_ids"] if record["output_ids"] else canonical_reply == []
        stable += is_stable
        same = canonical_reply == as_generated
        equal_as_generated += same
        if think and as_generated[: len(think)] == think and canonical_reply[: len(think)] != think:
            think_dropped += 1
        if not is_stable:
            text = hf.decode(record["output_ids"])
            reason = "reply trimmed by the template" if text != text.strip() else "re-tokenized differently"
            reasons[reason] = reasons.get(reason, 0) + 1
        bucket = identical if same else differing
        if len(bucket) < 2:
            diff = None if same else next(
                (i for i, (a, b) in enumerate(zip(as_generated, canonical_reply)) if a != b),
                min(len(as_generated), len(canonical_reply)))
            window = slice(max(0, (diff or 0) - 4), (diff or 0) + 8)
            bucket.append({
                "id": record["id"],
                "as_generated_tokens": len(as_generated),
                "canonical_tokens": len(canonical_reply),
                "first_difference": diff,
                "as_generated_window": hf.convert_ids_to_tokens(as_generated[window]),
                "canonical_window": hf.convert_ids_to_tokens(canonical_reply[window]),
                "as_generated_head": hf.decode(as_generated[:12]),
                "canonical_head": hf.decode(canonical_reply[:12]),
            })
    return {
        "compared": compared,
        "canonical_equals_as_generated": equal_as_generated,
        "reply_tokens_stable_on_rerender": stable,
        "think_block_dropped_on_rerender": think_dropped,
        "unstable_reasons": reasons,
        # Differing replies first: they are the ones that matter for "append as generated" (plan 4.5).
        "examples": (differing + identical)[:2],
    }


def run_model(repo, prompts, args):
    mx, load, _, _, _, _ = mlx_imports()
    result = {"repo": repo}
    start = time.perf_counter()
    model, tokenizer = load(repo)
    result["load_seconds"] = round(time.perf_counter() - start, 1)
    ids = special_ids(tokenizer)
    stop = set(int(t) for t in tokenizer.eos_token_ids) | {ids[n] for n in ("<|im_end|>", "<|endoftext|>") if ids[n] is not None}
    excluded = {ids[n] for n in ("<|im_start|>", "<|endoftext|>") if ids[n] is not None}
    template = chat_template_text(tokenizer)
    tool_format = infer_tool_format(template)
    result.update({"special_ids": ids, "stop_ids": sorted(stop), "draft_excluded_ids": sorted(excluded), "tool_call_format": tool_format})
    log(f"{repo}: loaded in {result['load_seconds']} s, tool format {tool_format}, stop {sorted(stop)}")

    records = []
    for item in lab_items(prompts, args.limit):
        record = {"set": item["set"], "id": item["id"], "input": item.get("input", "spoken"), "text": item["text"]}
        for key in ("expect", "expect_original"):
            if key in item:
                record[key] = item[key]
        try:
            prompt_text = render(tokenizer, messages_for(prompts, item), schemas_for(prompts, item["set"]))
            prompt_ids = encode(tokenizer, prompt_text)
            gen = greedy(model, prompt_ids, stop, args.max_tokens)
            trajectory = gen["tokens"] + ([gen["stop_token"]] if gen["stop_token"] is not None else [])
            dists, mismatch = teacher_forced(model, prompt_ids, trajectory)
            record.update({
                "prompt_ids": prompt_ids,
                "output_ids": gen["tokens"],
                "trajectory": trajectory,
                "dists": dists,
                "prompt_tokens": len(prompt_ids),
                "output_tokens": len(gen["tokens"]),
                "stopped": "eos" if gen["stop_token"] is not None else "length",
                "first_token_seconds": gen["first_token_seconds"],
                "prefill_tokens_per_second": gen["prefill_tokens_per_second"],
                "decode_tokens_per_second": gen["decode_tokens_per_second"],
                "teacher_forced_argmax_mismatches": mismatch,
                "output": hf_tokenizer(tokenizer).decode(gen["tokens"]),
            })
            log(f"{repo} {item['set']}/{item['id']}: {len(prompt_ids)} + {len(gen['tokens'])} tokens, "
                f"{gen['decode_tokens_per_second']} tok/s, tf mismatches {mismatch}")
        except Exception as error:  # noqa: BLE001
            record["error"] = describe(error)
            record["traceback"] = traceback.format_exc(limit=5)[-2000:]
            log(f"{repo} {item['set']}/{item['id']}: {record['error']}")
        records.append(record)

    single_ok = {i for i in added_token_ids(tokenizer)}
    for name, fn in (
        ("drafting", lambda: drafting_for_model(records, prompts, tokenizer, tool_format, excluded, single_ok)),
        ("tool_accuracy", lambda: tool_accuracy(records)),
        ("template_comparison", lambda: template_comparison(records, prompts, tokenizer, ids)),
    ):
        try:
            result[name] = fn()
        except Exception as error:  # noqa: BLE001
            result.setdefault("errors", {})[name] = describe(error)
            result.setdefault("tracebacks", {})[name] = traceback.format_exc(limit=6)[-3000:]

    result["outputs"] = summarize_outputs(records)
    result["prompts"] = [
        {k: v for k, v in r.items() if k not in ("prompt_ids", "output_ids", "trajectory", "dists")} for r in records
    ]
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    return result


def summarize_outputs(records):
    out = {}
    for set_name in SETS:
        rows = [r for r in records if r["set"] == set_name and not r.get("error")]
        if not rows:
            continue

        def mean(key):
            values = [r[key] for r in rows if r.get(key) is not None]
            return round(sum(values) / len(values), 2) if values else None

        out[set_name] = {
            "prompts": len(rows),
            "errors": sum(1 for r in records if r["set"] == set_name and r.get("error")),
            "mean_prompt_tokens": mean("prompt_tokens"),
            "mean_output_tokens": mean("output_tokens"),
            "stopped_by_eos": sum(1 for r in rows if r["stopped"] == "eos"),
            "mean_decode_tokens_per_second": mean("decode_tokens_per_second"),
            "mean_prefill_tokens_per_second": mean("prefill_tokens_per_second"),
            "mean_first_token_seconds": mean("first_token_seconds"),
            "teacher_forced_argmax_mismatches": sum(r.get("teacher_forced_argmax_mismatches") or 0 for r in rows),
            "positions": sum(len(r["trajectory"]) for r in rows),
        }
    return out


def tool_accuracy(records):
    out = {}
    for set_name in ("tools", "tools_local"):
        rows = [r for r in records if r["set"] == set_name and "expect" in r]
        if not rows:
            continue
        per_prompt = []
        for r in rows:
            entry = {"id": r["id"], "text": r["text"], "expected": r["expect"]}
            if "expect_original" in r:
                entry["expected_in_tools"] = r["expect_original"]
            if r.get("error"):
                per_prompt.append({**entry, "error": r["error"], "name_ok": False, "args_ok": False, "parsed": False})
                continue
            score = score_tool_call(r["expect"], r["output"])
            first = score["calls"][0] if score["calls"] else None
            per_prompt.append({
                **entry,
                "got": {"name": first["name"], "arguments": first["arguments"]} if first else None,
                "calls": len(score["calls"]),
                "parsed": score["parsed"],
                "wrapped": score["wrapped"],
                "name_ok": score["name_ok"],
                "args": score["args"],
                "args_ok": score["args_ok"],
                "text_before_call": score["text_before_call"],
                "output_tokens": r.get("output_tokens"),
            })
        n = len(per_prompt)
        out[set_name] = {
            "prompts": n,
            # Prompts whose tool is outside the on-device subset; they expect handoff_to_cloud (see local_expect).
            "expect_handoff": sum(1 for r in rows if "expect_original" in r),
            "parsed": sum(p["parsed"] for p in per_prompt),
            "name_accuracy": ratio(sum(p["name_ok"] for p in per_prompt), n),
            "args_accuracy": ratio(sum(p["args_ok"] for p in per_prompt), n),
            "args_accuracy_given_name": ratio(sum(p["args_ok"] for p in per_prompt), sum(p["name_ok"] for p in per_prompt)),
            "per_prompt": per_prompt,
        }
    return out


# ---------------------------------------------------------------------------
# DFlash

# Runs `dflash generate` (dflash_mlx/generate.py main) unchanged, but also writes the engine's
# SummaryEvent (tokens per cycle, cycles, acceptance history, generated token ids) to stderr as JSON.
# generate.py's run_generate calls the module-level stream_dflash_generate (checked in dflash-mlx 0.1.8),
# so wrapping it is enough. The CLI itself prints each token decoded on its own, which garbles characters
# split across tokens, so the text comparison decodes the token ids instead.
DFLASH_WRAPPER = r"""
import json, sys
import dflash_mlx.generate as g
_stream = g.stream_dflash_generate
def _wrapped(*args, **kwargs):
    inner = _stream(*args, **kwargs)
    try:
        for event in inner:
            if type(event).__name__ == "SummaryEvent":
                try:
                    payload = event.to_payload() if hasattr(event, "to_payload") else dict(vars(event))
                except Exception as error:
                    payload = {"payload_error": repr(error)}
                payload.pop("cycle_profile_us", None)
                sys.stderr.write("\nDFLASH_SUMMARY_JSON " + json.dumps(payload, default=str) + "\n")
                sys.stderr.flush()
            yield event
    finally:
        close = getattr(inner, "close", None)
        if close is not None:
            close()
g.stream_dflash_generate = _wrapped
g.main(sys.argv[1:], prog="dflash generate")
"""

DFLASH_LINE = re.compile(r"(\d+) tokens \| ([\d.]+) tok/s \| ([\d.]+)% acceptance")


def parse_dflash(stdout, stderr):
    out = {}
    for line in stderr.splitlines():
        if line.startswith("DFLASH_SUMMARY_JSON "):
            try:
                out["summary"] = json.loads(line[len("DFLASH_SUMMARY_JSON "):])
            except ValueError as error:
                out["summary_error"] = describe(error)
        m = DFLASH_LINE.search(line)
        if m:
            out["tokens"] = int(m.group(1))
            out["tokens_per_second"] = float(m.group(2))
            out["acceptance"] = float(m.group(3)) / 100.0
    summary = out.get("summary") or {}
    cycles = summary.get("cycles_completed")
    if summary.get("tokens_per_cycle"):
        out["tokens_per_round"] = round(float(summary["tokens_per_cycle"]), 3)
    elif cycles:
        out["tokens_per_round"] = round(float(summary.get("generation_tokens", 0)) / cycles, 3)
    elif out.get("acceptance") is not None and out["acceptance"] < 1:
        # acceptance = drafted tokens kept / tokens generated, and each round adds one target token.
        out["tokens_per_round"] = round(1.0 / (1.0 - out["acceptance"]), 3)
        out["tokens_per_round_derived"] = True
    history = summary.get("acceptance_history")
    if isinstance(history, list) and history:
        # Accepted draft tokens per verify cycle, without the target's own token.
        out["mean_accepted_per_round"] = round(sum(history) / len(history), 3)
        out["rounds"] = len(history)
    elif out.get("tokens_per_round") is not None:
        out["mean_accepted_per_round"] = round(out["tokens_per_round"] - 1.0, 3)
    out["output"] = stdout
    return out


def run_dflash(prompts, args):
    from huggingface_hub import snapshot_download

    result = {"target": args.model, "draft": args.dflash_draft, "max_tokens": args.max_tokens, "runs": []}
    try:
        import importlib.metadata as metadata

        result["dflash_mlx_version"] = metadata.version("dflash-mlx")
    except Exception as error:  # noqa: BLE001
        result["dflash_mlx_version"] = f"unknown ({describe(error)})"
    # The files mlx-lm downloads (mlx_lm.utils.DEFAULT_ALLOW_PATTERNS), so the lab's cache is reused.
    patterns = ["*.json", "model*.safetensors", "*.py", "tokenizer.model", "*.tiktoken", "tiktoken.model", "*.txt", "*.jsonl", "*.jinja"]
    snapshot = snapshot_download(args.model, allow_patterns=patterns, token=HF_TOKEN)
    result["snapshot"] = snapshot
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    baseline = {}
    if args.baseline and os.path.exists(args.baseline):
        with open(args.baseline, encoding="utf-8") as f:
            lab = json.load(f)
        primary = (lab.get("models") or [{}])[0]
        if primary.get("repo") == args.model:
            baseline = {(p["set"], p["id"]): p for p in primary.get("prompts", [])}
    wanted = args.dflash_prompts.split(",")
    items = [i for i in lab_items(prompts) if i["id"] in wanted and i["set"] in ("chat", "copy", "tools")]
    for item in items:
        run = {"set": item["set"], "id": item["id"]}
        try:
            text = render(tokenizer, messages_for(prompts, item), schemas_for(prompts, item["set"]))
            command = [sys.executable, "-I", "-c", DFLASH_WRAPPER, "--model", snapshot, "--draft", args.dflash_draft,
                       "--prompt", text, "--no-chat-template", "--max-tokens", str(args.max_tokens)]
            started = time.perf_counter()
            proc = subprocess.run(command, capture_output=True, text=True, timeout=1200)
            if proc.returncode != 0 and "DFLASH_SUMMARY_JSON" not in proc.stderr:
                # The wrapper depends on dflash internals; fall back to the plain CLI.
                run["wrapper_error"] = proc.stderr.strip().splitlines()[-1:] if proc.stderr.strip() else proc.returncode
                command = [sys.executable, "-I", "-m", "dflash_mlx.cli", "generate", "--model", snapshot, "--draft",
                           args.dflash_draft, "--prompt", text, "--no-chat-template", "--max-tokens", str(args.max_tokens)]
                proc = subprocess.run(command, capture_output=True, text=True, timeout=1200)
            run["seconds"] = round(time.perf_counter() - started, 1)
            run["returncode"] = proc.returncode
            run.update(parse_dflash(proc.stdout, proc.stderr))
            token_ids = (run.get("summary") or {}).pop("generated_token_ids", None)
            if isinstance(token_ids, list) and token_ids:
                stops = {tokenizer.eos_token_id}
                for name in ("<|im_end|>", "<|endoftext|>"):
                    stops.add(tokenizer.get_vocab().get(name))
                while token_ids and token_ids[-1] in stops:
                    token_ids = token_ids[:-1]
                run["output"] = tokenizer.decode(token_ids)
                run["output_tokens"] = len(token_ids)
            if proc.returncode != 0:
                run["stderr_tail"] = proc.stderr[-1500:]
            base = baseline.get((item["set"], item["id"]))
            if base and run.get("tokens_per_second") and base.get("decode_tokens_per_second"):
                run["speedup_vs_lab_greedy"] = round(run["tokens_per_second"] / base["decode_tokens_per_second"], 2)
            if base and run.get("tokens") and base.get("output") is not None:
                a, b = run["output"].strip(), base["output"].strip()
                common = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                run["same_text_as_lab_greedy"] = a == b
                run["common_prefix_chars"] = common
            log(f"dflash {item['id']}: {run.get('tokens')} tokens, {run.get('tokens_per_round')} per round, "
                f"acceptance {run.get('acceptance')}")
        except Exception as error:  # noqa: BLE001
            run["error"] = describe(error)
            log(f"dflash {item['id']}: {run['error']}")
        result["runs"].append(run)
    ok = [r for r in result["runs"] if r.get("tokens_per_round")]
    if ok:
        result["mean_tokens_per_round"] = round(sum(r["tokens_per_round"] for r in ok) / len(ok), 3)
        result["mean_accepted_per_round"] = round(sum(r["mean_accepted_per_round"] for r in ok) / len(ok), 3)
        result["mean_acceptance"] = round(sum(r.get("acceptance") or 0 for r in ok) / len(ok), 3)
        # The Swift DFlash port's go/no-go. Plan section 3 says ">= 2.5 accepted tokens per round"; WP01 item 8
        # says "DFlash tokens per round". Tokens per round include the target's own token each round, accepted
        # draft tokens do not, so the two readings differ by one. Both are reported; the orchestrator decides.
        result["port_threshold"] = {
            "value": PORT_THRESHOLD,
            "accepted_draft_tokens_per_round": {"mean": result["mean_accepted_per_round"],
                                                "met": result["mean_accepted_per_round"] >= PORT_THRESHOLD},
            "tokens_per_round": {"mean": result["mean_tokens_per_round"],
                                 "met": result["mean_tokens_per_round"] >= PORT_THRESHOLD},
        }
    return result


# ---------------------------------------------------------------------------
# Markdown summary


def fmt(value, digits=2):
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def md_table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(cell(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return lines


def lab_markdown(result):
    lines = ["# Drafter lab summary", "", f"Generated {result['generated_at']}; {json.dumps(result['environment'])}.", ""]
    for note in result.get("notes", []):
        lines.append(f"- {note}")
    for m in result.get("models", []):
        lines += ["", f"## {m['repo']} ({m.get('role', '')})", ""]
        if m.get("load_error"):
            lines += [f"Could not load: {m['load_error']}", ""]
            continue
        lines.append(f"Tool-call format {m.get('tool_call_format')}; stop ids {m.get('stop_ids')}; load {m.get('load_seconds')} s.")
        lines += ["", "### Greedy outputs", ""]
        rows = []
        for set_name, s in (m.get("outputs") or {}).items():
            rows.append([set_name, s["prompts"], s["errors"], fmt(s["mean_prompt_tokens"], 0), fmt(s["mean_output_tokens"], 1),
                         s["stopped_by_eos"], fmt(s["mean_decode_tokens_per_second"], 1),
                         fmt(s["mean_prefill_tokens_per_second"], 0), s["teacher_forced_argmax_mismatches"]])
        lines += md_table(["Set", "Prompts", "Errors", "Prompt tokens", "Output tokens", "Stopped by EOS", "Decode tok/s",
                           "Prefill tok/s", "Teacher-forced argmax ≠ greedy"], rows)
        drafting = m.get("drafting") or {}
        for set_name in [s for s in SETS if s in drafting] + (["all"] if "all" in drafting else []):
            title = f"all ({', '.join(ALL_SETS)}; each prompt once)" if set_name == "all" else set_name
            lines += ["", f"### Drafting: {title}", ""]
            rows = []
            for drafter, d in drafting[set_name].items():
                k4 = d["by_k"].get("4", {})
                speedups = " / ".join(fmt(d["by_k"].get(str(k), {}).get("speedup_t0")) for k in KS)
                rows.append([
                    drafter, fmt(d["proposal_rate"]), speedups,
                    f"K={d['best_k_t0']}: {fmt(d['best_speedup_t0'])}",
                    f"K={d.get('best_k_t07', '–')}: {fmt(d.get('best_speedup_t07'))}",
                    f"{fmt(k4.get('tokens_per_spec_round_t0'))} / {fmt(k4.get('tokens_per_spec_round_t07'))}",
                    " ".join(fmt(x) for x in k4.get("acceptance_by_depth_t0", [])),
                    " ".join(fmt(x) for x in k4.get("acceptance_by_depth_t07", [])),
                ])
            lines += md_table(["Drafter", "Positions with a proposal", "Speedup T=0 at K=" + "/".join(map(str, KS)),
                               "Best T=0", "Best T=0.7", "Tokens per drafted round K=4 (T=0 / T=0.7)",
                               "Acceptance by depth K=4, T=0", "Acceptance by depth K=4, T=0.7"], rows)
        pl = ((drafting.get("all") or {}).get("prompt_lookup") or {}).get("by_k", {}).get("4", {}).get("by_match_length")
        if pl:
            lines += ["", f"### Prompt lookup by match length ({', '.join(ALL_SETS)}; K=4)", ""]
            lines += md_table(["Match length", "Rounds", "Mean accepted T=0", "First draft accepted T=0", "Mean accepted T=0.7",
                               "First draft accepted T=0.7"],
                              [[b, v["rounds"], fmt(v["mean_accepted_t0"]), fmt(v["first_draft_accepted_t0"]),
                                fmt(v.get("mean_accepted_t07")), fmt(v.get("first_draft_accepted_t07"))] for b, v in pl.items()])
        tools = m.get("tool_accuracy") or {}
        for set_name, t in tools.items():
            handoff = (f" {t['expect_handoff']} prompt(s) ask for a tool outside the on-device subset and expect "
                       f"handoff_to_cloud instead." if t.get("expect_handoff") else "")
            lines += ["", f"### Tool calls: {set_name}", "",
                      f"Name accuracy {fmt(t['name_accuracy'])}, arguments {fmt(t['args_accuracy'])} "
                      f"({fmt(t['args_accuracy_given_name'])} when the name was right), parsed {t['parsed']}/{t['prompts']}.{handoff}", ""]
            rows = []
            for p in t["per_prompt"]:
                got = p.get("got")
                expected = (p.get("expected") or {}).get("name", "") or "no call"
                if p.get("expected_in_tools"):
                    expected += f" (not {p['expected_in_tools']['name']})"
                rows.append([p["id"], p.get("text", ""), expected,
                             got["name"] if got else "–", "yes" if p["name_ok"] else "no", "yes" if p["args_ok"] else "no",
                             json.dumps(got["arguments"], ensure_ascii=False)[:120] if got else (p.get("error") or "no call"),
                             p.get("output_tokens", "–")])
            lines += md_table(["Prompt", "Text", "Expected", "Called", "Name", "Arguments", "Arguments given", "Output tokens"], rows)
        tc = m.get("template_comparison")
        if tc:
            lines += ["", "### As-generated versus canonical reply tokens", "",
                      f"{tc['compared']} past replies re-rendered: canonical equals as-generated for {tc['canonical_equals_as_generated']}; "
                      f"the template drops the generation prompt's empty think block for {tc.get('think_block_dropped_on_rerender', '–')}; "
                      f"reply tokens unchanged (ignoring the think block) for {tc['reply_tokens_stable_on_rerender']}. "
                      f"Other differences: {json.dumps(tc['unstable_reasons'], ensure_ascii=False)}.", ""]
            for ex in tc.get("examples", []):
                if ex["first_difference"] is None:
                    lines.append(f"- {ex['id']}: identical, {ex['as_generated_tokens']} tokens starting {json.dumps(ex['as_generated_head'], ensure_ascii=False)}…")
                    continue
                lines.append(f"- {ex['id']}: as generated {json.dumps(ex['as_generated_head'], ensure_ascii=False)}…, "
                             f"canonical {json.dumps(ex['canonical_head'], ensure_ascii=False)}…; first difference at token "
                             f"{ex['first_difference']}: {ex['as_generated_window']} vs {ex['canonical_window']}")
        for name, message in (m.get("errors") or {}).items():
            lines.append(f"- Error in {name}: {message}")
    return "\n".join(lines) + "\n"


def dflash_markdown(result):
    lines = ["# DFlash summary", "", f"Target {result.get('target')}, draft {result.get('draft')}, dflash-mlx {result.get('dflash_mlx_version')}.", ""]
    rows = []
    for r in result.get("runs", []):
        summary = r.get("summary") or {}
        rows.append([r["id"], r.get("tokens", "–"), r.get("rounds", "–"), fmt(r.get("tokens_per_round")),
                     fmt(r.get("mean_accepted_per_round")), fmt(r.get("acceptance")),
                     # dflash-mlx also drafts by copying from the prompt ("copyspec"); these rounds are in the totals.
                     f"{summary.get('copyspec_hits', 0)} / {summary.get('copyspec_tokens', 0)}" if summary else "–",
                     fmt(r.get("tokens_per_second"), 1),
                     fmt(r.get("speedup_vs_lab_greedy")), {True: "yes", False: "no"}.get(r.get("same_text_as_lab_greedy"), "–"),
                     r.get("error") or (f"exit {r.get('returncode')}" if r.get("returncode") else "")])
    lines += md_table(["Prompt", "Tokens", "Rounds", "Tokens per round", "Accepted per round", "Acceptance",
                       "Copy-spec rounds / tokens", "tok/s", "vs mlx-lm greedy", "Same text", "Error"], rows)
    if "mean_tokens_per_round" in result:
        gate = result["port_threshold"]

        def verdict(reading):
            return "met" if gate[reading]["met"] else "not met"

        lines += ["", f"Mean tokens per round {result['mean_tokens_per_round']} (each round's accepted draft tokens plus "
                      f"the target's own token); mean accepted draft tokens per round {result['mean_accepted_per_round']}; "
                      f"mean acceptance {result['mean_acceptance']}.", "",
                  f"Swift port threshold {gate['value']}, two readings: tokens per round ≥ {gate['value']}: "
                  f"{verdict('tokens_per_round')}; accepted draft tokens per round ≥ {gate['value']}: "
                  f"{verdict('accepted_draft_tokens_per_round')}. Plan section 3 says \"accepted tokens per round\" "
                  f"and WP01 item 8 says \"tokens per round\"; when the readings disagree, the orchestrator decides."]
    if result.get("error"):
        lines += ["", f"Error: {result['error']}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------


def environment():
    env = {"python": platform.python_version(), "machine": platform.machine()}
    try:
        import importlib.metadata as metadata

        for name in ("mlx", "mlx-lm", "transformers", "huggingface_hub"):
            try:
                env[name] = metadata.version(name)
            except Exception:  # noqa: BLE001
                pass
        import mlx.core as mx

        info = mx.device_info() if hasattr(mx, "device_info") else {}
        env["device"] = {k: info.get(k) for k in ("device_name", "architecture", "memory_size", "max_recommended_working_set_size") if k in info}
    except Exception as error:  # noqa: BLE001
        env["error"] = describe(error)
    return env


def self_test():
    """Checks the pure parts (drafters, simulation, parsing) before spending GPU time."""
    import random

    rng = random.Random(1)
    for _ in range(300):
        prompt = [rng.randrange(6) for _ in range(rng.randrange(1, 30))]
        trajectory = [rng.randrange(6) for _ in range(rng.randrange(1, 30))]
        excluded = {5}
        fast = prompt_lookup_proposals(prompt, trajectory, 2, 4, 8, excluded)
        for pos in range(len(trajectory)):
            ref = prompt_lookup_reference(prompt + trajectory[:pos], 2, 4, 8, excluded)
            assert fast[pos] == ref, (prompt, trajectory, pos, fast[pos], ref)
    traj = [1, 2, 3, 4, 5]
    perfect = [(traj[i:], "x", 3) for i in range(5)]
    s = simulate(traj, perfect, [([t], [1.0]) for t in traj], 4)
    assert s["tokens"] == 5 and s["spec_rounds"] == 1 and s["rounds"] == 1, s
    assert abs(s["expected_tokens_t07"] - 5.0) < 1e-9, s
    assert relative_cost(7) == (2.3 + 3.07) / 2 and relative_cost(1) == 1.0 and relative_cost(10) > 3.4
    xml = "Sure.\n<tool_call>\n<function=create_reminder>\n<parameter=title>\nCall mum\n</parameter>\n<parameter=due>\n2026-10-07T17:00\n</parameter>\n</function>\n</tool_call>"
    score = score_tool_call({"name": "create_reminder", "args": {"title": "call mum", "due": "2026-10-07T17:00"}}, xml)
    assert score["name_ok"] and score["args_ok"], score
    js = '<tool_call>\n{"name": "set_timer", "arguments": {"seconds": 600, "label": "pasta"}}\n</tool_call>'
    assert score_tool_call({"name": "set_timer", "args": {"seconds": 600}}, js)["args_ok"]
    assert match_value("2026-10-08", "2026-10-08T00:00:00") and not match_value("2026-10-08T08:00", "2026-10-08")
    assert match_value("快递", "去取快递") and match_value("call mum", "Call Mum!") and not match_value("buy milk", "milk")
    assert match_value(1200, "1200") and match_value(["a b", "c"], "c d")
    assert score_tool_call({"name": None, "args": {}}, "Sorry, I can't do that here.")["args_ok"]
    assert not score_tool_call({"name": None, "args": {}}, js)["name_ok"]
    subset = {"local_tools": ["handoff_to_cloud", "set_timer"]}
    assert local_expect(subset, {"name": "set_timer", "args": {"seconds": 600}})["name"] == "set_timer"
    assert local_expect(subset, {"name": "list_timers", "args": {}}) == {"name": "handoff_to_cloud", "args": {}}
    assert local_expect({"local_tools": ["set_timer"]}, {"name": "list_timers", "args": {}})["name"] is None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", help="Markdown summary path (default: --out with .md).")
    parser.add_argument("--model", default=PRIMARY, help="Primary model (Woof 4B).")
    parser.add_argument("--fallback", default=FALLBACK, help="Used when mlx-lm cannot load the primary model.")
    parser.add_argument("--compare", default=COMPARE, help="Comparison model; empty to skip.")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="Prompts per set (0 = all); for quick local runs.")
    parser.add_argument("--dflash", action="store_true", help="Run the DFlash measurement instead.")
    parser.add_argument("--dflash-draft", default=DFLASH_DRAFT)
    parser.add_argument("--dflash-prompts", default=",".join(DFLASH_PROMPTS))
    parser.add_argument("--baseline", help="lab.json from the default mode, to compare DFlash speed and text.")
    args = parser.parse_args(argv)

    prompts = load_prompts(args.prompts)
    summary_path = args.summary or os.path.splitext(args.out)[0] + ".md"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if args.dflash:
        result = {"generated_at": started, "environment": environment()}
        try:
            result.update(run_dflash(prompts, args))
        except Exception as error:  # noqa: BLE001
            result["error"] = describe(error)
            result["traceback"] = traceback.format_exc(limit=6)
        write_json(args.out, result)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(dflash_markdown(result))
        return 0 if "mean_tokens_per_round" in result else 1

    self_test()
    result = {
        "generated_at": started,
        "environment": environment(),
        "settings": {"max_tokens": args.max_tokens, "ks": KS, "cost_curve": COST_CURVE, "temperature": TEMPERATURE,
                     "top_p": TOP_P, "top_k": TOP_K, "limit": args.limit},
        "notes": [],
        "models": [],
    }
    plan = [(args.model, "primary")]
    if args.compare:
        plan.append((args.compare, "comparison"))
    for repo, role in plan:
        try:
            entry = run_model(repo, prompts, args)
        except Exception as error:  # noqa: BLE001
            entry = {"repo": repo, "load_error": describe(error), "traceback": traceback.format_exc(limit=6)}
            log(f"{repo}: {entry['load_error']}")
            if role == "primary" and args.fallback:
                result["notes"].append(f"mlx-lm could not load {repo} ({entry['load_error']}); used {args.fallback} instead.")
                entry["role"] = role
                result["models"].append(entry)
                try:
                    entry = run_model(args.fallback, prompts, args)
                    role = "primary (fallback)"
                except Exception as error2:  # noqa: BLE001
                    entry = {"repo": args.fallback, "load_error": describe(error2), "traceback": traceback.format_exc(limit=6)}
        entry["role"] = role
        result["models"].append(entry)
        # Keep the measured models first, so --baseline finds the primary at index 0.
        result["models"].sort(key=lambda m: 1 if m.get("load_error") else 0)
    write_json(args.out, result)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(lab_markdown(result))
    log(f"wrote {args.out} and {summary_path}")
    return 0 if any(not m.get("load_error") for m in result["models"]) else 1


if __name__ == "__main__":
    sys.exit(main())
