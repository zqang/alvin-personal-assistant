#!/usr/bin/env python3
"""Convert the pinned official DSpark draft to the pinned mobile runtime's GGUF.

The upstream converter does not yet register Qwen3DSparkModel. This narrow
converter maps its actual tensors to src/models/dflash.cpp, preserves its full
RoPE and target-layer configuration, and copies the paired target's tokenizer.
It never rewrites the checkpoint as an unrelated architecture.

Requires numpy, tqdm and pyyaml plus the runtime's gguf-py. Build the quantizer:
  CMAKE --build build/minicpm/macos-arm64 --config Release --target llama-quantize
Download config.json and model.safetensors from SOURCE_URL below before running.
Every pin comes from the draftProvenance of MiniCPM35BManifest.json, which the app
reads too. All three inputs are SHA-256 checked, the output has a provenance JSON
sidecar, and an output that differs from the manifest's pin fails the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = json.loads((ROOT / "MiniCPM/Sources/MiniCPM/MiniCPM35BManifest.json").read_text())["draftProvenance"]
RUNTIME_REVISION = PROVENANCE["runtimeRevision"]
SOURCE_REPOSITORY = PROVENANCE["sourceRepository"]
SOURCE_REVISION = PROVENANCE["sourceRevision"]
SOURCE_URL = f"https://huggingface.co/{SOURCE_REPOSITORY}/resolve/{SOURCE_REVISION}"
CONFIG_SHA = PROVENANCE["sourceConfigSHA256"]
WEIGHTS_SHA = PROVENANCE["sourceWeightsSHA256"]
TARGET_SHA = PROVENANCE["pairedTargetSHA256"]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require_sha(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual}")


def tensor_mapping(config: dict) -> dict[str, tuple[str, list[int]]]:
    h = config["hidden_size"]
    ff = config["intermediate_size"]
    head = config["head_dim"]
    q = config["num_attention_heads"] * head
    kv = config["num_key_value_heads"] * head
    rank = config["markov_rank"]
    mapping = {
        "fc.weight": ("fc.weight", [h, h * len(config["dflash_config"]["target_layer_ids"])]),
        "hidden_norm.weight": ("enc.output_norm.weight", [h]),
        "norm.weight": ("output_norm.weight", [h]),
        "markov_head.markov_w1.weight": ("markov_w1.weight", [config["vocab_size"], rank]),
        "markov_head.markov_w2.weight": ("markov_w2.weight", [config["vocab_size"], rank]),
        "confidence_head.proj.weight": ("conf_proj.weight", [1, h + rank]),
        "confidence_head.proj.bias": ("conf_proj.bias", [1]),
    }
    layer_mapping = {
        "input_layernorm.weight": ("attn_norm.weight", [h]),
        "post_attention_layernorm.weight": ("ffn_norm.weight", [h]),
        "self_attn.q_proj.weight": ("attn_q.weight", [q, h]),
        "self_attn.k_proj.weight": ("attn_k.weight", [kv, h]),
        "self_attn.v_proj.weight": ("attn_v.weight", [kv, h]),
        "self_attn.o_proj.weight": ("attn_output.weight", [h, q]),
        "self_attn.q_norm.weight": ("attn_q_norm.weight", [head]),
        "self_attn.k_norm.weight": ("attn_k_norm.weight", [head]),
        "mlp.gate_proj.weight": ("ffn_gate.weight", [ff, h]),
        "mlp.up_proj.weight": ("ffn_up.weight", [ff, h]),
        "mlp.down_proj.weight": ("ffn_down.weight", [h, ff]),
    }
    for i in range(config["num_hidden_layers"]):
        for name, (mapped, shape) in layer_mapping.items():
            mapping[f"layers.{i}.{name}"] = (f"blk.{i}.{mapped}", shape)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, default=ROOT / "build/minicpm-dspark-source")
    parser.add_argument("--target-model", type=Path, default=ROOT / "build/minicpm-35b-models/MiniCPM-V-4_7-35B-A3B-mix_q3gud_im-les.gguf")
    parser.add_argument("--runtime-source", type=Path, default=ROOT / f"build/minicpm/llama.cpp-omni-{RUNTIME_REVISION}")
    parser.add_argument("--quantizer", type=Path, default=ROOT / "build/minicpm/macos-arm64/bin/Release/llama-quantize")
    parser.add_argument("--output", type=Path, default=ROOT / "build/minicpm-35b-models/MiniCPMV47-DSpark-Q8_0.gguf")
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime_source / "gguf-py"))
    import gguf
    import numpy as np

    config_path = args.source_directory / "config.json"
    weights_path = args.source_directory / "model.safetensors"
    for path, expected in ((config_path, CONFIG_SHA), (weights_path, WEIGHTS_SHA), (args.target_model, TARGET_SHA)):
        require_sha(path, expected)
    config = json.loads(config_path.read_text())
    if config["architectures"] != ["Qwen3DSparkModel"]:
        raise ValueError("Expected the pinned Qwen3DSparkModel checkpoint")

    with weights_path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    header.pop("__metadata__", None)
    mapping = tensor_mapping(config)
    if set(header) != set(mapping):
        raise ValueError("Checkpoint tensor names do not exactly match the DSpark mapping")
    if len(mapping) != PROVENANCE["tensorCount"] or sum(math.prod(value["shape"]) for value in header.values()) != PROVENANCE["parameterCount"]:
        raise ValueError("Unexpected draft tensor or parameter count")
    for name, info in header.items():
        if info["dtype"] != "BF16" or info["shape"] != mapping[name][1]:
            raise ValueError(f"Unexpected dtype/shape: {name}")
        if info["data_offsets"][1] - info["data_offsets"][0] != math.prod(info["shape"]) * 2:
            raise ValueError(f"Unexpected tensor byte count: {name}")

    target = gguf.GGUFReader(args.target_model)
    tokens = target.fields["tokenizer.ggml.tokens"].contents()
    if len(tokens) != config["vocab_size"]:
        raise ValueError("Paired target and draft vocabulary sizes differ")
    for config_key, gguf_key in (("bos_token_id", "bos"), ("eos_token_id", "eos"), ("pad_token_id", "padding")):
        if target.fields[f"tokenizer.ggml.{gguf_key}_token_id"].contents() != config[config_key]:
            raise ValueError(f"Paired tokenizer {config_key} differs")

    intermediate = args.source_directory / "MiniCPMV47-DSpark-F16.gguf"
    writer = gguf.GGUFWriter(intermediate, "dflash", use_temp_file=True)
    writer.add_name("MiniCPM-V-4.7-35B-A3B-DSpark")
    writer.add_file_type(1)
    writer.add_string("general.source.huggingface.repository", SOURCE_REPOSITORY)
    writer.add_string("minicpm.draft.source_revision", SOURCE_REVISION)
    writer.add_string("minicpm.draft.source_sha256", WEIGHTS_SHA)
    writer.add_string("minicpm.draft.runtime_revision", RUNTIME_REVISION)
    metadata = {
        "dflash.context_length": config["max_position_embeddings"],
        "dflash.embedding_length": config["hidden_size"],
        "dflash.block_count": config["num_hidden_layers"],
        "dflash.feed_forward_length": config["intermediate_size"],
        "dflash.attention.head_count": config["num_attention_heads"],
        "dflash.attention.head_count_kv": config["num_key_value_heads"],
        "dflash.attention.key_length": config["head_dim"],
        "dflash.attention.value_length": config["head_dim"],
        "dflash.rope.dimension_count": config["rope_parameters"]["rope_dim"],
        "dflash.vocab_size": config["vocab_size"],
        "dflash.block_size": config["block_size"],
    }
    for name, value in metadata.items():
        writer.add_uint32(name, value)
    writer.add_float32("dflash.attention.layer_norm_rms_epsilon", config["rms_norm_eps"])
    writer.add_float32("dflash.rope.freq_base", config["rope_parameters"]["rope_theta"])
    writer.add_array("dflash.target_layers", config["dflash_config"]["target_layer_ids"])
    for name, field in target.fields.items():
        if name.startswith("tokenizer."):
            writer.add_key_value(name, field.contents(), field.types[0], field.types[1] if len(field.types) > 1 else None)
    writer.add_uint32("tokenizer.ggml.mask_token_id", config["dflash_config"]["mask_token_id"])

    for name, info in header.items():
        mapped, shape = mapping[name]
        raw = np.memmap(weights_path, mode="r", dtype="<u2", offset=8 + header_size + info["data_offsets"][0], shape=tuple(shape))
        data = (raw.astype(np.uint32) << 16).view(np.float32)
        # Qwen3 uses NeoX RoPE; its projection matrices need no permutation.
        converted = data if data.ndim == 1 else data.astype(np.float16)
        if not np.all(np.isfinite(converted)):
            raise ValueError(f"Non-finite converted weights: {name}")
        writer.add_tensor(mapped, converted)
        print(f"{name} -> {mapped}: {shape}", flush=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staged = args.output.with_suffix(".gguf.partial")
    subprocess.run([str(args.quantizer), str(intermediate), str(staged), "Q8_0", "4"], check=True)
    converted = gguf.GGUFReader(staged)
    if converted.fields["general.architecture"].contents() != "dflash":
        raise ValueError("Quantizer changed architecture")
    if set(tensor.name for tensor in converted.tensors) != set(item[0] for item in mapping.values()):
        raise ValueError("Quantizer changed tensor names")
    for tensor in converted.tensors:
        # The upstream quantizer preserves the single-row confidence head at
        # F16; all other matrices are Q8_0 and normalization/bias vectors F32.
        allowed = (gguf.GGMLQuantizationType.F16,) if tensor.name == "conf_proj.weight" else (
            gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.F32)
        if tensor.tensor_type not in allowed:
            raise ValueError(f"Unexpected output tensor type: {tensor.name}: {tensor.tensor_type}")
    del converted
    # Checked before anything is replaced: a conversion that misses the pin leaves a good draft
    # at the output path as it was.
    provenance = {**PROVENANCE, "sizeBytes": staged.stat().st_size, "sha256": sha256(staged)}
    print(json.dumps(provenance, indent=2))
    if provenance != PROVENANCE:
        staged.unlink()
        raise ValueError("The converted draft differs from the manifest's pin, so the app wouldn't use it; the output was left as it was")
    staged.replace(args.output)
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
