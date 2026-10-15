#!/usr/bin/env python3
"""Resume and SHA256-verify the pinned MiniCPM-V 35B LES benchmark files.

Uses the same manifest as the app. Interrupted downloads stay in .partial files;
only complete, verified files are atomically installed. No third-party packages.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import time
import urllib.error
import urllib.request

MANIFEST = Path(__file__).resolve().parents[1] / "AssistantKit/Sources/AssistantKit/OnDevice/MiniCPM35BManifest.json"
CHUNK = 1_048_576


def safe_path(directory, relative):
    parts = PurePosixPath(relative).parts
    if not parts or relative.startswith("/") or "\\" in relative or any(p in ("", ".", "..") for p in relative.split("/")):
        raise ValueError(f"Unsafe artifact path: {relative}")
    path = directory.joinpath(*parts)
    if not path.resolve().is_relative_to(directory.resolve()) or path.is_symlink():
        raise ValueError(f"Artifact path leaves model directory: {relative}")
    return path


def valid(path, artifact):
    if not path.is_file() or path.stat().st_size != artifact["sizeBytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if artifact["path"].endswith(".gguf") and stream.read(4) != b"GGUF":
            return False
        stream.seek(0)
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest() == artifact["sha256"]


def download(directory, artifact, verify_only=False):
    target = safe_path(directory, artifact["path"])
    if valid(target, artifact):
        print(f"Verified {artifact['path']}", flush=True)
        return
    if verify_only or not artifact.get("sourceURL"):
        raise ValueError(f"Missing or invalid {artifact['path']}; prepare the DSpark draft with scripts/convert_minicpm_dspark.py if applicable")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = safe_path(directory, artifact["path"] + ".partial")
    size = artifact["sizeBytes"]
    for attempt in range(6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > size or (offset == size and not valid(partial, artifact)):
                partial.unlink()
                offset = 0
            if offset < size:
                headers = {"User-Agent": "personal-assistant-minicpm-benchmark/1"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(artifact["sourceURL"], headers=headers)
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status == 206:
                        content_range = response.headers.get("Content-Range", "")
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                        if not match or int(match[1]) != offset or int(match[3]) != size:
                            raise ValueError(f"Invalid resume response for {artifact['path']}: {content_range}")
                    elif response.status == 200:
                        offset = 0
                    else:
                        raise ValueError(f"Unexpected HTTP status {response.status}")
                    print(f"Downloading {artifact['path']} from {offset:,}/{size:,} bytes", flush=True)
                    with partial.open("ab" if offset else "wb") as output:
                        last_report = time.monotonic()
                        for block in iter(lambda: response.read(CHUNK), b""):
                            offset += len(block)
                            if offset > size:
                                raise ValueError("Download exceeds pinned size")
                            output.write(block)
                            if time.monotonic() - last_report > 20:
                                print(f"  {artifact['path']}: {offset / size:.1%}", flush=True)
                                last_report = time.monotonic()
                        output.flush()
                        os.fsync(output.fileno())
            if not valid(partial, artifact):
                raise ValueError(f"Incomplete or checksum mismatch: {artifact['path']}")
            os.replace(partial, target)
            print(f"Installed {artifact['path']} (SHA256 verified)", flush=True)
            return
        except (OSError, ValueError, urllib.error.URLError) as error:
            if attempt == 5:
                raise
            print(f"Retry {attempt + 1}/5 for {artifact['path']}: {error}", flush=True)
            time.sleep(min(2 ** attempt, 15))


def validate_expert_manifest(directory, artifacts):
    manifest = json.loads(safe_path(directory, "experts/manifest.json").read_text())
    allowed = {a["path"]: a["sizeBytes"] for a in artifacts if a["role"] == "expert"}
    if manifest.get("schema_version") != 1 or manifest.get("layout") != "layer_major_expert" or manifest.get("model", {}).get("arch") != "qwen35moe":
        raise ValueError("Unsupported LES expert manifest")
    entries = manifest.get("entries", [])
    if len(entries) != 120:
        raise ValueError("Expected three expert tensors in each of 40 layers")
    seen = set()
    for entry in entries:
        filename = entry["repacked_file"]
        safe_path(directory, "experts/" + filename)
        key = "experts/" + filename
        if key not in allowed or filename != f"layer_{entry['layer']:03d}.bin":
            raise ValueError("Unexpected expert shard reference")
        end = entry["repacked_offset"] + 255 * entry["expert_stride"] + entry["bytes_per_expert"]
        if min(entry["repacked_offset"], entry["expert_stride"], entry["bytes_per_expert"]) < 0 or end > allowed[key]:
            raise ValueError("Expert tensor leaves pinned shard bounds")
        seen.add(key)
    if seen != allowed.keys():
        raise ValueError("LES manifest does not cover the pinned expert set")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("build/minicpm-35b-models"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--target-only", action="store_true", help="Prepare target files while the DSpark draft is converted separately")
    options = parser.parse_args()
    try:
        directory = options.directory.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        artifacts = json.loads(MANIFEST.read_text())["artifacts"]
        if options.target_only:
            artifacts = [a for a in artifacts if a["role"] != "draft"]
        for artifact in artifacts:
            path = safe_path(directory, artifact["path"])
            if not artifact.get("sourceURL") and not valid(path, artifact):
                raise ValueError(f"Prepare the required local draft {artifact['path']} with scripts/convert_minicpm_dspark.py first, or use --target-only")
        missing = sum(a["sizeBytes"] for a in artifacts if not safe_path(directory, a["path"]).exists())
        required = missing + max(a["sizeBytes"] for a in artifacts) + 512 * 1_048_576
        if not options.verify_only and shutil.disk_usage(directory).free < required:
            raise ValueError("Insufficient storage for downloads, one replacement staging file, and 512 MiB headroom")
        with ThreadPoolExecutor(max_workers=max(1, min(options.jobs, 8))) as pool:
            list(pool.map(lambda artifact: download(directory, artifact, options.verify_only), artifacts))
        validate_expert_manifest(directory, artifacts)
        print(f"Verified {len(artifacts)} pinned artifacts, {sum(a['sizeBytes'] for a in artifacts):,} bytes", flush=True)
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"MiniCPM 35B preparation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
