#!/usr/bin/env python3
"""Download and verify the pinned MiniCPM-V 4.7 1B benchmark GGUF pair.

Only Python's standard library is required. Keep the manifest synchronized with
AssistantKit/OnDevice/MiniCPMArtifacts.swift. Downloaded files are installed only
after exact length, GGUF header, and full SHA256 verification succeeds.
"""

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import time
import urllib.request

REVISION = "8716e5209f308fd53f197ccd7d50c6fdcc2dc37d"
BASE_URL = (
    "https://huggingface.co/tc-mb/MiniCPM-V-Apps-gguf/resolve/"
    f"{REVISION}/MiniCPM-V-4.7-1B/"
)
ARTIFACTS = (
    (
        "MiniCPM-V-4_7-1B-Q4_K_M.gguf",
        672_003_584,
        "bdc2c0da2320129a4fd91b0d464c63f8047b511a7ef7f7e1d143b433c7165cd5",
    ),
    (
        "mmproj-model-f16.gguf",
        1_108_746_816,
        "febb329c29ec9f1643c9d690ca74d35f486b83fa48c2f54c108b135729f635fd",
    ),
)
CHUNK_SIZE = 1_048_576


def valid(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            return False
        stream.seek(0)
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == digest


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for filename, size, digest in ARTIFACTS:
        target = directory / filename
        if valid(target, size, digest):
            print(f"Verified cached {target}", flush=True)
            continue
        descriptor, staged_name = tempfile.mkstemp(prefix=f".{filename}.", suffix=".download", dir=directory)
        staged = Path(staged_name)
        try:
            hasher = hashlib.sha256()
            received = 0
            last_report = 0
            print(f"Downloading {filename} ({size:,} bytes)", flush=True)
            with os.fdopen(descriptor, "wb") as output:
                request = urllib.request.Request(BASE_URL + filename, headers={"User-Agent": "personal-assistant-minicpm-benchmark/1"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 200:
                        raise ValueError(f"Unexpected HTTP status {response.status}")
                    for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                        if received == 0 and chunk[:4] != b"GGUF":
                            raise ValueError("The response is not a GGUF model")
                        received += len(chunk)
                        if received > size:
                            raise ValueError("The download exceeds the pinned file size")
                        output.write(chunk)
                        hasher.update(chunk)
                        if time.monotonic() - last_report >= 5:
                            print(f"  {received / size:.1%} ({received:,}/{size:,} bytes)", flush=True)
                            last_report = time.monotonic()
                output.flush()
                os.fsync(output.fileno())
            if received != size:
                raise ValueError(f"Incomplete download: expected {size}, received {received}")
            if hasher.hexdigest() != digest:
                raise ValueError("SHA256 does not match the pinned model")
            os.replace(staged, target)
            print(f"Verified and installed {target}", flush=True)
        finally:
            staged.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("build/minicpm-models"))
    options = parser.parse_args()
    try:
        prepare(options.directory.resolve())
    except (OSError, ValueError) as error:
        print(f"MiniCPM download failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("MiniCPM download cancelled; staging file removed.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
