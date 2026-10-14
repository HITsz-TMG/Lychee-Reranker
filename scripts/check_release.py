"""Reject private paths, credentials and artifacts in files selected for Git."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    names = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root
        )
        .decode()
        .split("\0")
    )
    patterns = {
        "private absolute path": rb"/(?:mnt|root|home|Users)/",
        "private key": b"BEGIN " + rb"(?:RSA |OPENSSH |EC )?PRIVATE KEY",
        "access token": rb"\b(?:h" + rb"f_|gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}",
        "API credential": rb"\bs" + rb"k-[A-Za-z0-9_-]{20,}",
    }
    forbidden = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".parquet", ".log"}
    errors = []
    checked = 0
    for name in sorted(set(names) - {""}):
        path = root / name
        if not path.exists():
            continue
        checked += 1
        if path.is_symlink():
            errors.append(f"{name}: symlink")
            continue
        if path.suffix in forbidden or path.stat().st_size > 5_000_000:
            errors.append(f"{name}: generated artifact or oversized file")
            continue
        content = path.read_bytes()
        for label, pattern in patterns.items():
            if re.search(pattern, content):
                errors.append(f"{name}: {label}")
    if errors:
        raise SystemExit("Release scan failed:\n" + "\n".join(errors))
    print(f"Release scan passed: {checked} files")


if __name__ == "__main__":
    main()
