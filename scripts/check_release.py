"""Reject private paths, credentials, and artifacts from a release candidate.

The normal scan covers both the checked-out files and the index.  This is
deliberate: a staged blob can differ from its worktree file, and a staged
deletion can leave a sensitive blob unavailable through the filesystem.  The
optional ``--history`` scan walks only commits reachable from ``HEAD``; it
does not inspect unrelated local branches.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable


MAX_FILE_SIZE = 5_000_000
FORBIDDEN_SUFFIXES = frozenset(
    {
        ".bin",
        ".ckpt",
        ".log",
        ".msgpack",
        ".npy",
        ".npz",
        ".onnx",
        ".parquet",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
        ".safetensors",
    }
)


# Keep the path expression narrow.  System paths such as /lib and /usr/lib,
# placeholder examples under /path/to, and loopback fixtures are intentionally
# outside this expression.
_PRIVATE_PATH = re.compile(
    rb"/(?:mnt|root|home|Users|private|var/tmp)/",
    re.IGNORECASE,
)
_WINDOWS_PRIVATE_PATH = re.compile(
    rb"[A-Za-z]:[\\/]+(?:Users|home|private)[\\/]+",
    re.IGNORECASE,
)

_PATTERNS = (
    ("private absolute path", _PRIVATE_PATH),
    ("private absolute path", _WINDOWS_PRIVATE_PATH),
    (
        "private key",
        re.compile(rb"BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY", re.IGNORECASE),
    ),
    (
        "access token",
        re.compile(
            rb"\b(?:h"
            rb"f_|gh[pousr]_|github_pat_|glpat-|xox[baprs]-|npm_|pypi-)"
            rb"[A-Za-z0-9_.-]{20,}"
        ),
    ),
    (
        "API credential",
        re.compile(rb"\bsk-(?:proj-|admin-)?[A-Za-z0-9_-]{20,}"),
    ),
    (
        "cloud credential",
        re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    ),
    (
        "cloud credential",
        re.compile(rb"\bAIza[0-9A-Za-z_-]{30,}\b"),
    ),
    (
        "authorization credential",
        re.compile(rb"\bBearer[ \t]+[A-Za-z0-9._~+/=-]{20,}", re.IGNORECASE),
    ),
)

_ASSIGNMENT_PATTERN = re.compile(
    rb"\b(?:api[_-]?key|access[_-]?(?:key|token)|"
    rb"secret(?:[_-](?:access[_-]?)?key)?|password|passwd|token)"
    rb"[ \t]*[:=][ \t]*[\"']?"
    rb"([A-Za-z0-9][A-Za-z0-9._~+/=-]{15,})",
    re.IGNORECASE,
)
_PLACEHOLDER_VALUES = frozenset(
    {
        b"changeme",
        b"dummy",
        b"example",
        b"fixture",
        b"placeholder",
        b"replace-me",
        b"test",
        b"testing",
        b"your-token-here",
    }
)


class _Errors:
    def __init__(self) -> None:
        self.values: list[str] = []
        self._seen: set[str] = set()

    def add(self, scope: str, name: str, label: str) -> None:
        # Never include matching bytes in diagnostics.  A path is retained so
        # a release author can locate the offending file without exposing a
        # credential value in CI logs.
        value = f"{scope}:{name}: {label}"
        if value not in self._seen:
            self._seen.add(value)
            self.values.append(value)


def _git_prefix(root: Path) -> list[str]:
    """Return explicit repository arguments, including for linked worktrees."""

    dot_git = root / ".git"
    if dot_git.is_file():
        first_line = dot_git.read_text(encoding="utf-8").splitlines()[0]
        if not first_line.startswith("gitdir:"):
            raise RuntimeError(f"invalid Git worktree metadata in {dot_git}")
        git_dir = Path(first_line[7:].strip())
        if not git_dir.is_absolute():
            git_dir = (dot_git.parent / git_dir).resolve()
    elif dot_git.is_dir():
        git_dir = dot_git
    else:
        raise RuntimeError(f"not a Git worktree: {root}")
    return ["--git-dir", str(git_dir), "--work-tree", str(root)]


def _git_bytes(root: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    # The explicit worktree/git-dir pair is precise for both ordinary and
    # linked worktrees; no global safe-directory trust is needed.
    command = ["git", *_git_prefix(root), *arguments]
    try:
        result = subprocess.run(
            command,
            cwd=root,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Git query failed while checking the release") from error
    return result.stdout


def _nul_paths(payload: bytes) -> list[str]:
    return [os.fsdecode(value) for value in payload.split(b"\0") if value]


def _index_entries(root: Path) -> list[tuple[str, str, str]]:
    """Return ``(mode, object ID, path)`` entries from the current index."""

    entries = []
    for item in _git_bytes(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not item:
            continue
        try:
            metadata, path = item.split(b"\t", 1)
            mode, object_id, _stage = metadata.split()
        except ValueError as error:
            raise RuntimeError("unexpected Git index entry") from error
        entries.append((mode.decode("ascii"), object_id.decode("ascii"), os.fsdecode(path)))
    return entries


def _batch_blob_sizes(root: Path, object_ids: Iterable[str]) -> dict[str, int]:
    unique_ids = sorted(set(object_ids))
    if not unique_ids:
        return {}
    payload = ("\n".join(unique_ids) + "\n").encode("ascii")
    output = _git_bytes(root, "cat-file", "--batch-check", input_bytes=payload)
    lines = output.splitlines()
    if len(lines) != len(unique_ids):
        raise RuntimeError("truncated Git object response")
    result: dict[str, int] = {}
    for expected, line in zip(unique_ids, lines):
        header = line.split()
        if len(header) != 3 or header[0].decode("ascii") != expected:
            raise RuntimeError("unexpected Git object response")
        object_type = header[1].decode("ascii")
        try:
            size = int(header[2])
        except ValueError as error:
            raise RuntimeError("invalid Git object size") from error
        if object_type != "blob":
            raise RuntimeError("release index contains a non-blob entry")
        result[expected] = size
    return result


def _batch_blobs(root: Path, object_ids: Iterable[str]) -> dict[str, tuple[int, bytes | None]]:
    """Return sizes for all blobs and bytes only for blobs below the size cap."""

    sizes = _batch_blob_sizes(root, object_ids)
    readable_ids = [object_id for object_id, size in sizes.items() if size <= MAX_FILE_SIZE]
    result: dict[str, tuple[int, bytes | None]] = {
        object_id: (size, None) for object_id, size in sizes.items()
    }
    if not readable_ids:
        return result

    payload = ("\n".join(readable_ids) + "\n").encode("ascii")
    output = _git_bytes(root, "cat-file", "--batch", input_bytes=payload)
    offset = 0
    for expected in readable_ids:
        header_end = output.find(b"\n", offset)
        if header_end < 0:
            raise RuntimeError("truncated Git object response")
        header = output[offset:header_end].split()
        offset = header_end + 1
        if len(header) != 3 or header[0].decode("ascii") != expected:
            raise RuntimeError("unexpected Git object response")
        if header[1] != b"blob":
            raise RuntimeError("release index contains a non-blob entry")
        try:
            size = int(header[2])
        except ValueError as error:
            raise RuntimeError("invalid Git object size") from error
        if size != sizes[expected]:
            raise RuntimeError("Git object size changed during release scan")
        data = output[offset : offset + size]
        if len(data) != size or output[offset + size : offset + size + 1] != b"\n":
            raise RuntimeError("truncated Git blob response")
        offset += size + 1
        result[expected] = (size, data)
    return result


def _metadata_labels(name: str, size: int) -> list[str]:
    labels = []
    if Path(name).suffix.lower() in FORBIDDEN_SUFFIXES:
        labels.append("generated artifact or oversized file")
    elif size > MAX_FILE_SIZE:
        labels.append("generated artifact or oversized file")
    return labels


def _scan_content(data: bytes, scope: str, name: str, errors: _Errors) -> None:
    if b"\0" in data:
        errors.add(scope, name, "binary file")
        return
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        errors.add(scope, name, "binary file")
        return
    for label, pattern in _PATTERNS:
        if pattern.search(data):
            errors.add(scope, name, label)
    for match in _ASSIGNMENT_PATTERN.finditer(data):
        value = match.group(1).strip(b"\"'").lower()
        if value in _PLACEHOLDER_VALUES or value.startswith((b"your-", b"<")):
            continue
        errors.add(scope, name, "credential assignment")


def _scan_blob_entries(
    entries: Iterable[tuple[str, str, str]],
    blobs: dict[str, tuple[int, bytes | None]],
    scope: str,
    errors: _Errors,
) -> int:
    checked = 0
    for mode, object_id, name in entries:
        checked += 1
        size, data = blobs[object_id]
        if mode == "120000":
            errors.add(scope, name, "symlink")
        labels = _metadata_labels(name, size)
        for label in labels:
            errors.add(scope, name, label)
        if not labels:
            if data is None:
                raise RuntimeError("release blob content was not loaded")
            _scan_content(data, scope, name, errors)
    return checked


def _scan_index(root: Path, errors: _Errors) -> int:
    entries = _index_entries(root)
    blobs = _batch_blobs(root, (object_id for _mode, object_id, _name in entries))
    return _scan_blob_entries(entries, blobs, "index", errors)


def _scan_worktree(root: Path, errors: _Errors) -> int:
    names = _nul_paths(
        _git_bytes(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    )
    checked = 0
    for name in sorted(set(names)):
        checked += 1
        path = root / name
        # ``islink`` preserves dangling symlinks for this check.
        if os.path.islink(path):
            errors.add("worktree", name, "symlink")
            continue
        if not path.exists():
            # A tracked deletion is still covered by the index scan.  History
            # mode additionally covers a deletion that was already staged.
            continue
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
            data = path.read_bytes() if size <= MAX_FILE_SIZE else b""
        except OSError as error:
            raise RuntimeError(f"cannot read release file: {name}") from error
        labels = _metadata_labels(name, size)
        for label in labels:
            errors.add("worktree", name, label)
        if not labels:
            _scan_content(data, "worktree", name, errors)
    return checked


def _history_commits(root: Path) -> list[str]:
    return _git_bytes(root, "rev-list", "--topo-order", "HEAD").decode("ascii").splitlines()


def _history_entries(root: Path, commits: Iterable[str]) -> list[tuple[str, str, str, str]]:
    """Collect ``(commit, mode, object ID, path)`` entries reachable from HEAD."""

    entries = []
    for commit in commits:
        tree = _git_bytes(root, "ls-tree", "-r", "-z", "--full-tree", commit)
        for item in tree.split(b"\0"):
            if not item:
                continue
            try:
                metadata, path = item.split(b"\t", 1)
                mode, object_type, object_id = metadata.split()
            except ValueError as error:
                raise RuntimeError("unexpected Git tree entry") from error
            if object_type != b"blob":
                continue
            entries.append(
                (
                    commit,
                    mode.decode("ascii"),
                    object_id.decode("ascii"),
                    os.fsdecode(path),
                )
            )
    return entries


def _scan_history(root: Path, errors: _Errors) -> int:
    commits = _history_commits(root)
    checked = 0
    for commit in commits:
        message = _git_bytes(root, "show", "-s", "--format=%B", "--no-patch", commit)
        _scan_content(message, f"history:{commit[:12]}", "<commit-message>", errors)
    checked += len(commits)

    entries = _history_entries(root, commits)
    object_ids = [object_id for _commit, _mode, object_id, _name in entries]
    blobs = _batch_blobs(root, object_ids)
    for commit, mode, object_id, name in entries:
        checked += 1
        size, data = blobs[object_id]
        scope = f"history:{commit[:12]}"
        if mode == "120000":
            errors.add(scope, name, "symlink")
        labels = _metadata_labels(name, size)
        for label in labels:
            errors.add(scope, name, label)
        if not labels:
            _scan_content(data, scope, name, errors)
    return checked


def scan_repository(root: str | Path, *, history: bool = False) -> tuple[list[str], int]:
    """Scan a repository and return ``(errors, checked_entry_count)``.

    ``history=True`` deliberately follows only ``HEAD``.  Callers that want a
    release branch audit should run this command with that branch checked out,
    instead of accidentally scanning private worker branches through ``--all``.
    """

    root = Path(root).resolve()
    errors = _Errors()
    checked = _scan_index(root, errors)
    checked += _scan_worktree(root, errors)
    if history:
        checked += _scan_history(root, errors)
    return errors.values, checked


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scan files selected for a release")
    parser.add_argument(
        "--history",
        action="store_true",
        help="also scan every tree reachable from HEAD (never unrelated local branches)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(__file__).resolve().parents[1]
    try:
        errors, checked = scan_repository(root, history=args.history)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    if errors:
        raise SystemExit("Release scan failed:\n" + "\n".join(errors))
    suffix = " including HEAD history" if args.history else ""
    print(f"Release scan passed: {checked} entries{suffix}")


if __name__ == "__main__":
    main()
