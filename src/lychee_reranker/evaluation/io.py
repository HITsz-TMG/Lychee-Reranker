from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping


def stable_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def task_digest(task) -> str:
    digest = hashlib.sha256()
    for section, values in (
        (
            "meta",
            {"benchmark": task.benchmark, "name": task.name, "split": task.split},
        ),
        ("corpus", task.corpus),
        ("queries", task.queries),
        ("qrels", task.qrels),
        ("candidates", task.candidates or {}),
        ("instructions", task.instructions or {}),
    ):
        digest.update(section.encode())
        for key, value in sorted(values.items()):
            digest.update(str(key).encode())
            digest.update(b"\0")
            digest.update(
                json.dumps(
                    value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode()
            )
            digest.update(b"\n")
    return digest.hexdigest()


def candidate_digest(task) -> str:
    """Digest the exact query-to-document candidate mapping used by a task."""
    return stable_digest(task.candidates)


def _validate_run(task, run: Mapping[str, Mapping[str, float]]) -> None:
    expected_queries = set(task.queries)
    actual_queries = set(run)
    if actual_queries != expected_queries:
        raise ValueError(
            f"{task.name}: run/query coverage mismatch; "
            f"missing={sorted(expected_queries - actual_queries)[:3]}, "
            f"extra={sorted(actual_queries - expected_queries)[:3]}"
        )
    for query_id, scores in run.items():
        if not isinstance(scores, Mapping):
            raise ValueError(f"{task.name}/{query_id}: run scores must be a mapping")
        allowed = set(task.candidates[query_id]) if task.candidates is not None else set(task.corpus)
        unknown = sorted(set(scores) - allowed)
        if unknown:
            raise ValueError(f"{task.name}/{query_id}: run references unknown documents {unknown[:3]}")
        for document_id, score in scores.items():
            if not isinstance(document_id, str):
                raise ValueError(f"{task.name}/{query_id}: document IDs must be strings")
            if not math.isfinite(float(score)):
                raise ValueError(f"{task.name}/{query_id}/{document_id}: score must be finite")


def write_run(
    path: str | Path,
    *,
    task,
    run: Mapping[str, Mapping[str, float]],
    kind: str,
    model: Mapping[str, Any],
    config_digest: str,
    metrics: Mapping[str, float] | None = None,
) -> Path:
    _validate_run(task, run)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "kind": kind,
        "benchmark": task.benchmark,
        "dataset": task.name,
        "split": task.split,
        "model": dict(model),
        "config_digest": config_digest,
        "input_manifest_digest": task_digest(task),
        "candidate_manifest_digest": candidate_digest(task),
        "results": {
            query_id: [
                {"doc_id": doc_id, "score": score}
                for doc_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            ]
            for query_id, scores in sorted(run.items())
        },
    }
    if metrics is not None:
        payload["metrics"] = dict(metrics)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    return output


def read_run(path: str | Path, *, expected_task=None) -> Dict[str, Dict[str, float]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"candidate run not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("results"), dict):
        raise ValueError(f"unsupported candidate schema: {source}")
    if expected_task is not None:
        if (
            payload.get("benchmark") != expected_task.benchmark
            or payload.get("dataset") != expected_task.name
            or payload.get("split") != expected_task.split
        ):
            raise ValueError(f"candidate file does not match task {expected_task.name}: {source}")
        if payload.get("input_manifest_digest") != task_digest(expected_task):
            raise ValueError(f"candidate input manifest does not match current data: {source}")
        if payload.get("candidate_manifest_digest") != candidate_digest(expected_task):
            raise ValueError(f"candidate pool manifest does not match current data: {source}")
    result = {}
    for query_id, records in payload["results"].items():
        if not isinstance(records, list):
            raise ValueError(f"{source}: results[{query_id!r}] must be a list")
        scores = {}
        for record in records:
            if not isinstance(record, Mapping) or "doc_id" not in record or "score" not in record:
                raise ValueError(f"{source}: malformed result record for query {query_id}")
            doc_id = str(record["doc_id"])
            if doc_id in scores:
                raise ValueError(f"{source}: duplicate document {doc_id} for query {query_id}")
            score = float(record["score"])
            if not math.isfinite(score):
                raise ValueError(f"{source}: non-finite score for query {query_id}")
            scores[doc_id] = score
        result[str(query_id)] = scores
    if expected_task is not None:
        _validate_run(expected_task, result)
    return result


def write_summary(
    path: str | Path,
    rows: Mapping[str, Mapping[str, float]],
    *,
    group_rows: Mapping[str, Mapping[str, float]] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = sorted({metric for values in rows.values() for metric in values})
    lines = ["| Dataset | " + " | ".join(metrics) + " |", "|---|" + "---:|" * len(metrics)]
    for dataset, values in sorted(rows.items()):
        cells = [f"{100.0 * values.get(metric, 0.0):.2f}" for metric in metrics]
        lines.append("| " + dataset + " | " + " | ".join(cells) + " |")
    if rows:
        macro = {
            metric: sum(values.get(metric, 0.0) for values in rows.values()) / len(rows)
            for metric in metrics
        }
        lines.append(
            "| **"
            + ("Leaf macro average" if group_rows else "Macro average")
            + "** | "
            + " | ".join(f"{100.0 * macro[m]:.2f}" for m in metrics)
            + " |"
        )
    if group_rows:
        group_metrics = sorted(
            {metric for values in group_rows.values() for metric in values}
        )
        lines.append("")
        lines.append("LMEB group macro")
        lines.append("| Group | " + " | ".join(group_metrics) + " |")
        lines.append("|---|" + "---:|" * len(group_metrics))
        for group, values in sorted(group_rows.items()):
            lines.append(
                "| "
                + group
                + " | "
                + " | ".join(f"{100.0 * values.get(metric, 0.0):.2f}" for metric in group_metrics)
                + " |"
            )
        group_macro = {
            metric: sum(values.get(metric, 0.0) for values in group_rows.values())
            / len(group_rows)
            for metric in group_metrics
        }
        lines.append(
            "| **Group macro average** | "
            + " | ".join(f"{100.0 * group_macro[m]:.2f}" for m in group_metrics)
            + " |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
