from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping


def stable_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def task_digest(task) -> str:
    digest = hashlib.sha256()
    for section, values in (
        ("meta", {"name": task.name, "split": task.split}),
        ("corpus", task.corpus),
        ("queries", task.queries),
        ("qrels", task.qrels),
        ("candidates", task.candidates or {}),
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
        ):
            raise ValueError(f"candidate file does not match task {expected_task.name}: {source}")
        if payload.get("input_manifest_digest") != task_digest(expected_task):
            raise ValueError(f"candidate input manifest does not match current data: {source}")
    result = {}
    for query_id, records in payload["results"].items():
        if not isinstance(records, list):
            raise ValueError(f"{source}: results[{query_id!r}] must be a list")
        scores = {}
        for record in records:
            doc_id = str(record["doc_id"])
            if doc_id in scores:
                raise ValueError(f"{source}: duplicate document {doc_id} for query {query_id}")
            scores[doc_id] = float(record["score"])
        result[str(query_id)] = scores
    return result


def write_summary(path: str | Path, rows: Mapping[str, Mapping[str, float]]) -> None:
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
            "| **Macro average** | " + " | ".join(f"{100.0 * macro[m]:.2f}" for m in metrics) + " |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
