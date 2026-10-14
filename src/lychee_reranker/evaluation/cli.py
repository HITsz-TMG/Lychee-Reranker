from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from lychee_reranker.inference import LycheeReranker
from lychee_reranker.train.config import load_config

from .benchmarks import (
    BEIR_RERANKER_INSTRUCTIONS,
    BEIR_RETRIEVER_INSTRUCTIONS,
    BEIR_V1_TASKS,
    DEFAULT_METRICS,
    LMEB_DIALOGUE_TASKS,
    MIRACL_LANGUAGES,
)
from .data import EvaluationTask, load_beir, load_lmeb, load_miracl
from .embedder import DenseEmbedder
from .io import read_run, stable_digest, write_run, write_summary
from .metrics import evaluate_run
from .retrieval import dense_search, dense_search_restricted, rerank


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", name)


def _tasks(config: Dict[str, Any]) -> List[EvaluationTask]:
    benchmark = config["benchmark"].lower()
    root = config["data_dir"]
    if benchmark == "beir":
        names = config.get("tasks", list(BEIR_V1_TASKS))
        split = config.get("split", "test")
        split_overrides = config.get("splits", {})
        if not isinstance(split_overrides, dict):
            raise ValueError("splits must be a mapping from BEIR task to split")
        return [
            task
            for name in names
            for task in load_beir(root, name, split_overrides.get(name, split))
        ]
    if benchmark == "miracl":
        names = config.get("tasks", list(MIRACL_LANGUAGES))
        split = config.get("split", "dev")
        return [load_miracl(root, name, split) for name in names]
    if benchmark == "lmeb":
        return load_lmeb(root, config.get("tasks", list(LMEB_DIALOGUE_TASKS)))
    raise ValueError("benchmark must be one of: beir, miracl, lmeb")


def _instruction(task: EvaluationTask, config: Dict[str, Any], *, stage: str = "rerank") -> str:
    explicit = config.get("instruction")
    if explicit is not None:
        return str(explicit)
    for key in ("instruction", "query", "query_instruction"):
        if key in task.instructions:
            return task.instructions[key]
    if task.benchmark.casefold() == "beir":
        dataset = task.name.split("/", maxsplit=1)[0].casefold()
        instructions = (
            BEIR_RETRIEVER_INSTRUCTIONS if stage == "retrieve" else BEIR_RERANKER_INSTRUCTIONS
        )
        if dataset in instructions:
            return instructions[dataset]
    return "Given a query, retrieve documents that answer the query."


def _build_embedder(config: Dict[str, Any]) -> DenseEmbedder:
    value = dict(config.get("embedder", {}))
    value.setdefault(
        "model_name_or_path", "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"
    )
    value.setdefault("max_length", 512)
    return DenseEmbedder(**value)


def _build_reranker(config: Dict[str, Any]) -> LycheeReranker:
    value = dict(config.get("reranker", {}))
    if "model_name_or_path" not in value:
        raise ValueError("reranker.model_name_or_path is required for rerank/all stages")
    value.setdefault("query_max_length", 512)
    value.setdefault("max_length", 1024)
    value.setdefault("chunk_size", 4)
    return LycheeReranker(**value)


def _aggregate_rows(rows: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Treat CQADupStack's leaves as one BEIR dataset in the macro table."""
    result = {name: dict(values) for name, values in rows.items()}
    cqa_names = [name for name in result if name.startswith("cqadupstack/")]
    if cqa_names:
        metric_names = sorted({metric for name in cqa_names for metric in result[name]})
        result["cqadupstack"] = {
            metric: sum(result[name].get(metric, 0.0) for name in cqa_names) / len(cqa_names)
            for metric in metric_names
        }
        for name in cqa_names:
            del result[name]
    return result


def _aggregate_lmeb_groups(rows: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for name, values in rows.items():
        parts = name.split("/")
        # Standard LMEB paths are category/task/leaf.  The paper macro gives
        # each task group (the six Dialogue groups by default) equal weight,
        # after averaging its leaves.
        if len(parts) >= 3 and parts[0].casefold() in {
            "dialogue",
            "episodic",
            "procedural",
            "semantic",
        }:
            group = parts[1]
        else:
            group = parts[0]
        grouped.setdefault(group, []).append(values)
    return {
        group: {
            metric: sum(values.get(metric, 0.0) for values in group_values) / len(group_values)
            for metric in sorted({metric for values in group_values for metric in values})
        }
        for group, group_values in sorted(grouped.items())
    }


def run(
    config: Dict[str, Any], stage: str, candidate_dir: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    tasks = _tasks(config)
    output_dir = Path(config["output_dir"])
    candidates_root = Path(candidate_dir or config.get("candidate_dir", output_dir / "candidates"))
    results_root = output_dir / "results"
    config_digest = stable_digest(config)
    top_k = int(config.get("retrieval_top_k", 100))
    rerank_top_k = int(config.get("rerank_top_k", 100))
    metrics = tuple(config.get("metrics", DEFAULT_METRICS[config["benchmark"].lower()]))
    ignore_identical_ids = bool(
        config.get(
            "ignore_identical_ids",
            config["benchmark"].casefold() == "beir",
        )
    )
    embedder = _build_embedder(config) if stage in {"retrieve", "all"} else None
    reranker_model = _build_reranker(config) if stage in {"rerank", "all"} else None
    task_rows = {}
    for task in tasks:
        candidate_path = candidates_root / f"{_safe_name(task.name)}.json"
        if stage in {"retrieve", "all"}:
            query_ids = sorted(task.queries)
            document_ids = sorted(task.corpus)
            query_embeddings = embedder.encode_queries(
                [task.queries[query_id] for query_id in query_ids],
                _instruction(task, config, stage="retrieve"),
            )
            document_embeddings = embedder.encode_corpus(
                [task.corpus[document_id] for document_id in document_ids]
            )
            if task.candidates is None:
                candidates = dense_search(
                    query_embeddings,
                    document_embeddings,
                    query_ids,
                    document_ids,
                    top_k=top_k,
                    ignore_identical_ids=ignore_identical_ids,
                )
            else:
                candidates = dense_search_restricted(
                    query_embeddings,
                    document_embeddings,
                    query_ids,
                    document_ids,
                    task.candidates,
                    top_k=top_k,
                    ignore_identical_ids=ignore_identical_ids,
                )
            write_run(
                candidate_path,
                task=task,
                run=candidates,
                kind="candidates",
                model={
                    "id": embedder.model_name_or_path,
                    "revision": embedder.revision,
                    "top_k": top_k,
                    "ignore_identical_ids": ignore_identical_ids,
                },
                config_digest=config_digest,
            )
        else:
            candidates = read_run(
                candidate_path,
                expected_task=task,
                expected_ignore_identical_ids=ignore_identical_ids,
            )
        if stage == "retrieve":
            continue
        reranked = rerank(
            task,
            candidates,
            reranker_model,
            top_k=rerank_top_k,
            instruction=_instruction(task, config, stage="rerank"),
            ignore_identical_ids=ignore_identical_ids,
        )
        task_metrics = evaluate_run(task.qrels, reranked, metrics)
        task_rows[task.name] = task_metrics
        write_run(
            results_root / f"{_safe_name(task.name)}.json",
            task=task,
            run=reranked,
            kind="reranked",
            model={
                "id": reranker_model.model_name_or_path,
                "revision": reranker_model.revision,
                "chunk_size": reranker_model.chunk_size,
                "ignore_identical_ids": ignore_identical_ids,
            },
            config_digest=config_digest,
            metrics=task_metrics,
        )
    if stage != "retrieve":
        benchmark = config["benchmark"].lower()
        summary = _aggregate_rows(task_rows)
        group_summary = _aggregate_lmeb_groups(task_rows) if benchmark == "lmeb" else {}
        metric_names = sorted({metric for values in summary.values() for metric in values})
        leaf_macro = {
            metric: sum(values.get(metric, 0.0) for values in summary.values()) / len(summary)
            for metric in metric_names
        }
        group_metric_names = sorted(
            {metric for values in group_summary.values() for metric in values}
        )
        group_macro = {
            metric: sum(values.get(metric, 0.0) for values in group_summary.values())
            / len(group_summary)
            for metric in group_metric_names
        }
        macro = group_macro if benchmark == "lmeb" else leaf_macro
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "benchmark": config["benchmark"].lower(),
                    "tasks": task_rows,
                    "aggregate": summary,
                    "macro_average": macro,
                    "leaf_macro_average": leaf_macro,
                    "group_aggregate": group_summary,
                    "group_macro_average": group_macro,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        write_summary(output_dir / "summary.md", summary, group_rows=group_summary)
        return summary
    return {}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Lychee-Reranker-V1 on local data")
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", choices=["beir", "miracl", "lmeb"])
    parser.add_argument("--stage", choices=["retrieve", "rerank", "all"], default="all")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--candidate-dir")
    parser.add_argument("--reranker-name-or-path")
    parser.add_argument("--embedder-name-or-path")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    for key in ("benchmark", "data_dir", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.reranker_name_or_path:
        config.setdefault("reranker", {})["model_name_or_path"] = args.reranker_name_or_path
    if args.embedder_name_or_path:
        config.setdefault("embedder", {})["model_name_or_path"] = args.embedder_name_or_path
    missing = [key for key in ("benchmark", "data_dir", "output_dir") if key not in config]
    if missing:
        raise ValueError(f"missing evaluation configuration keys: {missing}")
    run(config, args.stage, candidate_dir=args.candidate_dir)


if __name__ == "__main__":
    main()
