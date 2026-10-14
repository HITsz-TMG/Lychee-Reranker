from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple


def ranked_document_ids(scores: Mapping[str, float]) -> list[str]:
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def _dcg(relevances: Sequence[float]) -> float:
    return sum(
        (2.0**relevance - 1.0) / math.log2(index + 2) for index, relevance in enumerate(relevances)
    )


def ndcg_at_k(qrels: Mapping[str, float], ranked: Sequence[str], k: int) -> float:
    actual = [max(0.0, float(qrels.get(doc_id, 0.0))) for doc_id in ranked[:k]]
    ideal = sorted((max(0.0, float(value)) for value in qrels.values()), reverse=True)[:k]
    denominator = _dcg(ideal)
    return 0.0 if denominator == 0 else _dcg(actual) / denominator


def recall_at_k(qrels: Mapping[str, float], ranked: Sequence[str], k: int) -> float:
    relevant = {doc_id for doc_id, score in qrels.items() if score > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def mrr_at_k(qrels: Mapping[str, float], ranked: Sequence[str], k: int) -> float:
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if qrels.get(doc_id, 0.0) > 0:
            return 1.0 / rank
    return 0.0


def _parse_metric(name: str) -> Tuple[str, int]:
    try:
        kind, raw_k = name.lower().split("@", maxsplit=1)
        k = int(raw_k)
    except (ValueError, AttributeError) as error:
        raise ValueError(f"invalid metric {name!r}; expected e.g. ndcg@10") from error
    if kind not in {"ndcg", "recall", "mrr"} or k <= 0:
        raise ValueError(f"unsupported metric: {name}")
    return kind, k


def evaluate_run(
    qrels: Mapping[str, Mapping[str, float]],
    run: Mapping[str, Mapping[str, float]],
    metrics: Sequence[str],
) -> Dict[str, float]:
    """Macro-average metrics over qrels queries, treating missing runs as empty."""
    if not qrels:
        raise ValueError("qrels must contain at least one query")
    result = {}
    for metric in metrics:
        kind, k = _parse_metric(metric)
        values = []
        for query_id, relevance in qrels.items():
            ranked = ranked_document_ids(run.get(query_id, {}))
            if kind == "ndcg":
                values.append(ndcg_at_k(relevance, ranked, k))
            elif kind == "recall":
                values.append(recall_at_k(relevance, ranked, k))
            else:
                values.append(mrr_at_k(relevance, ranked, k))
        result[metric.lower()] = sum(values) / len(values)
    return result
