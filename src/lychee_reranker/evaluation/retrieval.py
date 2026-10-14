from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np


def dense_search(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    query_ids: Sequence[str],
    document_ids: Sequence[str],
    *,
    top_k: int = 100,
) -> Dict[str, Dict[str, float]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if query_embeddings.ndim != 2 or document_embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if query_embeddings.shape[1] != document_embeddings.shape[1]:
        raise ValueError("query/document embedding dimensions differ")
    if (
        len(query_ids) != query_embeddings.shape[0]
        or len(document_ids) != document_embeddings.shape[0]
    ):
        raise ValueError("embedding counts do not match IDs")
    limit = min(top_k, len(document_ids))
    if not limit:
        return {query_id: {} for query_id in query_ids}
    try:
        import faiss

        index = faiss.IndexFlatIP(document_embeddings.shape[1])
        index.add(np.asarray(document_embeddings, dtype=np.float32))
        scores, indices = index.search(np.asarray(query_embeddings, dtype=np.float32), limit)
    except ImportError:
        if len(document_ids) > 50_000:
            raise ImportError("faiss-cpu is required for corpora larger than 50,000 documents")
        all_scores = (
            np.asarray(query_embeddings, dtype=np.float32)
            @ np.asarray(document_embeddings, dtype=np.float32).T
        )
        indices = np.argsort(-all_scores, axis=1, kind="stable")[:, :limit]
        scores = np.take_along_axis(all_scores, indices, axis=1)
    result: Dict[str, Dict[str, float]] = {}
    for row, query_id in enumerate(query_ids):
        pairs = [
            (document_ids[index], float(score)) for index, score in zip(indices[row], scores[row])
        ]
        pairs.sort(key=lambda item: (-item[1], item[0]))
        result[query_id] = dict(pairs[:limit])
    return result


def dense_search_restricted(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    query_ids: Sequence[str],
    document_ids: Sequence[str],
    candidates: Mapping[str, Sequence[str]],
    *,
    top_k: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Search a query-specific candidate pool, as required by LMEB."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    document_index = {doc_id: index for index, doc_id in enumerate(document_ids)}
    result = {}
    for row, query_id in enumerate(query_ids):
        candidate_ids = list(dict.fromkeys(candidates.get(query_id, [])))
        missing = [doc_id for doc_id in candidate_ids if doc_id not in document_index]
        if missing:
            raise ValueError(f"{query_id}: candidate pool references unknown docs {missing[:3]}")
        indices = [document_index[doc_id] for doc_id in candidate_ids]
        if not indices:
            result[query_id] = {}
            continue
        scores = np.asarray(document_embeddings[indices], dtype=np.float32) @ np.asarray(
            query_embeddings[row], dtype=np.float32
        )
        pairs = sorted(zip(candidate_ids, scores.tolist()), key=lambda item: (-item[1], item[0]))
        result[query_id] = dict(pairs[:top_k])
    return result


def restrict_candidates(
    run: Mapping[str, Mapping[str, float]],
    candidates: Optional[Mapping[str, Sequence[str]]],
) -> Dict[str, Dict[str, float]]:
    if candidates is None:
        return {query_id: dict(scores) for query_id, scores in run.items()}
    return {
        query_id: {
            doc_id: score
            for doc_id, score in run.get(query_id, {}).items()
            if doc_id in set(doc_ids)
        }
        for query_id, doc_ids in candidates.items()
    }


def rerank(
    task,
    candidates: Mapping[str, Mapping[str, float]],
    reranker,
    *,
    top_k: int = 100,
    instruction: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    result = {}
    for query_id, query in task.queries.items():
        candidate_scores = candidates.get(query_id, {})
        ranked_ids = [
            doc_id
            for doc_id, _ in sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
        ][:top_k]
        missing = [doc_id for doc_id in ranked_ids if doc_id not in task.corpus]
        if missing:
            raise ValueError(
                f"{task.name}/{query_id}: candidates reference unknown docs {missing[:3]}"
            )
        scores = reranker.predict(
            [(query, task.corpus[doc_id]) for doc_id in ranked_ids], instruction=instruction
        )
        pairs = sorted(zip(ranked_ids, scores), key=lambda item: (-item[1], item[0]))
        result[query_id] = dict(pairs)
    return result
