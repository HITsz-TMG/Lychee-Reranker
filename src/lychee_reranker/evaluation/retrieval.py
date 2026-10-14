from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence

import numpy as np


def dense_search(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    query_ids: Sequence[str],
    document_ids: Sequence[str],
    *,
    top_k: int = 100,
    ignore_identical_ids: bool = False,
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
    if len(set(query_ids)) != len(query_ids) or len(set(document_ids)) != len(document_ids):
        raise ValueError("query and document IDs must be unique")
    query_array = np.asarray(query_embeddings, dtype=np.float32)
    document_array = np.asarray(document_embeddings, dtype=np.float32)
    if not np.isfinite(query_array).all() or not np.isfinite(document_array).all():
        raise ValueError("embeddings must contain only finite values")
    # Search one extra item when identical query/document IDs are excluded so
    # filtering does not reduce a requested top-k result by one rank.
    search_limit = min(top_k + int(ignore_identical_ids), len(document_ids))
    limit = min(top_k, len(document_ids))
    if not limit:
        return {query_id: {} for query_id in query_ids}
    try:
        import faiss

        index = faiss.IndexFlatIP(document_embeddings.shape[1])
        index.add(document_array)
        scores, indices = index.search(query_array, search_limit)
    except ImportError:
        # Keep FAISS optional.  Compute one query row at a time so a local
        # CPU-only install remains runnable without allocating a QxD matrix.
        indices_rows = []
        scores_rows = []
        for query in query_array:
            row_scores = query @ document_array.T
            row_indices = np.argsort(-row_scores, kind="stable")[:search_limit]
            indices_rows.append(row_indices)
            scores_rows.append(row_scores[row_indices])
        indices = np.asarray(indices_rows, dtype=np.int64)
        scores = np.asarray(scores_rows, dtype=np.float32)
    result: Dict[str, Dict[str, float]] = {}
    for row, query_id in enumerate(query_ids):
        pairs = [
            (document_ids[index], float(score))
            for index, score in zip(indices[row], scores[row])
            if not (ignore_identical_ids and document_ids[index] == query_id)
        ][:limit]
        if any(not math.isfinite(score) for _, score in pairs):
            raise ValueError(f"{query_id}: retrieval produced a non-finite score")
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
    ignore_identical_ids: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Search a query-specific candidate pool, as required by LMEB."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if len(set(query_ids)) != len(query_ids) or len(set(document_ids)) != len(document_ids):
        raise ValueError("query and document IDs must be unique")
    if set(candidates) != set(query_ids):
        raise ValueError("candidate/query coverage mismatch")
    query_array = np.asarray(query_embeddings, dtype=np.float32)
    document_array = np.asarray(document_embeddings, dtype=np.float32)
    if query_array.ndim != 2 or document_array.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if query_array.shape[1] != document_array.shape[1]:
        raise ValueError("query/document embedding dimensions differ")
    if (
        len(query_ids) != query_array.shape[0]
        or len(document_ids) != document_array.shape[0]
    ):
        raise ValueError("embedding counts do not match IDs")
    if not np.isfinite(query_array).all() or not np.isfinite(document_array).all():
        raise ValueError("embeddings must contain only finite values")
    document_index = {doc_id: index for index, doc_id in enumerate(document_ids)}
    result = {}
    for row, query_id in enumerate(query_ids):
        candidate_values = candidates[query_id]
        candidate_ids = list(candidate_values)
        if not candidate_ids:
            raise ValueError(f"{query_id}: candidate pool is empty")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"{query_id}: candidate pool contains duplicates")
        missing = [doc_id for doc_id in candidate_ids if doc_id not in document_index]
        if missing:
            raise ValueError(f"{query_id}: candidate pool references unknown docs {missing[:3]}")
        candidate_pairs = [
            (doc_id, document_index[doc_id])
            for doc_id in candidate_ids
            if not (ignore_identical_ids and doc_id == query_id)
        ]
        if not candidate_pairs:
            result[query_id] = {}
            continue
        indices = [index for _, index in candidate_pairs]
        scores = document_array[indices] @ query_array[row]
        if not np.isfinite(scores).all():
            raise ValueError(f"{query_id}: retrieval produced a non-finite score")
        # FlagEmbedding's LMEB helper uses NumPy's default argsort on this
        # exact candidate-score vector.  Preserve its quicksort tie order;
        # a lexical or stable secondary key can change top-k membership.
        order = np.argsort(-scores)
        pairs = [
            (candidate_pairs[index][0], float(scores[index]))
            for index in order[:top_k]
        ]
        result[query_id] = dict(pairs)
    return result


def restrict_candidates(
    run: Mapping[str, Mapping[str, float]],
    candidates: Optional[Mapping[str, Sequence[str]]],
) -> Dict[str, Dict[str, float]]:
    if candidates is None:
        return {query_id: dict(scores) for query_id, scores in run.items()}
    if set(run) != set(candidates):
        raise ValueError("candidate/query coverage mismatch")
    if any(not math.isfinite(float(score)) for scores in run.values() for score in scores.values()):
        raise ValueError("candidate scores must be finite")
    return {
        query_id: {
            doc_id: score for doc_id, score in run[query_id].items() if doc_id in set(doc_ids)
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
    ignore_identical_ids: bool = False,
) -> Dict[str, Dict[str, float]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if set(candidates) != set(task.queries):
        raise ValueError(f"{task.name}: candidate/query coverage mismatch")
    result = {}
    for query_id, query in task.queries.items():
        candidate_scores = candidates[query_id]
        unknown = sorted(set(candidate_scores) - set(task.corpus))
        if unknown:
            raise ValueError(
                f"{task.name}/{query_id}: candidates reference unknown docs {unknown[:3]}"
            )
        if any(not math.isfinite(float(score)) for score in candidate_scores.values()):
            raise ValueError(f"{task.name}/{query_id}: candidate scores must be finite")
        if task.candidates is not None:
            allowed = set(task.candidates[query_id])
            outside_pool = sorted(set(candidate_scores) - allowed)
            if outside_pool:
                raise ValueError(
                    f"{task.name}/{query_id}: candidate run contains docs outside its pool "
                    f"{outside_pool[:3]}"
                )
        ranked_ids = [
            doc_id
            for doc_id, _ in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        ][:top_k]
        if ignore_identical_ids:
            ranked_ids = [doc_id for doc_id in ranked_ids if doc_id != query_id]
        scores = reranker.predict(
            [(query, task.corpus[doc_id]) for doc_id in ranked_ids], instruction=instruction
        )
        if len(scores) != len(ranked_ids):
            raise ValueError(
                f"{task.name}/{query_id}: reranker returned {len(scores)} scores for "
                f"{len(ranked_ids)} documents"
            )
        if any(not math.isfinite(float(score)) for score in scores):
            raise ValueError(f"{task.name}/{query_id}: reranker returned non-finite scores")
        # EvalReranker builds its output mapping in sentence-pair order.  Keep
        # that stable order when reranker scores tie for custom MRR parity.
        pairs = sorted(zip(ranked_ids, scores), key=lambda item: item[1], reverse=True)
        result[query_id] = dict(pairs)
    return result
