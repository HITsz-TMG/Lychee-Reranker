from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence


@dataclass
class EvaluationTask:
    name: str
    benchmark: str
    split: str
    corpus: Dict[str, str]
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, float]]
    candidates: Optional[Dict[str, List[str]]] = None
    instructions: Dict[str, str] = field(default_factory=dict)


def _read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"required data file not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def _identifier(record: Mapping[str, object], names: Sequence[str], location: str) -> str:
    for name in names:
        value = record.get(name)
        if value is not None:
            return str(value)
    raise ValueError(f"{location}: none of the identifier fields {list(names)} is present")


def _text(record: Mapping[str, object], location: str) -> str:
    value = record.get("text", record.get("query"))
    if not isinstance(value, str):
        raise ValueError(f"{location}: text/query must be a string")
    return value


def _document_text(record: Mapping[str, object], location: str) -> str:
    text = record.get("text", record.get("contents"))
    if not isinstance(text, str):
        raise ValueError(f"{location}: text/contents must be a string")
    title = record.get("title", "")
    if title is None:
        title = ""
    if not isinstance(title, str):
        raise ValueError(f"{location}: title must be a string")
    # FlagEmbedding's BEIR/MIRACL/LMEB loaders concatenate title and body
    # with one space.  Keep the same text contract for retrieval parity.
    return f"{title} {text}".strip()


def _load_corpus(path: Path, id_fields: Sequence[str]) -> Dict[str, str]:
    result = {}
    for index, record in enumerate(_read_jsonl(path), start=1):
        doc_id = _identifier(record, id_fields, f"{path}:{index}")
        if doc_id in result:
            raise ValueError(f"{path}:{index}: duplicate document id {doc_id}")
        result[doc_id] = _document_text(record, f"{path}:{index}")
    return result


def _load_queries(path: Path, id_fields: Sequence[str]):
    queries: Dict[str, str] = {}
    metadata: Dict[str, Mapping[str, object]] = {}
    for index, record in enumerate(_read_jsonl(path), start=1):
        query_id = _identifier(record, id_fields, f"{path}:{index}")
        if query_id in queries:
            raise ValueError(f"{path}:{index}: duplicate query id {query_id}")
        queries[query_id] = _text(record, f"{path}:{index}")
        metadata[query_id] = record
    return queries, metadata


def _load_jsonl_qrels(
    path: Path,
    query_fields: Sequence[str],
    document_fields: Sequence[str],
    score_fields: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for index, record in enumerate(_read_jsonl(path), start=1):
        query_id = _identifier(record, query_fields, f"{path}:{index}")
        document_id = _identifier(record, document_fields, f"{path}:{index}")
        value = None
        for score_field in score_fields:
            if score_field in record:
                value = record[score_field]
                break
        if value is None:
            raise ValueError(f"{path}:{index}: relevance score is missing")
        score = float(value)
        if not math.isfinite(score):
            raise ValueError(f"{path}:{index}: relevance score must be finite")
        if document_id in result.setdefault(query_id, {}):
            raise ValueError(f"{path}:{index}: duplicate relevance for {query_id}/{document_id}")
        result[query_id][document_id] = score
    return result


def _load_tsv_qrels(path: Path) -> Dict[str, Dict[str, float]]:
    """Read BEIR's standard qrels/{split}.tsv (with or without a header)."""
    result: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not row or not any(cell.strip() for cell in row):
                continue
            normalized = [cell.strip().casefold() for cell in row]
            if normalized[0] in {"query-id", "qid", "query_id"}:
                continue
            if len(row) >= 4:
                # TREC-style qrels have qid, iteration, docid, relevance.
                query_id, document_id, raw_score = row[0], row[2], row[3]
            elif len(row) >= 3:
                query_id, document_id, raw_score = row[0], row[1], row[2]
            else:
                raise ValueError(f"{path}:{row_number}: expected qid, docid, score")
            score = float(raw_score)
            if not math.isfinite(score):
                raise ValueError(f"{path}:{row_number}: relevance score must be finite")
            query_id, document_id = str(query_id), str(document_id)
            if document_id in result.setdefault(query_id, {}):
                raise ValueError(
                    f"{path}:{row_number}: duplicate relevance for {query_id}/{document_id}"
                )
            result[query_id][document_id] = score
    return result


def _require_consistent(task: EvaluationTask) -> EvaluationTask:
    unknown_queries = sorted(set(task.qrels) - set(task.queries))
    if unknown_queries:
        raise ValueError(f"{task.name}: qrels reference unknown queries: {unknown_queries[:3]}")
    unknown_documents = sorted(
        {doc_id for rels in task.qrels.values() for doc_id in rels} - set(task.corpus)
    )
    if unknown_documents:
        raise ValueError(f"{task.name}: qrels reference unknown documents: {unknown_documents[:3]}")
    for query_id, relevance in task.qrels.items():
        if any(not math.isfinite(float(score)) for score in relevance.values()):
            raise ValueError(f"{task.name}/{query_id}: qrels contain a non-finite score")
    if task.candidates is not None:
        missing_queries = sorted(set(task.queries) - set(task.candidates))
        extra_queries = sorted(set(task.candidates) - set(task.queries))
        if missing_queries or extra_queries:
            raise ValueError(
                f"{task.name}: candidate/query coverage mismatch; "
                f"missing={missing_queries[:3]}, extra={extra_queries[:3]}"
            )
        for query_id, document_ids in task.candidates.items():
            if not document_ids:
                raise ValueError(f"{task.name}/{query_id}: candidate pool is empty")
            if len(document_ids) != len(set(document_ids)):
                raise ValueError(f"{task.name}/{query_id}: candidate pool contains duplicates")
            unknown = sorted(set(document_ids) - set(task.corpus))
            if unknown:
                raise ValueError(
                    f"{task.name}/{query_id}: candidate pool references unknown documents "
                    f"{unknown[:3]}"
                )
    return task


def load_beir(root: str | Path, dataset: str, split: str = "test") -> List[EvaluationTask]:
    base = Path(root) / dataset
    if not base.is_dir():
        raise FileNotFoundError(f"BEIR dataset directory not found: {base}")
    leaves = [base]
    if not (base / "corpus.jsonl").is_file():
        leaves = sorted(path.parent for path in base.rglob("corpus.jsonl"))
        if dataset.casefold() == "cqadupstack":
            expected = {
                "android",
                "english",
                "gaming",
                "gis",
                "mathematica",
                "physics",
                "programmers",
                "stats",
                "tex",
                "unix",
                "webmasters",
                "wordpress",
            }
            available = {path.relative_to(base).parts[0] for path in leaves}
            if available != expected:
                raise ValueError(
                    f"{dataset}: expected 12 child datasets; "
                    f"missing={sorted(expected - available)}, extra={sorted(available - expected)}"
                )
    tasks = []
    for leaf in leaves:
        suffix = "" if leaf == base else f"/{leaf.relative_to(base).as_posix()}"
        queries_path = leaf / f"{split}_queries.jsonl"
        if not queries_path.is_file():
            queries_path = leaf / "queries.jsonl"
        jsonl_qrels_path = leaf / f"{split}_qrels.jsonl"
        tsv_qrels_path = leaf / "qrels" / f"{split}.tsv"
        if jsonl_qrels_path.is_file():
            qrels = _load_jsonl_qrels(
                jsonl_qrels_path,
                ("query-id", "qid", "query_id"),
                ("corpus-id", "docid", "doc_id"),
                ("score", "relevance"),
            )
        elif tsv_qrels_path.is_file():
            qrels = _load_tsv_qrels(tsv_qrels_path)
        else:
            raise FileNotFoundError(
                f"could not find {split}_qrels.jsonl or qrels/{split}.tsv below {leaf}"
            )
        task = EvaluationTask(
            name=f"{dataset}{suffix}",
            benchmark="beir",
            split=split,
            corpus=_load_corpus(leaf / "corpus.jsonl", ("_id", "id")),
            queries=_load_queries(queries_path, ("_id", "id", "query_id", "qid"))[0],
            qrels=qrels,
        )
        tasks.append(_require_consistent(task))
    if not tasks:
        raise FileNotFoundError(f"no BEIR task data found below {base}")
    return tasks


def load_miracl(root: str | Path, language: str, split: str = "dev") -> EvaluationTask:
    base = Path(root) / language
    if not base.is_dir():
        raise FileNotFoundError(f"MIRACL language directory not found: {base}")
    task = EvaluationTask(
        name=language,
        benchmark="miracl",
        split=split,
        corpus=_load_corpus(base / "corpus.jsonl", ("id", "_id")),
        queries=_load_queries(base / f"{split}_queries.jsonl", ("id", "_id"))[0],
        qrels=_load_jsonl_qrels(
            base / f"{split}_qrels.jsonl",
            ("qid", "query-id", "query_id"),
            ("docid", "corpus-id", "doc_id"),
            ("relevance", "score"),
        ),
    )
    return _require_consistent(task)


def _nearest_parent_file(leaf: Path, root: Path, name: str) -> Path:
    current = leaf
    while True:
        candidate = current / name
        if candidate.is_file():
            return candidate
        if current == root:
            break
        current = current.parent
    raise FileNotFoundError(f"could not find {name} for LMEB leaf {leaf}")


def _load_lmeb_qrels(path: Path) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not row or row[0].lower() in {"qid", "query-id", "query_id"}:
                continue
            if len(row) < 3:
                raise ValueError(f"{path}:{row_number}: expected qid, docid, score")
            query_id, document_id = str(row[0]), str(row[1])
            score = float(row[2])
            if not math.isfinite(score):
                raise ValueError(f"{path}:{row_number}: relevance score must be finite")
            if document_id in result.setdefault(query_id, {}):
                raise ValueError(
                    f"{path}:{row_number}: duplicate relevance for {query_id}/{document_id}"
                )
            result[query_id][document_id] = score
    return result


def _load_lmeb_candidates(path: Path) -> Dict[str, List[str]]:
    result = {}
    for index, record in enumerate(_read_jsonl(path), start=1):
        scene = _identifier(record, ("scene_id", "query_id", "qid", "id"), f"{path}:{index}")
        values = record.get("candidate_doc_ids")
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, (str, int)) for value in values
        ):
            raise ValueError(f"{path}:{index}: candidate_doc_ids must be a list")
        if scene in result:
            raise ValueError(f"{path}:{index}: duplicate candidate scene {scene}")
        document_ids = [str(value) for value in values]
        if not document_ids:
            raise ValueError(f"{path}:{index}: candidate_doc_ids must not be empty")
        if len(document_ids) != len(set(document_ids)):
            raise ValueError(f"{path}:{index}: candidate_doc_ids contains duplicates")
        result[scene] = document_ids
    return result


def _candidate_scene(
    query_id: str,
    record: Mapping[str, object],
    scene_candidates: Mapping[str, Sequence[str]],
    location: str,
) -> str:
    explicit_scene = record.get("scene_id")
    if explicit_scene is not None:
        scene = str(explicit_scene)
        if scene not in scene_candidates:
            raise ValueError(f"{location}: no candidate pool for scene/query {scene}")
        return scene
    if query_id in scene_candidates:
        return query_id
    # Standard LMEB query IDs are often scene_0_q_0 while candidates are
    # keyed by scene_0.  Longest-prefix matching also supports analogous
    # task-specific scene names and rejects ambiguous pools.
    matches = [
        scene
        for scene in scene_candidates
        if query_id.startswith(f"{scene}_")
    ]
    if not matches:
        raise ValueError(f"{location}: no candidate pool for scene/query {query_id}")
    longest = max(len(scene) for scene in matches)
    matches = [scene for scene in matches if len(scene) == longest]
    if len(matches) != 1:
        raise ValueError(f"{location}: ambiguous candidate pools for query {query_id}: {matches}")
    return matches[0]


def _normalize_instruction_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _resolve_lmeb_instruction(name: str, raw: Mapping[str, object]) -> Optional[str]:
    """Resolve the source LMEB task/subset instruction without absolute paths."""
    parts = name.replace("\\", "/").split("/")
    normalized_parts = [_normalize_instruction_name(part) for part in parts]
    task_instructions: Optional[Mapping[str, object]] = None
    for key, value in raw.items():
        if _normalize_instruction_name(str(key)) in normalized_parts and isinstance(value, dict):
            task_instructions = value
            break
    if task_instructions is None:
        for key, value in raw.items():
            normalized_key = _normalize_instruction_name(str(key))
            if isinstance(value, dict) and any(
                normalized_key.endswith(part) or part.endswith(normalized_key)
                for part in normalized_parts
            ):
                task_instructions = value
                break
    if task_instructions is None:
        return None
    for part in reversed(parts):
        normalized_part = _normalize_instruction_name(part)
        for sub_key, instruction in task_instructions.items():
            if _normalize_instruction_name(str(sub_key)) == normalized_part:
                return instruction if isinstance(instruction, str) else None
    if len(task_instructions) == 1:
        instruction = next(iter(task_instructions.values()))
        return instruction if isinstance(instruction, str) else None
    return None


def _load_task_instructions(root: Path) -> Mapping[str, object]:
    path = root / "task_instructions.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_lmeb(root: str | Path, groups: Sequence[str]) -> List[EvaluationTask]:
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"LMEB directory not found: {base}")
    instructions = _load_task_instructions(base)
    tasks = []
    if isinstance(groups, str):
        groups = [groups]
    requested = {str(group).casefold() for group in groups}
    for queries_path in sorted(base.rglob("queries.jsonl")):
        leaf = queries_path.parent
        relative_name = leaf.relative_to(base).as_posix()
        relative_parts = {part.casefold() for part in relative_name.split("/")}
        if requested and relative_name.casefold() not in requested and not (requested & relative_parts):
            continue
        qrels_path = leaf / "qrels.tsv"
        if not qrels_path.is_file():
            continue
        corpus_path = _nearest_parent_file(leaf, base, "corpus.jsonl")
        candidates_path = _nearest_parent_file(leaf, base, "candidates.jsonl")
        queries, metadata = _load_queries(queries_path, ("_id", "id", "query_id", "qid"))
        corpus = _load_corpus(corpus_path, ("_id", "id", "docid", "doc_id"))
        scene_candidates = _load_lmeb_candidates(candidates_path)
        unknown_candidate_documents = sorted(
            {
                document_id
                for document_ids in scene_candidates.values()
                for document_id in document_ids
                if document_id not in corpus
            }
        )
        if unknown_candidate_documents:
            raise ValueError(
                f"{candidates_path}: candidate pool references unknown documents "
                f"{unknown_candidate_documents[:3]}"
            )
        candidates: Dict[str, List[str]] = {}
        for query_id, record in metadata.items():
            scene = _candidate_scene(query_id, record, scene_candidates, str(queries_path))
            candidates[query_id] = scene_candidates[scene]
        qrels = _load_lmeb_qrels(qrels_path)
        unknown_queries = sorted(set(qrels) - set(queries))
        if unknown_queries:
            raise ValueError(
                f"{qrels_path}: qrels reference unknown queries: {unknown_queries[:3]}"
            )
        unknown_documents = sorted(
            {document_id for relevance in qrels.values() for document_id in relevance} - set(corpus)
        )
        if unknown_documents:
            raise ValueError(
                f"{qrels_path}: qrels reference unknown documents: {unknown_documents[:3]}"
            )
        filtered_qrels = {
            query_id: {
                doc_id: score
                for doc_id, score in relevance.items()
                # Keep non-positive judgments for parity with the source
                # loader; positive judgments outside a restricted pool cannot
                # be retrieved and are excluded from this task's denominator.
                if score <= 0 or doc_id in set(candidates[query_id])
            }
            for query_id, relevance in qrels.items()
        }
        kept = {
            query_id
            for query_id, relevance in filtered_qrels.items()
            if any(score > 0 for score in relevance.values())
        }
        name = leaf.relative_to(base).as_posix()
        resolved_instruction = _resolve_lmeb_instruction(name, instructions)
        task_instructions = {"instruction": resolved_instruction} if resolved_instruction else {}
        task = EvaluationTask(
            name=name,
            benchmark="lmeb",
            split="test",
            corpus=corpus,
            queries={query_id: queries[query_id] for query_id in sorted(kept)},
            qrels={query_id: filtered_qrels[query_id] for query_id in sorted(kept)},
            candidates={query_id: candidates[query_id] for query_id in sorted(kept)},
            instructions={str(k): str(v) for k, v in task_instructions.items()},
        )
        tasks.append(_require_consistent(task))
    if not tasks:
        raise FileNotFoundError(f"no matching LMEB leaves found below {base}")
    return tasks
