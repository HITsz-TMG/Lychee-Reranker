from __future__ import annotations

import csv
import json
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
    return f"{title}\n{text}".strip()


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
        result.setdefault(query_id, {})[document_id] = float(value)
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
    return task


def load_beir(root: str | Path, dataset: str, split: str = "test") -> List[EvaluationTask]:
    base = Path(root) / dataset
    if not base.is_dir():
        raise FileNotFoundError(f"BEIR dataset directory not found: {base}")
    leaves = [base]
    if not (base / "corpus.jsonl").is_file():
        leaves = sorted(path.parent for path in base.rglob("corpus.jsonl"))
    tasks = []
    for leaf in leaves:
        suffix = "" if leaf == base else f"/{leaf.relative_to(base).as_posix()}"
        task = EvaluationTask(
            name=f"{dataset}{suffix}",
            benchmark="beir",
            split=split,
            corpus=_load_corpus(leaf / "corpus.jsonl", ("_id", "id")),
            queries=_load_queries(leaf / f"{split}_queries.jsonl", ("_id", "id"))[0],
            qrels=_load_jsonl_qrels(
                leaf / f"{split}_qrels.jsonl",
                ("query-id", "qid", "query_id"),
                ("corpus-id", "docid", "doc_id"),
                ("score", "relevance"),
            ),
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
            result.setdefault(str(row[0]), {})[str(row[1])] = float(row[2])
    return result


def _load_lmeb_candidates(path: Path) -> Dict[str, List[str]]:
    result = {}
    for index, record in enumerate(_read_jsonl(path), start=1):
        scene = _identifier(record, ("scene_id", "query_id", "qid", "id"), f"{path}:{index}")
        values = record.get("candidate_doc_ids")
        if not isinstance(values, list) or any(
            not isinstance(value, (str, int)) for value in values
        ):
            raise ValueError(f"{path}:{index}: candidate_doc_ids must be a list")
        result[scene] = [str(value) for value in values]
    return result


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
    requested = {group.casefold() for group in groups}
    for queries_path in sorted(base.rglob("queries.jsonl")):
        leaf = queries_path.parent
        relative_parts = {part.casefold() for part in leaf.relative_to(base).parts}
        if requested and not (requested & relative_parts):
            continue
        qrels_path = leaf / "qrels.tsv"
        if not qrels_path.is_file():
            continue
        corpus_path = _nearest_parent_file(leaf, base, "corpus.jsonl")
        candidates_path = _nearest_parent_file(leaf, base, "candidates.jsonl")
        queries, metadata = _load_queries(queries_path, ("_id", "id", "query_id", "qid"))
        scene_candidates = _load_lmeb_candidates(candidates_path)
        candidates: Dict[str, List[str]] = {}
        for query_id, record in metadata.items():
            scene = str(record.get("scene_id", query_id))
            if scene not in scene_candidates:
                raise ValueError(f"{queries_path}: no candidate pool for scene/query {scene}")
            candidates[query_id] = scene_candidates[scene]
        qrels = _load_lmeb_qrels(qrels_path)
        filtered_qrels = {
            query_id: {
                doc_id: score
                for doc_id, score in relevance.items()
                if doc_id in set(candidates.get(query_id, []))
            }
            for query_id, relevance in qrels.items()
        }
        kept = {
            query_id
            for query_id, relevance in filtered_qrels.items()
            if any(score > 0 for score in relevance.values())
        }
        name = leaf.relative_to(base).as_posix()
        instruction_value = instructions.get(name, instructions.get(leaf.name, {}))
        task_instructions = instruction_value if isinstance(instruction_value, dict) else {}
        task = EvaluationTask(
            name=name,
            benchmark="lmeb",
            split="test",
            corpus=_load_corpus(corpus_path, ("_id", "id", "docid", "doc_id")),
            queries={query_id: queries[query_id] for query_id in sorted(kept)},
            qrels={query_id: filtered_qrels[query_id] for query_id in sorted(kept)},
            candidates={query_id: candidates[query_id] for query_id in sorted(kept)},
            instructions={str(k): str(v) for k, v in task_instructions.items()},
        )
        tasks.append(_require_consistent(task))
    if not tasks:
        raise FileNotFoundError(f"no matching LMEB leaves found below {base}")
    return tasks
