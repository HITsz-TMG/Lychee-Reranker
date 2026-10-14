from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from lychee_reranker.prompts import DEFAULT_SYSTEM_INSTRUCTION, build_decoder_prompt, build_document

SUPPORTED_SUFFIXES = {".json", ".jsonl", ".parquet"}
REQUIRED_FIELDS = ("query", "pos", "neg", "prompt")
SCORE_FIELDS = ("pos_scores", "neg_scores")


def discover_data_files(paths: Sequence[str | os.PathLike[str]]) -> List[Path]:
    """Discover supported shards deterministically while preserving repeated roots."""
    found: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_file():
            if path.suffix.lower() not in SUPPORTED_SUFFIXES or ".complete." in path.name:
                raise ValueError(f"unsupported training file: {path}")
            found.append(path)
        elif path.is_dir():
            found.extend(
                sorted(
                    candidate
                    for candidate in path.rglob("*")
                    if candidate.is_file()
                    and candidate.suffix.lower() in SUPPORTED_SUFFIXES
                    and ".complete." not in candidate.name
                )
            )
        else:
            raise FileNotFoundError(f"training data path not found: {path}")
    if not found:
        raise ValueError("no JSON, JSONL, or Parquet training files were found")
    return found


def data_manifest(files: Sequence[Path]) -> Dict[str, Any]:
    entries = []
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        entry = {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        entries.append(entry)
        digest.update(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode())
    return {"version": 1, "digest": digest.hexdigest(), "files": entries}


def _string_list(value: Any, field: str, location: str) -> List[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{location}: {field} must be a non-empty list of strings")
    result = list(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{location}: {field} must contain non-empty strings")
    return result


def _score_list(value: Any, field: str, expected: int, location: str) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) != expected:
        raise ValueError(f"{location}: {field} must contain exactly {expected} scores")
    tensor = torch.as_tensor(list(value), dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{location}: {field} contains a non-finite value")
    return [float(item) for item in value]


def validate_record(
    record: Mapping[str, Any], *, knowledge_distillation: bool, location: str = "record"
) -> Dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError(f"{location}: expected a mapping")
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"{location}: missing fields: {missing}")
    if not isinstance(record["query"], str) or not isinstance(record["prompt"], str):
        raise ValueError(f"{location}: query and prompt must be strings")
    pos = _string_list(record["pos"], "pos", location)
    neg = _string_list(record["neg"], "neg", location)
    result: Dict[str, Any] = {
        "query": record["query"],
        "pos": pos,
        "neg": neg,
        "prompt": record["prompt"],
    }
    if knowledge_distillation:
        missing_scores = [field for field in SCORE_FIELDS if field not in record]
        if missing_scores:
            raise ValueError(f"{location}: missing distillation fields: {missing_scores}")
        result["pos_scores"] = _score_list(record["pos_scores"], "pos_scores", len(pos), location)
        result["neg_scores"] = _score_list(record["neg_scores"], "neg_scores", len(neg), location)
    return result


def validate_dataset_records(
    records: Iterable[Mapping[str, Any]], *, knowledge_distillation: bool
) -> int:
    """Validate every loaded row before tokenizer/model construction.

    The training launcher calls this immediately after loading the shards so a
    malformed row or missing teacher field cannot fail after a model download
    or after a distributed job has started.
    """
    count = 0
    for count, record in enumerate(records, start=1):
        validate_record(
            record,
            knowledge_distillation=knowledge_distillation,
            location=f"record[{count - 1}]",
        )
    if count == 0:
        raise ValueError("training data contains no records")
    return count


def load_hf_dataset(files: Sequence[Path]):
    try:
        import datasets
    except ImportError as error:
        raise ImportError("training data loading requires the 'train' extra") from error
    groups: Dict[str, List[str]] = {"json": [], "parquet": []}
    for path in files:
        groups["parquet" if path.suffix.lower() == ".parquet" else "json"].append(str(path))
    loaded = []
    for kind in ("json", "parquet"):
        if groups[kind]:
            loaded.append(datasets.load_dataset(kind, data_files=groups[kind], split="train"))
    if len(loaded) == 1:
        return loaded[0]
    return datasets.concatenate_datasets(loaded)


class LycheeTrainDataset(Dataset):
    """Map-style training dataset that preserves source multiplicity."""

    def __init__(
        self,
        records,
        tokenizer,
        *,
        train_group_size: int = 16,
        query_max_length: int = 128,
        passage_max_length: int = 512,
        knowledge_distillation: bool = False,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        seed: int = 42,
    ) -> None:
        if train_group_size < 2:
            raise ValueError("train_group_size must be at least 2")
        self.records = records
        self.tokenizer = tokenizer
        self.train_group_size = train_group_size
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        self.knowledge_distillation = knowledge_distillation
        self.system_instruction = system_instruction
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = validate_record(
            self.records[index],
            knowledge_distillation=self.knowledge_distillation,
            location=f"record[{index}]",
        )
        query_ids = self.tokenizer(
            record["query"],
            add_special_tokens=False,
            truncation=True,
            max_length=self.query_max_length,
        )["input_ids"]
        query = self.tokenizer.decode(
            query_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        decoder_text = build_decoder_prompt(query, record["prompt"], self.system_instruction)
        decoder = self.tokenizer(
            decoder_text, truncation=True, padding=False, add_special_tokens=False
        )

        # Trainer may prefetch a resumed batch before restoring global RNG
        # state. Independent sampling makes the same record/epoch identical
        # regardless of worker scheduling, prefetching or checkpoint resume.
        rng = random.Random(f"{self.seed}:{self.epoch}:{index}")
        pos_index = rng.randrange(len(record["pos"]))
        negative_pool = list(range(len(record["neg"])))
        needed = self.train_group_size - 1
        if len(negative_pool) < needed:
            repetitions = math.ceil(needed / len(negative_pool))
            negative_indices = rng.sample(negative_pool * repetitions, needed)
        else:
            negative_indices = rng.sample(negative_pool, needed)
        documents = [record["pos"][pos_index]] + [record["neg"][i] for i in negative_indices]
        teacher_scores: Optional[List[float]] = None
        if self.knowledge_distillation:
            teacher_scores = [record["pos_scores"][pos_index]] + [
                record["neg_scores"][i] for i in negative_indices
            ]

        pairs = []
        for document in documents:
            encoded = self.tokenizer(
                build_document(document),
                max_length=self.passage_max_length,
                truncation=True,
                padding=False,
                add_special_tokens=False,
            )
            pairs.append(
                {
                    "encoder_input_ids": encoded["input_ids"],
                    "encoder_attention_mask": encoded["attention_mask"],
                    "decoder_input_ids": decoder["input_ids"],
                    "decoder_attention_mask": decoder["attention_mask"],
                }
            )
        return pairs, teacher_scores


@dataclass
class LycheeTrainCollator:
    tokenizer: Any
    passage_max_length: int = 512
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: Sequence[Tuple[List[Dict[str, Any]], Optional[List[float]]]]):
        if not features:
            raise ValueError("cannot collate an empty batch")
        teacher_groups = [feature[1] for feature in features]
        has_teacher = [group is not None for group in teacher_groups]
        if any(value != has_teacher[0] for value in has_teacher[1:]):
            raise ValueError("a batch cannot mix SFT rows and distillation rows")
        group_sizes = [len(group) for group, _ in features]
        if any(size < 2 for size in group_sizes):
            raise ValueError("each query must contain one positive and at least one negative")
        if len(set(group_sizes)) != 1:
            raise ValueError("all query groups in a batch must have the same size")
        teacher_scores = None
        if has_teacher[0]:
            if any(len(group) != group_sizes[index] for index, group in enumerate(teacher_groups)):
                raise ValueError("teacher score groups must match their query group size")
            teacher_scores = torch.tensor(
                [score for group in teacher_groups for score in group], dtype=torch.float32
            )
        pairs = [pair for group, _ in features for pair in group]
        encoder = self.tokenizer.pad(
            [
                {
                    "input_ids": pair["encoder_input_ids"],
                    "attention_mask": pair["encoder_attention_mask"],
                }
                for pair in pairs
            ],
            padding=True,
            max_length=self.passage_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        decoder = self.tokenizer.pad(
            [
                {
                    "input_ids": pair["decoder_input_ids"],
                    "attention_mask": pair["decoder_attention_mask"],
                }
                for pair in pairs
            ],
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        return {
            "encoder_input_ids": encoder["input_ids"],
            "encoder_attention_mask": encoder["attention_mask"],
            "decoder_input_ids": decoder["input_ids"],
            "decoder_attention_mask": decoder["attention_mask"],
            "teacher_scores": teacher_scores,
            # Trainer forwards this scalar to the wrapper; the wrapper checks
            # it against the flattened pair count for variable final batches.
            "num_query_groups": torch.tensor(len(features), dtype=torch.long),
        }
