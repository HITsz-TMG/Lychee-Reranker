from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from .config import (
    apply_path_overrides,
    load_config,
    require_sections,
    seed_everything,
    validate_training_config,
)
from .data import (
    LycheeTrainCollator,
    LycheeTrainDataset,
    data_manifest,
    discover_data_files,
    load_hf_dataset,
    validate_dataset_records,
)


def _dtype(name: str):
    values = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float32": torch.float32}
    if name not in values:
        raise ValueError(f"unsupported training dtype: {name}")
    return values[name]


def preflight_data(config: Dict[str, Any]):
    """Discover, load, and validate rows before importing model checkpoints."""
    require_sections(config, "data")
    data_cfg = config["data"]
    files = discover_data_files(data_cfg["train_data"])
    raw_dataset = load_hf_dataset(files)
    validate_dataset_records(
        raw_dataset,
        knowledge_distillation=bool(data_cfg.get("knowledge_distillation", False)),
    )
    return files, raw_dataset, data_manifest(files)


def _protected_local_paths(config: Dict[str, Any], files: Sequence[Path]) -> list[Path]:
    protected = list(files)
    model_name = config["model"]["name_or_path"]
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        protected.append(model_path)
    protected.extend(Path(path).expanduser() for path in config["data"]["train_data"])
    return protected


def ensure_safe_output_dir(
    output_dir: str | os.PathLike[str],
    *,
    protected_paths: Sequence[Path] = (),
    resume_from_checkpoint: str | os.PathLike[str] | None = None,
) -> Path:
    """Create an output directory without deleting or nesting in source data."""
    output = Path(output_dir).expanduser().resolve()
    for raw_path in protected_paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file() and output == path:
            raise ValueError(f"training output cannot overwrite input file: {path}")
        if path.is_dir() and (output == path or path in output.parents):
            raise ValueError(f"training output cannot be inside protected path: {path}")

    resume = None
    if resume_from_checkpoint is not None:
        resume = Path(resume_from_checkpoint).expanduser().resolve()
        if not resume.is_dir():
            raise FileNotFoundError(f"resume checkpoint directory not found: {resume}")
        if output != resume and output not in resume.parents:
            raise ValueError("resume checkpoint must be inside training.output_dir")

    if output.exists():
        if not output.is_dir():
            raise ValueError(f"training output exists and is not a directory: {output}")
        if resume is None and any(output.iterdir()):
            raise FileExistsError(
                f"training output is non-empty: {output}; pass --resume-from-checkpoint"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
    return output


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_components(
    config: Dict[str, Any],
    *,
    preflight: tuple[Sequence[Path], Any, Dict[str, Any]] | None = None,
):
    """Build tokenizer/model/dataset after static and data validation."""
    validate_training_config(config)
    seed_everything(int(config["training"]["seed"]))
    if preflight is None:
        files, raw_dataset, manifest = preflight_data(config)
    else:
        files, raw_dataset, manifest = preflight

    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
    except ImportError as error:
        raise ImportError("training requires `pip install -e '.[train]'`") from error

    from .model import LycheeRerankerTrainingModel

    model_cfg = dict(config["model"])
    data_cfg = dict(config["data"])
    training_cfg = dict(config["training"])
    model_name = model_cfg.pop("name_or_path")
    trust_remote_code = bool(model_cfg.pop("trust_remote_code", False))
    revision = model_cfg.pop("revision", None)
    dtype = _dtype(model_cfg.pop("dtype", "bfloat16"))
    attention = model_cfg.pop("attn_implementation", "sdpa")
    lora_cfg = dict(model_cfg.pop("lora"))
    chunk_sizes = model_cfg.pop("chunk_sizes", [2, 8, 32, 128])
    chunk_weights = model_cfg.pop("chunk_loss_weights", [1.0] * len(chunk_sizes))
    if model_cfg:
        raise ValueError(f"unknown model configuration keys: {sorted(model_cfg)}")

    data_cfg.pop("train_data")
    train_group_size = int(data_cfg.pop("train_group_size", 16))
    query_max_length = int(data_cfg.pop("query_max_length", 128))
    passage_max_length = int(data_cfg.pop("passage_max_length", 512))
    pad_to_multiple_of = data_cfg.pop("pad_to_multiple_of", 8)
    knowledge_distillation = bool(data_cfg.pop("knowledge_distillation", False))
    if data_cfg:
        raise ValueError(f"unknown data configuration keys: {sorted(data_cfg)}")

    common = {"trust_remote_code": trust_remote_code, "revision": revision}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **common)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must provide pad_token or eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    base_config = AutoConfig.from_pretrained(model_name, **common)
    base_config.use_cache = False
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        config=base_config,
        dtype=dtype,
        attn_implementation=attention,
        **common,
    )
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=int(lora_cfg["rank"]),
        lora_alpha=float(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.1)),
        target_modules=list(lora_cfg["target_modules"]),
        # T5Gemma2 checkpoints retain a SigLIP tower whose q/k/v/out modules
        # share these suffixes.  Reranker inputs are text-only; excluding it
        # keeps every trainable adapter on the executed text graph.
        exclude_modules=r".*vision_tower.*",
    )
    model = get_peft_model(model, peft_config)

    dataset = LycheeTrainDataset(
        raw_dataset,
        tokenizer,
        train_group_size=train_group_size,
        query_max_length=query_max_length,
        passage_max_length=passage_max_length,
        knowledge_distillation=knowledge_distillation,
    )
    wrapper = LycheeRerankerTrainingModel(
        model,
        tokenizer,
        train_group_size=train_group_size,
        chunk_sizes=chunk_sizes,
        chunk_loss_weights=chunk_weights,
        knowledge_distillation=knowledge_distillation,
    )
    training_cfg.setdefault("remove_unused_columns", False)
    arguments = TrainingArguments(**training_cfg)
    collator = LycheeTrainCollator(tokenizer, pad_to_multiple_of=pad_to_multiple_of)
    return tokenizer, wrapper, dataset, collator, arguments, manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Lychee encoder-decoder reranker")
    parser.add_argument("--config", required=True, help="JSON/YAML recipe")
    parser.add_argument("--train-data", nargs="+", help="portable override for data.train_data")
    parser.add_argument("--output-dir", help="portable override for training.output_dir")
    parser.add_argument("--model-name-or-path", help="portable override for model.name_or_path")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args(argv)


def _distributed_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = apply_path_overrides(
        load_config(args.config),
        data=args.train_data,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
    )
    validate_training_config(config)
    if args.print_config:
        print(json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True))
        return

    seed_everything(int(config["training"]["seed"]))
    files, raw_dataset, manifest = preflight_data(config)
    rank = _distributed_rank()
    output_dir = Path(config["training"]["output_dir"]).expanduser().resolve()
    if rank == 0:
        output_dir = ensure_safe_output_dir(
            output_dir,
            protected_paths=_protected_local_paths(config, files),
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    tokenizer, model, dataset, collator, training_args, _ = build_components(
        config, preflight=(files, raw_dataset, manifest)
    )

    from .trainer import LycheeTrainer

    trainer = LycheeTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    _distributed_barrier()
    if rank == 0:
        _atomic_json(output_dir / "effective_config.json", config)
        _atomic_json(output_dir / "input_manifest.json", manifest)
    _distributed_barrier()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    main()
