from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from .config import apply_path_overrides, load_config, require_sections
from .data import (
    LycheeTrainCollator,
    LycheeTrainDataset,
    data_manifest,
    discover_data_files,
    load_hf_dataset,
)
from .model import LycheeRerankerTrainingModel
from .trainer import LycheeTrainer


def _dtype(name: str):
    values = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float32": torch.float32}
    if name not in values:
        raise ValueError(f"unsupported training dtype: {name}")
    return values[name]


def build_components(config: Dict[str, Any]):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
    except ImportError as error:
        raise ImportError("training requires `pip install -e '.[train]'`") from error

    require_sections(config, "model", "data", "training")
    model_cfg, data_cfg, training_cfg = (
        dict(config["model"]),
        dict(config["data"]),
        dict(config["training"]),
    )
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
    )
    model = get_peft_model(model, peft_config)

    train_paths = data_cfg.pop("train_data")
    knowledge_distillation = bool(data_cfg.pop("knowledge_distillation", False))
    files = discover_data_files(train_paths)
    raw_dataset = load_hf_dataset(files)
    dataset = LycheeTrainDataset(
        raw_dataset,
        tokenizer,
        train_group_size=int(data_cfg.pop("train_group_size", 16)),
        query_max_length=int(data_cfg.pop("query_max_length", 128)),
        passage_max_length=int(data_cfg.pop("passage_max_length", 512)),
        knowledge_distillation=knowledge_distillation,
    )
    pad_to_multiple_of = data_cfg.pop("pad_to_multiple_of", 8)
    if data_cfg:
        raise ValueError(f"unknown data configuration keys: {sorted(data_cfg)}")
    wrapper = LycheeRerankerTrainingModel(
        model,
        tokenizer,
        train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        chunk_sizes=chunk_sizes,
        chunk_loss_weights=chunk_weights,
        knowledge_distillation=knowledge_distillation,
    )
    training_cfg.setdefault("remove_unused_columns", False)
    arguments = TrainingArguments(**training_cfg)
    collator = LycheeTrainCollator(tokenizer, pad_to_multiple_of=pad_to_multiple_of)
    return tokenizer, wrapper, dataset, collator, arguments, data_manifest(files)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a KaLM encoder-decoder reranker")
    parser.add_argument("--config", required=True, help="JSON/YAML recipe")
    parser.add_argument("--train-data", nargs="+", help="portable override for data.train_data")
    parser.add_argument("--output-dir", help="portable override for training.output_dir")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = apply_path_overrides(
        load_config(args.config), data=args.train_data, output_dir=args.output_dir
    )
    if args.print_config:
        print(json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True))
        return
    tokenizer, model, dataset, collator, training_args, manifest = build_components(config)
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trainer = LycheeTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    main()
