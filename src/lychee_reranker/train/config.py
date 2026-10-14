from __future__ import annotations

import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Mapping


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a JSON or YAML configuration without silently accepting an empty file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    if config_path.suffix.lower() == ".json":
        value = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml
        except ImportError as error:
            raise ImportError("YAML configs require the 'train' extra (PyYAML)") from error
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("the top-level configuration must be a mapping")
    return value


def require_sections(config: Mapping[str, Any], *sections: str) -> None:
    missing = [name for name in sections if not isinstance(config.get(name), dict)]
    if missing:
        raise ValueError(f"missing configuration section(s): {', '.join(missing)}")


def apply_path_overrides(
    config: Dict[str, Any],
    *,
    data: list[str] | None,
    output_dir: str | None,
    model_name_or_path: str | None = None,
) -> Dict[str, Any]:
    """Apply portable CLI overrides without mutating the loaded object."""
    result = copy.deepcopy(config)
    if data is not None:
        result.setdefault("data", {})["train_data"] = list(data)
    if output_dir is not None:
        result.setdefault("training", {})["output_dir"] = output_dir
    if model_name_or_path is not None:
        result.setdefault("model", {})["name_or_path"] = model_name_or_path
    return result


def _positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite_number(value: Any, name: str, *, minimum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and float(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def validate_training_config(
    config: Mapping[str, Any], *, check_data_paths: bool = False
) -> None:
    """Validate the supported training contract before model loading.

    The release recipes deliberately expose only native Trainer/DDP. FSDP and
    DeepSpeed would require different checkpoint gathering and resume logic, so
    accepting them here would silently produce incomplete adapter checkpoints.
    """
    require_sections(config, "model", "data", "training")
    model = config["model"]
    data = config["data"]
    training = config["training"]

    model_name = model.get("name_or_path")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model.name_or_path must be a non-empty model ID or path")
    chunk_sizes = model.get("chunk_sizes", [2, 8, 32, 128])
    if list(chunk_sizes) != [2, 8, 32, 128]:
        raise ValueError("model.chunk_sizes must be exactly [2, 8, 32, 128]")
    chunk_weights = model.get("chunk_loss_weights", [1.0] * 4)
    if list(chunk_weights) != [1.0, 1.0, 1.0, 1.0]:
        raise ValueError("model.chunk_loss_weights must be four unit weights")
    lora = model.get("lora")
    if not isinstance(lora, Mapping):
        raise ValueError("model.lora must be a mapping")
    rank = lora.get("rank")
    alpha = lora.get("alpha")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 2:
        raise ValueError("model.lora.rank must be an integer of at least 2")
    _finite_number(alpha, "model.lora.alpha", minimum=0.0)
    if float(alpha) <= 0:
        raise ValueError("model.lora.alpha must be positive")
    targets = lora.get("target_modules")
    attention_targets = {"q_proj", "k_proj", "v_proj", "out_proj"}
    feed_forward_targets = {"gate_proj", "up_proj", "down_proj"}
    if not isinstance(targets, (list, tuple)):
        raise ValueError("model.lora.target_modules must be a list")
    if any(not isinstance(target, str) for target in targets):
        raise ValueError("model.lora.target_modules must contain strings")
    target_set = set(targets)
    if target_set not in (attention_targets, attention_targets | feed_forward_targets):
        raise ValueError(
            "model.lora.target_modules must be attention-only or attention plus MLP projections"
        )
    _finite_number(lora.get("dropout", 0.1), "model.lora.dropout", minimum=0.0)
    if float(lora.get("dropout", 0.1)) >= 1.0:
        raise ValueError("model.lora.dropout must be less than 1")

    train_data = data.get("train_data")
    if not isinstance(train_data, (list, tuple)) or not train_data or any(
        not isinstance(path, str) or not path.strip() for path in train_data
    ):
        raise ValueError("data.train_data must be a non-empty list of paths")
    if check_data_paths:
        from .data import discover_data_files

        discover_data_files(train_data)
    train_group_size = data.get("train_group_size", 16)
    if (
        isinstance(train_group_size, bool)
        or not isinstance(train_group_size, int)
        or train_group_size < 2
    ):
        raise ValueError("data.train_group_size must be an integer of at least 2")
    if data.get("query_max_length", 128) != 128:
        raise ValueError("data.query_max_length must be 128")
    if data.get("passage_max_length", 512) != 512:
        raise ValueError("data.passage_max_length must be 512")
    _positive_int(data.get("pad_to_multiple_of", 8), "data.pad_to_multiple_of")
    if not isinstance(data.get("knowledge_distillation", False), bool):
        raise ValueError("data.knowledge_distillation must be boolean")

    output_dir = training.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("training.output_dir must be a non-empty path")
    for key in ("deepspeed", "fsdp", "fsdp_config"):
        value = training.get(key)
        if value not in (None, False, "", [], {}):
            raise ValueError(
                f"training.{key} is unsupported; use native torchrun/DDP with adapter checkpoints"
            )
    _positive_int(
        training.get("per_device_train_batch_size", 0),
        "training.per_device_train_batch_size",
    )
    _positive_int(
        training.get("gradient_accumulation_steps", 0),
        "training.gradient_accumulation_steps",
    )
    _finite_number(training.get("learning_rate", 0), "training.learning_rate", minimum=0.0)
    if float(training.get("learning_rate", 0)) <= 0:
        raise ValueError("training.learning_rate must be positive")
    _finite_number(training.get("weight_decay", 0), "training.weight_decay", minimum=0.0)
    _finite_number(training.get("warmup_ratio", 0), "training.warmup_ratio", minimum=0.0)
    if float(training.get("warmup_ratio", 0)) > 1:
        raise ValueError("training.warmup_ratio must be at most 1")
    if training.get("lr_scheduler_type") != "cosine":
        raise ValueError("training.lr_scheduler_type must be cosine")
    if training.get("optim") != "adamw_torch":
        raise ValueError("training.optim must be adamw_torch")
    if training.get("adam_beta1") != 0.9 or training.get("adam_beta2") != 0.999:
        raise ValueError("training Adam betas must be 0.9 and 0.999")
    if training.get("max_grad_norm") != 1.0:
        raise ValueError("training.max_grad_norm must be 1.0")
    if not isinstance(training.get("gradient_checkpointing", False), bool):
        raise ValueError("training.gradient_checkpointing must be boolean")
    seed = training.get("seed")
    _positive_int(seed, "training.seed")
    data_seed = training.get("data_seed", seed)
    _positive_int(data_seed, "training.data_seed")
    epochs = training.get("num_train_epochs")
    _finite_number(epochs, "training.num_train_epochs", minimum=0.0)
    if float(epochs) <= 0:
        raise ValueError("training.num_train_epochs must be positive")


def seed_everything(seed: int) -> None:
    """Seed all available RNGs before model or dataset objects are created."""
    _positive_int(seed, "seed")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
