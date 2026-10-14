from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._common import (
    check_finite_state,
    copy_relevant_metadata,
    CONFIG_VOLATILE_FIELDS,
    read_mapping,
    semantic_config,
    staged_output,
)


def _reference_kind(value: str | Path, *, relative_to: Path | None = None) -> tuple[str, str]:
    """Classify a model reference without guessing a hub ID is a filesystem path."""

    text = str(value)
    candidate = Path(text)
    if candidate.is_absolute():
        return "path", str(candidate.resolve())
    if relative_to is not None:
        relative_candidate = relative_to / candidate
        if relative_candidate.exists():
            return "path", str(relative_candidate.resolve())
    if candidate.exists():
        return "path", str(candidate.resolve())
    return "name", text.rstrip("/")


def _check_adapter_identity(adapter_path: Path, base_model: str | Path) -> dict[str, Any]:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")
    config = dict(read_mapping(config_path))
    peft_type = config.get("peft_type")
    if peft_type != "LORA":
        raise ValueError(f"adapter is not a LoRA adapter (peft_type={peft_type!r})")
    task_type = config.get("task_type")
    if task_type not in (None, "SEQ_2_SEQ_LM"):
        raise ValueError(f"adapter task type is incompatible with seq2seq merge: {task_type!r}")

    declared = config.get("base_model_name_or_path")
    if declared not in (None, ""):
        if not isinstance(declared, str):
            raise ValueError("adapter base_model_name_or_path must be a string when present")
        requested_kind, requested = _reference_kind(base_model)
        declared_kind, declared_ref = _reference_kind(
            declared, relative_to=adapter_path.parent.resolve()
        )
        # Compare references only when both sides identify the same kind of
        # object.  A hub ID versus a local path cannot be resolved reliably,
        # so loading and PEFT's target-module checks remain the authority.
        if requested_kind == declared_kind and requested != declared_ref:
            raise ValueError(
                "adapter was created for a different base model: "
                f"adapter={declared!r}, requested={str(base_model)!r}"
            )
    return config


def _load_seq2seq_config(base_model: str | Path, common: dict[str, Any]):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(base_model, **common)
    if getattr(config, "is_encoder_decoder", None) is not True:
        raise ValueError(
            "base model is not an encoder-decoder model; refusing incompatible checkpoint type"
        )
    return config


_RUNTIME_CONFIG_FIELDS = frozenset((*CONFIG_VOLATILE_FIELDS, "dtype", "torch_dtype"))


def _config_without_runtime_fields(value: Any) -> Any:
    """Remove only save/load metadata that is allowed to vary recursively."""

    if isinstance(value, dict):
        return {
            key: _config_without_runtime_fields(item)
            for key, item in value.items()
            if key not in _RUNTIME_CONFIG_FIELDS
        }
    if isinstance(value, list):
        return [_config_without_runtime_fields(item) for item in value]
    return value


def merge_lora(
    base_model: str,
    adapter: str,
    output_dir: str | Path,
    *,
    revision: str | None = None,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
) -> Path:
    """Merge one explicit LoRA adapter into a seq2seq base model.

    The destination is staged beside the requested path and atomically
    published only after model, adapter, tokenizer, and finite-weight checks
    pass. Existing destinations are never replaced.
    """

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as error:
        raise ImportError("LoRA merge requires the 'train' extra") from error

    dtypes = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float16": torch.float16,
    }
    if dtype not in dtypes:
        raise ValueError(f"unsupported dtype: {dtype}")

    adapter_path = Path(adapter)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"adapter directory not found: {adapter_path}")
    adapter_path = adapter_path.resolve()
    base_path = Path(base_model)
    local_base = base_path if base_path.is_dir() else None
    adapter_config = _check_adapter_identity(adapter_path, base_model)
    common = {"revision": revision, "trust_remote_code": trust_remote_code}
    common = {key: value for key, value in common.items() if value is not None}

    # Reserve the output name before loading potentially large checkpoints so
    # a concurrent invocation cannot publish over this invocation's result.
    with staged_output(output_dir) as temporary:
        config = _load_seq2seq_config(base_model, common)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            dtype=dtypes[dtype],
            **common,
        )
        if getattr(model.config, "is_encoder_decoder", None) is not True:
            raise ValueError("loaded base model is not an encoder-decoder model")
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        check_finite_state(model, "base+adapter")
        merged = model.merge_and_unload(safe_merge=True)
        check_finite_state(merged, "merged")

        tokenizer = AutoTokenizer.from_pretrained(base_model, **common)
        merged.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)

        # For local bases, preserve every relevant metadata file byte-for-byte
        # after the model save. This covers generation config, processor and
        # custom tokenizer/model code that save_pretrained may not emit.
        if local_base is not None:
            copy_relevant_metadata(local_base, temporary, exclude_names={"config.json"})

        # Ensure the saved semantic config still describes the loaded base.
        # This catches an unexpected model implementation rewriting a config
        # during save before the destination can become visible.
        semantic_config(temporary)
        if local_base is not None:
            # Transformers may materialize omitted defaults while loading a
            # nested config (T5Gemma2 does this for dropout, token IDs, and
            # tied embeddings). Compare canonical config objects so those
            # representation changes do not look semantic, while retaining
            # strict equality for every non-runtime field.
            saved_loaded_config = _load_seq2seq_config(temporary, common)
            comparable_saved = _config_without_runtime_fields(saved_loaded_config.to_dict())
            comparable_base = _config_without_runtime_fields(config.to_dict())
            if comparable_saved != comparable_base:
                raise ValueError("merged output config differs from the loaded base model config")

        manifest = {
            "schema_version": 1,
            "operation": "merge_lora",
            "base_model": str(base_model),
            "base_revision": revision,
            "adapter": str(adapter_path),
            "adapter_base_model_name_or_path": adapter_config.get("base_model_name_or_path"),
            "model_type": getattr(config, "model_type", None),
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        (temporary / "merge_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return Path(output_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a Lychee-Reranker LoRA adapter")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    merge_lora(
        args.base_model,
        args.adapter,
        args.output_dir,
        revision=args.revision,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
