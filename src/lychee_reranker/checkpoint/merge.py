from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Optional, Sequence


def merge_lora(
    base_model: str,
    adapter: str,
    output_dir: str | Path,
    *,
    revision: Optional[str] = None,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
) -> Path:
    """Merge one explicit LoRA adapter into a seq2seq base model atomically."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as error:
        raise ImportError("LoRA merge requires the 'train' extra") from error
    dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}
    if dtype not in dtypes:
        raise ValueError(f"unsupported dtype: {dtype}")
    adapter_path = Path(adapter)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"adapter directory not found: {adapter_path}")
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    common = {"revision": revision, "trust_remote_code": trust_remote_code}
    common = {key: value for key, value in common.items() if value is not None}
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model, dtype=dtypes[dtype], **common)
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    merged = model.merge_and_unload(safe_merge=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, **common)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as tmp:
        temporary = Path(tmp) / "model"
        temporary.mkdir()
        merged.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        manifest = {
            "schema_version": 1,
            "operation": "merge_lora",
            "base_model": base_model,
            "base_revision": revision,
            "adapter": str(adapter_path.resolve()),
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        (temporary / "merge_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    return destination


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a Lychee-Reranker LoRA adapter")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
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
