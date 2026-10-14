from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.model",
    "added_tokens.json",
    "generation_config.json",
)


def _semantic_config(path: Path) -> Mapping[str, object]:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found in {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for volatile in ("_name_or_path", "transformers_version"):
        config.pop(volatile, None)
    return config


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokenizer_fingerprint(path: Path) -> Mapping[str, str]:
    return {name: _file_digest(path / name) for name in TOKENIZER_FILES if (path / name).is_file()}


def _weight_map(model_dir: Path) -> Dict[str, Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        mapping = index.get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"invalid safetensors index: {index_path}")
        return {key: model_dir / filename for key, filename in mapping.items()}
    single = model_dir / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"model.safetensors or its index not found in {model_dir}")
    from safetensors import safe_open

    with safe_open(single, framework="pt", device="cpu") as handle:
        return {key: single for key in handle.keys()}


def _tensor_metadata(
    weight_map: Mapping[str, Path],
) -> Dict[str, Tuple[object, Tuple[int, ...], int]]:
    from safetensors import safe_open

    result = {}
    readers = {}
    try:
        for key, path in weight_map.items():
            reader = readers.get(path)
            if reader is None:
                reader = safe_open(path, framework="pt", device="cpu")
                readers[path] = reader
            tensor = reader.get_slice(key)
            shape = tuple(tensor.get_shape())
            dtype = tensor.get_dtype()
            # Safetensors dtype strings use standard widths except BOOL.
            bits = {
                "BOOL": 8,
                "U8": 8,
                "I8": 8,
                "I16": 16,
                "I32": 32,
                "I64": 64,
                "F16": 16,
                "BF16": 16,
                "F32": 32,
                "F64": 64,
            }.get(str(dtype), 32)
            count = 1
            for dimension in shape:
                count *= dimension
            result[key] = (dtype, shape, count * bits // 8)
    finally:
        readers.clear()
    return result


def _shard_plan(metadata: Mapping[str, Tuple[object, Tuple[int, ...], int]], max_bytes: int):
    shards: List[List[str]] = []
    current: List[str] = []
    current_size = 0
    for key in sorted(metadata):
        size = metadata[key][2]
        if current and current_size + size > max_bytes:
            shards.append(current)
            current, current_size = [], 0
        current.append(key)
        current_size += size
    if current:
        shards.append(current)
    return shards


def soup_models(
    model_paths: Sequence[str | Path],
    weights: Sequence[float],
    output_dir: str | Path,
    *,
    max_shard_size_gb: float = 5.0,
) -> Path:
    """Create a strict tensor-wise model soup from merged safetensors models."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if len(model_paths) < 2 or len(model_paths) != len(weights):
        raise ValueError("provide at least two models and one weight per model")
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    normalized = [float(weight) / sum(weights) for weight in weights]
    models = [Path(path) for path in model_paths]
    if any(not path.is_dir() for path in models):
        missing = [str(path) for path in models if not path.is_dir()]
        raise FileNotFoundError(f"model directories not found: {missing}")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    reference_config = _semantic_config(models[0])
    reference_tokenizer = _tokenizer_fingerprint(models[0])
    if not reference_tokenizer:
        raise FileNotFoundError(f"no tokenizer files found in {models[0]}")
    maps = [_weight_map(path) for path in models]
    keys = set(maps[0])
    for path, mapping in zip(models[1:], maps[1:]):
        if _semantic_config(path) != reference_config:
            raise ValueError(f"model config differs: {path}")
        if _tokenizer_fingerprint(path) != reference_tokenizer:
            raise ValueError(f"tokenizer files differ: {path}")
        if set(mapping) != keys:
            raise ValueError(f"tensor keys differ: {path}")

    metadata = [_tensor_metadata(mapping) for mapping in maps]
    for key in keys:
        signatures = [(item[key][0], item[key][1]) for item in metadata]
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError(f"tensor dtype/shape differs for key {key}")
    max_bytes = int(max_shard_size_gb * 1024**3)
    if max_bytes <= 0:
        raise ValueError("max_shard_size_gb must be positive")
    shard_keys = _shard_plan(metadata[0], max_bytes)

    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as tmp:
        temporary = Path(tmp) / "model"
        temporary.mkdir()
        weight_index: Dict[str, str] = {}
        total_size = sum(value[2] for value in metadata[0].values())
        for shard_number, names in enumerate(shard_keys, start=1):
            filename = (
                "model.safetensors"
                if len(shard_keys) == 1
                else f"model-{shard_number:05d}-of-{len(shard_keys):05d}.safetensors"
            )
            output_tensors = OrderedDict()
            for key in names:
                tensors = []
                for mapping in maps:
                    with safe_open(mapping[key], framework="pt", device="cpu") as handle:
                        tensors.append(handle.get_tensor(key))
                reference = tensors[0]
                if reference.is_floating_point():
                    merged = sum(
                        weight * tensor.float() for weight, tensor in zip(normalized, tensors)
                    )
                    output_tensors[key] = merged.to(reference.dtype).contiguous()
                else:
                    if any(not torch.equal(reference, tensor) for tensor in tensors[1:]):
                        raise ValueError(f"non-floating tensor differs for key {key}")
                    output_tensors[key] = reference.contiguous()
                weight_index[key] = filename
            save_file(output_tensors, temporary / filename, metadata={"format": "pt"})
            del output_tensors
        if len(shard_keys) > 1:
            index = {"metadata": {"total_size": total_size}, "weight_map": weight_index}
            (temporary / "model.safetensors.index.json").write_text(
                json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        for name in ("config.json",) + TOKENIZER_FILES:
            source = models[0] / name
            if source.is_file():
                shutil.copy2(source, temporary / name)
        manifest = {
            "schema_version": 1,
            "operation": "model_soup",
            "models": [str(path.resolve()) for path in models],
            "weights": normalized,
            "tensor_count": len(keys),
            "total_size": total_size,
            "config_sha256": hashlib.sha256(
                json.dumps(reference_config, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        (temporary / "soup_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    return destination


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average merged Lychee-Reranker checkpoints")
    parser.add_argument("--model", action="append", required=True, help="repeat for every model")
    parser.add_argument("--weight", action="append", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-shard-size-gb", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    soup_models(
        args.model,
        args.weight,
        args.output_dir,
        max_shard_size_gb=args.max_shard_size_gb,
    )


if __name__ == "__main__":
    main()
