from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path

from ._common import (
    TOKENIZER_EXACT_FILES,
    check_finite_tensor,
    copy_relevant_metadata,
    file_digest,
    relevant_metadata_files,
    semantic_config,
    staged_output,
    tensor_metadata,
    tokenizer_fingerprint,
    weight_map,
)

# Retain this exported name for callers that used the first release helper.
TOKENIZER_FILES = tuple(sorted(TOKENIZER_EXACT_FILES))


# Keep private compatibility aliases for the original focused package tests.
_semantic_config = semantic_config
_file_digest = file_digest
_tokenizer_fingerprint = tokenizer_fingerprint
_weight_map = weight_map
_tensor_metadata = tensor_metadata


def _shard_plan(
    metadata: Mapping[str, tuple[str, tuple[int, ...], int]], max_bytes: int
) -> list[list[str]]:
    shards: list[list[str]] = []
    current: list[str] = []
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


def _normalise_weights(weights: Sequence[float]) -> list[float]:
    values = []
    for weight in weights:
        if isinstance(weight, bool):
            raise TypeError("weights must be finite non-negative numbers")
        try:
            value = float(weight)
        except (TypeError, ValueError) as error:
            raise ValueError("weights must be finite non-negative numbers") from error
        if not math.isfinite(value) or value < 0:
            raise ValueError("weights must be finite non-negative numbers")
        values.append(value)
    total = math.fsum(values)
    if total <= 0:
        raise ValueError("weights must have a positive finite sum")
    return [value / total for value in values]


def _validate_inputs(
    model_paths: Sequence[str | Path], weights: Sequence[float]
) -> tuple[list[Path], list[float], list[dict[str, object]], list[dict[str, str]]]:
    if isinstance(model_paths, (str, Path)):
        raise TypeError("model_paths must contain at least two model directories")
    if isinstance(weights, (str, bytes)):
        raise TypeError("weights must contain one numeric value per model")
    if len(model_paths) < 2 or len(model_paths) != len(weights):
        raise ValueError("provide at least two models and one weight per model")
    normalized = _normalise_weights(weights)
    models = [Path(path) for path in model_paths]
    missing = [str(path) for path in models if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"model directories not found: {missing}")
    resolved = [path.resolve() for path in models]
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate model paths are not valid soup inputs")

    configs = []
    tokenizers = []
    maps = []
    for path in models:
        configs.append(semantic_config(path))
        tokenizers.append(tokenizer_fingerprint(path))
        maps.append(weight_map(path))
    reference_keys = set(maps[0])
    for index, (path, config, tokenizer, mapping) in enumerate(
        zip(models[1:], configs[1:], tokenizers[1:], maps[1:]), start=1
    ):
        if config != configs[0]:
            raise ValueError(f"model config differs: {path}")
        if tokenizer != tokenizers[0]:
            raise ValueError(f"tokenizer files differ: {path}")
        if set(mapping) != reference_keys:
            raise ValueError(f"tensor keys differ: {path}")
    metadata = [tensor_metadata(mapping) for mapping in maps]
    for key in reference_keys:
        signatures = [(item[key][0], item[key][1]) for item in metadata]
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError(f"tensor dtype/shape differs for key {key}")
    return models, normalized, metadata, maps


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

    try:
        shard_size = float(max_shard_size_gb)
    except (TypeError, ValueError) as error:
        raise ValueError("max_shard_size_gb must be a positive finite number") from error
    if not math.isfinite(shard_size) or shard_size <= 0:
        raise ValueError("max_shard_size_gb must be a positive finite number")

    models, normalized, metadata, maps = _validate_inputs(model_paths, weights)
    keys = set(maps[0])
    max_bytes = int(shard_size * 1024**3)
    if max_bytes <= 0:
        raise ValueError("max_shard_size_gb is too small to represent one byte")
    shard_keys = _shard_plan(metadata[0], max_bytes)
    reference_config = semantic_config(models[0])
    total_size = sum(value[2] for value in metadata[0].values())

    with staged_output(output_dir) as temporary:
        weight_index: dict[str, str] = {}
        for shard_number, names in enumerate(shard_keys, start=1):
            filename = (
                "model.safetensors"
                if len(shard_keys) == 1
                else f"model-{shard_number:05d}-of-{len(shard_keys):05d}.safetensors"
            )
            output_tensors = OrderedDict()
            with ExitStack() as stack:
                handles = {}
                for key in names:
                    for mapping in maps:
                        source = mapping[key]
                        if source not in handles:
                            handles[source] = stack.enter_context(
                                safe_open(source, framework="pt", device="cpu")
                            )
                for key in names:
                    tensors = [handles[mapping[key]].get_tensor(key) for mapping in maps]
                    reference = tensors[0]
                    for model_number, tensor in enumerate(tensors):
                        check_finite_tensor(tensor, f"model[{model_number}]:{key}")
                    if reference.is_floating_point():
                        accumulator_dtype = (
                            torch.float64 if reference.dtype == torch.float64 else torch.float32
                        )
                        merged = torch.zeros(
                            reference.shape, dtype=accumulator_dtype, device="cpu"
                        )
                        for weight, tensor in zip(normalized, tensors):
                            merged.add_(tensor.to(accumulator_dtype), alpha=weight)
                        output_tensor = merged.to(reference.dtype).contiguous()
                        check_finite_tensor(output_tensor, f"soup:{key}")
                        output_tensors[key] = output_tensor
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
        copy_relevant_metadata(models[0], temporary)
        output_config = semantic_config(temporary)
        if output_config != reference_config:
            raise ValueError("soup output config differs from the reference model config")

        manifest = {
            "schema_version": 1,
            "operation": "model_soup",
            "models": [str(path.resolve()) for path in models],
            "weights": normalized,
            "model_type": reference_config["model_type"],
            "tensor_count": len(keys),
            "total_size": total_size,
            "config_sha256": hashlib.sha256(
                json.dumps(reference_config, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "metadata_files": [candidate.name for candidate in relevant_metadata_files(models[0])],
        }
        (temporary / "soup_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return Path(output_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average merged Lychee-Reranker checkpoints")
    parser.add_argument("--model", action="append", required=True, help="repeat for every model")
    parser.add_argument("--weight", action="append", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-shard-size-gb", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    soup_models(
        args.model,
        args.weight,
        args.output_dir,
        max_shard_size_gb=args.max_shard_size_gb,
    )


if __name__ == "__main__":
    main()
