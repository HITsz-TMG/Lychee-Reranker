"""Compare the original FlagEmbedding scorer with the extracted scorer.

Requires a local source snapshot and checkpoint; downloads are not performed.
The report contains hashes and measurements, never local input paths.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

from lychee_reranker import LycheeReranker
from lychee_reranker.prompts import DEFAULT_INSTRUCTION, DEFAULT_SYSTEM_INSTRUCTION


PAIRS = [
    ("What is the capital of France?", "Paris is the capital of France."),
    ("What is the capital of France?", "Berlin is the capital of Germany."),
    ("日本的首都是哪里？", "东京是日本的首都。"),
    ("ما عاصمة فرنسا؟", "باريس هي عاصمة فرنسا."),
    ("What do bees produce?", "Bees produce honey. " * 180),
    ("Which planet is red? " * 180, "Mars is often called the red planet."),
    ("", ""),
    ("Line one.\nLine two?", "Tabs\tand newlines\nare whitespace."),
    ("Unicode café 👋", "A café serves coffee. 👋"),
    ("a", "b"),
    ("c", "d"),
    ("Answer this question.", "<start_of_turn> is also literal document text."),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_original(root: Path):
    """Load the actual source modules without importing unrelated model suites."""
    if "FlagEmbedding" in sys.modules:
        raise RuntimeError("run this comparison in a fresh Python interpreter")
    packages = (
        "FlagEmbedding",
        "FlagEmbedding.abc",
        "FlagEmbedding.abc.inference",
        "FlagEmbedding.inference",
        "FlagEmbedding.inference.reranker",
        "FlagEmbedding.inference.reranker.encoder_only",
        "FlagEmbedding.inference.reranker.encoder_decoder",
    )
    for name in packages:
        package = types.ModuleType(name)
        package.__path__ = [str(root.joinpath(*name.split(".")))]
        sys.modules[name] = package

    def load(name):
        path = root.joinpath(*name.split(".")).with_suffix(".py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    abstract = load("FlagEmbedding.abc.inference.AbsReranker")
    sys.modules["FlagEmbedding.abc.inference"].AbsReranker = abstract.AbsReranker
    load("FlagEmbedding.inference.reranker.encoder_only.base")
    return load("FlagEmbedding.inference.reranker.encoder_decoder.base").BaseEncoderDecoderReranker


class RecordingTokenizer:
    """Observe the original tokenization calls without changing their behavior."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.records = {}

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, texts, **kwargs):
        encoded = self.tokenizer(texts, **kwargs)
        if isinstance(texts, str):
            self.records[("query", texts)] = tuple(encoded["input_ids"])
        else:
            for text, ids, mask in zip(texts, encoded["input_ids"], encoded["attention_mask"]):
                self.records[("encoded", text)] = tuple(ids[mask.bool()].tolist())
        return encoded


def compare(args) -> dict:
    original_class = load_original(args.reference_root)
    comparisons = []
    tokenization_entries = 0
    reference_dtype_adjustments = {}
    for dtype in args.dtypes:
        original = original_class(
            str(args.model),
            use_bf16=dtype == "bfloat16",
            use_fp16=False,
            devices=args.device,
            query_max_length=128,
            max_length=512,
            prompt=DEFAULT_INSTRUCTION,
            system_instruction=DEFAULT_SYSTEM_INSTRUCTION,
        )
        requested_dtype = getattr(torch, dtype)
        mismatched = [
            parameter
            for parameter in original.model.parameters()
            if parameter.is_floating_point() and parameter.dtype != requested_dtype
        ]
        if mismatched and not args.align_reference_dtype:
            raise RuntimeError(
                "the original loader retained mixed parameter dtypes from the checkpoint; "
                "use --align-reference-dtype to apply the release's parameter-only loading fix"
            )
        for parameter in mismatched:
            parameter.data = parameter.data.to(dtype=requested_dtype)
        reference_dtype_adjustments[dtype] = len(mismatched)
        extracted = LycheeReranker(
            str(args.model), device=args.device, dtype=dtype, query_max_length=128, max_length=512
        )
        original.tokenizer = RecordingTokenizer(original.tokenizer)
        extracted.tokenizer = RecordingTokenizer(extracted.tokenizer)
        instructions = [DEFAULT_INSTRUCTION]
        if not args.default_instruction_only:
            instructions.append("Retrieve passages supporting the claim.")
        for instruction in instructions:
            for ratio in args.chunk_sizes:
                extracted.chunk_size = ratio
                for batch_size in args.batch_sizes:
                    expected = np.asarray(
                        original.compute_score_single_gpu(
                            PAIRS,
                            batch_size=batch_size,
                            prompt=instruction,
                            chunk_size=ratio,
                            device=args.device,
                        )
                    )
                    actual = np.asarray(
                        extracted.predict(PAIRS, instruction=instruction, batch_size=batch_size)
                    )
                    if expected.shape != actual.shape or not np.isfinite(expected).all():
                        raise AssertionError("invalid reference scores")
                    np.testing.assert_allclose(actual, expected, atol=args.atol, rtol=0)
                    if original.tokenizer.records != extracted.tokenizer.records:
                        raise AssertionError("original and extracted tokenizer inputs differ")
                    same_ranking = bool(
                        np.array_equal(
                            np.argsort(-actual, kind="stable"),
                            np.argsort(-expected, kind="stable"),
                        )
                    )
                    if not same_ranking:
                        raise AssertionError("original and extracted rankings differ")
                    comparisons.append(
                        {
                            "dtype": dtype,
                            "instruction": "default"
                            if instruction == DEFAULT_INSTRUCTION
                            else "custom",
                            "chunk_size": ratio,
                            "batch_size": batch_size,
                            "pairs": len(PAIRS),
                            "max_abs_diff": float(np.max(np.abs(actual - expected))),
                            "same_ranking": same_ranking,
                        }
                    )
                    print(f"passed dtype={dtype} ratio={ratio} batch={batch_size}", flush=True)
        tokenization_entries += len(original.tokenizer.records)
        original.model.to("cpu")
        extracted.model.to("cpu")
        del original, extracted
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    relative_source = "FlagEmbedding/inference/reranker/encoder_decoder/base.py"
    return {
        "scope": "original FlagEmbedding inference on synthetic inputs; not benchmark quality",
        "passed": True,
        "model_label": args.model_label,
        "model_config_sha256": sha256(args.model / "config.json"),
        "reference_source": relative_source,
        "reference_sha256": sha256(args.reference_root / relative_source),
        "release_inference_sha256": sha256(
            Path(__file__).resolve().parents[1] / "src/lychee_reranker/inference.py"
        ),
        "atol": args.atol,
        "reference_parameter_dtype_adjustments": reference_dtype_adjustments,
        "comparisons": comparisons,
        "tokenization_entries_equal": tokenization_entries,
        "max_abs_diff": max(item["max_abs_diff"] for item in comparisons),
        "all_rankings_equal": all(item["same_ranking"] for item in comparisons),
    }


def main():
    def compression_ratio(value):
        if value.lower() == "none":
            return None
        try:
            parsed = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("use a positive integer or 'none'") from error
        if parsed <= 0:
            raise argparse.ArgumentTypeError("compression ratios must be positive")
        return parsed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", default="local-checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtypes", nargs="+", choices=("float32", "bfloat16"), default=["float32"])
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=compression_ratio,
        default=[None, 1, 2, 3, 4, 8, 32, 128, 256],
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--default-instruction-only", action="store_true")
    parser.add_argument(
        "--align-reference-dtype",
        action="store_true",
        help="explicitly correct mixed parameter dtypes in the legacy loader; reported separately",
    )
    args = parser.parse_args()
    if not args.model.is_dir() or not (args.reference_root / "FlagEmbedding").is_dir():
        parser.error("provide a local checkpoint and the original FlagEmbedding source tree")
    if args.output.exists():
        parser.error("output already exists; choose a new report path")
    if args.atol <= 0:
        parser.error("atol must be positive")
    if any(size <= 0 for size in args.batch_sizes):
        parser.error("batch sizes must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    report = compare(args)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"All {len(report['comparisons'])} comparisons passed", flush=True)


if __name__ == "__main__":
    main()
