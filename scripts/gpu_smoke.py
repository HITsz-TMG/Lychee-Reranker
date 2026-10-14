"""Exercise the release pipeline on local models and synthetic, public test text.

This is an execution check, not a quality benchmark or full training run.
All generated data, checkpoints and logs stay in a new --work-dir.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = "Given a query, retrieve documents that answer the query."
PAIRS = [
    ("What is the capital of France?", "Paris is the capital of France."),
    ("What is the capital of France?", "Berlin is the capital of Germany."),
    ("Which city is the capital of Japan, and where is it located?", "Tokyo is in Japan."),
    ("日本的首都是哪里？", "东京是日本的首都。"),
    ("What do bees produce?", "Bees produce honey. " * 70),
    ("What do bees produce?", "A bicycle has two wheels."),
]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_digest() -> str:
    digest = hashlib.sha256()
    for folder in ("src", "configs", "scripts", "tests"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml", ".json", ".sh"}:
                digest.update(path.relative_to(ROOT).as_posix().encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def command(work: Path, name: str, args: list[str]) -> None:
    log = work / "logs" / f"{name}.log"
    log.parent.mkdir(exist_ok=True)
    print(f"[{name}] starting", flush=True)
    with log.open("w") as handle:
        result = subprocess.run(
            args, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode:
        raise RuntimeError(f"{name} failed (exit {result.returncode}); inspect {log}")
    print(f"[{name}] passed", flush=True)


def module(work: Path, name: str, mod: str, *args: str) -> None:
    command(work, name, [sys.executable, "-m", mod, *args])


def train_config(work: Path, name: str, base: Path, data: Path, rank: int, kd: bool) -> Path:
    recipe = "nano_distill_r64.yaml" if kd else "nano_sft.yaml"
    config = yaml.safe_load((ROOT / "configs" / "train" / recipe).read_text())
    config["model"]["name_or_path"] = str(base)
    config["model"]["lora"].update(rank=rank, alpha=rank / 2)
    config["data"]["train_data"] = [str(data)]
    config["training"].update(
        output_dir=str(work / name),
        max_steps=2,
        save_steps=1,
        logging_steps=1,
        dataloader_num_workers=0,
        disable_tqdm=True,
    )
    config["training"].pop("deepspeed", None)
    path = work / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def train(work: Path, name: str, config: Path, resume: Path | None = None) -> dict:
    args = [
        "--standalone",
        "--nproc-per-node=1",
        "--module",
        "lychee_reranker.train.cli",
        "--config",
        str(config),
    ]
    if resume:
        args.extend(["--resume-from-checkpoint", str(resume)])
    module(work, name + ("-resume" if resume else ""), "torch.distributed.run", *args)
    state = json.loads((work / name / "trainer_state.json").read_text())
    if state["global_step"] != 2:
        raise AssertionError("training did not complete two optimizer steps")
    rows = [row for row in state["log_history"] if "loss" in row]
    if not rows or not all(math.isfinite(row["loss"]) for row in rows):
        raise AssertionError("missing or non-finite training losses")
    gradients = [row["grad_norm"] for row in rows if "grad_norm" in row]
    if not gradients or not all(math.isfinite(g) and g > 0 for g in gradients):
        raise AssertionError("expected finite, nonzero gradients")
    with safe_open(work / name / "adapter_model.safetensors", framework="pt") as tensors:
        norms = [
            tensors.get_tensor(k).float().norm().item() for k in tensors.keys() if "lora_B" in k
        ]
    if not norms or max(norms) <= 0:
        raise AssertionError("LoRA B parameters were not updated")
    return {
        "optimizer_steps": 2,
        "losses": [r["loss"] for r in rows],
        "gradient_norms": gradients,
        "max_lora_B_norm": max(norms),
    }


def merge(work: Path, name: str, base: Path) -> Path:
    output = work / f"{name}-merged"
    module(
        work,
        f"{name}-merge",
        "lychee_reranker.checkpoint.merge",
        "--base-model",
        str(base),
        "--adapter",
        str(work / name),
        "--output-dir",
        str(output),
    )
    return output


def release_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def merge_parity(base: Path, adapter: Path, merged: Path) -> dict:
    from peft import PeftModel

    from lychee_reranker import LycheeReranker

    kwargs = {"device": "cuda", "dtype": "bfloat16", "batch_size": 3, "attn_implementation": "sdpa"}
    model = LycheeReranker(str(base), **kwargs)
    model.model = PeftModel.from_pretrained(model.model, str(adapter)).eval()
    expected = model.predict(PAIRS)
    del model
    release_gpu()
    model = LycheeReranker(str(merged), **kwargs)
    actual = model.predict(PAIRS)
    diff = float(np.max(np.abs(np.asarray(actual) - expected)))
    del model
    release_gpu()
    # Merging into BF16 rounds the dense weights once. CPU unit tests separately
    # check a real PEFT merge in FP32 at a stricter tolerance.
    if diff > 0.01:
        raise AssertionError(f"adapter/merged BF16 scoring deviation {diff} > 0.01")
    return {"pairs": len(PAIRS), "max_abs_diff": diff, "tolerance": 0.01}


def soup_tensor_checks(models: list[Path], soup: Path) -> dict:
    with ExitStack() as stack:
        readers = []
        for directory in [*models, soup]:
            mapped = {}
            for shard in sorted(directory.glob("*.safetensors")):
                handle = stack.enter_context(safe_open(shard, framework="pt"))
                for key in handle.keys():
                    if key in mapped:
                        raise AssertionError(f"duplicate tensor {key}")
                    mapped[key] = handle
            readers.append(mapped)
        if not readers[0] or not (set(readers[0]) == set(readers[1]) == set(readers[2])):
            raise AssertionError("soup tensor keys differ")
        for key in readers[0]:
            left, right, actual = [r[key].get_tensor(key) for r in readers]
            expected = (
                ((left.float() + right.float()) * 0.5).to(left.dtype)
                if left.is_floating_point()
                else left
            )
            if not torch.equal(actual, expected):
                raise AssertionError(f"soup is not an equal-weight average: {key}")
        return {"tensor_count": len(readers[0]), "equal_weight_exact": True}


def inference_checks(model_path: Path) -> dict:
    from lychee_reranker import LycheeReranker

    model = LycheeReranker(
        str(model_path), device="cuda", dtype="bfloat16", batch_size=3, attn_implementation="sdpa"
    )
    all_scores = {}
    for ratio in (1, 2, 4, 8, 32, 128):
        model.chunk_size = ratio
        scores = model.predict(PAIRS)
        if len(scores) != len(PAIRS) or not all(math.isfinite(s) and 0 <= s <= 1 for s in scores):
            raise AssertionError(f"invalid inference scores for ratio {ratio}")
        all_scores[str(ratio)] = scores
    model.chunk_size = 4
    single = model.predict(PAIRS, batch_size=1)
    delta = float(np.max(np.abs(np.array(single) - all_scores["4"])))
    if delta > 0.02:
        raise AssertionError(f"batch/single BF16 deviation {delta} > 0.02")
    if model.predict([]) != [] or len(model.rank(PAIRS[0][0], [p[1] for p in PAIRS], top_k=2)) != 2:
        raise AssertionError("empty/rank API contract failed")
    del model
    release_gpu()
    return {
        "ratios": list(all_scores),
        "finite_pairs_per_ratio": len(PAIRS),
        "batch_single_max_abs_diff": delta,
    }


def parity_checks(reference_file: Path, model_path: Path) -> dict:
    from lychee_reranker import LycheeReranker

    spec = importlib.util.spec_from_file_location("reference_reranker", reference_file)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    kwargs = dict(device="cuda", dtype="bfloat16", batch_size=3, attn_implementation="sdpa")
    old = reference.KaLMReranker(str(model_path), **kwargs)
    expected = {}
    for ratio in (1, 4, 128):
        old.chunk_size = ratio
        expected[ratio] = old.predict(PAIRS)
    del old
    release_gpu()
    new = LycheeReranker(str(model_path), **kwargs)
    diffs = {}
    for ratio in expected:
        new.chunk_size = ratio
        actual = new.predict(PAIRS)
        diff = float(np.max(np.abs(np.array(actual) - expected[ratio])))
        if diff > 1e-5:
            raise AssertionError(f"reference parity failed at ratio {ratio}: {diff}")
        diffs[str(ratio)] = diff
    del new
    release_gpu()
    return {
        "model_label": model_path.name,
        "reference_sha256": sha256(reference_file),
        "pairs": len(PAIRS),
        "max_abs_diff_by_ratio": diffs,
        "tolerance": 1e-5,
    }


def eval_fixtures(work: Path) -> dict[str, list[str]]:
    corpus = [{"_id": f"d{i}", "title": "", "text": p[1]} for i, p in enumerate(PAIRS)]
    queries = [{"_id": "q0", "text": PAIRS[0][0]}, {"_id": "q1", "text": PAIRS[2][0]}]
    qrels = [
        {"query-id": "q0", "corpus-id": "d0", "score": 2},
        {"query-id": "q1", "corpus-id": "d2", "score": 1},
    ]
    for benchmark, task, split in (("beir", "fiqa", "test"), ("miracl", "en", "dev")):
        root = work / "data" / benchmark / task
        jsonl(root / "corpus.jsonl", corpus)
        jsonl(root / f"{split}_queries.jsonl", queries)
        jsonl(root / f"{split}_qrels.jsonl", qrels)
    root = work / "data" / "lmeb"
    dump(
        root / "task_instructions.json",
        {"ConvoMem/demo": {"query": INSTRUCTION}, "ConvoMem/second": {"query": INSTRUCTION}},
    )
    jsonl(root / "ConvoMem" / "corpus.jsonl", corpus)
    jsonl(
        root / "ConvoMem" / "candidates.jsonl",
        [{"scene_id": q["_id"], "candidate_doc_ids": [d["_id"] for d in corpus]} for q in queries],
    )
    for leaf in ("demo", "second"):
        jsonl(root / "ConvoMem" / leaf / "queries.jsonl", queries)
        (root / "ConvoMem" / leaf / "qrels.tsv").write_text(
            "qid\tdocid\tscore\nq0\td0\t2\nq1\td2\t1\n"
        )
    return {"beir": ["fiqa"], "miracl": ["en"], "lmeb": ["ConvoMem"]}


def evaluate(work: Path, reranker: Path, embedder: Path) -> dict:
    summaries = {}
    for benchmark, tasks in eval_fixtures(work).items():
        config = yaml.safe_load((ROOT / "configs" / "eval" / f"{benchmark}.yaml").read_text())
        config.update(
            data_dir=str(work / "data" / benchmark),
            tasks=tasks,
            output_dir=str(work / "eval" / benchmark),
            retrieval_top_k=6,
            rerank_top_k=6,
        )
        config["embedder"].update(model_name_or_path=str(embedder), batch_size=3, device="cuda")
        config["reranker"].update(model_name_or_path=str(reranker), batch_size=3, device="cuda")
        path = work / f"eval-{benchmark}.yaml"
        path.write_text(yaml.safe_dump(config))
        module(
            work,
            f"eval-{benchmark}",
            "lychee_reranker.evaluation.cli",
            "--config",
            str(path),
            "--stage",
            "all",
        )
        before = json.loads((Path(config["output_dir"]) / "summary.json").read_text())
        original_output = Path(config["output_dir"])
        config["output_dir"] = str(work / "eval-reuse" / benchmark)
        path.write_text(yaml.safe_dump(config))
        module(
            work,
            f"eval-{benchmark}-reuse",
            "lychee_reranker.evaluation.cli",
            "--config",
            str(path),
            "--stage",
            "rerank",
            "--candidate-dir",
            str(original_output / "candidates"),
        )
        after = json.loads((Path(config["output_dir"]) / "summary.json").read_text())
        if before["tasks"] != after["tasks"]:
            raise AssertionError(f"candidate reuse changed {benchmark} metrics")
        summaries[benchmark] = {"leaves": len(before["tasks"]), "candidate_reuse_equal": True}
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--embedder-model", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="new output directory")
    parser.add_argument("--reference-implementation", type=Path)
    parser.add_argument("--reference-model", type=Path)
    args = parser.parse_args()
    for path in (args.base_model, args.embedder_model):
        if not path.is_dir():
            parser.error(f"local model directory does not exist: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; set the GPU environment before launching Python")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("this smoke test requires BF16-capable CUDA")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ["HF_DATASETS_CACHE"] = str(work / "cache")
    torch.set_num_threads(4)
    started = time.monotonic()
    report = {
        "scope": "synthetic execution smoke; not benchmark quality or full training",
        "release_source_sha256": release_digest(),
        "gpu": torch.cuda.get_device_name(0),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "peft", "accelerate", "datasets", "safetensors")
        },
    }
    rows = []
    for i in range(8):
        rows.append(
            {
                "query": PAIRS[i % len(PAIRS)][0],
                "prompt": INSTRUCTION,
                "pos": [PAIRS[i % len(PAIRS)][1]],
                "neg": [
                    f"Unrelated example {j}. " + "A bicycle has two wheels. " * 24
                    for j in range(16)
                ],
                "pos_scores": [2.5],
                "neg_scores": [-2.0 + j / 10 for j in range(16)],
            }
        )
    data = work / "train.jsonl"
    jsonl(data, rows)
    sft_cfg = train_config(work, "sft", args.base_model.resolve(), data, 96, False)
    report["sft"] = train(work, "sft", sft_cfg)
    before_resume = sha256(work / "sft" / "adapter_model.safetensors")
    train(work, "sft", sft_cfg, work / "sft" / "checkpoint-1")
    report["sft"]["resume_exact_adapter_sha256_match"] = before_resume == sha256(
        work / "sft" / "adapter_model.safetensors"
    )
    if not report["sft"]["resume_exact_adapter_sha256_match"]:
        raise AssertionError("resumed adapter differs from uninterrupted training")
    sft = merge(work, "sft", args.base_model.resolve())
    report["sft"]["merge_parity"] = merge_parity(args.base_model.resolve(), work / "sft", sft)
    merged = []
    for rank in (64, 96):
        name = f"kd-r{rank}"
        config = train_config(work, name, sft, data, rank, True)
        report[name] = train(work, name, config)
        merged_model = merge(work, name, sft)
        merged.append(merged_model)
        report[name]["merge_parity"] = merge_parity(sft, work / name, merged_model)
    soup = work / "soup"
    module(
        work,
        "soup",
        "lychee_reranker.checkpoint.soup",
        "--model",
        str(merged[0]),
        "--weight",
        "0.5",
        "--model",
        str(merged[1]),
        "--weight",
        "0.5",
        "--output-dir",
        str(soup),
    )
    report["soup"] = soup_tensor_checks(merged, soup)
    report["inference"] = inference_checks(soup)
    report["evaluation"] = evaluate(work, soup, args.embedder_model.resolve())
    if args.reference_implementation:
        report["reference_parity"] = parity_checks(
            args.reference_implementation, args.reference_model or args.base_model
        )
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    report["passed"] = True
    dump(work / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
