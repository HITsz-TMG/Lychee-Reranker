# Validation evidence

Validated on 2026-10-14 using locally available checkpoints and synthetic public
text. The checks used a **CUDA-capable NVIDIA GPU with BF16 support**, PyTorch 2.8.0+cu129,
Transformers 5.3.0 and PEFT 0.18.1.
Core runtime versions and measured checks are in
[validation.json](validation.json). Private paths, credentials, datasets, checkpoints
and raw machine logs are excluded from this repository.

## Executed checks

| Check | Result |
| --- | --- |
| CPU tests | 61 passed, zero failures or skips |
| Tiny T5Gemma2 native DDP | Two CPU processes; real forward/backward, adapter save and exact resumed adapter hash |
| Nano GPU SFT | Rank 96 / alpha 48; two optimizer steps; finite losses, nonzero finite gradients and adapter updates |
| Nano GPU resume | Checkpoint 1 to step 2; final adapter file hash exactly matches uninterrupted training |
| Nano GPU KD | Independent ranks 64/96, alphas 32/48; two optimizer steps each from merged SFT |
| Real checkpoint merging | All three adapters merged and reloaded; BF16 score differences below 0.01 |
| ModelSoup | All 911 saved tensors match the exact equal-weight average; output reload succeeds |
| Inference | Six mixed English/Chinese, variable-length pairs at ratios 1, 2, 4, 8, 32 and 128; finite scores and batch/individual consistency |
| Evaluation | Real embedder and reranker on BEIR/FiQA, MIRACL/en and two LMEB/ConvoMem leaves; candidate reuse preserves metrics |
| Reference reranker parity | Released Nano checkpoint, six pairs at ratios 1/4/128: maximum score difference 0.0 |
| Reference embedding parity | Four texts, 896-dimensional FP32 vectors: maximum difference 6.51925802e-08, below 1e-5 |
| Static checks | Ruff lint/format, shell syntax, nine recipes, three workflow manifests and four CLI entrypoints pass |
| Packaging | Wheel and source distribution build; wheel source bytes match the tested package; isolated wheel import and all four installed entrypoints pass |
| Publication scan | Repository files, Git history and distribution contents checked for private paths, credentials and model/data artifacts |

The GPU pipeline took 679.84 seconds. Its LoRA ranks,
query group size 16, Nano batch size 2, accumulation 2, input limits and MEP
ratios match the supplied training recipes; the number of steps is deliberately
shortened. KD uses synthetic teacher margins to exercise the objective.
The measured adapter/merged score differences were sft: 1.09e-09, kd-r64: 0.00177, kd-r96: 0.00831.
The FP32 embedding reference was independently generated with the checkpoint's
original local model implementation under Transformers 4.51.0; the release uses
native Transformers 5.3.0 with a bidirectional padding mask.

The initial GPU check exposed a real resume discrepancy: global random negative
sampling ran during prefetch before Trainer restored its RNG state. Sampling
now depends on the data seed, epoch and record index. Both the stronger CPU
fixture and the subsequent real Nano GPU run verify exact resumed weights.

## Reproduce the checks

Install the versions in `requirements/repro.txt`, then run:

```bash
python -m pytest
ruff check src tests scripts
ruff format --check src tests scripts
python -m build
python scripts/check_release.py

bash scripts/gpu_smoke.sh \
  --base-model /path/to/t5gemma-2-270m-270m \
  --embedder-model /path/to/KaLM-Embedding-V2.5 \
  --work-dir /path/to/new-smoke-output
```

The shell launcher contains the required CUDA driver-library exports. Optional
`--reference-implementation` and `--reference-model` arguments run the legacy
reranker comparison. `--embedder-reference-vectors` accepts an independently
captured `.npy` matrix and a sibling JSON file containing the corresponding
`texts` list. Every invocation requires a fresh output directory.

## Verified boundary

These are execution, numerical-parity and data-contract checks. They do not
establish benchmark quality or reproduce the paper's published scores.
No complete training epoch on the real corpus, teacher annotation job, full
BEIR/MIRACL/LMEB evaluation, 32-GPU run, or real Small/Large GPU run was performed.
Small/Large recipes were checked structurally against the paper; their full
memory use and distributed performance remain unmeasured. The tested CPU
resume case and one-GPU resume case use the default non-persistent data workers.
