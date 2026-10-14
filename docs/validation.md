# Validation evidence

Validated on 2026-10-15 using locally available checkpoints and synthetic public
text. Checks used a **CUDA-capable NVIDIA GPU with BF16 support**, PyTorch
2.8.0+cu129, Transformers 5.3.0 and PEFT 0.18.1. Core versions and measurements
are in [validation.json](validation.json).
Private paths, credentials, datasets, checkpoints and raw machine logs are
excluded from this repository.

## Executed checks

| Check | Result |
| --- | --- |
| CPU tests | 92 passed, zero failures or skips; includes installed FAISS and pytrec_eval reference checks |
| Tiny T5Gemma2 native DDP | Two CPU processes; real forward/backward, adapter save and exact resumed adapter hash |
| Nano GPU SFT | Rank 96 / alpha 48; two optimizer steps; finite losses, nonzero finite gradients and adapter updates |
| Nano GPU resume | Checkpoint 1 to step 2; final adapter hash exactly matches uninterrupted training |
| Nano GPU KD | Independent ranks 64/96, alphas 32/48; two optimizer steps each from merged SFT |
| Real checkpoint merging | All three adapters merged and reloaded; BF16 score differences below 0.01 |
| ModelSoup | All 911 saved tensors match the exact equal-weight average; output reload succeeds |
| Inference | Six mixed English/Chinese, variable-length pairs at ratios 1, 2, 4, 8, 32 and 128; finite scores and batch/individual consistency |
| Evaluation | Real embedder and reranker on BEIR/FiQA, MIRACL/en and two LMEB/ConvoMem leaves; candidate reuse preserves metrics |
| Actual FlagEmbedding scorer | Nano: 72 configurations / 864 pair scores, maximum difference 1.37090683e-6; Small/Large: 36 scores each, exact BF16 match; tokenization and rankings agree |
| Actual FlagEmbedding embedder | Six dtype/batch configurations, 16 texts / 896 dimensions; FP16/BF16 exact match, FP32 maximum difference 2.16066837e-7 |
| Actual training components | Data/collator IDs, masks, labels and teacher margins agree; shared tiny-model SFT/KD gradients agree within measured floating-point tolerance |
| Saved candidate ties | Two regressions fail before the writer fix and pass afterward; same top-k membership and MRR after save/reload |
| Dataset aggregation | Complete synthetic CQADupStack with twelve children; expected macro nDCG@1 of 0.5 |
| Static checks | Ruff lint/format, shell syntax, nine recipes, three workflow manifests and four CLI entrypoints pass |
| Packaging | Wheel and source distribution build; wheel source bytes match the tested package; isolated wheel import and all four installed entrypoints pass |
| Publication scan | Index, worktree, HEAD-reachable history and distribution contents checked for private paths, credentials, binary/model/data artifacts and machine identifiers |

The Nano GPU pipeline took 825.8 seconds. Its LoRA ranks,
query group size 16, batch size 2, accumulation 2, input limits and MEP ratios
match the supplied recipes; each training stage is limited to two steps.
KD uses synthetic teacher margins to exercise the objective.
Measured adapter/merged score differences were SFT: 1.09e-9, KD-r64: 0.00177
and KD-r96: 0.00831. The earlier standalone Nano scorer comparison also passed
exactly at ratios 1/4/128. The stronger actual FlagEmbedding comparisons above
are separate from that standalone comparison. The Nano FP32 reference required
parameter-dtype alignment; the BF16 checks required no reference dtype adjustment.

The original embedder reference was generated with its local custom model
under Transformers 4.51.0; the release uses native Transformers 5.3.0 with a
bidirectional mask, FP32 mean pooling and matching length-based batching.

The full pipeline ran on the snapshot recorded by `release_source_sha256`.
Afterward, only the evaluation run writer changed in runtime code; a Nano
provenance note and two regression tests were also updated. The final source
passed the complete CPU suite and a new GPU evaluation run using the existing
smoke ModelSoup checkpoint. That evaluation rerun took
230.9 seconds and repeated all three benchmarks and their
candidate-reuse paths. `publication_source_sha256` and `evaluation_followup`
identify this final snapshot. Training, inference and checkpoint code are
unchanged from the full pipeline; full training was not repeated for the writer fix.

The initial release check had exposed a resume discrepancy from global random
sampling during data prefetch. Sampling now depends on data seed, epoch and
record index. Both the two-process CPU fixture and the Nano GPU run verify
exact resumed weights. This intentionally changes historical sample order.

## Reproduce the checks

Install the versions in `requirements/repro.txt` with the training, evaluation,
FAISS and development extras, then run:

```bash
python -m pip install pytrec-eval-terrier==0.5.10
python -m pytest
ruff check src tests scripts
ruff format --check src tests scripts
python -m build
python scripts/check_release.py --history

bash scripts/gpu_smoke.sh \
  --base-model /path/to/t5gemma-2-270m-270m \
  --embedder-model /path/to/KaLM-Embedding-V2.5 \
  --work-dir /path/to/new-smoke-output
```

The shell launcher contains the required CUDA driver-library exports. Optional
`--reference-implementation` and `--reference-model` arguments run the legacy
standalone scorer comparison. For the actual FlagEmbedding scorer comparison,
run `python scripts/compare_flagembedding.py --help`.
`--embedder-reference-vectors` accepts an independently captured `.npy` matrix
and a sibling JSON file containing the corresponding `texts` list.
Every full smoke invocation requires a fresh output directory. On filesystems
without working file locks, set `HF_DATASETS_CACHE=/path/to/local-cache`.

## Verified boundary

These are execution, numerical-parity and data-contract checks. They do not
establish benchmark quality or reproduce the paper's published scores.
No complete training epoch on the real corpus, teacher annotation job, full
BEIR/MIRACL/LMEB evaluation or 32-GPU run was performed. Small/Large inference
was checked on supplied checkpoints; their full training memory use and
distributed performance remain unmeasured. The tested CPU and one-GPU resume
cases use the default non-persistent data workers. Sampling order differs from
the historical global RNG sequence. At the recipe's temperature 1, the reported
KD loss differs from the original by a teacher-entropy constant; the tested
student gradients agree.
