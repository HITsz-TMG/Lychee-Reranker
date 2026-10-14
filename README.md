# Lychee-Reranker-V1

Minimal, standalone training, inference, checkpoint, and evaluation code for
[Lychee-Reranker-V1](https://arxiv.org/abs/2606.22807). This repository supports
the Nano, Small, and Large encoder-decoder rerankers without importing
FlagEmbedding at runtime.

This repository contains code and reproducibility configurations only. Model
weights, training data, benchmark data, and generated results are not bundled.

## Install

Python 3.10 is recommended. The original code path was validated with PyTorch
2.8.0, Transformers 5.3.0, Datasets 4.8.4, Accelerate 1.13.0, PEFT 0.18.1,
DeepSpeed 0.18.8, and PyArrow 23.0.1. See `requirements/repro.txt` for the
complete reference environment.

```bash
pip install -e '.[train,eval,deepspeed]'
```

Install a PyTorch wheel compatible with your CUDA driver before installing the
remaining requirements. Flash Attention is not required. The T5Gemma2 models
load natively in Transformers 5.3.0, so `trust_remote_code` defaults to false.

## Inference

```python
from lychee_reranker import LycheeReranker

model = LycheeReranker(
    "models/Lychee-Reranker-V1-Nano",
    dtype="bfloat16",
    chunk_size=4,
)
scores = model.predict([
    ("query", "relevant document"),
    ("query", "unrelated document"),
])
```

The returned score is `P(yes)` from the lowercase `yes` and `no` logits. Query
and document inputs are truncated independently, encoder states are mean-pooled
in chunks, and scores are restored to caller order after length-sorted batching.

## Training data

JSON, JSONL, and Parquet shards share one schema:

```json
{
  "query": "...",
  "pos": ["..."],
  "neg": ["..."],
  "prompt": "Given a query, retrieve documents that answer the query.",
  "pos_scores": [4.2],
  "neg_scores": [-2.3]
}
```

`pos_scores` and `neg_scores` are mandatory only for distillation and represent
the teacher's `yes_logit - no_logit`. The loader preserves duplicate rows and
repeated source paths because these encode training multiplicity. It validates
non-empty candidates, score lengths, and finite values.

## Training and release2 construction

Nine explicit recipes live in `configs/train`: one SFT recipe and two KD recipes
for each model size. Use `--train-data` and `--output-dir` to replace the portable
placeholder paths in a recipe.

```bash
torchrun --standalone --nproc-per-node=8 -m lychee_reranker.train.cli \
  --config configs/train/nano_sft.yaml \
  --train-data /path/to/sft/parquet \
  --output-dir outputs/nano/sft-adapter

lychee-reranker-merge \
  --base-model google/t5gemma-2-270m-270m \
  --adapter outputs/nano/sft-adapter \
  --output-dir outputs/nano/sft-merged

torchrun --standalone --nproc-per-node=8 -m lychee_reranker.train.cli \
  --config configs/train/nano_distill_r64.yaml \
  --train-data /path/to/distillation/parquet

torchrun --standalone --nproc-per-node=8 -m lychee_reranker.train.cli \
  --config configs/train/nano_distill_r96.yaml \
  --train-data /path/to/distillation/parquet

lychee-reranker-merge --base-model outputs/nano/sft-merged \
  --adapter outputs/nano/kd-r64-adapter --output-dir outputs/nano/kd-r64-merged
lychee-reranker-merge --base-model outputs/nano/sft-merged \
  --adapter outputs/nano/kd-r96-adapter --output-dir outputs/nano/kd-r96-merged

lychee-reranker-soup \
  --model outputs/nano/kd-r64-merged --weight 0.5 \
  --model outputs/nano/kd-r96-merged --weight 0.5 \
  --output-dir outputs/nano/release2
```

SFT uses independent binary cross-entropy over `[no, yes]` for every passage.
KD matches the teacher logit difference with a Bernoulli KL-equivalent loss.
Losses from chunk sizes 2, 8, 32, and 128 are summed, matching the release
training code; returned scores are the normalized average across chunks.

The soup command accepts only merged safetensors models with identical tensor
keys, shapes, dtypes, model config, and tokenizer files. Floating tensors are
averaged; integer tensors must be identical. Outputs are written atomically with
a manifest.

### Provenance limitation

The canonical Nano r96 KD launcher was missing from the source snapshot. Its
public recipe was reconstructed from the final adapter metadata and the common
release2 KD settings. The legacy `_ck37` launcher used a different mixed loss
and is intentionally excluded. Consequently, the provided Nano recipe is not
claimed to reproduce the original checkpoint bit-for-bit. The source snapshot
also lacked Git history and exact base-model revisions; pin revisions before an
archival reproduction run.

## Evaluation

Evaluation is local-data-only and fails when required files are missing. It
never downloads benchmark data implicitly.

```bash
# Generate Top-100 candidates and rerank them.
lychee-reranker-eval --config configs/eval/beir.yaml --stage all \
  --data-dir /path/to/beir --output-dir results/beir

# Generate candidates only.
lychee-reranker-eval --config configs/eval/miracl.yaml --stage retrieve

# Reuse candidate JSON files from another run.
lychee-reranker-eval --config configs/eval/miracl.yaml --stage rerank \
  --candidate-dir /path/to/candidates
```

Candidate and result files include a schema version, benchmark/split, model ID
and revision, configuration digest, input-manifest digest, and deterministic
ranked results. JSON metrics use the `[0, 1]` scale; Markdown summaries display
percentages.

### Dataset layouts

BEIR:

```text
<root>/<dataset>/corpus.jsonl
<root>/<dataset>/test_queries.jsonl
<root>/<dataset>/test_qrels.jsonl
```

MIRACL:

```text
<root>/<language>/corpus.jsonl
<root>/<language>/dev_queries.jsonl
<root>/<language>/dev_qrels.jsonl
```

LMEB:

```text
<root>/task_instructions.json
<root>/<group>/corpus.jsonl
<root>/<group>/candidates.jsonl
<root>/<group>/<leaf>/queries.jsonl
<root>/<group>/<leaf>/qrels.tsv
```

For LMEB, qrels are restricted to each query's candidate pool and queries with
no positive candidate are removed, matching the original evaluator. Retrieval
is also restricted to that pool.

The supplied configs reproduce the paper task sets: 13 BEIR datasets, 18 MIRACL
languages, and LMEB Dialogue groups ConvoMem, LoCoMo, LongMemEval, MemBench,
REALTALK, and TMD. The main metrics are nDCG@10 plus the benchmark-specific
recall, nDCG@100, and MRR values declared in each config.

## Validation status

The repository's unit tests cover schema validation, JSONL/Parquet loading,
prompt construction, chunk pooling, SFT/KD losses, benchmark loaders, candidate
round-tripping, deterministic ranking, metrics, and model-soup rejection paths.
CI does not download models or datasets.

Full training and full benchmark evaluation are intentionally not run as part
of repository construction. The local Nano checkpoint passed both an FP32 CPU
single-pair parity smoke test and the fixed six-pair FiQA BF16 GPU comparison
against the original FlagEmbedding wrapper. Both produced `max_abs_diff=0.0`;
the FiQA ranking order was identical under the `1e-5` strict tolerance.

## License

Code is released under the MIT License. See `THIRD_PARTY_NOTICES.md` for the
FlagEmbedding attribution. Model and dataset licenses must be reviewed
separately before redistributing those artifacts.
