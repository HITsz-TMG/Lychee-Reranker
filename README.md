# Lychee-Reranker

Focused training, inference and evaluation code for
[Lychee-Reranker](https://arxiv.org/abs/2606.22807v4), formerly KaLM-Reranker-V1.
The package supports the Nano, Small and Large T5Gemma2 encoder–decoder models.
It contains SFT, teacher-score distillation, LoRA merging, ModelSoup, and
BEIR/MIRACL/LMEB evaluation. It does not depend on FlagEmbedding at runtime.

Model weights, real datasets and experiment outputs are supplied separately.
All paths below are placeholders. See [training details](docs/training.md) and
[validation evidence](docs/validation.md) before reproducing an experiment.

## Installation

Use Python 3.10 or newer, in a dedicated environment. Install a PyTorch build
suited to your CUDA driver first, then install this repository:

```bash
python -m pip install -e '.[train,eval,faiss,dev]' -c requirements/repro.txt
```

`requirements/repro.txt` records the tested dependency versions. The `faiss`
extra accelerates exact retrieval on larger corpora. Small fixtures can use the
NumPy backend without it. Neither Flash Attention nor DeepSpeed is required.
The default launcher uses native PyTorch distributed training.

## Inference

```python
from lychee_reranker import LycheeReranker

reranker = LycheeReranker(
    "/path/to/merged-reranker",
    device="cuda",
    dtype="bfloat16",
    chunk_size=4,
)
scores = reranker.predict([
    ("What is the capital of France?", "Paris is the capital of France."),
    ("What is the capital of France?", "Berlin is the capital of Germany."),
])
```

Scores are `P(yes)` from the `yes`/`no` logits, in input order. `rank(query,
 documents, top_k=10)` returns ranked input indices and scores. The compression
ratio is controlled by `chunk_size`; `1` retains the full representation.

## Training

The paper workflow is **SFT → two independent KD branches → merge each adapter
→ equal-weight ModelSoup**. The two KD branches both start from the merged SFT
checkpoint. Nine recipes cover all three model sizes in `configs/train/`.

```bash
# Stage 1: use the Nano backbone, or the corresponding Small/Large recipe.
torchrun --standalone --nproc-per-node=1 -m lychee_reranker.train.cli \
  --config configs/train/nano_sft.yaml \
  --model-name-or-path /path/to/t5gemma-2-270m-270m \
  --train-data /path/to/supervised-shards \
  --output-dir outputs/nano/sft-adapter

lychee-reranker-merge \
  --base-model /path/to/t5gemma-2-270m-270m \
  --adapter outputs/nano/sft-adapter \
  --output-dir outputs/nano/sft-merged

# Stage 2: independent KD branches, initialized from Stage 1.
for rank in 64 96; do
  torchrun --standalone --nproc-per-node=1 -m lychee_reranker.train.cli \
    --config "configs/train/nano_distill_r${rank}.yaml" \
    --model-name-or-path outputs/nano/sft-merged \
    --train-data /path/to/teacher-scored-shards \
    --output-dir "outputs/nano/kd-r${rank}-adapter"

  lychee-reranker-merge \
    --base-model outputs/nano/sft-merged \
    --adapter "outputs/nano/kd-r${rank}-adapter" \
    --output-dir "outputs/nano/kd-r${rank}-merged"
done

# Stage 3: average merged models, not raw adapters of different ranks.
lychee-reranker-soup \
  --model outputs/nano/kd-r64-merged --weight 0.5 \
  --model outputs/nano/kd-r96-merged --weight 0.5 \
  --output-dir outputs/nano/final
```

These example commands use one GPU. The paper uses **32 GPUs** with an effective
batch of 128 query groups. Adjust the distributed launch and accumulation as
explained in [training.md](docs/training.md); one GPU with the unchanged recipe
has a smaller effective batch. Resume a training run with
`--resume-from-checkpoint /path/to/output/checkpoint-N` and the same recipe.

Training accepts JSON, JSONL or Parquet with this schema:

```json
{
  "query": "What do bees produce?",
  "prompt": "Given a query, retrieve documents that answer the query.",
  "pos": ["Bees produce honey."],
  "neg": ["A bicycle has two wheels."],
  "pos_scores": [2.5],
  "neg_scores": [-1.8]
}
```

The score fields are required for KD and contain the teacher's
`yes_logit - no_logit`, **not probabilities**. The loader applies sigmoid to
obtain soft labels. The released workflow consumes prepared teacher scores;
it does not run the 8B teacher or regenerate the training corpus implicitly.
Repeated records and repeated shard paths preserve training multiplicity.

## Evaluation

Evaluation uses local benchmark files. The supplied configurations cover the
paper's 13 BEIR tasks, 18 MIRACL languages and six LMEB Dialogue groups. The
first stage retrieves 100 candidates using KaLM-Embedding-V2.5, then the
reranker scores the same candidates. The main metric is nDCG@10.

```bash
# Retrieve and rerank.
lychee-reranker-eval --config configs/eval/beir.yaml --stage all \
  --data-dir /path/to/beir \
  --embedder-name-or-path /path/to/KaLM-Embedding-V2.5 \
  --reranker-name-or-path /path/to/merged-reranker \
  --output-dir results/beir

# Reuse candidates without loading the embedder.
lychee-reranker-eval --config configs/eval/beir.yaml --stage rerank \
  --data-dir /path/to/beir \
  --reranker-name-or-path /path/to/another-merged-reranker \
  --candidate-dir results/beir/candidates \
  --output-dir results/beir-another-model
```

See [evaluation.md](docs/evaluation.md) for layouts, split selection, candidate
integrity, LMEB filtering, aggregation and metrics. JSON metrics are on the
0–1 scale; Markdown summaries display percentages.

## Tests

CPU checks do not download models or datasets:

```bash
python -m pip install pytrec-eval-terrier==0.5.10
python -m pytest
ruff check src tests scripts
ruff format --check src tests scripts
python -m build
python scripts/check_release.py --history
```

The GPU smoke test uses local models, synthetic text, and a **new** output
directory. It executes SFT, checkpoint resume, both KD branches, merging,
ModelSoup, multiple compression ratios, and all three evaluation pipelines.
It uses the Nano recipe's LoRA ranks, batch/group sizes and MEP ratios, but only
two optimizer steps per branch. Synthetic teacher scores check the KD path;
they are not outputs from Qwen3-Reranker-8B.

```bash
bash scripts/gpu_smoke.sh \
  --base-model /path/to/t5gemma-2-270m-270m \
  --embedder-model /path/to/KaLM-Embedding-V2.5 \
  --work-dir /path/to/new-smoke-output
```

For comparison with an earlier standalone implementation, additionally pass
`--reference-implementation /path/to/kalm_reranker.py` and
`--reference-model /path/to/released-Nano-checkpoint`. GPU environment exports
are included in the shell launcher. Set `PYTHON=/path/to/python` to select an
environment. Generated training data, logs, dataset caches and weights stay in
the output directory by default. If the output filesystem does not support
file locks, set `HF_DATASETS_CACHE=/path/to/local-cache` to put only the dataset
cache on a local filesystem.

## License and provenance

Code is MIT-licensed. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) retains
FlagEmbedding attribution. Models and datasets have separate licenses.
This is a reviewed extraction and correction of the original implementation;
exact historical base revisions and a complete original training-data order
were unavailable. The recipes implement arXiv v4 and do not claim bitwise
reproduction of the published model weights.
