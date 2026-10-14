# Training contract

The reference is [arXiv:2606.22807v4](https://arxiv.org/html/2606.22807v4),
Sections 4 and Appendices B/D. The retained objectives are pointwise binary
SFT and teacher-soft-label binary cross-entropy. There is no contrastive,
listwise ranking, or mixed experimental objective.

| Setting | Nano | Small | Large |
| --- | --- | --- | --- |
| T5Gemma2 backbone | 270m–270m | 1b–1b | 4b–4b |
| LoRA SFT rank / alpha | 96 / 48 | 96 / 48 | 96 / 48 |
| KD ranks / alphas | 64 / 32 and 96 / 48 | 64 / 32 and 96 / 48 | 64 / 32 and 96 / 48 |
| Query groups per GPU | 2 | 2 | 1 |
| Accumulation steps | 2 | 2 | 4 |
| Target world size | 32 | 32 | 32 |
| Global query-group batch | 128 | 128 | 128 |

Shared settings: one epoch for each optimization stage, learning rate 0.0002,
cosine schedule, warmup ratio 0.1, AdamW betas (0.9, 0.999), weight decay 0.00001,
gradient clipping 1.0 and BF16. Each sampled query group has one positive and
15 negatives. Raw queries and prefixed passages are truncated to 128 and 512
tokens; the decoder also contains the instruction and prompt template.

Nano and Small adapt `q_proj`, `k_proj`, `v_proj`, `out_proj`, `gate_proj`,
`up_proj` and `down_proj` in both encoder and decoder. Large adapts the four
attention projections. All stages pool the same encoder states at ratios
2, 8, 32 and 128. Each ratio has weight 1; losses are summed across ratios,
not divided by four. Padded document tokens are excluded from each mean.

For logits `[z_no, z_yes]`, SFT is binary-label cross-entropy. KD uses
`y = sigmoid(teacher_yes_logit - teacher_no_logit)` and
`BCEWithLogits(z_yes - z_no, y)`. The teacher is Qwen3-Reranker-8B. The original
code's KL-equivalent objective subtracted teacher entropy; this extraction
uses the paper's BCE value directly, with the same gradient at temperature 1.

Data preparation is a separate, explicit step. The paper describes mining
hard negatives and producing teacher scores before training. This package
reads those prepared shards and preserves multiplicity; it neither mines a
corpus nor replaces missing teacher scores with hard labels. The one-negative
README record illustrates the schema only; use the prepared negative pools
for paper training.

## Launch and effective batch

`global_query_groups = world_size * per_device_train_batch_size * gradient_accumulation_steps`.
The effective number of query–passage pairs is this value multiplied by 16.
On one GPU, Nano's paper batch can be maintained by setting accumulation to 64
instead of 2; this changes the runtime and is not a multi-GPU validation.

For four nodes with eight GPUs each, run on every node with its own `NODE_RANK`:

```bash
# MASTER_ADDR is the routable address of node 0, supplied by your scheduler.
torchrun --nnodes=4 --nproc-per-node=8 \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" \
  --master-port="${MASTER_PORT:-29500}" \
  -m lychee_reranker.train.cli \
  --config configs/train/nano_sft.yaml \
  --model-name-or-path /path/to/base-model \
  --train-data /path/to/prepared-shards \
  --output-dir /path/to/shared-new-output
```

Use the same model, data, seed, world size and training settings on every rank.
The default recipes use native PyTorch DDP; DeepSpeed/FSDP are outside the
supported checkpoint path. `--print-config` shows the resolved configuration
without loading a model. Do not reuse output directories for unrelated runs.

## Checkpoints

Training saves adapters and tokenizer files. Merge the SFT adapter into its
original backbone before starting either KD branch. Merge each KD adapter
into that same SFT model, then average the resulting complete models. The
soup checks tensor keys, shapes and dtypes, model configuration, tokenizer
compatibility and non-floating tensor equality. Outputs are created atomically
and existing destinations are refused.

The recipe manifests describe the stage dependencies. They are portable
instructions, not proof of historical checkpoint identity. Pin model revisions
and archive the prepared data, effective configuration and training manifests
when starting a reproducible full run.
