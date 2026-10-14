#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/train/nano_sft.yaml}
DATA_ROOT=${2:?usage: examples/train.sh CONFIG DATA_ROOT OUTPUT_DIR}
OUTPUT_DIR=${3:?usage: examples/train.sh CONFIG DATA_ROOT OUTPUT_DIR}

torchrun --standalone --nproc-per-node="${NPROC_PER_NODE:-1}" \
  -m lychee_reranker.train.cli \
  --config "$CONFIG" \
  --train-data "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR"
