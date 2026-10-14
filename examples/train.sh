#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?usage: examples/train.sh CONFIG [training CLI options]}
shift

exec torchrun --standalone --nproc-per-node="${NPROC_PER_NODE:-1}" \
  -m lychee_reranker.train.cli \
  --config "$CONFIG" \
  "$@"
