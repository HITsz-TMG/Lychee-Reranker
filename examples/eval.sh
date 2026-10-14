#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/eval/beir.yaml}
STAGE=${2:-all}

lychee-reranker-eval --config "$CONFIG" --stage "$STAGE"
