#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/eval/beir.yaml}
STAGE=${2:-all}
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi

exec lychee-reranker-eval --config "$CONFIG" --stage "$STAGE" "$@"
