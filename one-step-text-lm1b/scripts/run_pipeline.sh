#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run.sh"

DATASET="${DATASET:-lm1b}"
STAGE_A_CONFIG="${STAGE_A_CONFIG:-configs/stage_a.yaml}"
STAGE_B_CONFIG="${STAGE_B_CONFIG:-configs/stage_b.yaml}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-coupling_${DATASET}}"
PREPARE_DATA="${PREPARE_DATA:-1}"
RUN_EVAL="${RUN_EVAL:-0}"

if [[ "$PREPARE_DATA" == "1" ]]; then
  "$RUNNER" python prepare.py --dataset "$DATASET" --tokenizer qwen
fi

"$RUNNER" python train_stage_a.py --dataset "$DATASET" --config "$STAGE_A_CONFIG"
"$RUNNER" python train_stage_b.py --dataset "$DATASET" --config "$STAGE_B_CONFIG"

if [[ "$RUN_EVAL" == "1" ]]; then
  "$RUNNER" python eval.py --checkpoint "stage_b/v3_${DATASET}/${EXPERIMENT_NAME}/checkpoint.ckpt" --temperature 0.65 --top_p 0.95
fi
