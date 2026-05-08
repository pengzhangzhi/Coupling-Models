#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LTLM_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

_default_runtime_root() {
  if [[ -n "${LTLM_RUNTIME_ROOT:-}" ]]; then
    printf '%s\n' "$LTLM_RUNTIME_ROOT"
  elif [[ -n "${SCRATCH:-}" ]]; then
    printf '%s\n' "$SCRATCH/latent-transport-lm"
  else
    printf '%s\n' "$LTLM_REPO_ROOT/.runtime"
  fi
}

export LTLM_RUNTIME_ROOT="$(_default_runtime_root)"
export LTLM_DATA_ROOT="${LTLM_DATA_ROOT:-$LTLM_RUNTIME_ROOT/data}"
export LTLM_CHECKPOINT_ROOT="${LTLM_CHECKPOINT_ROOT:-$LTLM_RUNTIME_ROOT/checkpoints}"
export LTLM_WANDB_DIR="${LTLM_WANDB_DIR:-$LTLM_RUNTIME_ROOT/wandb}"
export LTLM_TMPDIR="${LTLM_TMPDIR:-$LTLM_RUNTIME_ROOT/tmp}"
export LTLM_UV_CACHE_DIR="${LTLM_UV_CACHE_DIR:-$LTLM_RUNTIME_ROOT/uv-cache}"
export UV_PROJECT_ENVIRONMENT="${LTLM_VENV_PATH:-$LTLM_RUNTIME_ROOT/.venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$LTLM_UV_CACHE_DIR}"
export TMPDIR="${TMPDIR:-$LTLM_TMPDIR}"
export TEMP="${TEMP:-$LTLM_TMPDIR}"
export TMP="${TMP:-$LTLM_TMPDIR}"

mkdir -p \
  "$LTLM_RUNTIME_ROOT" \
  "$LTLM_DATA_ROOT" \
  "$LTLM_CHECKPOINT_ROOT" \
  "$LTLM_WANDB_DIR" \
  "$LTLM_TMPDIR" \
  "$LTLM_UV_CACHE_DIR" \
  "$(dirname "$UV_PROJECT_ENVIRONMENT")"

if [[ "${1:-}" == "--create-venv" ]]; then
  PYTHON_BIN="${LTLM_BOOTSTRAP_PYTHON:-python3}"
  cd "$LTLM_REPO_ROOT"
  uv venv "$UV_PROJECT_ENVIRONMENT" --python "$PYTHON_BIN"
  # shellcheck source=/dev/null
  source "$UV_PROJECT_ENVIRONMENT/bin/activate"
  uv sync --project "$LTLM_REPO_ROOT" --dev --active
  echo "Environment ready at $UV_PROJECT_ENVIRONMENT"
fi
