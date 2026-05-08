#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/setup_env.sh"

if [[ $# -eq 0 ]]; then
  echo "Usage: scripts/run.sh <command> [args...]" >&2
  exit 1
fi

if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
  echo "Python environment not found at $UV_PROJECT_ENVIRONMENT" >&2
  echo "Create it with: bash scripts/setup_env.sh --create-venv" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
exec uv run --active --project "$LTLM_REPO_ROOT" "$@"
