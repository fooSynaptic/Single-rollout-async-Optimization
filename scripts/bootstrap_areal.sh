#!/usr/bin/env bash
# Clone the tested AReaL revision and apply this repository's SAO integration.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AREAL_REPO="${AREAL_REPO:-https://github.com/inclusionAI/AReaL.git}"
AREAL_COMMIT="${AREAL_COMMIT:-3cf0dfbd2b0fbeabd6977184980e189d1567747a}"
AREAL_ROOT="${AREAL_ROOT:-$ROOT/vendor/AReaL}"
PATCH="$ROOT/patches/areal-sao.patch"
INSTALL="${INSTALL:-0}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"

if [[ ! -f "$PATCH" ]]; then
  echo "missing patch: $PATCH" >&2
  exit 1
fi

if [[ ! -d "$AREAL_ROOT/.git" ]]; then
  mkdir -p "$(dirname "$AREAL_ROOT")"
  git clone "$AREAL_REPO" "$AREAL_ROOT"
fi

git -C "$AREAL_ROOT" fetch origin "$AREAL_COMMIT"
git -C "$AREAL_ROOT" checkout --detach "$AREAL_COMMIT"

if git -C "$AREAL_ROOT" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo "SAO patch already applied: $AREAL_ROOT"
elif git -C "$AREAL_ROOT" apply --check "$PATCH"; then
  git -C "$AREAL_ROOT" apply "$PATCH"
  echo "applied SAO patch: $PATCH"
else
  echo "patch does not apply cleanly to AReaL $AREAL_COMMIT" >&2
  exit 1
fi

if [[ "$INSTALL" == "1" ]]; then
  command -v uv >/dev/null 2>&1 || python3 -m pip install uv
  cd "$AREAL_ROOT"
  if [[ "$INFERENCE_BACKEND" == "vllm" ]]; then
    cp pyproject.vllm.toml pyproject.toml
    cp uv.vllm.lock uv.lock
  elif [[ "$INFERENCE_BACKEND" != "sglang" ]]; then
    echo "INFERENCE_BACKEND must be vllm or sglang" >&2
    exit 2
  fi
  uv sync --extra cuda
fi

cat <<EOF
AReaL ready at: $AREAL_ROOT
Revision: $AREAL_COMMIT
INSTALL=$INSTALL (1 = uv sync cuda env under $AREAL_ROOT/.venv)

Next:
  export AREAL_ROOT="$AREAL_ROOT"
  export PYTHONPATH="$AREAL_ROOT:$ROOT"
$([ "$INSTALL" = "1" ] || echo "  # for training: INSTALL=1 INFERENCE_BACKEND=vllm $0")
  # mirror override: AREAL_REPO=<git-url> $0
EOF
