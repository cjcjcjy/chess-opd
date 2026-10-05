#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VERL_DIR=${VERL_DIR:-$ROOT/vendor/verl}
REVISION=$(cat "$ROOT/VERL_REVISION")
if [[ ! -d "$VERL_DIR/.git" ]]; then
  if [[ -e "$VERL_DIR" ]]; then
    echo "Refusing to overwrite existing non-git directory: $VERL_DIR" >&2
    exit 1
  fi
  mkdir -p "$VERL_DIR"
  git -C "$VERL_DIR" init
  git -C "$VERL_DIR" remote add origin https://github.com/verl-project/verl.git
  git -C "$VERL_DIR" fetch --depth 1 origin "$REVISION"
  git -C "$VERL_DIR" checkout --detach FETCH_HEAD
fi
if [[ $(git -C "$VERL_DIR" rev-parse HEAD) != "$REVISION" ]]; then
  echo "VERL_DIR must point to the revision in VERL_REVISION; use a fresh directory." >&2
  exit 1
fi
CONTEXT_PATCH="$ROOT/patches/teacher_prompt.patch"
THINKING_PATCH="$ROOT/patches/teacher_thinking.patch"
if git -C "$VERL_DIR" apply --reverse --check "$THINKING_PATCH" 2>/dev/null; then
  echo "Teacher-context and reasoning patches already applied."
else
  if ! git -C "$VERL_DIR" apply --reverse --check "$CONTEXT_PATCH" 2>/dev/null; then
    git -C "$VERL_DIR" apply --check "$CONTEXT_PATCH"
    git -C "$VERL_DIR" apply "$CONTEXT_PATCH"
  fi
  git -C "$VERL_DIR" apply --check "$THINKING_PATCH"
  git -C "$VERL_DIR" apply "$THINKING_PATCH"
fi
if [[ ${1:-} == --checkout-only ]]; then
  exit 0
fi
if [[ $# != 0 ]]; then
  echo "Usage: $0 [--checkout-only]" >&2
  exit 1
fi
cd "$VERL_DIR"
uv sync --frozen --all-packages --extra vllm --extra fsdp
