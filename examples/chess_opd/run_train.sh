#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "${PYTHON:-python}" -u -m examples.chess_opd.train_top3_opd \
  --student "${STUDENT_MODEL:-models/Qwen3-4B}" \
  --teacher "${TEACHER_MODEL:-models/Qwen3-8B}" \
  --train-data "${TRAIN_DATA:-data/opd/train.jsonl}" \
  --output-dir "${OUTPUT_DIR:-runs/opd}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-2048}" \
  --lr "${LR:-5e-7}" \
  --all-output-tokens --eval-every 0 --no-stability-stop "$@"
