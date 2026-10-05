#!/usr/bin/env bash
# Delegate training and all distillation losses to the pinned official example.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VERL_DIR=${VERL_DIR:-$ROOT/vendor/verl}
export STUDENT_MODEL=${STUDENT_MODEL:-$ROOT/models/Qwen3-4B}
export TEACHER_MODEL=${TEACHER_MODEL:-$ROOT/models/Qwen3-8B}
TRAIN_DATA=${TRAIN_DATA:-$ROOT/datasets/chess_opd_100k/train.parquet}
VAL_DATA=${VAL_DATA:-$ROOT/datasets/chess_opd_100k/dev.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/runs/official_opd_100k_teacher_reasoning}
PREFLIGHT_PYTHON=${PREFLIGHT_PYTHON:-python3}
if [[ $PREFLIGHT_PYTHON == */* ]]; then
  PREFLIGHT_PYTHON=$(realpath "$PREFLIGHT_PYTHON")
fi
for name in STUDENT_MODEL TEACHER_MODEL TRAIN_DATA VAL_DATA OUTPUT_DIR VERL_DIR; do
  # Resolve before changing into the official checkout.
  printf -v "$name" '%s' "$(realpath -m "${!name}")"
done
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NNODES=1 NGPUS_PER_NODE=1 TEACHER_WORLD_SIZE=1 ROLLOUT_TP=1 TEACHER_TP=1
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
MAX_TEACHER_PROMPT_LENGTH=${MAX_TEACHER_PROMPT_LENGTH:-3072}
TEACHER_THINK_MAX_TOKENS=${TEACHER_THINK_MAX_TOKENS:-8192}
if [[ ! $TEACHER_THINK_MAX_TOKENS =~ ^[1-9][0-9]*$ ]]; then
  echo "TEACHER_THINK_MAX_TOKENS must be a positive integer" >&2
  exit 1
fi
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
export ACTOR_LR=${ACTOR_LR:-5e-7}
export TOTAL_EPOCHS=1
export SAVE_FREQ=${SAVE_FREQ:-250} TEST_FREQ=${TEST_FREQ:--1}
export DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k1}
export USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}
export VERL_USE_UV=${VERL_USE_UV:-1}
export PROJECT_NAME=chess_opd EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_nothink_from_8b_think_100k}
if [[ $(git -C "$VERL_DIR" rev-parse HEAD) != "$(cat "$ROOT/VERL_REVISION")" ]]; then
  echo "Run scripts/setup_verl.sh; official verl revision mismatch." >&2
  exit 1
fi
git -C "$VERL_DIR" apply --reverse --check "$ROOT/patches/teacher_thinking.patch"
PREFLIGHT=(python3)
if [[ $VERL_USE_UV != 0 ]]; then
  PREFLIGHT=(uv run --frozen --all-packages --extra vllm --extra fsdp python3)
fi
cd "$VERL_DIR"
# This is the official script, with chess data/configuration overrides only.
LAUNCH=(bash examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh
  "data.train_files=$TRAIN_DATA"
  "data.val_files=$VAL_DATA"
  data.shuffle=True
  data.filter_overlong_prompts=False
  data.dataloader_num_workers=0
  '+data.apply_chat_template_kwargs.enable_thinking=False'
  "actor_rollout_ref.model.lora_rank=${LORA_RANK:-0}"
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.rollout.agent.num_workers=1
  "+chess_opd.teacher_think_max_tokens=$TEACHER_THINK_MAX_TOKENS"
  "distillation.teacher_models.teacher_model.inference.max_model_len=$((MAX_TEACHER_PROMPT_LENGTH + TEACHER_THINK_MAX_TOKENS + MAX_RESPONSE_LENGTH + 2))"
  "reward.custom_reward_function.path=$ROOT/examples/chess_opd/reward.py"
  reward.custom_reward_function.name=compute_score
  'trainer.logger=[console]'
  "trainer.default_local_dir=$OUTPUT_DIR"
)
if [[ ${1:-} == --dry-run ]]; then
  shift
  # No Ray, model loading or uv environment installation during a dry run.
  "${PREFLIGHT_PYTHON:-python3}" "$ROOT/examples/chess_opd/preflight_verl.py" \
    --train "$TRAIN_DATA" --val "$VAL_DATA" --student "$STUDENT_MODEL" --teacher "$TEACHER_MODEL" \
    --batch-size "$TRAIN_BATCH_SIZE" --mini-batch-size "$PPO_MINI_BATCH_SIZE" \
    --max-prompt "$MAX_PROMPT_LENGTH" --max-teacher-prompt "$MAX_TEACHER_PROMPT_LENGTH"
  printf '%q ' "${LAUNCH[@]}" "$@"
  printf '\n'
  exit 0
fi
"${PREFLIGHT[@]}" "$ROOT/examples/chess_opd/preflight_verl.py" \
  --train "$TRAIN_DATA" --val "$VAL_DATA" --student "$STUDENT_MODEL" --teacher "$TEACHER_MODEL" \
  --batch-size "$TRAIN_BATCH_SIZE" --mini-batch-size "$PPO_MINI_BATCH_SIZE" \
  --max-prompt "$MAX_PROMPT_LENGTH" --max-teacher-prompt "$MAX_TEACHER_PROMPT_LENGTH"
exec "${LAUNCH[@]}" "$@"
