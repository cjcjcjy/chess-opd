# Chess OPD — official verl trainer

当前训练入口使用 [verl 官方 On-Policy Distillation 示例](https://github.com/verl-project/verl/tree/main/examples/on_policy_distillation_trainer)，具体调用 `run_qwen3_8b_fsdp.sh` → `verl.trainer.main_ppo`。不再使用手写 `train_top3_opd.py`，该文件已从当前分支删除；历史实现仍可在 Git 历史中找到。

学生为 Qwen3-4B，教师为 Qwen3-8B，均开启 thinking。棋类 prompt 沿用 [完整示例](examples/chess_opd/prompt_example.md)：三条走法解释，最后一行 `Best Move: MOVE`。学生只看棋盘和合法走法；教师额外看 Stockfish 最佳三步及事实、cp/mate 分数。

## 官方实现与适配范围

上游固定到 [`8718ca30a3f002f93b7c4fd99b9b2506718681bc`](https://github.com/verl-project/verl/tree/8718ca30a3f002f93b7c4fd99b9b2506718681bc)，记录在 `VERL_REVISION`。不使用本机旧 `/home/.../verl` checkout。

- 官方负责 Ray 资源池、vLLM 学生采样和教师打分、FSDP 训练、蒸馏损失、optimizer 和 checkpoint。
- 本仓库负责棋类 prompt、数据转换、验证指标和启动参数。
- 官方默认让教师读取学生同一个 prompt。为保留教师私有 Stockfish 信息，附带 [输入适配补丁](patches/teacher_prompt.patch)，仅修改 `_compute_teacher_logprobs`：教师读取 `extra_info.teacher_prompt_ids + 原样学生 response_ids`；返回概率映射回学生序列坐标。响应 token 不解码重编码、不增删、不位移。prompt 区域填占位值并由官方 response mask 排除。
- 补丁不重写训练循环或损失，也不让教师生成固定答案。缺少棋类教师私有输入时直接报错，避免悄悄退回相同 prompt。

默认遵循官方示例的 **`k1 + use_policy_gradient=True`**，不是历史手写版的完整词表 forward KL。所有采样 response token（包括 thinking 和生成的 EOS）进入官方 response mask，不做走法/解释差异加权。官方示例的损失裁剪设置保留。默认 `use_task_rewards=False`，棋类 reward 只用于验证。

如需官方的 top-k forward KL，可设置 `DISTILLATION_LOSS_MODE=forward_kl_topk USE_POLICY_GRADIENT=False DISTILLATION_TOPK=64`。它是 top-k 近似，不是完整词表 KL。

## 环境

**不要复用历史手写版的 torch 2.8 / vLLM 0.11 环境运行本版。** 固定上游的 `uv.lock` 使用 torch 2.13.0、vLLM 0.29.0、Transformers 5.12.1、CUDA 13.0。需要支持 CUDA 13 的驱动（通常 Linux 580+）和 Python 3.10–3.12；以官方依赖与 NVIDIA 兼容性要求为准。数据准备可以在单独的 CPU 环境完成。

默认单机两张可见 GPU：一张给学生 actor/rollout，一张给教师。原始示例默认全参数 FSDP，启动脚本也默认 `LORA_RANK=0`，不是之前手写版的 LoRA 训练。支持用 `LORA_RANK=8` 选择官方 LoRA 配置。新版两卡训练的实际显存峰值尚未实测；两张 48 GiB 是目标环境，不是已经验证的显存保证。

```bash
git clone https://github.com/cjcjcjy/chess-opd.git
cd chess-opd
# 安装 uv（若已有则跳过）
python -m pip install uv
# 获取固定官方版本，应用输入补丁，并安装官方锁定环境
bash scripts/setup_verl.sh
```

官方 checkout 默认位于 `vendor/verl`，不会提交到本仓库。脚本拒绝覆盖已有非 Git 目录或切换不同版本的 checkout，重复执行不会重复打补丁。只获取代码不安装环境：`bash scripts/setup_verl.sh --checkout-only`。`VERL_DIR` 可指定其他独立目录。

仓库私有，对方需要 GitHub 访问权限。

## 数据和模型准备

以下命令在 chess-opd 根目录执行：

```bash
uv venv --python 3.11 .venv-data
uv pip install --python .venv-data/bin/python -r examples/chess_opd/requirements.txt
source .venv-data/bin/activate
sudo apt-get update
sudo apt-get install -y stockfish
hf download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
hf download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
```

先用内置的合成局面生成示例数据（8 train / 2 dev / 2 test，仅用于链路检查）：

```bash
python -m examples.chess_opd.build_top3_engine_data \
  --train-source examples/chess_opd/sample_data/train.raw.jsonl \
  --test-source examples/chess_opd/sample_data/heldout.raw.jsonl \
  --dev-positions 2 --depth 4 --workers 2 --output-dir data/engine_smoke
python -m examples.chess_opd.prepare_verl_data \
  --input-dir data/engine_smoke --output-dir data/verl
```

第二步生成官方 RLHFDataset 可读取的 train/dev/test parquet：

| 字段 | 内容 |
|---|---|
| `prompt` | 仅学生 user message |
| `data_source` | `chess_opd` |
| `extra_info.teacher_prompt_ids` | 用教师 chat template 编码的私有 prompt |
| `extra_info.fen` / `index` | 棋盘与样本标识 |
| `reward_model.ground_truth` | 验证用合法走法/前三名，不进入学生 prompt |

转换时校验学生/教师词表和特殊 token 映射一致，检查全部合法走法覆盖、跨 split 重复和长度。保存 tokenizer/chat template 指纹；启动时不一致会要求重新生成数据。`teacher_prompt_ids` 包含 generation prefix，thinking 开关为 True。

正式数据可从有权使用的 PGN 按对局划分导出十万级局面，再标注、转换：

```bash
python -m examples.chess_opd.export_pgn_positions \
  --pgn /path/to/games.pgn --output-dir data/raw \
  --max-positions 100000 --stride 4 --heldout-fraction 0.05
python -m examples.chess_opd.build_top3_engine_data \
  --train-source data/raw/train.raw.jsonl --test-source data/raw/heldout.raw.jsonl \
  --dev-positions 128 --depth 14 --workers 16 --output-dir data/engine
python -m examples.chess_opd.prepare_verl_data \
  --input-dir data/engine --output-dir data/verl_full
```

也可自行提供每行含 `fen` 的原始 JSONL。Stockfish 默认 `/usr/games/stockfish`，其他位置用 `--stockfish`。官方训练使用 parquet；JSONL 引擎数据仍保留给独立棋类评估。输出目录应使用新目录。

## 检查启动配置

```bash
bash examples/chess_opd/run_train.sh --dry-run
```

此命令检查数据和 tokenizer，打印调用官方示例的命令，不启动 Ray、不加载 GPU 模型、不安装训练环境。数据环境 Python 需在 PATH 上，也可设置 `PREFLIGHT_PYTHON`。

默认 `TRAIN_BATCH_SIZE=4`、`PPO_MINI_BATCH_SIZE=4`、`ACTOR_LR=5e-7`，一个 epoch。官方 dataloader 会丢弃不满 batch 的尾部，本仓库启动检查因此要求训练行数能被 batch size 整除，避免“完整 epoch”遗漏尾部；必要时设置 `TRAIN_BATCH_SIZE=1 PPO_MINI_BATCH_SIZE=1`。

`MAX_PROMPT_LENGTH=2048`、`MAX_TEACHER_PROMPT_LENGTH=3072`、`MAX_RESPONSE_LENGTH=4096` 分别控制两个 prompt 和共同回答预算。4096 不保证 thinking 一定完成，应先在真实 dev 集检查截断率。改 prompt/tokenizer 后重新转换 parquet。

## 训练

先做一次官方链路检查（不用于判断学习效果）：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash examples/chess_opd/run_train.sh \
  trainer.total_training_steps=1 trainer.save_freq=1
```

后台跑正式数据的完整 epoch：

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0,1 \
  TRAIN_DATA=data/verl_full/train.parquet VAL_DATA=data/verl_full/dev.parquet \
  OUTPUT_DIR=runs/official_opd \
  nohup bash examples/chess_opd/run_train.sh > logs/official_opd.log 2>&1 < /dev/null &
echo $! > logs/official_opd.pid
```

模型目录可设置 `STUDENT_MODEL` / `TEACHER_MODEL`。脚本默认通过官方 `uv run --frozen --all-packages --extra vllm --extra fsdp` 启动 driver 和 Ray worker，避免跑到本机旧 verl。仅在手工准备了兼容的官方环境时使用 `VERL_USE_UV=0`。

官方 checkpoint 位于 `OUTPUT_DIR/global_step_N`。相同配置、数据和输出目录下再次运行时，官方 `trainer.resume_mode=auto` 恢复；显式路径可加 `trainer.resume_mode=resume_path trainer.resume_from_path=/absolute/path/global_step_N`。不要加载历史手写版 checkpoint。

默认只记录 console，`SAVE_FREQ=250`，`TEST_FREQ=-1` 关闭周期验证；设置 `TEST_FREQ=250` 可启用官方验证。关注官方 distillation loss、生成长度、截断和留出集指标，不能单凭 KL 下降断言学生进步。

## 导出与棋类评估

默认全参数 FSDP checkpoint 用官方 merger 导出（替换 N）：

```bash
cd vendor/verl
uv run --frozen --all-packages --extra vllm --extra fsdp python -m verl.model_merger merge \
  --backend fsdp --local_dir ../../runs/official_opd/global_step_N/actor \
  --target_dir ../../models/chess-opd-4b
cd ../..
uv pip install --python vendor/verl/.venv/bin/python python-chess==1.999
CUDA_VISIBLE_DEVICES=0 vendor/verl/.venv/bin/python -m examples.chess_opd.evaluate_top3_opd \
  --data data/engine/test.jsonl --model models/chess-opd-4b --mode student \
  --batch-size 1 --max-tokens 4096 --max-model-len 8192 --output-dir runs/eval_trained
```

用原始 `models/Qwen3-4B` 重复同一评估得到基线；用 `--mode teacher --model models/Qwen3-8B` 检查教师。若使用官方 LoRA 配置，请遵循上游对应的 adapter 导出流程，不套用历史手写版 checkpoint 路径。

棋类验证统计格式、三步合法性、top-1/top-3、cp regret 和截断，不自动证明解释正确，也不是 depth 0/1/2 对弈胜率。教师拿到引擎前三名后的准确率不能代表独立棋力。

## 验证范围

见 [VALIDATION.md](VALIDATION.md)。此前手写版的 GPU 冒烟测试不代表官方训练已经跑通。本次迁移验证数据、补丁对齐与配置，不声称在当前旧驱动/依赖环境上完成了官方 GPU 训练。

代码 Apache-2.0；模型、Stockfish 和棋谱遵守各自许可证。仓库不含权重或正式私有数据。
