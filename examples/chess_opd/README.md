# Chess OPD — official verl trainer

当前训练入口使用 [verl 官方 On-Policy Distillation 示例](https://github.com/verl-project/verl/tree/main/examples/on_policy_distillation_trainer)，具体调用 `run_qwen3_8b_fsdp.sh` → `verl.trainer.main_ppo`。不再使用手写 `train_top3_opd.py`，该文件已从当前分支删除；历史实现仍可在 Git 历史中找到。

学生为 Qwen3-4B，`enable_thinking=False`；教师为 Qwen3-8B，`enable_thinking=True`。棋类 prompt 见 [完整示例](examples/chess_opd/prompt_example.md)：三条走法解释，最后一行 `Best Move: MOVE`。学生只看棋盘和合法走法；教师额外看 Stockfish 最佳三步及事实、cp/mate 分数。已移除 `Use three distinct legal UCI moves. Best Move must match the first move.`。

Teacher 现在先基于自己的 prompt 生成 `<think>...</think>`，在 `</think>` 处停止，再把学生原样采样的回答接在后面计算概率。teacher 的思考只作为其私有上下文，不作为 student 的训练目标。流程如下：

```text
student: student prompt → 学生回答 y
teacher: teacher prompt → <think>教师思考</think> → 停止生成
teacher 打分输入: teacher prompt + <think>教师思考</think> + 两个换行 + y
训练位置: 仅 y 的 token（包含生成的 EOS）
```

教师思考生成时看不到学生回答。打分时，对每个学生 token 使用 `P_teacher(y_t | teacher prompt, teacher thinking, y_<t)`；不拿教师自己生成的最终答案对齐学生答案。student 关闭 thinking 的模板前缀不包含在 response loss 中。

## 官方实现与适配范围

上游固定到 [`8718ca30a3f002f93b7c4fd99b9b2506718681bc`](https://github.com/verl-project/verl/tree/8718ca30a3f002f93b7c4fd99b9b2506718681bc)，记录在 `VERL_REVISION`。不使用本机旧 `/home/.../verl` checkout。

- 官方负责 Ray 资源池、vLLM 学生采样和教师打分、FSDP 训练、蒸馏损失、optimizer 和 checkpoint。
- 本仓库负责棋类 prompt、数据转换、验证指标和启动参数。
- 官方默认让教师读取学生同一个 prompt。[输入适配补丁](patches/teacher_prompt.patch) 加入私有 prompt；随后应用 [教师思考补丁](patches/teacher_thinking.patch)，通过官方 teacher client 先生成思考，再调用原有概率接口。返回概率去掉 teacher prompt 和 thinking，映射回学生序列坐标。学生响应 token 不解码重编码、不增删、不位移；prompt 区域由官方 response mask 排除。
- [文本输入补丁](patches/teacher_prompt_text.patch) 让数据直接保存英文 `teacher_prompt`，训练时使用对应教师模型的 tokenizer 和 `enable_thinking=True` 编码；tokenizer 按教师缓存。
- 三个补丁只适配教师上下文和请求，不重写训练循环或损失。缺少私有 prompt、思考为空/被截断/未闭合、上下文预算不足时，该样本报错且不调用打分。官方 rollout 层处理失败状态；请检查 worker 日志，不能把失败样本当成完成监督。

默认遵循官方示例的 **`k1 + use_policy_gradient=True`**，不是历史手写版的完整词表 forward KL。所有学生采样 response token（包括生成的 EOS）进入官方 response mask，不做走法/解释差异加权。官方示例的损失裁剪设置保留。默认 `use_task_rewards=False`，棋类 reward 不进入训练目标。

如需官方的 top-k forward KL，可设置 `DISTILLATION_LOSS_MODE=forward_kl_topk USE_POLICY_GRADIENT=False DISTILLATION_TOPK=64`。它是 top-k 近似，不是完整词表 KL。

## 已准备的 100K 数据

仓库直接包含 [datasets/chess_opd_100k](datasets/chess_opd_100k)，不使用 Git LFS 或外部下载链接。克隆仓库即可得到训练数据，无需运行 Stockfish、重新选样或生成 prompt。

| 文件 | 用途 | 条数 |
|---|---|---:|
| `train.parquet` | 训练 | 100,000 |
| `dev.parquet` | 验证 | 128 |
| `test.parquet` | 测试 | 825 |

训练集为开局 28,000、中局 44,000、残局 28,000，来自现有 Stockfish 11 depth-12 分析；验证/测试集沿用固定 depth-14 分析。三个 split 按 FEN 前四个字段去重并检查无交集；不声称来自完全独立的对局。数据不含旧模型回答。来源、抽样种子、SHA256 和长度统计见该目录的说明和 JSON 清单。

默认启动脚本已指向这份数据。batch size 4 时，一个完整 epoch 是 **25,000 个训练 batch**。

## 环境

训练入口默认使用当前环境的 `python3`，直接进入官方 trainer，不调用 `uv` 或运行全量数据预检查。需要锁定环境时设置 `VERL_USE_UV=1`；需要启动前数据检查时设置 `RUN_PREFLIGHT=1`。`--dry-run` 仍会检查数据。跳过环境管理不代表当前依赖已经兼容。

**不要复用历史手写版的 torch 2.8 / vLLM 0.11 环境运行本版。** 固定上游的 `uv.lock` 使用 torch 2.13.0、vLLM 0.29.0、Transformers 5.12.1、CUDA 13.0。需要支持 CUDA 13 的驱动（通常 Linux 580+）和 Python 3.10–3.12；以官方依赖与 NVIDIA 兼容性要求为准。数据准备可以在单独的 CPU 环境完成。

默认单机两张可见 GPU：一张给学生 actor/rollout，一张给教师。学生使用官方 FSDP + LoRA，默认 `LORA_RANK=8`、`LORA_ALPHA=16`、`LORA_TARGET_MODULES=all-linear`；基座权重冻结，只训练 adapter，教师保持冻结推理。可设置 `LORA_RANK=0` 使用全参数训练。LoRA 减少学生的梯度和 optimizer 显存，不降低独立教师服务的打分显存；日志中的教师 OOM 仍需通过教师推理参数处理。

```bash
git clone https://github.com/cjcjcjy/chess-opd.git
cd chess-opd
# 安装 uv（若已有则跳过）
python -m pip install uv
# 获取固定官方版本，应用输入补丁，并安装官方锁定环境
bash scripts/setup_verl.sh
```

官方 checkout 默认位于 `vendor/verl`，不会提交到本仓库。脚本拒绝覆盖已有非 Git 目录或切换不同版本的 checkout，重复执行不会重复打补丁。已有旧版补丁的 checkout 会自动补上教师思考和文本输入改动。更新本仓库后运行 `bash scripts/setup_verl.sh --checkout-only` 即可更新补丁而不重装环境。`VERL_DIR` 可指定其他独立目录。

仓库私有，对方需要 GitHub 访问权限。

## 数据和模型准备

**只需跑训练的人不用准备数据。** 官方环境安装完成后，下载模型或通过环境变量指定已有模型路径：

```bash
vendor/verl/.venv/bin/hf download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
vendor/verl/.venv/bin/hf download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
PREFLIGHT_PYTHON=vendor/verl/.venv/bin/python bash examples/chess_opd/run_train.sh --dry-run
```

下面仅用于需要重新生成其他数据的情况，在 chess-opd 根目录执行：

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
  --input-dir data/engine_smoke --output-dir data/verl_smoke
```

第二步生成官方 RLHFDataset 可读取的 train/dev/test parquet：

| 字段 | 内容 |
|---|---|
| `prompt` | 仅学生 user message |
| `data_source` | `chess_opd` |
| `extra_info.teacher_prompt` | 完整英文教师 user prompt，含棋盘、引擎信息和真实换行 |
| `extra_info.teacher_prompt_length` | 教师模板编码后的 token 数，仅用于检查长度 |
| `extra_info.fen` / `index` | 棋盘与样本标识 |
| `reward_model.ground_truth` | 验证用合法走法/前三名，不进入学生 prompt |

转换时校验学生/教师词表和特殊 token 映射一致，检查全部合法走法覆盖、跨 split 重复和长度。保存 tokenizer/chat template 指纹；启动时校验文件 SHA256、行数和指纹。student 编码为 thinking=False；teacher 的自然语言文本在训练时以 thinking=True 编码。数据不保存教师 token ID 数组或生成的思考。旧版存储 token ID 的 parquet 会被拒绝，仓库中的 100K 数据已转换好。

查看 [一条真实数据的完整 prompt](datasets/chess_opd_100k/sample.md)，或直接打印 Parquet 中的教师文本（`print` 会显示实际换行）：

```bash
python - <<'PY'
import pyarrow.parquet as pq
row = next(pq.ParquetFile('datasets/chess_opd_100k/train.parquet').iter_batches(batch_size=1)).to_pylist()[0]
print(row['extra_info']['teacher_prompt'])
PY
```

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

`MAX_PROMPT_LENGTH=2048`、`MAX_TEACHER_PROMPT_LENGTH=3072`、`MAX_RESPONSE_LENGTH=4096` 分别控制两个 prompt 和学生回答预算。新增 `TEACHER_THINK_MAX_TOKENS=8192` 控制教师思考预算，采样使用 temperature 0.6、top-p 0.95、top-k 20。教师服务上下文上限自动设为 `teacher prompt + teacher thinking + student response + 2`（一个换行分隔 token 和一个打分接口生成 token）。当前默认合计 15,362 token。

教师必须实际生成闭合的 `</think>` 才会打分；不会给截断思考强行补标签，也不自动反复采样。若 worker 日志出现 `Teacher did not generate </think>`，先检查生成，必要时增加 `TEACHER_THINK_MAX_TOKENS`。成功样本记录 `teacher_thinking_tokens` 和 `teacher_scoring_prompt_tokens`，并有教师完成长度日志。思考越长，教师生成时间和上下文显存开销越大。

100K parquet 内的 prompt 无需更改，教师思考在训练时生成，不预存在数据中。改 prompt/tokenizer 才需要重新转换 parquet。

## 训练

可选：先做一次官方链路检查（使用独立输出目录，不用于判断学习效果）：

```bash
CUDA_VISIBLE_DEVICES=0,1 OUTPUT_DIR=runs/official_smoke bash examples/chess_opd/run_train.sh \
  trainer.total_training_steps=1 trainer.save_freq=1
```

对方可以直接后台跑已准备的 100K 数据、完整一个 epoch：

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0,1 \
  STUDENT_MODEL=models/Qwen3-4B TEACHER_MODEL=models/Qwen3-8B \
  LORA_RANK=8 LORA_ALPHA=16 \
  OUTPUT_DIR=runs/official_opd_100k_teacher_reasoning_lora8 \
  nohup bash examples/chess_opd/run_train.sh > logs/official_opd.log 2>&1 < /dev/null &
echo $! > logs/official_opd.pid
```

模型目录可设置 `STUDENT_MODEL` / `TEACHER_MODEL`。脚本默认使用当前环境的 `python3`；设置 `VERL_USE_UV=1` 时，通过官方 `uv run --frozen --all-packages --extra vllm --extra fsdp` 启动 driver 和 Ray worker。`RUN_PREFLIGHT=1` 启用全量数据预检查，`--dry-run` 始终检查数据。

### LoRA 环境检查

用训练所用的 Python 运行下面的 CPU 检查。它会在一个小型 Qwen3 上执行真实的 PEFT LoRA 初始化、前向和反向计算，并检查仅 adapter 获得梯度，不下载模型或占用 GPU：

```bash
python -m examples.chess_opd.check_lora_environment
```

若在 `peft → awq` 导入过程中出现 `cannot import name 'PytorchGELUTanh'`，原因是可选的 `autoawq` 与 Transformers 不兼容。此项目使用的 Qwen3-4B/8B 原始权重不需要 AWQ；在训练环境中移除该可选包，再运行上述检查：

```bash
python -m pip uninstall -y autoawq
python -m examples.chess_opd.check_lora_environment
```

这解决 LoRA 初始化依赖错误，不代表已验证完整 Ray/vLLM/FSDP 训练。教师推理显存独立于学生 LoRA：当前教师 `max_num_batched_tokens=4096`，`TEACHER_GPU_MEM_UTIL` 默认为 `0.8`，实际运行仍需验证教师打分是否有足够显存。

官方 checkpoint 位于 `OUTPUT_DIR/global_step_N`。相同配置、数据和输出目录下再次运行时，官方 `trainer.resume_mode=auto` 恢复；显式路径可加 `trainer.resume_mode=resume_path trainer.resume_from_path=/absolute/path/global_step_N`。默认输出目录为 `runs/official_opd_100k_teacher_reasoning_lora8`，避免自动恢复旧的全参数训练；切换 rank、alpha 或 target modules 时也应使用新目录。不要加载历史手写版 checkpoint。

默认同时记录 console 和 **Weights & Biases**，通过官方 verl logger 上传训练配置和每步指标。首次在线运行前，在训练环境执行 `wandb login`。默认项目为 `chess_opd`，运行名包含模型和 LoRA rank；可设置 `PROJECT_NAME` / `EXPERIMENT_NAME`，也支持 `WANDB_PROJECT` / `WANDB_NAME`（前两者优先）。`WANDB_ENTITY` 可指定账号或团队。

```bash
wandb login
PROJECT_NAME=chess_opd EXPERIMENT_NAME=qwen3_4b_lora8_opd \
  bash examples/chess_opd/run_train.sh
```

后台训练仍使用上面的 `nohup` 命令，W&B 会自动启用。本地 W&B 文件默认写入 `logs/wandb/`（`WANDB_DIR` 可修改父目录）。不联网时设置 `WANDB_MODE=offline`，之后可用 `wandb sync logs/wandb/offline-run-*` 上传；仅保留终端日志可在命令末尾加 `'trainer.logger=[console]'`。

断点恢复训练默认创建新的 W&B run；若需要接续原在线 run，显式设置相同的 `WANDB_RUN_ID` 和 `WANDB_RESUME=allow`，并恢复对应 checkpoint。

`SAVE_FREQ=250`，`TEST_FREQ=-1` 关闭周期验证；设置 `TEST_FREQ=250` 可启用官方验证及其 W&B 指标。关注官方 distillation loss、学习率、梯度范数、生成长度、截断、耗时和留出集指标，不能单凭 KL 下降断言学生进步。W&B 记录以官方 trainer 实际输出的指标为准。

## 导出与棋类评估

默认全参数 FSDP checkpoint 用官方 merger 导出（替换 N）：

```bash
cd vendor/verl
uv run --frozen --all-packages --extra vllm --extra fsdp python -m verl.model_merger merge \
  --backend fsdp --local_dir ../../runs/official_opd_100k_teacher_reasoning/global_step_N/actor \
  --target_dir ../../models/chess-opd-4b
cd ../..
uv pip install --python vendor/verl/.venv/bin/python python-chess==1.999
CUDA_VISIBLE_DEVICES=0 vendor/verl/.venv/bin/python -m examples.chess_opd.evaluate_top3_opd \
  --data datasets/chess_opd_100k/test.engine.jsonl --model models/chess-opd-4b --mode student \
  --batch-size 1 --max-tokens 4096 --max-model-len 8192 --output-dir runs/eval_trained
```

用原始 `models/Qwen3-4B` 重复同一评估得到基线；用 `--mode teacher --model models/Qwen3-8B` 检查教师。若使用官方 LoRA 配置，请遵循上游对应的 adapter 导出流程，不套用历史手写版 checkpoint 路径。

棋类验证统计格式、三步合法性、top-1/top-3、cp regret 和截断，不自动证明解释正确，也不是 depth 0/1/2 对弈胜率。教师拿到引擎前三名后的准确率不能代表独立棋力。

## 验证范围

见 [VALIDATION.md](VALIDATION.md)。此前手写版的 GPU 冒烟测试不代表官方训练已经跑通。本次迁移验证数据、补丁对齐与配置，不声称在当前旧驱动/依赖环境上完成了官方 GPU 训练。

代码 Apache-2.0；模型、Stockfish 和棋谱遵守各自许可证。仓库包含上述已准备的棋局数据，不包含模型权重或训练 checkpoint。
