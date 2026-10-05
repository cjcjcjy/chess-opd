# Chess OPD — Qwen3-4B student / Qwen3-8B teacher

学生从当前策略采样，冻结的教师接收同一局面和 Stockfish 私有信息，对学生的同一段 response prefix 计算完整词表分布。学生用 LoRA 优化 `KL(P_teacher || P_student)`；默认所有输出 token（包括 thinking 和 EOS）等权计算，prompt 不计 loss。每次更新后重新采样，不使用旧模型回答或拒绝采样。

这是同步 Transformers 实现，不依赖安装 verl、Ray、Megatron 或 vLLM。所有命令在仓库根目录执行，使用 `python -m examples.chess_opd...`。

## Prompt 与输出

教师只有一条 user message：棋局任务、Unicode 图形棋盘、行棋方/易位权/吃过路兵格、全部合法 UCI 走法、Stockfish 最佳三步的事实与 cp/mate、分数说明和回答模板。示例参考信息：

```text
f8d8: the rook moves from f8 to d8; cp=230
a7a5: the pawn moves from a7 to a5; cp=213
f6d5: the knight moves from f6 to d5; cp=212
```

事实从 FEN 用 python-chess 重新计算。输出模板中的前三走法与引擎排名对应：

```text
1. f8d8: describe the outcome of this move and judge the value of it.
2. a7a5: describe the outcome of this move and judge the value of it.
3. f6d5: describe the outcome of this move and judge the value of it.
Best Move: f8d8
```

学生使用相同回答结构，但模板以 `MOVE` 占位，不接收引擎分数、排名或答案。两者保留 `enable_thinking=True`。解析 `</think>` 后的四行，要求三步不重复，末行与第一步一致；未闭合 thinking 算格式失败。

教师不另生成训练答案，而是对学生实际采样的 token prefix 提供概率分布。教师已知最佳走法的准确率不能代表独立棋力。更换 prompt 不代表解释事实错误已经解决；格式和排名评估不能替代攻击、将杀、王车易位等陈述的事实审计。

## 安装

推荐 Linux、Python 3.11、支持 BF16 的 NVIDIA GPU。训练用两张卡，学生/教师各一张。验证机器为两张 48 GiB RTX 5880 Ada，更小显存未验证。模型约需 24 GB 磁盘，另需依赖、数据和 checkpoint 空间。

```bash
git clone https://github.com/cjcjcjy/chess-opd.git
cd chess-opd
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r examples/chess_opd/requirements.txt
sudo apt-get update
sudo apt-get install -y stockfish
hf download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
hf download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
```

CUDA wheel 应与驱动兼容。引擎默认 `/usr/games/stockfish`，其他位置用 `--stockfish /path/to/stockfish`。引擎版本/深度改变时，排名和分数可能变化。训练/评估只从本地加载模型，启动时校验两个 tokenizer 的词表与特殊 token 映射相同。

仓库默认私有，克隆者需要 GitHub 访问权限；仓库所有者可在 Settings 中授予访问或改为公开。

## 跑通示例

无需模型即可预览完整 prompt：

```bash
python -m examples.chess_opd.preview_top3_prompts \
  --fen 'r1b2rk1/ppp2ppp/2nbpn2/8/PP3P2/2PB1N2/6PP/RNB2R1K b - - 0 11' \
  --depth 12 --skip-tokenizer --output prompt_preview.md
```

去掉 `--skip-tokenizer` 可校验 tokenizer 并展示实际 chat prefix；`--student`、`--teacher` 指定本地模型。输出后缀 `.json` 可导出分析和 messages。

内置 12 个局面由固定随机合法走子生成，只用于链路检查，不能代表棋力或正式数据分布。生成 8 train / 2 dev / 2 test：

```bash
python -m examples.chess_opd.build_top3_engine_data \
  --train-source examples/chess_opd/sample_data/train.raw.jsonl \
  --test-source examples/chess_opd/sample_data/heldout.raw.jsonl \
  --dev-positions 2 --depth 4 --workers 2 --output-dir data/smoke
python -m unittest examples.chess_opd.test_contract -v
CUDA_VISIBLE_DEVICES=0,1 TRAIN_DATA=data/smoke/train.jsonl OUTPUT_DIR=runs/smoke \
  MAX_NEW_TOKENS=128 bash examples/chess_opd/run_train.sh --max-steps 1
```

128 token 用于快速链路检查，很可能截断 thinking，不验证回答质量。输出目录必须不存在，checkpoint 包含 LoRA adapter、optimizer 和随机状态。

## 正式数据

输入 JSONL 每行至少包含 `fen`，可选 `source_index`、`phase`：

```json
{"fen":"r1b2rk1/ppp2ppp/2nbpn2/8/PP3P2/2PB1N2/6PP/RNB2R1K b - - 0 11","source_index":0}
```

从有权使用的 PGN 导出最多 100,000 个局面，按对局划分 train/heldout：

```bash
python -m examples.chess_opd.export_pgn_positions \
  --pgn /path/to/games.pgn --output-dir data/raw \
  --max-positions 100000 --stride 4 --heldout-fraction 0.05
python -m examples.chess_opd.build_top3_engine_data \
  --train-source data/raw/train.raw.jsonl \
  --test-source data/raw/heldout.raw.jsonl \
  --dev-positions 128 --depth 14 --workers 16 --output-dir data/opd
```

实际数量取决于棋谱长度、去重和终局/不足三步过滤；检查 `metadata.json`。heldout 有效局面必须多于 `--dev-positions`。引擎 MultiPV 分析覆盖全部合法走法，可能耗时很长；仅前三进入 teacher prompt。train/dev/test 按 FEN 前四个字段去重，正式评估应使用独立来源或按对局留出的数据。

## 后台训练一个完整 epoch

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0,1 \
  STUDENT_MODEL=models/Qwen3-4B TEACHER_MODEL=models/Qwen3-8B \
  TRAIN_DATA=data/opd/train.jsonl OUTPUT_DIR=runs/opd \
  nohup bash examples/chess_opd/run_train.sh > logs/opd.log 2>&1 < /dev/null &
echo $! > logs/opd.pid
```

启动脚本默认 batch size 1、学习率 `5e-7`、输出上限 2048、LoRA rank 8，每 250 步及最后一步保存。可设置 `BATCH_SIZE`、`LR`、`MAX_NEW_TOKENS`。不指定 `--max-steps` 时每个训练局面使用一次，跑完整 epoch。两卡分别放学生/教师，不用 torchrun。默认关闭训练期间额外 GPU 的评估，训练后再评估。

`tail -n 5 logs/opd.log` 查看进度。`steps.jsonl` 包含 KL、梯度范数、生成长度、格式/合法计数、EOS 完成数和采样回答。KL 下降不能单独证明棋力进步；若 thinking 经常截断，需要增大 token 预算并检查显存。

实际链路验证中，单个合成局面的 student 和 teacher 在 2048 token 都出现了 thinking 截断。因此 2048 是可配置的初始上限，不是保证完整回答的预算。正式训练前应在真实 dev 集检查完成率，再确定 `MAX_NEW_TOKENS`；评估同步调整 `--max-tokens`，必要时提高 `--max-model-len`。不要用截断样例的格式准确率判断棋力。

恢复时保持原参数，替换实际 checkpoint 路径：

```bash
CUDA_VISIBLE_DEVICES=0,1 TRAIN_DATA=data/opd/train.jsonl OUTPUT_DIR=runs/opd \
  bash examples/chess_opd/run_train.sh --resume runs/opd/checkpoint_step_00250
```

恢复会校验 prompt 版本、数据哈希、关键训练参数。旧 prompt checkpoint 不可直接恢复。仅加载自己信任的 checkpoint；optimizer 使用 PyTorch pickle。

## 留出集评估

训练释放显存后，用相同测试集和解码设置比较原始模型与 adapter。默认 Transformers，单卡运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m examples.chess_opd.evaluate_top3_opd \
  --data data/opd/test.jsonl --model models/Qwen3-4B --mode student \
  --batch-size 1 --max-tokens 2048 --output-dir runs/eval_base
CUDA_VISIBLE_DEVICES=0 python -m examples.chess_opd.evaluate_top3_opd \
  --data data/opd/test.jsonl --model models/Qwen3-4B --mode student \
  --adapter runs/opd/checkpoint_step_XXXXX/adapter \
  --batch-size 1 --max-tokens 2048 --output-dir runs/eval_trained
CUDA_VISIBLE_DEVICES=0 python -m examples.chess_opd.evaluate_top3_opd \
  --data data/opd/dev.jsonl --model models/Qwen3-8B --mode teacher \
  --batch-size 1 --max-tokens 2048 --output-dir runs/eval_teacher
```

`XXXXX` 替换为实际步数。`summary.json` 给出格式、三步合法、top-1、top-3 集合、允许 30 cp 差异的 top-3、截断计数及 cp regret；计数除以 `positions` 得到比例。`responses.jsonl` 用于审计解释。这是局面评估，不是 depth 0/1/2 对弈胜率。若另装了兼容 vLLM，可加 `--backend vllm`。

## 入口

- `preview_top3_prompts.py` / `board.py`：prompt 和图形棋盘。
- `export_pgn_positions.py`：PGN → JSONL。
- `build_top3_engine_data.py`：Stockfish 分析、去重和划分。
- `train_top3_opd.py` / `run_train.sh`：训练和恢复。
- `evaluate_top3_opd.py`：基线/adapter/teacher 评估。
- `test_contract.py`：CPU 检查。

代码沿用 Apache-2.0；模型、棋谱和 Stockfish 遵守各自许可证。不包含模型权重、正式私有数据或旧训练输出。
