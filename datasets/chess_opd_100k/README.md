# Ready-to-train chess OPD data

This directory ships actual Parquet data in Git, not Git LFS pointers. No engine
run, sampling, or prompt generation is needed before training.

| Split | Positions | Engine reference |
|---|---:|---|
| train.parquet | 100,000 | Stockfish 11 64, depth 12 |
| dev.parquet | 128 | Existing fixed Stockfish 11, depth 14 split |
| test.parquet | 825 | Existing fixed Stockfish 11, depth 14 split |

Training phases: 28,000 opening, 44,000 middlegame, 28,000 endgame. Phase is a
material/fullmove-number heuristic, defined in `select_engine_data.py`.

All three splits have unique FEN first-four-field keys and no shared key.
This is a position-level exclusion, not proof of game-level independence.
The source pool contains 1,320,343 existing Stockfish-annotated positions derived
from `ChessExplained_2500k_qwen3.parquet`. Its upstream filtering and reference-move
agreement policy were inherited; no new model-response filtering is performed.
Only board positions and engine evaluations were selected, never prior model prose.

`selection.json` records the exact source SHA256, seed 20261005, eligible counts,
phase quotas and exclusions. `metadata.json` records the Parquet hashes, lengths
and tokenizer identities. `audit.json` records the completed dataset checks.
`text_migration.json` records the token-identical migration from stored IDs to text.
`dev.engine.jsonl` and `test.engine.jsonl` support the standalone chess evaluator;
they are not extra training rows.

## Model inputs

- Student: Qwen3-4B, `enable_thinking=False`, board and legal moves only.
- Teacher: Qwen3-8B, `enable_thinking=True`, board plus the three best engine moves,
  cp/mate scores, and immediate occupancy/capture facts computed from the board.
- Student output: three move explanations and `Best Move: MOVE`.
- The redundant instruction beginning `Use three distinct legal UCI moves.` is absent.
- Teacher context is readable English in `extra_info.teacher_prompt`, with actual
  newlines and no chat special tokens or integer arrays. `teacher_prompt_length`
  records its token count. See [a real training sample](sample.md).
- At training time the adapter encodes this text with the teacher's own tokenizer
  and chat template (`enable_thinking=True`), generates reasoning through `</think>`,
  then appends two newlines and the student's raw sampled response token IDs for
  scoring. Reasoning is private context and never becomes a student loss target.
  The Parquet files do not contain generated reasoning. All 100,953 rows have been
  converted to text; re-encoding gives exactly the former teacher token IDs.

The two model tokenizers must match the saved vocabulary/chat-template fingerprints.
Local model directory names may differ: the launcher checks tokenizer content, not
the machine's original absolute paths.

## Run

After provisioning the official verl environment described in the repository README,
run from the repository root using your local model directories:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  STUDENT_MODEL=/path/to/Qwen3-4B TEACHER_MODEL=/path/to/Qwen3-8B \
  bash examples/chess_opd/run_train.sh
```

The launcher defaults to the train/dev files in this directory. With the default
batch size 4, one epoch covers all 100,000 positions in 25,000 training batches.
Use a fresh output directory for this prompt/data version; old checkpoints were
created with different inputs.
