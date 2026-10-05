# Official verl migration validation — 2026-10-05

## Ready 100K dataset and asymmetric thinking update

Student chat templates now use `enable_thinking=False`, while serialized teacher
prefixes and teacher-generation evaluation use `enable_thinking=True`. The redundant
student instruction about distinct moves and matching Best Move has been removed.
Default training paths point to the committed `datasets/chess_opd_100k` data.

Nine CPU unittest groups pass, including an additional real-Qwen3-tokenizer test
checking the student's empty thinking block and the teacher's open assistant prefix.
The official Hydra configuration test now asserts student thinking is disabled.

The source scan selected exactly 100,000 training positions: 28,000 opening,
44,000 middlegame and 28,000 endgame. The selected training and fixed held-out engine
records (100,953 total) all have scores ordered best-first under cp/mate ordering.
Full artifact audit details are recorded with the dataset in `audit.json`.

The full Parquet artifact audit passed: 100,953 distinct FEN keys, no cross-split
overlap, exact legal-move coverage on every row, no student reference leaks and no
occurrences of the removed instruction. All three SHA256 checks passed. Maximum
training prompt lengths are 906 student tokens and 1,032 teacher tokens.
The pinned official RLHFDataset loaded all 100,000 training rows; first/middle/last
rows were collated and checked with real Qwen3 tokenizers for the correct prefixes.
The default launcher's full-data preflight and dry run also passed: 100,000 train
rows plus 128 validation rows, 25,000 batches per epoch, student thinking disabled.

No training was started for this data preparation task.

## Earlier migration checks

The current training entry point delegates to the official
`examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh` at
`8718ca30a3f002f93b7c4fd99b9b2506718681bc`.
The former custom training loop is no longer part of the current tree.

Checks completed for this migration:

- Fetched the pinned official Git checkout; the private-teacher-context patch
  applies cleanly and repeated setup recognizes it as already applied.
- Eight CPU unittest groups: chess output contract, special move facts, private
  teacher reference, sampled-logprob alignment with longer/shorter teacher
  prefixes, top-k alignment, missing teacher metadata rejection, validation skip,
  and exact official launcher/Hydra configuration composition.
- Converted Stockfish-annotated sample data into 8 train / 2 dev / 2 test parquet
  records using the real local Qwen3-4B and Qwen3-8B tokenizers. Their token maps
  and chat templates match; private teacher context stays outside the student prompt.
- Loaded those 8 training rows with the pinned official `RLHFDataset` and
  `collate_fn`; private teacher metadata survives both operations and is absent
  from the student's raw prompt.
- Ran the launcher's data/tokenizer preflight and dry run: 8 training positions,
  batch size 4, two steps per epoch; one student GPU and one teacher GPU;
  official `k1`, policy gradient enabled, task rewards disabled.
- Composed the *actual official script's* complete Hydra arguments without
  importing its GPU stack. A temporary Python shim captured the invocation;
  this does not constitute a Ray/vLLM training run.
- Python compilation and shell syntax checks.

Reproduce CPU checks after checking out the upstream dependency, in an environment
with torch, transformers, python-chess and hydra-core (the official training
environment supplies all except python-chess):

```bash
bash scripts/setup_verl.sh --checkout-only
python -m unittest examples.chess_opd.test_contract \
  examples.chess_opd.test_official_adapter examples.chess_opd.test_official_config -v
```

No official GPU training or full epoch was run during this migration. The available
host uses NVIDIA driver 570.133.20, torch 2.8.0+cu128 and vLLM 0.11.0; the pinned
official environment instead specifies CUDA 13.0, torch 2.13.0 and vLLM 0.29.0.
Provision a compatible training machine/environment before the one-step official
smoke command in README, then run a full epoch for a learning comparison.

Historical real-model training/resume checks at commit `38050a8` applied only to
the removed handwritten trainer. They are **not evidence** that the official
trainer runs on this host or that the current prompts produce factual explanations.
