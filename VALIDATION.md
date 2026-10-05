# Official verl migration validation — 2026-10-05

## Readable teacher prompts

All committed Parquet splits now store natural-language `extra_info.teacher_prompt`
with actual newlines, instead of `teacher_prompt_ids`. The adapter encodes the text
with the routed teacher's own cached tokenizer and `enable_thinking=True` at runtime.
Prompt wording, student thinking=False, teacher thinking=True and response-only
distillation are preserved.

The complete migration comparison checked all 100,953 rows: re-encoding each text
produces exactly the previous teacher token sequence. All other row fields are
unchanged. `datasets/chess_opd_100k/text_migration.json` records split counts and
old/new SHA256 values. The full dataset audit passed with no overlapping FEN keys
and complete legal-move coverage. `sample.md` in that directory displays a real
training row's student and teacher prompts.

Sixteen CPU tests pass, including text validation, obsolete-schema rejection,
per-teacher tokenizer caching/routing, exact response probability alignment,
thinking failure handling, and setup from clean, context-only and two-patch
checkouts. Setup remains idempotent. Shell syntax and whitespace checks pass.

The pinned official `RLHFDataset` loaded all 100,000 training rows. First, middle
and last samples retained the complete teacher text through `collate_fn`, with
matching token lengths and no private reference in the student's raw prompt.
The launcher's full-data preflight and dry run also passed on the new artifacts:
100,000 train and 128 validation positions, 25,000 batches for one epoch.

The earlier inference results below predate this storage-only migration; they
are retained as historical evidence. No GPU inference or training was run for
the text migration.

## Earlier teacher reasoning before student-token scoring

The adapter now makes two sequential calls through the official teacher client:
generate private reasoning through `</think>`, then compute probabilities for the
same student's answer conditioned on that reasoning. Student tokens, response mask
and EOS are preserved; teacher prefix and reasoning positions are excluded.

Fourteen CPU unittest groups pass. The added cases verify:

- The reasoning request receives only the private teacher prompt, with a token
  stop on `</think>`; scoring happens afterward on prompt + thinking + student tokens.
- Single-token and top-k logprobs align with the student even when prefix lengths differ.
- Any teacher answer after the closing tag is excluded; only student tokens are scored.
- Missing close tags, EOS before closing, malformed/nested/empty thinking, aborted
  requests and insufficient context never proceed to student scoring.
- Validation skips the teacher; non-chess paths retain their previous behavior.
- Exact official Hydra arguments include the reasoning budget and expanded context.
- Clean upstream checkout setup, upgrade from the earlier context-only patch, and
  repeated setup all produce identical patched source files.

At this earlier stage, the 100K data files were unchanged. Teacher reasoning is
generated at training time; adding reasoning did not require regenerating the dataset.

The local real-Qwen3 smoke at a 4,096-token reasoning limit reached that cap without
`</think>` and correctly refused scoring. The default thinking budget was increased
to 8,192; this is a configurable ceiling, not a guarantee of completion on every
position. The teacher service context length now reserves room for both the entire
reasoning budget and the student response (15,362 tokens by default).

`smoke_teacher_reasoning.py` is an inference-only test using a Transformers client
adapter to invoke the actual patched official method bodies. It performs no loss,
gradient update or checkpoint writing. Example (existing local model paths):

```bash
CUDA_VISIBLE_DEVICES=2 python -m examples.chess_opd.smoke_teacher_reasoning \
  --student /path/to/Qwen3-4B --teacher /path/to/Qwen3-8B \
  --thinking-tokens 8192 --output runs/teacher_reasoning_smoke.json
```

This diagnostic requires torch, transformers, pyarrow and the applied patches.
It does not validate the official Ray/vLLM/FSDP deployment or GPU memory requirements
of concurrent two-model training.

The real Qwen3-4B / Qwen3-8B inference check passed with the 8,192-token reasoning
budget on one 48 GiB RTX 5880 Ada (models loaded sequentially). On dev source index
370239, the teacher generated 5,309 reasoning tokens through the closing tag; its
scoring prefix was 6,033 tokens. The next call scored exactly the student's 256
sampled tokens with finite probabilities. The student hit the deliberately short
256-token smoke cap without EOS, so this is a generation/scoring alignment check,
not a successful final-answer or explanation-quality benchmark. EOS inclusion is
covered by the CPU tests. The mean teacher log probability of these student tokens
was -0.22425951. The test process exited and released its GPU.

The full-data launcher dry run passed after this change, still using the existing
100,000 train / 128 validation rows and 25,000 batches per epoch. It includes the
new teacher budget of 8,192 and the teacher context length of 15,362. All three
committed Parquet SHA256 hashes were unchanged at that stage. No training was launched.

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
