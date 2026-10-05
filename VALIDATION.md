# Validation — 2026-10-05

The standalone checkout was checked on Linux with Python 3.11, PyTorch 2.8.0+cu128,
Transformers 5.12.0, PEFT 0.18.1, python-chess 1.999, and two 48 GiB RTX 5880 Ada GPUs.
This used an existing environment, not a freshly provisioned machine.

Completed checks:

- Four CPU unittest groups: private teacher reference, final-answer format and
  thinking removal, en passant/promotion/castling occupancy facts, and dense
  teacher-to-student KL values and gradients.
- Stockfish depth-4 annotation: 8 train / 2 dev / 2 test synthetic positions.
- PGN exporter: 100 unique positions, 74 train / 26 heldout, with game-level splitting.
- Real Qwen3-4B / Qwen3-8B, one 128-token training update; saved a checkpoint,
  restored optimizer/RNG/adapter, and completed the next update.
- A separate real-model update at the default 2048-token limit completed and saved
  a checkpoint on two 48 GiB GPUs (batch size 1, full-vocabulary KL).
- Transformers evaluation loaded the saved student adapter and the frozen teacher
  and wrote both response records and summary metrics.
- Standalone Python compilation, shell syntax, and Git whitespace checks.

Both evaluation samples reached the 2048-token cap inside thinking. Their format
success rate was therefore 0/1. These tests demonstrate executable code paths,
not correct explanations or improved chess strength. Before a full training run,
choose a response budget from completion rates on a real development split.

No full training epoch, chess-strength comparison, explanation-factuality benchmark,
vLLM backend test, or smaller-GPU memory test was performed for this release.
