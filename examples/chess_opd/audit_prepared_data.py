"""Audit the ready-to-train parquet artifacts without model weights or a GPU."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import chess
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("datasets/chess_opd_100k"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.directory / "metadata.json").read_text())
    assert metadata["student_enable_thinking"] is False
    assert metadata["teacher_enable_thinking"] is True
    assert metadata["teacher_prompt_format"] == "text"
    seen, splits, phases = set(), {}, Counter()
    for split in ("train", "dev", "test"):
        path = args.directory / f"{split}.parquet"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == metadata["splits"][split]["sha256"], f"SHA256 mismatch: {split}"
        count = max_student = max_teacher = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            for row in batch.to_pylist():
                extra = row["extra_info"]
                board = chess.Board(extra["fen"])
                key = " ".join(board.fen().split()[:4])
                assert key not in seen, f"Repeated FEN: {key}"
                seen.add(key)
                assert board.is_valid() and not board.is_game_over()
                legal = {move.uci() for move in board.legal_moves}
                truth = json.loads(row["reward_model"]["ground_truth"])
                assert set(truth["legal"]) == legal and len(truth["legal"]) == len(legal) >= 3
                assert truth["top3"] == truth["legal"][:3]
                assert row["data_source"] == "chess_opd"
                assert len(row["prompt"]) == 1 and row["prompt"][0]["role"] == "user"
                content = row["prompt"][0]["content"]
                assert "stockfish engine:" not in content
                assert "Use three distinct legal UCI moves." not in content
                assert content.endswith("Best Move: MOVE")
                assert "teacher_prompt_ids" not in extra
                assert isinstance(extra["teacher_prompt"], str) and extra["teacher_prompt"].strip()
                assert "stockfish engine:" in extra["teacher_prompt"]
                assert isinstance(extra["teacher_prompt_length"], int) and extra["teacher_prompt_length"] > 0
                max_student = max(max_student, extra["student_prompt_length"])
                max_teacher = max(max_teacher, extra["teacher_prompt_length"])
                if split == "train":
                    phases[extra["phase"]] += 1
                count += 1
        expected = metadata["splits"][split]
        assert count == expected["positions"]
        assert max_student == expected["max_student_prompt_tokens"]
        assert max_teacher == expected["max_teacher_prompt_tokens"]
        splits[split] = {"positions": count, "file_bytes": path.stat().st_size,
                         "sha256": digest.hexdigest(), "max_student_prompt_tokens": max_student,
                         "max_teacher_prompt_tokens": max_teacher}
    assert {key: value["positions"] for key, value in splits.items()} == {"train": 100000, "dev": 128, "test": 825}
    assert dict(phases) == {"opening": 28000, "middlegame": 44000, "endgame": 28000}
    report = {"splits": splits, "train_phases": dict(phases), "unique_fen_keys": len(seen),
              "duplicate_or_overlapping_fens": 0, "legal_move_coverage": "passed for every row",
              "student_private_reference_leaks": 0, "removed_instruction_occurrences": 0,
              "student_enable_thinking": False, "teacher_enable_thinking": True,
              "teacher_prompt_format": "text",
              "default_batch_size": 4, "training_batches_per_epoch": 25000}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
