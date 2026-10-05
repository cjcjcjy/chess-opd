"""Validate official OPD data, private context and epoch coverage before Ray starts."""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--mini-batch-size", type=int, required=True)
    parser.add_argument("--max-prompt", type=int, required=True)
    parser.add_argument("--max-teacher-prompt", type=int, required=True)
    args = parser.parse_args()
    student = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    teacher = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    if student.get_vocab() != teacher.get_vocab() or student.special_tokens_map != teacher.special_tokens_map:
        parser.error("Student/teacher token IDs differ")
    if args.batch_size < 1 or args.mini_batch_size < 1 or args.batch_size % args.mini_batch_size:
        parser.error("Batch size must be positive and divisible by mini batch size")
    train_rows = pq.ParquetFile(args.train).metadata.num_rows
    if train_rows < args.batch_size or train_rows % args.batch_size:
        parser.error("Official dataloader drops incomplete batches; choose a batch size dividing the row count")
    seen = set()
    sizes = {}
    for split, path in (("train", args.train), ("val", args.val)):
        metadata = json.loads((path.parent / "metadata.json").read_text())
        if metadata.get("student_enable_thinking") is not False or metadata.get("teacher_enable_thinking") is not True:
            parser.error("Expected student thinking=False and teacher thinking=True; regenerate old parquet")
        if metadata.get("teacher_prompt_format") != "text":
            parser.error("Expected readable teacher_prompt text; regenerate old parquet")
        expected = metadata["splits"].get(path.stem)
        if expected is None:
            parser.error(f"No manifest entry for {path.name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected["sha256"] or pq.ParquetFile(path).metadata.num_rows != expected["positions"]:
            parser.error(f"Dataset checksum or row count mismatch: {path}")
        for name, tokenizer in (("student", student), ("teacher", teacher)):
            payload = {"vocab": tokenizer.get_vocab(), "special": tokenizer.special_tokens_map,
                       "chat_template": tokenizer.chat_template}
            fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            if metadata.get(name + "_tokenizer_sha256") != fingerprint:
                parser.error(f"{name} tokenizer or chat template changed; regenerate parquet")
        count = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                if row["data_source"] != "chess_opd":
                    parser.error("Expected chess_opd data_source")
                extra = row["extra_info"]
                text = extra.get("teacher_prompt")
                if not isinstance(text, str) or not text.strip() or "teacher_prompt_ids" in extra:
                    parser.error("Expected private teacher_prompt text, not stored token IDs")
                ids = teacher.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                    add_generation_prompt=True, enable_thinking=True)
                if hasattr(ids, "keys"):
                    ids = ids["input_ids"]
                if not ids or len(ids) > args.max_teacher_prompt:
                    parser.error("Missing/overlong private teacher prompt; regenerate data or increase its limit")
                if len(ids) != extra.get("teacher_prompt_length"):
                    parser.error("Teacher prompt token count differs from saved metadata")
                key = " ".join(extra["fen"].split()[:4])
                if key in seen:
                    parser.error("Duplicate FEN within or across train/val")
                seen.add(key)
                # Tokenize with the actual runtime tokenizer, not just saved length metadata.
                prompt_ids = student.apply_chat_template(row["prompt"], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
                if hasattr(prompt_ids, "keys"):
                    prompt_ids = prompt_ids["input_ids"]
                if len(prompt_ids) > args.max_prompt:
                    parser.error("Student prompt exceeds configured limit")
                # The readable text is encoded with the teacher's own chat template above.
                if "The best three moves and their scores provided by stockfish engine:" not in text:
                    parser.error("Teacher context does not contain the expected Stockfish reference")
                if "stockfish engine:" in row["prompt"][0]["content"]:
                    parser.error("Private teacher reference leaked into the student prompt")
                count += 1
        if not count:
            parser.error(f"Empty {split}")
        sizes[split] = count
    print(json.dumps({"validated_positions": sizes, "steps_per_epoch": train_rows // args.batch_size}))


if __name__ == "__main__":
    main()
