#!/usr/bin/env python3
"""Convert engine JSONL into official verl parquet, with private teacher context."""
import argparse
import hashlib
import json
from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from examples.chess_opd.preview_top3_prompts import student_prompt, teacher_prompt


def encode(tokenizer, messages):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def tokenizer_fingerprint(tokenizer):
    payload = {"vocab": tokenizer.get_vocab(), "special": tokenizer.special_tokens_map,
               "chat_template": tokenizer.chat_template}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def convert_row(row, student, teacher, index, max_prompt_length, max_teacher_prompt_length):
    board = chess.Board(row["fen"])
    legal = {move.uci() for move in board.legal_moves}
    ranked = [entry["move"] for entry in row["moves"]]
    if len(ranked) != len(legal) or set(ranked) != legal:
        raise ValueError(f"Incomplete or duplicate move ranking: {row['fen']}")
    messages = [{"role": "user", "content": student_prompt(row["fen"])}]
    teacher_messages = [{"role": "user", "content": teacher_prompt(row)}]
    student_ids = encode(student, messages)
    teacher_ids = encode(teacher, teacher_messages)
    if len(student_ids) > max_prompt_length or len(teacher_ids) > max_teacher_prompt_length:
        raise ValueError(f"Overlong prompt at row {index}: student={len(student_ids)}, teacher={len(teacher_ids)}")
    return {
        "data_source": "chess_opd", "ability": "chess", "prompt": messages,
        "reward_model": {"style": "rule", "ground_truth": json.dumps({"legal": ranked, "top3": ranked[:3]})},
        "extra_info": {
            "index": index, "source_index": row.get("source_index", index), "fen": row["fen"],
            "teacher_prompt_ids": teacher_ids, "student_prompt_length": len(student_ids),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="train/dev/test.jsonl from engine annotation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student", type=Path, default=Path("models/Qwen3-4B"))
    parser.add_argument("--teacher", type=Path, default=Path("models/Qwen3-8B"))
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-teacher-prompt-length", type=int, default=3072)
    args = parser.parse_args()
    student = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    teacher = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    if student.get_vocab() != teacher.get_vocab() or student.special_tokens_map != teacher.special_tokens_map:
        parser.error("Student and teacher token mappings must match exactly")
    if args.output_dir.exists():
        parser.error("Use a new output directory")
    args.output_dir.mkdir(parents=True)
    seen = set()
    stats = {}
    for split in ("train", "dev", "test"):
        source = args.input_dir / f"{split}.jsonl"
        output = args.output_dir / f"{split}.parquet"
        count = max_student = max_teacher = 0
        pending = []
        writer = None
        try:
            with source.open(encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    row = json.loads(line)
                    key = " ".join(chess.Board(row["fen"]).fen().split()[:4])
                    if key in seen:
                        raise ValueError(f"Repeated position across/within splits: {split}:{index}")
                    seen.add(key)
                    converted = convert_row(row, student, teacher, index,
                                            args.max_prompt_length, args.max_teacher_prompt_length)
                    max_student = max(max_student, converted["extra_info"]["student_prompt_length"])
                    max_teacher = max(max_teacher, len(converted["extra_info"]["teacher_prompt_ids"]))
                    pending.append(converted)
                    count += 1
                    if len(pending) == 256:
                        table = pa.Table.from_pylist(pending)
                        if writer is None:
                            writer = pq.ParquetWriter(output, table.schema)
                        writer.write_table(table)
                        pending = []
            if pending:
                table = pa.Table.from_pylist(pending)
                if writer is None:
                    writer = pq.ParquetWriter(output, table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        if not count:
            raise ValueError(f"Empty {split} split")
        digest = hashlib.sha256()
        with output.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        stats[split] = {"positions": count, "max_student_prompt_tokens": max_student,
                        "max_teacher_prompt_tokens": max_teacher,
                        "sha256": digest.hexdigest()}
    manifest = {"student": str(args.student.resolve()), "teacher": str(args.teacher.resolve()),
                "student_tokenizer_sha256": tokenizer_fingerprint(student),
                "teacher_tokenizer_sha256": tokenizer_fingerprint(teacher),
                "enable_thinking": True, "prompt_version": "top3-colon-best-move-20261005", "splits": stats}
    (args.output_dir / "metadata.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
