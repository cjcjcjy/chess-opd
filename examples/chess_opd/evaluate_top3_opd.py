#!/usr/bin/env python3
"""Evaluate the raw/LoRA student or privileged teacher on position-only splits."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from examples.chess_opd.preview_top3_prompts import student_prompt, teacher_prompt


LINE = re.compile(r"^([123])\.\s+([a-h][1-8][a-h][1-8][qrbn]?):\s+(.+?)\s*$")
BEST_LINE = re.compile(r"^Best Move: ([a-h][1-8][a-h][1-8][qrbn]?)$")


def parse_answer(text: str) -> tuple[list[str], list[str], bool]:
    text = visible_answer(text)
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    matches = [LINE.fullmatch(line) for line in lines[:3]]
    best = BEST_LINE.fullmatch(lines[-1]) if lines else None
    valid = (len(lines) == 4 and all(matches) and best is not None and
             [int(match.group(1)) for match in matches] == [1, 2, 3])
    if not valid:
        return [], lines, False
    moves = [match.group(2) for match in matches]
    return moves, lines, len(set(moves)) == 3 and best.group(1) == moves[0]


def parse_answer_flexible(text: str) -> tuple[list[str], list[str], bool]:
    """The shared output contract includes a consistent Best Move line."""
    return parse_answer(text)


def visible_answer(text: str) -> str:
    """Remove Qwen3's private thinking block before checking the requested format."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    if "<think>" in text:
        return ""
    return text.strip()


def score_value(move: dict) -> int:
    score = move.get("score") or {"cp": move["score_cp"], "mate": move["score_mate"]}
    if score["mate"] is None:
        return int(score["cp"])
    mate = int(score["mate"])
    return 100000 - 100 * mate if mate > 0 else -100000 - 100 * mate


def generate_responses(args, tokenizer, prompts):
    """Transformers needs no vLLM installation; vLLM is an optional faster backend."""
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        engine = LLM(
            model=str(args.model), dtype="bfloat16", tensor_parallel_size=1,
            max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.batch_size, enforce_eager=True, seed=args.seed,
            enable_lora=args.adapter is not None,
            **({"max_lora_rank": 16} if args.adapter else {}),
        )
        requests = engine.generate(
            prompts, SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                    max_tokens=args.max_tokens, seed=args.seed),
            lora_request=LoRARequest("top3_opd", 1, str(args.adapter.resolve())) if args.adapter else None,
        )
        yield from (request.outputs[0] for request in requests)
        return
    import torch
    from types import SimpleNamespace
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": "cuda:0"}, local_files_only=True,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    tokenizer.padding_side = "left"
    for start in range(0, len(prompts), args.batch_size):
        inputs = tokenizer(prompts[start:start + args.batch_size], padding=True,
                           add_special_tokens=False, return_tensors="pt").to("cuda:0")
        sampling = {"temperature": args.temperature, "top_p": args.top_p} if args.temperature > 0 else {}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens,
                                     do_sample=args.temperature > 0, **sampling,
                                     pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=tokenizer.eos_token_id)
        for output in outputs:
            ids = output[inputs.input_ids.shape[1]:].tolist()
            complete = tokenizer.eos_token_id in ids
            if complete:
                ids = ids[:ids.index(tokenizer.eos_token_id) + 1]
            yield SimpleNamespace(text=tokenizer.decode(ids, skip_special_tokens=True),
                                  token_ids=ids, finish_reason="stop" if complete else "length")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("student", "teacher"), required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positions", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"output directory exists: {args.output_dir}")
    if args.adapter and args.mode != "student":
        parser.error("adapter is only supported in student mode")
    if args.positions is not None and args.positions <= 0:
        parser.error("positions must be positive")

    rows = []
    with args.data.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if args.positions is not None and len(rows) >= args.positions:
                break
    if not rows:
        parser.error("no data rows")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = []
    for row in rows:
        content = teacher_prompt(row) if args.mode == "teacher" else student_prompt(row["fen"])
        messages = [{"role": "user", "content": content}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        length = len(tokenizer.encode(prompt, add_special_tokens=False))
        if length + args.max_tokens > args.max_model_len:
            parser.error(f"source_index={row['source_index']}: prompt needs {length} tokens")
        prompts.append(prompt)

    requests = generate_responses(args, tokenizer, prompts)
    args.output_dir.mkdir(parents=True)
    stats = Counter()
    rank_sum = 0
    regret_sum = 0
    records = []
    for row, output in zip(rows, requests, strict=True):
        text = output.text.strip()
        _, _, format_valid = parse_answer(text)
        moves, lines, semantic_valid = parse_answer_flexible(text)
        engine_moves = [m["move"] for m in row["moves"]]
        legal = set(engine_moves)
        all_legal = semantic_valid and all(move in legal for move in moves)
        move_rank = engine_moves.index(moves[0]) + 1 if all_legal else None
        third_score = score_value(row["moves"][2])
        tie_set_valid = all_legal and all(
            score_value(row["moves"][engine_moves.index(move)]) >= third_score - 30
            for move in moves
        )
        regret = None
        if move_rank is not None:
            best = row["moves"][0]
            chosen = row["moves"][move_rank - 1]
            if (best.get("score_cp", best.get("score", {}).get("cp")) is not None and
                chosen.get("score_cp", chosen.get("score", {}).get("cp")) is not None):
                regret = min(1000, max(0, score_value(best) - score_value(chosen)))
                regret_sum += regret
            rank_sum += move_rank
        stats["positions"] += 1
        stats["format_valid"] += format_valid
        stats["three_distinct_legal"] += all_legal
        stats["top1_exact"] += move_rank == 1
        stats["top3_exact_set"] += all_legal and set(moves) == set(engine_moves[:3])
        stats["top3_tie_aware"] += tie_set_valid
        stats["top1_bottom_quartile"] += (move_rank is not None and
                                           move_rank > math.floor(0.75 * len(engine_moves)))
        stats["truncated"] += output.finish_reason == "length"
        stats["total_response_tokens"] += len(output.token_ids)
        stats["total_response_words"] += len(text.split())
        records.append({
            "source_index": row["source_index"], "fen": row["fen"],
            "phase": row.get("phase"), "engine_top3": engine_moves[:3],
            "engine_all_moves": engine_moves, "response": text,
            "response_lines": lines, "moves": moves,
            "format_valid": format_valid, "three_distinct_legal": all_legal,
            "top1_rank": move_rank, "top1_regret_cp_capped": regret,
            "top3_tie_aware": tie_set_valid,
            "finish_reason": output.finish_reason,
            "response_tokens": len(output.token_ids),
        })
    with (args.output_dir / "responses.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "counts": dict(stats),
        "mean_top1_rank_when_valid": rank_sum / stats["three_distinct_legal"]
        if stats["three_distinct_legal"] else None,
        "mean_capped_cp_regret_when_available": regret_sum / sum(
            r["top1_regret_cp_capped"] is not None for r in records)
        if any(r["top1_regret_cp_capped"] is not None for r in records) else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
