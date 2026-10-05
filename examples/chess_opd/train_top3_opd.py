#!/usr/bin/env python3
"""Synchronous, position-only top-three chess on-policy distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.chess_opd.evaluate_top3_opd import parse_answer, parse_answer_flexible
from examples.chess_opd.preview_top3_prompts import student_prompt, teacher_prompt


MOVE_SPAN = re.compile(r"(?m)^(?:[123]\.\s+|Best Move: )([a-h][1-8][a-h][1-8][qrbn]?)")
PROMPT_VERSION = "top3-colon-best-move-20261005"


def encode_messages(tokenizer, messages: list[dict]) -> list[int]:
    value = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=True,
    )
    return value["input_ids"] if hasattr(value, "keys") else value


def response_weights(tokenizer, response_ids: list[int], *, move_weight: float,
                     other_weight: float, eos_weight: float, device: str) -> tuple[torch.Tensor, int, bool]:
    text = tokenizer.decode(response_ids, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    canonical_tokens = encoded["input_ids"] == response_ids
    if canonical_tokens:
        offsets = encoded["offset_mapping"]
    else:
        # Some generated BPE sequences are not canonical on re-encoding.
        prefix_lengths = [
            len(tokenizer.decode(response_ids[:i], skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False))
            for i in range(len(response_ids) + 1)
        ]
        offsets = list(zip(prefix_lengths[:-1], prefix_lengths[1:]))
    weights = torch.full((len(response_ids),), other_weight, device=device)
    answer_start = text.rfind("</think>") + len("</think>") if "</think>" in text else 0
    spans = [match.span(1) for match in MOVE_SPAN.finditer(text, answer_start)]
    for index, (start, end) in enumerate(offsets):
        if any(start < right and end > left for left, right in spans):
            weights[index] = move_weight
    if response_ids and response_ids[-1] == tokenizer.eos_token_id:
        weights[-1] = eos_weight
    return weights, len(spans), canonical_tokens


def full_teacher_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                    weights: torch.Tensor) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape or student_logits.shape[0] != len(weights):
        raise ValueError("response logits or weights are misaligned")
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_prob = teacher_logp.exp()
    log_ratio = torch.where(teacher_prob > 0, teacher_logp - student_logp, 0.0)
    token_kl = (teacher_prob * log_ratio).sum(-1)
    return (token_kl * weights).sum() / weights.sum()


def load_model(path: Path, device: str, *, trainable: bool, adapter: Path | None = None):
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
        device_map={"": device}, low_cpu_mem_usage=True,
    )
    model.config.use_cache = not trainable
    if trainable:
        if adapter:
            model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
            ))
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    else:
        model.requires_grad_(False)
        model.eval()
    return model


def save_checkpoint(output_dir: Path, step: int, cursor: int, student, tokenizer,
                    optimizer) -> Path:
    directory = output_dir / f"checkpoint_step_{step:05d}"
    temporary = output_dir / f".checkpoint_step_{step:05d}.tmp"
    if directory.exists() or temporary.exists():
        raise RuntimeError(f"checkpoint path already exists: {directory}")
    temporary.mkdir()
    student.save_pretrained(temporary / "adapter")
    tokenizer.save_pretrained(temporary / "adapter")
    torch.save({
        "step": step, "cursor": cursor,
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }, temporary / "state.pt")
    temporary.rename(directory)
    return directory


def run_dev_eval(args, checkpoint: Path, step: int) -> dict:
    output = args.output_dir / f"dev_eval_step_{step:05d}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.eval_gpu
    env["OMP_NUM_THREADS"] = "1"
    command = [
        str(args.eval_python), "-m", "examples.chess_opd.evaluate_top3_opd",
        "--data", str(args.dev_data), "--model", str(args.student),
        "--mode", "student", "--adapter", str(checkpoint / "adapter"),
        "--output-dir", str(output), "--max-tokens", str(args.max_new_tokens),
        "--positions", str(args.dev_positions),
    ]
    subprocess.run(command, check=True, env=env, cwd=Path.cwd(), stdout=subprocess.DEVNULL)
    return json.loads((output / "summary.json").read_text())


def semantic_legal_rate(responses_path: Path) -> float:
    rows = [json.loads(line) for line in responses_path.open(encoding="utf-8")]
    if not rows:
        raise RuntimeError(f"empty evaluation responses: {responses_path}")
    valid = 0
    for row in rows:
        moves, _, parsed = parse_answer_flexible(row["response"])
        valid += bool(parsed and all(move in row["engine_all_moves"] for move in moves))
    return valid / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path, default=Path("models/Qwen3-4B"))
    parser.add_argument("--teacher", type=Path, default=Path("models/Qwen3-8B"))
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--dev-data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--student-device", default="cuda:0")
    parser.add_argument("--teacher-device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--move-weight", type=float, default=1.0)
    parser.add_argument("--other-weight", type=float, default=1.0)
    parser.add_argument("--eos-weight", type=float, default=1.0)
    parser.add_argument(
        "--all-output-tokens",
        action="store_true",
        help="Use equal KL weight 1.0 for every sampled response token, including EOS.",
    )
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--dev-positions", type=int, default=128)
    parser.add_argument("--eval-python", type=Path,
                        default=Path(sys.executable))
    parser.add_argument("--eval-gpu", default="2")
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument(
        "--no-stability-stop",
        action="store_true",
        help="Run all planned steps even when development metrics fall below baseline.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.student_device == args.teacher_device:
        parser.error("student and teacher require separate devices")
    if args.batch_size <= 0 or args.max_new_tokens <= 0 or args.save_every <= 0:
        parser.error("batch-size, max-new-tokens, and save-every must be positive")
    if min(args.move_weight, args.other_weight, args.eos_weight) <= 0:
        parser.error("all token weights must be positive")
    if args.all_output_tokens:
        args.move_weight = args.other_weight = args.eos_weight = 1.0
    if args.output_dir.exists() and not args.resume:
        parser.error(f"output directory exists: {args.output_dir}")
    if args.resume and not args.output_dir.exists():
        parser.error("resume requires the existing output directory")
    if args.eval_every and not args.dev_data:
        parser.error("dev-data is required for periodic evaluation")
    if args.eval_every < 0:
        parser.error("eval-every cannot be negative")
    if torch.cuda.device_count() < 2:
        parser.error("two CUDA devices are required")

    rows = [json.loads(line) for line in args.train_data.open(encoding="utf-8")]
    if not rows:
        parser.error("empty train data")
    data_sha256 = hashlib.sha256(args.train_data.read_bytes()).hexdigest()
    config_path = args.output_dir / "config.json"
    if args.resume:
        previous = json.loads(config_path.read_text())
        if previous.get("prompt_version") != PROMPT_VERSION or previous.get("data_sha256") != data_sha256:
            parser.error("resume requires the same prompt version and training data; start a new output directory")
        for key in ("student", "teacher", "batch_size", "seed", "lr", "max_new_tokens",
                    "move_weight", "other_weight", "eos_weight"):
            current = getattr(args, key)
            if previous.get(key) != (str(current) if isinstance(current, Path) else current):
                parser.error(f"resume configuration mismatch: {key}")
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    total_steps = math.ceil(len(order) / args.batch_size)
    if args.max_steps is not None:
        if args.max_steps <= 0:
            parser.error("max-steps must be positive")
        total_steps = min(total_steps, args.max_steps)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    tokenizer.padding_side = "left"
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab() or tokenizer.special_tokens_map != teacher_tokenizer.special_tokens_map:
        parser.error("student and teacher token IDs must match")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        config_path.write_text(json.dumps({
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "positions": len(rows), "planned_steps": total_steps,
            "prompt_version": PROMPT_VERSION, "data_sha256": data_sha256,
        }, indent=2) + "\n")

    teacher = load_model(args.teacher, args.teacher_device, trainable=False)
    student = load_model(
        args.student, args.student_device, trainable=True,
        adapter=args.resume / "adapter" if args.resume else None,
    )
    optimizer = torch.optim.AdamW(
        (p for p in student.parameters() if p.requires_grad), lr=args.lr,
        weight_decay=0.01,
    )
    start_step = 0
    cursor = 0
    if args.resume:
        state = torch.load(args.resume / "state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_step, cursor = state["step"], state["cursor"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    baseline = json.loads(args.baseline_summary.read_text()) if args.baseline_summary else None
    train_log = args.output_dir / "steps.jsonl"
    if args.resume and train_log.exists():
        # Drop logs after the durable checkpoint, since those updates will be replayed.
        retained = [line for line in train_log.read_text().splitlines()
                    if json.loads(line)["step"] <= start_step]
        train_log.write_text("\n".join(retained) + ("\n" if retained else ""))
    with train_log.open("a" if args.resume else "w", encoding="utf-8") as log:
        for step in range(start_step + 1, total_steps + 1):
            started = time.perf_counter()
            batch_indices = order[cursor:cursor + args.batch_size]
            batch = [rows[i] for i in batch_indices]
            users = [{"role": "user", "content": student_prompt(row["fen"])} for row in batch]
            student_prompt_ids_batch = [encode_messages(tokenizer, [user]) for user in users]
            padded = tokenizer.pad(
                {"input_ids": student_prompt_ids_batch}, padding=True, return_tensors="pt",
            ).to(args.student_device)
            student.eval()
            with torch.no_grad():
                generated = student.generate(
                    **padded, max_new_tokens=args.max_new_tokens,
                    do_sample=True, temperature=0.7, top_p=0.95,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id, use_cache=True,
                )
            sampled_responses = []
            for generated_row in generated:
                tokens = generated_row[padded["input_ids"].shape[1]:].tolist()
                if tokenizer.eos_token_id in tokens:
                    tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
                if not tokens:
                    raise RuntimeError("student generated zero response tokens")
                sampled_responses.append(tokens)
            del generated, padded
            optimizer.zero_grad(set_to_none=True)
            losses = []
            sample_results = []
            for row, user, student_prompt_ids, response_ids in zip(
                batch, users, student_prompt_ids_batch, sampled_responses, strict=True,
            ):
                teacher_prompt_ids = encode_messages(
                    teacher_tokenizer,
                    [{"role": "user", "content": teacher_prompt(row)}],
                )
                student_sequence = torch.tensor(
                    [student_prompt_ids + response_ids], dtype=torch.long,
                    device=args.student_device,
                )
                weights, move_spans, canonical_tokens = response_weights(
                    tokenizer, response_ids, move_weight=args.move_weight,
                    other_weight=args.other_weight, eos_weight=args.eos_weight,
                    device=args.student_device,
                )
                teacher_ids = torch.tensor(
                    [teacher_prompt_ids + response_ids], dtype=torch.long,
                    device=args.teacher_device,
                )
                with torch.no_grad():
                    teacher_logits_local = teacher(teacher_ids, use_cache=False).logits[
                        0, len(teacher_prompt_ids) - 1:-1,
                    ].float()
                    if not torch.isfinite(teacher_logits_local).all():
                        raise RuntimeError(
                            f"nonfinite teacher logits on source device at step {step}, "
                            f"source_index={row['source_index']}, "
                            f"bad_positions={(~torch.isfinite(teacher_logits_local).all(-1)).nonzero().flatten().tolist()}"
                        )
                    # Direct GPU-to-GPU copies corrupt some rows on this host's GPU pair.
                    # A synchronized host staging copy preserves the teacher distribution.
                    teacher_logits = teacher_logits_local.cpu().to(args.student_device)
                    if not torch.isfinite(teacher_logits).all():
                        raise RuntimeError(
                            f"nonfinite teacher logits after device transfer at step {step}, "
                            f"source_index={row['source_index']}, "
                            f"bad_positions={(~torch.isfinite(teacher_logits).all(-1)).nonzero().flatten().tolist()}"
                        )
                student.train()
                student_logits = student(student_sequence, use_cache=False).logits[
                    0, len(student_prompt_ids) - 1:-1,
                ]
                loss = full_teacher_kl(student_logits, teacher_logits, weights)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"nonfinite KL at step {step}, source_index={row['source_index']}, "
                        f"student_prefix={len(student_prompt_ids)}, "
                        f"teacher_prefix={len(teacher_prompt_ids)}, response={len(response_ids)}, "
                        f"student_finite={float(torch.isfinite(student_logits).float().mean())}, "
                        f"teacher_finite={float(torch.isfinite(teacher_logits).float().mean())}, "
                        f"student_abs_max={float(torch.nan_to_num(student_logits.detach()).abs().max())}, "
                        f"teacher_bad_positions={(~torch.isfinite(teacher_logits).all(-1)).nonzero().flatten().tolist()}, "
                        f"teacher_nan={int(torch.isnan(teacher_logits).sum())}, "
                        f"teacher_posinf={int(torch.isposinf(teacher_logits).sum())}, "
                        f"teacher_neginf={int(torch.isneginf(teacher_logits).sum())}, "
                        f"canonical_tokens={canonical_tokens}, response_ids={response_ids}, "
                        f"response={tokenizer.decode(response_ids, skip_special_tokens=False)!r}"
                    )
                (loss / len(batch)).backward()
                losses.append(float(loss.detach()))
                text = tokenizer.decode(response_ids, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False).strip()
                _, _, strict_valid = parse_answer(text)
                moves, _, semantic_valid = parse_answer_flexible(text)
                legal = {entry["move"] for entry in row["moves"]}
                sample_results.append({
                    "source_index": row["source_index"], "response": text,
                    "moves": moves, "format_valid": strict_valid,
                    "legal_distinct": semantic_valid and all(m in legal for m in moves),
                    "response_tokens": len(response_ids),
                    "complete": response_ids[-1] == tokenizer.eos_token_id,
                    "move_spans": move_spans,
                    "canonical_response_tokens": canonical_tokens,
                })
                del teacher_logits, teacher_logits_local, student_logits, student_sequence, teacher_ids, loss
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in student.parameters() if p.requires_grad), 1.0,
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"nonfinite gradient norm at step {step}")
            optimizer.step()
            cursor += len(batch)
            record = {
                "step": step, "cursor": cursor, "mean_kl": sum(losses) / len(losses),
                "grad_norm": float(grad_norm),
                "mean_response_tokens": sum(r["response_tokens"] for r in sample_results) / len(batch),
                "format_valid": sum(r["format_valid"] for r in sample_results),
                "legal_distinct": sum(r["legal_distinct"] for r in sample_results),
                "complete": sum(r["complete"] for r in sample_results),
                "noncanonical_response_tokens": sum(not r["canonical_response_tokens"] for r in sample_results),
                "samples": sample_results,
                "seconds": time.perf_counter() - started,
            }
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps({k: record[k] for k in (
                "step", "cursor", "mean_kl", "mean_response_tokens", "format_valid", "complete", "seconds",
            )}), flush=True)

            checkpoint_due = step % args.save_every == 0 or step == total_steps
            eval_due = args.eval_every and (step % args.eval_every == 0 or step == total_steps)
            if checkpoint_due or eval_due:
                checkpoint = save_checkpoint(args.output_dir, step, cursor, student, tokenizer, optimizer)
                print(f"checkpoint: {checkpoint}", flush=True)
                if eval_due:
                    dev = run_dev_eval(args, checkpoint, step)
                    counts = dev["counts"]
                    positions = counts["positions"]
                    strict_format_rate = counts.get("format_valid", 0) / positions
                    semantic_rate = semantic_legal_rate(
                        args.output_dir / f"dev_eval_step_{step:05d}" / "responses.jsonl"
                    )
                    truncation_rate = counts.get("truncated", 0) / positions
                    mean_tokens = counts.get("total_response_tokens", 0) / positions
                    print(json.dumps({"dev_step": step, "strict_format_rate": strict_format_rate,
                                      "three_legal_rate": semantic_rate,
                                      "truncation_rate": truncation_rate,
                                      "mean_response_tokens": mean_tokens,
                                      "top1_exact": counts.get("top1_exact", 0)}), flush=True)
                    if baseline and not args.no_stability_stop:
                        base_counts = baseline["counts"]
                        base_n = base_counts["positions"]
                        base_semantic_rate = semantic_legal_rate(
                            args.baseline_summary.parent / "responses.jsonl"
                        )
                        base_tokens = base_counts.get("total_response_tokens", 0) / base_n
                        earlier = sorted(
                            path for path in args.output_dir.glob("dev_eval_step_*")
                            if path.name < f"dev_eval_step_{step:05d}"
                            and (path / "responses.jsonl").exists()
                        )
                        prior_semantic_rate = (semantic_legal_rate(earlier[-1] / "responses.jsonl")
                                               if earlier else None)
                        persistently_low = (
                            semantic_rate < base_semantic_rate - 0.05
                            and prior_semantic_rate is not None
                            and prior_semantic_rate < base_semantic_rate - 0.05
                        )
                        if (persistently_low or truncation_rate > 0.02 or
                                mean_tokens > 1.25 * base_tokens):
                            (args.output_dir / "stopped_for_stability.json").write_text(
                                json.dumps({"step": step,
                                            "strict_format_rate": strict_format_rate,
                                            "three_legal_rate": semantic_rate,
                                            "prior_three_legal_rate": prior_semantic_rate,
                                            "baseline_three_legal_rate": base_semantic_rate,
                                            "truncation_rate": truncation_rate,
                                            "mean_response_tokens": mean_tokens}, indent=2) + "\n"
                            )
                            print("Stopped for development stability; last checkpoint saved.", flush=True)
                            break


if __name__ == "__main__":
    main()
