"""Validation-only chess metrics; official OPD has use_task_rewards=False."""
import json
import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # Standalone file: the official reward loader need not import this repository.
    truth = json.loads(ground_truth)
    if "</think>" in solution_str:
        solution_str = solution_str.rsplit("</think>", 1)[1]
    elif "<think>" in solution_str:
        solution_str = ""
    lines = [line.strip() for line in solution_str.splitlines() if line.strip()]
    pattern = r"^([123])\.\s+([a-h][1-8][a-h][1-8][qrbn]?):\s+(.+)$"
    matches = [re.fullmatch(pattern, line) for line in lines[:3]]
    valid = len(lines) == 4 and all(matches)
    moves = [match.group(2) for match in matches] if valid else []
    valid = bool(valid and [match.group(1) for match in matches] == ["1", "2", "3"]
                 and len(set(moves)) == 3 and lines[-1] == f"Best Move: {moves[0]}")
    legal = valid and all(move in truth["legal"] for move in moves)
    correct = legal and moves[0] == truth["top3"][0]
    return {"score": float(correct), "acc": float(correct), "format_valid": float(valid),
            "three_legal": float(legal), "top3_exact": float(legal and set(moves) == set(truth["top3"]))}
