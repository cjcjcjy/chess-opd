#!/usr/bin/env python3
"""Preview prompt-only student input and engine-informed teacher input for OPD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import chess.engine
from examples.chess_opd.board import render_unicode_board


def move_facts(board: chess.Board, move: chess.Move) -> dict:
    """Minimal, verifiable occupancy and capture facts for a legal move."""
    if move not in board.legal_moves:
        raise ValueError(f"illegal move {move.uci()} for {board.fen()}")
    piece = board.piece_at(move.from_square)
    assert piece is not None
    if board.is_en_passant(move):
        capture_square = move.to_square - 8 if board.turn else move.to_square + 8
    else:
        capture_square = move.to_square
    captured = board.piece_at(capture_square)
    if captured is None and board.piece_at(move.to_square) is not None:
        raise RuntimeError(f"unexplained occupied destination for {move.uci()}")
    return {
        "uci": move.uci(),
        "san": board.san(move),
        "destination": chess.square_name(move.to_square),
        "destination_empty_before_move": board.piece_at(move.to_square) is None,
        "capture": None if captured is None else {
            "mover_color": "white" if piece.color else "black",
            "mover_piece": chess.piece_name(piece.piece_type),
            "captured_color": "white" if captured.color else "black",
            "piece": chess.piece_name(captured.piece_type),
            "square": chess.square_name(capture_square),
        },
    }


def analyse_all_moves(fen: str, engine_path: Path, depth: int) -> dict:
    board = chess.Board(fen)
    if not board.is_valid() or board.is_game_over():
        raise ValueError("FEN must describe a valid nonterminal position")
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        engine.configure({"Threads": 1, "Hash": 64})
        infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=board.legal_moves.count())
        engine_name = engine.id.get("name", "unknown")
    rows = []
    for info in infos:
        pv = info.get("pv", [])
        if not pv:
            raise RuntimeError("engine returned a move without PV")
        score = info["score"].pov(board.turn)
        rows.append({
            "move": pv[0].uci(),
            "score": {"cp": score.score(), "mate": score.mate()},
            "pv": [move.uci() for move in pv[:4]],
            "facts": move_facts(board, pv[0]),
            "reply_facts": move_facts(board_after(board, pv[0]), pv[1]) if len(pv) > 1 else None,
            "depth": info.get("depth"),
        })
    expected = {move.uci() for move in board.legal_moves}
    if {row["move"] for row in rows} != expected or len(rows) != len(expected):
        raise RuntimeError("engine did not evaluate each legal move exactly once")
    return {"engine": engine_name, "depth": depth, "score_pov": "side_to_move",
            "fen": fen, "moves": rows}


def board_after(board: chess.Board, move: chess.Move) -> chess.Board:
    result = board.copy(stack=False)
    result.push(move)
    return result


def student_prompt(fen: str) -> str:
    return position_prompt(fen) + """\n\nPlease choose and evaluate the best three legal moves, best first, and respond with the following format:

1. MOVE: describe the outcome of this move and judge the value of it.
2. MOVE: describe the outcome of this move and judge the value of it.
3. MOVE: describe the outcome of this move and judge the value of it.
Best Move: MOVE"""


def position_prompt(fen: str) -> str:
    board = chess.Board(fen)
    if not board.is_valid() or board.is_game_over() or board.legal_moves.count() < 3:
        raise ValueError("Expected a valid nonterminal position with at least three legal moves")
    legal = ", ".join(move.uci() for move in board.legal_moves)
    return f"""You are a strong chess player. Analyze the position, evaluate every listed legal move in order, and choose the strongest move for the side to move.

{render_unicode_board(fen)}

Legal moves (UCI): {legal}"""


def verified_move_description(board: chess.Board, facts: dict) -> str:
    """Describe only the moving piece and its verified destination or capture."""
    move = chess.Move.from_uci(facts["uci"])
    mover = board.piece_at(move.from_square)
    if move not in board.legal_moves or mover is None:
        raise ValueError(f"illegal verified move {move.uci()} for {board.fen()}")
    captured = facts["capture"]
    piece = chess.piece_name(mover.piece_type)
    if captured:
        description = f"the {piece} captures the {captured['piece']} on {captured['square']}"
        if captured["square"] != facts["destination"]:
            description += " en passant"
    else:
        description = (
            f"the {piece} moves from {chess.square_name(move.from_square)} "
            f"to {facts['destination']}"
        )
    if move.promotion:
        description += f" and promotes to a {chess.piece_name(move.promotion)}"
    return description


def teacher_reference(analysis: dict) -> str:
    top_three = analysis["moves"][:3]
    board = chess.Board(analysis["fen"])
    if len(top_three) != 3 or len({row["move"] for row in top_three}) != 3:
        raise ValueError("Teacher reference needs three distinct legal moves, best first")
    move_reference = []
    for row in top_three:
        fact = verified_move_description(board, move_facts(board, chess.Move.from_uci(row["move"])))
        score = row.get("score")
        if score is None:
            score = {"cp": row["score_cp"], "mate": row["score_mate"]}
        evaluation = (f"mate={score['mate']}" if score["mate"] is not None
                      else f"cp={score['cp']}")
        move_reference.append(f"{row['move']}: {fact}; {evaluation}")
    facts_text = "\n".join(move_reference)
    answer_format = "\n".join(
        f"{i}. {row['move']}: describe the outcome of this move and judge the value of it."
        for i, row in enumerate(top_three, 1)
    )
    return f"""The best three moves and their scores provided by stockfish engine:
{facts_text}

Scores are from the side-to-move perspective. cp is centipawns; 100 cp is roughly one pawn, and positive values favor the side to move. Positive mate means a predicted forced win in that many moves, and negative mate means a predicted forced loss.

Please evaluate the best three moves and response with the following format:

{answer_format}
Best Move: {top_three[0]['move']}"""


def teacher_prompt(analysis: dict) -> str:
    """Combine the shared chess task and teacher reference into one user message."""
    return position_prompt(analysis["fen"]) + "\n\n" + teacher_reference(analysis)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fen", default="1r3rk1/6p1/1bQ1pq1p/p3p3/4P3/1P3P2/P1P1N1PP/3R1R1K w - - 3 22")
    parser.add_argument("--stockfish", type=Path, default=Path("/usr/games/stockfish"))
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--student", type=Path, default=Path("models/Qwen3-4B"))
    parser.add_argument("--teacher", type=Path, default=Path("models/Qwen3-8B"))
    parser.add_argument("--output", type=Path, default=Path("prompt_preview.md"))
    parser.add_argument("--skip-tokenizer", action="store_true", help="Preview without downloading models")
    args = parser.parse_args()
    if args.depth <= 0:
        parser.error("--depth must be positive")
    analysis = analyse_all_moves(args.fen, args.stockfish, args.depth)
    user = student_prompt(args.fen)
    teacher_user = teacher_prompt(analysis)
    student_prefix, teacher_prefix = "", ""
    if not args.skip_tokenizer:
        from transformers import AutoTokenizer
        student = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
        teacher = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
        if student.get_vocab() != teacher.get_vocab() or student.special_tokens_map != teacher.special_tokens_map:
            parser.error("student and teacher token mappings differ")
        student_prefix = student.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        teacher_prefix = teacher.apply_chat_template(
            [{"role": "user", "content": teacher_user}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
    result = {
        "models": {"student": str(args.student), "teacher": str(args.teacher)},
        "analysis": analysis,
        "student_messages": [{"role": "user", "content": user}],
        "teacher_messages": [{"role": "user", "content": teacher_user}],
        "student_chat_prefix": student_prefix,
        "teacher_chat_prefix": teacher_prefix,
    }
    if args.output and args.output.suffix.lower() == ".md":
        rendered = (
            "# Top-three chess OPD prompt preview\n\n"
            f"Student: `{args.student}`  \nTeacher: `{args.teacher}`  \n"
            "Student uses `enable_thinking=False`; teacher uses `enable_thinking=True`.\n\n"
            "## Student user message\n\n```text\n" + user + "\n```\n\n"
            "## Teacher user message\n\n```text\n" + teacher_user + "\n```\n\n"
            "## Complete teacher thinking-enabled chat prefix\n\n```text\n"
            + result["teacher_chat_prefix"] + "\n```\n\n"
            "During OPD, the student's sampled answer tokens are appended to this prefix. "
            "The sampled answer differs on each rollout; the teacher scores its tokens "
            "rather than generating a replacement answer.\n"
        )
    else:
        rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    else:
        print(rendered)


if __name__ == "__main__":
    main()
