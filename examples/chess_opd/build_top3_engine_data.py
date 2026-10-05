#!/usr/bin/env python3
"""Build position-only OPD splits with all-legal-move Stockfish analysis."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import chess.engine

from examples.chess_opd.preview_top3_prompts import move_facts


def _analyse_record(item: dict, engine: chess.engine.SimpleEngine, depth: int) -> dict:
    board = chess.Board(item["fen"])
    infos = engine.analyse(
        board, chess.engine.Limit(depth=depth), multipv=board.legal_moves.count(),
    )
    moves = []
    for info in infos:
        pv = info.get("pv", [])
        if not pv:
            raise RuntimeError(f"missing PV for {item['fen']}")
        score = info["score"].pov(board.turn)
        moves.append({
            "move": pv[0].uci(),
            "score_cp": score.score(),
            "score_mate": score.mate(),
            "pv": [move.uci() for move in pv[:4]],
            "facts": move_facts(board, pv[0]),
            "depth": int(info.get("depth", 0)),
        })
    legal = {move.uci() for move in board.legal_moves}
    if len(moves) != len(legal) or {m["move"] for m in moves} != legal:
        raise RuntimeError(f"incomplete MultiPV analysis for {item['fen']}")
    return {**item, "moves": moves}


def _analyse_chunk(args: tuple[list[dict], str, int, int]) -> list[dict]:
    items, engine_path, depth, hash_mb = args
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        engine.configure({"Threads": 1, "Hash": hash_mb})
        return [_analyse_record(item, engine, depth) for item in items]


def _load_positions(path: Path, *, limit: int | None, seen: set[str]) -> tuple[list[dict], Counter]:
    rows: list[dict] = []
    counts: Counter = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and counts["read"] >= limit:
                break
            counts["read"] += 1
            source = json.loads(line)
            board = chess.Board(str(source["fen"]))
            if not board.is_valid() or board.is_game_over():
                counts["invalid_or_terminal"] += 1
                continue
            if board.legal_moves.count() < 3:
                counts["fewer_than_three_moves"] += 1
                continue
            key = " ".join(board.fen().split()[:4])
            if key in seen:
                counts["duplicate"] += 1
                continue
            seen.add(key)
            rows.append({
                "fen": board.fen(),
                "source_index": int(source.get("source_index", counts["read"] - 1)),
                "phase": source.get("phase", "unknown"),
            })
    counts["accepted"] = len(rows)
    return rows, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--test-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stockfish", type=Path, default=Path("/usr/games/stockfish"))
    parser.add_argument("--depth", type=int, default=14)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--dev-positions", type=int, default=128)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    args = parser.parse_args()
    if min(args.depth, args.workers, args.hash_mb, args.dev_positions) <= 0:
        parser.error("depth, workers, hash-mb, and dev-positions must be positive")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    seen: set[str] = set()
    train, train_counts = _load_positions(args.train_source, limit=args.train_limit, seen=seen)
    if not train:
        parser.error("no valid training positions")
    heldout, heldout_counts = _load_positions(args.test_source, limit=args.test_limit, seen=seen)
    if len(heldout) <= args.dev_positions:
        parser.error("not enough held-out positions for development and test splits")
    splits = {"train": train, "dev": heldout[:args.dev_positions],
              "test": heldout[args.dev_positions:]}
    args.output_dir.mkdir(parents=True)
    metadata = {
        "stockfish": str(args.stockfish), "depth": args.depth, "workers": args.workers,
        "train_source": str(args.train_source), "test_source": str(args.test_source),
        "train_source_counts": dict(train_counts), "heldout_source_counts": dict(heldout_counts),
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "input_policy": "Only FEN, source_index, and phase are read from source records; no old model responses or best-move labels are used.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for split, rows in splits.items():
            output = args.output_dir / f"{split}.jsonl"
            chunk_size = 64
            chunks = [
                (rows[start:start + chunk_size], str(args.stockfish), args.depth, args.hash_mb)
                for start in range(0, len(rows), chunk_size)
            ]
            with output.open("w", encoding="utf-8") as handle:
                index = 0
                for results in executor.map(_analyse_chunk, chunks):
                    for result in results:
                        handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    index += len(results)
                    if index % 256 == 0 or index == len(rows):
                        print(f"{split}: {index}/{len(rows)} positions", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
