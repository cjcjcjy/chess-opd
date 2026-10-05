#!/usr/bin/env python3
"""Export legal positions from PGN; split by game before engine annotation."""
import argparse
import json
import random
from pathlib import Path

import chess.pgn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-positions", type=int, default=100000)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--heldout-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_positions <= 0 or args.stride <= 0 or not 0 < args.heldout_fraction < 1:
        parser.error("positive sizes and a heldout fraction between 0 and 1 are required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    seen = set()
    counts = {"train": 0, "heldout": 0}
    with args.pgn.open(encoding="utf-8-sig") as source, \
            (args.output_dir / "train.raw.jsonl").open("w") as train, \
            (args.output_dir / "heldout.raw.jsonl").open("w") as heldout:
        while sum(counts.values()) < args.max_positions:
            game = chess.pgn.read_game(source)
            if game is None:
                break
            if game.errors:
                raise ValueError(f"Invalid PGN game: {game.errors}")
            split = "heldout" if rng.random() < args.heldout_fraction else "train"
            output = heldout if split == "heldout" else train
            board = game.board()
            for ply, move in enumerate(game.mainline_moves(), 1):
                board.push(move)
                if ply < 8 or ply % args.stride or board.is_game_over() or board.legal_moves.count() < 3:
                    continue
                key = " ".join(board.fen().split()[:4])
                if key in seen:
                    continue
                seen.add(key)
                output.write(json.dumps({"fen": board.fen(), "source_index": sum(counts.values())}) + "\n")
                counts[split] += 1
                if sum(counts.values()) >= args.max_positions:
                    break
    print(json.dumps(counts))
    if not all(counts.values()):
        raise ValueError("Need more PGN games to obtain nonempty train and heldout splits")


if __name__ == "__main__":
    main()
