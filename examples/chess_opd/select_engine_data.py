"""Select a reproducible phase-balanced dataset from existing Stockfish JSONL."""
import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import chess


def phase(board):
    material = sum({chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}.get(p.piece_type, 0)
                   for p in board.piece_map().values())
    return "endgame" if material <= 24 else "opening" if board.fullmove_number <= 10 else "middlegame"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dev-source", type=Path, required=True)
    parser.add_argument("--test-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-positions", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    if args.output_dir.exists() or args.train_positions <= 0:
        parser.error("Use a new output directory and a positive row count")
    quotas = {"opening": int(args.train_positions * .28), "middlegame": int(args.train_positions * .44)}
    quotas["endgame"] = args.train_positions - sum(quotas.values())
    seen = set()
    holdouts = {}
    for name, path in (("dev", args.dev_source), ("test", args.test_source)):
        count = 0
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = " ".join(chess.Board(row["fen"]).fen().split()[:4])
            if key in seen:
                raise ValueError("Duplicate holdout FEN")
            seen.add(key)
            count += 1
        holdouts[name] = count
    rng = random.Random(args.seed)
    reservoirs = {name: [] for name in quotas}
    available, excluded = Counter(), Counter()
    digest = hashlib.sha256()
    scanned = 0
    with args.source.open("rb") as source:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            digest.update(line)
            scanned += 1
            row = json.loads(line)
            board = chess.Board(row["fen"])
            key = " ".join(board.fen().split()[:4])
            if key in seen:
                excluded["duplicate_or_holdout"] += 1
                continue
            if not board.is_valid() or board.is_game_over() or board.legal_moves.count() < 3:
                excluded["invalid_terminal_or_fewer_than_three"] += 1
                continue
            seen.add(key)
            category = phase(board)
            available[category] += 1
            bucket = reservoirs[category]
            if len(bucket) < quotas[category]:
                bucket.append((offset, category))
            else:
                replacement = rng.randrange(available[category])
                if replacement < quotas[category]:
                    bucket[replacement] = (offset, category)
            if scanned % 100000 == 0:
                print(f"scanned {scanned}; eligible={dict(available)}", flush=True)
    selected = sorted(item for bucket in reservoirs.values() for item in bucket)
    if len(selected) != args.train_positions:
        raise ValueError(f"Insufficient eligible rows: requested={quotas}, available={dict(available)}")
    # Read selected offsets in source order to avoid random disk access. Retain
    # only chess/engine fields, never prior model text or quality labels.
    rows = []
    engines, depths = Counter(), Counter()
    with args.source.open("rb") as source:
        for offset, category in selected:
            source.seek(offset)
            item = json.loads(source.readline())
            board = chess.Board(item["fen"])
            legal = {move.uci() for move in board.legal_moves}
            ranked = item["ranked_moves"]
            evaluations = {entry["move"]: entry for entry in item["evaluations"]}
            if len(ranked) != len(legal) or set(ranked) != legal or set(evaluations) != legal:
                raise ValueError(f"Incomplete engine analysis: {item['source_index']}")
            if item.get("score_pov") != "side_to_move":
                raise ValueError("Expected side-to-move scores")
            moves = []
            for move in ranked:
                score = evaluations[move]
                if (score.get("cp") is None) == (score.get("mate") is None):
                    raise ValueError("Expected exactly one of cp/mate")
                moves.append({"move": move, "score_cp": score.get("cp"), "score_mate": score.get("mate"),
                              "depth": score.get("depth", item["depth"])})
            rows.append({"fen": board.fen(), "source_index": item["source_index"], "phase": category,
                         "moves": moves})
            engines[item.get("stockfish_name", "unknown")] += 1
            depths[item["depth"]] += 1
    rng.shuffle(rows)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "train.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    for name, path in (("dev", args.dev_source), ("test", args.test_source)):
        shutil.copyfile(path, args.output_dir / f"{name}.jsonl")
    metadata = {"source_file": args.source.name, "source_sha256": digest.hexdigest(),
                "scanned": scanned, "train_positions": len(rows), "holdout_positions": holdouts,
                "phase_counts": dict(Counter(row["phase"] for row in rows)), "engine_counts": dict(engines),
                "depth_counts": dict(depths), "seed": args.seed, "available": dict(available),
                "excluded": dict(excluded), "split_policy": "Unique FEN first four fields; fixed dev/test excluded",
                "source_policy": "Only FEN and engine analysis are used; no previous model responses"}
    (args.output_dir / "selection.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
