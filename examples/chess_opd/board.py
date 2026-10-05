"""Unicode board rendering for the standalone OPD example."""

import chess

UNICODE_PIECES = {
    "P": "♙",
    "R": "♖",
    "N": "♘",
    "B": "♗",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "r": "♜",
    "n": "♞",
    "b": "♝",
    "q": "♛",
    "k": "♚",
}

def render_unicode_board(fen: str) -> str:
    """Render a coordinate grid using the starter kit's Unicode pieces."""
    board = chess.Board(fen)
    fen_parts = fen.split()
    files = tuple("abcdefgh")
    coord_line = "   " + "".join(f" {file} " for file in files) + "  "
    rows = [
        "The facing unicode piece legend is:",
        "White: ♔ king, ♕ queen, ♖ rook, ♗ bishop, ♘ knight, ♙ pawn",
        "Black: ♚ king, ♛ queen, ♜ rook, ♝ bishop, ♞ knight, ♟ pawn",
        "Board (ranks 8 to 1, files a to h):",
        coord_line,
        "   +" + "-" * 24 + "+",
    ]

    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            cells.append(UNICODE_PIECES[piece.symbol()] if piece else "·")
        rows.append(f"{rank + 1} |" + "".join(f" {cell} " for cell in cells) + f"| {rank + 1}")

    rows.extend(
        [
            "   +" + "-" * 24 + "+",
            coord_line,
            f"Side to move: {'white' if board.turn == chess.WHITE else 'black'}",
            f"Castling rights: {fen_parts[2]}",
            f"En-passant square: {fen_parts[3]}",
        ]
    )
    return "\n".join(rows)
