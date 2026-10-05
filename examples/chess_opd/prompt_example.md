# Prompt example

Scores below reproduce the user-provided example; fresh Stockfish analysis may differ.

Chat templates: student `enable_thinking=False`; teacher `enable_thinking=True`.

## Student user message

```text
You are a strong chess player. Analyze the position, evaluate every listed legal move in order, and choose the strongest move for the side to move.

The facing unicode piece legend is:
White: ♔ king, ♕ queen, ♖ rook, ♗ bishop, ♘ knight, ♙ pawn
Black: ♚ king, ♛ queen, ♜ rook, ♝ bishop, ♞ knight, ♟ pawn
Board (ranks 8 to 1, files a to h):
    a  b  c  d  e  f  g  h
   +------------------------+
8 | ♜  ·  ♝  ·  ·  ♜  ♚  · | 8
7 | ♟  ♟  ♟  ·  ·  ♟  ♟  ♟ | 7
6 | ·  ·  ♞  ♝  ♟  ♞  ·  · | 6
5 | ·  ·  ·  ·  ·  ·  ·  · | 5
4 | ♙  ♙  ·  ·  ·  ♙  ·  · | 4
3 | ·  ·  ♙  ♗  ·  ♘  ·  · | 3
2 | ·  ·  ·  ·  ·  ·  ♙  ♙ | 2
1 | ♖  ♘  ♗  ·  ·  ♖  ·  ♔ | 1
   +------------------------+
    a  b  c  d  e  f  g  h
Side to move: black
Castling rights: -
En-passant square: -

Legal moves (UCI): g8h8, f8e8, f8d8, c8d7, a8b8, f6e8, f6d7, f6h5, f6d5, f6g4, f6e4, d6e7, d6e5, d6c5, d6f4, d6b4, c6d8, c6b8, c6e7, c6e5, c6a5, c6d4, c6b4, h7h6, g7g6, b7b6, a7a6, e6e5, h7h5, g7g5, b7b5, a7a5

Please choose and evaluate the best three legal moves, best first, and respond with the following format:

1. MOVE: describe the outcome of this move and judge the value of it.
2. MOVE: describe the outcome of this move and judge the value of it.
3. MOVE: describe the outcome of this move and judge the value of it.
Best Move: MOVE

```

## Teacher user message

```text
You are a strong chess player. Analyze the position, evaluate every listed legal move in order, and choose the strongest move for the side to move.

The facing unicode piece legend is:
White: ♔ king, ♕ queen, ♖ rook, ♗ bishop, ♘ knight, ♙ pawn
Black: ♚ king, ♛ queen, ♜ rook, ♝ bishop, ♞ knight, ♟ pawn
Board (ranks 8 to 1, files a to h):
    a  b  c  d  e  f  g  h
   +------------------------+
8 | ♜  ·  ♝  ·  ·  ♜  ♚  · | 8
7 | ♟  ♟  ♟  ·  ·  ♟  ♟  ♟ | 7
6 | ·  ·  ♞  ♝  ♟  ♞  ·  · | 6
5 | ·  ·  ·  ·  ·  ·  ·  · | 5
4 | ♙  ♙  ·  ·  ·  ♙  ·  · | 4
3 | ·  ·  ♙  ♗  ·  ♘  ·  · | 3
2 | ·  ·  ·  ·  ·  ·  ♙  ♙ | 2
1 | ♖  ♘  ♗  ·  ·  ♖  ·  ♔ | 1
   +------------------------+
    a  b  c  d  e  f  g  h
Side to move: black
Castling rights: -
En-passant square: -

Legal moves (UCI): g8h8, f8e8, f8d8, c8d7, a8b8, f6e8, f6d7, f6h5, f6d5, f6g4, f6e4, d6e7, d6e5, d6c5, d6f4, d6b4, c6d8, c6b8, c6e7, c6e5, c6a5, c6d4, c6b4, h7h6, g7g6, b7b6, a7a6, e6e5, h7h5, g7g5, b7b5, a7a5

The best three moves and their scores provided by stockfish engine:
f8d8: the rook moves from f8 to d8; cp=230
a7a5: the pawn moves from a7 to a5; cp=213
f6d5: the knight moves from f6 to d5; cp=212

Scores are from the side-to-move perspective. cp is centipawns; 100 cp is roughly one pawn, and positive values favor the side to move. Positive mate means a predicted forced win in that many moves, and negative mate means a predicted forced loss.

Please evaluate the best three moves and response with the following format:

1. f8d8: describe the outcome of this move and judge the value of it.
2. a7a5: describe the outcome of this move and judge the value of it.
3. f6d5: describe the outcome of this move and judge the value of it.
Best Move: f8d8
```
