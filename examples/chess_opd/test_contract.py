"""CPU checks for the chess prompt, output contract and special moves."""
import unittest

import chess

from examples.chess_opd.preview_top3_prompts import (
    move_facts, student_prompt, teacher_prompt, verified_move_description,
)
from examples.chess_opd.evaluate_top3_opd import parse_answer

FEN = "r1b2rk1/ppp2ppp/2nbpn2/8/PP3P2/2PB1N2/6PP/RNB2R1K b - - 0 11"
ANSWER = "1. f8d8: Rook to d8.\n2. a7a5: Attacks b4.\n3. f6d5: Attacks c3.\nBest Move: f8d8"


class ContractTest(unittest.TestCase):
    def test_teacher_reference_is_private(self):
        row = {"fen": FEN, "moves": [
            {"move": move, "score_cp": score, "score_mate": None}
            for move, score in zip(("f8d8", "a7a5", "f6d5"), (230, 213, 212))
        ]}
        teacher = teacher_prompt(row)
        student = student_prompt(FEN)
        self.assertIn("f8d8: the rook moves from f8 to d8; cp=230", teacher)
        self.assertTrue(teacher.endswith("Best Move: f8d8"))
        self.assertNotIn("cp=", student)
        self.assertNotIn("stockfish", student)
        self.assertIn("Best Move: MOVE", student)

    def test_output_and_thinking(self):
        self.assertTrue(parse_answer(ANSWER)[2])
        self.assertTrue(parse_answer("<think>candidate text</think>\n" + ANSWER)[2])
        self.assertFalse(parse_answer("<think>" + ANSWER)[2])
        self.assertFalse(parse_answer(ANSWER.replace("Best Move: f8d8", "Best Move: a7a5"))[2])
        self.assertFalse(parse_answer(ANSWER.replace("2. a7a5", "2. f8d8"))[2])
        self.assertFalse(parse_answer(ANSWER + "\nMore text")[2])
        self.assertFalse(parse_answer(ANSWER.rsplit("\n", 1)[0])[2])

    def test_special_moves(self):
        board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
        facts = move_facts(board, chess.Move.from_uci("e5d6"))
        self.assertEqual(facts["capture"]["square"], "d5")
        self.assertIn("en passant", verified_move_description(board, facts))
        board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        facts = move_facts(board, chess.Move.from_uci("a7a8q"))
        self.assertIn("promotes to a queen", verified_move_description(board, facts))
        board = chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
        facts = move_facts(board, chess.Move.from_uci("e1g1"))
        self.assertIsNone(facts["capture"])
        self.assertEqual(verified_move_description(board, facts), "the king moves from e1 to g1")

if __name__ == "__main__":
    unittest.main()
