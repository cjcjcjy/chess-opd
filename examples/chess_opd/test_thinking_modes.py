"""Check role-specific prefixes with the actual Qwen3 tokenizer (no GPU needed)."""
import os
import unittest
from pathlib import Path

import chess
from transformers import AutoTokenizer

from examples.chess_opd.prepare_verl_data import convert_row, encode


class ThinkingModesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ.get("QWEN3_TOKENIZER", "models/Qwen3-4B"))
        if not path.exists():
            raise unittest.SkipTest("Set QWEN3_TOKENIZER to a local Qwen3 tokenizer directory")
        cls.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    def test_distinct_chat_prefixes_and_teacher_metadata(self):
        board = chess.Board()
        row = {"fen": board.fen(), "moves": [
            {"move": move.uci(), "score_cp": -i, "score_mate": None}
            for i, move in enumerate(board.legal_moves)]}
        converted = convert_row(row, self.tokenizer, self.tokenizer, 0, 2048, 3072)
        student_ids = encode(self.tokenizer, converted["prompt"], enable_thinking=False)
        student_prefix = self.tokenizer.decode(student_ids, skip_special_tokens=False)
        teacher_ids = converted["extra_info"]["teacher_prompt_ids"]
        teacher_prefix = self.tokenizer.decode(teacher_ids, skip_special_tokens=False)
        self.assertTrue(student_prefix.endswith("<think>\n\n</think>\n\n"))
        self.assertTrue(teacher_prefix.endswith("<|im_start|>assistant\n"))
        self.assertEqual(len(student_ids), converted["extra_info"]["student_prompt_length"])
        self.assertNotIn("stockfish engine:", student_prefix)
        self.assertIn("stockfish engine:", teacher_prefix)
        self.assertNotIn("Use three distinct legal UCI moves.", student_prefix)


if __name__ == "__main__":
    unittest.main()
