"""Exercise the patched official method directly without starting Ray or a GPU."""
import ast
import asyncio
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import torch

ROOT = Path(__file__).resolve().parents[2]


class TeacherContextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        upstream = Path(os.environ.get("VERL_DIR", ROOT / "vendor/verl"))
        source = upstream / "verl/experimental/agent_loop/agent_loop.py"
        if not source.exists():
            raise unittest.SkipTest("Run bash scripts/setup_verl.sh --checkout-only")
        tree = ast.parse(source.read_text())
        worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AgentLoopWorker")
        method = next(node for node in worker.body if getattr(node, "name", None) == "_compute_teacher_logprobs")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {"torch": torch}
        exec(compile(module, str(source), "exec"), namespace)
        cls.method = staticmethod(namespace["_compute_teacher_logprobs"])

    def run_case(self, student_length, teacher_length, width):
        responses = [17, 18, 19, 20]  # Raw sampled IDs, including any think/EOS tokens, never retokenized.
        private = list(range(100, 100 + teacher_length))
        length = teacher_length + len(responses)
        ids = torch.arange(length * width).reshape(length, width)
        logprobs = -ids.float() / 10
        manager = SimpleNamespace(compute_teacher_logprobs_single=AsyncMock(return_value=(ids, logprobs)))
        worker = SimpleNamespace(distillation_enabled=True, teacher_key="data_source",
                                 teacher_server_manager=manager, tokenizer=SimpleNamespace(pad_token_id=0))
        output = SimpleNamespace(extra_fields={}, multi_modal_data=None, mm_processor_kwargs=None)
        asyncio.run(self.method(worker, output, [5] * student_length, responses, False,
                               {"data_source": "chess_opd", "extra_info": {"teacher_prompt_ids": private}}))
        call = manager.compute_teacher_logprobs_single.call_args.kwargs
        self.assertEqual(call["sequence_ids"], private + responses)
        self.assertEqual(output.extra_fields["teacher_ids"].shape, (student_length + 4, width))
        torch.testing.assert_close(output.extra_fields["teacher_ids"][student_length:], ids[teacher_length:])
        torch.testing.assert_close(output.extra_fields["teacher_logprobs"][student_length:], logprobs[teacher_length:])
        self.assertEqual(output.extra_fields["teacher_logprobs"][:student_length].count_nonzero(), 0)

    def test_sampled_logprob_alignment(self):
        self.run_case(3, 8, 1)
        self.run_case(8, 3, 1)

    def test_topk_alignment(self):
        self.run_case(3, 8, 7)

    def test_missing_private_context_fails(self):
        worker = SimpleNamespace(distillation_enabled=True, teacher_key="data_source")
        with self.assertRaisesRegex(ValueError, "requires private"):
            asyncio.run(self.method(worker, None, [1], [2], False, {"data_source": "chess_opd"}))

    def test_validation_skips_teacher(self):
        # Validation evaluates the student without scoring from the private teacher.
        worker = SimpleNamespace(distillation_enabled=True)
        asyncio.run(self.method(worker, None, [1], [2], True))


if __name__ == "__main__":
    unittest.main()
