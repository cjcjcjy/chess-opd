"""Exercise the patched official teacher generation/scoring methods without Ray."""
import ast
import asyncio
import logging
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import torch

ROOT = Path(__file__).resolve().parents[2]


def load_patched_methods():
    """Load the exact installed method bodies, not copies of their implementations."""
    upstream = Path(os.environ.get("VERL_DIR", ROOT / "vendor/verl"))
    agent_path = upstream / "verl/experimental/agent_loop/agent_loop.py"
    manager_path = upstream / "verl/experimental/teacher_loop/teacher_manager.py"
    if not agent_path.exists():
        raise unittest.SkipTest("Run bash scripts/setup_verl.sh --checkout-only")
    tree = ast.parse(agent_path.read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AgentLoopWorker")
    method = next(node for node in worker.body if getattr(node, "name", None) == "_compute_teacher_logprobs")
    tree = ast.parse(manager_path.read_text())
    manager = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "AsyncTeacherLLMServerManager")
    sampling = next(node for node in tree.body if getattr(node, "name", None) == "_get_teacher_sampling_params")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             method, sampling, manager], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "uuid4": uuid4, "logger": logging.getLogger("teacher_test")}
    exec(compile(module, str(agent_path), "exec"), namespace)
    return namespace["_compute_teacher_logprobs"], namespace["AsyncTeacherLLMServerManager"]


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and add_generation_prompt and enable_thinking
        assert len(messages) == 1 and messages[0]["role"] == "user"
        assert messages[0]["content"] == "Private board and Stockfish reference."
        return getattr(self, "prompt_ids", [100])

    def encode(self, text, **kwargs):
        return {"<think>": [10], "</think>": [11], "\n\n": [12]}[text]

    def decode(self, ids, **kwargs):
        return "reasoning" if 50 in ids else ""


class TeacherContextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        method, cls.manager_class = load_patched_methods()
        cls.method = staticmethod(method)

    def make_worker(self, generated=None, width=1, context_limit=1000, stop_reason="completed"):
        manager = self.manager_class.__new__(self.manager_class)
        manager.teacher_model_configs = {"teacher": SimpleNamespace(
            model_path="teacher-model", inference=SimpleNamespace(max_model_len=context_limit, temperature=1.0))}
        manager._teacher_tokenizers = {"teacher": FakeTokenizer()}
        manager.distillation_loss_config = SimpleNamespace(
            topk=width, loss_settings=SimpleNamespace(use_topk=width > 1))
        if generated is None:
            generated = [10, 50, 11]

        async def generate(**kwargs):
            if "prompt_logprobs" not in kwargs["sampling_params"]:
                return SimpleNamespace(token_ids=generated, stop_reason=stop_reason)
            length = len(kwargs["prompt_ids"])
            ids = torch.arange(length * width).reshape(length, width)
            return SimpleNamespace(extra_fields={"prompt_ids": ids.tolist(),
                                                "prompt_logprobs": (-ids.float() / 10).tolist()})

        client = SimpleNamespace(generate=AsyncMock(side_effect=generate))
        manager.teacher_client = {"teacher": client}
        worker = SimpleNamespace(distillation_enabled=True, teacher_key="data_source",
                                 config={"chess_opd": {"teacher_think_max_tokens": 100}},
                                 teacher_server_manager=manager, tokenizer=FakeTokenizer())
        return worker, client

    def run_case(self, student_length, teacher_length, width, generated=None):
        response = [17, 18, 19, 99]  # EOS is included in the scored student tokens.
        private = list(range(100, 100 + teacher_length))
        worker, client = self.make_worker(generated=generated, width=width)
        worker.teacher_server_manager._teacher_tokenizers["teacher"].prompt_ids = private
        output = SimpleNamespace(extra_fields={}, multi_modal_data=None, mm_processor_kwargs=None,
                                 response_mask=[1] * len(response), response_ids=list(response))
        asyncio.run(self.method(worker, output, [5] * student_length, response, False,
                               {"data_source": "chess_opd", "extra_info": {
                                   "teacher_prompt": "Private board and Stockfish reference."}}))
        self.assertEqual(client.generate.await_count, 2)
        thinking, scoring = [call.kwargs for call in client.generate.call_args_list]
        self.assertEqual(thinking["prompt_ids"], private)  # No student answer leaks into reasoning.
        self.assertEqual(thinking["sampling_params"]["stop_token_ids"], [11])
        self.assertNotIn("prompt_logprobs", thinking["sampling_params"])
        self.assertEqual(scoring["prompt_ids"], private + [10, 50, 11, 12] + response)
        self.assertEqual(scoring["sampling_params"]["max_tokens"], 1)
        self.assertEqual(scoring["sampling_params"]["prompt_logprobs"], width if width > 1 else 0)
        prefix_length = teacher_length + 4
        ids = torch.arange((prefix_length + len(response)) * width).reshape(-1, width)
        self.assertEqual(output.extra_fields["teacher_ids"].shape, (student_length + len(response), width))
        torch.testing.assert_close(output.extra_fields["teacher_ids"][student_length:], ids[prefix_length:].int())
        torch.testing.assert_close(output.extra_fields["teacher_logprobs"][student_length:], -ids[prefix_length:].float() / 10)
        self.assertEqual(output.extra_fields["teacher_logprobs"][:student_length].count_nonzero(), 0)
        self.assertEqual(output.extra_fields["teacher_thinking_tokens"], 3)
        self.assertEqual(output.response_mask, [1] * len(response))
        self.assertEqual(output.response_ids, response)

    def test_sampled_logprob_alignment(self):
        self.run_case(3, 8, 1)
        self.run_case(8, 3, 1)

    def test_topk_alignment(self):
        self.run_case(3, 8, 7)

    def test_teacher_final_answer_is_never_a_label(self):
        self.run_case(3, 8, 1, generated=[10, 50, 11, 88, 89, 99])

    def test_failed_thinking_never_scores(self):
        for generated, stop_reason in (([10, 50], "completed"), ([10, 50, 99], "completed"),
                                       ([50, 11], "completed"), ([10, 11], "completed"),
                                       ([10, 10, 50, 11], "completed"), ([10, 50, 11], "aborted")):
            with self.subTest(generated=generated, stop_reason=stop_reason):
                worker, client = self.make_worker(generated=generated, stop_reason=stop_reason)
                output = SimpleNamespace(extra_fields={})
                with self.assertRaises(RuntimeError):
                    asyncio.run(self.method(worker, output, [5], [17, 99], False,
                        {"data_source": "chess_opd", "extra_info": {
                            "teacher_prompt": "Private board and Stockfish reference."}}))
                self.assertEqual(client.generate.await_count, 1)
                self.assertNotIn("teacher_logprobs", output.extra_fields)

    def test_insufficient_context_never_generates(self):
        worker, client = self.make_worker(context_limit=100)
        with self.assertRaisesRegex(ValueError, "context too small"):
            asyncio.run(self.method(worker, SimpleNamespace(extra_fields={}), [5], [17, 99], False,
                {"data_source": "chess_opd", "extra_info": {
                    "teacher_prompt": "Private board and Stockfish reference."}}))
        client.generate.assert_not_called()

    def test_invalid_or_obsolete_private_context_fails(self):
        for extra in ({"teacher_prompt": " "}, {"teacher_prompt": [100]}, {"teacher_prompt_ids": [100]}):
            with self.subTest(extra=extra):
                worker, client = self.make_worker()
                with self.assertRaises(ValueError):
                    asyncio.run(self.method(worker, None, [1], [2], False,
                                           {"data_source": "chess_opd", "extra_info": extra}))
                client.generate.assert_not_called()

    def test_teacher_tokenizer_is_loaded_once_per_routed_model(self):
        worker, _ = self.make_worker()
        manager = worker.teacher_server_manager
        del manager._teacher_tokenizers
        manager.teacher_model_configs["second"] = SimpleNamespace(model_path="second-model")
        teacher, second = FakeTokenizer(), FakeTokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=[teacher, second]) as loader:
            self.assertIs(manager.get_teacher_tokenizer("teacher"), teacher)
            self.assertIs(manager.get_teacher_tokenizer("teacher"), teacher)
            self.assertIs(manager.get_teacher_tokenizer("second"), second)
            self.assertEqual([call.args for call in loader.call_args_list], [("teacher-model",), ("second-model",)])

    def test_missing_private_context_fails(self):
        worker, client = self.make_worker()
        with self.assertRaisesRegex(ValueError, "requires private"):
            asyncio.run(self.method(worker, None, [1], [2], False, {"data_source": "chess_opd"}))
        client.generate.assert_not_called()

    def test_validation_skips_teacher(self):
        worker, client = self.make_worker()
        asyncio.run(self.method(worker, None, [1], [2], True))
        client.generate.assert_not_called()

    def test_non_chess_path_is_unchanged(self):
        worker, client = self.make_worker()
        output = SimpleNamespace(extra_fields={}, multi_modal_data=None, mm_processor_kwargs=None)
        asyncio.run(self.method(worker, output, [5, 6], [17, 99], False, {"data_source": "other"}))
        self.assertEqual(client.generate.await_count, 1)
        self.assertEqual(client.generate.call_args.kwargs["prompt_ids"], [5, 6, 17, 99])


if __name__ == "__main__":
    unittest.main()
