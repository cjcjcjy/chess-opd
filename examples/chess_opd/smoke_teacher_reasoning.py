"""Real-model inference check of the patched methods; not an OPD trainer.

Uses a Transformers client adapter to exercise teacher generation followed by
student-token scoring on machines where the official vLLM stack is unavailable.
This does not validate Ray, vLLM, FSDP or optimizer updates.
"""
import argparse
import asyncio
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.chess_opd.test_official_adapter import load_patched_methods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("datasets/chess_opd_100k/dev.parquet"))
    parser.add_argument("--thinking-tokens", type=int, default=8192)
    parser.add_argument("--response-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    row = next(pq.ParquetFile(args.data).iter_batches(batch_size=1)).to_pylist()[0]
    tokenizer = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    assert tokenizer.get_vocab() == teacher_tokenizer.get_vocab()
    assert tokenizer.special_tokens_map == teacher_tokenizer.special_tokens_map
    device = "cuda:0"

    def load(path):
        return AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map={"": device}, local_files_only=True).eval()

    student = load(args.student)
    prompt_ids = tokenizer.apply_chat_template(row["prompt"], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    if hasattr(prompt_ids, "keys"):
        prompt_ids = prompt_ids["input_ids"]
    inputs = torch.tensor([prompt_ids], device=device)
    with torch.no_grad():
        generated = student.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
            do_sample=True, temperature=.7, top_p=.95, max_new_tokens=args.response_tokens,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    response = generated[0, len(prompt_ids):].tolist()
    del student, inputs, generated
    gc.collect()
    torch.cuda.empty_cache()
    teacher = load(args.teacher)
    observed = []

    async def client_generate(*, prompt_ids, sampling_params, **kwargs):
        inputs = torch.tensor([prompt_ids], device=device)
        if "prompt_logprobs" not in sampling_params:
            observed.append("thinking")
            assert prompt_ids == row["extra_info"]["teacher_prompt_ids"]
            with torch.no_grad():
                generated = teacher.generate(
                    input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    do_sample=True, temperature=sampling_params["temperature"],
                    top_p=sampling_params["top_p"], top_k=sampling_params["top_k"],
                    max_new_tokens=sampling_params["max_tokens"],
                    eos_token_id=sampling_params["stop_token_ids"] + [teacher_tokenizer.eos_token_id],
                    pad_token_id=teacher_tokenizer.pad_token_id)
            return SimpleNamespace(token_ids=generated[0, len(prompt_ids):].tolist(), stop_reason="completed")
        observed.append("scoring")
        assert prompt_ids[-len(response):] == response
        with torch.no_grad():
            logits = teacher(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                             use_cache=False, logits_to_keep=len(response) + 1).logits[0, :-1].float()
            logprobs = logits.log_softmax(-1).gather(
                -1, torch.tensor(response, device=device).unsqueeze(-1)).squeeze(-1).cpu()
        # The official method consumes a full-sequence array, then discards the
        # prompt part. Only response entries need true values in this smoke client.
        values = torch.zeros(len(prompt_ids), 1)
        values[-len(response):, 0] = logprobs
        return SimpleNamespace(extra_fields={"prompt_ids": [[i] for i in prompt_ids],
                                            "prompt_logprobs": values.tolist()})

    method, manager_class = load_patched_methods()
    manager = manager_class.__new__(manager_class)
    context = len(row["extra_info"]["teacher_prompt_ids"]) + args.thinking_tokens + len(response) + 2
    manager.teacher_model_configs = {"teacher": SimpleNamespace(
        inference=SimpleNamespace(max_model_len=context, temperature=1.0))}
    manager.distillation_loss_config = SimpleNamespace(topk=0, loss_settings=SimpleNamespace(use_topk=False))
    manager.teacher_client = {"teacher": SimpleNamespace(generate=client_generate)}
    worker = SimpleNamespace(distillation_enabled=True, teacher_key="data_source", tokenizer=tokenizer,
        config={"chess_opd": {"teacher_think_max_tokens": args.thinking_tokens}}, teacher_server_manager=manager)
    output = SimpleNamespace(extra_fields={}, multi_modal_data=None, mm_processor_kwargs=None)
    asyncio.run(method(worker, output, prompt_ids, response, False, row))
    scored_ids = output.extra_fields["teacher_ids"][len(prompt_ids):, 0].tolist()
    values = output.extra_fields["teacher_logprobs"][len(prompt_ids):, 0]
    assert observed == ["thinking", "scoring"] and scored_ids == response
    assert torch.isfinite(values).all()
    report = {"backend": "Transformers client exercising patched official methods; not vLLM/Ray",
        "source_index": row["extra_info"]["source_index"], "call_order": observed,
        "student_response_tokens": len(response), "student_eos": response[-1] == tokenizer.eos_token_id,
        "teacher_thinking_tokens": output.extra_fields["teacher_thinking_tokens"],
        "teacher_scoring_prompt_tokens": output.extra_fields["teacher_scoring_prompt_tokens"],
        "scored_tokens_match_student_exactly": True, "finite_logprobs": True,
        "mean_teacher_student_token_logprob": values.mean().item()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
