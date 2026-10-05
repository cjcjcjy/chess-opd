"""Exercise real PEFT LoRA initialization and backprop on a tiny CPU Qwen3.

Run with the same Python environment as training. No model download or GPU is
needed. This checks dependency compatibility, not the full distributed trainer.
"""
import importlib.metadata
import importlib.util
import json
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM


def main():
    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
    )
    try:
        model = get_peft_model(
            Qwen3ForCausalLM(config).cpu(),
            LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", task_type="CAUSAL_LM"),
        )
    except ImportError:
        if importlib.util.find_spec("awq") is not None:
            print(
                "LoRA initialization failed with AutoAWQ installed. For non-AWQ Qwen3 models, "
                f"remove this optional backend using: {sys.executable} -m pip uninstall autoawq",
                file=sys.stderr,
            )
        raise
    inputs = torch.randint(0, config.vocab_size, (1, 8), device="cpu")
    loss = model(input_ids=inputs, labels=inputs).loss
    loss.backward()
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    assert torch.isfinite(loss)
    assert trainable and all("lora_" in name for name, _ in trainable)
    assert all(param.grad is not None and torch.isfinite(param.grad).all() for _, param in trainable)
    assert any(param.grad.count_nonzero() > 0 for _, param in trainable)
    assert all(param.grad is None for param in model.parameters() if not param.requires_grad)
    print(json.dumps({
        "python": sys.executable,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")},
        "device": "cpu", "lora_rank": 8, "lora_alpha": 16, "target_modules": "all-linear",
        "trainable_parameters": sum(param.numel() for _, param in trainable),
        "loss": loss.item(), "lora_initialization_and_backward": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
