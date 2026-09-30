import sys
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _tiny_qwen2():
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(cfg)


@pytest.fixture
def base_dir(tmp_path):
    path = tmp_path / "base"
    _tiny_qwen2().save_pretrained(path)
    return path


def _save_adapter(base_dir, out_dir, lora_b_value):
    model = Qwen2ForCausalLM.from_pretrained(base_dir)
    peft_model = get_peft_model(
        model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
    )
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.fill_(lora_b_value)
    peft_model.save_pretrained(out_dir)
    return out_dir


@pytest.fixture
def trained_adapter(base_dir, tmp_path):
    return _save_adapter(base_dir, tmp_path / "trained", 0.05)


@pytest.fixture
def untrained_adapter(base_dir, tmp_path):
    return _save_adapter(base_dir, tmp_path / "untrained", 0.0)


@pytest.fixture
def inner_mangled_adapter(trained_adapter, tmp_path):
    """The pre-v0.72.1 streaming bug: wrapper name '.inner.' leaks into saved keys."""
    import re
    import shutil

    from safetensors.torch import load_file, save_file

    out = tmp_path / "mangled"
    shutil.copytree(trained_adapter, out)
    path = out / "adapter_model.safetensors"
    tensors = load_file(str(path))
    save_file({re.sub(r"(layers\.\d+\.)", r"\1inner.", k): v for k, v in tensors.items()}, str(path))
    return out


@pytest.fixture
def prompt_ids():
    return [[1, 5, 9, 17, 33, 2], [3, 4, 5, 6]]


@pytest.fixture
def pairs():
    """Held-out preference pairs as token ids (chat template applied upstream)."""
    return [
        {"prompt_ids": [1, 5, 9], "chosen_ids": [17, 33, 2], "rejected_ids": [40, 41, 42, 2]},
        {"prompt_ids": [3, 4], "chosen_ids": [6, 7, 8, 2], "rejected_ids": [60, 2]},
        {"prompt_ids": [11, 12, 13, 14], "chosen_ids": [20, 21], "rejected_ids": [90, 91, 92]},
    ]


CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> {% endfor %}"
    "{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
)


@pytest.fixture
def base_with_tokenizer(base_dir):
    """Tiny whitespace tokenizer with a ChatML-like template, saved beside the tiny model."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["<unk>", "<pad>", "<|im_end|>", "<|im_start|>", "user", "assistant", "system"]
    words += [f"w{i}" for i in range(128 - len(words))]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<|im_end|>"
    )
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(base_dir)
    return base_dir


@pytest.fixture
def heldout_jsonl(tmp_path):
    import json

    rows = [
        {"prompt": [{"role": "user", "content": "w1 w2 w3"}],
         "chosen": [{"role": "assistant", "content": "w10 w11"}],
         "rejected": [{"role": "assistant", "content": "w20 w21 w22"}]},
        {"prompt": [{"role": "user", "content": "w4 w5"}],
         "chosen": [{"role": "assistant", "content": "w12"}],
         "rejected": [{"role": "assistant", "content": "w30 w31"}]},
    ]
    path = tmp_path / "heldout.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


@pytest.fixture
def tiny_lr_adapter(base_dir, tmp_path):
    """What an lr=1e-10 control looks like: lora_B non-zero, but only by ~lr."""
    return _save_adapter(base_dir, tmp_path / "tiny_lr", 1e-10)
