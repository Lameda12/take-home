import json
from types import SimpleNamespace

from soup_memlog import MemoryLogCallback


def test_callback_writes_one_timestamped_row_per_step(tmp_path):
    path = tmp_path / "mem.jsonl"
    cb = MemoryLogCallback(path)
    cb.on_train_begin(None, SimpleNamespace(), None)
    for step in (1, 2):
        cb.on_step_end(None, SimpleNamespace(global_step=step), None)
    rows = [json.loads(line) for line in path.read_text().split("\n") if line]
    assert [r["step"] for r in rows] == [1, 2]
    for r in rows:
        assert r["ts"].endswith("Z")
        assert {"max_allocated", "max_reserved", "allocated", "free", "total"} <= set(r)


def test_no_hf_grad_ckpt_patch_turns_off_trl_dpo_default(tmp_path):
    from trl import DPOConfig

    from soup_memlog import disable_trl_dpo_grad_ckpt

    assert DPOConfig(output_dir=str(tmp_path)).gradient_checkpointing is True  # TRL 0.24 default
    disable_trl_dpo_grad_ckpt()
    assert DPOConfig(output_dir=str(tmp_path)).gradient_checkpointing is False
