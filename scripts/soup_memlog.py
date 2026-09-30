"""Run the `soup` CLI with a per-step peak-VRAM logger attached (ADR 0004).

Soup 0.72.4 reports torch.cuda.memory_allocated() at log time, a point value, not a
peak. This wrapper adds a TrainerCallback to every HF Trainer Soup builds, via a
documented monkeypatch of Trainer.__init__, without modifying Soup itself.

Usage: python scripts/soup_memlog.py [--no-hf-grad-ckpt] <mem.jsonl> train --config configs/main.yaml ...

--no-hf-grad-ckpt works around a Soup 0.72.4 bug: its DPO trainer never overrides
TRL 0.24's DPOConfig.gradient_checkpointing=True default, so HF's reentrant
checkpointing recomputes streamed layers outside Soup's weight substitution and
crashes on meta tensors at step 0. Soup's SFT trainer already turns HF checkpointing
off under streaming (should_enable_hf_gradient_checkpointing); this applies the same
rule to DPO. Streaming's own per-layer checkpointing is unaffected.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from transformers import TrainerCallback


def _cuda_stats() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError
        free, total = torch.cuda.mem_get_info()
        return {
            "allocated": torch.cuda.memory_allocated(),
            "max_allocated": torch.cuda.max_memory_allocated(),
            "max_reserved": torch.cuda.max_memory_reserved(),
            "free": free,
            "total": total,
        }
    except Exception:
        return {k: None for k in ("allocated", "max_allocated", "max_reserved", "free", "total")}


class MemoryLogCallback(TrainerCallback):
    def __init__(self, path):
        self.path = Path(path)
        self.t0 = None

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        self.t0 = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        row = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "step": state.global_step,
            "elapsed_s": round(time.time() - (self.t0 or time.time()), 3),
            **_cuda_stats(),
        }
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def disable_trl_dpo_grad_ckpt() -> None:
    from trl import DPOConfig

    original = DPOConfig.__post_init__

    def patched(self):
        self.gradient_checkpointing = False
        original(self)

    DPOConfig.__post_init__ = patched


def _install(path) -> None:
    import transformers

    original = transformers.Trainer.__init__

    def patched(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.add_callback(MemoryLogCallback(path))

    transformers.Trainer.__init__ = patched


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--no-hf-grad-ckpt":
        argv = argv[1:]
        disable_trl_dpo_grad_ckpt()
        print("[soup_memlog] DEVIATION: TRL DPOConfig.gradient_checkpointing forced False", flush=True)
    _install(argv[0])
    sys.argv = ["soup"] + argv[1:]
    from soup_cli.cli import run

    sys.exit(run())
