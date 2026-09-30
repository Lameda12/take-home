# ADR 0003: Precision — run bf16 as shipped, measure fp16 separately

- Status: Accepted
- Date: 2026-09-29

## Context

Soup 0.72.4 sets `DPOConfig(bf16=self.device == "cuda")` (`trainer/dpo.py:160`) and stores the streamed base as bfloat16 on any CUDA device (`trainer/stream_setup.py:122`). The T4 (compute capability 7.5) has no native bf16, so torch emulates it without warning. `soup doctor` reports "All checks passed" and does not mention this.

## Decision

- The main run and the controls use bf16 **as shipped**. The verdict is about Soup as a user gets it.
- One short fp16 comparison run (about 50–100 steps, same config and seed) measures step time, tok/s, peak VRAM, the loss trajectory and any NaN/inf or skipped steps. It needs a documented patch to `bf16`/`fp16`, which is logged as a deviation.
- NaN/inf in the loss or gradients is checked on every run.

## Consequences

- If bf16 emulation is only slower, it counts as a measured, non-blocking defect (ADR 0001). If it corrupts training (NaNs, stalled loss, a diverging margin), it blocks SHIP.
- The fp16 run is the first one dropped if GPU time runs out.
