# ADR 0005: Work around the streaming-DPO gradient-checkpointing crash

- Status: Accepted
- Date: 2026-09-29, after two crashed lr0 attempts and before any completed run

## Context

As shipped, `soup train` with `task: dpo` + `stream_layers: true` crashes at step 0 of the backward pass: `RuntimeError: Tensor on device cuda:0 is not on the expected device meta!`. Crash bundles: `.soup-crashes/crash_20260929T031416Z_2cedad2a.crash` and `crash_20260929T031549Z_574e76ec.crash`. Log: `runs/*_lr0/train_verbose.log`.

Root cause, confirmed in the source:
1. `trl/trainer/dpo_config.py:241-242` (TRL 0.24.0): `gradient_checkpointing` defaults to `True`.
2. Soup 0.72.4 turns HF checkpointing off under streaming only in `trainer/sft.py:408-413` (`should_enable_hf_gradient_checkpointing`). `trainer/dpo.py` never sets `gradient_checkpointing` on `DPOConfig`.
3. So HF enables its reentrant checkpointing (`transformers/modeling_utils.py:3692`, `use_reentrant=True`) on every `Qwen2DecoderLayer`. In backward, `torch/utils/checkpoint.py:308` re-runs the *inner* layer (`modeling_qwen2.py:232`) outside Soup's `StreamedDecoderLayer` weight substitution, so its weights are still `meta` placeholders.
4. Related warning at step 0: `checkpoint.py:232: None of the inputs have requires_grad=True. Gradients will be None`.

## Decision

All streaming runs are launched with `python scripts/soup_memlog.py --no-hf-grad-ckpt ...`. That patches `DPOConfig.__post_init__` to set `gradient_checkpointing=False`, the same rule Soup applies to SFT. Soup's own per-layer checkpointing (`StreamedDecoderLayer`, `use_reentrant=False`) still bounds activation memory, so the memory estimate is unchanged. The patch prints a `DEVIATION` line into every run log.

## Consequences

- "As shipped" streaming DPO does not run. This is defect #1 in the report and weighs on the verdict for Soup's config.
- Proposed fix in Soup (`trainer/dpo.py`): pass `gradient_checkpointing=should_enable_hf_gradient_checkpointing(tcfg.gradient_checkpointing, stream_layers=tcfg.stream_layers)` to `DPOConfig`, and add a streaming-DPO smoke test that runs one backward step.
- The same omission probably affects the other streaming preference trainers (`kto.py`, `orpo.py`, `simpo.py`), which the grep shows don't reference the helper either. Not tested.
