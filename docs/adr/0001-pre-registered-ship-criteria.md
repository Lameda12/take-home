# ADR 0001: Pre-registered SHIP criteria

- Status: Accepted
- Date: 2026-09-29, before any training run

## Context

The brief warns that a completed run, a falling loss or `soup ship` printing SHIP does not prove correct training. Reading the source (`docs/research/soup-dpo-research.md`) shows that in 0.72.4 `soup ship` says SHIP whenever the tuned score is strictly above base and no general benchmark drops by more than 0.05. It has no noise floor, and its general suite is not Russian. To avoid fitting the bar to the results, the bar is fixed now.

## Decision

The verdict is **SHIP** only if all of the following hold:

1. **A: adapter trained.** Every `lora_B` tensor has a non-zero norm, and at least one norm is at least 1e-4 (amended 2026-09-29, before any run; see below), and all adapter parameters are in the optimizer's `param_groups`.
2. **B: artifact round-trips.** The saved adapter loads into a plain (non-streamed) HF model with no missing or unexpected LoRA keys, and its logits on held-out prompts differ from the base model's.
3. **C: learned on held-out data.** On 100 held-out pairs, reward accuracy and mean reward margin are above the reference (0.5 / 0) by more than the spread seen across the control runs. Mean `logps/chosen` does not collapse; this rules out "both chosen and rejected pushed down" as the only effect.
4. **D: controls fail the same checks.** The lr=0 run fails A and C. The swapped-labels run shows a reversed held-out margin.
5. **No unresolved silent failure.** Every item in the silent-failure table has been checked and either ruled out or measured and worked around.

**Numeric thresholds** (fixed 2026-09-29, before any run; implemented in `scripts/verify_training.py::verdict`):
- C: the mean held-out margin must be greater than 2 standard errors of the per-pair margins, reward accuracy must be above 0.5, and mean Δlogp(chosen) must not fall below −10 nats (summed over the sequence).
- D: the lr≈0 run (lr=1e-10, see the amendment) must fail A and show no margin greater than 2 SE; the swapped-labels run must have a mean margin below 0.
- Response length more than 1.2× the base triggers a **warning**, not a blocker (the length bias is known and in scope for the report).

A known Soup defect (for example forced bf16 on a T4) does **not** block SHIP if its effect has been measured and does not invalidate the checks above. Otherwise the verdict is **DON'T SHIP**, with a list of the required changes.

`soup ship` is run and reported, but its verdict is *evidence*, not the decision. A disagreement with checks A–D is reported as a finding.

## Consequences

- A clean negative result is acceptable and expected to be possible.
- Check C needs the control runs to exist, so they are mandatory, not optional.

## Amendment (2026-09-29, before any training run)

Soup's schema requires `lr > 0` (`TrainingConfig.lr: gt=0`), so a true lr=0 control is impossible. The control runs at lr=1e-10 instead. At that learning rate `lora_B` becomes non-zero (about 1e-10), so the original "non-zero" check A would have **passed** the control. Check A now also fails when every `lora_B` norm is below 1e-4. Why 1e-4: Adam moves each element by about lr per step. The control (lr 1e-10, about 50 optimizer steps) reaches a norm of at most about 8e-7 per `lora_B` of about 25k elements; the main run (lr 2e-5, about 125 steps) should reach about 1e-1. 1e-4 sits between the two with several orders of magnitude of margin on each side. Changed before any result was seen; the test is `tests/test_adapter_trained.py::test_adapter_moved_only_by_a_negligible_lr_fails`.
