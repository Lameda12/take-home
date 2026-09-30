# Soup DPO + layer streaming on a T4: **SHIP this run, with conditions on the pipeline**

The adapter passes all four checks **pre-registered before any run** (ADR 0001), against two controls. **But Soup 0.72.4 can't produce this run as shipped:** streaming DPO crashes at step 0, and needed a one-line workaround (ADR 0005). My pre-results draft predicted DON'T SHIP; I kept the pre-registered rule. Full version: `docs/findings/report-long.md`.

**Setup:**
- Qwen2.5-1.5B-Instruct; DPO β 0.1; LoRA r16 on q/v (2.18M params)
- streaming with 2 buffers; bf16 as shipped
- batch 2 × accumulation 4; max_length 1024; lr 2e-5; 57 steps
- data: 600 filtered Russian pairs from `saiga_preferences` (reproducible hash)
- Soup silently trains on 450 of the 500 (a 10% `val_split`); 100 pairs held out
- controls with identical batches: lr≈1e-10 (Soup forbids 0) and swapped labels

## 1. Memory: estimated before training, then measured

| | GB |
|---|---|
| **Logits** 151,936 × 1024 × 4 rows × 14 B | 8.71 (88%) |
| Embeddings 0.47 + buffers 0.19 + activations 0.52 + LoRA/AdamW 0.04 | 1.22 |
| DPO reference pass (same weights, adapter off, no_grad) | ≈0 |
| **Hand estimate / Soup streaming pre-flight** | **9.9 / 9.95** |
| torch max allocated / **max reserved** | 8.56 / **15.07** |
| **nvidia-smi peak** | **15.25 (95% of the card)** |
| Soup's displayed VRAM · `soup profile` · `soup plan` | **0.7** · 5.5 (says batch 4 fits) · 8.0 |

**Gaps:**
- **Over-estimate:** logits cost about 12 B per element, not 14 (likely TRL's `selective_log_softmax`).
- **+6.5 GB reserved:** fragmentation from differently sized logits tensors; `expandable_segments` was off.
- **+0.18 GB:** the CUDA context.
- **Soup's 0.7 GB:** read *after* training.
- `profile` ignores streaming and logits.

**Throughput:** 157 tok/s padded vs a forecast of 172–253. The forecast assumes SFT compute and **emulated bf16**: 1.8–2.4 TFLOPS on a card that does about 65 in fp16.

## 2. Proof the model trained

**Loss is not proof here.** Soup's summary compares the first step with the *last single batch*: correct run **0.6931 → 0.6972** (looks worse), swapped-label run **0.6931 → 0.6812** (looks better). The mean loss moved about 0.002. Loss also can't exclude:
- both answers pushed down
- length bias (chosen is longer in 72.6% of pairs)
- an adapter that never loads
- reference == policy

**`scripts/verify_training.py`** (independent of Soup, written test-first with 37 tests, 100 held-out pairs):

| | main | lr≈0 | swapped |
|---|---|---|---|
| A: max lora_B norm (≥1e-4) | **0.0188 ✓** | 9.4e-08 ✗ | 0.0187 |
| B: loads into plain HF (tensors, logit diff) | **112/112, 0.25 ✓** | 112/112, 0.0003 | 112/112, 0.25 |
| C: held-out accuracy | **0.68** | 0.51 | **0.34** |
| C: margin ± 2 SE | **+0.0140 ± 0.0063 ✓** | 0.0000 | **−0.0141 ± 0.0066** |
| C: Δlogp chosen / rejected | +0.153 / +0.013 | 0 / 0 | −0.174 / −0.033 |
| D: controls behave as broken | **✓** | | |

**What the numbers show:**
- The swapped control mirrors the main run, and the do-nothing control sits at chance, so the checks track the training *signal*, not just "weights changed".
- Chosen became *more* likely, so the gain isn't "both pushed down".
- Generated length was unchanged (119/119 tokens), but on only 5 prompts.
- **The effect is small.**

**What the check misses:**
- whether the answers are actually *better* (it measures agreement with the labels)
- forgetting outside these pairs
- failures the controls don't simulate
- streaming-vs-resident equivalence (that comparison run was lost to the GPU quota)
- its thresholds are judgment calls
- B passes lr0 (a 0.0003 diff), so B proves loading, not usefulness

## 3. Silent failures

| Risk | How checked | Does Soup catch it? |
|---|---|---|
| **Streaming DPO + TRL's default HF checkpointing → crash / "Gradients will be None"** (fix exists in `sft.py:408`, not in `dpo.py`) | traceback + source | **No**: `doctor` and `--dry-run` pass |
| Adapter never trains / reference == policy | checks A and C + controls | No |
| Saved adapter loads 0 tensors (PEFT only warns; reproduced 0/8) | check B | Fixed in 0.72.1, not verified at run time |
| bf16 emulated on the T4 | panel TFLOPS | No warning |
| Loss summary shows the wrong direction; VRAM misreported; near-OOM | logs, torch, nvidia-smi | **Soup causes it** |
| Bad data: 63 planted faults of 7 types | `soup data lint` | **1/7 caught**; a skipped check shows "OK"; `data doctor` refuses DPO |
| Length bias | lint d=0.39 | Rated "MINOR" |
| `soup ship` on noise | rule probed offline | **0.00→0.01 = SHIP; a −0.049 general drop = SHIP** |
| Environment drift | `doctor` rates torchao 0.10 OK; training needs ≥0.16 | No |

## 4. Conditions (must change before this is used unattended)

1. Pass `gradient_checkpointing=should_enable_hf_gradient_checkpointing(...)` in `DPOConfig` (KTO, ORPO and SimPO likely need the same), and add a one-step backward smoke test to `--dry-run`.
2. Use fp16 on compute capability < 8.0.
3. Memory: enable `expandable_segments`, report reserved memory, make `profile` stream-aware. This config runs at 95% of the card.
4. Summarize loss as a mean or trend, not the last step.
5. `soup ship`: require a significant win, score pairwise for DPO, and check forgetting in the data's language.
6. Before a product claim: more steps, verbosity on ≥100 prompts, and a streaming-vs-resident check.

## What surprised me / what still concerns me

The feature under test didn't run at all as shipped: the fix sat in Soup's SFT trainer, unwired for DPO, and every pre-flight check passed. The defaults are easy to read but wrong:
- a loss summary that ranks the correct run below the swapped one
- 0.7 GB reported on a card that was 95% full
- a linter showing a skipped check as OK

The clean ±0.014 mirror between main and swapped convinced me the small effect is real. What still concerns me:
- the effect's size and usefulness
- no resident comparison
- verbosity measured on 5 prompts
- verification rests on the saved artifact, not the live streamed model

## Deviations (kept, not hidden)

- **Crashes:** two crashed lr0 attempts and a torchao error.
- **Workaround:** the flag on every streaming run.
- **Plan changes:** main run cut to 1 epoch; fp16 and resident comparisons not run (GPU quota).
- **Verification:** the first attempt died silently (my code); the final one ran on Apple MPS with generation cut to 5 prompts.
- **Lost outputs:** the Soup tool outputs are transcribed in `logs_raw/`.
- **AI use and 5 AI mistakes:** `AI_USAGE.md`.
