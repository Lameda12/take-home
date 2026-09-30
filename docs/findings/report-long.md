# Soup DPO + layer streaming on a free T4: **SHIP this run (with conditions on the pipeline)**

**Verdict: SHIP.** The trained adapter meets all four criteria that were **pre-registered before any run** (ADR 0001), and two control runs show the checks can tell real training from fake. **But Soup 0.72.4 cannot produce this run as shipped**: streaming DPO crashes at step 0. The run exists only because of a one-line workaround (ADR 0005). The pipeline problems below must be fixed before this setup is used again unattended.

The draft of this report, written before verification, predicted DON'T SHIP. The evidence came out the other way, and I kept the pre-registered rule rather than moving the goalposts.

Setup: Qwen2.5-1.5B-Instruct, DPO β 0.1, LoRA r16/α32 on q/v (2.18M params), streaming (2 buffers), bf16 as shipped, batch 2 × accumulation 4, max_length 1024, lr 2e-5, 57 steps. Data: 600 filtered Russian pairs from `IlyaGusev/saiga_preferences` (seed 42; the split hash was reproduced on two machines). Soup trains on 450 of the 500 because of a silent 10% `val_split`; 100 pairs are held out. Controls use the same batches in the same order (identical step-1 grad_norm, 5.45108): lr≈1e-10 (Soup forbids lr=0) and swapped labels.

## 1. Memory budget (estimate written before training)

| Term | Estimate |
|---|---|
| Logits: V·S·rows·14 B = 151,936·1024·4·14 | **8.71 GB (88%)** |
| Embedding (tied) + 2 layer buffers | 0.47 + 0.19 GB |
| Activations (checkpointed per layer) | 0.52 GB |
| LoRA + grads + AdamW (2.18M × 16 B) | 0.035 GB |
| DPO reference pass (same weights, adapter off, no_grad) | ≈0 at peak |
| **Hand estimate / Soup streaming pre-flight** | **9.9 / 9.95 GB** |

| Measured (identical in all 3 runs) | Value |
|---|---|
| torch `max_memory_allocated` / `max_memory_reserved` | 8.56 / 15.07 GB |
| nvidia-smi peak | 14,539 MiB = 15.25 GB (**94.7% of the card**) |
| Soup's in-run display | **0.7 GB** |
| `soup profile` / `soup plan` | **5.5 GB** (and recommends batch 4) / 8.0 GB |

**Gaps:**
- **Estimate − allocated (−1.4 GB):** the per-element logits cost is about 12 B, not the 14 B Soup measured on SFT; the likely cause is TRL's `selective_log_softmax`. It's not padding: 14 pairs are ≥1000 tokens.
- **Reserved − allocated (+6.5 GB):** allocator fragmentation from logits tensors whose size changes every batch. `expandable_segments` was "not enabled"; no Soup estimate models this.
- **nvidia-smi − reserved (+0.18 GB):** the CUDA context.
- **Soup's 0.7 GB:** `memory_allocated()` read *after* training.
- `soup profile` ignores both streaming and logits, and recommends batch 4, which my estimate puts at about 18 GB.

**Throughput:** measured about 157 tok/s padded (116 real) against a forecast of 172–253. The forecast uses SFT compute (6·P per token vs about 8·P for DPO) and a speed test in **emulated bf16**: the T4 has no bf16 hardware, measured 1.8–2.4 TFLOPS against about 65 TFLOPS fp16.

## 2. Did the model train? Yes: verified independently of Soup

**Why falling loss proves nothing here:** Soup's summary compares the first step with the *last single step* (2 pairs). It printed **0.6931 → 0.6972** for the correct run (looks worse) and **0.6931 → 0.6812** for the swapped-label run (looks better). The mean training loss moved by only about 0.002 (0.6914 vs 0.6945). Loss also can't rule out:
- pushing chosen *and* rejected both down
- learning "longer is better" (chosen is longer in 72.6% of pairs)
- an adapter that trains but never loads
- a reference model identical to the policy (then the loss stays at 0.693 and nothing is learned)

**Check:** `scripts/verify_training.py` (plain PEFT + transformers, no Soup code, written test-first with 37 tests, thresholds pre-registered). Run 2026-09-30T14:22Z on the saved adapters. Output: `logs/verify_*.json|log`.

| Check | main | lr≈0 control | swapped control |
|---|---|---|---|
| A: max lora_B norm (pass ≥ 1e-4) | **0.0188 ✓** | 9.4e-08 ✗ (as intended) | 0.0187 |
| B: saved adapter → plain HF model | 112/112 tensors, logit diff 0.251 ✓ | 112/112, diff 0.0003 | 112/112, diff 0.246 |
| C: held-out reward accuracy (100 pairs) | **0.68** | 0.51 | **0.34** |
| C: mean margin ± 2 SE | **+0.0140 ± 0.0063 ✓** | 0.0000 | **−0.0141 ± 0.0066** |
| C: Δlogp chosen / rejected vs reference | **+0.153 / +0.013** | 0 / 0 | −0.174 / −0.033 |
| Generated length, tuned / base (5 prompts, ≤128 tokens) | 119 / 119 | | |
| **D: controls fail as expected** | **✓** | | |

**Reading the table:**
- Main prefers chosen on pairs it never saw, beyond 2 SE.
- The swapped control is a near-exact mirror, and the do-nothing control sits at chance. So the checks respond to the training *signal*, not just to "weights changed".
- Chosen became **more** likely (+0.153 nats): the improvement is not "both pushed down".
- A logit diff of 0.25 in a plain HF model shows the adapter is real outside Soup. It also shows the adapter-off reference path gives the base model.
- **The effect is small:** about 0.14 nats per sequence after 57 steps at lr 2e-5 on q/v only.

**What the check does NOT detect:**
- whether the preferred answers are actually *better* (it measures agreement with the labels)
- forgetting outside these 100 Russian pairs
- failure modes the two controls don't simulate
- the verbosity drift, which I measured on only 5 prompts (time-limited)
- whether streaming's forward pass matches a resident one (verification uses the saved artifact on a plain model; the resident comparison run was lost to the GPU quota)
- its thresholds are judgment calls: 2 SE, 10 nats, 1e-4
- lr0's B "passes" at a 0.0003 logit diff, so B proves the adapter loads, not that it matters. That's why A has a threshold.

## 3. Silent failures

| What could go wrong | How checked | Does Soup catch it? | Evidence |
|---|---|---|---|
| Streaming DPO + TRL's default HF checkpointing → crash, or "Gradients will be None" | traceback; source: TRL `dpo_config.py:241` (default True); Soup wires the off-switch into `sft.py:408` but not `dpo.py` | **No**: `doctor` and `--dry-run` both pass | ADR 0005; crash bundles |
| Adapter never trains | check A + lr≈0 control | No | §2 |
| Saved adapter silently loads 0 tensors | check B; reproduced (PEFT only warns, 0/8 loaded) | Fixed in 0.72.1, not verified at run time | tests |
| Reference == policy | Δlogp vs reference; mirrored controls | No | §2 |
| bf16 emulated on the T4 | panel TFLOPS | No warning | 1.8–2.4 TFLOPS |
| Loss summary shows the wrong direction | mean vs first→last step | **Soup causes it** | §2 |
| VRAM misreported; near-OOM from fragmentation | torch + nvidia-smi | No | §1 |
| Bad data rows | 63 planted faults, 7 types | **1/7 caught**; a skipped check shows "OK"; `data doctor` refuses DPO | `docs/findings` |
| Length bias drives the win | lint d=0.39 (rated "MINOR"); generated length | Understated | 119 vs 119 (n=5) |
| `soup ship` passes on noise | rule probed offline | **0.00→0.01 (one lucky answer) = SHIP; a −0.049 general drop = SHIP** | `probe_ship_rule.py` |
| Environment drift | `doctor` rates torchao 0.10 OK; training needs ≥0.16 | No | crash log |

## 4. Conditions attached to SHIP (what must change in the pipeline)

1. **Soup:**
   - pass `gradient_checkpointing=should_enable_hf_gradient_checkpointing(...)` to `DPOConfig` (the KTO, ORPO and SimPO trainers probably need the same fix; not tested)
   - add a one-backward-step streaming smoke test to `--dry-run`
2. **Precision:** pick fp16 on compute capability < 8.0 (`auto_mixed_precision` exists but defaults to off).
3. **Memory:**
   - enable `expandable_segments`
   - report `max_memory_reserved`
   - make `profile` account for streaming and logits
   - this config ran at 95% of the card, so batch 3 would likely OOM despite passing the gate
4. **Reporting:** summarize loss as a mean or trend, not first vs last step.
5. **`soup ship`:**
   - require a significant task win
   - score pairwise preference for DPO
   - check forgetting in the data's language
6. **Before a production claim:** more steps (the effect is small), measure verbosity on ≥100 prompts, and a resident-vs-streaming equivalence check.

## What surprised me / what still concerns me

I expected subtle numerics. Instead:
- **The feature under test didn't run at all as shipped.** The fix already existed in Soup's SFT trainer and simply wasn't wired into DPO, and every pre-flight check passed.
- **The defaults are easy to read but wrong:** a loss summary that ranks the correct run below the swapped one, a 0.7 GB memory figure on a card that was 95% full, and a data linter that shows a skipped check as OK.
- **I didn't expect the controls to mirror so cleanly** (±0.014). That symmetry is what convinced me the small effect is real, and it's why I kept the pre-registered SHIP even though I had drafted DON'T SHIP.

What still concerns me:
- The effect is small, and 100 held-out pairs can't tell me whether it's *useful*.
- Streaming and resident training were never compared numerically.
- Verbosity was checked on only 5 prompts.
- All of this rests on the saved artifact, not on the live streamed model.

## Deviations and failures (kept, not hidden)

- **Crashes:** two crashed lr0 attempts (ADR 0005) and one torchao ImportError.
- **Deviation:** the workaround flag on every streaming run.
- **Plan changes (ADR 0004):** main run cut to 1 epoch (80–86 s/step); fp16 and no-streaming comparisons not run (free GPU quota exhausted; the fp16 folder holds only its config).
- **Verification problems:** first Colab verification killed silently (my fp32-on-CPU load, then the runtime was lost); final verification ran on a local Apple MPS GPU against the same adapters, with generation cut to 5 prompts to fit the deadline.
- **Lost outputs:** the Soup tool outputs were lost from Drive and are transcribed verbatim in `logs_raw/`.
- **AI use:** logged in `AI_USAGE.md`, including 5 cases where the AI was wrong and how each was caught.
