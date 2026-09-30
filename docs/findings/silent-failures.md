# Silent-failure log (running)

Evidence for report Part 3. Each entry lists what could go wrong, how it was checked, whether Soup catches it, and the evidence. Log paths are relative to the Drive `logs/` folder.

## Data checks: planted-fault set vs `soup data lint` (soup 0.72.4, 2026-09-29T02:55Z)

63 rows, 9 per fault (`scripts/make_faults.py`, seed 7). Logs: `lint_faults.txt`, `lint_train.txt`.

| Fault (9 rows each) | Caught by `soup data lint`? | Notes |
|---|---|---|
| exact_duplicate | **Yes**: `identical_pairs` MAJOR, 9/63 | the only fault flagged MAJOR |
| near_duplicate | **No**, even with `datasketch` installed | Without `datasketch` (not in `soup-cli[train]`) the check is *skipped* but shows OK. With it installed it runs and reports 0/63, because it compares whole rows *with each other*, not chosen *with* rejected inside one row. Our 9 rows whose sides differ only in whitespace have no preference signal and pass. Log: `lint_faults_datasketch.txt` |
| shared_prefix_truncation | **No** | lint has no max_length/truncation check; `data doctor` has one but refuses DPO data |
| length_bias | **No**: `length_bias` OK, d=0.030 | a dataset-level effect size dilutes 9 extreme rows in 63; per-row outliers are invisible |
| homoglyph (Latin in Cyrillic) | **No** | no script/charset check |
| empty_completion | **No** | 9 rows with a whitespace-only chosen answer passed silently |
| prompt_echo | **Partly**: 6/9 | `min_prompt_len=40`, so short prompts aren't checked |

Result: **1 of 7 fault types fully caught, 1 partly.** The overall verdict (MAJOR) comes entirely from the exact duplicates. Remove those 9 rows and the same file would score MINOR.

## Real training data (`data/train.jsonl`, 500 rows)

- `length_bias` MINOR: Cohen's d = 0.392, chosen longer in 72.6% of rows (token lengths, Qwen tokenizer). **MINOR understates the risk for DPO**: a 70%+ "longer wins" signal is exactly what a DPO model can learn as a shortcut. Tracked after training through generated length, tuned vs base.
- `prompt_leak` MINOR: 2/500 rows.
- `near_duplicates`: skipped (see above) but reported OK.

## `soup data doctor`

`Error: format='dpo' is preference data — use soup data lint instead of soup data doctor.` So the chat-template, EOS and truncation checks never run on preference data (log: `data_doctor.txt`).

## Tooling friction

- `soup data lint -o <path>` refuses any path outside the current directory (`must stay under cwd`), so the report JSON can't go straight to the Drive log folder.

## Pre-flight tools

- `soup profile` ignores streaming, logits and DPO's doubled rows. It says "fits" and recommends **batch 4**, which our estimate puts at about 18 GB. The failure would be a loud OOM later, but the tool itself is silently wrong. See `memory-budget.md`.
- `soup train --dry-run` reports "Config valid. Ready to train!" without running the streaming VRAM pre-flight or saying anything about bf16 on a T4.
- `soup data` silently takes 10% of the training set for validation (`val_split` default 0.1): 450 of 500 pairs are trained on.

## Environment and runtime (first training attempt, 2026-09-29T03:07Z, lr0 control)

- **`soup doctor` passes a torchao version that crashes training.** Doctor lists torchao 0.10.0 as OK (optional, ≥0.4.0). `soup train` then failed after the layer shards were prepared: `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported`. It fails loudly, but the doctor's version table doesn't match what training needs. Fix used: `pip uninstall torchao` (not needed for this run).
- **bf16 emulation cost is visible in Soup's own panel**: "2.42 TFLOPS measured on this card". A T4 peaks at about 65 TFLOPS fp16 on tensor cores and about 8 TFLOPS fp32. Soup measures, then trains in emulated bf16 without warning. The forecast of 178–262 tok/s rests on that figure.
- **Mixed GB/GiB units**: "Memory: 14.6 GB" (actually GiB) vs "free VRAM 15.53 GB" (decimal GB). Free memory appears larger than the card.
- Streaming pre-flight: **9.95 GB** peak, logits 8.71 GB. Our hand estimate: 9.9 GB (`memory-budget.md`).

## Streaming DPO crashes at step 0 as shipped (see ADR 0005)

`RuntimeError: Tensor on device cuda:0 is not on the expected device meta!` in backward. Cause: TRL 0.24's `DPOConfig.gradient_checkpointing=True` default is never overridden by Soup's DPO trainer (only SFT uses `should_enable_hf_gradient_checkpointing`). HF's reentrant checkpointing then re-runs the inner decoder layer with meta weights. Loud, not silent. But the preceding warning, `None of the inputs have requires_grad=True. Gradients will be None`, is the silent version of the same problem wherever the weights happen to be real. `soup train --dry-run` and `soup doctor` both passed this config.

## lr≈0 control run (2026-09-29T03:19–04:36Z, 57 steps, with the ADR 0005 workaround)

- Behaved as a no-op should: loss 0.6931 on every logged step, `rewards/*` exactly 0.0. `grad_norm` was 4–8 at every step, so **gradients do reach the LoRA parameters under streaming** once HF checkpointing is off (contrast with the step-0 warning "Gradients will be None").
- **Soup's final live panel shows `Loss: 0.0000`** while the summary panel right below says `Loss: 0.6931 -> 0.6931`. Two different loss values for one run.
- **Soup's panel shows `GPU: 0.7/14.6 GB`**, read after training. The true peaks were 8.56 GB allocated, 15.07 GB reserved and 15.25 GB in nvidia-smi (`memory-budget.md`).
- Throughput: 80.8 s/step against a forecast of 172–253 tok/s. `train_samples_per_second` = 0.098.

## Throughput: forecast vs measured

Tokens per optimizer step, computed from the data (`data/train.jsonl`, Qwen tokenizer, 2,000 random batches of 2 pairs × 4 accumulation steps): **about 12,665 padded tokens (about 9,378 real)**. At 80.8 s/step (lr0 run) that's **about 157 tok/s padded, about 116 tok/s real**, against Soup's forecast of **172–253 tok/s**. The forecast uses 6·P FLOPs per token (SFT) and a GEMM rate of 2.34–2.42 TFLOPS measured **in emulated bf16**. DPO adds a reference forward pass (about 8·P per token), which alone would lower the forecast to about 129–190 tok/s. Real throughput is below even that.

## `soup ship` decision rule (probed offline: `scripts/probe_ship_rule.py`, soup 0.72.4)

| Evidence | Verdict |
|---|---|
| task 0.00 → 0.01 (one lucky answer out of 100), general benchmark flat | **SHIP** |
| task 0.500 → 0.501, general benchmark −0.049 | **SHIP** |
| task 0.50 → 0.51, general benchmark −0.051 | DON'T SHIP (regression) |
| task +0.01, no general benchmarks | DON'T SHIP (missing baseline) |
| tuned == base | DON'T SHIP (task_win) |

There's no noise floor or significance test on the task win (strict `tuned > base`). The forgetting threshold is a hard 0.05 cliff. For preference data, leg 1's `metric` mode scores `exact`/`contains`/`regex` against an `expected` string, which doesn't fit "chosen is better than rejected"; the fitting modes (`judge_score`, `pairwise`) need an external judge-model URL.

## Main and swapped runs (2026-09-30, 57 steps each, same batches/order as lr0: identical step-1 grad_norm 5.45108)

| Run | Soup summary "Loss a -> b" | HF `train_loss` (mean over steps) | s/step | torch alloc / reserved | nvidia-smi |
|---|---|---|---|---|---|
| lr≈0 | 0.6931 -> 0.6931 | 0.6931 | 80.8 | 8.56 / 15.07 GB | 14,539 MiB |
| main | 0.6931 -> **0.6972** | **0.6914** | 86.2 | 8.56 / 15.07 GB | 14,539 MiB |
| swapped | 0.6931 -> **0.6812** | **0.6945** | 86.4 | 8.56 / 15.07 GB | 14,539 MiB |

**Soup's completion summary reads the result backwards.** It compares the first logged step with the *last single step* (one micro-batch of 2 pairs). By that summary the correctly-labelled run got worse and the label-swapped run improved, while the mean training loss says the opposite, by only about 0.002. The per-step margins (±0.1) and accuracies (0.125–0.875) are dominated by batch noise. Whether any real learning exists is decided on held-out data (check C), not by these logs.
