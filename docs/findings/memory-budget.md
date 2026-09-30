# Memory budget: estimate written BEFORE training

Written 2026-09-29, before any training step. The measured columns get filled in after the runs; this estimate is not revised.

Config: `configs/main.yaml`: Qwen2.5-1.5B-Instruct, DPO, layer streaming (2 buffers), bf16 as shipped, batch 2, max_length 1024, LoRA r=16 on q/v.

Model facts (Qwen2.5-1.5B config): hidden h=1536, intermediate i=8960, L=28 layers, vocab V=151,936, 12 query heads and 2 KV heads (KV dim 256), tied embeddings.

## Estimate (VRAM held by the torch allocator)

| # | Consumer | Calculation | Estimate |
|---|---|---|---|
| 1 | Streamed layer buffers | 1 layer ≈ attention (1536·1536·2 + 1536·256·2 ≈ 5.5M) + MLP (3·1536·8960 ≈ 41.3M) ≈ 46.8M params × 2 B (bf16) = 94 MB; × 2 buffers | **0.19 GB** |
| 2 | Resident embedding (tied with lm_head) | 151,936 × 1536 = 233M params × 2 B | **0.47 GB** |
| 3 | LoRA adapter + gradients + AdamW states | q: 16·(1536+1536)=49.2k; v: 16·(1536+256)=28.7k; × 28 layers = 2.18M params × 16 B (fp32 weight, grad, m, v) | **0.035 GB** |
| 4 | Activations (checkpointed per layer) | tokens = 1024 × 4 rows (batch 2 × chosen+rejected) = 4096. Saved layer inputs: 2 B × 28 × 1536 = 86 KB/token → 0.35 GB. The one layer being recomputed: 2·2 B·(1536+8960) = 42 KB/token → 0.17 GB | **0.52 GB** |
| 5 | **Logits** (policy forward + loss + backward) | V × S × rows × 14 B = 151,936 × 1024 × 4 × 14. The 14 B per element is 2 (bf16 logits) + 4 (fp32 upcast) + 4 (log-softmax) + 4 (gradient), matching Soup's measured `LOGITS_BYTES_PER_ELEMENT = 14` | **8.71 GB** |
| 6 | DPO reference pass | Same weights with the adapter off, so **no extra weights**. Runs under `no_grad`, so no checkpointed activations and no gradients. Its logits (V × S × 4 × 2–6 B ≈ 1.2–3.7 GB) exist *before* the policy forward and are freed once reduced to per-sequence log-probs, so they should **not** add to the peak | **~0 GB at peak** (risk: +1.2–3.7 GB if they stay alive) |
| 7 | Fixed slack (Soup constant) | `STREAM_FIXED_SLACK_BYTES` | 0.014 GB |
| | **Allocator peak (torch `max_memory_allocated`)** | 1+2+3+4+5+7 | **≈ 9.9 GB** |
| 8 | Allocator cache and fragmentation | `max_memory_reserved` − `max_memory_allocated`; large, varying logits tensors fragment the cache | +0.3 to 1.5 GB (guess) |
| 9 | CUDA context + cuBLAS workspace (outside the allocator) | seen only by nvidia-smi | +0.3 to 0.5 GB |
| | **Predicted nvidia-smi peak** | | **≈ 10.5 to 12 GB of 14.6 GB** |

## What the estimate says

- **Logits are 88% of the budget.** Weights, which people usually budget for first, are under 7% thanks to streaming. Batch 2 → 1 halves the peak (≈ 5.5 GB); max_length 1024 → 512 halves it too.
- Streaming moves the weight cost to **CPU RAM**: about 3.1 GB of bf16 store (all 1.54B params × 2 B) out of 13 GB, plus the Python/HF process itself.
- **Soup's pre-flight should predict slightly *more* than this table.** It counts the adapter as 4 targets × h×h (it assumes 4 targets for `auto`, and even with our explicit 2-target list it charges every target as h×h, although v_proj is only h×256), about 88 MB instead of 35 MB. Its other terms use the same formulas.
- **Things that would make the measured peak differ:**
  - reference-pass logits staying alive (row 6)
  - TRL's `concatenated_forward` padding chosen and rejected to the same length (the estimate already assumes full 1024)
  - bf16 emulation on the T4 creating extra fp32 temporaries (unknown)
  - Soup reporting `memory_allocated` (a point value) instead of a peak
  - `nvidia-smi` including rows 8 and 9

## Soup's own pre-training estimates (2026-09-29T03:05Z, same config)

Soup produced **three different peak-VRAM numbers** for one config. Logs: `profile_main.txt`, `plan_main.txt`, `dryrun_main.txt`.

| Tool | Peak VRAM | Problems |
|---|---|---|
| `soup profile` | **5.5 GB** ("OK Fits in 15 GB", **"Recommended batch_size: 4"**) | Charges 3.0 GB of *resident* model, so it ignores streaming. Has **no logits term** (8.7 GB here) and no DPO ×2 rows. Reports "trainable: 12,845,056" params where the real q/v r=16 adapter is about 2.18M. Its recommended batch 4 means about 17.4 GB of logits alone, which would OOM. |
| `soup plan` | **8.0 GB**, 20 min, $0.10 | A third number; the method is not shown. Written to `soup.tfstate` for `soup apply`. |
| `soup train --dry-run` | *(none)* | Validates the data (450 train / 50 val) and prints "Config valid. Ready to train!" **without running the streaming VRAM pre-flight**, which only runs at real train time. No bf16/T4 warning. |
| Streaming pre-flight (`_stream_budget_lines`) | **9.95 GB** (logits 8.71 GB), shown only once `soup train` starts | Matches our 9.9 GB; same formulas. It's the only Soup estimate that accounts for streaming and logits. |

Our hand estimate: **9.9 GB** allocator peak. The two Soup numbers a user sees *before* committing a GPU (5.5 and 8.0) are both below it, and `profile` recommends a config that should not fit.

## Measured (fill in after runs; do not edit the estimate above)

| Source | Value | Log |
|---|---|---|
Run: lr≈0 control, `runs/20260929T0318*_lr0` (57 steps, 03:19–04:36Z). Same shapes as the main run (batch 2, max_length 1024, streaming).

| Source | Value | Log |
|---|---|---|
| Soup pre-flight (`soup train` panel) | 9.95 GB predicted | `train.log` |
| Soup's in-run reported VRAM | **0.7 GB** ("GPU: 0.7/14.6 GB", final panel) | `train.log` |
| torch `max_memory_allocated` (our callback) | **8.56 GB** | `mem.jsonl` |
| torch `max_memory_reserved` | **15.07 GB** (14.04 GiB) | `mem.jsonl` |
| nvidia-smi peak `memory.used` | **14,539 MiB = 15.25 GB**, 94.7% of the T4's 15,360 MiB | `nvidia_smi.csv` |

### Explaining the gaps

1. **Estimate vs allocated (9.9 → 8.56 GB, 14% over).** *Corrected 2026-09-30.* The first explanation (padding to the longest sequence, not `max_length`) does **not** hold. The training set has 14 pairs of ≥1000 tokens (max 1024, 51 of ≥950), and 57 steps × 8 pairs covers essentially the whole set, so the peak batch was about 1024 long. The gap is in the logits term itself: 8.56 GB minus about 1.2 GB of other terms leaves about 7.4 GB for 151,936 × 1024 × 4 elements, which is **about 12 bytes per element, not the 14** that Soup's constant uses (measured on SFT). The mechanism isn't isolated. The likely candidate is TRL 0.24's `selective_log_softmax`, which avoids materialising a full fp32 log-softmax tensor. Still safe as a refusal gate, since it over-predicts.
2. **Allocated vs reserved (8.56 → 15.07 GB, +6.5 GB).** The caching allocator kept blocks from logits tensors whose size changes every batch (different padded lengths), so the cache fragmented. The panel says the `expandable_segments` allocator setting, the standard mitigation, was "not enabled" ("unavailable on this platform"), even though this is Linux. **No Soup estimate models this term, yet it's the one that nearly filled the card.**
3. **Reserved vs nvidia-smi (15.07 → 15.25 GB, +0.18 GB).** The CUDA context and cuBLAS workspace, outside the allocator.
4. **Soup's 0.7 GB**: `memory_allocated()` read after training, when the logits were freed. Off by 12× from the true allocated peak and 22× from what the card actually held.

**Bottom line:** the run needed 8.6 GB but held 15.25 GB, leaving about 0.8 GB of headroom on the card. Soup's pre-flight approved it with a "fits" margin of about 5 GB. A longer batch, or batch 3, would likely OOM despite passing the gate.
| Resident run (no streaming), same batch | | |
