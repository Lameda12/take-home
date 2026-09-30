# Soup tool outputs: transcribed from notebook cell output

The original `.txt` files written with `tee` were not found on Drive when the logs were collected (probably written to the ephemeral Colab disk on a day `$LOG` expanded empty). The text below is copied verbatim from the notebook cell output as it appeared, with the UTC timestamps each cell printed. Rich table borders are simplified.

## `soup version; soup doctor`: 2026-09-29T02:08:44Z
```
soup v0.72.4
Python: 3.13.15 | Platform: Linux 6.6.122+ | x86_64
`soup` runs under /usr/bin/python3; `python` on your PATH is /usr/local/bin/python.
CUDA: available (v12.8) | GPU 0: Tesla T4 (14.6 GB) | RAM 13 GB | Disk 65 GB free
torch 2.11.0+cu128 OK | transformers 4.57.6 OK | peft 0.20.0 OK | trl 0.24.0 OK | datasets 4.8.5 OK
bitsandbytes 0.50.2 OK | accelerate 1.14.0 OK | torchao 0.10.0 OK (optional, >=0.4.0) | datasketch not installed
All checks passed! Your environment is ready.
```

## `soup data doctor data/train.jsonl -m Qwen/Qwen2.5-1.5B-Instruct --max-length 1024`: 2026-09-29T02:55:03Z
```
Error: format='dpo' is preference data — use soup data lint instead of soup data doctor.
```

## `soup data lint data/train.jsonl --model Qwen/Qwen2.5-1.5B-Instruct`: 2026-09-29T02:55:34Z
```
length_bias     MINOR  effect size (Cohen's d) = 0.392 (chosen longer than rejected); chosen is longer in 72.6% of rows | mean chosen=529.8, mean rejected=438.9
identical_pairs OK     0/500 rows (0.0%) have chosen == rejected
near_duplicates OK     near-dup check skipped (datasketch not installed) | pip install "soup-cli[data]" to enable
prompt_leak     MINOR  2/500 rows (0.4%) have the prompt echoed verbatim inside the completion | min_prompt_len=40
overall: MINOR — 500/500 rows scanned
Error: cannot write --output: output 'lint_train.json' must stay under cwd
```

## `soup data lint data/faults.jsonl ...`: 2026-09-29T02:55:53Z (without datasketch)
```
length_bias     OK     effect size (Cohen's d) = 0.030; chosen is longer in 44.4% of rows | mean chosen=716.8, mean rejected=699.0
identical_pairs MAJOR  9/63 rows (14.3%) have chosen == rejected
near_duplicates OK     near-dup check skipped (datasketch not installed)
prompt_leak     MINOR  6/63 rows (9.5%) have the prompt echoed verbatim inside the completion | min_prompt_len=40
overall: MAJOR — 63/63 rows scanned
```

## Same, after `pip install datasketch`: 2026-09-29T02:59:54Z
```
length_bias     OK     d = 0.030; 44.4%
identical_pairs MAJOR  9/63 rows (14.3%)
near_duplicates OK     0/63 rows (0.0%) have a near-duplicate elsewhere in the dataset | threshold=0.85
prompt_leak     MINOR  6/63 rows (9.5%)
overall: MAJOR — 63/63 rows scanned
```

## `soup profile -c configs/main.yaml`: 2026-09-29T03:05:37Z
```
Params: 1.5B (trainable: 12,845,056 with LoRA r=16) | Quantization: none
GPU Memory Estimate: Model ~3.0 GB | LoRA ~0.0 GB | Optimizer ~0.1 GB | Activations (bs=2, seq=1024) ~0.9 GB | Overhead ~1.5 GB | Total ~5.5 GB
Speed Estimate: Tokens/sec ~1,250 | Samples/sec ~1.2
Recommendations: OK Fits in 15 GB VRAM | OK Recommended batch_size: 4
```

## `soup plan -c configs/main.yaml --state logs/soup.tfstate`: 2026-09-29T03:05:41Z
```
config_sha 9c9d820a94a16f54... | dataset_sha a75b3296d1c31f63... | estimated_cost $0.1000 | estimated_minutes 20.0 | peak_vram_gb 8.0 | spot_price_usd_per_hour $0.30
```

## `soup train -c configs/main.yaml --dry-run -y`: 2026-09-29T03:05:41Z
```
Device: Tesla T4 | Memory: 14.6 GB | Model: Qwen/Qwen2.5-1.5B-Instruct | Task: dpo | LoRA: r=16, alpha=32 | Quant: none
Dry run - validating data...
Data OK: 450 train samples
Val: 50 samples
Config valid. Ready to train!
```

## First training attempt (lr0), torchao crash: 2026-09-29T03:07:12Z
```
peak VRAM ~9.95 GB at batch 2 x seq 1024 (4 rows — chosen+rejected are one concatenated tensor) (logits 8.71 GB)
free VRAM 15.53 GB
forecast 178-262 tok/s — a compute-bound bound, not a promise (from 2.42 TFLOPS measured on this card now @ 1185 MHz)
Error: ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported
```

## Colab run chronology (from `run_remaining.log` and run folders)
- `20260929T030712Z_lr0`: torchao crash (above); only the config was saved.
- `20260929T031338Z_lr0`: step-0 meta-device crash, as shipped (`train.log`, `train_verbose.log`; crash bundles in `crashes/`).
- `20260929T031859Z_lr0`: complete, with the ADR 0005 workaround.
- `20260930T021628Z_main`: complete.
- `20260930T034012Z_swapped`: interrupted after step 1 (notebook cell restarted); superseded.
- `20260930T034258Z_swapped`: complete.
- Verification started 2026-09-30T05:05:48Z and was killed without a traceback when the Colab GPU runtime was lost. `20260930T052803Z_fp16` was then started by the script and never trained (config only). The free GPU quota was exhausted, so the fp16 and no-streaming comparisons were not run.
- Final verification ran on Apple MPS (local machine) against the same adapters: `logs/verify_*.log` / `.json`.
