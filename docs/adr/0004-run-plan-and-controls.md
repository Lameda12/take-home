# ADR 0004: Run plan, controls, measurement and logging

- Status: Accepted
- Date: 2026-09-29

## Decision

### Run order (step 5 is dropped first if the GPU budget runs out)

1. **Before training:**
   - run `soup data lint` on the main data and on the planted-fault set
   - run `soup data doctor`, and record that it refuses DPO data
   - write the hand memory estimate
   - run `soup profile` and `soup plan`
2. **lr=0 control**, about 50 steps. This is the cheapest check that the pipeline and the verification checks work.
3. **Main run**, full length.
4. **Swapped-labels control**, about 100 steps.
5. **Short fp16 comparison** (ADR 0003) and a **short run without streaming**, both about 50–100 steps.
6. **After training:**
   - `scripts/verify_training.py` (checks A–D, independent of Soup: plain PEFT + transformers)
   - `soup ship`, with a held-out preference win-rate as the task eval and the default general suite, recording that it is not Russian
   - `soup diff` on 20 Russian prompts, as a spot check for forgetting

### Measuring memory

- A custom callback logs `max_memory_allocated`, `max_memory_reserved` and `mem_get_info` per step to JSONL, after `reset_peak_memory_stats()` at the start. If Soup does not accept custom callbacks, a documented monkeypatch or a same-config TRL profiling run is used instead.
- The report compares four figures: the hand estimate, Soup's pre-flight estimate, torch's peak and `nvidia-smi` (a 1 s CSV logger running for the whole session). The gaps are explained: CUDA context, allocator cache, fragmentation, and Soup's point-in-time `memory_allocated`.

### Logging and durability

- Everything goes to Google Drive as it runs. Each run has its own folder `runs/<UTC-timestamp>_<name>/` holding:
  - the config
  - `soup train` stdout/stderr through `tee`
  - the memory JSONL
  - checkpoints (`save_steps: 25`)
- The session-wide `nvidia-smi` CSV and the environment capture are in `logs/`.
- Failed or odd runs are kept and reported.

### Buffers

`stream_buffers: 2` for all runs. A 1.5B decoder layer is about 110 MB, which is negligible next to the logits term (several GB at vocabulary 151,936), so no buffer sweep.

## Consequences

The verdict (ADR 0001) needs runs 2–4 at a minimum. With runs 1–4 alone, the report can still give a verdict.

## Amendment (2026-09-29, after the lr0 control, before the main run)

The lr0 control measured **80.8 s per optimizer step** (57 steps in 1 h 16 m). Soup forecast 172–253 tok/s; the real tok/s is still to be computed from the token counts. The planned 2-epoch main run (≈113 steps) would take about 2.5 h, which doesn't fit the free-Colab budget together with the controls. Changes:
- Main run: **1 epoch (57 steps)**, the same number of steps as both controls, which makes check D a like-for-like comparison.
- fp16 and no-streaming comparisons: **9 steps each** on an 80-row subset (`data/train_small.jsonl`), because Soup has no `max_steps`. They measure speed and memory only, not learning.
- Order: main → swapped → verify → fp16 → nostream (`scripts/run_remaining.sh`). A verdict is possible as soon as verify finishes.
- The no-streaming run does **not** get `--no-hf-grad-ckpt`: it uses the resident path as shipped.
