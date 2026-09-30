# ADR 0002: Base model and data

- Status: Accepted
- Date: 2026-09-29

## Context

Streaming keeps the whole frozen base model in CPU RAM, and Colab has 13 GB. A 7B model in bf16 (about 14 GB) does not fit. Streaming with 4-bit weights has a documented LoRA-class mismatch (`layer_stream_runtime.py:900-904`). Every bundled DPO recipe is 7B or larger and uses `batch_size: auto`, which the streaming validator refuses. The brief allows an open-source dataset in place of the private ticket package.

## Decision

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`, with `quantization: none` (about 3.1 GB of bf16 in RAM, 28 layers, vocabulary 151,936).
- **Main data:** `IlyaGusev/saiga_preferences` (Russian, message-list format, so TRL applies the chat template). Filtered to about 600 rows: 500 train and 100 held-out, split with a fixed seed. The dataset card states no license; the report says so.
- **Planted-fault set:** about 60 rows produced by a reproducible script (`scripts/make_faults.py`) that mutates real rows. Seven fault types:
  1. exact duplicate
  2. near-duplicate (whitespace or punctuation)
  3. long shared start, so truncation hides the difference
  4. length bias
  5. mixed Latin/Cyrillic look-alike letters
  6. empty answer
  7. prompt copied into the answer

  Used only to test `soup data lint`, `soup data doctor` and our own checks.
- **LoRA:** `target_modules: [q_proj, v_proj]`, set explicitly (equal to `auto`, the shipped default), r=16, alpha=32.
- **Hyperparameters:** lr 2e-5 (Soup's default), 2 epochs, beta 0.1, sigmoid loss, `stream_buffers: 2`. `max_length: 1024` (Soup caps the prompt at 512). `batch_size: 2`, `gradient_accumulation_steps: 4` (effective batch 8, about 125 optimizer steps). If Soup's pre-flight check refuses this, fall back to batch 1 × accumulation 8 and record the refusal.
- **Row filters** (`docs/research/saiga-stats.md`): keep rows with `sonnet_approved` and not `is_bad_by_regex`, a single user turn, chosen ≠ rejected, and that fit at 1024 without truncation. Drop the `pippa_and_multiturn_10_10` roleplay source. Take a seeded sample of 600 and split it 500 / 100.
- **Length bias is kept, not corrected.** Chosen is longer in about 71% of the filtered rows. It is measured with `soup data lint` and by comparing response length before and after tuning, and reported as a potential silent failure.

## Consequences

- saiga is general chat, not support tickets. The report states this limits domain claims.
- Using q/v only keeps the setup as shipped. Covering all linear layers is listed as a possible improvement, not tested.
