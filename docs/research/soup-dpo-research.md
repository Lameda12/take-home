# Soup DPO + layer streaming on a Colab T4: research notes

Scope: primary sources only. The versions are **pinned to what the user's Colab actually installed**: soup-cli 0.72.4, trl 0.24.0, transformers 4.57.6, peft 0.20.0, torch 2.11.0+cu128, bitsandbytes 0.50.2, Python 3.13.
Method: I pulled the wheels with `pip download soup-cli==0.72.4 trl==0.24.0 peft==0.20.0 transformers==4.57.6 --no-deps`, unpacked them and read the source. `file:line` citations point into those wheels. Paths are relative to the package root (`soup_cli/`, `trl/`, …).
For contrast I also read soup-cli 0.75.1 (the current release). Where I cite it, it is tagged **[0.75.1]**.
Anything I did not verify is tagged **UNVERIFIED**.

## Summary (highest-signal findings)

1. **Version trap.** soup-cli 0.72.4 is the last release that allows Python 3.13. Releases 0.73.0 through 0.75.1 declare `Requires-Python <3.13,>=3.10` (PyPI JSON for each release).
   The `[train]` extra of 0.72.4 pins `transformers<5.0.0` and `trl<0.25` (wheel METADATA). That is why pip downgraded transformers 5.16.1→4.57.6 and huggingface_hub.
   So the Colab runs *older* Soup code, and it lacks several fixes that later releases describe as silent-failure fixes (see #2, #3, #6).
2. **T4 runs bf16 under emulation, silently.** `trainer/dpo.py:160` sets `bf16=self.device == "cuda"`, so a T4 gets bf16 too. `trainer/stream_setup.py:122` hard-codes the streamed base to `dtype = "bfloat16"` on CUDA.
   A T4 is sm_75 and has no bf16 units. transformers gates bf16 on `torch.cuda.is_bf16_supported()` (`transformers/utils/import_utils.py:644`), and that call counts emulation as support.
   Soup's own later docstring **[0.75.1] `utils/gpu.py:376-397`** says: "a T4 answers True… on the current stack nothing raised; the run simply proceeded in an emulated dtype. Established by running it on a real T4." 0.75.1 fixes this by routing pre-Ampere cards to fp16 (`bf16_fp16_flags`, `resolve_frozen_base_load_dtype`).
   Consequences: tok/s, VRAM and numerics all differ from the forecasts, and there is no error.
3. **The reported VRAM is the inter-step trough, not the peak.** In 0.72.4, `monitoring/callback.py:163` shows `torch.cuda.memory_allocated()` (current). **[0.75.1] `monitoring/callback.py:175-184`** switched to `max_memory_allocated()` with the comment "rather than the inter-step trough (#650)".
   Neither value is `memory_reserved` and neither is nvidia-smi.
4. **"tok/s" is a forecast, not a measurement.** `trainer/stream_setup.py:~430-446` prints `forecast X–Y tok/s — a compute-bound bound, not a promise`. It is computed from a GEMM microbenchmark (`measure_gemm_tflops`) as TFLOPS/(6·P) (`utils/layer_stream.py`, `FLOPS_PER_PARAM_PER_TOKEN = 6`).
   The live display shows `train_steps_per_second` as "it/s" (`monitoring/callback.py:174`, `monitoring/display.py:115-116`). That is the HF Trainer key; I did not verify when it is populated during training (UNVERIFIED).
   The GEMM probe runs in the resolved stream dtype. On a T4 that is emulated bf16, so the forecast itself is distorted (UNVERIFIED for 0.72.4 specifically; the 0.75.1 probe uses `resolve_stream_dtype`).
5. **The reference model is the policy with its adapter disabled.** Soup passes an already-wrapped `PeftModel` and no `ref_model`. TRL 0.24 then sets `self.ref_model = None` (`trl/trainer/dpo_trainer.py:365-369`) and computes reference log-probs under `null_ref_context()` → `disable_adapter()` (`:915-927`, `:929-938`).
   There is no second copy of the weights. If the adapter is never actually engaged in the streamed forward, policy == reference, logits are identical, and the loss sits at ln 2 ≈ 0.693 with zero gradient.
6. **Soup's checks never look at training outcomes.**
   - `soup data doctor` **refuses** DPO data (exit 2, "is preference data — use soup data lint"; `commands/data_doctor.py:199`).
   - `soup data lint` runs 5 static checks (length_bias, label_imbalance, near_duplicates, identical_pairs, prompt_leak; `utils/data_lint.py:10-40`). None measures truncation, and none checks the chat template on preference data.
   - `soup ship` in 0.72.4 SHIPs iff `tuned > base` strictly (no noise floor; `utils/ship_verdict.py:247`) and the leg‑2 benchmarks did not drop by more than 0.05 (`:65`, `:140-146`). One extra exact-match hit is a SHIP.
   - No check verifies adapter delta, reward margin, or policy-vs-reference divergence.
7. **Plain-string rows get no chat template.** Soup's `_convert_dpo` passes `prompt/chosen/rejected` through untouched (`data/formats.py:190-204`). TRL applies the chat template only if a row is conversational (message lists; `trl/data_utils.py:79` `is_conversational`, used at `dpo_trainer.py:653-659`).
   A string-format dataset therefore trains on raw text with no role tokens. The model is then served with the chat template, so train and serve see different distributions. Nothing in Soup flags this for DPO, because doctor refuses preference data.

---

## Q1. soup-cli internals (0.72.4)

**Where the source lives.**
- PyPI `soup-cli`. Homepage and repo: https://github.com/MakazhanAlpamys/Soup (PyPI project_urls).
- Current release: 0.75.1, uploaded 2026‑09‑21. The Colab has 0.72.4, uploaded 2026‑08‑03, `soup_cli/__init__.py:3`.

**How layer streaming works** (`utils/layer_stream.py` docstring lines 1-18, `utils/layer_stream_runtime.py` docstring):
- The frozen base lives in CPU RAM, pinned when possible, or on disk (the NVMe-only "disk" tier).
- Decoder layers are copied one at a time into a pool of pre-allocated VRAM buffers (`stream_buffers`, default 2 = double buffering; schema `stream_buffers`).
- Embeddings and an untied lm_head share one large single-slot buffer.
- Only the LoRA adapters, their gradients and the optimizer state stay resident.
- Per step:
  - FORWARD layer i: `wait(i) → prefetch(i+1) → checkpoint(body_i)`.
  - BACKWARD layer i: `wait(i) → prefetch(i-1) → recompute + backward`.
  - So every layer is read twice per step (runtime docstring).
- Every layer is wrapped in `checkpoint(use_reentrant=False)`, so streaming always implies gradient checkpointing.
- Pre-flight (`trainer/stream_setup.py:100-330`):
  - An architecture allowlist: llama, qwen2, qwen3, mistral, gemma, gemma2, gemma3_text, phi, phi3 (`utils/layer_stream.py:74-84`).
  - A RAM-headroom check (store < 0.7 × free RAM).
  - Checkpoint sharding into `shard_dir`.
  - Then a VRAM fit check: predicted peak vs `torch.cuda.mem_get_info()[0]` free memory, raising if it does not fit (`stream_setup.py:~418-425`).
- For DPO the budget uses 2 rows per example (`trainer/dpo.py:28` `_STREAM_ROWS_PER_EXAMPLE = 2`), because chosen and rejected are concatenated.
- Schema constraints (`config/schema.py` `_validate_stream_layers_compat`):
  - tasks sft/dpo/orpo/simpo/kto; grpo/ppo are permanently refused
  - backend transformers, text only
  - quantization none|4bit
  - `batch_size` must be concrete, not `auto`
  - LoRA r ≥ 1
  - no DoRA/VeRA/PiSSA
  - no packing
- The schema comment claims DPO's reference "costs no extra weights: measured 0.914x SFT peak" (a Soup-internal measurement; UNVERIFIED by me).

**What the pre-flight checks do NOT cover:**
- numerics/dtype suitability for the card (the bf16-on-T4 problem)
- whether the adapter actually receives gradients
- the chat template
- truncation of the chosen/rejected difference
- any quality metric

**`soup data doctor`** (`utils/data_doctor.py:35-44`):
- Checks: chat_template, template_render, generation_markers, eos_in_labels, bos_duplication, system_role, unknown_roles, truncation_risk. The default sample is 200 rows.
- **It rejects `dpo`/`kto` formats** (`commands/data_doctor.py:199`).

**`soup data lint`** (`utils/data_lint.py`):
- **length_bias**: Cohen's d between chosen and rejected lengths. MAJOR ≥ 0.8, MINOR ≥ 0.3 (`:207-250`). Without `--model` the length is a whitespace word count (`commands/data_doctor.py:~300-313` fallback).
- **identical_pairs**: exact `strip()` equality only; near-identical pairs pass (`:333-353`).
- **near_duplicates**: MinHash.
- **prompt_leak**: needs at least 40 chars (`_MIN_PROMPT_LEAK_LEN`).
- **label_imbalance**: KTO only.
- Sample size is 2000 (`:48`).
- It does NOT check:
  - token-level truncation
  - the shared-prefix length of chosen vs rejected
  - language or encoding
  - the chat template for preference data
  - label noise
  - whether the "rejected" answer is actually worse

**`soup ship`** (`commands/ship.py` docstring):
- Leg 1 is the task win (`metric` = `eval/custom.run_eval` accuracy with exact/contains/regex/semantic scoring, `eval/custom.py:16,142-166`), or a judge score, or a pairwise score.
- Leg 2 is a general suite (bundled offline MCQ/arithmetic/tool/JSON/safety, or lm-eval).
- Rule, 0.72.4 `utils/ship_verdict.py:200-260`: SHIP ⇔ `tuned > base` (strict, **no noise floor in 0.72.4**; the floor was added later, [0.75.1] `ship_verdict.py:431`) AND leg‑2 is non-empty AND no benchmark dropped by more than 0.05 absolute.
- `--evidence` mode trusts a JSON of pre-computed scores and loads no model.
- It does NOT check:
  - statistical significance
  - whether the eval set overlaps the training set
  - DPO-specific signals (margins, KL)
  - adapter liveness
  - whether the adapter being evaluated is the one that was trained (0.72.4 has no provenance hash; [0.75.1] adds `_compute_provenance`/staleness)

**Defaults (0.72.4 schema / TRL 0.24):**

| item | value | source |
|---|---|---|
| base model | required, no default (`base: str = Field(...)`); the DPO recipes use Llama-3.1-8B-Instruct / Qwen2.5-7B-Instruct | `config/schema.py` SoupConfig; `recipes/catalog.py:83-110,314-` |
| LoRA r / alpha / dropout | 64 / 16 / 0.05 (DPO recipe overrides to 16/32) | schema LoraConfig; recipe |
| target_modules | `auto` → `None` → PEFT default mapping: llama & qwen2 = `["q_proj","v_proj"]` only | `trainer/dpo.py:238-240`, `stream_setup.py` same; `peft/utils/constants.py:85,103` |
| beta | 0.1 (`dpo_beta`) | schema |
| loss_type | not passed → TRL default `["sigmoid"]` | `trl/trainer/dpo_config.py:370` |
| precision | `bf16=True` on any CUDA device, including T4 | `trainer/dpo.py:160` |
| max_length / max_prompt_length | `data.max_length` (default 2048) / `max_length//2` | `dpo.py:166-167`; schema `max_length` |
| truncation_mode | TRL default `keep_end` (whole prompt+completion sequence, `dpo_trainer.py:1336-1352`); the prompt is left-truncated to `max_prompt_length` in `tokenize_row` (`:738-739`) | trl |
| max_completion_length | None | dpo_config.py:319 |
| lr | 2e-5 (schema default; the recipe uses 5e-6); cosine; warmup_ratio 0.03; AdamW torch; epochs 3; grad_accum 4; max_grad_norm 1.0 | schema |
| gradient_checkpointing | not passed → TRL 0.24 DPOConfig default **True** (`dpo_config.py:241-246`), stacked on top of streaming's own per-layer checkpoint. [0.75.1] `trainer/dpo.py:174-185` notes that the double checkpoint crashed on torch 2.13 (#328). Behaviour on torch 2.11: UNVERIFIED | |
| seed | not passed in 0.72.4 DPO → HF default 42 (UNVERIFIED path) | |
| reference | policy with the adapter disabled (`null_ref_context`), no second copy | `trl/trainer/dpo_trainer.py:365-369,915-927` |

**Logs and metrics:**
- TRL logs `rewards/chosen`, `rewards/rejected`, `rewards/accuracies`, `rewards/margins`, `logps/chosen`, `logps/rejected`, `logits/chosen`, `logits/rejected` (in trl 0.29.1 at `dpo_trainer.py:1105-1122`; that metric set also exists in 0.24 `get_batch_loss_metrics`; exact 0.24 lines UNVERIFIED).
- The Soup live display shows loss, lr, "Speed it/s" and "GPU used/total" (`monitoring/display.py:115-118`).
- The return value keeps only `initial_loss`, `final_loss`, `duration` and `total_steps` (`dpo.py:354-361`). **No margin or accuracy summary is surfaced.**
- Loss watchdog is off by default (schema `loss_watchdog=False`).

## Q2. T4 facts

- **Datasheet** (NVIDIA T4 datasheet MAR19, p.1):
  - Turing, 320 Tensor Cores, 2,560 CUDA cores
  - FP32 8.1 TFLOPS; mixed-precision FP16/FP32 65 TFLOPS; INT8 130 TOPS
  - **16 GB GDDR6, 300 GB/s**
  - PCIe Gen3 x16, 32 GB/s interconnect; 70 W; passive cooling
  - bf16 is not listed.
- **Compute capability 7.5** (sm_75): confirmed indirectly by Soup [0.75.1] `utils/gpu.py:379` ("A T4 (sm_75, Colab's free tier)"). The NVIDIA CUDA-GPUs page was not fetched (UNVERIFIED direct).
- **bf16**: there is no hardware support (bf16 needs Ampere sm_80+). `torch.cuda.is_bf16_supported()` defaults to `including_emulation=True` and returns True on a T4 ([0.75.1] `utils/gpu.py:384-408`, based on Soup's own T4 run). I did not read the torch source (UNVERIFIED direct).
- **fp16 pitfalls**:
  - max about 65504; overflow gives inf/NaN
  - fp16 needs a GradScaler; HF Trainer uses it when `fp16=True`
  - GradScaler **silently skips optimizer steps** whose grads are inf/NaN and lowers the scale, so a run can finish with many skipped steps and only a log line (torch AMP docs; UNVERIFIED fetch)
  - [0.75.1] `utils/mixed_precision.py:84-112` documents that GradScaler raises `…not implemented for 'BFloat16'` if the LoRA params are bf16 under fp16
  - In 0.72.4 fp16 is never selected, so the pitfall is instead emulated bf16.
- **FlashAttention-2**: Ampere/Ada/Hopper only. For "Turing GPUs (T4, RTX 2080), see the separate flash-attention-turing repo" (github.com/Dao-AILab/flash-attention README). Use SDPA or eager on a T4.
- **Allocator vs nvidia-smi** (PyTorch 2.11 CUDA semantics, "Memory management"):
  - The caching allocator keeps freed blocks, and "unused memory managed by the allocator will still show as if used in nvidia-smi".
  - `memory_allocated`/`max_memory_allocated` count tensors; `memory_reserved`/`max_memory_reserved` count everything the caching allocator holds.
  - nvidia-smi also includes the CUDA context. Soup notes the context and driver reservation sit outside the allocator (0.85 GB on their dev box; [0.75.1] `utils/layer_stream.py` docstring of the predict function). Soup also measured reserved at 1.08–1.41× allocated ([0.75.1] `layer_stream.py:~1295`).
  - Ordering: nvidia-smi ≥ reserved ≥ max_allocated ≥ allocated (the last value is what 0.72.4 displays).

## Q3. Preference datasets (ranked for ~500 pairs, Russian / support-style)

| # | dataset | size | license | format | fit |
|---|---|---|---|---|---|
| 1 | `IlyaGusev/saiga_preferences` | 30,590 train | **not stated in the card** (UNVERIFIED) | `prompt`/`chosen`/`rejected` as **message lists** (role/content) + `chosen_model`, `rejected_model`, `source`, `sonnet_approved`, `is_bad_by_regex` | Native Russian, conversational (so TRL applies the chat template), and it has quality flags to filter on (`sonnet_approved=True`, `is_bad_by_regex=False`). Best choice. |
| 2 | `eridai/russian_dpo_qa` | 9.8K | MIT | string `prompt`/`chosen`/`rejected` (5 cols) | Russian QA. Created 2026‑08, low downloads, provenance unclear (UNVERIFIED quality). Strings, so no chat template (see summary #7). |
| 3 | `d0rj/full-hh-rlhf-ru` | 112K train / 12.5K test | not stated | strings `prompt`,`response`,`chosen`,`rejected` | Machine-translated Anthropic HH. Assistant-style dialogue, noisy translation, and it has a real test split. |
| 4 | `stindardlogic/customer-support-dpo-100k` | 100K | Apache‑2.0 | 5 cols, synthetic | **English** support tickets across 23 scenarios. Synthetic; the "rejected" answers are likely strawmen (run a length-bias check). |
| 5 | `argilla/ultrafeedback-binarized-preferences-cleaned` | 60.9K | MIT | message lists | English, general. A well-known control/baseline. |

Sources: the HF Hub dataset pages and READMEs (the saiga schema comes from its README YAML). The row previews were rate-limited (429), so I have not inspected the text content of any row (UNVERIFIED).

## Q4. DPO silent failures and how to verify them

The DPO loss per pair, with the sigmoid loss (`trl` loss code; Rafailov et al. 2023), is
`-log σ(β[(log πθ(yw|x) - log πref(yw|x)) - (log πθ(yl|x) - log πref(yl|x))])`.
At step 0 the adapter's B=0, so πθ=πref and the loss is exactly **ln 2 = 0.6931**. `rewards/margins` starts at 0 and `rewards/accuracies` at 0 (or 0.5 with ties).

Each failure below lists the failure, the evidence I have, and the verification.

- **Adapter not attached, not engaged, or frozen.**
  - Evidence: under streaming, the weights live on the meta device and are substituted per layer. [0.75.1] `utils/layer_stream_runtime.py:1956-2000` shows a later bug where PEFT built an adapter on `meta` and TRL's `copy_` "is a no-op". That is a precedent for meta-skeleton adapter bugs. In 0.72.4 with TRL 0.24 there is no ref adapter, so that specific path is absent.
  - Verify:
    - a) Diff the saved `adapter_model.safetensors` against a fresh-init adapter. With B initialised at 0, ‖B‖ > 0 must hold for every targeted module.
    - b) Confirm `requires_grad` and a non-None `.grad` on lora_A/B after one step.
    - c) Compute per-module ‖ΔW‖=‖(α/r)·B·A‖.
- **Reference == policy.** Symptoms: loss stuck at 0.693 and margins at 0. Or the inverse: the adapter-disabled path still runs the adapter, so ref==policy for all time.
  - Verify: on held-out pairs, compute log πθ − log πref with the adapter on vs `disable_adapter()`. The difference must be non-zero and larger on the chosen side than the rejected side.
- **Falling loss without learning.** The loss drops if the rejected log-prob falls faster than the chosen one. Both can fall together; this is the known "likelihood displacement" (Razin et al. 2024, arXiv:2410.08847; UNVERIFIED fetch).
  - Watch `logps/chosen`: if it trends down, the model is getting worse at the preferred answers.
  - Length hacking: DPO favours longer outputs when chosen is systematically longer (Park et al. 2024, "Disentangling Length from Quality in DPO", arXiv:2403.19159; UNVERIFIED fetch). `soup data lint length_bias` catches only the data side.
  - Verify with a **held-out** reward margin and accuracy (train metrics can be memorised on 500 pairs) and the generated-length distribution before vs after.
- **Truncation erases the difference.**
  - With `max_prompt_length = max_length/2` and left truncation of the prompt, then `keep_end` on the full sequence (`dpo_trainer.py:738-741,1336-1352`), a long ticket loses its start.
  - If chosen and rejected share a long prefix and differ only after the cut, the pair has zero signal. Nothing in Soup checks this.
  - Verify: tokenise with the actual tokenizer and count pairs where `chosen_ids[:L] == rejected_ids[:L]` after the caps.
- **Identical or near-identical pairs.** lint catches exact equality only (`data_lint.py:333-353`). Verify with normalised or token-level equality and edit distance.
- **Chat template / tokenizer with Cyrillic.**
  - String rows mean no template (summary #7).
  - `tokenize_row` tokenises the prompt and the completion separately (`dpo_trainer.py:724-726`), which can split a BPE merge at the boundary. It appends EOS to chosen and rejected but adds no BOS for decoder-only models (`:728-735`).
  - `pad_token = eos_token` fallback (`dpo.py:208-209`).
  - Cyrillic token efficiency: Russian text uses more tokens per character in most English-centric BPEs, so truncation bites earlier (UNVERIFIED quantitatively; measure the tokens/char ratio).
  - Verify: decode a batch back to text and eyeball it; check for mojibake or U+FFFD.
- **fp16 NaN / skipped steps.** Not applicable to the 0.72.4 path (bf16 is always used). If a patched run uses fp16, count the GradScaler scale drops and skipped steps, and assert `isfinite` on the loss and grads every step.
- **Emulated bf16 on a T4.** This is silent: slow, and different numerics (summary #2). Verify by printing `trainer.args.bf16`, the dtype of the lora params, and the dtype of the streamed buffers.
- **Streaming/offload correctness.** Soup's allowlist exists because "a half-supported architecture streams weights into the wrong module and mis-trains silently" (`utils/layer_stream.py` comment above the allowlist).
  - Verify: logits parity streamed vs resident on the same small model and batch (max |Δ| within bf16/fp16 tolerance).
  - Also check that layer read counts per step equal 2×n_layers.
- **Sabotage and control runs** (the strongest proof):
  - a) **lr=0 run**: same pipeline. Loss must stay at 0.693, and the adapter delta must be 0.
  - b) **Swapped labels** (chosen↔rejected): held-out margin must go negative.
  - c) **Shuffled pairs** (random rejected): the train loss still drops, but held-out accuracy must stay near chance.
  - d) **Adapter-deleted eval**: the ship metric must revert to the base score.
  - A check that cannot tell the real run from (a) or (c) is not a check.
- **What these checks do not detect:** whether the preference labels match real support quality, and regressions outside the eval distribution.

## Q5. Memory budget inputs

Notation: P = base params, b = bytes per weight element (bf16/fp16 = 2, NF4 ≈ 0.516 incl. double-quant stats: `0.5 + 1/64 + 4/(64·256)`, `utils/layer_stream.py` NF4 constant), L = layers, h = hidden, i = intermediate (FFN), V = vocab, S = seq len (≤ max_length), B = per-device batch, R = 2B rows for DPO.

- **Weights, resident (no streaming):** P·b. **Weights, streamed:** `layer_bytes × stream_buffers + max(embed, untied lm_head) + non-decoder extras` (Soup `predict` function; [0.75.1] `layer_stream.py` docstring). layer_bytes ≈ (P_decoder/L)·b.
- **LoRA params:** for each targeted linear of shape d_in×d_out, r·(d_in + d_out). For q_proj+v_proj on a GQA model: per layer r·(h + h) + r·(h + kv_dim).
  Optimizer and adapter states: Soup charges **16 B/param** (fp32 weight 4 + grad 4 + Adam m 4 + v 4; `utils/layer_stream.py:100`).
- **Activations with checkpointing** (Soup `estimate_activation_bytes`, `layer_stream.py`): `R·S·( e·L·h  +  2·e·(h + i) )`, e = 2 bytes. The first term is the saved layer inputs; the second is the transient tensors inside the one layer being recomputed.
  This ignores attention scores. With SDPA memory-efficient attention there is no S² term; with eager attention add ≈ R·n_heads·S²·e (UNVERIFIED which attention implementation Soup picks on a T4).
- **Logits:** base tensor R·S·V·2 bytes (bf16). Peak through the loss: Soup 0.72.4 charges **14 B/element** = bf16 logits 2 + fp32 upcast 4 + log-softmax 4 + fp32 grad 4 (`utils/layer_stream.py:86-92`). [0.75.1] re-derives this as 12 + 2 retention.
  Because DPO uses R = 2B, **logits peak ≈ 14·2B·S·V**. TRL's `use_logits_to_keep` is False by default (`dpo_config.py:290`).
- **Reference pass:** no extra weights (adapter disabled). The pass runs under `torch.no_grad()` (`dpo_trainer.py:935`), so it adds one transient R·S·V logits tensor plus log-softmax work: about ≥ 2 + 4 bytes/element transient. It is not held across the policy backward if computed first (TRL computes the ref inside `get_batch_loss_metrics` before the policy forward; order UNVERIFIED for 0.24).
- **Fixed:** CUDA context plus cuBLAS workspace (outside the allocator; seen in nvidia-smi, not in `max_memory_allocated`). Soup reserves a DEFAULT_WORKSPACE of 1 GB in planning (`layer_stream.py:128`), plus a 13.5 MB slack.
- **Gap sources, predicted vs measured:** reserved vs allocated (1.08–1.41×), the CUDA context (~0.3–0.9 GB), actual S < max_length (dynamic padding lowers the real peak), and 0.72.4 displaying the trough (`memory_allocated`), not the peak.

Worked template: fill in P, L, h, i, V from the base model's `config.json`. The base model in the package config is unknown to me (UNVERIFIED).

## Open questions / UNVERIFIED

- Does double gradient checkpointing (TRL default GC=True plus the streamed per-layer checkpoint) crash, silently no-op, or behave correctly on torch 2.11? [0.75.1] reports a crash on 2.13.
- Does `disable_adapter()` actually bypass LoRA inside Soup 0.72.4's streamed `functional_call` layer wrapper? This needs the ref-vs-policy logits test.
- When does `train_steps_per_second` appear in the logs mid-training? Is "Speed" 0 until the end?
- The torch `is_bf16_supported` emulation branch on a T4: sourced only from Soup's own docstring, not from the torch source.
- The GradScaler skip behaviour and the length-bias and likelihood-displacement papers are cited from memory; I did not fetch them.
- The licenses of saiga_preferences and full-hh-rlhf-ru are unstated. I have not inspected any dataset row content.
- The exact TRL 0.24 metric-logging line numbers and the ref/policy forward order.
- Whether the Colab config sets `stream_layers: true` with `quantization: 4bit` (NF4) or `none`, and what the base model is.

## Sources

- PyPI soup-cli JSON: https://pypi.org/pypi/soup-cli/json, …/0.72.4/json, …/0.75.1/json
- soup-cli 0.72.4 wheel (files cited above); soup-cli 0.75.1 wheel for the [0.75.1] tags; repo https://github.com/MakazhanAlpamys/Soup
- trl 0.24.0 wheel: `trl/trainer/dpo_trainer.py`, `trl/trainer/dpo_config.py`, `trl/data_utils.py`; trl 0.29.1 for contrast
- peft 0.20.0 wheel `peft/utils/constants.py`
- transformers 4.57.6 wheel `transformers/utils/import_utils.py:644`, `training_args.py:1740-1743`
- NVIDIA T4 datasheet: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
- FlashAttention README: https://github.com/Dao-AILab/flash-attention
- PyTorch 2.11 CUDA semantics: https://docs.pytorch.org/docs/2.11/notes/cuda.html
- HF datasets: https://hf.co/datasets/IlyaGusev/saiga_preferences, https://hf.co/datasets/eridai/russian_dpo_qa, https://hf.co/datasets/d0rj/full-hh-rlhf-ru, https://hf.co/datasets/stindardlogic/customer-support-dpo-100k, https://hf.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned
