# CONTEXT

Glossary and ground rules for the Soup DPO take-home. The brief is in `take-home.md`, the source research in `docs/research/`, and the decisions in `docs/adr/`.

## Glossary

- **Policy**: the model being trained: the base model plus the trainable LoRA adapter.
- **Reference model**: the frozen model DPO measures against. In TRL 0.24, and so in Soup, this is the *same* base model with the adapter switched off (`null_ref_context` / `disable_adapter()`), not a second copy. If the adapter never affects the forward pass, policy and reference are identical and the loss stays at ln 2 ≈ 0.693.
- **Adapter**: the LoRA weights (`lora_A`, `lora_B`) on `q_proj` and `v_proj`. `lora_B` starts at zero, so an untrained adapter changes nothing.
- **Adapter delta**: how far the adapter weights moved from their starting values. For `lora_B` that is simply its norm, because it starts at zero.
- **Layer streaming**: Soup's beta feature (v0.72). The frozen base weights stay in CPU RAM and are copied into a small pool of VRAM buffers one decoder layer at a time. Only the adapters, their gradients and the optimizer state stay on the GPU.
- **Meta materialization**: under streaming, PEFT creates the adapters on the `meta` device (no memory). `materialize_meta_adapters` then replaces them with *new* `nn.Parameter` objects. If this happens after the optimizer is built, the optimizer holds stale parameters.
- **Reward margin**: DPO's implicit reward for chosen minus rejected, β·(Δlogp_chosen − Δlogp_rejected), where each Δ is policy minus reference.
- **Held-out pairs**: preference pairs never seen during training (100 rows), used for checks C and D.
- **Control run**: a deliberately broken short run (lr=0, or chosen and rejected swapped) that the verification checks must flag as a failure.
- **Planted-fault set**: about 60 rows made by mutating real pairs with a script, each labelled with the fault it carries. Used only to test the data checks, never for training.
- **Silent failure**: a run that completes, reports normal metrics and still produces a wrong or unchanged model.
- **As shipped**: Soup 0.72.4 with no patches. Every deviation from it is logged as its own experiment.
- **Peak VRAM (three sources)**: Soup's reported `memory_allocated` (a point value), torch's `max_memory_allocated` / `max_memory_reserved`, and `nvidia-smi` (which adds the CUDA context and the allocator cache).

## Environment (pinned)

Colab T4 (14.6 GB usable, compute capability 7.5, no native bf16), 13 GB RAM, Python 3.13.
soup-cli 0.72.4, trl 0.24.0, transformers 4.57.6, peft 0.20.0, torch 2.11.0+cu128, bitsandbytes 0.50.2.
soup-cli 0.72.4 is the last release that allows Python 3.13. Installing it downgraded transformers (5.16.1 → 4.57.6) and huggingface_hub (1.29 → 0.36.2).

## Ground rules

- Failed or odd runs are kept, never deleted or rerun silently.
- All logs have UTC timestamps and are written to Drive as they happen.
- Every use of an AI tool goes in `AI_USAGE.md`.
- The ship criteria (ADR 0001) were fixed before any results.
