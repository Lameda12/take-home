# AI Tool Usage Log

The brief asks where AI tools were used, which suggestions were accepted, and what was verified, changed or redone by hand. Entries are chronological.

| # | Date | Tool | Task | What it produced | Accepted? | Verified / changed / redone by me |
|---|------|------|------|------------------|-----------|-----------------------------------|
| 1 | 2026-09-28 | Claude Code (Opus 5.5) | Picked a workflow for the take-home | Suggested order: research → plan → prototype → verification script (test-first) → silent-failure debugging → review | Partly | |
| 2 | 2026-09-28 | Claude Code background research agent | Research on soup-cli internals, T4 limits, preference datasets, DPO silent failures and memory formulas | `docs/research/soup-dpo-research.md`: claims cite file:line in soup-cli 0.72.4 and trl 0.24.0 source; the agent marked some claims UNVERIFIED | Pending: each claim must be checked on Colab | TODO: confirm bf16 on T4, VRAM metric, disable_adapter under streaming, target_modules |
| 3 | 2026-09-29 | Claude Code (grilling skill) | Worked through the plan decisions with me | I answered each question; decisions recorded in `CONTEXT.md` and `docs/adr/0001-0004` | Mostly the recommendations | I made every decision myself |
| 4 | 2026-09-29 | Claude Code background agent | saiga_preferences length and quality stats with the Qwen tokenizer | `docs/research/saiga-stats.md` (includes the script) | Pending | TODO: rerun the script myself on Colab and spot-check 10 rows |
| 5 | 2026-09-29 | Claude Code | Diagnosed the step-0 crash from the traceback and the TRL/Soup source; wrote the `--no-hf-grad-ckpt` workaround (ADR 0005), `run_remaining.sh` and `run_peaks.py` | Root cause with file:line references, the workaround and scripts | Yes | I confirmed the TRL default by running it; the lr0 run confirmed gradients flow (grad_norm about 5) |

## Where the AI tools were unreliable

<!-- Log every wrong claim, broken code or source that didn't check out, and how you caught it. -->

- 2026-09-29: Claude said Soup's VRAM pre-flight probably ignored DPO's doubled batch (chosen + rejected). Reading `trainer/stream_setup.py` (`_stream_budget_lines`, `rows = batch * _STREAM_ROWS_PER_EXAMPLE`) showed the claim was wrong: the estimate does double the rows. It was a guess made before reading the code. Lesson: read the source before claiming anything.
- 2026-09-29: the JSONL readers Claude wrote used `str.splitlines()`. That crashed on real saiga rows on Colab (`JSONDecodeError: Unterminated string`) because splitlines also breaks on U+2028/U+2029 inside strings. The unit tests had passed because the test data was ASCII. Fixed with a regression test (commit 910f90f). Lesson: the test data didn't look like the real data.
- 2026-09-30: Claude's first explanation of the memory over-estimate ("TRL pads to the longest sequence in a batch, not max_length") was wrong. Computing the token lengths showed batches of about 1024 tokens occur, so the gap is in the per-element logits cost (about 12 B vs 14 B). Corrected in `memory-budget.md`. Lesson: check an explanation against the data before writing it down.
- 2026-09-30: the verifier Claude wrote loaded the fp32 model on the CPU before moving it to the GPU. On Colab (12.7 GB of RAM) the process died with no traceback about 2 minutes in, most likely killed for running out of memory (not proven: no dmesg access). Fixed by loading straight onto the GPU and freeing memory between runs; progress lines added so a silent death is visible. Lesson: the tests ran a tiny model on a laptop and never exercised real memory limits.
- 2026-09-30: Claude drafted the report's verdict as DON'T SHIP before verification ran, predicting check C would fail. Verification passed all pre-registered checks. I kept the pre-registered SHIP instead of the draft, and the report says so.

## Summary (for report)

I used Claude Code throughout: to research Soup's source, plan, write the scripts test-first, and diagnose the crash. I made every decision myself (ADRs 0001–0005). I verified its claims against the source or the running system, and 5 of them turned out wrong (listed above). The most useful pattern was treating every AI explanation as a hypothesis until a log, a test or the source code confirmed it.
