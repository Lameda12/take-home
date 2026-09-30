# Take-Home: Soup AI Engineer, DPO + layer streaming on a free T4

Done by **Alamedin Sabit** ([asabitt29@gmail.com](mailto:asabitt29@gmail.com)).

**Verdict: SHIP this run, with conditions on the pipeline.** The trained adapter passes all four pre-registered checks against two control runs, but Soup 0.72.4 streaming DPO crashes as shipped and needed a one-line workaround. Full reasoning is in [`report/REPORT.md`](report/REPORT.md) (≤2 pages; the long version is in `docs/findings/report-long.md`).

## Deliverables

| Brief item | Where |
|---|---|
| Report (≤2 pages) | [`report/REPORT.md`](report/REPORT.md) |
| Raw timestamped logs | [`logs_raw/`](logs_raw): per-run `train.log`, `mem.jsonl` (per-step torch peaks), `peaks.json`; Soup `profile`/`plan`/`lint`/`doctor` outputs; crash bundles |
| Raw nvidia-smi output | [`logs_raw/nvidia_smi.csv`](logs_raw) (1 s sampling for the whole session) |
| Part 2 verification script | [`scripts/verify_training.py`](scripts/verify_training.py), with its output in [`logs/`](logs) |
| Notebook / code changes | [`scripts/`](scripts) and [`configs/`](configs); Soup itself is not modified (see ADR 0005 for the one runtime workaround) |
| Surprises and concerns | last section of the report |
| AI tool use | [`AI_USAGE.md`](AI_USAGE.md) |

## Supporting material

- [`docs/findings/`](docs/findings): silent-failure log and memory budget (the estimate was written **before** training; measurements were added after, without editing it)
- [`docs/adr/`](docs/adr): decisions, including the **pre-registered ship criteria** (0001) and the crash workaround (0005)
- [`docs/research/`](docs/research): source-level notes on soup-cli 0.72.4 / TRL 0.24 and the dataset statistics
- [`CONTEXT.md`](CONTEXT.md): glossary; [`docs/EXPLAINER.md`](docs/EXPLAINER.md): plain-language walkthrough
- [`adapters/`](adapters): the three trained LoRA adapters (main, lr≈0 control, swapped-labels control)

## Reproduce

Environment: Colab T4, Python 3.13, `pip install "soup-cli[train]"` (resolves to soup-cli 0.72.4, trl 0.24.0, transformers 4.57.6, peft 0.20.0), then `pip uninstall torchao` (see the report).

```bash
python scripts/prepare_data.py          # 500 train / 100 held-out; sha256 a75b3296… / d58b213a…
python scripts/make_faults.py           # 63 planted-fault rows for data-check testing
python scripts/swap_labels.py           # swapped-labels control data
soup data lint data/train.jsonl --model Qwen/Qwen2.5-1.5B-Instruct
LOG=<dir> bash scripts/run_remaining.sh # main -> swapped -> verify -> fp16 -> nostream
python scripts/soup_memlog.py --no-hf-grad-ckpt <mem.jsonl> train -c configs/lr0.yaml -y
python scripts/probe_ship_rule.py       # soup ship decision rule, offline
```

The data is not committed: `IlyaGusev/saiga_preferences` declares no license. `prepare_data.py` rebuilds the identical split (the hashes were checked on two machines).

Tests (CPU or Apple MPS, tiny random Qwen2 model): `pip install -r requirements-dev.txt && pytest tests` (37 tests).
