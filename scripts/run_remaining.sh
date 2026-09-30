#!/usr/bin/env bash
# Runs everything left after the lr0 control, in priority order (ADR 0004 amendment):
#   main -> swapped -> verify (verdict possible from here) -> fp16 -> nostream
# Usage (Colab, from the repo root):  LOG=/content/drive/MyDrive/soup-takehome/logs bash scripts/run_remaining.sh
# Re-running skips runs whose runs/<name>/adapter_model.safetensors already exists.
set -uo pipefail
: "${LOG:?set LOG to the Drive logs dir}"
RUNS="$LOG/../runs"
mkdir -p "$RUNS"
stamp() { date -u +%Y%m%dT%H%M%SZ; }

pgrep -f "nvidia-smi --query" > /dev/null || \
  (nohup nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 1 >> "$LOG/nvidia_smi.csv" 2>&1 &)

[ -f data/train_swapped.jsonl ] || python scripts/swap_labels.py
[ -f data/train_small.jsonl ] || head -n 80 data/train.jsonl > data/train_small.jsonl

run() {  # run <name> <config> [extra flag for soup_memlog]
  local name=$1 cfg=$2 flag=${3:-}
  if [ -f "runs/$name/adapter_model.safetensors" ]; then echo "== skip $name (done)"; return 0; fi
  local dir="$RUNS/$(stamp)_$name"; mkdir -p "$dir"; cp "$cfg" "$dir/"
  echo "== $(date -u +%FT%TZ) start $name -> $dir"
  (date -u +%FT%TZ; python scripts/soup_memlog.py $flag "$dir/mem.jsonl" train -c "$cfg" -y; echo "exit=$?"; date -u +%FT%TZ) 2>&1 | tee "$dir/train.log"
  [ -d "runs/$name" ] && cp -r "runs/$name" "$dir/output"
  [ -f "$dir/mem.jsonl" ] && python scripts/run_peaks.py "$dir" "$LOG/nvidia_smi.csv" | tee -a "$LOG/peaks.jsonl"
  cp -r .soup-crashes "$LOG/" 2>/dev/null || true
}

run main    configs/main.yaml    --no-hf-grad-ckpt
run swapped configs/swapped.yaml --no-hf-grad-ckpt

echo "== $(date -u +%FT%TZ) verify"
python scripts/verify_training.py --base Qwen/Qwen2.5-1.5B-Instruct \
  --main runs/main --lr0 runs/lr0 --swapped runs/swapped \
  --heldout data/heldout.jsonl --beta 0.1 --max-new-tokens 256 --gen-pairs 30 \
  --out "$LOG/verify_$(stamp).json" 2>&1 | tee "$LOG/verify_$(stamp).log"

run fp16     configs/fp16.yaml     --no-hf-grad-ckpt
run nostream configs/nostream.yaml            # resident path, TRL/HF checkpointing as shipped
echo "== $(date -u +%FT%TZ) all done"
