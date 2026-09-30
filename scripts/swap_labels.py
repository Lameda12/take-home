"""Swapped-labels control data: chosen <-> rejected on every row (ADR 0004)."""
import json
import sys
from pathlib import Path

from verify_training import read_jsonl

src, dst = (sys.argv[1:3] + ["data/train.jsonl", "data/train_swapped.jsonl"][len(sys.argv[1:3]):])
rows = [{**r, "chosen": r["rejected"], "rejected": r["chosen"]} for r in read_jsonl(src)]
Path(dst).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print(f"wrote {len(rows)} swapped rows to {dst}")
