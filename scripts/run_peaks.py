"""Summarise peak memory for one run dir: torch peaks (mem.jsonl) + nvidia-smi peak in the run's time window."""
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from verify_training import read_jsonl

run_dir, smi_csv = Path(sys.argv[1]), Path(sys.argv[2])
rows = read_jsonl(run_dir / "mem.jsonl")
# first row is logged at the END of step 1: widen the window back by that step's duration
t0 = datetime.strptime(rows[0]["ts"][:19], "%Y-%m-%dT%H:%M:%S") - timedelta(seconds=rows[0]["elapsed_s"] + 5)
start = t0.strftime("%Y/%m/%d %H:%M:%S")
end = rows[-1]["ts"][:19].replace("-", "/").replace("T", " ")
smi = 0
with smi_csv.open() as f:
    for r in csv.reader(f):
        if len(r) > 1 and r[0][:4].isdigit() and start <= r[0].strip()[:19] <= end:
            smi = max(smi, int(r[1].strip().split()[0]))
steps = [r["elapsed_s"] for r in rows]
out = {
    "run": run_dir.name,
    "steps_logged": len(rows),
    "window_utc": [rows[0]["ts"], rows[-1]["ts"]],
    "sec_per_step": round((steps[-1] - steps[0]) / max(len(steps) - 1, 1), 2),
    "torch_max_allocated_gb": round(max(r["max_allocated"] or 0 for r in rows) / 1e9, 3),
    "torch_max_reserved_gb": round(max(r["max_reserved"] or 0 for r in rows) / 1e9, 3),
    "nvidia_smi_peak_mib": smi,
}
(run_dir / "peaks.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out))
