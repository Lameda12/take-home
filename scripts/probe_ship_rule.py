"""Probe soup ship's decision rule offline (no model load) with hand-made evidence.

Shows what score changes flip SHIP / DON'T SHIP in soup 0.72.4.
Run: python scripts/probe_ship_rule.py
"""
from soup_cli.commands.ship import _verdict_from_evidence

CASES = {
    "+0.01 task, no general benchmarks": {"task": {"mode": "metric", "base": 0.50, "tuned": 0.51}},
    "0.00 -> 0.01 task (one lucky answer in 100), benchmark flat": {
        "task": {"mode": "metric", "base": 0.0, "tuned": 0.01},
        "benchmarks": {"mcq": {"base": 0.6, "tuned": 0.6}},
    },
    "0.500 -> 0.501 task, benchmark -0.049": {
        "task": {"mode": "metric", "base": 0.5, "tuned": 0.501},
        "benchmarks": {"mcq": {"base": 0.6, "tuned": 0.551}},
    },
    "0.50 -> 0.51 task, benchmark -0.051": {
        "task": {"mode": "metric", "base": 0.50, "tuned": 0.51},
        "benchmarks": {"mcq": {"base": 0.6, "tuned": 0.549}},
    },
    "tuned == base, benchmark flat": {
        "task": {"mode": "metric", "base": 0.5, "tuned": 0.5},
        "benchmarks": {"mcq": {"base": 0.6, "tuned": 0.6}},
    },
}

for name, evidence in CASES.items():
    v = _verdict_from_evidence(evidence, forgetting_threshold=0.05)
    print(f"{v.decision:11s} failed_rule={v.failed_rule!s:18s} {name}")
