"""Planted-fault set: mutate real pairs so each row carries one labelled, known defect (ADR 0002).

Used only to test what `soup data lint` / `soup data doctor` and our own checks catch.
Never trained on.
"""
import argparse
import copy
import json
import random
from pathlib import Path

from verify_training import read_jsonl

FAULTS = (
    "exact_duplicate",
    "near_duplicate",
    "shared_prefix_truncation",
    "length_bias",
    "homoglyph",
    "empty_completion",
    "prompt_echo",
)

# Cyrillic -> visually identical Latin
HOMOGLYPHS = {"а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x"}
FILLER = "Спасибо за обращение в службу поддержки, мы внимательно изучили ваш вопрос. "


def _set(row, side, text):
    row[side] = [{"role": "assistant", "content": text}]


def _get(row, side):
    return row[side][-1]["content"]


def make_fault(row, fault, rng) -> dict:
    out = copy.deepcopy(row)
    chosen, rejected = _get(row, "chosen"), _get(row, "rejected")
    if fault == "exact_duplicate":
        _set(out, "rejected", chosen)
    elif fault == "near_duplicate":
        _set(out, "rejected", "  " + chosen.replace(" ", "  ").rstrip(".") + "!!")
    elif fault == "shared_prefix_truncation":
        prefix = FILLER * 60  # ~4.5k chars, past a 512-token completion budget
        _set(out, "chosen", prefix + chosen)
        _set(out, "rejected", prefix + rejected)
    elif fault == "length_bias":
        _set(out, "chosen", chosen + " " + FILLER * 8)
    elif fault == "homoglyph":
        _set(out, "chosen", "".join(HOMOGLYPHS.get(ch, ch) if rng.random() < 0.5 else ch for ch in chosen))
        if _get(out, "chosen") == chosen:  # guarantee at least one swap
            _set(out, "chosen", "".join(HOMOGLYPHS.get(ch, ch) for ch in chosen))
    elif fault == "empty_completion":
        _set(out, "chosen", " ")
    elif fault == "prompt_echo":
        _set(out, "chosen", row["prompt"][-1]["content"] + "\n" + chosen)
    else:
        raise ValueError(f"unknown fault {fault!r}")
    out["fault"] = fault
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="data/train.jsonl")
    ap.add_argument("--out", default="data/faults.jsonl")
    ap.add_argument("--per-fault", type=int, default=9)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    rows = read_jsonl(args.src)
    picked = rng.sample(rows, args.per_fault * len(FAULTS))
    faults = [make_fault(r, FAULTS[i % len(FAULTS)], rng) for i, r in enumerate(picked)]
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in faults), encoding="utf-8")
    print(f"wrote {len(faults)} rows ({args.per_fault} per fault x {len(FAULTS)}) to {args.out}")


if __name__ == "__main__":
    main()
