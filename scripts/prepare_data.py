"""Build the 500 train / 100 held-out DPO split from IlyaGusev/saiga_preferences (ADR 0002).

Filters: sonnet_approved and not is_bad_by_regex, one user turn (system prompt allowed),
chosen != rejected, no roleplay source, and no truncation at max_length with Soup's
prompt cap of max_length // 2. Lengths are measured with the training model's chat
template, the same way TRL splits prompt and completion.
"""
import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

DROP_SOURCES = {"pippa_and_multiturn_10_10"}
KEEP_FIELDS = ("prompt", "chosen", "rejected", "source", "chosen_model", "rejected_model")


def keep_row(row, lens, max_length) -> bool:
    if not row.get("sonnet_approved") or row.get("is_bad_by_regex"):
        return False
    if row.get("source") in DROP_SOURCES:
        return False
    roles = [m["role"] for m in row["prompt"]]
    if roles.count("user") != 1 or "assistant" in roles:
        return False
    if row["chosen"] == row["rejected"]:
        return False
    if lens["prompt"] > max_length // 2:
        return False
    return lens["prompt"] + max(lens["chosen"], lens["rejected"]) <= max_length


def token_lengths(tokenizer, row) -> dict:
    prompt = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=True)
    lens = {"prompt": len(prompt)}
    for side in ("chosen", "rejected"):
        full = tokenizer.apply_chat_template(row["prompt"] + row[side], tokenize=True)
        lens[side] = len(full) - len(prompt)
    return lens


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dataset", default="IlyaGusev/saiga_preferences")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-heldout", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args(argv)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset(args.dataset, split="train")
    rng = random.Random(args.seed)
    order = list(range(len(ds)))
    rng.shuffle(order)

    need = args.n_train + args.n_heldout
    kept, scanned = [], 0
    for i in order:  # shuffle first, then filter until we have enough: avoids tokenizing all 30k rows
        scanned += 1
        row = ds[i]
        if not row.get("sonnet_approved") or row.get("is_bad_by_regex"):
            continue  # cheap filters first; tokenizing is the slow part
        if keep_row(row, token_lengths(tok, row), args.max_length):
            kept.append({k: row[k] for k in KEEP_FIELDS if k in row} | {"saiga_index": i})
            if len(kept) == need:
                break
    if len(kept) < need:
        raise SystemExit(f"only {len(kept)} rows passed the filters, need {need}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": kept[: args.n_train], "heldout": kept[args.n_train :]}
    for name, rows in splits.items():
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "args": vars(args),
        "rows_scanned": scanned,
        "pass_rate": round(len(kept) / scanned, 4),
        "sha256_16": {n: _sha(out / f"{n}.jsonl") for n in splits},
        "chosen_longer_frac": {
            n: round(sum(len(r["chosen"][-1]["content"]) > len(r["rejected"][-1]["content"]) for r in rows) / len(rows), 3)
            for n, rows in splits.items()
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
