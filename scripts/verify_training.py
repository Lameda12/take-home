"""Independent verification that a Soup DPO run actually trained (ADR 0001).

Uses only PEFT + transformers + safetensors, never Soup's own code, so a bug in
Soup cannot hide itself from these checks.
"""
import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file


def read_jsonl(path) -> list:
    """Split on "\n" only. str.splitlines() also breaks on U+2028/U+2029/U+0085,
    which json.dumps(ensure_ascii=False) leaves raw inside strings."""
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


@dataclass
class CheckResult:
    name: str
    passed: bool
    reason: str
    details: dict = field(default_factory=dict)


def check_adapter_trained(adapter_dir, min_norm=1e-4) -> CheckResult:
    """Check A: lora_B must have moved meaningfully away from its zero init.

    Non-zero is not enough: Soup forbids lr=0, and an lr=1e-10 control leaves
    lora_B at ~1e-10. Fails if any lora_B is exactly zero or all are below min_norm.
    """
    tensors = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    norms = {k: float(v.float().norm()) for k, v in tensors.items() if "lora_B" in k}
    if not norms:
        return CheckResult("A_adapter_trained", False, "no lora_B tensors in adapter", {"lora_B_norms": norms})
    zero = [k for k, n in norms.items() if n == 0.0]
    if zero:
        return CheckResult(
            "A_adapter_trained",
            False,
            f"{len(zero)}/{len(norms)} lora_B tensors are exactly zero (untrained)",
            {"lora_B_norms": norms, "zero": zero},
        )
    if max(norms.values()) < min_norm:
        return CheckResult(
            "A_adapter_trained",
            False,
            f"all {len(norms)} lora_B norms below {min_norm:g} (max {max(norms.values()):.3g}): moved, not trained",
            {"lora_B_norms": norms},
        )
    return CheckResult(
        "A_adapter_trained", True, f"all {len(norms)} lora_B tensors non-zero, max norm {max(norms.values()):.3g}", {"lora_B_norms": norms}
    )


def _load_peft(base_dir, adapter_dir):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    # Load straight onto the GPU: an fp32 1.5B model staged in CPU RAM first (~6 GB plus a
    # conversion copy) was silently OOM-killed on Colab's 12.7 GB host.
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    base = AutoModelForCausalLM.from_pretrained(
        str(base_dir), dtype=torch.float32, device_map=device, low_cpu_mem_usage=True
    )
    return PeftModel.from_pretrained(base, str(adapter_dir)).eval()


def _free() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_roundtrip(base_dir, adapter_dir, prompt_ids) -> CheckResult:
    """Check B: the saved adapter loads into a plain HF model and changes its logits.

    Counts loaded tensors ourselves instead of trusting PEFT, which only warns
    on missing keys and silently returns the untuned base.
    """
    file_tensors = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    model = _load_peft(base_dir, adapter_dir)
    params = dict(model.named_parameters())

    loaded, unmatched = 0, []
    for key, tensor in file_tensors.items():
        name = key.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight"
        )
        param = params.get(name)
        if param is not None and torch.equal(param.detach().cpu().to(tensor.dtype), tensor):
            loaded += 1
        else:
            unmatched.append(key)

    max_diff = 0.0
    with torch.no_grad():
        for ids in prompt_ids:
            x = torch.tensor([ids], device=model.device)
            tuned = model(input_ids=x).logits
            with model.disable_adapter():
                base = model(input_ids=x).logits
            max_diff = max(max_diff, float((tuned - base).abs().max()))

    details = {"loaded": loaded, "total": len(file_tensors), "unmatched_keys": unmatched, "max_logit_diff": max_diff}
    if unmatched:
        return CheckResult("B_roundtrip", False, f"{len(unmatched)}/{len(file_tensors)} adapter tensors did not load", details)
    if max_diff == 0.0:
        return CheckResult("B_roundtrip", False, "adapter loaded but logits identical to base", details)
    return CheckResult("B_roundtrip", True, f"all {loaded} tensors loaded; max logit diff {max_diff:.3g}", details)


def _completion_logp(model, prompt_ids, completion_ids) -> float:
    """Sum of log p(completion | prompt), the quantity DPO's implicit reward uses."""
    ids = torch.tensor([prompt_ids + completion_ids], device=model.device)
    logits = model(input_ids=ids).logits[0, :-1].float()
    logps = torch.log_softmax(logits, dim=-1)
    targets = ids[0, 1:]
    token_logps = logps.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(token_logps[len(prompt_ids) - 1 :].sum())


def _gen_len(model, prompt_ids, max_new_tokens, eos_id) -> int:
    out = model.generate(
        input_ids=torch.tensor([prompt_ids], device=model.device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
    )
    return out.shape[1] - len(prompt_ids)


def heldout_metrics(base_dir, adapter_dir, pairs, beta, max_new_tokens=256, eos_id=2, gen_limit=None) -> dict:
    """Check C inputs: DPO implicit reward on held-out pairs, policy vs adapter-off reference.

    Reports d-logp for chosen and rejected separately so "both pushed down,
    rejected a bit more" is visible instead of hidden inside a positive margin.
    """
    model = _load_peft(base_dir, adapter_dir)
    margins, d_chosen, d_rejected, len_tuned, len_base = [], [], [], [], []
    with torch.no_grad():
        for i, p in enumerate(pairs):
            gen = gen_limit is None or i < gen_limit
            pc = _completion_logp(model, p["prompt_ids"], p["chosen_ids"])
            pr = _completion_logp(model, p["prompt_ids"], p["rejected_ids"])
            if gen:
                len_tuned.append(_gen_len(model, p["prompt_ids"], max_new_tokens, eos_id))
            with model.disable_adapter():
                rc = _completion_logp(model, p["prompt_ids"], p["chosen_ids"])
                rr = _completion_logp(model, p["prompt_ids"], p["rejected_ids"])
                if gen:
                    len_base.append(_gen_len(model, p["prompt_ids"], max_new_tokens, eos_id))
            d_chosen.append(pc - rc)
            d_rejected.append(pr - rr)
            margins.append(beta * ((pc - rc) - (pr - rr)))

    n = len(pairs)
    return {
        "n": n,
        "beta": beta,
        "reward_accuracy": sum(m > 0 for m in margins) / n,
        "mean_margin": sum(margins) / n,
        "mean_dlogp_chosen": sum(d_chosen) / n,
        "mean_dlogp_rejected": sum(d_rejected) / n,
        "n_gen": len(len_tuned),
        "mean_gen_len_tuned": sum(len_tuned) / max(len(len_tuned), 1),
        "mean_gen_len_base": sum(len_base) / max(len(len_base), 1),
        "margins": margins,
    }


def _std_err(xs) -> float:
    n = len(xs)
    if n < 2:
        return float("inf")
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return (var / n) ** 0.5


def verdict(main, *, lr0, swapped, max_chosen_drop=10.0, max_len_ratio=1.2) -> dict:
    """Apply the pre-registered ADR 0001 criteria. Pure function over check results.

    Each run is {"A": bool, "B": bool, "heldout": heldout_metrics(...)}.
    Thresholds (2 standard errors, chosen-drop in summed nats, length ratio)
    are judgment calls fixed before the runs and reported with the verdict.
    """
    h = main["heldout"]
    checks = {
        "A_adapter_trained": {"passed": bool(main["A"]), "reason": "every lora_B non-zero" if main["A"] else "adapter untrained"},
        "B_roundtrip": {"passed": bool(main["B"]), "reason": "adapter reloads and changes logits" if main["B"] else "adapter does not round-trip"},
    }

    se = _std_err(h["margins"])
    c_reasons = []
    if not h["mean_margin"] > 2 * se:
        c_reasons.append(f"mean margin {h['mean_margin']:.3g} within 2 SE ({2 * se:.3g}) of zero")
    if not h["reward_accuracy"] > 0.5:
        c_reasons.append(f"reward accuracy {h['reward_accuracy']:.2f} not above 0.5")
    if h["mean_dlogp_chosen"] < -max_chosen_drop:
        c_reasons.append(
            f"chosen log-prob collapsed ({h['mean_dlogp_chosen']:.3g} nats); margin comes from pushing both down"
        )
    checks["C_heldout_learning"] = {
        "passed": not c_reasons,
        "reason": "; ".join(c_reasons) or f"margin {h['mean_margin']:.3g} > 2 SE, accuracy {h['reward_accuracy']:.2f}",
    }

    d_reasons = []
    if lr0["A"]:
        d_reasons.append("lr=0 control passes check A, so A cannot tell trained from untrained")
    if lr0["heldout"]["mean_margin"] > 2 * _std_err(lr0["heldout"]["margins"]):
        d_reasons.append("lr=0 control shows a held-out margin")
    if not swapped["heldout"]["mean_margin"] < 0:
        d_reasons.append(f"swapped-labels margin {swapped['heldout']['mean_margin']:.3g} not reversed")
    checks["D_controls"] = {"passed": not d_reasons, "reason": "; ".join(d_reasons) or "controls fail as expected"}

    warnings = []
    if h["mean_gen_len_base"] and h["mean_gen_len_tuned"] / h["mean_gen_len_base"] > max_len_ratio:
        warnings.append(
            f"response length grew {h['mean_gen_len_tuned'] / h['mean_gen_len_base']:.2f}x vs base (length bias in data)"
        )

    ship = all(c["passed"] for c in checks.values())
    return {
        "verdict": "SHIP" if ship else "DON'T SHIP",
        "checks": checks,
        "warnings": warnings,
        "thresholds": {"margin_se_multiple": 2, "max_chosen_drop_nats": max_chosen_drop, "max_len_ratio": max_len_ratio},
    }


def encode_pair(tokenizer, row) -> dict:
    """Tokenize a message-list preference row the way TRL does: completion = full - prompt."""
    prompt_ids = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=True)
    out = {"prompt_ids": list(prompt_ids)}
    for side in ("chosen", "rejected"):
        full = tokenizer.apply_chat_template(row["prompt"] + row[side], tokenize=True)
        if list(full[: len(prompt_ids)]) != list(prompt_ids):
            raise ValueError(f"{side}: chat template does not extend the prompt; cannot split completion")
        out[f"{side}_ids"] = list(full[len(prompt_ids):])
    return out


def _evaluate_run(base_dir, adapter_dir, pairs, prompt_ids, beta, max_new_tokens, eos_id, gen_limit=None) -> dict:
    a = check_adapter_trained(adapter_dir)
    print(f"  [{adapter_dir}] A: {a.reason}", flush=True)
    b = check_roundtrip(base_dir, adapter_dir, prompt_ids)
    print(f"  [{adapter_dir}] B: {b.reason}", flush=True)
    _free()
    h = heldout_metrics(base_dir, adapter_dir, pairs, beta, max_new_tokens=max_new_tokens, eos_id=eos_id, gen_limit=gen_limit)
    _free()
    print(f"  [{adapter_dir}] C: acc {h['reward_accuracy']:.2f} margin {h['mean_margin']:.4f}", flush=True)
    return {"A": asdict(a), "B": asdict(b), "heldout": {k: v for k, v in h.items()}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base model id/path (with tokenizer)")
    parser.add_argument("--main", required=True, help="adapter dir of the run under test")
    parser.add_argument("--lr0", required=True, help="adapter dir of the lr=0 control")
    parser.add_argument("--swapped", required=True, help="adapter dir of the swapped-labels control")
    parser.add_argument("--heldout", required=True, help="held-out JSONL with prompt/chosen/rejected message lists")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gen-pairs", type=int, default=None, help="measure generated length on the first N pairs only")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    rows = read_jsonl(args.heldout)
    rows = rows[: args.limit] if args.limit else rows
    pairs = [encode_pair(tok, r) for r in rows]
    prompt_ids = [p["prompt_ids"] for p in pairs[:8]]

    runs = {
        name: _evaluate_run(
            args.base, path, pairs, prompt_ids, args.beta, args.max_new_tokens, tok.eos_token_id, args.gen_pairs
        )
        for name, path in (("main", args.main), ("lr0", args.lr0), ("swapped", args.swapped))
    }
    as_input = {k: {"A": v["A"]["passed"], "B": v["B"]["passed"], "heldout": v["heldout"]} for k, v in runs.items()}
    result = verdict(as_input["main"], lr0=as_input["lr0"], swapped=as_input["swapped"])

    report = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "args": vars(args),
        "eos_token_id": tok.eos_token_id,
        **result,
        "runs": runs,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"{report['timestamp_utc']} {result['verdict']}")
    for name, c in result["checks"].items():
        print(f"  {'PASS' if c['passed'] else 'FAIL'} {name}: {c['reason']}")
    for w in result["warnings"]:
        print(f"  WARN {w}")
    return 0 if result["verdict"] == "SHIP" else 1


if __name__ == "__main__":
    sys.exit(main())
