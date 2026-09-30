from verify_training import verdict


def _heldout(margins, d_chosen=0.5, d_rejected=-1.0, len_tuned=100, len_base=100):
    n = len(margins)
    return {
        "n": n,
        "reward_accuracy": sum(m > 0 for m in margins) / n,
        "mean_margin": sum(margins) / n,
        "mean_dlogp_chosen": d_chosen,
        "mean_dlogp_rejected": d_rejected,
        "mean_gen_len_tuned": len_tuned,
        "mean_gen_len_base": len_base,
        "margins": margins,
    }


def _run(a=True, b=True, heldout=None):
    return {"A": a, "B": b, "heldout": heldout}


GOOD_MAIN = _run(heldout=_heldout([0.4, 0.3, 0.5, 0.2, 0.35, 0.45, -0.05, 0.3]))
LR0 = _run(a=False, b=False, heldout=_heldout([0.0] * 8, d_chosen=0.0, d_rejected=0.0))
SWAPPED = _run(heldout=_heldout([-0.3, -0.4, -0.2, 0.05, -0.35, -0.3, -0.25, -0.4]))


def test_healthy_run_with_controls_behaving_is_ship():
    v = verdict(GOOD_MAIN, lr0=LR0, swapped=SWAPPED)
    assert v["verdict"] == "SHIP", v


def test_margin_inside_noise_is_dont_ship():
    noisy = _run(heldout=_heldout([0.3, -0.3, 0.25, -0.2, 0.1, -0.15, 0.05, 0.0]))
    v = verdict(noisy, lr0=LR0, swapped=SWAPPED)
    assert v["verdict"] == "DON'T SHIP"
    assert not v["checks"]["C_heldout_learning"]["passed"]


def test_chosen_likelihood_collapse_is_dont_ship_even_with_positive_margin():
    collapsed = _run(heldout=_heldout(GOOD_MAIN["heldout"]["margins"], d_chosen=-40.0, d_rejected=-45.0))
    v = verdict(collapsed, lr0=LR0, swapped=SWAPPED)
    assert v["verdict"] == "DON'T SHIP"
    assert "chosen" in v["checks"]["C_heldout_learning"]["reason"]


def test_swapped_control_not_reversed_means_checks_cannot_discriminate():
    v = verdict(GOOD_MAIN, lr0=LR0, swapped=GOOD_MAIN)
    assert v["verdict"] == "DON'T SHIP"
    assert not v["checks"]["D_controls"]["passed"]


def test_lr0_control_that_looks_trained_fails_controls():
    v = verdict(GOOD_MAIN, lr0=_run(a=True, heldout=LR0["heldout"]), swapped=SWAPPED)
    assert not v["checks"]["D_controls"]["passed"]


def test_failed_roundtrip_is_dont_ship():
    v = verdict(_run(b=False, heldout=GOOD_MAIN["heldout"]), lr0=LR0, swapped=SWAPPED)
    assert v["verdict"] == "DON'T SHIP"


def test_verbosity_drift_is_a_warning_not_a_blocker():
    verbose = _run(heldout=_heldout(GOOD_MAIN["heldout"]["margins"], len_tuned=160, len_base=100))
    v = verdict(verbose, lr0=LR0, swapped=SWAPPED)
    assert v["verdict"] == "SHIP"
    assert any("length" in w for w in v["warnings"])
