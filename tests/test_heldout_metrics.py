import pytest

from verify_training import heldout_metrics


def _swap(pairs):
    return [{**p, "chosen_ids": p["rejected_ids"], "rejected_ids": p["chosen_ids"]} for p in pairs]


def test_untrained_adapter_is_indistinguishable_from_reference(base_dir, untrained_adapter, pairs):
    m = heldout_metrics(base_dir, untrained_adapter, pairs, beta=0.1, max_new_tokens=8)
    assert m["mean_margin"] == 0.0
    assert m["mean_dlogp_chosen"] == 0.0
    assert m["mean_dlogp_rejected"] == 0.0
    assert m["reward_accuracy"] == 0.0  # ties never count as a win
    assert m["mean_gen_len_tuned"] == m["mean_gen_len_base"]


def test_trained_adapter_moves_logps_away_from_reference(base_dir, trained_adapter, pairs):
    m = heldout_metrics(base_dir, trained_adapter, pairs, beta=0.1, max_new_tokens=8)
    assert m["mean_dlogp_chosen"] != 0.0
    assert m["mean_margin"] != 0.0
    assert m["n"] == 3


def test_swapping_chosen_and_rejected_negates_the_margin(base_dir, trained_adapter, pairs):
    m = heldout_metrics(base_dir, trained_adapter, pairs, beta=0.1, max_new_tokens=8)
    s = heldout_metrics(base_dir, trained_adapter, _swap(pairs), beta=0.1, max_new_tokens=8)
    assert s["mean_margin"] == pytest.approx(-m["mean_margin"], abs=1e-6)
    assert s["reward_accuracy"] == pytest.approx(1.0 - m["reward_accuracy"])


def test_gen_limit_measures_length_on_first_n_pairs_only(base_dir, trained_adapter, pairs):
    m = heldout_metrics(base_dir, trained_adapter, pairs, beta=0.1, max_new_tokens=8, gen_limit=1)
    assert m["n"] == 3
    assert m["n_gen"] == 1
