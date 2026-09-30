from verify_training import check_roundtrip


def test_good_adapter_loads_fully_and_changes_logits(base_dir, trained_adapter, prompt_ids):
    result = check_roundtrip(base_dir, trained_adapter, prompt_ids)
    assert result.passed, result.reason
    assert result.details["loaded"] == 8  # 2 layers x 2 modules x (A, B)
    assert result.details["unmatched_keys"] == []
    assert result.details["max_logit_diff"] > 0


def test_inner_mangled_keys_load_nothing(base_dir, inner_mangled_adapter, prompt_ids):
    result = check_roundtrip(base_dir, inner_mangled_adapter, prompt_ids)
    assert not result.passed
    assert result.details["loaded"] == 0
    assert len(result.details["unmatched_keys"]) == 8


def test_zero_lora_b_adapter_leaves_logits_identical(base_dir, untrained_adapter, prompt_ids):
    result = check_roundtrip(base_dir, untrained_adapter, prompt_ids)
    assert not result.passed
    assert result.details["max_logit_diff"] == 0.0
