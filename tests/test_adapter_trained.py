from verify_training import check_adapter_trained


def test_adapter_with_nonzero_lora_b_passes(trained_adapter):
    result = check_adapter_trained(trained_adapter)
    assert result.passed
    assert len(result.details["lora_B_norms"]) == 4  # 2 layers x (q_proj, v_proj)


def test_adapter_with_all_zero_lora_b_fails(untrained_adapter):
    result = check_adapter_trained(untrained_adapter)
    assert not result.passed
    assert "zero" in result.reason


def test_adapter_moved_only_by_a_negligible_lr_fails(tiny_lr_adapter):
    # Soup forbids lr=0, so the control runs at lr=1e-10: non-zero is not "trained"
    result = check_adapter_trained(tiny_lr_adapter)
    assert not result.passed
    assert "0.0001" in result.reason
