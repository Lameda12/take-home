from prepare_data import keep_row


def _row(**kw):
    base = {
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "good"}],
        "rejected": [{"role": "assistant", "content": "bad"}],
        "source": "lmsys_clean_ru_queries",
        "sonnet_approved": True,
        "is_bad_by_regex": False,
    }
    base.update(kw)
    return base


LENS = {"prompt": 100, "chosen": 300, "rejected": 200}


def test_clean_single_turn_row_that_fits_is_kept():
    assert keep_row(_row(), LENS, max_length=1024)


def test_unapproved_or_regex_bad_rows_are_dropped():
    assert not keep_row(_row(sonnet_approved=False), LENS, max_length=1024)
    assert not keep_row(_row(is_bad_by_regex=True), LENS, max_length=1024)


def test_multi_turn_prompt_is_dropped_but_system_prompt_is_allowed():
    multi = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]
    with_system = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    assert not keep_row(_row(prompt=multi), LENS, max_length=1024)
    assert keep_row(_row(prompt=with_system), LENS, max_length=1024)


def test_identical_chosen_and_rejected_is_dropped():
    same = [{"role": "assistant", "content": "x"}]
    assert not keep_row(_row(chosen=same, rejected=same), LENS, max_length=1024)


def test_roleplay_source_is_dropped():
    assert not keep_row(_row(source="pippa_and_multiturn_10_10"), LENS, max_length=1024)


def test_rows_that_would_be_truncated_are_dropped():
    # Soup caps the prompt at max_length // 2 = 512
    assert not keep_row(_row(), {"prompt": 513, "chosen": 10, "rejected": 10}, max_length=1024)
    # prompt + longest completion must fit in max_length
    assert not keep_row(_row(), {"prompt": 500, "chosen": 525, "rejected": 10}, max_length=1024)
    assert keep_row(_row(), {"prompt": 500, "chosen": 524, "rejected": 10}, max_length=1024)
