import random

from make_faults import FAULTS, make_fault


def _row(i):
    return {
        "prompt": [{"role": "user", "content": f"Как вернуть заказ номер {i}?"}],
        "chosen": [{"role": "assistant", "content": f"Откройте раздел заказов и нажмите возврат {i}."}],
        "rejected": [{"role": "assistant", "content": f"Не знаю {i}."}],
    }


def _text(row, side):
    return row[side][0]["content"]


def test_every_fault_type_is_labelled():
    rng = random.Random(0)
    for fault in FAULTS:
        out = make_fault(_row(1), fault, rng)
        assert out["fault"] == fault


def test_exact_duplicate_has_identical_sides():
    out = make_fault(_row(1), "exact_duplicate", random.Random(0))
    assert out["chosen"] == out["rejected"]


def test_near_duplicate_differs_only_in_whitespace_and_punctuation():
    out = make_fault(_row(1), "near_duplicate", random.Random(0))
    strip = lambda s: "".join(ch for ch in s if ch.isalnum())
    assert _text(out, "chosen") != _text(out, "rejected")
    assert strip(_text(out, "chosen")) == strip(_text(out, "rejected"))


def test_shared_prefix_pushes_the_difference_past_the_char_budget():
    out = make_fault(_row(1), "shared_prefix_truncation", random.Random(0))
    c, r = _text(out, "chosen"), _text(out, "rejected")
    common = next(i for i, (a, b) in enumerate(zip(c, r)) if a != b)
    assert common > 4000  # well past a 512-token completion budget


def test_length_bias_makes_chosen_much_longer():
    out = make_fault(_row(1), "length_bias", random.Random(0))
    assert len(_text(out, "chosen")) > 5 * len(_text(out, "rejected"))


def test_homoglyphs_inject_latin_lookalikes_into_cyrillic():
    out = make_fault(_row(1), "homoglyph", random.Random(0))
    assert any(ch in "aeopcx" for ch in _text(out, "chosen"))
    assert _text(out, "chosen") != _text(_row(1), "chosen")


def test_empty_completion():
    out = make_fault(_row(1), "empty_completion", random.Random(0))
    assert _text(out, "chosen").strip() == ""


def test_prompt_echo_puts_prompt_verbatim_into_completion():
    out = make_fault(_row(1), "prompt_echo", random.Random(0))
    assert _text(_row(1), "prompt") in _text(out, "chosen")


def test_jsonl_with_unicode_line_separator_inside_a_string_round_trips(tmp_path):
    import json

    from make_faults import main

    row = _row(1)
    row["chosen"][0]["content"] = "строка перенос"
    src = tmp_path / "train.jsonl"
    src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in [row] * 7))
    out = tmp_path / "faults.jsonl"
    main(["--src", str(src), "--out", str(out), "--per-fault", "1"])
    assert len(out.read_text(encoding="utf-8").split("\n")) - 1 == 7
