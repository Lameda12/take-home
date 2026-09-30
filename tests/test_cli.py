import json

import pytest

from verify_training import encode_pair, main


def test_encode_pair_splits_completion_after_generation_prompt(base_with_tokenizer):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_with_tokenizer)
    row = {"prompt": [{"role": "user", "content": "w1 w2"}],
           "chosen": [{"role": "assistant", "content": "w10"}],
           "rejected": [{"role": "assistant", "content": "w20 w21"}]}
    pair = encode_pair(tok, row)
    # "<|im_start|> user w1 w2 <|im_end|> <|im_start|> assistant"
    assert tok.decode(pair["prompt_ids"]).split() == ["<|im_start|>", "user", "w1", "w2", "<|im_end|>", "<|im_start|>", "assistant"]
    assert tok.decode(pair["chosen_ids"]).split() == ["w10", "<|im_end|>"]
    assert tok.decode(pair["rejected_ids"]).split() == ["w20", "w21", "<|im_end|>"]


def test_main_writes_timestamped_report_with_verdict(
    base_with_tokenizer, trained_adapter, untrained_adapter, heldout_jsonl, tmp_path
):
    out = tmp_path / "report.json"
    code = main([
        "--base", str(base_with_tokenizer),
        "--main", str(trained_adapter),
        "--lr0", str(untrained_adapter),
        "--swapped", str(trained_adapter),
        "--heldout", str(heldout_jsonl),
        "--max-new-tokens", "4",
        "--out", str(out),
    ])
    report = json.loads(out.read_text())
    assert report["timestamp_utc"].endswith("Z")
    assert report["verdict"] in ("SHIP", "DON'T SHIP")
    assert set(report["runs"]) == {"main", "lr0", "swapped"}
    assert report["runs"]["lr0"]["A"]["passed"] is False
    # swapped == main here, so the controls cannot discriminate
    assert report["verdict"] == "DON'T SHIP"
    assert code == 1


def test_heldout_reader_survives_unicode_line_separator(tmp_path):
    from verify_training import read_jsonl

    path = tmp_path / "h.jsonl"
    path.write_text(json.dumps({"x": "a b"}, ensure_ascii=False) + "\n", encoding="utf-8")
    assert read_jsonl(path) == [{"x": "a b"}]
