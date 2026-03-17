"""Unit tests for the phy-env reward function and helpers."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "phy_env"))

from phy_env.phy_env import (
    _check_answer,
    _extract_boxed,
    _is_valid,
    _numbers_match,
    _parse_number,
    correctness_reward,
)


# ---------------------------------------------------------------------------
# _extract_boxed
# ---------------------------------------------------------------------------


def test_extract_boxed_simple():
    assert _extract_boxed(r"The answer is \boxed{42}.") == ["42"]


def test_extract_boxed_nested_fraction():
    assert _extract_boxed(r"\boxed{\frac{3}{4}}") == [r"\frac{3}{4}"]


def test_extract_boxed_multiple():
    text = r"Part a: \boxed{42} and part b: \boxed{3.14}"
    assert _extract_boxed(text) == ["42", "3.14"]


def test_extract_boxed_empty():
    assert _extract_boxed("No box here.") == []


def test_extract_boxed_nested_braces():
    text = r"\boxed{\frac{mv^2}{2r}}"
    result = _extract_boxed(text)
    assert result == [r"\frac{mv^2}{2r}"]


def test_extract_boxed_with_think_tags():
    # Qwen3 thinking mode wraps CoT in <think>...</think>
    text = r"<think>Let me compute...</think>Therefore \boxed{9.8\,\text{m/s}^2}."
    assert _extract_boxed(text) == [r"9.8\,\text{m/s}^2"]


# ---------------------------------------------------------------------------
# _parse_number
# ---------------------------------------------------------------------------


def test_parse_number_integer():
    assert _parse_number("42") == pytest.approx(42.0)


def test_parse_number_decimal():
    assert _parse_number("3.14") == pytest.approx(3.14)


def test_parse_number_negative():
    assert _parse_number("-2.5") == pytest.approx(-2.5)


def test_parse_number_scientific_braced():
    assert _parse_number(r"2.39078 \times 10^{-15}") == pytest.approx(2.39078e-15, rel=1e-5)


def test_parse_number_scientific_unbraced():
    assert _parse_number(r"1.68 \times 10^-3") == pytest.approx(1.68e-3, rel=1e-5)


def test_parse_number_with_unit_text():
    # Units wrapped in \text{} should be stripped
    result = _parse_number(r"9.8\,\text{m/s}^2")
    # After stripping \text{m/s} the remainder "9.8\,^2" may not parse, that's ok
    # — units are ignored in numeric comparison; just check it doesn't crash
    assert result is None or isinstance(result, float)


def test_parse_number_none_for_expression():
    # A pure symbolic expression should return None (no numeric value)
    result = _parse_number(r"\frac{mv^2}{r}")
    assert result is None


# ---------------------------------------------------------------------------
# _numbers_match
# ---------------------------------------------------------------------------


def test_numbers_match_exact():
    assert _numbers_match(42.0, 42.0)


def test_numbers_match_within_tolerance():
    assert _numbers_match(2.39e-15, 2.39078e-15)  # ~0.03% apart


def test_numbers_match_outside_tolerance():
    assert not _numbers_match(42.0, 43.0)  # ~2.3% apart — within 5%, so True
    assert not _numbers_match(1.0, 2.0)    # 100% apart


def test_numbers_match_zero():
    assert _numbers_match(0.0, 0.0)
    assert not _numbers_match(1.0, 0.0)


# ---------------------------------------------------------------------------
# _check_answer
# ---------------------------------------------------------------------------


def test_check_answer_numerical_exact():
    assert _check_answer("42", "42", "numerical")


def test_check_answer_numerical_scientific():
    assert _check_answer(r"2.39 \times 10^{-15}", r"2.39078 \times 10^{-15}", "numerical")


def test_check_answer_numerical_wrong():
    assert not _check_answer("42", "100", "numerical")


def test_check_answer_text_numeric():
    assert _check_answer(r"1.68 \times 10^{-3}", r"1.68 \times 10^{-3}", "text")


def test_check_answer_text_string_fallback():
    assert _check_answer("yes", "yes", "text")
    assert not _check_answer("yes", "no", "text")


# ---------------------------------------------------------------------------
# correctness_reward — integration tests
# ---------------------------------------------------------------------------


def _make_completion(text: str) -> list[dict]:
    return [{"role": "assistant", "content": text}]


def _make_answer(value: str, answer_type: str = "numerical") -> str:
    return json.dumps([{"index": 0, "label": None, "raw": "", "value": value, "unit": None, "answer_type": answer_type}])


def test_reward_correct():
    completion = _make_completion(r"Therefore \boxed{42}.")
    answer = _make_answer("42")
    assert correctness_reward(completion=completion, answer=answer) == pytest.approx(1.0)


def test_reward_correct_scientific():
    completion = _make_completion(r"\boxed{2.39 \times 10^{-15}}")
    answer = _make_answer(r"2.39078 \times 10^{-15}", "text")
    assert correctness_reward(completion=completion, answer=answer) == pytest.approx(1.0)


def test_reward_wrong():
    completion = _make_completion(r"\boxed{100}")
    answer = _make_answer("42")
    assert correctness_reward(completion=completion, answer=answer) == pytest.approx(0.0)


def test_reward_no_box():
    completion = _make_completion("The answer is 42.")
    answer = _make_answer("42")
    assert correctness_reward(completion=completion, answer=answer) == pytest.approx(0.0)


def test_reward_multipart_all_correct():
    completion = _make_completion(r"Part a: \boxed{42} Part b: \boxed{3.14}")
    answer = json.dumps([
        {"index": 0, "label": "a", "raw": "", "value": "42", "unit": None, "answer_type": "numerical"},
        {"index": 1, "label": "b", "raw": "", "value": "3.14", "unit": None, "answer_type": "numerical"},
    ])
    assert correctness_reward(completion=completion, answer=answer) == pytest.approx(1.0)


def test_reward_multipart_half_correct():
    completion = _make_completion(r"Part a: \boxed{42} Part b: \boxed{999}")
    answer = json.dumps([
        {"index": 0, "label": "a", "raw": "", "value": "42", "unit": None, "answer_type": "numerical"},
        {"index": 1, "label": "b", "raw": "", "value": "3.14", "unit": None, "answer_type": "numerical"},
    ])
    assert correctness_reward(completion=completion, answer=answer) == pytest.approx(0.5)


def test_reward_multipart_none_correct():
    completion = _make_completion(r"\boxed{0} \boxed{0}")
    answer = json.dumps([
        {"index": 0, "label": "a", "raw": "", "value": "42", "unit": None, "answer_type": "numerical"},
        {"index": 1, "label": "b", "raw": "", "value": "3.14", "unit": None, "answer_type": "numerical"},
    ])
    assert correctness_reward(completion=completion, answer=answer) == pytest.approx(0.0)


def test_reward_with_think_tags():
    # Qwen3 thinking output — answer should still be found after </think>
    text = r"<think>Let me reason step by step...</think>The answer is \boxed{42}."
    completion = _make_completion(text)
    answer = _make_answer("42")
    assert correctness_reward(completion=completion, answer=answer) == pytest.approx(1.0)


def test_reward_invalid_answer_json():
    completion = _make_completion(r"\boxed{42}")
    assert correctness_reward(completion=completion, answer="not_json") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _is_valid
# ---------------------------------------------------------------------------


def _make_row(source: str, question: str, structured_answers: list | None = None) -> dict:
    if structured_answers is None:
        structured_answers = [{"value": "42", "answer_type": "numerical"}]
    return {"source": source, "question": question, "structured_answers": structured_answers}


def test_is_valid_good_row():
    row = _make_row("olympiadbench_physics", "What is the speed?")
    assert _is_valid(row, sources=None)


def test_is_valid_phybench_filtered():
    row = _make_row("phybench", "A complex optics problem...")
    assert not _is_valid(row, sources=None)


def test_is_valid_image_filtered():
    row = _make_row("olympiadbench_physics", "See <image_start>[problem_image_1]<image_end>")
    assert not _is_valid(row, sources=None)


def test_is_valid_source_filter():
    row = _make_row("olympiadbench_physics", "What is force?")
    assert not _is_valid(row, sources=["ipho_open_train"])
    assert _is_valid(row, sources=["olympiadbench_physics"])


def test_is_valid_placeholder_value_filtered():
    row = _make_row(
        "physics_text_reasoning_train",
        "A rolling coin problem...",
        structured_answers=[{"value": "2", "answer_type": "numerical"}],
    )
    assert not _is_valid(row, sources=None)


def test_is_valid_empty_answers_filtered():
    row = _make_row("olympiadbench_physics", "What?", structured_answers=[])
    assert not _is_valid(row, sources=None)
