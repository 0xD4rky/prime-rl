"""
Physics RLVR environment for prime-rl.

Loads the Darkyy/phy-rl-base dataset, filters invalid examples, and trains
a model to solve physics problems using binary/fractional correctness reward.

Reward design:
- Reward = correct_sub_answers / total_sub_answers  (fractional for multi-part)
- A sub-answer is correct if the model's \boxed{...} content matches the ground
  truth within 5% relative tolerance (numerical) or symbolic equality (expression).
- No format bonus — the gradient signal comes entirely from correctness.
"""

import json
import os
import random
import re
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset

# ---------------------------------------------------------------------------
# System prompt — guides physics CoT and enforces \boxed{} output format
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert physics problem solver with deep knowledge of classical \
mechanics, electromagnetism, thermodynamics, statistical mechanics, quantum \
mechanics, optics, and special relativity.

Approach every problem as follows:
1. **Identify** the relevant physical domain, principles, and governing equations.
2. **Define** all variables, coordinate systems, and approximations explicitly.
3. **Derive** the solution step by step, showing key algebraic and calculus steps.
4. **Verify** units and check limiting cases where possible.
5. **State** the final answer clearly.

Answer format rules:
- Always enclose your final answer(s) in \\boxed{}.
- For multi-part problems (a, b, c, …), provide one box per part in order:
  \\boxed{<answer_a>} for part (a), \\boxed{<answer_b>} for part (b), etc.
- Include units inside the box when the answer is a physical quantity:
  \\boxed{9.8\\,\\text{m/s}^2}, \\boxed{2.4\\times10^{-3}\\,\\text{kg}}.
- For symbolic answers: \\boxed{\\dfrac{mv^2}{r}}.
- For dimensionless numbers: \\boxed{3.14}.

Failure to place your answer in \\boxed{} will result in zero credit.\
"""

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_SCIENTIFIC_RE = re.compile(
    r"^(-?)\s*(\d+(?:\.\d+)?)\s*(?:\\times|×)\s*10\^{?(-?\d+)}?$"
)


def _extract_boxed(text: str) -> list[str]:
    """Return all \\boxed{...} contents from *text*, handling nested braces."""
    results: list[str] = []
    i = 0
    while i < len(text):
        idx = text.find(r"\boxed{", i)
        if idx == -1:
            break
        start = idx + 7  # len(r"\boxed{")
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start : j - 1])
        i = idx + 1
    return results


def _parse_number(s: str) -> float | None:
    """Parse a LaTeX numeric expression to float, returning None on failure."""
    s = s.strip()
    # Strip non-numeric LaTeX wrappers (units, text, etc.)
    s = re.sub(r"\\(?:text|mathrm|mbox)\{[^}]*\}", "", s).strip()
    # Strip trailing/leading commas or spaces that sometimes appear
    s = s.strip(",").strip()

    try:
        return float(s)
    except ValueError:
        pass

    # Handle \times 10^{n} / \times 10^n (scientific notation)
    m = _SCIENTIFIC_RE.match(s)
    if m:
        sign_str, mantissa_str, exp_str = m.groups()
        sign = -1.0 if sign_str == "-" else 1.0
        return sign * float(mantissa_str) * 10 ** int(exp_str)

    # Sympy as last resort (handles \frac, \sqrt, \pi, etc.)
    try:
        from sympy.parsing.latex import parse_latex  # noqa: PLC0415

        result = float(parse_latex(s).evalf())
        if result != result:  # NaN guard
            return None
        return result
    except Exception:
        pass

    return None


def _numbers_match(a: float, b: float, rtol: float = 0.05) -> bool:
    """True if |a-b|/|b| ≤ rtol (or both near-zero)."""
    if b == 0.0:
        return abs(a) < 1e-10
    return abs(a - b) / abs(b) <= rtol


def _expressions_match(expr1: str, expr2: str) -> bool:
    """Symbolically compare two LaTeX expressions via sympy."""
    try:
        from sympy import simplify  # noqa: PLC0415
        from sympy.parsing.latex import parse_latex  # noqa: PLC0415

        e1 = parse_latex(expr1)
        e2 = parse_latex(expr2)
        return bool(simplify(e1 - e2) == 0)
    except Exception:
        pass
    return expr1.strip() == expr2.strip()


def _check_answer(model_ans: str, expected_value: str, answer_type: str) -> bool:
    """Return True if *model_ans* matches *expected_value* for *answer_type*."""
    model_ans = model_ans.strip()
    expected_value = expected_value.strip()

    if answer_type in ("numerical", "text"):
        model_num = _parse_number(model_ans)
        expected_num = _parse_number(expected_value)
        if model_num is not None and expected_num is not None:
            return _numbers_match(model_num, expected_num)
        # Fall back to case-insensitive string match
        return model_ans.lower() == expected_value.lower()

    if answer_type == "expression":
        return _expressions_match(model_ans, expected_value)

    # Unknown type: string fallback
    return model_ans.lower() == expected_value.lower()


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def correctness_reward(completion: list[dict], answer: str, **kwargs) -> float:
    """
    Fractional correctness reward in [0.0, 1.0].

    Extracts all \\boxed{} answers from the model's completion and checks how
    many of the expected sub-answers are matched.  Returns correct / total.
    """
    # Extract assistant text (last assistant message in completion)
    text = ""
    for msg in reversed(completion):
        if msg.get("role") == "assistant":
            text = msg.get("content", "")
            break

    model_boxed = _extract_boxed(text)

    try:
        expected: list[dict[str, Any]] = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        return 0.0

    if not expected or not model_boxed:
        return 0.0

    correct = sum(
        1
        for sa in expected
        if any(
            _check_answer(m, sa["value"], sa["answer_type"]) for m in model_boxed
        )
    )
    return correct / len(expected)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

# Sources where the `value` field is a reliable ground-truth (not a placeholder).
# phybench always has value="2" regardless of the actual answer.
_UNRELIABLE_SOURCES = {"phybench"}

_IMAGE_RE = re.compile(r"<image_start>|\[problem_image")


def _is_valid(row: dict[str, Any], sources: list[str] | None) -> bool:
    if row["source"] in _UNRELIABLE_SOURCES:
        return False
    if _IMAGE_RE.search(row["question"]):
        return False
    if sources and row["source"] not in sources:
        return False
    sa = row.get("structured_answers") or []
    if not sa:
        return False
    # Skip examples where every sub-answer has a suspiciously trivial value,
    # which often indicates placeholder data in physics_text_reasoning_train.
    if all(s.get("value", "").strip() in ("2", "") for s in sa):
        return False
    return True


def _to_verifiers_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": row["question"],
        "answer": json.dumps(row["structured_answers"]),
    }


# ---------------------------------------------------------------------------
# Environment entry point
# ---------------------------------------------------------------------------


def load_environment(
    hf_token: str | None = None,
    sources: str | list[str] | None = None,
    num_train: int | None = None,
    num_eval: int = 100,
    seed: int = 42,
    **kwargs,
) -> vf.Environment:
    """
    Build and return the physics SingleTurnEnv.

    Args:
        hf_token:  HuggingFace token.  Falls back to HF_TOKEN /
                   HUGGING_FACE_HUB_TOKEN environment variables.
        sources:   Comma-separated source name(s) or list of source names to
                   include.  None = all reliable sources.
        num_train: Max training examples.  None = all available.
        num_eval:  Evaluation set size (drawn before train split).
        seed:      Random seed for shuffling.
    """
    token = (
        hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    # Normalise sources: accept comma-separated string or list
    if isinstance(sources, str):
        sources = [s.strip() for s in sources.split(",") if s.strip()]
    if not sources:
        sources = None

    raw = load_dataset("Darkyy/phy-rl-base", token=token)["train"]
    valid = [row for row in raw if _is_valid(row, sources)]

    rng = random.Random(seed)
    rng.shuffle(valid)

    # Eval comes first so it's deterministic regardless of num_train
    eval_take = min(num_eval, max(1, len(valid) // 5))
    eval_rows = valid[:eval_take]
    train_rows = valid[eval_take:]
    if num_train is not None:
        train_rows = train_rows[:num_train]

    train_ds = Dataset.from_list([_to_verifiers_row(r) for r in train_rows])
    eval_ds = Dataset.from_list([_to_verifiers_row(r) for r in eval_rows])

    rubric = vf.Rubric(funcs=[correctness_reward], weights=[1.0])
    return vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
    )
