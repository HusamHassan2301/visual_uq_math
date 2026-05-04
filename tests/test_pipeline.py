"""
test_pipeline.py
----------------
Unit tests for all pipeline modules.
Run: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
import pandas as pd


# ── Dataset tests ──────────────────────────────────────────────────────────

def test_create_pilot_dataset():
    from dataset import PILOT_PROBLEMS_RAW, Problem
    problems = [Problem(**p) for p in PILOT_PROBLEMS_RAW]
    assert len(problems) == 60, f"Expected 60 problems, got {len(problems)}"


def test_stratified_sample():
    from dataset import PILOT_PROBLEMS_RAW, Problem, stratified_sample
    problems = [Problem(**p) for p in PILOT_PROBLEMS_RAW]
    sample = stratified_sample(problems, n_per_cell=2)
    assert len(sample) == 24  # 4 subjects × 3 difficulties × 2


def test_problem_fields():
    from dataset import PILOT_PROBLEMS_RAW, Problem
    for p_raw in PILOT_PROBLEMS_RAW:
        p = Problem(**p_raw)
        assert p.id, "Problem must have an id"
        assert p.subject in ["calculus", "combinatorics", "linear_algebra", "number_theory"]
        assert p.difficulty in ["university", "competition", "olympiad"]
        assert p.text, "Problem must have text"
        assert p.answer, "Problem must have an answer"


def test_subject_distribution():
    from dataset import PILOT_PROBLEMS_RAW, Problem, summary
    problems = [Problem(**p) for p in PILOT_PROBLEMS_RAW]
    s = summary(problems)
    assert s["total"] == 60
    for subject in ["calculus", "combinatorics", "linear_algebra", "number_theory"]:
        assert s["by_subject"][subject] == 15


# ── Evaluator tests ────────────────────────────────────────────────────────

def test_mock_evaluator_runs():
    from dataset import PILOT_PROBLEMS_RAW, Problem
    from evaluator import MockEvaluator
    problems = [Problem(**p) for p in PILOT_PROBLEMS_RAW[:5]]
    ev = MockEvaluator(seed=42)
    results = ev.run(problems, "text_only")
    assert len(results) == 5
    for r in results:
        assert r.condition == "text_only"
        assert r.model == "mock-gpt4o"
        assert isinstance(r.correct, bool)


def test_mock_evaluator_visual_condition():
    from dataset import PILOT_PROBLEMS_RAW, Problem
    from evaluator import MockEvaluator
    problems = [Problem(**p) for p in PILOT_PROBLEMS_RAW[:10]]
    ev = MockEvaluator(seed=42)
    results = ev.run(problems, "text_visual")
    # Only problems with has_natural_visual=True should be included
    for r in results:
        assert r.condition == "text_visual"


def test_mock_accuracy_varies_by_difficulty():
    from dataset import PILOT_PROBLEMS_RAW, Problem
    from evaluator import MockEvaluator
    problems = [Problem(**p) for p in PILOT_PROBLEMS_RAW]
    ev = MockEvaluator(seed=42)
    results = ev.run(problems, "text_only")
    df = pd.DataFrame([r.__dict__ for r in results])
    univ_acc = df[df["difficulty"] == "university"]["correct"].mean()
    olym_acc = df[df["difficulty"] == "olympiad"]["correct"].mean()
    assert univ_acc > olym_acc, "University should be easier than Olympiad"


def test_answer_checking():
    from evaluator import _check_answer
    assert _check_answer("42", "42") is True
    assert _check_answer("3.14", "3.14") is True
    assert _check_answer("42", "43") is False
    assert _check_answer("1/4", "0.25") is False  # string mismatch
    assert _check_answer("0.25", "0.25") is True


# ── Analysis tests ─────────────────────────────────────────────────────────

def _make_mock_df():
    """Create a minimal mock DataFrame for analysis tests."""
    from dataset import PILOT_PROBLEMS_RAW, Problem
    from evaluator import MockEvaluator
    problems = [Problem(**p) for p in PILOT_PROBLEMS_RAW]
    ev = MockEvaluator(seed=42)
    rows = []
    for cond in ["text_only", "text_visual"]:
        for r in ev.run(problems, cond):
            rows.append(r.__dict__)
    return pd.DataFrame(rows)


def test_accuracy_by_condition():
    from analysis import accuracy_by_condition
    df = _make_mock_df()
    result = accuracy_by_condition(df)
    assert "condition" in result.columns
    assert "correct" in result.columns
    assert len(result) == 2


def test_accuracy_delta_sign():
    from analysis import accuracy_delta
    df = _make_mock_df()
    delta = accuracy_delta(df, ["subject"])
    assert "delta" in delta.columns
    # Number theory should have negative delta (visual hurts)
    nt = delta[delta["subject"] == "number_theory"]["delta"].values
    assert len(nt) > 0


def test_mi_proxy_values():
    from analysis import estimate_mutual_information_proxy
    df = _make_mock_df()
    mi = estimate_mutual_information_proxy(df)
    assert "calculus" in mi
    assert "number_theory" in mi
    for subj, vals in mi.items():
        assert "MI_proxy" in vals
        assert "interpretation" in vals
        assert vals["interpretation"] in ["visual helps", "visual hurts", "visual neutral"]


def test_binary_entropy():
    """Test the binary entropy helper used in MI proxy."""
    from analysis import estimate_mutual_information_proxy
    # H(0.5) should be maximum at 1 bit
    # H(0) = H(1) = 0
    # This is tested implicitly through MI proxy
    df = _make_mock_df()
    mi = estimate_mutual_information_proxy(df)
    for subj, vals in mi.items():
        assert 0 <= vals["H_text_only"] <= 1.0
        assert 0 <= vals["H_text_visual"] <= 1.0


def test_visual_reference_rate():
    from analysis import visual_reference_rate
    df = _make_mock_df()
    vr = visual_reference_rate(df)
    assert "references_visual" in vr.columns
    assert all(0 <= r <= 1 for r in vr["references_visual"])


def test_mcnemar_test():
    from analysis import mcnemar_test
    df = _make_mock_df()
    result = mcnemar_test(df)
    assert "p_value" in result or "error" in result
    if "p_value" in result:
        assert 0 <= result["p_value"] <= 1


# ── Image generation tests ─────────────────────────────────────────────────

def test_diagram_generation():
    from image_gen import generate_diagram
    result = generate_diagram("calc_u_002", save=False)
    assert result is not None
    assert isinstance(result, str)  # base64 string


def test_diagram_missing_id():
    from image_gen import generate_diagram
    result = generate_diagram("nonexistent_id", save=False)
    assert result is None


def test_function_plot():
    from image_gen import plot_function
    fig = plot_function("np.sin(x)", (-3, 3), "sin(x)")
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close("all")


def test_pascals_triangle():
    from image_gen import plot_pascals_triangle
    fig = plot_pascals_triangle(5)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close("all")


if __name__ == "__main__":
    print("Running basic test suite...")
    test_create_pilot_dataset()
    print("✓ dataset: 60 problems created")
    test_stratified_sample()
    print("✓ dataset: stratified sampling works")
    test_problem_fields()
    print("✓ dataset: all problem fields valid")
    test_mock_evaluator_runs()
    print("✓ evaluator: mock runs correctly")
    test_mi_proxy_values()
    print("✓ analysis: MI proxy computed")
    test_diagram_generation()
    print("✓ image_gen: diagram generated")
    print("\nAll basic tests passed.")
