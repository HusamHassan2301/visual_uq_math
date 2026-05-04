"""
analysis.py
-----------
Statistical analysis of evaluation results.
Computes accuracy, accuracy deltas, reasoning fidelity,
and information-theoretic uncertainty measures.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Dict, Tuple
from scipy import stats

FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "raw"

SUBJECTS     = ["calculus", "combinatorics", "linear_algebra", "number_theory"]
DIFFICULTIES = ["university", "competition", "olympiad"]
COLORS = {"calculus": "#2E6FAD", "combinatorics": "#E8530A",
          "linear_algebra": "#0D7C72", "number_theory": "#5B2D8E"}
DIFF_COLORS = {"university": "#2E6FAD", "competition": "#E8530A", "olympiad": "#B07D1A"}


# ── Data loading ───────────────────────────────────────────────────────────

def load_all_results(model: str = "mock-gpt4o") -> pd.DataFrame:
    """Load text_only and text_visual results into a single DataFrame."""
    rows = []
    for condition in ["text_only", "text_visual"]:
        path = RESULTS_DIR / f"{model}_{condition}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        rows.extend(data)
    return pd.DataFrame(rows)


# ── Core accuracy analysis ─────────────────────────────────────────────────

def accuracy_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Overall accuracy per condition."""
    return df.groupby("condition")["correct"].mean().reset_index()


def accuracy_by_subject_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy broken down by subject × condition."""
    return df.groupby(["subject", "condition"])["correct"].agg(
        ["mean", "count", "sum"]
    ).reset_index().rename(columns={"mean": "accuracy", "count": "n", "sum": "correct_n"})


def accuracy_by_difficulty_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy broken down by difficulty × condition."""
    return df.groupby(["difficulty", "condition"])["correct"].agg(
        ["mean", "count"]
    ).reset_index().rename(columns={"mean": "accuracy", "count": "n"})


def accuracy_delta(df: pd.DataFrame, group_by: List[str] = None) -> pd.DataFrame:
    """
    Compute accuracy delta = accuracy(text_visual) - accuracy(text_only).
    Positive = visual helps. Negative = visual hurts.
    """
    group_by = group_by or ["subject"]
    pivot = df.pivot_table(
        index=group_by,
        columns="condition",
        values="correct",
        aggfunc="mean"
    ).reset_index()
    if "text_visual" in pivot.columns and "text_only" in pivot.columns:
        pivot["delta"] = pivot["text_visual"] - pivot["text_only"]
    return pivot


# ── Reasoning fidelity analysis ────────────────────────────────────────────

def visual_reference_rate(df: pd.DataFrame) -> pd.DataFrame:
    """What fraction of text_visual responses actually reference the visual?"""
    visual_df = df[df["condition"] == "text_visual"]
    return visual_df.groupby("subject")["references_visual"].mean().reset_index()


def fidelity_vs_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    For text_visual condition: does referencing the diagram correlate with correctness?
    """
    visual_df = df[df["condition"] == "text_visual"].copy()
    return visual_df.groupby("references_visual")["correct"].agg(
        ["mean", "count"]
    ).reset_index().rename(columns={"mean": "accuracy", "count": "n"})


# ── Information-theoretic measures ─────────────────────────────────────────

def estimate_mutual_information_proxy(df: pd.DataFrame) -> Dict[str, float]:
    """
    Estimate conditional mutual information proxy:
    I_proxy(answer; image | text) ≈ H(correct | text_only) - H(correct | text_visual)

    Where H is binary entropy. Positive = image reduces uncertainty (helps).
    Negative = image increases uncertainty (hurts). Near zero = image irrelevant.

    This is a behavioural proxy, not a true MI estimate, but it connects
    the empirical results to the information-theoretic framing of the MRes.
    """
    def binary_entropy(p: float) -> float:
        """H(p) = -p log p - (1-p) log(1-p)"""
        if p <= 0 or p >= 1:
            return 0.0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    results = {}
    for subject in SUBJECTS:
        subj = df[df["subject"] == subject]
        text_only = subj[subj["condition"] == "text_only"]["correct"].mean()
        text_visual = subj[subj["condition"] == "text_visual"]["correct"].mean()
        if pd.isna(text_only) or pd.isna(text_visual):
            continue
        h_text = binary_entropy(text_only)
        h_visual = binary_entropy(text_visual)
        mi_proxy = h_text - h_visual
        results[subject] = {
            "acc_text_only": round(text_only, 3),
            "acc_text_visual": round(text_visual, 3),
            "H_text_only": round(h_text, 4),
            "H_text_visual": round(h_visual, 4),
            "MI_proxy": round(mi_proxy, 4),
            "interpretation": "visual helps" if mi_proxy > 0.01
                              else "visual hurts" if mi_proxy < -0.01
                              else "visual neutral"
        }
    return results


def mcnemar_test(df: pd.DataFrame, subject: str = None) -> Dict:
    """
    McNemar's test for paired accuracy comparison (text_only vs text_visual).
    Tests H0: accuracy is the same in both conditions.
    """
    sub = df if subject is None else df[df["subject"] == subject]
    text_only  = sub[sub["condition"] == "text_only"].set_index("problem_id")["correct"]
    text_visual = sub[sub["condition"] == "text_visual"].set_index("problem_id")["correct"]
    common_ids = text_only.index.intersection(text_visual.index)
    if len(common_ids) < 4:
        return {"error": "insufficient paired data"}
    to = text_only[common_ids].values.astype(int)
    tv = text_visual[common_ids].values.astype(int)
    n01 = ((to == 0) & (tv == 1)).sum()  # text_only wrong, visual correct
    n10 = ((to == 1) & (tv == 0)).sum()  # text_only correct, visual wrong
    if n01 + n10 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n01": 0, "n10": 0}
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_val = stats.chi2.sf(stat, df=1)
    return {
        "statistic": round(stat, 4),
        "p_value": round(p_val, 4),
        "n01_visual_better": int(n01),
        "n10_text_better": int(n10),
        "significant": p_val < 0.05
    }


# ── Visualisation ──────────────────────────────────────────────────────────

def plot_accuracy_comparison(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Main figure: accuracy by subject and condition (4-panel)."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    fig.suptitle("Accuracy by Subject: Text-Only vs Text+Visual\n"
                 "(Pilot Study — Mock Results)", fontsize=13, fontweight="bold")

    acc = accuracy_by_subject_condition(df)
    for ax, subject in zip(axes, SUBJECTS):
        subj_data = acc[acc["subject"] == subject]
        conditions = ["text_only", "text_visual"]
        labels = ["Text Only", "Text + Visual"]
        accs = [subj_data[subj_data["condition"] == c]["accuracy"].values
                for c in conditions]
        accs = [a[0] if len(a) > 0 else 0 for a in accs]
        bars1 = ax.bar(labels[:1], accs[:1], color=[COLORS[subject]], alpha=1.0, edgecolor="white", linewidth=1.5, width=0.5)
        bars2 = ax.bar(labels[1:], accs[1:], color=[COLORS[subject]], alpha=0.6, edgecolor="white", linewidth=1.5, width=0.5)
        bars = list(bars1) + list(bars2)
        ax.set_ylim(0, 1.05)
        ax.set_title(subject.replace("_", " ").title(), fontsize=11, fontweight="bold",
                     color=COLORS[subject])
        ax.set_ylabel("Accuracy" if subject == "calculus" else "")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        for bar, acc_val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{acc_val:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "fig1_accuracy_comparison.png", dpi=150, bbox_inches="tight")
    return fig


def plot_accuracy_delta(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Figure: accuracy delta (visual - text_only) per subject × difficulty."""
    fig, ax = plt.subplots(figsize=(10, 5))
    delta = accuracy_delta(df, ["subject", "difficulty"])

    x = np.arange(len(SUBJECTS))
    width = 0.25
    for i, diff in enumerate(DIFFICULTIES):
        deltas = []
        for subj in SUBJECTS:
            row = delta[(delta["subject"] == subj) & (delta["difficulty"] == diff)]
            deltas.append(row["delta"].values[0] if len(row) > 0 else 0)
        bars = ax.bar(x + i * width, deltas, width,
                      label=diff.title(), color=DIFF_COLORS[diff], alpha=0.85,
                      edgecolor="white", linewidth=1)
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.fill_between([-0.3, len(SUBJECTS) - 0.1 + 2*width], [0, 0], [-0.5, -0.5],
                    alpha=0.05, color="red")
    ax.fill_between([-0.3, len(SUBJECTS) - 0.1 + 2*width], [0, 0], [0.5, 0.5],
                    alpha=0.05, color="green")
    ax.text(0.01, 0.03, "← Visual hurts", transform=ax.transAxes,
            color="red", fontsize=9, alpha=0.7)
    ax.text(0.01, 0.97, "Visual helps →", transform=ax.transAxes,
            va="top", color="green", fontsize=9, alpha=0.7)
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.replace("_", "\n").title() for s in SUBJECTS], fontsize=10)
    ax.set_ylabel("Accuracy Delta (Visual − Text Only)")
    ax.set_title("Effect of Visual Context on Accuracy by Subject & Difficulty\n"
                 "(Positive = visual helps, Negative = visual hurts)", fontsize=12)
    ax.legend(title="Difficulty", loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "fig2_accuracy_delta.png", dpi=150, bbox_inches="tight")
    return fig


def plot_mi_proxy(mi_results: Dict, save: bool = True) -> plt.Figure:
    """Figure: information-theoretic MI proxy per subject."""
    fig, ax = plt.subplots(figsize=(8, 5))
    subjects = list(mi_results.keys())
    mi_vals  = [mi_results[s]["MI_proxy"] for s in subjects]
    colors = [COLORS[s] for s in subjects]
    bars = ax.barh(subjects, mi_vals, color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    for bar, val in zip(bars, mi_vals):
        ax.text(val + (0.003 if val >= 0 else -0.003), bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=10)
    ax.set_xlabel("MI Proxy: H(correct|text) − H(correct|text+visual) [bits]")
    ax.set_title("Information-Theoretic Proxy for Visual Contribution\n"
                 "Positive = visual reduces uncertainty about correct answer", fontsize=11)
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels([s.replace("_", " ").title() for s in subjects])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "fig3_mi_proxy.png", dpi=150, bbox_inches="tight")
    return fig


def plot_reasoning_fidelity(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Figure: visual reference rate and its correlation with accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: visual reference rate by subject
    ref_rate = visual_reference_rate(df)
    ax1.barh(ref_rate["subject"],
             ref_rate["references_visual"],
             color=[COLORS[s] for s in ref_rate["subject"]],
             alpha=0.85, edgecolor="white")
    ax1.set_xlabel("Fraction of responses referencing visual")
    ax1.set_title("Reasoning Fidelity:\nHow Often Do Models Reference the Diagram?", fontsize=11)
    ax1.set_yticks(range(len(ref_rate["subject"])))
    ax1.set_yticklabels([s.replace("_", " ").title() for s in ref_rate["subject"]])
    ax1.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlim(0, 1)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", alpha=0.3)

    # Right: accuracy with and without visual reference
    fid = fidelity_vs_accuracy(df)
    labels = ["Did Not Reference\nVisual", "Referenced\nVisual"]
    accs = []
    for ref in [False, True]:
        row = fid[fid["references_visual"] == ref]["accuracy"].values
        accs.append(row[0] if len(row) > 0 else 0)
    ax2.bar(labels, accs, color=["#AAAAAA", "#2E6FAD"],
            alpha=0.85, edgecolor="white", width=0.4)
    for i, acc in enumerate(accs):
        ax2.text(i, acc + 0.02, f"{acc:.1%}", ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy vs Whether Model\nReferenced Visual Content", fontsize=11)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "fig4_reasoning_fidelity.png", dpi=150, bbox_inches="tight")
    return fig


def generate_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """Full results table for paper appendix."""
    agg = df.groupby(["subject", "difficulty", "condition"])["correct"].agg(
        accuracy="mean", n="count"
    ).reset_index()
    agg["accuracy_pct"] = (agg["accuracy"] * 100).round(1).astype(str) + "%"
    return agg


# ── Full analysis pipeline ─────────────────────────────────────────────────

def run_full_analysis(model: str = "mock-gpt4o", save_figures: bool = True):
    print("Loading results...")
    df = load_all_results(model)
    if df.empty:
        print("No results found. Run the experiment first.")
        return

    print(f"Loaded {len(df)} evaluation records.")
    print(f"Conditions: {df['condition'].unique()}")
    print(f"Subjects:   {df['subject'].unique()}\n")

    print("─" * 50)
    print("ACCURACY BY CONDITION")
    print(accuracy_by_condition(df).to_string(index=False))

    print("\n─" * 50)
    print("ACCURACY BY SUBJECT × CONDITION")
    print(accuracy_by_subject_condition(df).to_string(index=False))

    print("\n─" * 50)
    print("ACCURACY DELTA (visual − text_only) BY SUBJECT")
    delta = accuracy_delta(df, ["subject"])
    print(delta.to_string(index=False))

    print("\n─" * 50)
    print("INFORMATION-THEORETIC MI PROXY")
    mi = estimate_mutual_information_proxy(df)
    for subj, vals in mi.items():
        print(f"  {subj:20s}: MI_proxy={vals['MI_proxy']:+.4f}  ({vals['interpretation']})")

    print("\n─" * 50)
    print("REASONING FIDELITY (text_visual only)")
    print(visual_reference_rate(df).to_string(index=False))
    print("\nAccuracy by visual reference:")
    print(fidelity_vs_accuracy(df).to_string(index=False))

    print("\n─" * 50)
    print("MCNEMAR TESTS (H0: no difference between conditions)")
    for subj in SUBJECTS:
        result = mcnemar_test(df, subject=subj)
        sig = "**significant**" if result.get("significant") else "not significant"
        p_val = result.get('p_value', 'N/A')
        p_str = f"{p_val:.4f}" if isinstance(p_val, float) else str(p_val)
        print(f"  {subj:20s}: p={p_str}  ({sig})")

    print("\n─" * 50)
    print("Generating figures...")
    plot_accuracy_comparison(df, save=save_figures)
    plot_accuracy_delta(df, save=save_figures)
    mi_data = estimate_mutual_information_proxy(df)
    plot_mi_proxy(mi_data, save=save_figures)
    plot_reasoning_fidelity(df, save=save_figures)
    print(f"Figures saved to {FIGURES_DIR}/")

    return {
        "accuracy_by_condition": accuracy_by_condition(df),
        "accuracy_by_subject": accuracy_by_subject_condition(df),
        "delta": delta,
        "mi_proxy": mi,
    }


if __name__ == "__main__":
    run_full_analysis()
