"""
reanalyse.py — Re-run answer extraction + type-aware correctness checking.
Usage:
    python reanalyse.py
    python reanalyse.py --model llava-ollama-llava
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

RESULTS_DIR   = Path("results/raw")
PROBLEMS_FILE = Path("data/pilot_problems.json")


def load_ground_truth() -> dict:
    """Return {problem_id: {answer, answer_type}} from pilot_problems.json."""
    if not PROBLEMS_FILE.exists():
        from dataset import create_pilot_dataset
        create_pilot_dataset()
    with open(PROBLEMS_FILE) as f:
        problems = json.load(f)
    return {p["id"]: {"answer": p["answer"],
                      "answer_type": p.get("answer_type", "auto")}
            for p in problems}


def reextract_and_recheck(model: str = "llava-ollama-llava") -> int:
    from llava_evaluator import _extract_answer
    from evaluator import _check_answer

    gt_lookup   = load_ground_truth()
    total_flips = 0

    for condition in ["text_only", "text_visual"]:
        path = RESULTS_DIR / f"{model}_{condition}.json"
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue

        with open(path) as f:
            records = json.load(f)

        n_flips = 0
        for r in records:
            pid         = r["problem_id"]
            reasoning   = r.get("reasoning_chain", "")
            old_correct = r.get("correct", False)

            # Ground truth — prefer stored, fall back to problems file
            gt_info     = gt_lookup.get(pid, {})
            gt          = r.get("answer", "") or gt_info.get("answer", "")
            answer_type = gt_info.get("answer_type", "auto")
            r["answer"] = gt

            # Re-extract + re-check
            new_pred    = _extract_answer(reasoning)
            r["predicted_answer"] = new_pred

            if gt:
                new_correct = _check_answer(new_pred, gt, answer_type)
                r["correct"] = new_correct
                if new_correct != old_correct:
                    n_flips += 1
                    arrow = "✗→✓" if new_correct else "✓→✗"
                    print(f"    {arrow} {pid} [{answer_type}]: "
                          f"pred='{new_pred[:40]}' truth='{gt[:40]}'")

        with open(path, "w") as f:
            json.dump(records, f, indent=2)

        n_correct = sum(r["correct"] for r in records)
        print(f"  {path.name}: {len(records)} records, "
              f"{n_flips} flips, accuracy={n_correct/len(records):.1%}")
        total_flips += n_flips

    return total_flips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava-ollama-llava")
    args = parser.parse_args()

    print(f"Re-analysing: {args.model}\n")
    print("[1/2] Re-extracting + re-checking with type-aware checker...")
    n = reextract_and_recheck(args.model)
    print(f"\n      Total correctness flips: {n}")

    print("\n[2/2] Running full analysis...")
    from analysis import run_full_analysis
    run_full_analysis(args.model, save_figures=True)
    print("\nDone. Figures saved to results/figures/")

if __name__ == "__main__":
    main()
