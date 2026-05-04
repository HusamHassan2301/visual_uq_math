# Experimental Results

## LLaVA-7B Full Experiment (60 problems, May 2026)

**Model:** LLaVA-7B via Ollama (CPU inference)  
**Dataset:** 60 problems across 4 subjects × 3 difficulty levels  
**Conditions:** text-only (n=60) and text+visual (n=46)

---

### Overall Accuracy

| Condition | n | Correct | Accuracy |
|---|---|---|---|
| Text-only | 60 | 27 | 45.0% |
| Text+Visual | 46 | 23 | 50.0% |
| **Delta** | — | — | **+5.0 pp** |

---

### Accuracy by Subject × Condition

| Subject | Text-only | n | Text+Visual | n | Δ | MI Proxy (bits) |
|---|---|---|---|---|---|---|
| Calculus | 60.0% | 15 | 60.0% | 15 | 0.0 pp | 0.000 |
| Combinatorics | 33.3% | 15 | 15.4% | 13 | **−17.9 pp** | +0.299 |
| Linear Algebra | 53.3% | 15 | 72.7% | 11 | **+19.4 pp** | +0.151 |
| Number Theory | 33.3% | 15 | 57.1% | 7 | **+23.8 pp** | −0.067 |

---

### Reasoning Fidelity

| Subject | Visual Reference Rate |
|---|---|
| Number Theory | 85.7% |
| Combinatorics | 69.2% |
| Calculus | 66.7% |
| Linear Algebra | 63.6% |
| **Overall** | **69.6%** |

**The Fidelity Paradox:**

| Visual Engagement | Accuracy | n |
|---|---|---|
| References diagram | 43.8% | 32 |
| Does NOT reference diagram | 64.3% | 14 |
| **Difference** | **−20.5 pp** | — |

---

### Pilot (12 problems) vs Full Study (60 problems)

| Metric | 12-problem pilot | 60-problem full |
|---|---|---|
| Overall delta | +11.7 pp | +5.0 pp |
| Number theory Δ | +66.7 pp | +23.8 pp |
| Linear algebra Δ | 0.0 pp | +19.4 pp |
| Combinatorics Δ | 0.0 pp | −17.9 pp |
| Fidelity direction | ref → higher acc | ref → lower acc |

---

### Reproduce These Results

```bash
# Full 60-problem run
python run_experiment.py --mode llava --backend ollama --n_problems 60

# Re-run analysis on saved results
python reanalyse.py
```
