# Visual UQ Math

A pilot study on whether programmatically generated diagrams improve mathematical reasoning in open-source vision-language models. Evaluated using LLaVA-7B on 60 problems across four subjects at university and Olympiad difficulty.

This is a preliminary experiment accompanying an MRes research proposal in uncertainty quantification and multimodal AI.

---

## Research Question

When a vision-language model is given a diagram alongside a university-level mathematics problem, does the visual input genuinely improve reasoning — or does the model ignore it?

This is evaluated through two lenses: accuracy (does adding a diagram change correctness?) and reasoning fidelity (does the model's chain-of-thought actually reference the visual?).

---

## Results (LLaVA-7B, 60 problems)

| Subject | Text-only | Text+Visual | Delta |
|---|---|---|---|
| Calculus | 60.0% | 60.0% | 0.0 pp |
| Combinatorics | 33.3% | 15.4% | −17.9 pp |
| Linear Algebra | 53.3% | 72.7% | +19.4 pp |
| Number Theory | 33.3% | 57.1% | +23.8 pp |
| Overall | 45.0% | 50.0% | +5.0 pp |

The most unexpected finding: when LLaVA explicitly references the diagram in its reasoning, accuracy is 43.8% — compared to 64.3% when it does not reference it. Visual engagement correlates negatively with correctness.

---

## Repository Structure

```
visual_uq_math/
├── src/
│   ├── dataset.py           # 60-problem dataset with stratified sampling
│   ├── image_gen.py         # Diagram generation (matplotlib, sympy, networkx)
│   ├── evaluator.py         # Type-aware answer checker
│   ├── llava_evaluator.py   # LLaVA inference via Ollama or HuggingFace
│   └── analysis.py          # Accuracy, MI proxy, McNemar, reasoning fidelity
├── paper/
│   └── draft_paper.md       # Draft paper with full experimental results
├── results/
│   ├── raw/                 # JSON results per model and condition
│   └── figures/             # Generated plots
├── data/
│   └── pilot_problems.json  # Problem dataset (auto-generated on first run)
├── run_experiment.py        # Main pipeline entry point
└── reanalyse.py             # Re-run analysis on saved results
```

---

## Setup

**Requirements:** Python 3.10+, Ollama, LLaVA

```bash
# Install Ollama from https://ollama.com then pull the model
ollama pull llava

# Clone and install dependencies
git clone https://github.com/HusamHassan2301/visual_uq_math
cd visual_uq_math

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the Experiment

```bash
# Quick test — 12 problems, roughly 30 minutes on CPU
python run_experiment.py --mode llava --backend ollama --n_problems 12

# Full experiment — 60 problems, roughly 4-8 hours on CPU
python run_experiment.py --mode llava --backend ollama --n_problems 60

# Re-run analysis on already saved results
python reanalyse.py

# Mock mode — instant, no model needed
python run_experiment.py --mode mock
```

---

## Other Run Options

```bash
# Larger model
python run_experiment.py --mode llava --backend ollama --ollama_model llava:13b

# HuggingFace backend (for GPU servers or HPC clusters)
python run_experiment.py --mode llava --backend hf \
    --hf_model llava-hf/llava-1.5-7b-hf --quantize 4bit

# GPT-4o (requires OPENAI_API_KEY in .env)
python run_experiment.py --mode api --model gpt4o --n_problems 20
```

---

## Methods

**Dataset:** 60 hand-curated problems across calculus, combinatorics, linear algebra, and number theory. Each subject has 5 problems at each of three difficulty levels: university, competition, and Olympiad. Problems are self-contained but have natural visual representations.

**Diagrams:** Generated programmatically using matplotlib, sympy, and networkx. Each problem type maps to a specific visual format — function plots for calculus, graph diagrams for combinatorics, eigenspace visualisations for linear algebra, modular grids for number theory.

**Evaluation:** Each problem is run in two conditions — text-only and text+visual. Accuracy is measured using a three-tier type-aware checker: exact numeric matching, symbolic equality via sympy, and keyword overlap for proof-style answers.

**Metrics:** Accuracy delta (text+visual minus text-only), visual reference rate (how often the model mentions the diagram), and an MI proxy (binary entropy reduction from visual context).

---

## Paper

The full draft paper is in `paper/draft_paper.md`. It covers the experimental design, all results tables, discussion of the fidelity paradox, and a proposed three-phase MRes research programme connecting these findings to Bayesian uncertainty quantification and imprecise probability.

---

## Requirements

```
matplotlib, sympy, networkx, pillow, numpy, pandas, scipy
python-dotenv, tqdm, requests
transformers, accelerate  (for HuggingFace backend)
openai                    (for GPT-4o mode)
```

---

## References

- Zhang et al. (2024). MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems? ECCV 2024.
- Lu et al. (2023). MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts. NeurIPS.
- Fang et al. (2024). MathOdyssey: Benchmarking Mathematical Problem-Solving Skills in LLMs. Scientific Data.
- Liu et al. (2023). Visual Instruction Tuning (LLaVA). NeurIPS.
- Gal and Ghahramani (2016). Dropout as a Bayesian Approximation. ICML.
- Caprio et al. (2023). Credal Bayesian Deep Learning. arXiv:2302.09656.
