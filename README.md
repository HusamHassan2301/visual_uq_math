# Visual Uncertainty in Mathematical Reasoning
### Does Visual Context Help or Hurt LLMs on University-Level Mathematics?

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Pilot Study](https://img.shields.io/badge/status-pilot--study-orange.svg)]()

> **Pilot experiment supporting an MRes research proposal**  
> *Evaluating and Modelling Uncertainty in Machine Learning Systems*

---

## Overview

This repository contains the code, data pipeline, and draft paper for a pilot study investigating a fundamental open question in multimodal AI:

**When a vision-language model is given a diagram alongside a mathematical problem, does the visual input genuinely improve reasoning — or does the model ignore it?**

This question has been studied on geometry-heavy benchmarks (MathVista, MathVerse) but has **never been systematically evaluated on university and Olympiad-level difficulty** — precisely the level targeted by MathOdyssey (Fang et al., 2024). This pilot study fills that gap using a curated 60-problem subset spanning four non-geometric subject areas.

---

## Research Question

> *Do multimodal LLMs genuinely use visual representations when reasoning about university-level mathematics, and does this vary by subject area?*

This is evaluated through two lenses:

1. **Accuracy delta** — does adding a diagram improve or hurt correctness?
2. **Reasoning fidelity** — does the model's chain-of-thought actually reference the visual content?

---

## Project Structure

```
visual_uq_math/
│
├── src/
│   ├── dataset.py          # Dataset loading and stratified sampling
│   ├── image_gen.py        # Programmatic diagram generation (matplotlib/sympy)
│   ├── evaluator.py        # API runner — GPT-4o, Gemini, mock mode
│   ├── analysis.py         # Accuracy, delta, reasoning fidelity analysis
│   └── uncertainty.py      # Information-theoretic measures (mutual info proxy)
│
├── data/
│   ├── pilot_problems.json # 60 curated problems with metadata
│   └── README.md           # Data documentation
│
├── results/
│   ├── raw/                # JSON results per model per condition
│   └── figures/            # Generated charts and tables
│
├── notebooks/
│   └── pilot_analysis.ipynb  # End-to-end walkthrough notebook
│
├── paper/
│   └── draft_paper.md      # Draft paper (NeurIPS workshop style)
│
├── tests/
│   └── test_pipeline.py    # Unit tests for all modules
│
├── run_experiment.py       # Main entry point
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
└── README.md
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/visual_uq_math.git
cd visual_uq_math

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API keys (optional — mock mode works without keys)
cp .env.example .env
# Edit .env and add your OpenAI API key

# 4. Run in mock mode (no API key needed — uses synthetic results)
python run_experiment.py --mode mock --n_problems 20

# 5. Run with real API (requires OpenAI key)
python run_experiment.py --mode api --model gpt4o --n_problems 20

# 6. View results
python run_experiment.py --mode analyse
```

---

## Pilot Dataset

60 problems hand-curated across four non-geometric subject areas and three difficulty levels:

| Subject | Problems | Difficulty Range |
|---|---|---|
| Calculus & Analysis | 15 | University → Olympiad |
| Combinatorics | 15 | University → Olympiad |
| Linear Algebra | 15 | University → Olympiad |
| Number Theory | 15 | University → Olympiad |

Each problem exists in two conditions:
- **Text-only**: the problem statement as plain text
- **Text + Diagram**: the problem statement plus a programmatically generated visual representation

---

## Key Hypotheses

**H1 (Visual Null):** Adding a diagram does not significantly change accuracy on university-level problems (accuracy delta ≈ 0).

**H2 (Subject Interaction):** The effect of visual context varies by subject — diagrams may help on calculus (function plots) but hurt on number theory (no natural visual representation).

**H3 (Reasoning Fidelity):** Models that show accuracy improvement with visual input also show higher rates of visual reference in their chain-of-thought reasoning.

---

## Connecting to Uncertainty Research

This pilot connects to the broader uncertainty quantification agenda through an information-theoretic lens:

If `I(answer ; image | text) ≈ 0` across subject areas, visual input contributes no information beyond the text — the model's uncertainty about the correct answer is not reduced by the diagram. This is a measurable, quantifiable claim that connects directly to the MRes research programme on uncertainty in ML systems.

---

## Draft Paper

See [`paper/draft_paper.md`](paper/draft_paper.md) for the full draft written in NeurIPS workshop style. The paper presents:
- Motivation and related work
- Methodology
- Pilot results and analysis
- Discussion of implications for the MRes research programme

---

## Dependencies

```
datasets>=2.18.0
openai>=1.12.0
google-generativeai>=0.4.0
matplotlib>=3.8.0
sympy>=1.12
networkx>=3.2
pillow>=10.2.0
numpy>=1.26.0
pandas>=2.2.0
scipy>=1.12.0
python-dotenv>=1.0.0
tqdm>=4.66.0
```

---

## Citation

If you use this code or dataset in your work, please cite:

```bibtex
@misc{visualuq2025,
  title  = {Visual Uncertainty in Mathematical Reasoning: Does Diagram Context Help or Hurt LLMs?},
  author = {[Author Name]},
  year   = {2025},
  note   = {Pilot study, MRes research proposal supporting document},
  url    = {https://github.com/YOUR_USERNAME/visual_uq_math}
}
```

---

## Related Work

- **MathOdyssey** (Fang et al., 2024) — University and Olympiad-level text math benchmark
- **MathVista** (Lu et al., 2023) — Multimodal math benchmark (geometry-heavy)
- **MathVerse** (Zhang et al., 2024) — Visual ablation study for multimodal math
- **Credal Bayesian Deep Learning** (Caprio et al., 2023) — Imprecise probability UQ
- **Uncertainty in Deep Learning** (Gal, 2016) — Foundational Bayesian UQ framework

---

*This repository is part of an MRes application portfolio. The pilot study demonstrates research capability and methodological competence in the proposed area of study.*
