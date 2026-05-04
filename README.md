<div align="center">

# 📐 Visual UQ Math

### *Do Vision-Language Models Actually Use Diagrams in Advanced Mathematics?*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![LLaVA](https://img.shields.io/badge/Model-LLaVA--7B-orange)](https://ollama.com/library/llava)
[![Ollama](https://img.shields.io/badge/Runtime-Ollama-black)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**A pilot study evaluating visual information integration in mathematical reasoning using open-source VLMs.**

*MRes Research Pilot · University-to-Olympiad Level · 60 Problems · 4 Subjects*

</div>

---

## 🔬 What This Is

A reproducible pilot experiment investigating whether visual context (programmatically generated diagrams) genuinely improves mathematical reasoning in large vision-language models — specifically **LLaVA-7B** running locally via Ollama.

The experiment targets **four non-geometric subjects** at three difficulty levels:

| Subject | Difficulty | N problems |
|---|---|---|
| Calculus | University → Olympiad | 15 |
| Combinatorics | University → Olympiad | 15 |
| Linear Algebra | University → Olympiad | 15 |
| Number Theory | University → Olympiad | 15 |

Each problem is evaluated in two conditions: **text-only** and **text + visual** (programmatic diagram).

---

## 📊 Key Results (LLaVA-7B, 60 problems)

| Subject | Text-only | Text+Visual | Δ | MI Proxy |
|---|---|---|---|---|
| Calculus | 60.0% | 60.0% | 0.0 pp | 0.000 bits |
| Combinatorics | 33.3% | 15.4% | **−17.9 pp** | +0.299 bits |
| Linear Algebra | 53.3% | 72.7% | **+19.4 pp** | +0.151 bits |
| Number Theory | 33.3% | 57.1% | **+23.8 pp** | −0.067 bits |
| **Overall** | **45.0%** | **50.0%** | **+5.0 pp** | — |

### 🔑 The Fidelity Paradox

> When LLaVA **references** the diagram: **43.8% accuracy**  
> When LLaVA **ignores** the diagram: **64.3% accuracy**

Visual engagement correlates **negatively** with correctness — a surprising finding that motivates deeper uncertainty-theoretic investigation.

---

## 🗂️ Repository Structure

```
visual_uq_math/
│
├── src/
│   ├── dataset.py           # 60-problem pilot dataset with stratified sampling
│   ├── image_gen.py         # Programmatic diagram generation (matplotlib, sympy, networkx)
│   ├── evaluator.py         # Type-aware answer checker (integer / symbolic / proof-sketch)
│   ├── llava_evaluator.py   # LLaVA inference via Ollama or HuggingFace
│   └── analysis.py          # Statistics: accuracy, MI proxy, McNemar, reasoning fidelity
│
├── paper/
│   └── draft_paper.md       # Full draft paper with real experimental results
│
├── results/
│   ├── raw/                 # JSON results (one file per model × condition)
│   └── figures/             # Auto-generated plots (fig1–fig4 + diagrams)
│
├── data/
│   └── pilot_problems.json  # Generated dataset (auto-created on first run)
│
├── notebooks/               # Exploratory analysis notebooks
├── tests/                   # Unit tests for pipeline components
│
├── run_experiment.py        # Main entry point — runs full pipeline
├── reanalyse.py             # Re-run answer extraction + analysis on saved results
└── requirements.txt
```

---

## 🚀 Quick Start

### Prerequisites

**1. Install Ollama** → [https://ollama.com](https://ollama.com)

**2. Pull LLaVA:**
```bash
ollama pull llava
```

**3. Set up Python environment:**
```bash
git clone https://github.com/HusamHassan2301/visual_uq_math
cd visual_uq_math

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Experiment

```bash
# Quick test — 12 problems (~30 min on CPU)
python run_experiment.py --mode llava --backend ollama --n_problems 12

# Full experiment — 60 problems (~4–8 hours on CPU)
python run_experiment.py --mode llava --backend ollama --n_problems 60
```

### Re-run Analysis on Saved Results

```bash
python reanalyse.py
```

### Mock Mode (no GPU/API needed — instant)

```bash
python run_experiment.py --mode mock
```

---

## ⚙️ All Run Modes

```bash
# LLaVA via Ollama (local — recommended)
python run_experiment.py --mode llava --backend ollama
python run_experiment.py --mode llava --backend ollama --ollama_model llava:13b

# LLaVA via HuggingFace (GPU server / HPC)
python run_experiment.py --mode llava --backend hf \
    --hf_model llava-hf/llava-1.5-7b-hf --quantize 4bit

# GPT-4o API (requires OPENAI_API_KEY in .env)
python run_experiment.py --mode api --model gpt4o --n_problems 20

# Generate diagrams only
python run_experiment.py --mode generate_diagrams

# Re-analyse existing results
python run_experiment.py --mode analyse --model llava-ollama-llava
```

---

## 📐 Methods

### Dataset

60 hand-curated problems spanning university, competition, and Olympiad difficulty levels across four non-geometric subjects. Problems are self-contained (solvable from text alone) but amenable to meaningful visual representation.

### Diagram Generation

Diagrams are generated programmatically using `matplotlib`, `sympy`, and `networkx`. Each problem type maps to a principled visual representation:

- **Calculus:** Function plots, Taylor approximation curves, area-under-curve shading
- **Combinatorics:** Pascal's triangle, lattice path grids, Ramsey/tournament graphs
- **Linear Algebra:** Eigenspace visualisations, vector projections, transformation diagrams
- **Number Theory:** Prime factor trees, modular arithmetic grids, Pell equation hyperbola plots

### Metrics

| Metric | Definition |
|---|---|
| **Accuracy** | Proportion correct per condition |
| **Accuracy Delta (Δ)** | `Acc(text+visual) − Acc(text-only)` |
| **Visual Reference Rate** | % of text+visual responses mentioning the diagram |
| **MI Proxy** | `H(correct\|text-only) − H(correct\|text+visual)` in bits |

### Answer Checker

Three-tier type-aware system:
1. **Integer/float** — exact numeric match within 1e-4
2. **Symbolic** — expression equality via sympy
3. **Proof/sketch** — keyword Jaccard overlap ≥ 0.25 or core-hit rate ≥ 33%

---

## 📄 Paper

The draft paper (`paper/draft_paper.md`) follows NeurIPS workshop format and includes:

- Full experimental results with real LLaVA-7B inference data
- Subject-level accuracy, MI-proxy, and fidelity analysis
- Discussion of the fidelity paradox
- Connection to Bayesian UQ and imprecise probability literature
- Proposed MRes research programme

---

## 🔧 Requirements

```
Core: matplotlib, sympy, networkx, pillow, numpy, pandas, scipy, python-dotenv, tqdm, requests
LLaVA (HF backend): transformers, accelerate, bitsandbytes (optional, for quantisation)
GPT-4o: openai
```

For HPC (Liverpool Barkla):
```bash
module load cuda/12.1
pip install transformers accelerate bitsandbytes --user
python run_experiment.py --mode llava --backend hf \
    --hf_model llava-hf/llava-1.5-7b-hf --quantize 4bit
```

---

## 📜 Citation

If you use this codebase or dataset, please cite:

```bibtex
@misc{sadig2026visualuqmath,
  author = {Hussam Sadig},
  title  = {Visual UQ Math: A Pilot Study of Visual Information Integration
             in Advanced Mathematical Reasoning},
  year   = {2026},
  url    = {https://github.com/HusamHassan2301/visual_uq_math}
}
```

---

## 📚 References

- Zhang et al. (2024). *MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?* ECCV 2024.
- Lu et al. (2023). *MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts.* NeurIPS.
- Fang et al. (2024). *MathOdyssey: Benchmarking Mathematical Problem-Solving Skills in LLMs.* Scientific Data.
- Liu et al. (2023). *Visual Instruction Tuning (LLaVA).* NeurIPS.
- Gal & Ghahramani (2016). *Dropout as a Bayesian Approximation.* ICML.
- Caprio et al. (2023). *Credal Bayesian Deep Learning.* arXiv:2302.09656.

---

<div align="center">
<sub>Pilot study for MRes by Research in Uncertainty Quantification and Multimodal AI</sub>
</div>
