# Visual Information in Advanced Mathematical Reasoning: A Pilot Study with LLaVA-7B

**Husam Hassan**  
husamsadig@gmail.com

---

## Abstract

The central finding of this pilot is unexpected: when LLaVA-7B explicitly references a diagram in its reasoning chain, its accuracy is 43.8% — compared to 64.3% when it ignores the diagram entirely. Visual engagement correlates negatively with correctness. This fidelity paradox, observed across 106 inference calls on a 60-problem dataset spanning calculus, combinatorics, linear algebra, and number theory, suggests that the relationship between visual input and mathematical reasoning in open-source VLMs is more complex than a simple "diagrams help" or "diagrams hurt" narrative.

Overall accuracy improves modestly from 45.0% (text-only) to 50.0% (text+visual), but this aggregate conceals large subject-level divergence: linear algebra gains +19.4 pp with visual context while combinatorics loses −17.9 pp. An information-theoretic MI proxy further complicates the picture — combinatorics shows a positive proxy (+0.299 bits) despite the accuracy drop, suggesting the diagrams carry relevant information that the model cannot correctly use. These findings motivate a proposed MRes programme in uncertainty quantification and multimodal AI.

---

## 1. Introduction

Can an open-source vision-language model use a diagram to improve its mathematical reasoning? The question has practical consequences: VLMs are increasingly deployed in educational contexts where mathematical reasoning with visual aids is routine. The dominant narrative from MathVerse (Zhang et al., 2024) suggests a pessimistic answer — models often score equally or worse when given visual context. Yet this finding emerged predominantly from geometry-heavy benchmarks where diagrams are structurally redundant with the text.

This pilot investigates the question in four non-geometric subjects — calculus, combinatorics, linear algebra, and number theory — where diagrams are supplementary rather than essential. Using LLaVA-7B on a curated 60-problem dataset, we find a more complex picture than any simple narrative captures. Subject-level effects are large and bidirectional; the information-theoretic proxy and raw accuracy sometimes disagree; and the reasoning fidelity metric produces a surprising inversion that we did not anticipate at the outset. The pipeline is fully open-source and runs locally via Ollama, requiring no API access.

---

## 2. Related Work

**Multimodal Mathematical Benchmarks.** MathVista (Lu et al., 2023) introduced 6,141 multimodal math problems, predominantly geometry. MathVerse (Zhang et al., 2024) isolated visual contribution through six problem variants from text-dominant to vision-only; their finding that models often perform as well or better in text-only conditions motivates the present work. MATH-Vision (Wang et al., 2024) introduced competition-level difficulty with visual context but remains geometry-heavy.

**MathOdyssey.** Fang et al. (2024) introduced a text-only benchmark of 387 university-to-Olympiad problems. No multimodal extension exists; this pilot is a step toward one.

**Open-Source VLMs.** LLaVA (Liu et al., 2023) demonstrated visual instruction tuning connecting CLIP encoders with language models. LLaVA-1.5 (7B) achieves competitive results while remaining locally runnable — a key reproducibility advantage over API-dependent research.

**Uncertainty in LLMs.** The question of when models are uncertain about their outputs connects to Bayesian deep learning (Gal & Ghahramani, 2016), conformal prediction (Vovk et al., 2005), and imprecise probability (Caprio et al., 2023). The fidelity paradox we observe — visual engagement correlating negatively with correctness — is precisely the kind of phenomenon that demands formal uncertainty modelling.

---

## 3. Methodology

### 3.1 Dataset

60 problems distributed evenly across four subjects and three difficulty levels:

| Subject | University | Competition | Olympiad | Total |
|---|---|---|---|---|
| Calculus | 5 | 5 | 5 | 15 |
| Combinatorics | 5 | 5 | 5 | 15 |
| Linear Algebra | 5 | 5 | 5 | 15 |
| Number Theory | 5 | 5 | 5 | 15 |

Of the 60 problems, 46 (76.7%) have natural visual representations and appear in the text+visual condition.

### 3.2 Diagram Generation

Diagrams generated programmatically using `matplotlib`, `sympy`, and `networkx`:

- **Calculus:** Function plots, area shading, Taylor approximation curves, convergence visualisations
- **Combinatorics:** Pascal's triangle, lattice path grids, tournament and Ramsey graphs
- **Linear Algebra:** Vector diagrams, eigenspace visualisations, projection illustrations
- **Number Theory:** Prime factor trees, modular arithmetic grids, Pell equation hyperbola plots

### 3.3 Model and Setup

**Model:** LLaVA-7B (llava:latest via Ollama), CPU inference, temperature 0 for deterministic output.

**Conditions:**
- **Text-only:** Problem statement with chain-of-thought instruction
- **Text+visual:** Problem statement + programmatically generated PNG diagram

**Scale:** 60 text-only + 46 text+visual = 106 total inference calls. Mean latency ~130s/problem on CPU (~3.8 hours total).

### 3.4 Metrics

**Accuracy (Acc):** Proportion correct per condition.

**Accuracy Delta (Δ):** `Acc(text+visual) − Acc(text-only)`.

**Visual Reference Rate (VRR):** Proportion of text+visual responses where reasoning explicitly mentions the diagram.

**MI Proxy:** `MI_proxy ≈ H(correct | text-only) − H(correct | text+visual)` where H is binary entropy. Positive = visual reduces uncertainty about correctness.

**Answer Checker:** Three-tier type-aware system: (1) integer/float exact matching, (2) sympy symbolic equality, (3) keyword-overlap Jaccard ≥ 0.25 or core-hit rate ≥ 33% for proof-sketch answers. Validated against ground truth before analysis.

---

## 4. Results

### 4.1 Overall Accuracy

| Condition | n | Correct | Accuracy |
|---|---|---|---|
| Text-only | 60 | 27 | **45.0%** |
| Text+visual | 46 | 23 | **50.0%** |
| **Delta** | — | — | **+5.0 pp** |

LLaVA-7B shows a modest but positive overall effect of visual context. This contrasts with MathVerse's finding of visual context hurting performance, though the effect here is small and subject composition is very different.

### 4.2 Accuracy by Subject × Condition

| Subject | Text-Only | n | Text+Visual | n | Δ | MI Proxy |
|---|---|---|---|---|---|---|
| Calculus | 60.0% (9/15) | 15 | 60.0% (9/15) | 15 | **0.0 pp** | 0.000 bits |
| Combinatorics | 33.3% (5/15) | 15 | 15.4% (2/13) | 13 | **−17.9 pp** | +0.299 bits |
| Linear Algebra | 53.3% (8/15) | 15 | 72.7% (8/11) | 11 | **+19.4 pp** | +0.151 bits |
| Number Theory | 33.3% (5/15) | 15 | 57.1% (4/7) | 7 | **+23.8 pp** | −0.067 bits |

**Key findings from this table:**

**Calculus** is completely neutral — visual context adds nothing. This suggests diagrams of standard calculus functions (derivatives, integrals) provide no additional information beyond what the text already conveys.

**Linear algebra** shows the strongest positive effect (+19.4 pp). Visual representations of eigenspaces, projections, and matrix transformations appear to genuinely help LLaVA reason about geometric linear algebra concepts.

**Number theory** shows a substantial improvement (+23.8 pp) despite a low MI proxy (−0.067 bits). The accuracy finding and MI proxy disagree here — likely because the text+visual subset (n=7) contains only problems amenable to visual representation, creating a selection effect.

**Combinatorics** is the most striking finding: a **−17.9 pp accuracy drop** with a **+0.299 bits MI proxy**. Visual context is informationally relevant to combinatorics — the diagrams (tournament graphs, Ramsey graphs, lattice paths) are mathematically meaningful — but LLaVA appears to use them in a way that increases errors. This suggests the model is attending to the wrong visual features, or that the visual representation creates interference with the combinatorial reasoning.

### 4.3 Reasoning Fidelity — The Fidelity Paradox

| Subject | Visual Reference Rate |
|---|---|
| Number Theory | 85.7% |
| Combinatorics | 69.2% |
| Calculus | 66.7% |
| Linear Algebra | 63.6% |
| **Overall** | **69.6%** |

LLaVA references the diagram in 69.6% of text+visual trials — substantially higher than the ~35% reported for GPT-4o in MathVerse.

**The fidelity paradox:**

| Visual Engagement | Accuracy | n |
|---|---|---|
| References diagram | **43.8%** | 32 |
| Does not reference | **64.3%** | 14 |
| **Difference** | **−20.5 pp** | — |

This is the most scientifically interesting finding of the full study. When LLaVA explicitly engages with the diagram in its reasoning, it is **20.5 percentage points less accurate** than when it ignores the diagram. This inverts the expected relationship.

**Possible interpretations:**

1. **Selective disengagement:** The problems where LLaVA ignores the diagram may be problems where the text alone is sufficient and the diagram would add noise — a form of implicit uncertainty management.

2. **Distraction under load:** When LLaVA attempts to integrate visual information into a complex mathematical argument, the additional processing creates errors in the symbolic reasoning.

3. **Diagram quality interaction:** For some problem types, the programmatically generated diagrams may emphasise the wrong features, leading to incorrect reasoning when attended to.

### 4.4 MI Proxy Summary

| Subject | MI_proxy (bits) | Interpretation |
|---|---|---|
| Combinatorics | **+0.299** | Diagram is informative — but model uses it poorly |
| Linear Algebra | **+0.151** | Diagram genuinely helps performance |
| Calculus | 0.000 | Diagram is informationally neutral |
| Number Theory | **−0.067** | Diagram slightly increases uncertainty |

The combinatorics result is theoretically important: positive MI proxy with negative accuracy delta means the diagram carries information the model could in principle use, but does not use correctly. This is precisely the kind of uncertainty gap that Bayesian and information-theoretic frameworks could formalise.

---

## 5. Discussion

### 5.1 The Fidelity Paradox — Scientific Significance

The finding that visual engagement correlates negatively with correctness (−20.5 pp) is unexpected and requires careful interpretation. Three non-exclusive explanations deserve investigation:

**Causal hypothesis A — Distraction:** Attending to the visual increases cognitive load on the language model's reasoning process, leading to errors in symbolic manipulation.

**Causal hypothesis B — Selection effect:** Problems where LLaVA spontaneously ignores the visual may be problems where the diagram is genuinely redundant (easy text-only problems), leading to spurious correlation between non-reference and high accuracy.

**Causal hypothesis C — Diagram quality:** The programmatic diagrams, while mathematically principled, may not communicate the same information a human-drawn diagram would. When LLaVA engages with them, it may be using incorrect visual cues.

A full MRes programme would distinguish these hypotheses through controlled ablation: replacing programmatic with human-expert diagrams, varying diagram complexity, and applying conformal prediction to quantify model uncertainty per condition.

### 5.2 Why Combinatorics Shows Accuracy Drop Despite Positive MI Proxy

The combinatorics discrepancy is worth dwelling on. The MI proxy is +0.299 bits — meaning the diagram is correlated with correct outcomes at the population level — yet accuracy drops by 17.9 pp when the diagram is shown. The most plausible explanation is that the visual information is genuinely there, but LLaVA cannot read combinatorial graphs accurately enough to use it. Tournament graphs and Ramsey diagrams are informationally rich but visually dense; the model may be extracting noise rather than structure. This distinction — between information being available and information being usable — is exactly what I want to investigate more formally in the MRes.

### 5.3 Connection to Uncertainty Quantification

The fidelity paradox and the MI proxy–accuracy discrepancy both point to the same underlying phenomenon: **the model's subjective uncertainty is not well-calibrated with respect to the value of visual information**. A model with good epistemic uncertainty estimates would ideally:

- Reference the diagram when it reduces uncertainty about the answer
- Ignore the diagram when it increases uncertainty

LLaVA appears to do the opposite. This miscalibration is the central motivating phenomenon for the proposed MRes programme, which would apply:

- **Bayesian deep learning** (Gal, 2016): Epistemic uncertainty in model predictions across conditions
- **Conformal prediction** (Deisenroth et al., 2026): Distribution-free coverage guarantees for model outputs
- **Imprecise probability** (Caprio et al., 2023): Credal sets representing uncertainty when visual and textual evidence conflict

### 5.4 Limitations

The most significant limitation is sample size. With n=46 in the text+visual condition, the subject-level results — particularly number theory (n=7) — are too small for confident conclusions. The answer checker's keyword-overlap tier also introduces noise: some answers marked correct may not be, and vice versa. A human review of a random sample of 20 responses suggested the error rate is around 10–15%, which is enough to shift subject-level accuracy by several percentage points. The programmatic diagrams are another limitation; they are consistent and reproducible but may not capture the visual intuition that a human tutor's diagram would convey.

---

## 6. Proposed MRes Research Programme

The most immediate next step is scaling up: the current sample size is too small to draw confident conclusions at the subject level, and a single model (LLaVA-7B) is not enough to know whether the fidelity paradox is specific to this architecture or more general. Running the same pipeline on LLaVA-13B, LLaVA-NeXT, and InternVL-2 on a 300+ problem dataset would clarify both questions, and Liverpool Barkla would make the GPU inference feasible.

The more interesting theoretical question — which is what I want to focus on in the MRes — is whether the fidelity paradox can be formally explained. The observation that a model attends to visual information precisely when that information harms it points to something deeper than a benchmark finding: it suggests the model's internal uncertainty estimates are miscalibrated with respect to the value of visual evidence. Bayesian deep learning frameworks (Gal, 2016) provide tools for measuring epistemic uncertainty in model predictions, and conformal prediction (Deisenroth et al., 2026) could provide coverage guarantees that quantify when the model's outputs can be trusted across modalities. The imprecise probability literature (Caprio et al., 2023) is relevant too — when text and visual evidence conflict, credal sets may be a more honest representation of what the model actually knows than a single probability estimate.

I do not yet know which of these frameworks will be most tractable, and part of what I hope to get from the MRes is the mathematical grounding to make that judgement properly.

---

## 7. Conclusion

This pilot study reveals that visual context does not have a uniform effect on mathematical reasoning in open-source VLMs — it is strongly modulated by both subject and the model's own reasoning strategy. The most significant finding is a fidelity paradox: when LLaVA explicitly references the diagram in its reasoning, accuracy drops by 20.5 percentage points compared to when it ignores the diagram. This suggests fundamental miscalibration between the model's uncertainty about when visual information is useful and whether it actually is.

The subject-level results reveal a further paradox in combinatorics: visual context is informationally positive (MI proxy +0.299 bits) yet accuracy drops significantly (−17.9 pp), indicating that the model has access to relevant visual information but cannot use it correctly. These phenomena — information availability without information utilisation, and visual engagement without accuracy benefit — are precisely the kinds of gaps that formal uncertainty quantification frameworks are designed to diagnose.

The code, dataset, and results are all available at https://github.com/HusamHassan2301/visual_uq_math and can be reproduced locally without any API access.

---

## References

- Caprio, M. et al. (2023). *Credal Bayesian Deep Learning*. arXiv:2302.09656.
- Deisenroth, M.P. et al. (2026). *Uncertainty Quantification of Surrogate Models Using Conformal Prediction*. Machine Learning: Science and Technology.
- Fang, M. et al. (2024). *MathOdyssey: Benchmarking Mathematical Problem-Solving Skills in LLMs*. Scientific Data.
- Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML.
- Liu, H. et al. (2023). *Visual Instruction Tuning (LLaVA)*. NeurIPS.
- Lu, P. et al. (2023). *MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts*. NeurIPS.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
- Wang, K. et al. (2024). *MATH-Vision: Measuring Multimodal Mathematical Reasoning*. NeurIPS 2024.
- Zhang, R. et al. (2024). *MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?* ECCV 2024.

---

*Code: https://github.com/HusamHassan2301/visual_uq_math*  
*Reproduce: `python run_experiment.py --mode llava --backend ollama --n_problems 60`*  
*Re-analyse: `python reanalyse.py`*
