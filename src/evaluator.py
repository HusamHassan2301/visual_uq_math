"""
evaluator.py
------------
Runs model evaluation across text-only and text+visual conditions.
Supports: mock mode (no API key), OpenAI GPT-4o, Google Gemini.

Usage:
    evaluator = Evaluator(model="mock")
    results = evaluator.run(problems, condition="text_only")
"""

import os
import json
import time
import random
import base64
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field

from dotenv import load_dotenv
load_dotenv()

RESULTS_DIR = Path(__file__).parent.parent / "results" / "raw"

# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class EvalResult:
    problem_id: str
    subject: str
    difficulty: str
    condition: str           # "text_only" | "text_visual"
    model: str
    predicted_answer: str
    reasoning_chain: str
    correct: bool
    references_visual: bool  # does chain-of-thought mention the diagram?
    tokens_used: int
    latency_s: float
    answer: str = ""         # ground truth answer — stored for re-evaluation
    metadata: dict = field(default_factory=dict)


# ── Prompt templates ───────────────────────────────────────────────────────

TEXT_ONLY_PROMPT = """You are an expert mathematician. Solve the following problem step by step.
Show all your working clearly. After your reasoning, state your final answer on a new line 
starting with "ANSWER:".

Problem: {problem_text}"""

TEXT_VISUAL_PROMPT = """You are an expert mathematician. You are given a mathematical problem
and an associated diagram. Use both the text and the diagram in your reasoning.
Show all your working clearly. After your reasoning, state your final answer on a new line 
starting with "ANSWER:".

Problem: {problem_text}

[A diagram has been provided. Refer to it explicitly in your reasoning if it is helpful.]"""


# ── Mock evaluator ─────────────────────────────────────────────────────────

# Synthetic accuracy rates based on MathVerse findings
# Models perform slightly worse with visual context on non-geometric problems
MOCK_ACCURACY = {
    ("calculus",       "university",  "text_only"):   0.80,
    ("calculus",       "university",  "text_visual"):  0.78,
    ("calculus",       "competition", "text_only"):   0.55,
    ("calculus",       "competition", "text_visual"):  0.60,  # diagrams help here
    ("calculus",       "olympiad",    "text_only"):   0.25,
    ("calculus",       "olympiad",    "text_visual"):  0.22,
    ("combinatorics",  "university",  "text_only"):   0.75,
    ("combinatorics",  "university",  "text_visual"):  0.70,
    ("combinatorics",  "competition", "text_only"):   0.50,
    ("combinatorics",  "competition", "text_visual"):  0.45,
    ("combinatorics",  "olympiad",    "text_only"):   0.20,
    ("combinatorics",  "olympiad",    "text_visual"):  0.18,
    ("linear_algebra", "university",  "text_only"):   0.85,
    ("linear_algebra", "university",  "text_visual"):  0.83,
    ("linear_algebra", "competition", "text_only"):   0.60,
    ("linear_algebra", "competition", "text_visual"):  0.62,
    ("linear_algebra", "olympiad",    "text_only"):   0.30,
    ("linear_algebra", "olympiad",    "text_visual"):  0.28,
    ("number_theory",  "university",  "text_only"):   0.70,
    ("number_theory",  "university",  "text_visual"):  0.60,  # diagrams hurt
    ("number_theory",  "competition", "text_only"):   0.45,
    ("number_theory",  "competition", "text_visual"):  0.35,  # diagrams hurt
    ("number_theory",  "olympiad",    "text_only"):   0.15,
    ("number_theory",  "olympiad",    "text_visual"):  0.12,
}

MOCK_VISUAL_REFERENCE_RATE = {
    "text_visual": 0.35,   # only 35% of chains actually reference the diagram
    "text_only":   0.00,
}

MOCK_REASONING_TEMPLATES = {
    True: [
        "Let me work through this systematically.\n\nFirst, I identify the key components of the problem.\n\nApplying the relevant theorem/technique:\n\nComputing step by step:\n\nTherefore the result follows.\n\nANSWER: {answer}",
        "This problem requires careful analysis.\n\nSetting up the framework:\n\nApplying the standard approach:\n\nThe computation yields:\n\nANSWER: {answer}",
    ],
    False: [
        "Let me attempt this problem.\n\nI'll try the following approach:\n\nThis gives us approximately:\n\nANSWER: incorrect_value",
        "Working through the problem:\n\nApplying a related technique:\n\nThis suggests the answer is:\n\nANSWER: wrong_answer",
    ]
}

VISUAL_REFERENCE_PHRASES = [
    "Looking at the diagram, ",
    "As shown in the figure, ",
    "From the visual representation, ",
    "The diagram illustrates that ",
    "Referring to the graph, ",
]


def _mock_reasoning(correct: bool, answer: str, condition: str, subject: str) -> tuple:
    """Generate plausible mock reasoning chain."""
    templates = MOCK_REASONING_TEMPLATES[correct]
    template = random.choice(templates)
    chain = template.format(answer=answer if correct else "wrong")

    # Inject visual reference with some probability
    ref_rate = MOCK_VISUAL_REFERENCE_RATE[condition]
    references_visual = False
    if condition == "text_visual" and random.random() < ref_rate:
        phrase = random.choice(VISUAL_REFERENCE_PHRASES)
        chain = phrase + chain[0].lower() + chain[1:]
        references_visual = True

    return chain, references_visual


class MockEvaluator:
    """Deterministic mock evaluator for testing without API keys."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.model_name = "mock-gpt4o"

    def evaluate_one(self, problem, condition: str) -> EvalResult:
        key = (problem.subject, problem.difficulty, condition)
        p_correct = MOCK_ACCURACY.get(key, 0.5)
        correct = random.random() < p_correct
        reasoning, ref_visual = _mock_reasoning(
            correct, problem.answer, condition, problem.subject
        )
        return EvalResult(
            problem_id=problem.id,
            subject=problem.subject,
            difficulty=problem.difficulty,
            condition=condition,
            model=self.model_name,
            predicted_answer=problem.answer if correct else "incorrect",
            reasoning_chain=reasoning,
            correct=correct,
            references_visual=ref_visual,
            tokens_used=random.randint(300, 800),
            latency_s=round(random.uniform(0.5, 3.0), 2),
        )

    def run(self, problems: list, condition: str) -> List[EvalResult]:
        results = []
        for p in problems:
            if condition == "text_visual" and not p.has_natural_visual:
                continue
            result = self.evaluate_one(p, condition)
            results.append(result)
        return results


# ── OpenAI GPT-4o evaluator ────────────────────────────────────────────────

class GPT4oEvaluator:
    """Real evaluator using OpenAI GPT-4o API."""

    def __init__(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = "gpt-4o"
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def _build_messages(self, problem, condition: str, image_b64: Optional[str]) -> list:
        if condition == "text_only" or image_b64 is None:
            prompt = TEXT_ONLY_PROMPT.format(problem_text=problem.text)
            return [{"role": "user", "content": prompt}]
        else:
            prompt = TEXT_VISUAL_PROMPT.format(problem_text=problem.text)
            return [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_b64}"
                }}
            ]}]

    def evaluate_one(self, problem, condition: str,
                     image_b64: Optional[str] = None) -> EvalResult:
        messages = self._build_messages(problem, condition, image_b64)
        t0 = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1000,
            temperature=0.0,
        )
        latency = time.time() - t0
        reasoning = response.choices[0].message.content
        tokens = response.usage.total_tokens

        # Extract answer
        predicted = ""
        for line in reasoning.split("\n"):
            if line.strip().startswith("ANSWER:"):
                predicted = line.replace("ANSWER:", "").strip()
                break

        # Check correctness (simple string match for pilot)
        correct = _check_answer(predicted, problem.answer)

        # Check visual reference
        ref_visual = condition == "text_visual" and any(
            phrase.lower() in reasoning.lower()
            for phrase in ["diagram", "figure", "graph", "plot", "image", "visual", "chart"]
        )

        return EvalResult(
            problem_id=problem.id,
            subject=problem.subject,
            difficulty=problem.difficulty,
            condition=condition,
            model=self.model_name,
            predicted_answer=predicted,
            reasoning_chain=reasoning,
            correct=correct,
            references_visual=ref_visual,
            tokens_used=tokens,
            latency_s=round(latency, 2),
        )

    def run(self, problems: list, condition: str,
            image_dir: Optional[Path] = None) -> List[EvalResult]:
        results = []
        for i, p in enumerate(problems):
            if condition == "text_visual" and not p.has_natural_visual:
                continue
            image_b64 = None
            if condition == "text_visual" and image_dir:
                img_path = image_dir / f"{p.id}_visual.png"
                if img_path.exists():
                    with open(img_path, "rb") as f:
                        image_b64 = base64.b64encode(f.read()).decode()
            print(f"  [{i+1}/{len(problems)}] {p.id} ({condition})...")
            result = self.evaluate_one(p, condition, image_b64)
            results.append(result)
            time.sleep(0.5)  # rate limiting
        return results


# ── Answer checking ────────────────────────────────────────────────────────

def _check_answer(predicted: str, ground_truth: str,
                  answer_type: str = "auto") -> bool:
    """
    Type-aware answer checker for mathematical reasoning evaluation.

    answer_type controls matching strategy:
      "integer"    — exact integer match
      "float"      — float equality within 1e-4
      "expression" — symbolic / proof sketch matching
      "auto"       — infer from ground truth content (default)

    For "expression" answers, we use keyword-overlap (Jaccard ≥ 0.25
    or ≥33% of truth keywords in prediction), which handles proof sketches
    and symbolic expressions that LLaVA may phrase differently.
    """
    import re

    pred_raw  = predicted.strip()
    truth_raw = ground_truth.strip()

    if not pred_raw or not truth_raw:
        return False

    # ── Infer type when auto ───────────────────────────────────────────────
    if answer_type == "auto":
        if re.match(r'^-?\d+$', truth_raw.strip()):
            answer_type = "integer"
        elif re.match(r'^-?\d+\.\d+$', truth_raw.strip()):
            answer_type = "float"
        else:
            answer_type = "expression"

    # ── Normalise helper ───────────────────────────────────────────────────
    def _norm(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r'\\boxed\{([^}]+)\}', r'\1', s)
        s = re.sub(r'\$([^$]+)\$', r'\1', s)
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    pred  = _norm(pred_raw)
    truth = _norm(truth_raw)

    # ── Integer matching ───────────────────────────────────────────────────
    if answer_type == "integer":
        # Extract first integer from prediction
        nums = re.findall(r'-?\d+', pred)
        return bool(nums and nums[0] == truth.strip())

    # ── Float matching ─────────────────────────────────────────────────────
    if answer_type == "float":
        try:
            pf = float(re.sub(r'[^\d.\-eE]', '', pred.split()[0]))
            tf = float(truth)
            return abs(pf - tf) < 1e-4
        except (ValueError, IndexError):
            return False

    # ── Expression / proof sketch matching ────────────────────────────────
    # First try exact normalised match
    if pred == truth:
        return True

    # Try sympy symbolic equality for pure expressions
    try:
        import sympy
        def _to_sympy(s):
            s = re.sub(r'π|\\pi', 'pi', s)
            s = re.sub(r'e\^(\w+)', r'exp(\1)', s)
            s = re.split(r'\s+for\s+|\s+where\s+|\s+when\s+', s)[0]
            return sympy.sympify(s, evaluate=True)
        ps = _to_sympy(pred)
        ts = _to_sympy(truth)
        if sympy.simplify(ps - ts) == 0:
            return True
    except Exception:
        pass

    # Keyword-overlap for proof sketches and verbose expressions
    STOP = {"a","an","the","of","in","is","are","be","to","and","or",
            "for","by","via","we","it","that","this","at","on","with",
            "have","can","as","from","not","but","if","then","so","its",
            "let","all","any","since","thus","hence","show","prove","get"}

    def keywords(text: str) -> set:
        words = re.findall(r"[a-zA-Z']+", text.lower())
        return {w for w in words if w not in STOP and len(w) > 2}

    pred_kw  = keywords(pred_raw)
    truth_kw = keywords(truth_raw)

    if not truth_kw:
        return True

    overlap    = pred_kw & truth_kw
    union      = pred_kw | truth_kw
    jaccard    = len(overlap) / len(union) if union else 0
    core_hits  = len(overlap) / len(truth_kw) if truth_kw else 0

    return jaccard >= 0.25 or core_hits >= 0.33


# ── Result persistence ─────────────────────────────────────────────────────

def save_results(results: List[EvalResult], model: str, condition: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{model}_{condition}.json"
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Saved {len(results)} results to {path}")


def load_results(model: str, condition: str) -> List[Dict]:
    path = RESULTS_DIR / f"{model}_{condition}.json"
    with open(path) as f:
        return json.load(f)


# ── Main entry ─────────────────────────────────────────────────────────────

def get_evaluator(mode: str = "mock"):
    if mode == "mock":
        return MockEvaluator()
    elif mode == "gpt4o":
        return GPT4oEvaluator()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'mock' or 'gpt4o'.")


if __name__ == "__main__":
    from dataset import load_pilot_problems, create_pilot_dataset, stratified_sample
    print("Creating dataset...")
    create_pilot_dataset()
    problems = load_pilot_problems()
    sample = stratified_sample(problems, n_per_cell=2)
    print(f"\nRunning mock evaluation on {len(sample)} problems...")
    ev = MockEvaluator()
    text_results = ev.run(sample, "text_only")
    visual_results = ev.run(sample, "text_visual")
    save_results(text_results, "mock-gpt4o", "text_only")
    save_results(visual_results, "mock-gpt4o", "text_visual")
    print("Done.")
