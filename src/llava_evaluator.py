"""
llava_evaluator.py
------------------
LLaVA evaluator for the Visual UQ Math pilot experiment.

Supports two backends:
  1. Ollama  — easiest for local machines (CPU or GPU)
             Install: https://ollama.com  then:  ollama pull llava
             Run:     python run_experiment.py --mode llava --backend ollama

  2. HuggingFace — best for HPC clusters (Liverpool Barkla, etc.)
             Requires: pip install transformers accelerate bitsandbytes pillow
             Run:      python run_experiment.py --mode llava --backend hf
                       python run_experiment.py --mode llava --backend hf --hf_model llava-hf/llava-1.5-13b-hf

Tested models
-------------
  Ollama  : llava:7b  (default), llava:13b, llava-phi3
  HF      : llava-hf/llava-1.5-7b-hf   (default)
            llava-hf/llava-1.5-13b-hf
            llava-hf/llava-v1.6-mistral-7b-hf  (LLaVA-1.6 / LLaVA-NeXT)

Output format is identical to MockEvaluator / GPT4oEvaluator so the
existing analysis.py pipeline works without any modification.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

# ── Reuse the shared EvalResult dataclass ─────────────────────────────────

from evaluator import EvalResult, save_results, _check_answer

RESULTS_DIR = Path(__file__).parent.parent / "results" / "raw"

# ── Shared prompt templates (same as GPT-4o for fair comparison) ──────────

TEXT_ONLY_PROMPT = (
    "You are an expert mathematician. Solve the following problem step by step.\n"
    "Show all your working clearly. After your reasoning, state your final answer "
    "on a new line starting with \"ANSWER:\".\n\n"
    "Problem: {problem_text}"
)

TEXT_VISUAL_PROMPT = (
    "You are an expert mathematician. You are given a mathematical problem and an "
    "associated diagram. Use both the text and the diagram in your reasoning.\n"
    "Show all your working clearly. After your reasoning, state your final answer "
    "on a new line starting with \"ANSWER:\".\n\n"
    "Problem: {problem_text}\n\n"
    "[A diagram has been provided above. Refer to it explicitly in your reasoning "
    "if it is helpful.]"
)

VISUAL_KEYWORDS = [
    "diagram", "figure", "graph", "plot", "image", "visual",
    "chart", "picture", "illustration", "shown", "depicted",
]


def _extract_answer(text: str) -> str:
    """
    Robust answer extractor for LLaVA output.

    LLaVA often ignores the 'ANSWER:' format instruction. We try multiple
    patterns in priority order before falling back to heuristics.

    Priority
    --------
    1. Explicit 'ANSWER: ...' tag (as instructed in the prompt)
    2. 'The answer is ...' / 'The final answer is ...' phrasing
    3. Boxed LaTeX  \\boxed{...}
    4. 'Thus/Hence/Therefore ...' conclusion lines containing a value
    5. Last non-empty line (final fallback)
    """
    # 1. Explicit ANSWER: tag (case-insensitive)
    for line in text.splitlines():
        if re.match(r'^\s*answer\s*:', line, re.IGNORECASE):
            ans = re.split(r'answer\s*:', line, flags=re.IGNORECASE, maxsplit=1)[1].strip()
            if ans:
                return ans

    # 2. "The answer is X" / "The final answer is X"
    m = re.search(
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s+[:\-]?\s*([^\n\.]{1,80})',
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).strip().rstrip('.')

    # 3. LaTeX \boxed{...}
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m.group(1).strip()

    # 4. "Thus/Hence/Therefore ..." conclusion lines
    m = re.search(
        r'(?:thus|hence|therefore|so)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([^\n\.]{1,80})',
        text, re.IGNORECASE
    )
    if m:
        candidate = m.group(1).strip().rstrip('.')
        # Only use if it looks like a value (contains a digit or equation)
        if re.search(r'\d|[a-zA-Z]\s*=', candidate):
            return candidate

    # 5. Last non-empty line fallback
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else "NO_ANSWER"


def _references_visual(text: str) -> bool:
    return any(kw in text.lower() for kw in VISUAL_KEYWORDS)


def _encode_image(path: Path) -> str:
    """Return base64-encoded PNG string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ══════════════════════════════════════════════════════════════════════════
#  BACKEND 1 — Ollama
# ══════════════════════════════════════════════════════════════════════════

class OllamaLLaVAEvaluator:
    """
    Runs LLaVA through a local Ollama server.

    Prerequisites
    -------------
    1. Install Ollama:  https://ollama.com
    2. Pull the model:  ollama pull llava          # 7B, ~4 GB
                        ollama pull llava:13b       # 13B, ~8 GB
    3. Ollama starts automatically as a background service on most systems.
       If not: ollama serve

    The default endpoint is http://localhost:11434 — override with
    the OLLAMA_HOST environment variable.
    """

    def __init__(self, model: str = "llava", host: str | None = None):
        try:
            import requests
            self._requests = requests
        except ImportError:
            raise ImportError("requests not installed. Run: pip install requests")

        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model_name = f"llava-ollama-{model.replace(':', '-')}"
        self._check_connection()

    def _check_connection(self):
        try:
            r = self._requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            tags = [m["name"] for m in r.json().get("models", [])]
            if not any(self.model.split(":")[0] in t for t in tags):
                print(f"  [WARNING] Model '{self.model}' not found in Ollama.")
                print(f"  Run:  ollama pull {self.model}")
                print(f"  Available: {tags}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.host}.\n"
                f"Make sure Ollama is running: ollama serve\n"
                f"Error: {e}"
            )

    def _call(self, prompt: str, image_b64: str | None = None,
              timeout: int = 300, retries: int = 2) -> tuple[str, float]:
        """
        Call Ollama /api/generate. Returns (response_text, latency_s).
        Retries up to `retries` times on timeout before giving up gracefully.
        Timeout is 300s (5 min) — safe for CPU inference on long problems.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,   # deterministic for reproducibility
                "num_predict": 1024,
            },
        }
        if image_b64:
            payload["images"] = [image_b64]

        last_error = None
        for attempt in range(retries + 1):
            try:
                t0 = time.time()
                r = self._requests.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=timeout,
                )
                r.raise_for_status()
                latency = time.time() - t0
                return r.json()["response"], latency
            except Exception as e:
                last_error = e
                if attempt < retries:
                    wait = 10 * (attempt + 1)
                    print(f"\n    [timeout/error — retrying in {wait}s, attempt {attempt+2}/{retries+1}]",
                          end="", flush=True)
                    time.sleep(wait)

        # All retries exhausted — return a graceful fallback so the run continues
        print(f"\n    [SKIPPED after {retries+1} attempts: {type(last_error).__name__}]",
              end="", flush=True)
        return "TIMEOUT — model did not respond in time. ANSWER: SKIP", 0.0

    def evaluate_one(
        self,
        problem,
        condition: str,
        image_b64: str | None = None,
    ) -> EvalResult:
        if condition == "text_only" or image_b64 is None:
            prompt = TEXT_ONLY_PROMPT.format(problem_text=problem.text)
            img = None
        else:
            prompt = TEXT_VISUAL_PROMPT.format(problem_text=problem.text)
            img = image_b64

        reasoning, latency = self._call(prompt, img)
        predicted = _extract_answer(reasoning)
        correct = _check_answer(predicted, problem.answer)
        ref_visual = condition == "text_visual" and _references_visual(reasoning)
        skipped = "TIMEOUT" in reasoning

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
            tokens_used=len(reasoning.split()),   # Ollama doesn't return token count
            latency_s=round(latency, 2),
            answer=problem.answer,                # store ground truth for re-evaluation
            metadata={"skipped": skipped},
        )

    def run(
        self,
        problems: list,
        condition: str,
        image_dir: Path | None = None,
    ) -> List[EvalResult]:
        results = []
        skipped = 0
        for i, p in enumerate(problems):
            if condition == "text_visual" and not p.has_natural_visual:
                continue
            image_b64 = None
            if condition == "text_visual" and image_dir:
                img_path = image_dir / f"{p.id}_visual.png"
                if img_path.exists():
                    image_b64 = _encode_image(img_path)
            print(f"  [{i+1}/{len(problems)}] {p.id} ({condition}) ... ", end="", flush=True)
            result = self.evaluate_one(p, condition, image_b64)
            if result.metadata.get("skipped"):
                status = "⚠ SKIP"
                skipped += 1
            else:
                status = "✓" if result.correct else "✗"
            print(f"{status}  ({result.latency_s:.1f}s)")
            results.append(result)

        if skipped:
            print(f"\n  [Warning] {skipped} problem(s) skipped due to timeout.")
            print(f"  Tip: close other apps to free RAM, or reduce --n_problems.")
        return results


# ══════════════════════════════════════════════════════════════════════════
#  BACKEND 2 — HuggingFace Transformers (GPU / HPC)
# ══════════════════════════════════════════════════════════════════════════

class HuggingFaceLLaVAEvaluator:
    """
    Runs LLaVA via HuggingFace Transformers — best for GPU servers / HPC.

    Prerequisites
    -------------
        pip install transformers accelerate pillow
        pip install bitsandbytes   # optional, for 4-bit quantisation on small GPUs

    Recommended models (HF hub IDs)
    --------------------------------
        llava-hf/llava-1.5-7b-hf          ~14 GB VRAM (or 8 GB with 4-bit)
        llava-hf/llava-1.5-13b-hf         ~27 GB VRAM (or 12 GB with 4-bit)
        llava-hf/llava-v1.6-mistral-7b-hf  LLaVA-NeXT (stronger reasoning)

    On Barkla HPC
    -------------
        module load cuda/12.1
        pip install transformers accelerate bitsandbytes pillow --user
        python run_experiment.py --mode llava --backend hf \\
               --hf_model llava-hf/llava-1.5-7b-hf --quantize 4bit
    """

    def __init__(
        self,
        hf_model: str = "llava-hf/llava-1.5-7b-hf",
        quantize: str | None = None,   # "4bit" | "8bit" | None
        device: str | None = None,
    ):
        try:
            import torch
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            from transformers import LlavaForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers not installed.\n"
                "Run: pip install transformers accelerate pillow"
            )

        self.model_name = f"llava-hf-{hf_model.split('/')[-1]}"
        print(f"  Loading {hf_model} ...")

        import torch
        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {self._device}")

        # Choose correct class depending on model variant
        is_next = "v1.6" in hf_model or "next" in hf_model.lower()

        load_kwargs: dict = {"low_cpu_mem_usage": True}
        if quantize == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif quantize == "8bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self._device == "cuda":
            load_kwargs["torch_dtype"] = torch.float16

        if is_next:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            self._processor = LlavaNextProcessor.from_pretrained(hf_model)
            self._model = LlavaNextForConditionalGeneration.from_pretrained(
                hf_model, **load_kwargs
            )
        else:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            self._processor = AutoProcessor.from_pretrained(hf_model)
            self._model = LlavaForConditionalGeneration.from_pretrained(
                hf_model, **load_kwargs
            )

        if quantize is None and self._device == "cuda":
            self._model = self._model.to(self._device)

        self._model.eval()
        print(f"  Model loaded.")

    def _call(self, prompt: str, image_path: Path | None = None) -> tuple[str, float]:
        """Run one inference pass. Returns (response_text, latency_s)."""
        from PIL import Image

        # LLaVA expects a specific conversation format
        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            image = None
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]

        # Apply chat template (handles <image> token placement)
        text_input = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self._processor(
            text=text_input,
            images=image,
            return_tensors="pt",
        ).to(self._device)

        t0 = time.time()
        with self._torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,       # greedy = deterministic
                temperature=1.0,       # ignored when do_sample=False
            )
        latency = time.time() - t0

        # Decode only the newly generated tokens (skip the prompt)
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        response = self._processor.decode(generated, skip_special_tokens=True)
        return response, latency

    def evaluate_one(
        self,
        problem,
        condition: str,
        image_path: Path | None = None,
    ) -> EvalResult:
        if condition == "text_only" or image_path is None:
            prompt = TEXT_ONLY_PROMPT.format(problem_text=problem.text)
            img = None
        else:
            prompt = TEXT_VISUAL_PROMPT.format(problem_text=problem.text)
            img = image_path

        reasoning, latency = self._call(prompt, img)
        predicted = _extract_answer(reasoning)
        correct = _check_answer(predicted, problem.answer)
        ref_visual = condition == "text_visual" and _references_visual(reasoning)

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
            tokens_used=len(reasoning.split()),
            latency_s=round(latency, 2),
        )

    def run(
        self,
        problems: list,
        condition: str,
        image_dir: Path | None = None,
    ) -> List[EvalResult]:
        results = []
        for i, p in enumerate(problems):
            if condition == "text_visual" and not p.has_natural_visual:
                continue
            image_path = None
            if condition == "text_visual" and image_dir:
                img_path = image_dir / f"{p.id}_visual.png"
                if img_path.exists():
                    image_path = img_path
            print(f"  [{i+1}/{len(problems)}] {p.id} ({condition}) ... ", end="", flush=True)
            result = self.evaluate_one(p, condition, image_path)
            status = "✓" if result.correct else "✗"
            print(f"{status}  ({result.latency_s:.1f}s)")
            results.append(result)
        return results


# ══════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════

def get_llava_evaluator(
    backend: str = "ollama",
    ollama_model: str = "llava",
    hf_model: str = "llava-hf/llava-1.5-7b-hf",
    quantize: str | None = None,
):
    """
    Returns the appropriate LLaVA evaluator.

    Parameters
    ----------
    backend      : "ollama" | "hf"
    ollama_model : Ollama model tag, e.g. "llava", "llava:13b", "llava-phi3"
    hf_model     : HuggingFace model ID, e.g. "llava-hf/llava-1.5-7b-hf"
    quantize     : "4bit" | "8bit" | None  (HF backend only)
    """
    if backend == "ollama":
        return OllamaLLaVAEvaluator(model=ollama_model)
    elif backend == "hf":
        return HuggingFaceLLaVAEvaluator(hf_model=hf_model, quantize=quantize)
    else:
        raise ValueError(f"Unknown backend: '{backend}'. Use 'ollama' or 'hf'.")


# ══════════════════════════════════════════════════════════════════════════
#  Standalone test
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset import load_pilot_problems, create_pilot_dataset, stratified_sample
    from image_gen import generate_all_diagrams, DIAGRAM_MAP

    backend = sys.argv[1] if len(sys.argv) > 1 else "ollama"
    print(f"Testing LLaVA evaluator — backend: {backend}")

    create_pilot_dataset()
    problems = load_pilot_problems()
    # Small test: 1 problem per cell
    sample = stratified_sample(problems, n_per_cell=1, seed=42)
    print(f"Test sample: {len(sample)} problems")

    diagram_ids = [p.id for p in sample if p.has_natural_visual and p.id in DIAGRAM_MAP]
    generate_all_diagrams(diagram_ids, save=True)
    image_dir = Path(__file__).parent.parent / "results" / "figures"

    ev = get_llava_evaluator(backend=backend)

    print("\n--- TEXT ONLY ---")
    text_results = ev.run(sample, "text_only")
    save_results(text_results, ev.model_name, "text_only")

    print("\n--- TEXT + VISUAL ---")
    visual_results = ev.run(sample, "text_visual", image_dir=image_dir)
    save_results(visual_results, ev.model_name, "text_visual")

    print(f"\nDone. Results in results/raw/{ev.model_name}_*.json")
