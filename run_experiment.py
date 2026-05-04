"""
run_experiment.py
-----------------
Main entry point for the visual UQ math pilot experiment.

Usage:
    # Mock mode (no API key, no GPU needed — instant results)
    python run_experiment.py --mode mock --n_problems 60

    # LLaVA via Ollama (local machine, CPU/GPU)
    #   First: ollama pull llava
    python run_experiment.py --mode llava --backend ollama
    python run_experiment.py --mode llava --backend ollama --ollama_model llava:13b

    # LLaVA via HuggingFace (HPC cluster / GPU server)
    python run_experiment.py --mode llava --backend hf
    python run_experiment.py --mode llava --backend hf \\
        --hf_model llava-hf/llava-v1.6-mistral-7b-hf --quantize 4bit

    # GPT-4o (requires OpenAI API key in .env)
    python run_experiment.py --mode api --model gpt4o --n_problems 20

    # Re-run analysis on existing results
    python run_experiment.py --mode analyse --model llava-ollama-llava
    python run_experiment.py --mode analyse  # defaults to mock results

    # Generate diagrams only
    python run_experiment.py --mode generate_diagrams
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import create_pilot_dataset, load_pilot_problems, stratified_sample, summary
from evaluator import get_evaluator, save_results
from analysis import run_full_analysis
from image_gen import generate_all_diagrams, DIAGRAM_MAP


def run_mock(n_problems: int = 60, seed: int = 42):
    """Run full pipeline in mock mode — no API key required."""
    print("=" * 60)
    print("VISUAL UQ MATH — PILOT EXPERIMENT (MOCK MODE)")
    print("=" * 60)

    # Step 1: Create dataset
    print("\n[1/4] Creating pilot dataset...")
    create_pilot_dataset()
    problems = load_pilot_problems()
    per_cell = max(1, n_problems // 12)  # 4 subjects × 3 difficulties
    sample = stratified_sample(problems, n_per_cell=per_cell, seed=seed)
    print(f"      Sampled {len(sample)} problems (target: {n_problems})")

    # Step 2: Generate diagrams
    print("\n[2/4] Generating diagrams...")
    diagram_ids = [p.id for p in sample if p.has_natural_visual and p.id in DIAGRAM_MAP]
    generate_all_diagrams(diagram_ids, save=True)

    # Step 3: Evaluate
    print("\n[3/4] Running mock evaluation...")
    ev = get_evaluator("mock")
    text_results   = ev.run(sample, "text_only")
    visual_results = ev.run(sample, "text_visual")
    print(f"      Text-only:   {len(text_results)} results")
    print(f"      Text+visual: {len(visual_results)} results")
    save_results(text_results,   "mock-gpt4o", "text_only")
    save_results(visual_results, "mock-gpt4o", "text_visual")

    # Step 4: Analyse
    print("\n[4/4] Running analysis...")
    run_full_analysis("mock-gpt4o", save_figures=True)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print(f"Results:  results/raw/")
    print(f"Figures:  results/figures/")
    print("=" * 60)


def run_api(model: str = "gpt4o", n_problems: int = 20):
    """Run with real API — requires API key in .env."""
    print("=" * 60)
    print(f"VISUAL UQ MATH — PILOT EXPERIMENT ({model.upper()} MODE)")
    print("=" * 60)
    print("\nChecking API key...")
    import os
    from dotenv import load_dotenv
    load_dotenv()
    if model == "gpt4o" and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in .env file.")
        print("Run in mock mode: python run_experiment.py --mode mock")
        sys.exit(1)

    create_pilot_dataset()
    problems = load_pilot_problems()
    per_cell = max(1, n_problems // 12)
    sample = stratified_sample(problems, n_per_cell=per_cell)
    print(f"Running on {len(sample)} problems...\n")

    diagram_ids = [p.id for p in sample if p.has_natural_visual and p.id in DIAGRAM_MAP]
    generate_all_diagrams(diagram_ids, save=True)

    ev = get_evaluator(model)
    image_dir = Path("results/figures")
    print("\n[Text-only condition]")
    text_results = ev.run(sample, "text_only", image_dir=None)
    save_results(text_results, model, "text_only")
    print("\n[Text+visual condition]")
    visual_results = ev.run(sample, "text_visual", image_dir=image_dir)
    save_results(visual_results, model, "text_visual")
    run_full_analysis(model, save_figures=True)


def run_llava(
    backend: str = "ollama",
    ollama_model: str = "llava",
    hf_model: str = "llava-hf/llava-1.5-7b-hf",
    quantize: str | None = None,
    n_problems: int = 60,
    seed: int = 42,
):
    """Run full pipeline with LLaVA (Ollama or HuggingFace backend)."""
    from llava_evaluator import get_llava_evaluator

    print("=" * 60)
    print(f"VISUAL UQ MATH — LLaVA EXPERIMENT ({backend.upper()} BACKEND)")
    print("=" * 60)

    # Step 1: Dataset
    print("\n[1/4] Creating pilot dataset...")
    create_pilot_dataset()
    problems = load_pilot_problems()
    per_cell = max(1, n_problems // 12)
    sample = stratified_sample(problems, n_per_cell=per_cell, seed=seed)
    print(f"      Sampled {len(sample)} problems")

    # Step 2: Diagrams
    print("\n[2/4] Generating diagrams...")
    diagram_ids = [p.id for p in sample if p.has_natural_visual and p.id in DIAGRAM_MAP]
    generate_all_diagrams(diagram_ids, save=True)

    # Step 3: Evaluate
    print("\n[3/4] Loading LLaVA evaluator...")
    ev = get_llava_evaluator(
        backend=backend,
        ollama_model=ollama_model,
        hf_model=hf_model,
        quantize=quantize,
    )
    image_dir = Path("results/figures")

    print("\n      Running TEXT-ONLY condition...")
    text_results = ev.run(sample, "text_only")
    save_results(text_results, ev.model_name, "text_only")

    print("\n      Running TEXT+VISUAL condition...")
    visual_results = ev.run(sample, "text_visual", image_dir=image_dir)
    save_results(visual_results, ev.model_name, "text_visual")

    print(f"\n      Text-only:   {len(text_results)} results")
    print(f"      Text+visual: {len(visual_results)} results")

    # Step 4: Analyse
    print("\n[4/4] Running analysis...")
    run_full_analysis(ev.model_name, save_figures=True)

    print("\n" + "=" * 60)
    print("LLAVA EXPERIMENT COMPLETE")
    print(f"Model:    {ev.model_name}")
    print(f"Results:  results/raw/{ev.model_name}_*.json")
    print(f"Figures:  results/figures/")
    print("=" * 60)


def run_analyse(model: str = "mock-gpt4o"):
    """Re-run analysis on existing results."""
    print(f"Running analysis on {model} results...")
    run_full_analysis(model, save_figures=True)


def run_generate_diagrams():
    """Generate all diagrams without running evaluation."""
    print("Generating all pilot diagrams...")
    create_pilot_dataset()
    ids = list(DIAGRAM_MAP.keys())
    out = generate_all_diagrams(ids, save=True)
    ok = sum(1 for v in out.values() if v)
    print(f"\nGenerated {ok}/{len(ids)} diagrams in results/figures/")


def main():
    parser = argparse.ArgumentParser(
        description="Visual UQ Math Pilot Experiment Runner"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "api", "llava", "analyse", "generate_diagrams"],
        default="mock",
        help="Execution mode",
    )
    # GPT-4o API options
    parser.add_argument("--model", default="gpt4o",
                        help="Model tag for API mode or analysis (e.g. gpt4o)")
    # LLaVA options
    parser.add_argument("--backend", choices=["ollama", "hf"], default="ollama",
                        help="LLaVA backend: 'ollama' (local) or 'hf' (HuggingFace/HPC)")
    parser.add_argument("--ollama_model", default="llava",
                        help="Ollama model tag, e.g. 'llava', 'llava:13b', 'llava-phi3'")
    parser.add_argument("--hf_model", default="llava-hf/llava-1.5-7b-hf",
                        help="HuggingFace model ID for the HF backend")
    parser.add_argument("--quantize", choices=["4bit", "8bit"], default=None,
                        help="Quantisation for HF backend (reduces VRAM usage)")
    # Common
    parser.add_argument("--n_problems", type=int, default=60,
                        help="Number of problems to evaluate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.mode == "mock":
        run_mock(n_problems=args.n_problems, seed=args.seed)

    elif args.mode == "api":
        run_api(model=args.model, n_problems=args.n_problems)

    elif args.mode == "llava":
        run_llava(
            backend=args.backend,
            ollama_model=args.ollama_model,
            hf_model=args.hf_model,
            quantize=args.quantize,
            n_problems=args.n_problems,
            seed=args.seed,
        )

    elif args.mode == "analyse":
        run_analyse(model=args.model)

    elif args.mode == "generate_diagrams":
        run_generate_diagrams()


if __name__ == "__main__":
    main()
