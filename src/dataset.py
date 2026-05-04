"""
dataset.py
----------
Loads, filters, and stratifies the pilot problem set for the visual UQ
math experiment. Supports both local JSON and HuggingFace loading.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

DATA_DIR = Path(__file__).parent.parent / "data"

SUBJECTS = ["calculus", "combinatorics", "linear_algebra", "number_theory"]
DIFFICULTIES = ["university", "competition", "olympiad"]


@dataclass
class Problem:
    id: str
    subject: str
    difficulty: str
    text: str
    answer: str
    answer_type: str          # "integer", "float", "expression", "multiple_choice"
    visual_description: str   # what the diagram should show
    has_natural_visual: bool  # whether a meaningful diagram exists
    source: str               # "pilot" | "mathvista" | "mathvision"
    metadata: dict


def load_pilot_problems(path: Optional[Path] = None) -> List[Problem]:
    """Load the curated pilot problem set from JSON."""
    path = path or DATA_DIR / "pilot_problems.json"
    with open(path) as f:
        raw = json.load(f)
    return [Problem(**p) for p in raw]


def stratified_sample(
    problems: List[Problem],
    n_per_cell: int = 5,
    subjects: Optional[List[str]] = None,
    difficulties: Optional[List[str]] = None,
    seed: int = 42,
) -> List[Problem]:
    """
    Return a balanced sample: n_per_cell problems per (subject, difficulty) cell.
    Default: 4 subjects x 3 difficulties x 5 = 60 problems.
    """
    random.seed(seed)
    subjects = subjects or SUBJECTS
    difficulties = difficulties or DIFFICULTIES
    sampled = []
    for subj in subjects:
        for diff in difficulties:
            pool = [p for p in problems if p.subject == subj and p.difficulty == diff]
            k = min(n_per_cell, len(pool))
            sampled.extend(random.sample(pool, k))
    return sampled


def split_by_condition(problems: List[Problem]):
    """
    Returns two lists representing the two experimental conditions.
    In practice each problem exists in both conditions —
    this returns the same list twice (text_only, text_plus_visual).
    """
    text_only = [p for p in problems]
    text_visual = [p for p in problems if p.has_natural_visual]
    return text_only, text_visual


def summary(problems: List[Problem]) -> Dict:
    """Print dataset summary statistics."""
    from collections import Counter
    subj_counts = Counter(p.subject for p in problems)
    diff_counts = Counter(p.difficulty for p in problems)
    visual_count = sum(1 for p in problems if p.has_natural_visual)
    return {
        "total": len(problems),
        "by_subject": dict(subj_counts),
        "by_difficulty": dict(diff_counts),
        "has_natural_visual": visual_count,
        "visual_fraction": round(visual_count / len(problems), 3) if problems else 0,
    }


def to_json(problems: List[Problem], path: Path):
    """Serialise problem list to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(p) for p in problems], f, indent=2)


# ── Pilot problem set (60 problems, hard-coded for reproducibility) ─────────

PILOT_PROBLEMS_RAW = [
    # ── CALCULUS (15 problems) ──────────────────────────────────────────────
    {
        "id": "calc_u_001", "subject": "calculus", "difficulty": "university",
        "text": "Find the derivative of f(x) = x^3 * sin(x) at x = pi/2.",
        "answer": "3*(pi/2)^2", "answer_type": "expression",
        "visual_description": "Plot of f(x) = x^3 sin(x) with tangent line at x=pi/2",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "product rule"}
    },
    {
        "id": "calc_u_002", "subject": "calculus", "difficulty": "university",
        "text": "Evaluate the integral from 0 to 1 of x*e^x dx.",
        "answer": "1", "answer_type": "integer",
        "visual_description": "Area under curve x*e^x from 0 to 1",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "integration by parts"}
    },
    {
        "id": "calc_u_003", "subject": "calculus", "difficulty": "university",
        "text": "Find all critical points of f(x) = x^4 - 4x^3 + 4x^2.",
        "answer": "x=0, x=1, x=2", "answer_type": "expression",
        "visual_description": "Plot of f(x) = x^4 - 4x^3 + 4x^2 showing critical points",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "critical points"}
    },
    {
        "id": "calc_u_004", "subject": "calculus", "difficulty": "university",
        "text": "Determine whether the series sum_{n=1}^{inf} 1/n^2 converges.",
        "answer": "Converges (Basel problem, sum = pi^2/6)", "answer_type": "expression",
        "visual_description": "Bar chart of partial sums converging to pi^2/6",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "series convergence"}
    },
    {
        "id": "calc_u_005", "subject": "calculus", "difficulty": "university",
        "text": "Find the Taylor series of cos(x) centred at 0 up to the x^4 term.",
        "answer": "1 - x^2/2 + x^4/24", "answer_type": "expression",
        "visual_description": "Plot comparing cos(x) and its Taylor approximations",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "Taylor series"}
    },
    {
        "id": "calc_c_001", "subject": "calculus", "difficulty": "competition",
        "text": "Let f: R -> R be differentiable with f(0)=1 and f'(x) = f(x) for all x. Find f(3).",
        "answer": "e^3", "answer_type": "expression",
        "visual_description": "Exponential curve f(x)=e^x with point (3, e^3) marked",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "ODEs"}
    },
    {
        "id": "calc_c_002", "subject": "calculus", "difficulty": "competition",
        "text": "Compute lim_{x->0} (sin(x) - x + x^3/6) / x^5.",
        "answer": "-1/120", "answer_type": "float",
        "visual_description": "Plot of (sin x - x + x^3/6)/x^5 near x=0",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "L'Hopital/Taylor"}
    },
    {
        "id": "calc_c_003", "subject": "calculus", "difficulty": "competition",
        "text": "Find the maximum value of xy subject to x + y = 1, x,y >= 0.",
        "answer": "1/4", "answer_type": "float",
        "visual_description": "Constraint x+y=1 with level curves of xy",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "optimisation"}
    },
    {
        "id": "calc_c_004", "subject": "calculus", "difficulty": "competition",
        "text": "Evaluate the double integral over the unit disk of (x^2 + y^2) dA.",
        "answer": "pi/2", "answer_type": "expression",
        "visual_description": "Unit disk with radial integration regions",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "double integrals"}
    },
    {
        "id": "calc_c_005", "subject": "calculus", "difficulty": "competition",
        "text": "Let f(x) = x sin(1/x) for x ≠ 0. Is f uniformly continuous on (0,1)?",
        "answer": "Yes, extendable to [0,1]", "answer_type": "expression",
        "visual_description": "Oscillating plot of x sin(1/x) near origin",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "uniform continuity"}
    },
    {
        "id": "calc_o_001", "subject": "calculus", "difficulty": "olympiad",
        "text": "Prove that for all x in [0,1]: x(1-x) <= 1/4.",
        "answer": "AM-GM: x(1-x) <= ((x+1-x)/2)^2 = 1/4", "answer_type": "expression",
        "visual_description": "Parabola y=x(1-x) with maximum at (1/2, 1/4)",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "inequalities"}
    },
    {
        "id": "calc_o_002", "subject": "calculus", "difficulty": "olympiad",
        "text": "Find all continuous f: R->R satisfying f(x+y) = f(x) + f(y) for all x,y.",
        "answer": "f(x) = cx for some constant c", "answer_type": "expression",
        "visual_description": "Linear functions passing through origin",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "functional equations"}
    },
    {
        "id": "calc_o_003", "subject": "calculus", "difficulty": "olympiad",
        "text": "Show that e^x >= 1 + x for all real x.",
        "answer": "By convexity / tangent line at x=0", "answer_type": "expression",
        "visual_description": "e^x and line 1+x, tangent at origin",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "convexity"}
    },
    {
        "id": "calc_o_004", "subject": "calculus", "difficulty": "olympiad",
        "text": "Evaluate: integral from 0 to infinity of sin(x)/x dx.",
        "answer": "pi/2", "answer_type": "expression",
        "visual_description": "Plot of sinc function sin(x)/x from 0 to 4*pi",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "improper integrals"}
    },
    {
        "id": "calc_o_005", "subject": "calculus", "difficulty": "olympiad",
        "text": "For f twice differentiable with f(0)=0, f(1)=1, f'(0)=f'(1)=0, show there exists c in (0,1) with |f''(c)| >= 4.",
        "answer": "MVT applied twice to f'", "answer_type": "expression",
        "visual_description": "Smooth S-curve from (0,0) to (1,1) with zero derivatives at endpoints",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "mean value theorem"}
    },

    # ── COMBINATORICS (15 problems) ─────────────────────────────────────────
    {
        "id": "comb_u_001", "subject": "combinatorics", "difficulty": "university",
        "text": "How many ways can 5 students be seated in a row?",
        "answer": "120", "answer_type": "integer",
        "visual_description": "Five labelled seats in a row",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "permutations"}
    },
    {
        "id": "comb_u_002", "subject": "combinatorics", "difficulty": "university",
        "text": "In how many ways can you choose 3 books from a shelf of 10?",
        "answer": "120", "answer_type": "integer",
        "visual_description": "10 books on a shelf with 3 highlighted",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "combinations"}
    },
    {
        "id": "comb_u_003", "subject": "combinatorics", "difficulty": "university",
        "text": "Find the number of subsets of {1,2,3,4,5} that contain at least one even number.",
        "answer": "24", "answer_type": "integer",
        "visual_description": "Set diagram showing even and odd elements",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "inclusion-exclusion"}
    },
    {
        "id": "comb_u_004", "subject": "combinatorics", "difficulty": "university",
        "text": "What is the coefficient of x^3 in (1+x)^7?",
        "answer": "35", "answer_type": "integer",
        "visual_description": "Pascal's triangle rows 0 through 7",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "binomial theorem"}
    },
    {
        "id": "comb_u_005", "subject": "combinatorics", "difficulty": "university",
        "text": "How many binary strings of length 8 have exactly three 1s?",
        "answer": "56", "answer_type": "integer",
        "visual_description": "Binary strings with positions marked",
        "has_natural_visual": False, "source": "pilot", "metadata": {"topic": "binary strings"}
    },
    {
        "id": "comb_c_001", "subject": "combinatorics", "difficulty": "competition",
        "text": "In how many ways can 8 rooks be placed on an 8x8 chessboard so that no two attack each other?",
        "answer": "40320", "answer_type": "integer",
        "visual_description": "8x8 chessboard with one rook per row and column",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "permutation matrices"}
    },
    {
        "id": "comb_c_002", "subject": "combinatorics", "difficulty": "competition",
        "text": "How many paths are there from (0,0) to (4,4) using steps right or up, not crossing the diagonal y=x?",
        "answer": "14", "answer_type": "integer",
        "visual_description": "Grid paths from (0,0) to (4,4) staying below diagonal",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "Catalan numbers"}
    },
    {
        "id": "comb_c_003", "subject": "combinatorics", "difficulty": "competition",
        "text": "A graph has 6 vertices each with degree 3. How many edges does it have?",
        "answer": "9", "answer_type": "integer",
        "visual_description": "6-vertex 3-regular graph (Petersen-like)",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "graph theory"}
    },
    {
        "id": "comb_c_004", "subject": "combinatorics", "difficulty": "competition",
        "text": "Prove that among any 6 people, there are either 3 mutual acquaintances or 3 mutual strangers.",
        "answer": "Ramsey R(3,3)=6, pigeonhole on edges", "answer_type": "expression",
        "visual_description": "K6 graph with edges coloured red/blue",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "Ramsey theory"}
    },
    {
        "id": "comb_c_005", "subject": "combinatorics", "difficulty": "competition",
        "text": "How many permutations of {1,2,3,4,5} have no fixed points?",
        "answer": "44", "answer_type": "integer",
        "visual_description": "Derangement table for n=5",
        "has_natural_visual": False, "source": "pilot", "metadata": {"topic": "derangements"}
    },
    {
        "id": "comb_o_001", "subject": "combinatorics", "difficulty": "olympiad",
        "text": "Show that C(2n, n) is always even for n >= 1.",
        "answer": "C(2n,n) = 2*C(2n-1,n-1), which is even", "answer_type": "expression",
        "visual_description": "Pascal's triangle with central column highlighted",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "divisibility"}
    },
    {
        "id": "comb_o_002", "subject": "combinatorics", "difficulty": "olympiad",
        "text": "In a tournament with n players where every pair plays once, show there is a player who beats all players they have beaten.",
        "answer": "Player with maximum wins", "answer_type": "expression",
        "visual_description": "Tournament directed graph with 5 vertices",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "tournaments"}
    },
    {
        "id": "comb_o_003", "subject": "combinatorics", "difficulty": "olympiad",
        "text": "Prove that the number of partitions of n into odd parts equals the number of partitions of n into distinct parts.",
        "answer": "Bijection via Euler's theorem", "answer_type": "expression",
        "visual_description": "Partition diagrams (Young diagrams) for n=6",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "partition theory"}
    },
    {
        "id": "comb_o_004", "subject": "combinatorics", "difficulty": "olympiad",
        "text": "How many ways can you tile a 2xn board with 1x2 dominoes?",
        "answer": "F(n+1) (Fibonacci)", "answer_type": "expression",
        "visual_description": "2xn board with domino tiling illustrated for n=4",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "tiling"}
    },
    {
        "id": "comb_o_005", "subject": "combinatorics", "difficulty": "olympiad",
        "text": "In a group of n people, each person knows at least n/2 others. Show there is a Hamiltonian cycle.",
        "answer": "Dirac's theorem", "answer_type": "expression",
        "visual_description": "Graph with minimum degree n/2 and Hamiltonian cycle highlighted",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "Hamiltonian graphs"}
    },

    # ── LINEAR ALGEBRA (15 problems) ────────────────────────────────────────
    {
        "id": "linalg_u_001", "subject": "linear_algebra", "difficulty": "university",
        "text": "Find the eigenvalues of the matrix [[2,1],[1,2]].",
        "answer": "1 and 3", "answer_type": "expression",
        "visual_description": "2x2 matrix with eigenvectors shown as arrows",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "eigenvalues"}
    },
    {
        "id": "linalg_u_002", "subject": "linear_algebra", "difficulty": "university",
        "text": "Is the set {(1,2,3),(4,5,6),(7,8,9)} linearly independent?",
        "answer": "No, det=0", "answer_type": "expression",
        "visual_description": "Three vectors in 3D space lying in a plane",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "linear independence"}
    },
    {
        "id": "linalg_u_003", "subject": "linear_algebra", "difficulty": "university",
        "text": "Compute the determinant of [[1,2,3],[4,5,6],[7,8,10]].",
        "answer": "-3", "answer_type": "integer",
        "visual_description": "3x3 matrix with cofactor expansion illustrated",
        "has_natural_visual": False, "source": "pilot", "metadata": {"topic": "determinants"}
    },
    {
        "id": "linalg_u_004", "subject": "linear_algebra", "difficulty": "university",
        "text": "What is the rank of the matrix [[1,2],[2,4],[3,6]]?",
        "answer": "1", "answer_type": "integer",
        "visual_description": "Three row vectors all pointing in the same direction",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "rank"}
    },
    {
        "id": "linalg_u_005", "subject": "linear_algebra", "difficulty": "university",
        "text": "Find the projection of vector (3,4) onto vector (1,0).",
        "answer": "(3,0)", "answer_type": "expression",
        "visual_description": "Vectors (3,4) and (1,0) with projection shown",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "projections"}
    },
    {
        "id": "linalg_c_001", "subject": "linear_algebra", "difficulty": "competition",
        "text": "Let A be an n×n matrix with A^2 = A. Show eigenvalues are 0 or 1.",
        "answer": "If Av=λv then λ^2 v = A^2 v = Av = λv, so λ^2=λ", "answer_type": "expression",
        "visual_description": "Projection matrix splitting R^n into eigenspaces",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "idempotent matrices"}
    },
    {
        "id": "linalg_c_002", "subject": "linear_algebra", "difficulty": "competition",
        "text": "If A is a 3×3 matrix with det(A)=6 and det(2A)=?",
        "answer": "48", "answer_type": "integer",
        "visual_description": "Scaling diagram showing det(cA)=c^n det(A)",
        "has_natural_visual": False, "source": "pilot", "metadata": {"topic": "determinant properties"}
    },
    {
        "id": "linalg_c_003", "subject": "linear_algebra", "difficulty": "competition",
        "text": "Show that if A and B are similar matrices, they have the same eigenvalues.",
        "answer": "A=PBP^{-1}, det(A-λI)=det(B-λI)", "answer_type": "expression",
        "visual_description": "Commutative diagram of similar matrices",
        "has_natural_visual": False, "source": "pilot", "metadata": {"topic": "similarity"}
    },
    {
        "id": "linalg_c_004", "subject": "linear_algebra", "difficulty": "competition",
        "text": "Find the SVD of the matrix [[1,0],[0,2],[0,0]].",
        "answer": "U diag(2,1) V^T with standard U,V", "answer_type": "expression",
        "visual_description": "Geometric interpretation of SVD as rotation-scale-rotation",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "SVD"}
    },
    {
        "id": "linalg_c_005", "subject": "linear_algebra", "difficulty": "competition",
        "text": "For a symmetric matrix A, show that eigenvectors for distinct eigenvalues are orthogonal.",
        "answer": "λ1(v1·v2)=Av1·v2=v1·Av2=λ2(v1·v2), so (λ1-λ2)(v1·v2)=0", "answer_type": "expression",
        "visual_description": "Two orthogonal eigenvectors in 2D",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "spectral theorem"}
    },
    {
        "id": "linalg_o_001", "subject": "linear_algebra", "difficulty": "olympiad",
        "text": "Prove that rank(A) + nullity(A) = n for an m×n matrix A.",
        "answer": "Rank-nullity theorem via basis extension", "answer_type": "expression",
        "visual_description": "Diagram of null space, column space, and dimensions",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "rank-nullity"}
    },
    {
        "id": "linalg_o_002", "subject": "linear_algebra", "difficulty": "olympiad",
        "text": "Show that for any real matrix A, A^T A is positive semi-definite.",
        "answer": "x^T A^T A x = ||Ax||^2 >= 0", "answer_type": "expression",
        "visual_description": "Ellipse showing quadratic form of A^T A",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "PSD matrices"}
    },
    {
        "id": "linalg_o_003", "subject": "linear_algebra", "difficulty": "olympiad",
        "text": "Prove Cauchy-Schwarz: |<u,v>| <= ||u|| ||v|| for vectors u,v in R^n.",
        "answer": "Consider ||u - tv||^2 >= 0 as quadratic in t", "answer_type": "expression",
        "visual_description": "Vectors u and v with angle theta between them",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "Cauchy-Schwarz"}
    },
    {
        "id": "linalg_o_004", "subject": "linear_algebra", "difficulty": "olympiad",
        "text": "Let A be an n×n real symmetric matrix. Show it is diagonalisable over R.",
        "answer": "Spectral theorem for real symmetric matrices", "answer_type": "expression",
        "visual_description": "Orthogonal basis of eigenvectors in R^3",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "spectral theorem"}
    },
    {
        "id": "linalg_o_005", "subject": "linear_algebra", "difficulty": "olympiad",
        "text": "Show det(e^A) = e^{tr(A)} for any square matrix A.",
        "answer": "Via Jordan form and property of trace/det", "answer_type": "expression",
        "visual_description": "Commutative diagram of matrix exponential",
        "has_natural_visual": False, "source": "pilot", "metadata": {"topic": "matrix exponential"}
    },

    # ── NUMBER THEORY (15 problems) ─────────────────────────────────────────
    {
        "id": "numth_u_001", "subject": "number_theory", "difficulty": "university",
        "text": "Find all integers x satisfying x ≡ 3 (mod 7) and x ≡ 1 (mod 5).",
        "answer": "x ≡ 31 (mod 35)", "answer_type": "expression",
        "visual_description": "Number line with residues mod 35 marked",
        "has_natural_visual": False, "source": "pilot", "metadata": {"topic": "CRT"}
    },
    {
        "id": "numth_u_002", "subject": "number_theory", "difficulty": "university",
        "text": "Find gcd(252, 198) using the Euclidean algorithm.",
        "answer": "18", "answer_type": "integer",
        "visual_description": "Euclidean algorithm steps shown as rectangle subdivisions",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "GCD"}
    },
    {
        "id": "numth_u_003", "subject": "number_theory", "difficulty": "university",
        "text": "How many positive integers less than 100 are coprime to 100?",
        "answer": "40", "answer_type": "integer",
        "visual_description": "Euler totient function phi(100) visualised",
        "has_natural_visual": False, "source": "pilot", "metadata": {"topic": "Euler totient"}
    },
    {
        "id": "numth_u_004", "subject": "number_theory", "difficulty": "university",
        "text": "Find all prime factors of 360.",
        "answer": "2, 3, 5", "answer_type": "expression",
        "visual_description": "Prime factor tree for 360",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "factorisation"}
    },
    {
        "id": "numth_u_005", "subject": "number_theory", "difficulty": "university",
        "text": "Prove that the square of any odd integer is ≡ 1 (mod 8).",
        "answer": "(2k+1)^2 = 4k^2+4k+1 = 4k(k+1)+1, k(k+1) even", "answer_type": "expression",
        "visual_description": "Mod 8 clock showing squares of odd numbers",
        "has_natural_visual": False, "source": "pilot", "metadata": {"topic": "modular arithmetic"}
    },
    {
        "id": "numth_c_001", "subject": "number_theory", "difficulty": "competition",
        "text": "Show there are infinitely many primes.",
        "answer": "Euclid's proof: assume finite, multiply all and add 1", "answer_type": "expression",
        "visual_description": "Prime distribution on number line to 100",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "infinitude of primes"}
    },
    {
        "id": "numth_c_002", "subject": "number_theory", "difficulty": "competition",
        "text": "Find all solutions to x^2 ≡ -1 (mod 13).",
        "answer": "x ≡ 5 or x ≡ 8 (mod 13)", "answer_type": "expression",
        "visual_description": "Mod 13 multiplication table quadratic residues",
        "has_natural_visual": False, "source": "pilot", "metadata": {"topic": "quadratic residues"}
    },
    {
        "id": "numth_c_003", "subject": "number_theory", "difficulty": "competition",
        "text": "For which primes p is 2 a quadratic residue mod p?",
        "answer": "p ≡ ±1 (mod 8) by quadratic reciprocity", "answer_type": "expression",
        "visual_description": "Primes coloured by residue class mod 8",
        "has_natural_visual": False, "source": "pilot", "metadata": {"topic": "quadratic reciprocity"}
    },
    {
        "id": "numth_c_004", "subject": "number_theory", "difficulty": "competition",
        "text": "Prove Fermat's Little Theorem: a^p ≡ a (mod p) for prime p.",
        "answer": "Via group theory: order of (Z/pZ)* is p-1", "answer_type": "expression",
        "visual_description": "Cyclic group Z/pZ structure diagram",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "Fermat's theorem"}
    },
    {
        "id": "numth_c_005", "subject": "number_theory", "difficulty": "competition",
        "text": "Find all Pythagorean triples with hypotenuse 65.",
        "answer": "(25,60,65) and (33,56,65) and (16,63,65) and (39,52,65)", "answer_type": "expression",
        "visual_description": "Right triangles with hypotenuse 65 drawn to scale",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "Pythagorean triples"}
    },
    {
        "id": "numth_o_001", "subject": "number_theory", "difficulty": "olympiad",
        "text": "Prove that n^5 - n is divisible by 30 for all integers n.",
        "answer": "n^5-n = n(n-1)(n+1)(n^2+1), divisible by 2,3,5", "answer_type": "expression",
        "visual_description": "Factor diagram of n^5 - n",
        "has_natural_visual": False, "source": "pilot", "metadata": {"topic": "divisibility"}
    },
    {
        "id": "numth_o_002", "subject": "number_theory", "difficulty": "olympiad",
        "text": "Show that if p and p^2+2 are both prime, then p=3.",
        "answer": "p mod 3: if p≡1 or 2, then p^2+2≡0 (mod 3)", "answer_type": "expression",
        "visual_description": "Residue classes mod 3 for p and p^2+2",
        "has_natural_visual": False, "source": "pilot", "metadata": {"topic": "prime properties"}
    },
    {
        "id": "numth_o_003", "subject": "number_theory", "difficulty": "olympiad",
        "text": "Find all integer solutions to x^2 - 2y^2 = 1.",
        "answer": "Infinite solutions via Pell equation: (1,0),(3,2),(17,12),...", "answer_type": "expression",
        "visual_description": "Hyperbola x^2 - 2y^2 = 1 with integer points marked",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "Pell equation"}
    },
    {
        "id": "numth_o_004", "subject": "number_theory", "difficulty": "olympiad",
        "text": "Prove that sqrt(2) is irrational.",
        "answer": "Assume p/q in lowest terms, show 2|p and 2|q, contradiction", "answer_type": "expression",
        "visual_description": "Geometric proof: diagonal of unit square is irrational",
        "has_natural_visual": True, "source": "pilot", "metadata": {"topic": "irrationality"}
    },
    {
        "id": "numth_o_005", "subject": "number_theory", "difficulty": "olympiad",
        "text": "Show that the sum of the divisors function sigma(n) satisfies sigma(mn) = sigma(m)sigma(n) when gcd(m,n)=1.",
        "answer": "Multiplicativity of sigma via unique factorisation", "answer_type": "expression",
        "visual_description": "Divisor lattice diagram for mn with gcd=1",
        "has_natural_visual": False, "source": "pilot", "metadata": {"topic": "multiplicative functions"}
    },
]


def create_pilot_dataset():
    """Write the pilot problem set to data/pilot_problems.json."""
    problems = [Problem(**p) for p in PILOT_PROBLEMS_RAW]
    to_json(problems, DATA_DIR / "pilot_problems.json")
    print(f"Created {len(problems)} problems in {DATA_DIR}/pilot_problems.json")
    print(json.dumps(summary(problems), indent=2))
    return problems


if __name__ == "__main__":
    create_pilot_dataset()
