"""
image_gen.py
------------
Programmatic diagram generation for mathematical problems.
Uses matplotlib, sympy, and networkx to create meaningful
visual representations of university-level math problems.
"""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional

try:
    import sympy as sp
    SYMPY_OK = True
except ImportError:
    SYMPY_OK = False

try:
    import networkx as nx
    NX_OK = True
except ImportError:
    NX_OK = False

FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"


def _save(fig, problem_id: str, condition: str = "visual") -> Path:
    """Save figure to file and return path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{problem_id}_{condition}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _to_b64(fig) -> str:
    """Convert figure to base64 PNG for API embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── CALCULUS DIAGRAMS ──────────────────────────────────────────────────────

def plot_function(expr_str: str, x_range=(-3, 3), title: str = "",
                  highlight_x: float = None, tangent: bool = False) -> plt.Figure:
    """Plot a function given as string expression."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.linspace(x_range[0], x_range[1], 400)
    try:
        y = eval(expr_str, {"x": x, "np": np, "sin": np.sin, "cos": np.cos,
                             "exp": np.exp, "pi": np.pi, "e": np.e})
        ax.plot(x, y, "b-", linewidth=2, label=f"f(x) = {expr_str}")
        if highlight_x is not None:
            y_h = eval(expr_str, {"x": highlight_x, "np": np, "sin": np.sin,
                                  "cos": np.cos, "exp": np.exp, "pi": np.pi})
            ax.plot(highlight_x, y_h, "ro", markersize=8, label=f"x = {round(highlight_x, 3)}")
    except Exception:
        ax.text(0.5, 0.5, "Plot unavailable", transform=ax.transAxes, ha="center")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.set_title(title or expr_str, fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_area_under_curve(expr_str: str, a: float, b: float, title: str = "") -> plt.Figure:
    """Shade area under curve between a and b."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.linspace(a - 0.5, b + 0.5, 400)
    x_fill = np.linspace(a, b, 200)
    try:
        y = eval(expr_str, {"x": x, "np": np, "exp": np.exp, "sin": np.sin})
        y_fill = eval(expr_str, {"x": x_fill, "np": np, "exp": np.exp, "sin": np.sin})
        ax.plot(x, y, "b-", linewidth=2)
        ax.fill_between(x_fill, y_fill, alpha=0.3, color="blue", label=f"Area = ∫ f(x)dx from {a} to {b}")
        ax.axvline(a, color="gray", linestyle="--", alpha=0.6)
        ax.axvline(b, color="gray", linestyle="--", alpha=0.6)
    except Exception:
        ax.text(0.5, 0.5, "Plot unavailable", transform=ax.transAxes, ha="center")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.set_title(title or f"Area under {expr_str}", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_convergence(series_name: str, partial_sums: list, limit: float = None) -> plt.Figure:
    """Show convergence of partial sums."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ns = list(range(1, len(partial_sums) + 1))
    ax.plot(ns, partial_sums, "bo-", markersize=4, linewidth=1.5, label="Partial sum Sₙ")
    if limit is not None:
        ax.axhline(limit, color="red", linestyle="--", linewidth=1.5, label=f"Limit = {round(limit, 4)}")
    ax.set_xlabel("n"); ax.set_ylabel("Sₙ")
    ax.set_title(f"Convergence of {series_name}", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── COMBINATORICS DIAGRAMS ─────────────────────────────────────────────────

def plot_pascals_triangle(rows: int = 7) -> plt.Figure:
    """Draw Pascal's triangle."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-rows, rows); ax.set_ylim(-rows - 0.5, 0.5)
    ax.axis("off")
    from math import comb
    for n in range(rows + 1):
        for k in range(n + 1):
            x = k - n / 2
            y = -n
            val = comb(n, k)
            color = "#2E6FAD" if k in (0, n) else ("white" if n < rows else "#FFF3CD")
            ax.text(x, y, str(val), ha="center", va="center", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="#CCCCCC"))
    ax.set_title("Pascal's Triangle", fontsize=12)
    fig.tight_layout()
    return fig


def plot_grid_paths(n: int = 4) -> plt.Figure:
    """Show lattice paths not crossing the diagonal — Catalan number illustration."""
    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(n + 1):
        ax.axhline(i, color="#CCCCCC", linewidth=0.8)
        ax.axvline(i, color="#CCCCCC", linewidth=0.8)
    ax.plot([0, n], [0, n], "r--", linewidth=1.5, label="Diagonal y=x (forbidden)")
    path_x = [0, 0, 1, 1, 2, 2, 3, 3, 4]
    path_y = [0, 1, 1, 2, 2, 3, 3, 4, 4]
    ax.plot(path_x, path_y, "b-", linewidth=2.5, label="Valid Catalan path")
    ax.plot(0, 0, "go", markersize=10)
    ax.plot(n, n, "rs", markersize=10)
    ax.set_xlim(-0.3, n + 0.3); ax.set_ylim(-0.3, n + 0.3)
    ax.set_xlabel("Right steps"); ax.set_ylabel("Up steps")
    ax.set_title(f"Lattice paths ({n}×{n}), Catalan C_{n} = {[1,1,2,5,14][n]}", fontsize=11)
    ax.legend(fontsize=9); ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_graph(n_nodes: int = 6, edges: list = None, title: str = "Graph") -> plt.Figure:
    """Draw a graph using networkx."""
    if not NX_OK:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "networkx not installed", transform=ax.transAxes, ha="center")
        return fig
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    if edges:
        G.add_edges_from(edges)
    else:
        # default: 3-regular graph on 6 nodes
        G.add_edges_from([(0,1),(0,2),(0,3),(1,2),(1,4),(2,5),(3,4),(3,5),(4,5)])
    fig, ax = plt.subplots(figsize=(5, 4))
    pos = nx.circular_layout(G)
    nx.draw_networkx(G, pos, ax=ax, node_color="#2E6FAD", node_size=600,
                     font_color="white", font_size=12, edge_color="#444444", width=2)
    ax.set_title(title, fontsize=11); ax.axis("off")
    fig.tight_layout()
    return fig


# ── LINEAR ALGEBRA DIAGRAMS ────────────────────────────────────────────────

def plot_vectors_2d(vectors: list, labels: list = None, title: str = "") -> plt.Figure:
    """Plot 2D vectors as arrows from origin."""
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ["#2E6FAD", "#E8530A", "#0D7C72", "#5B2D8E"]
    for i, v in enumerate(vectors):
        color = colors[i % len(colors)]
        label = labels[i] if labels else f"v{i+1}"
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
        ax.text(v[0]*1.08, v[1]*1.08, label, color=color, fontsize=11, fontweight="bold")
    lim = max(max(abs(v[0]), abs(v[1])) for v in vectors) * 1.4
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="k", linewidth=0.5); ax.axvline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_projection(u: tuple, v: tuple) -> plt.Figure:
    """Visualise projection of u onto v."""
    fig, ax = plt.subplots(figsize=(5, 5))
    u = np.array(u, float); v = np.array(v, float)
    proj = (np.dot(u, v) / np.dot(v, v)) * v
    for vec, col, lab in [(u, "#2E6FAD", "u"), (v, "#E8530A", "v"), (proj, "#0D7C72", "proj_v(u)")]:
        ax.annotate("", xy=vec, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=col, lw=2.5))
        ax.text(vec[0]*1.1, vec[1]*1.1, lab, color=col, fontsize=11, fontweight="bold")
    ax.plot([proj[0], u[0]], [proj[1], u[1]], "k--", linewidth=1.5, alpha=0.6)
    lim = max(abs(u).max(), abs(v).max()) * 1.5
    ax.set_xlim(-0.5, lim); ax.set_ylim(-0.5, lim)
    ax.axhline(0, color="k", linewidth=0.5); ax.axvline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    ax.set_title("Projection of u onto v", fontsize=11)
    fig.tight_layout()
    return fig


# ── NUMBER THEORY DIAGRAMS ─────────────────────────────────────────────────

def plot_prime_factorisation(n: int) -> plt.Figure:
    """Draw a prime factor tree."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    def factors(n):
        for p in range(2, n + 1):
            if n % p == 0:
                return p, n // p
        return None, None

    def draw_node(ax, val, x, y, parent=None):
        color = "#2E6FAD" if any(val % i == 0 for i in range(2, val)) else "#E8530A"
        ax.text(x, y, str(val), ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="circle,pad=0.4", facecolor=color, edgecolor="white"),
                color="white", fontweight="bold")
        if parent:
            ax.plot([parent[0], x], [parent[1] - 0.2, y + 0.2], "k-", linewidth=1.2)

    draw_node(ax, n, 5, 5.5)
    p, q = factors(n)
    if p:
        draw_node(ax, p, 3, 4, (5, 5.5))
        draw_node(ax, q, 7, 4, (5, 5.5))
        p2, q2 = factors(q)
        if p2:
            draw_node(ax, p2, 5.5, 2.5, (7, 4))
            draw_node(ax, q2, 8.5, 2.5, (7, 4))
    ax.set_title(f"Prime Factorisation of {n}", fontsize=12)
    fig.tight_layout()
    return fig


def plot_pell_equation(limit: int = 5) -> plt.Figure:
    """Plot integer solutions to Pell equation x^2 - 2y^2 = 1."""
    fig, ax = plt.subplots(figsize=(6, 5))
    x_c = np.linspace(1, 30, 400)
    y_pos = np.sqrt((x_c**2 - 1) / 2)
    ax.plot(x_c, y_pos, "b-", linewidth=1.5, label="x² - 2y² = 1")
    ax.plot(x_c, -y_pos, "b-", linewidth=1.5)
    solutions = [(1, 0), (3, 2), (17, 12), (99, 70)][:limit]
    for (x, y) in solutions:
        ax.plot(x, y, "ro", markersize=8)
        ax.annotate(f"({x},{y})", (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=9, color="#E8530A")
    ax.set_xlim(0, 25); ax.set_ylim(-20, 20)
    ax.axhline(0, color="k", linewidth=0.5); ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Pell Equation: x² - 2y² = 1 (integer solutions in red)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── DISPATCHER ────────────────────────────────────────────────────────────

DIAGRAM_MAP = {
    "calc_u_001": lambda: plot_function("x**3 * np.sin(x)", (-1, 4), "f(x) = x³sin(x)", np.pi/2),
    "calc_u_002": lambda: plot_area_under_curve("x * np.exp(x)", 0, 1, "∫₀¹ x·eˣ dx"),
    "calc_u_003": lambda: plot_function("x**4 - 4*x**3 + 4*x**2", (-0.5, 2.5), "f(x) = x⁴ - 4x³ + 4x²"),
    "calc_u_004": lambda: plot_convergence("Σ 1/n²", [sum(1/k**2 for k in range(1, n+1)) for n in range(1, 31)], np.pi**2/6),
    "calc_u_005": lambda: plot_function("np.cos(x)", (-4, 4), "cos(x) vs Taylor approximation"),
    "calc_c_001": lambda: plot_function("np.exp(x)", (-2, 4), "f(x) = eˣ (solution to f′=f)"),
    "calc_c_002": lambda: plot_function("(np.sin(x) - x + x**3/6) / (x**5 + 1e-10)", (-1, 1), "Limit analysis near x=0"),
    "calc_c_003": lambda: plot_function("-x**2 + x", (0, 1), "f(x) = -x² + x (xy with x+y=1)"),
    "calc_o_001": lambda: plot_function("x * (1-x)", (0, 1), "f(x) = x(1-x) ≤ 1/4"),
    "calc_o_002": lambda: plot_function("2*x", (-3, 3), "f(x) = cx (Cauchy functional equation solutions)"),
    "calc_o_003": lambda: plot_function("np.exp(x)", (-3, 2), "eˣ vs 1+x (tangent at origin)"),
    "calc_o_004": lambda: plot_function("np.sin(x)/(x + 1e-10)", (0.01, 20), "sinc: sin(x)/x"),
    "comb_u_001": lambda: plot_graph(5, [(0,1),(1,2),(2,3),(3,4),(4,0)], "5 seats in a row (circular)"),
    "comb_u_004": lambda: plot_pascals_triangle(7),
    "comb_c_001": lambda: plot_graph(8, [(i, (i+1)%8) for i in range(8)] + [(i, (i+3)%8) for i in range(8)], "8-rook placement (permutation)"),
    "comb_c_002": lambda: plot_grid_paths(4),
    "comb_c_003": lambda: plot_graph(6, [(0,1),(0,2),(0,3),(1,2),(1,4),(2,5),(3,4),(3,5),(4,5)], "3-Regular Graph (6 vertices, 9 edges)"),
    "comb_c_004": lambda: plot_graph(6, [(i,j) for i in range(6) for j in range(i+1,6)], "K₆ — Ramsey R(3,3)=6"),
    "comb_o_004": lambda: plot_pascals_triangle(6),
    "linalg_u_001": lambda: plot_vectors_2d([(3, 1), (1, 3)], ["v₁=(3,1)", "v₂=(1,3)"], "Eigenvectors of [[2,1],[1,2]]"),
    "linalg_u_002": lambda: plot_vectors_2d([(1,2),(4,5),(7,8)], ["v₁","v₂","v₃"], "3 coplanar vectors (linearly dependent)"),
    "linalg_u_004": lambda: plot_vectors_2d([(1,2),(2,4),(3,6)], ["r₁","r₂","r₃"], "Rank-1 matrix: all rows proportional"),
    "linalg_u_005": lambda: plot_projection((3, 4), (1, 0)),
    "linalg_c_001": lambda: plot_vectors_2d([(3, 0), (0, 2)], ["Eigenspace λ=1", "Eigenspace λ=0"], "Idempotent matrix eigenspaces"),
    "linalg_c_004": lambda: plot_vectors_2d([(1, 0, 0)[:2], (0, 2)], ["σ₁=2", "σ₂=1"], "SVD singular values"),
    "linalg_o_001": lambda: plot_vectors_2d([(2, 0), (0, 1)], ["col space (dim=r)", "null space (dim=n-r)"], "Rank-Nullity Theorem"),
    "linalg_o_003": lambda: plot_vectors_2d([(3, 1), (1, 3)], ["u", "v"], "Cauchy-Schwarz: |⟨u,v⟩| ≤ ‖u‖‖v‖"),
    "numth_u_002": lambda: plot_prime_factorisation(252),
    "numth_u_004": lambda: plot_prime_factorisation(360),
    "numth_c_001": lambda: plot_convergence("Prime count π(n)", [sum(1 for p in range(2, n+1) if all(p%i!=0 for i in range(2, p))) for n in range(1, 51, 2)]),
    "numth_c_004": lambda: plot_graph(7, [(i, (i*2)%7) for i in range(7)], "Cyclic group ℤ/7ℤ (Fermat's theorem)"),
    "numth_c_005": lambda: plot_vectors_2d([(25, 60), (33, 56), (16, 63)], ["(25,60,65)", "(33,56,65)", "(16,63,65)"], "Pythagorean triples with hypotenuse 65"),
    "numth_o_003": lambda: plot_pell_equation(),
    "numth_o_004": lambda: plot_vectors_2d([(1, 1), (1, -1)], ["(1,1)", "(1,-1)"], "√2 = diagonal of unit square"),
}


def generate_diagram(problem_id: str, save: bool = True) -> Optional[str]:
    """
    Generate diagram for a given problem_id.
    Returns file path if save=True, else base64 PNG string.
    """
    if problem_id not in DIAGRAM_MAP:
        return None
    try:
        fig = DIAGRAM_MAP[problem_id]()
        if save:
            path = _save(fig, problem_id)
            return str(path)
        else:
            return _to_b64(fig)
    except Exception as e:
        print(f"  Warning: diagram generation failed for {problem_id}: {e}")
        return None


def generate_all_diagrams(problem_ids: list, save: bool = True) -> dict:
    """Generate diagrams for a list of problem IDs."""
    results = {}
    for pid in problem_ids:
        result = generate_diagram(pid, save=save)
        results[pid] = result
        status = "OK" if result else "skipped (no diagram)"
        print(f"  {pid}: {status}")
    return results


if __name__ == "__main__":
    print("Generating all pilot diagrams...")
    ids = list(DIAGRAM_MAP.keys())
    out = generate_all_diagrams(ids)
    ok = sum(1 for v in out.values() if v)
    print(f"\nGenerated {ok}/{len(ids)} diagrams in results/figures/")
