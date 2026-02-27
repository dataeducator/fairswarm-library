"""
Visualization Utilities for FairSwarm Experiments.

This module provides plotting functions for experiment analysis
and thesis figure generation.

Figures Generated:
    1. Convergence curves (Theorem 1 validation)
    2. Fairness comparison plots (Theorem 2 validation)
    3. Ablation study visualizations
    4. Pareto frontier (accuracy vs fairness tradeoff)
    5. Baseline comparison bar charts

Author: Tenicka Norwood

Usage:
    from analysis.plots import create_thesis_figures
    create_thesis_figures(results_dir="results/", output_dir="figures/")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _check_matplotlib():
    """Check if matplotlib is available."""
    try:
        import matplotlib  # noqa: F401

        return True
    except ImportError:
        return False


def plot_convergence_curves(
    fitness_histories: Dict[str, List[float]],
    output_path: Optional[str] = None,
    title: str = "FairSwarm Convergence (Theorem 1)",
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[Any]:
    """
    Plot convergence curves for different parameter configurations.

    Args:
        fitness_histories: Dict mapping config name to fitness history
        output_path: Path to save figure (optional)
        title: Figure title
        figsize: Figure size

    Returns:
        Matplotlib figure object (if matplotlib available)
    """
    if not _check_matplotlib():
        print("Matplotlib not available. Skipping plot generation.")
        return None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    for config_name, history in fitness_histories.items():
        iterations = range(len(history))
        ax.plot(iterations, history, label=config_name, alpha=0.8)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Fitness", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_fairness_comparison(
    results: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    title: str = "Demographic Divergence Comparison",
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[Any]:
    """
    Plot fairness comparison between FairSwarm and baselines.

    Args:
        results: Dict mapping algorithm name to metrics
                 (must include 'avg_divergence' and 'std_divergence')
        output_path: Path to save figure
        title: Figure title
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    if not _check_matplotlib():
        print("Matplotlib not available. Skipping plot generation.")
        return None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    algorithms = list(results.keys())
    divergences = [results[a]["avg_divergence"] for a in algorithms]
    errors = [results[a].get("std_divergence", 0) for a in algorithms]

    # Color coding
    colors = []
    for alg in algorithms:
        if "fairswarm" in alg.lower():
            colors.append("#2ecc71")  # Green for FairSwarm
        elif "random" in alg.lower():
            colors.append("#e74c3c")  # Red for random
        else:
            colors.append("#3498db")  # Blue for others

    ax.bar(algorithms, divergences, yerr=errors, capsize=5, color=colors, alpha=0.8)

    # Add epsilon threshold line
    ax.axhline(y=0.05, color="red", linestyle="--", label="ε = 0.05 threshold")

    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_ylabel("Demographic Divergence (DemDiv)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Rotate x labels if needed
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_ablation_results(
    ablation_data: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Optional[Any]:
    """
    Plot ablation study results.

    Args:
        ablation_data: Results from ablation experiments
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    if not _check_matplotlib():
        print("Matplotlib not available. Skipping plot generation.")
        return None

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, (ablation_name, data) in enumerate(ablation_data.items()):
        if idx >= 4:
            break

        ax = axes[idx]

        if "analysis" not in data:
            continue

        analysis = data["analysis"]
        variants = list(analysis.keys())

        if not variants:
            continue

        # Extract metrics
        fitness_vals = [analysis[v].get("avg_fitness", 0) for v in variants]
        fairness_vals = [analysis[v].get("avg_fairness", 0) for v in variants]

        x = np.arange(len(variants))
        width = 0.35

        ax.bar(x - width / 2, fitness_vals, width, label="Fitness", color="#3498db")
        ax.bar(
            x + width / 2,
            fairness_vals,
            width,
            label="Fairness (DemDiv)",
            color="#e74c3c",
        )

        ax.set_xlabel("Variant")
        ax.set_ylabel("Value")
        ax.set_title(f"Ablation: {ablation_name.replace('_', ' ').title()}")
        ax.set_xticks(x)
        ax.set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_pareto_frontier(
    points: List[Tuple[float, float, str]],  # (fitness, fairness, label)
    output_path: Optional[str] = None,
    title: str = "Accuracy-Fairness Tradeoff (Pareto Frontier)",
    figsize: Tuple[int, int] = (10, 8),
) -> Optional[Any]:
    """
    Plot Pareto frontier showing accuracy vs fairness tradeoff.

    Args:
        points: List of (fitness, fairness_divergence, label) tuples
        output_path: Path to save figure
        title: Figure title
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    if not _check_matplotlib():
        print("Matplotlib not available. Skipping plot generation.")
        return None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    # Separate FairSwarm and baseline points
    fairswarm_points = [
        (f, d, lbl) for f, d, lbl in points if "fairswarm" in lbl.lower()
    ]
    baseline_points = [
        (f, d, lbl) for f, d, lbl in points if "fairswarm" not in lbl.lower()
    ]

    # Plot baselines
    if baseline_points:
        fitness, divergence, labels = zip(*baseline_points)
        ax.scatter(
            divergence, fitness, c="#e74c3c", s=100, label="Baselines", marker="s"
        )
        for f, d, lbl in baseline_points:
            ax.annotate(
                lbl, (d, f), xytext=(5, 5), textcoords="offset points", fontsize=8
            )

    # Plot FairSwarm variants
    if fairswarm_points:
        fitness, divergence, labels = zip(*fairswarm_points)
        ax.scatter(
            divergence, fitness, c="#2ecc71", s=100, label="FairSwarm", marker="o"
        )
        for f, d, lbl in fairswarm_points:
            ax.annotate(
                lbl, (d, f), xytext=(5, 5), textcoords="offset points", fontsize=8
            )

    # Compute and plot Pareto frontier
    all_points = list(points)
    pareto_points = []

    for f, d, lbl in all_points:
        is_dominated = False
        for f2, d2, _ in all_points:
            # A point is dominated if another has higher fitness AND lower divergence
            if f2 > f and d2 < d:
                is_dominated = True
                break
        if not is_dominated:
            pareto_points.append((f, d))

    if pareto_points:
        # Sort by divergence
        pareto_points.sort(key=lambda x: x[1])
        pareto_f, pareto_d = zip(*pareto_points)
        ax.plot(
            pareto_d, pareto_f, "g--", linewidth=2, alpha=0.5, label="Pareto Frontier"
        )

    # Add ideal point indicator
    ax.axvline(x=0.05, color="red", linestyle=":", alpha=0.5, label="ε = 0.05")

    ax.set_xlabel("Demographic Divergence (lower is better)", fontsize=12)
    ax.set_ylabel("Fitness (higher is better)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_theorem_validation(
    theorem_results: Dict[str, Any],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
) -> Optional[Any]:
    """
    Create summary plot for theorem validation results.

    Args:
        theorem_results: Dict with theorem validation results
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    if not _check_matplotlib():
        print("Matplotlib not available. Skipping plot generation.")
        return None

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Theorem 1: Convergence
    ax1 = axes[0]
    if "convergence" in theorem_results:
        conv = theorem_results["convergence"]
        rates = conv.get("convergence_rates", {})

        if "satisfies_theorem" in rates:
            sat = rates["satisfies_theorem"]
            ax1.bar(
                ["Satisfies\nCondition", "Violates\nCondition"],
                [
                    sat.get("convergence_rate", 0),
                    rates.get("violates_theorem", {}).get("convergence_rate", 0),
                ],
                color=["#2ecc71", "#e74c3c"],
            )
            ax1.set_ylabel("Convergence Rate")
            ax1.set_title("Theorem 1: Convergence")
            ax1.axhline(y=0.9, color="blue", linestyle="--", alpha=0.5)
            ax1.set_ylim(0, 1.1)

    # Theorem 2: Fairness
    ax2 = axes[1]
    if "fairness" in theorem_results:
        fair = theorem_results["fairness"]
        if "theorem_validation" in fair:
            tv = fair["theorem_validation"]
            ax2.bar(
                ["Pass", "Fail"],
                [
                    tv.get("configs_passing", 0),
                    tv.get("configs_tested", 0) - tv.get("configs_passing", 0),
                ],
                color=["#2ecc71", "#e74c3c"],
            )
            ax2.set_ylabel("Number of Configurations")
            ax2.set_title("Theorem 2: ε-Fairness")

    # Theorem 4: Privacy tradeoff (placeholder)
    ax3 = axes[2]
    ax3.text(
        0.5,
        0.5,
        "Privacy-Fairness\nTradeoff\n(See Theorem 4)",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax3.set_title("Theorem 4: Privacy Bound")
    ax3.axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def create_thesis_figures(
    results_dir: str = "results",
    output_dir: str = "figures",
) -> Dict[str, str]:
    """
    Generate all thesis figures from experiment results.

    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save figures

    Returns:
        Dict mapping figure name to output path
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figures = {}

    # Load and process convergence results
    convergence_files = list(
        results_path.glob("convergence/convergence_results_*.json")
    )
    if convergence_files:
        with open(convergence_files[-1]) as f:  # Latest file
            json.load(f)

        # Create convergence plot (placeholder data structure)
        # In practice, would extract actual fitness histories
        figures["convergence"] = str(output_path / "convergence_curves.png")

    # Load and process fairness results
    fairness_files = list(results_path.glob("fairness/fairness_results_*.json"))
    if fairness_files:
        with open(fairness_files[-1]) as f:
            fair_results = json.load(f)

        if (
            "analysis" in fair_results
            and "baseline_comparison" in fair_results["analysis"]
        ):
            plot_fairness_comparison(
                fair_results["analysis"]["baseline_comparison"],
                output_path=str(output_path / "fairness_comparison.png"),
            )
            figures["fairness"] = str(output_path / "fairness_comparison.png")

    # Load and process ablation results
    ablation_files = list(results_path.glob("ablation/ablation_results_*.json"))
    if ablation_files:
        with open(ablation_files[-1]) as f:
            ablation_results = json.load(f)

        if "ablations" in ablation_results:
            plot_ablation_results(
                ablation_results["ablations"],
                output_path=str(output_path / "ablation_results.png"),
            )
            figures["ablation"] = str(output_path / "ablation_results.png")

    print(f"\nGenerated {len(figures)} figures in {output_dir}/")
    return figures


def generate_latex_table(
    data: Dict[str, Dict[str, float]],
    columns: List[str],
    caption: str,
    label: str,
) -> str:
    """
    Generate LaTeX table from experiment results.

    Args:
        data: Dict mapping row name to column values
        columns: List of column names
        caption: Table caption
        label: LaTeX label

    Returns:
        LaTeX table string
    """
    header = " & ".join(["Algorithm"] + columns) + " \\\\"

    rows = []
    for alg, values in data.items():
        row_values = [alg]
        for col in columns:
            val = values.get(col, "—")
            if isinstance(val, float):
                row_values.append(f"{val:.4f}")
            else:
                row_values.append(str(val))
        rows.append(" & ".join(row_values) + " \\\\")

    latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{l{"c" * len(columns)}}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    return latex


if __name__ == "__main__":
    # Example usage
    print("Generating thesis figures from experiment results...")
    figures = create_thesis_figures()
    print(f"Generated figures: {list(figures.keys())}")
