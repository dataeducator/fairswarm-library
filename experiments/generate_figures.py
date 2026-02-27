"""
Generate publication-quality figures and LaTeX tables from experiment results.

Usage:
    python experiments/generate_figures.py

Outputs:
    figures/   - PNG figures for the paper
    tables/    - LaTeX table fragments
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Consistent style
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# Colors: FairSwarm green, baselines in distinguishable palette
COLORS = {
    "fairswarm": "#2ecc71",
    "all_clients": "#3498db",
    "greedy_fair": "#9b59b6",
    "greedy_acc_fair": "#e67e22",
    "random": "#e74c3c",
    "round_robin": "#95a5a6",
}

LABELS = {
    "fairswarm": "FairSwarm (ours)",
    "all_clients": "FedAvg (all)",
    "greedy_fair": "Greedy-Fair",
    "greedy_acc_fair": "Greedy-Acc+Fair",
    "random": "Random",
    "round_robin": "Round Robin",
}

METHOD_ORDER = [
    "fairswarm",
    "all_clients",
    "greedy_fair",
    "greedy_acc_fair",
    "random",
    "round_robin",
]


def load_results(path: str = "data/processed/experiment_results.json") -> dict:
    with open(path) as f:
        return json.load(f)


def fig1_auc_demdiv_barplot(data: dict, out_dir: str):
    """Figure 1: Side-by-side AUC and DemDiv comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    methods = [m for m in METHOD_ORDER if m in data and m != "_meta"]
    x = np.arange(len(methods))
    width = 0.6

    # AUC bars
    aucs = [data[m]["auc_mean"] for m in methods]
    auc_errs = [data[m]["auc_std"] for m in methods]
    colors = [COLORS[m] for m in methods]

    ax1.bar(
        x,
        aucs,
        width,
        yerr=auc_errs,
        capsize=4,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.set_ylabel("AUC-ROC")
    ax1.set_title("(a) Predictive Performance")
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS[m] for m in methods], rotation=35, ha="right")
    ax1.set_ylim(0.68, 0.76)
    ax1.axhline(
        y=data["all_clients"]["auc_mean"],
        color="gray",
        linestyle=":",
        alpha=0.5,
        linewidth=0.8,
    )
    ax1.grid(axis="y", alpha=0.3)

    # DemDiv bars
    divs = [data[m]["demdiv_mean"] for m in methods]
    div_errs = [data[m]["demdiv_std"] for m in methods]

    ax2.bar(
        x,
        divs,
        width,
        yerr=div_errs,
        capsize=4,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    ax2.set_ylabel("Demographic Divergence (DemDiv)")
    ax2.set_title("(b) Fairness (lower is better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABELS[m] for m in methods], rotation=35, ha="right")
    ax2.axhline(
        y=0.05,
        color="red",
        linestyle="--",
        alpha=0.7,
        linewidth=1,
        label=r"$\epsilon = 0.05$",
    )
    ax2.legend(loc="upper right")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_auc_demdiv.png")
    plt.savefig(path)
    plt.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


def fig2_pareto_frontier(data: dict, out_dir: str):
    """Figure 2: Accuracy-Fairness Pareto frontier."""
    fig, ax = plt.subplots(figsize=(7, 5))

    methods = [m for m in METHOD_ORDER if m in data and m != "_meta"]

    for m in methods:
        auc = data[m]["auc_mean"]
        div = data[m]["demdiv_mean"]
        auc_err = data[m]["auc_std"]
        div_err = data[m]["demdiv_std"]

        marker = "D" if m == "fairswarm" else "o"
        size = 120 if m == "fairswarm" else 80
        zorder = 10 if m == "fairswarm" else 5

        ax.errorbar(
            div,
            auc,
            xerr=div_err,
            yerr=auc_err,
            fmt="none",
            ecolor=COLORS[m],
            alpha=0.4,
            capsize=3,
            zorder=zorder - 1,
        )
        ax.scatter(
            div,
            auc,
            c=COLORS[m],
            s=size,
            marker=marker,
            label=LABELS[m],
            zorder=zorder,
            edgecolors="white",
            linewidth=0.5,
        )

    # Epsilon threshold
    ax.axvline(
        x=0.05,
        color="red",
        linestyle="--",
        alpha=0.5,
        linewidth=1,
        label=r"$\epsilon = 0.05$",
    )

    # Ideal region annotation
    ax.annotate(
        "Ideal region\n(high AUC, low DemDiv)",
        xy=(0.0005, 0.742),
        fontsize=9,
        color="gray",
        fontstyle="italic",
        ha="left",
    )

    ax.set_xlabel("Demographic Divergence (lower is better)")
    ax.set_ylabel("AUC-ROC (higher is better)")
    ax.set_title("Accuracy-Fairness Trade-off")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_pareto.png")
    plt.savefig(path)
    plt.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


def fig3_per_group_auc(data: dict, out_dir: str):
    """Figure 3: Per-group AUC heatmap / grouped bar chart."""
    groups = ["white", "black", "hispanic", "asian", "other"]
    methods = [m for m in METHOD_ORDER if m in data and m != "_meta"]

    # Build per-group AUC from TPR data (we have confusion matrices)
    # Actually, per-group AUC was printed in experiment output but not saved in JSON
    # Use TPR from confusion matrices as a fairness proxy instead
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(groups))
    n_methods = len(methods)
    width = 0.8 / n_methods

    for i, m in enumerate(methods):
        pgcm = data[m].get("per_group_confusion_matrices", {})
        tprs = []
        for g in groups:
            if g in pgcm:
                tprs.append(pgcm[g].get("tpr", 0))
            else:
                tprs.append(0)
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(
            x + offset,
            tprs,
            width,
            label=LABELS[m],
            color=COLORS[m],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.3,
        )

    ax.set_xlabel("Demographic Group")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Per-Group Sensitivity Across Methods")
    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in groups])
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 0.8)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_per_group_tpr.png")
    plt.savefig(path)
    plt.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


def fig4_fairness_metrics(data: dict, out_dir: str):
    """Figure 4: DemDiv + Equalized Odds comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    methods = [m for m in METHOD_ORDER if m in data and m != "_meta"]
    x = np.arange(len(methods))
    width = 0.6
    colors = [COLORS[m] for m in methods]

    # DemDiv with CI
    divs = [data[m]["demdiv_mean"] for m in methods]
    div_ci_lo = [data[m]["demdiv_mean"] - data[m]["demdiv_ci95"][0] for m in methods]
    div_ci_hi = [data[m]["demdiv_ci95"][1] - data[m]["demdiv_mean"] for m in methods]

    ax1.bar(
        x,
        divs,
        width,
        yerr=[div_ci_lo, div_ci_hi],
        capsize=4,
        color=colors,
        alpha=0.85,
        edgecolor="white",
    )
    ax1.axhline(
        y=0.05, color="red", linestyle="--", alpha=0.7, label=r"$\epsilon = 0.05$"
    )
    ax1.set_ylabel("DemDiv (KL Divergence)")
    ax1.set_title("(a) Demographic Divergence")
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS[m] for m in methods], rotation=35, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Equalized Odds
    eqs = [data[m]["eqodds_mean"] for m in methods]
    eq_errs = [data[m]["eqodds_std"] for m in methods]

    ax2.bar(
        x,
        eqs,
        width,
        yerr=eq_errs,
        capsize=4,
        color=colors,
        alpha=0.85,
        edgecolor="white",
    )
    ax2.set_ylabel("Equalized Odds Gap")
    ax2.set_title("(b) Equalized Odds Gap (lower is better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABELS[m] for m in methods], rotation=35, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_fairness_metrics.png")
    plt.savefig(path)
    plt.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


def fig5_confusion_matrix_heatmap(data: dict, out_dir: str):
    """Figure 5: FairSwarm per-group confusion matrix heatmap."""
    groups = ["white", "black", "hispanic", "asian", "other"]
    metrics = ["tpr", "fpr"]
    metric_labels = ["TPR (Sensitivity)", "FPR (False Positive Rate)"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    methods = [m for m in METHOD_ORDER if m in data and m != "_meta"]

    for ax, metric, metric_label in zip(axes, metrics, metric_labels):
        matrix = []
        for m in methods:
            row = []
            pgcm = data[m].get("per_group_confusion_matrices", {})
            for g in groups:
                row.append(pgcm.get(g, {}).get(metric, 0))
            matrix.append(row)

        matrix = np.array(matrix)
        im = ax.imshow(
            matrix,
            cmap="RdYlGn" if metric == "tpr" else "RdYlGn_r",
            aspect="auto",
            vmin=0,
            vmax=0.8 if metric == "tpr" else 0.3,
        )

        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g.capitalize() for g in groups])
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([LABELS[m] for m in methods])
        ax.set_title(metric_label)

        # Add text annotations
        for i in range(len(methods)):
            for j in range(len(groups)):
                val = matrix[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                    fontweight="bold",
                )

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig5_fairness_heatmap.png")
    plt.savefig(path)
    plt.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


def generate_latex_main_table(data: dict, out_dir: str):
    """Table 1: Main results table for the paper."""
    methods = [m for m in METHOD_ORDER if m in data and m != "_meta"]

    rows = []
    for m in methods:
        d = data[m]
        name = LABELS[m]
        auc = f"{d['auc_mean']:.4f}"
        f"[{d['auc_ci95'][0]:.4f}, {d['auc_ci95'][1]:.4f}]"
        auprc = f"{d['auprc_mean']:.4f}"
        demdiv = f"{d['demdiv_mean']:.4f}"
        f"[{d['demdiv_ci95'][0]:.4f}, {d['demdiv_ci95'][1]:.4f}]"
        eqodds = f"{d['eqodds_mean']:.4f}"
        f1 = f"{d['f1_mean']:.4f}"

        # Bold the best values
        is_best_auc = d["auc_mean"] == max(
            data[mm]["auc_mean"] for mm in methods if mm != "all_clients"
        )
        is_best_div = d["demdiv_mean"] == min(
            data[mm]["demdiv_mean"] for mm in methods if mm != "all_clients"
        )

        if is_best_auc and m != "all_clients":
            auc = f"\\textbf{{{auc}}}"
        if is_best_div and m != "all_clients":
            demdiv = f"\\textbf{{{demdiv}}}"

        rows.append(f"    {name} & {auc} & {auprc} & {f1} & {demdiv} & {eqodds} \\\\")

    meta = data["_meta"]

    latex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Experimental results on MIMIC-III ({meta["n_trials"]} trials, {meta["n_rounds"]} rounds, {meta["n_clients"]} clients, coalition size {meta["coalition_size"]}). Best selection method values in \\textbf{{bold}} (excluding all-clients upper bound).}}
\\label{{tab:main-results}}
\\begin{{tabular}}{{lcccccc}}
\\toprule
Method & AUC-ROC & AUPRC & F1 & DemDiv & EqOdds Gap \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    path = os.path.join(out_dir, "table1_main_results.tex")
    with open(path, "w") as f:
        f.write(latex)
    print(f"  Saved: {path}")
    return latex


def generate_latex_statistical_table(data: dict, out_dir: str):
    """Table 2: Statistical comparison (FairSwarm vs baselines)."""
    # Hardcoded from experiment output since p-values aren't in JSON
    comparisons = [
        ("Random", "AUC", 0.0000, "+1.277", "large"),
        ("Random", "DemDiv", 0.0009, "-0.902", "large"),
        ("Random", "EqOdds", 0.0315, "-0.569", "medium"),
        ("Round Robin", "AUC", 0.0000, "+9.537", "large"),
        ("Round Robin", "DemDiv", 0.0000, "-1.820", "large"),
        ("Round Robin", "EqOdds", 0.0000, "-2.124", "large"),
        ("FedAvg (all)", "AUC", 0.0003, "-1.007", "large"),
        ("FedAvg (all)", "DemDiv", 0.0000, "+2.599", "large"),
        ("FedAvg (all)", "EqOdds", 0.0000, "+1.805", "large"),
        ("Greedy-Fair", "AUC", 0.0000, "+1.917", "large"),
        ("Greedy-Fair", "DemDiv", 0.0000, "+2.068", "large"),
        ("Greedy-Fair", "EqOdds", 0.0000, "-1.465", "large"),
        ("Greedy-Acc+Fair", "AUC", 0.0000, "+4.000", "large"),
        ("Greedy-Acc+Fair", "DemDiv", 0.0000, "-2.747", "large"),
        ("Greedy-Acc+Fair", "EqOdds", 0.2817, "-0.281", "small"),
    ]

    rows = []
    for baseline, metric, pval, cohend, effect in comparisons:
        sig = (
            "***"
            if pval < 0.001
            else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        )
        rows.append(
            f"    {baseline} & {metric} & {pval:.4f} & {sig} & {cohend} & {effect} \\\\"
        )

    latex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Statistical comparisons: FairSwarm vs.~baselines (Holm--Bonferroni corrected, 30 trials). Significance: ${{***}}$ $p<0.001$, ${{**}}$ $p<0.01$, ${{*}}$ $p<0.05$.}}
\\label{{tab:statistical-tests}}
\\begin{{tabular}}{{llcccc}}
\\toprule
Baseline & Metric & $p$-value & Sig. & Cohen's $d$ & Effect \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    path = os.path.join(out_dir, "table2_statistical_tests.tex")
    with open(path, "w") as f:
        f.write(latex)
    print(f"  Saved: {path}")
    return latex


def generate_latex_per_group_table(data: dict, out_dir: str):
    """Table 3: Per-group fairness metrics."""
    groups = ["white", "black", "hispanic", "asian", "other"]
    methods = [m for m in METHOD_ORDER if m in data and m != "_meta"]

    rows = []
    for m in methods:
        pgcm = data[m].get("per_group_confusion_matrices", {})
        tprs = [f"{pgcm.get(g, {}).get('tpr', 0):.3f}" for g in groups]
        [f"{pgcm.get(g, {}).get('fpr', 0):.3f}" for g in groups]

        # TPR range (max - min) as fairness measure
        tpr_vals = [pgcm.get(g, {}).get("tpr", 0) for g in groups]
        tpr_range = max(tpr_vals) - min(tpr_vals)

        rows.append(f"    {LABELS[m]} & {' & '.join(tprs)} & {tpr_range:.3f} \\\\")

    latex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Per-group true positive rates (sensitivity) across demographic groups. TPR Range measures the gap between the highest and lowest group TPR (lower is fairer).}}
\\label{{tab:per-group-tpr}}
\\begin{{tabular}}{{lccccc|c}}
\\toprule
Method & White & Black & Hispanic & Asian & Other & TPR Range \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    path = os.path.join(out_dir, "table3_per_group_tpr.tex")
    with open(path, "w") as f:
        f.write(latex)
    print(f"  Saved: {path}")
    return latex


def main():
    print("=" * 60)
    print("Generating publication figures and tables")
    print("=" * 60)

    data = load_results()

    fig_dir = "figures"
    tab_dir = "tables"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    print(
        f"\nLoaded results for {len([k for k in data if k != '_meta'])} methods, "
        f"{data['_meta']['n_trials']} trials\n"
    )

    print("Generating figures:")
    fig1_auc_demdiv_barplot(data, fig_dir)
    fig2_pareto_frontier(data, fig_dir)
    fig3_per_group_auc(data, fig_dir)
    fig4_fairness_metrics(data, fig_dir)
    fig5_confusion_matrix_heatmap(data, fig_dir)

    print("\nGenerating LaTeX tables:")
    generate_latex_main_table(data, tab_dir)
    generate_latex_statistical_table(data, tab_dir)
    generate_latex_per_group_table(data, tab_dir)

    print(f"\nDone! {5} figures in {fig_dir}/, {3} tables in {tab_dir}/")


if __name__ == "__main__":
    main()
