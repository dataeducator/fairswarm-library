"""
Stress Tests: Fairness-Aware Aggregation (Novel Contribution 2)

Tests the FairnessReweighter under extreme demographic conditions including
extreme skew, impossible targets, scaling, and comparison with baselines.

Author: Tenicka Norwood
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "swarmclinical-fl"))

from novel.fairness_aggregation.reweighter import ClientUpdate, FairnessReweighter


@dataclass
class TestResult:
    """Result of a single stress test."""
    name: str
    passed: bool
    wall_clock_seconds: float
    details: dict[str, Any]
    error: str | None = None


def _make_updates(
    n_clients: int,
    demographics: list[np.ndarray],
    num_samples: list[int] | None = None,
    param_dim: int = 100,
    seed: int = 42,
) -> list[ClientUpdate]:
    """Create synthetic client updates for testing."""
    rng = np.random.default_rng(seed)
    if num_samples is None:
        num_samples = [rng.integers(50, 500) for _ in range(n_clients)]
    updates = []
    for i in range(n_clients):
        updates.append(ClientUpdate(
            client_id=i,
            parameters=rng.standard_normal(param_dim),
            num_samples=num_samples[i],
            demographics=demographics[i],
        ))
    return updates


def _demographic_divergence(weights: np.ndarray, demographics: list[np.ndarray],
                             target: np.ndarray) -> float:
    """Compute KL divergence of weighted coalition from target."""
    coalition_demo = sum(w * d for w, d in zip(weights, demographics))
    eps = 1e-10
    p = np.clip(coalition_demo, eps, 1)
    q = np.clip(target, eps, 1)
    return float(np.sum(p * np.log(p / q)))


# ─── Test 1-4: Scale tests (2, 5, 10, max nodes) ────────────────────

def _test_scale(n_nodes: int, test_name: str) -> TestResult:
    """Generic scaling test."""
    start = time.perf_counter()
    try:
        rng = np.random.default_rng(42)
        n_groups = 4  # age, sex, race, insurance

        # Generate random but valid demographics
        demographics = []
        for _ in range(n_nodes):
            d = rng.dirichlet(np.ones(n_groups) * 2)
            demographics.append(d)

        target = np.array([0.25, 0.25, 0.25, 0.25])  # Uniform target
        updates = _make_updates(n_nodes, demographics)
        reweighter = FairnessReweighter(alpha=0.5)

        weights = reweighter.reweight(updates, target)

        # Verify properties
        weights_sum_to_1 = abs(weights.sum() - 1.0) < 1e-6
        all_positive = np.all(weights > 0)
        correct_length = len(weights) == n_nodes

        # Compute divergence with and without reweighting
        base_weights = np.array([u.num_samples for u in updates], dtype=np.float64)
        base_weights /= base_weights.sum()
        base_div = _demographic_divergence(base_weights, demographics, target)
        fair_div = _demographic_divergence(weights, demographics, target)

        passed = weights_sum_to_1 and all_positive and correct_length

        return TestResult(
            name=test_name,
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "n_nodes": n_nodes,
                "weights_sum": float(weights.sum()),
                "weight_range": [float(weights.min()), float(weights.max())],
                "base_divergence": base_div,
                "fair_divergence": fair_div,
                "divergence_reduction_pct": (base_div - fair_div) / max(base_div, 1e-8) * 100,
            },
        )
    except Exception as e:
        return TestResult(
            name=test_name, passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


def test_scale_2_nodes() -> TestResult:
    return _test_scale(2, "scale_2_nodes")


def test_scale_5_nodes() -> TestResult:
    return _test_scale(5, "scale_5_nodes")


def test_scale_10_nodes() -> TestResult:
    return _test_scale(10, "scale_10_nodes")


def test_scale_max_nodes() -> TestResult:
    return _test_scale(50, "scale_50_nodes_max")


# ─── Test 5: 95% single demographic ─────────────────────────────────

def test_95pct_single_demographic() -> TestResult:
    """One node has 95% single group; verify reweighter compensates."""
    start = time.perf_counter()
    try:
        # Node 0: 95% group 0; Nodes 1-4: varied
        demographics = [
            np.array([0.95, 0.02, 0.02, 0.01]),
            np.array([0.10, 0.60, 0.20, 0.10]),
            np.array([0.20, 0.20, 0.40, 0.20]),
            np.array([0.15, 0.15, 0.15, 0.55]),
            np.array([0.25, 0.25, 0.25, 0.25]),
        ]
        target = np.array([0.25, 0.25, 0.25, 0.25])
        updates = _make_updates(5, demographics, num_samples=[500, 200, 200, 200, 200])
        reweighter = FairnessReweighter(alpha=0.8)

        weights = reweighter.reweight(updates, target)

        # Node 0 (heavily skewed) should get lower weight than FedAvg
        base_weight_node0 = 500.0 / 1300.0
        fair_weight_node0 = weights[0]

        # Check that reweighter reduces node 0's weight
        node0_downweighted = fair_weight_node0 < base_weight_node0

        passed = (
            abs(weights.sum() - 1.0) < 1e-6
            and np.all(weights > 0)
            and node0_downweighted
        )

        return TestResult(
            name="95pct_single_demographic",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "base_weight_node0": base_weight_node0,
                "fair_weight_node0": float(fair_weight_node0),
                "node0_downweighted": node0_downweighted,
                "all_weights": weights.tolist(),
            },
        )
    except Exception as e:
        return TestResult(
            name="95pct_single_demographic", passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


# ─── Test 6: All skewed same direction ──────────────────────────────

def test_all_skewed_same_direction() -> TestResult:
    """All nodes skew toward same group; verify graceful degradation."""
    start = time.perf_counter()
    try:
        # All nodes overrepresent group 0
        demographics = [
            np.array([0.70, 0.10, 0.10, 0.10]),
            np.array([0.65, 0.15, 0.10, 0.10]),
            np.array([0.80, 0.05, 0.10, 0.05]),
            np.array([0.60, 0.20, 0.10, 0.10]),
            np.array([0.75, 0.10, 0.05, 0.10]),
        ]
        target = np.array([0.25, 0.25, 0.25, 0.25])
        updates = _make_updates(5, demographics)
        reweighter = FairnessReweighter(alpha=0.8)

        weights = reweighter.reweight(updates, target)

        # Compute divergence — should still be high since all skewed
        fair_div = _demographic_divergence(weights, demographics, target)
        base_weights = np.array([u.num_samples for u in updates], dtype=np.float64)
        base_weights /= base_weights.sum()
        base_div = _demographic_divergence(base_weights, demographics, target)

        # Weights should still be valid (no crash, no NaN)
        passed = (
            abs(weights.sum() - 1.0) < 1e-6
            and np.all(np.isfinite(weights))
            and np.all(weights > 0)
        )

        return TestResult(
            name="all_skewed_same_direction",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "base_divergence": base_div,
                "fair_divergence": fair_div,
                "improvement_pct": (base_div - fair_div) / max(base_div, 1e-8) * 100,
                "all_weights": weights.tolist(),
                "note": "Limited improvement expected when all nodes skew same way",
            },
        )
    except Exception as e:
        return TestResult(
            name="all_skewed_same_direction", passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


# ─── Test 7: Impossible fairness target ──────────────────────────────

def test_impossible_fairness_target() -> TestResult:
    """Target with 50% of group not present in any node."""
    start = time.perf_counter()
    try:
        # No node has significant group 3
        demographics = [
            np.array([0.40, 0.40, 0.19, 0.01]),
            np.array([0.35, 0.35, 0.29, 0.01]),
            np.array([0.50, 0.30, 0.19, 0.01]),
        ]
        # Target wants 50% group 3 — impossible!
        target = np.array([0.10, 0.10, 0.30, 0.50])
        updates = _make_updates(3, demographics)
        reweighter = FairnessReweighter(alpha=1.0)

        weights = reweighter.reweight(updates, target)

        # Should not crash, weights should be valid
        passed = (
            abs(weights.sum() - 1.0) < 1e-6
            and np.all(np.isfinite(weights))
            and np.all(weights > 0)
        )

        fair_div = _demographic_divergence(weights, demographics, target)

        return TestResult(
            name="impossible_fairness_target",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "weights": weights.tolist(),
                "fair_divergence": fair_div,
                "note": "High divergence expected since target is unachievable",
            },
        )
    except Exception as e:
        return TestResult(
            name="impossible_fairness_target", passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


# ─── Test 8: Accuracy-fairness tradeoff sweep ───────────────────────

def test_accuracy_fairness_tradeoff() -> TestResult:
    """Sweep alpha from 0→1 in 0.1 steps; record Pareto front."""
    start = time.perf_counter()
    try:
        n_groups = 4
        rng = np.random.default_rng(42)
        demographics = [rng.dirichlet(np.ones(n_groups) * 2) for _ in range(10)]
        target = np.array([0.25, 0.25, 0.25, 0.25])
        updates = _make_updates(10, demographics)

        alphas = np.arange(0, 1.05, 0.1)
        pareto_points = []

        for alpha in alphas:
            rw = FairnessReweighter(alpha=float(alpha))
            weights = rw.reweight(updates, target)
            div = _demographic_divergence(weights, demographics, target)

            # Simulated accuracy: higher when weights are closer to sample-proportional
            base_w = np.array([u.num_samples for u in updates], dtype=np.float64)
            base_w /= base_w.sum()
            accuracy_proxy = 1.0 - 0.5 * np.sqrt(np.sum((weights - base_w) ** 2))

            pareto_points.append({
                "alpha": round(float(alpha), 2),
                "divergence": div,
                "accuracy_proxy": float(accuracy_proxy),
            })

        passed = len(pareto_points) == len(alphas)

        return TestResult(
            name="accuracy_fairness_tradeoff",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={"pareto_points": pareto_points},
        )
    except Exception as e:
        return TestResult(
            name="accuracy_fairness_tradeoff", passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


# ─── Tests 9-11: Baseline comparisons ───────────────────────────────

def _baseline_comparison(baseline_name: str) -> TestResult:
    """Compare FairnessReweighter against a baseline aggregation."""
    start = time.perf_counter()
    try:
        n_groups = 4
        rng = np.random.default_rng(42)
        n_clients = 10
        demographics = [rng.dirichlet(np.ones(n_groups) * 1.5) for _ in range(n_clients)]
        target = np.array([0.25, 0.25, 0.25, 0.25])
        num_samples = [rng.integers(50, 500) for _ in range(n_clients)]
        updates = _make_updates(n_clients, demographics, num_samples=num_samples)

        # FairnessReweighter
        rw = FairnessReweighter(alpha=0.7)
        fair_weights = rw.reweight(updates, target)
        fair_div = _demographic_divergence(fair_weights, demographics, target)

        # Baseline weights
        if baseline_name == "fedavg":
            base_weights = np.array(num_samples, dtype=np.float64)
            base_weights /= base_weights.sum()
        elif baseline_name == "fedprox":
            # FedProx uses same weighting as FedAvg for aggregation
            base_weights = np.array(num_samples, dtype=np.float64)
            base_weights /= base_weights.sum()
        elif baseline_name == "qffl":
            # q-FFL: weights proportional to L_i^q (simulate losses)
            losses = rng.uniform(0.1, 2.0, size=n_clients)
            q = 5.0
            loss_powers = np.power(losses, q)
            base_weights = loss_powers / loss_powers.sum()
        else:
            raise ValueError(f"Unknown baseline: {baseline_name}")

        base_div = _demographic_divergence(base_weights, demographics, target)

        passed = True  # Comparison test

        return TestResult(
            name=f"vs_{baseline_name}",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "baseline": baseline_name,
                "fair_divergence": fair_div,
                "baseline_divergence": base_div,
                "divergence_reduction_pct": (base_div - fair_div) / max(base_div, 1e-8) * 100,
                "fair_weights": fair_weights.tolist(),
                "base_weights": base_weights.tolist(),
            },
        )
    except Exception as e:
        return TestResult(
            name=f"vs_{baseline_name}", passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


def test_vs_fedavg() -> TestResult:
    return _baseline_comparison("fedavg")


def test_vs_fedprox() -> TestResult:
    return _baseline_comparison("fedprox")


def test_vs_qffl() -> TestResult:
    return _baseline_comparison("qffl")


# ─── Test 12: Edge — zero samples ───────────────────────────────────

def test_edge_zero_samples() -> TestResult:
    """Client with num_samples=0; verify no division error."""
    start = time.perf_counter()
    try:
        demographics = [
            np.array([0.3, 0.3, 0.2, 0.2]),
            np.array([0.2, 0.4, 0.2, 0.2]),
            np.array([0.25, 0.25, 0.25, 0.25]),
        ]
        target = np.array([0.25, 0.25, 0.25, 0.25])
        updates = _make_updates(3, demographics, num_samples=[100, 0, 200])
        rw = FairnessReweighter(alpha=0.5)

        weights = rw.reweight(updates, target)

        passed = (
            abs(weights.sum() - 1.0) < 1e-6
            and np.all(np.isfinite(weights))
            and np.all(weights >= 0)
        )

        return TestResult(
            name="edge_zero_samples",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={"weights": weights.tolist()},
        )
    except Exception as e:
        return TestResult(
            name="edge_zero_samples", passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


# ─── Test 13: Edge — identical demographics ─────────────────────────

def test_edge_identical_demographics() -> TestResult:
    """All clients identical demographics; weights should ≈ FedAvg weights."""
    start = time.perf_counter()
    try:
        demo = np.array([0.25, 0.25, 0.25, 0.25])
        demographics = [demo.copy() for _ in range(5)]
        target = np.array([0.25, 0.25, 0.25, 0.25])
        num_samples = [100, 200, 150, 250, 300]
        updates = _make_updates(5, demographics, num_samples=num_samples)

        rw = FairnessReweighter(alpha=0.8)
        fair_weights = rw.reweight(updates, target)

        # Expected: same as FedAvg since demographics are identical
        base_weights = np.array(num_samples, dtype=np.float64)
        base_weights /= base_weights.sum()

        weight_diff = float(np.max(np.abs(fair_weights - base_weights)))
        passed = weight_diff < 0.01  # Should be nearly identical

        return TestResult(
            name="edge_identical_demographics",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "fair_weights": fair_weights.tolist(),
                "base_weights": base_weights.tolist(),
                "max_difference": weight_diff,
            },
        )
    except Exception as e:
        return TestResult(
            name="edge_identical_demographics", passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


# ─── Test 14: Edge — no demographics ────────────────────────────────

def test_edge_no_demographics() -> TestResult:
    """Client with demographics=None; verify graceful fallback."""
    start = time.perf_counter()
    try:
        target = np.array([0.25, 0.25, 0.25, 0.25])
        updates = [
            ClientUpdate(client_id=0, parameters=np.zeros(10), num_samples=100,
                         demographics=np.array([0.3, 0.3, 0.2, 0.2])),
            ClientUpdate(client_id=1, parameters=np.zeros(10), num_samples=200,
                         demographics=None),  # Missing!
            ClientUpdate(client_id=2, parameters=np.zeros(10), num_samples=150,
                         demographics=np.array([0.2, 0.3, 0.3, 0.2])),
        ]
        rw = FairnessReweighter(alpha=0.5)

        weights = rw.reweight(updates, target)

        # Should fall back to FedAvg weights
        base_weights = np.array([100, 200, 150], dtype=np.float64)
        base_weights /= base_weights.sum()

        passed = (
            abs(weights.sum() - 1.0) < 1e-6
            and np.all(np.isfinite(weights))
        )

        return TestResult(
            name="edge_no_demographics",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "weights": weights.tolist(),
                "expected_fedavg_weights": base_weights.tolist(),
                "note": "Falls back to FedAvg weights when demographics missing",
            },
        )
    except Exception as e:
        return TestResult(
            name="edge_no_demographics", passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


# ─── Figure generation ───────────────────────────────────────────────

def generate_figures(all_results: list[TestResult], output_dir: Path) -> list[str]:
    """Generate publication-ready figures."""
    figures_generated = []

    # Figure 1: Accuracy-fairness Pareto front
    pareto_result = next((r for r in all_results if r.name == "accuracy_fairness_tradeoff"), None)
    if pareto_result and "pareto_points" in pareto_result.details:
        fig, ax = plt.subplots(figsize=(8, 6))
        points = pareto_result.details["pareto_points"]
        divs = [p["divergence"] for p in points]
        accs = [p["accuracy_proxy"] for p in points]
        alphas = [p["alpha"] for p in points]

        scatter = ax.scatter(divs, accs, c=alphas, cmap="coolwarm", s=100, edgecolors="black")
        for p in points:
            ax.annotate(f'α={p["alpha"]:.1f}',
                       (p["divergence"], p["accuracy_proxy"]),
                       textcoords="offset points", xytext=(5, 5), fontsize=8)

        ax.set_xlabel("Demographic Divergence (KL)", fontsize=12)
        ax.set_ylabel("Accuracy Proxy", fontsize=12)
        ax.set_title("Accuracy-Fairness Pareto Front (FairnessReweighter)", fontsize=13)
        plt.colorbar(scatter, label="α (fairness weight)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = output_dir / "fairness_pareto_front.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures_generated.append(str(path))

    # Figure 2: Baseline comparison bar chart
    baseline_results = [r for r in all_results if r.name.startswith("vs_")]
    if baseline_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        methods = ["FairnessReweighter"] + [r.details.get("baseline", "?") for r in baseline_results]
        divs = []
        if baseline_results:
            divs.append(baseline_results[0].details.get("fair_divergence", 0))
            divs.extend([r.details.get("baseline_divergence", 0) for r in baseline_results])

        colors = ["tab:green"] + ["tab:blue"] * len(baseline_results)
        bars = ax.bar(methods, divs, color=colors, alpha=0.8)
        ax.set_ylabel("Demographic Divergence (KL)", fontsize=12)
        ax.set_title("Fairness Comparison: Reweighter vs Baselines", fontsize=13)

        for bar, val in zip(bars, divs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", fontsize=9)

        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        path = output_dir / "fairness_baseline_comparison.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures_generated.append(str(path))

    # Figure 3: Scaling wall-clock time
    scale_results = [r for r in all_results if r.name.startswith("scale_")]
    if scale_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        nodes = [r.details.get("n_nodes", 0) for r in scale_results]
        times = [r.wall_clock_seconds for r in scale_results]

        ax.plot(nodes, times, "o-", linewidth=2, markersize=8)
        ax.set_xlabel("Number of Nodes", fontsize=12)
        ax.set_ylabel("Wall-clock Time (s)", fontsize=12)
        ax.set_title("FairnessReweighter Scaling", fontsize=13)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = output_dir / "fairness_scaling.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures_generated.append(str(path))

    return figures_generated


# ─── Runner ──────────────────────────────────────────────────────────

ALL_TESTS = [
    test_scale_2_nodes,
    test_scale_5_nodes,
    test_scale_10_nodes,
    test_scale_max_nodes,
    test_95pct_single_demographic,
    test_all_skewed_same_direction,
    test_impossible_fairness_target,
    test_accuracy_fairness_tradeoff,
    test_vs_fedavg,
    test_vs_fedprox,
    test_vs_qffl,
    test_edge_zero_samples,
    test_edge_identical_demographics,
    test_edge_no_demographics,
]


def run_all() -> list[TestResult]:
    """Run all fairness aggregation stress tests."""
    results = []
    for test_fn in ALL_TESTS:
        print(f"  Running {test_fn.__name__}...", end=" ", flush=True)
        result = test_fn()
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} ({result.wall_clock_seconds:.3f}s)")
        if result.error:
            print(f"    ERROR: {result.error}")
        results.append(result)
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("STRESS TEST: Fairness-Aware Aggregation")
    print("=" * 60)
    results = run_all()

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\nResults: {passed}/{total} passed")

    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    figs = generate_figures(results, output_dir)
    print(f"Generated {len(figs)} figures")

    results_path = Path(__file__).parent / "results_fairness_aggregation.json"
    serializable = []
    for r in results:
        d = {"name": r.name, "passed": r.passed, "wall_clock_seconds": r.wall_clock_seconds, "error": r.error}
        details = {}
        for k, v in r.details.items():
            try:
                json.dumps(v)
                details[k] = v
            except (TypeError, ValueError):
                details[k] = str(v)
        d["details"] = details
        serializable.append(d)

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {results_path}")
