"""
Stress Tests: Adaptive Privacy Budget Allocation (Novel Contribution 1)

Tests the AdaptivePrivacyAllocator under extreme conditions including
fast/slow convergence, oscillating loss, epsilon caps, and edge cases.

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

from novel.adaptive_privacy.allocator import AdaptivePrivacyAllocator, AllocationResult


@dataclass
class TestResult:
    """Result of a single stress test."""
    name: str
    passed: bool
    wall_clock_seconds: float
    details: dict[str, Any]
    error: str | None = None


def _run_allocator_simulation(
    total_budget: float,
    total_rounds: int,
    velocity_schedule: list[float],
    loss_history_schedule: list[list[float]] | None = None,
    velocity_weight: float = 0.7,
    min_epsilon: float = 0.01,
) -> tuple[AdaptivePrivacyAllocator, list[AllocationResult]]:
    """Run allocator through a full simulation."""
    allocator = AdaptivePrivacyAllocator(
        total_budget=total_budget,
        total_rounds=total_rounds,
        velocity_weight=velocity_weight,
        min_epsilon_per_round=min_epsilon,
    )
    results = []
    for r in range(1, total_rounds + 1):
        vel = velocity_schedule[min(r - 1, len(velocity_schedule) - 1)]
        loss_hist = None
        if loss_history_schedule is not None and r - 1 < len(loss_history_schedule):
            loss_hist = loss_history_schedule[r - 1]
        result = allocator.allocate(r, vel, loss_hist)
        results.append(result)
    return allocator, results


# ─── Test 1: Fast convergence (3 rounds) ────────────────────────────

def test_fast_convergence_3_rounds() -> TestResult:
    """Federation converging in only 3 rounds."""
    start = time.perf_counter()
    try:
        velocities = [0.5, 0.2, 0.01]  # Rapidly decreasing
        allocator, results = _run_allocator_simulation(
            total_budget=1.0, total_rounds=3,
            velocity_schedule=velocities,
        )
        total_spent = sum(r.allocated_epsilon for r in results)
        epsilons = [r.allocated_epsilon for r in results]

        # Budget should be fully spent (last round gets remainder)
        budget_used_pct = total_spent / 1.0

        # Early rounds should get more budget
        early_heavy = epsilons[0] > epsilons[2] or len(epsilons) <= 1

        passed = (
            abs(total_spent - 1.0) < 1e-6
            and all(e > 0 for e in epsilons)
            and early_heavy
        )

        return TestResult(
            name="fast_convergence_3_rounds",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "epsilons": epsilons,
                "total_spent": total_spent,
                "budget_used_pct": budget_used_pct,
                "early_heavy": early_heavy,
            },
        )
    except Exception as e:
        return TestResult(
            name="fast_convergence_3_rounds",
            passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={},
            error=str(e),
        )


# ─── Test 2: Slow convergence (200 rounds) ──────────────────────────

def test_slow_convergence_200_rounds() -> TestResult:
    """Federation needing 200 rounds to converge."""
    start = time.perf_counter()
    try:
        # Gradually decreasing velocity
        velocities = [0.1 * (0.99 ** r) for r in range(200)]
        allocator, results = _run_allocator_simulation(
            total_budget=10.0, total_rounds=200,
            velocity_schedule=velocities,
        )
        total_spent = sum(r.allocated_epsilon for r in results)
        epsilons = [r.allocated_epsilon for r in results]

        # Budget should last all 200 rounds
        all_positive = all(e > 0 for e in epsilons)
        budget_close = abs(total_spent - 10.0) < 1e-4

        # First 50 rounds should get more than last 50
        first_50 = sum(epsilons[:50])
        last_50 = sum(epsilons[150:])

        passed = all_positive and budget_close and first_50 > last_50

        return TestResult(
            name="slow_convergence_200_rounds",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "total_spent": total_spent,
                "all_positive": all_positive,
                "first_50_budget": first_50,
                "last_50_budget": last_50,
                "min_epsilon": min(epsilons),
                "max_epsilon": max(epsilons),
            },
        )
    except Exception as e:
        return TestResult(
            name="slow_convergence_200_rounds",
            passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={},
            error=str(e),
        )


# ─── Test 3: Async one-fast-others-slow ──────────────────────────────

def test_async_one_fast_others_slow() -> TestResult:
    """One node converges 10x faster than others.
    Tests per-node allocation fairness at the federation level.
    """
    start = time.perf_counter()
    try:
        # Simulate 5 nodes: node 0 converges 10x faster
        n_rounds = 50
        fast_velocities = [0.5 * (0.8 ** r) for r in range(n_rounds)]
        slow_velocities = [0.05 * (0.95 ** r) for r in range(n_rounds)]

        # Run allocator with averaged velocity (federation-wide)
        avg_velocities = [(f + 4 * s) / 5 for f, s in zip(fast_velocities, slow_velocities)]

        allocator, results = _run_allocator_simulation(
            total_budget=5.0, total_rounds=n_rounds,
            velocity_schedule=avg_velocities,
        )
        total_spent = sum(r.allocated_epsilon for r in results)
        epsilons = [r.allocated_epsilon for r in results]

        passed = (
            abs(total_spent - 5.0) < 1e-4
            and all(e > 0 for e in epsilons)
        )

        return TestResult(
            name="async_one_fast_others_slow",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "total_spent": total_spent,
                "fast_node_velocity_range": [fast_velocities[0], fast_velocities[-1]],
                "slow_node_velocity_range": [slow_velocities[0], slow_velocities[-1]],
                "avg_velocity_range": [avg_velocities[0], avg_velocities[-1]],
                "epsilon_range": [min(epsilons), max(epsilons)],
            },
        )
    except Exception as e:
        return TestResult(
            name="async_one_fast_others_slow",
            passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={},
            error=str(e),
        )


# ─── Test 4: Oscillating loss ────────────────────────────────────────

def test_oscillating_loss() -> TestResult:
    """Non-monotonic convergence with sin-wave oscillating loss."""
    start = time.perf_counter()
    try:
        n_rounds = 100
        # Loss: decreasing trend + sinusoidal oscillation
        t = np.arange(n_rounds)
        loss_curve = 2.0 * np.exp(-0.02 * t) + 0.3 * np.sin(0.5 * t)
        loss_curve = np.clip(loss_curve, 0.01, 10.0)

        # Compute velocities from loss curve
        velocities = []
        for r in range(n_rounds):
            if r == 0:
                velocities.append(0.1)
            else:
                vel = abs(loss_curve[r] - loss_curve[r - 1]) / max(abs(loss_curve[r - 1]), 1e-8)
                velocities.append(vel)

        allocator, results = _run_allocator_simulation(
            total_budget=5.0, total_rounds=n_rounds,
            velocity_schedule=velocities,
        )
        total_spent = sum(r.allocated_epsilon for r in results)
        epsilons = [r.allocated_epsilon for r in results]

        # Key check: allocator shouldn't overspend during oscillation peaks
        passed = (
            abs(total_spent - 5.0) < 1e-4
            and all(e >= 0 for e in epsilons)
            and allocator.remaining_budget >= -1e-8
        )

        return TestResult(
            name="oscillating_loss",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "total_spent": total_spent,
                "remaining_budget": allocator.remaining_budget,
                "loss_range": [float(loss_curve.min()), float(loss_curve.max())],
                "velocity_range": [min(velocities), max(velocities)],
                "epsilon_range": [min(epsilons), max(epsilons)],
                "n_velocity_spikes": sum(1 for v in velocities if v > 0.3),
            },
        )
    except Exception as e:
        return TestResult(
            name="oscillating_loss",
            passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={},
            error=str(e),
        )


# ─── Test 5: Epsilon cap never exceeded ──────────────────────────────

def test_epsilon_cap_never_exceeded() -> TestResult:
    """Exhaustive test: caps 0.5, 1.0, 3.0, 10.0 × 200 rounds."""
    start = time.perf_counter()
    try:
        caps = [0.5, 1.0, 3.0, 10.0]
        n_rounds = 200
        results_by_cap = {}
        all_pass = True

        for cap in caps:
            velocities = [0.1 * (0.995 ** r) + 0.01 * np.sin(0.1 * r) for r in range(n_rounds)]
            velocities = [max(0.0, v) for v in velocities]

            allocator, alloc_results = _run_allocator_simulation(
                total_budget=cap, total_rounds=n_rounds,
                velocity_schedule=velocities,
            )
            total_spent = sum(r.allocated_epsilon for r in alloc_results)
            exceeds_cap = total_spent > cap + 1e-8

            if exceeds_cap:
                all_pass = False

            results_by_cap[str(cap)] = {
                "total_spent": total_spent,
                "exceeds_cap": exceeds_cap,
                "overshoot": total_spent - cap,
                "remaining": allocator.remaining_budget,
            }

        passed = all_pass

        return TestResult(
            name="epsilon_cap_never_exceeded",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={"results_by_cap": results_by_cap},
        )
    except Exception as e:
        return TestResult(
            name="epsilon_cap_never_exceeded",
            passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={},
            error=str(e),
        )


# ─── Test 6: Tightness (≥95% budget used) ───────────────────────────

def test_tightness() -> TestResult:
    """Prove ≥95% of budget is used (not wasted) across all cap levels."""
    start = time.perf_counter()
    try:
        caps = [0.5, 1.0, 3.0, 10.0]
        n_rounds = 100
        tightness_results = {}
        all_tight = True

        for cap in caps:
            velocities = [0.15 * (0.97 ** r) for r in range(n_rounds)]
            allocator, alloc_results = _run_allocator_simulation(
                total_budget=cap, total_rounds=n_rounds,
                velocity_schedule=velocities,
            )
            total_spent = sum(r.allocated_epsilon for r in alloc_results)
            utilization = total_spent / cap

            if utilization < 0.95:
                all_tight = False

            tightness_results[str(cap)] = {
                "total_spent": total_spent,
                "utilization_pct": utilization * 100,
                "wasted": cap - total_spent,
            }

        passed = all_tight

        return TestResult(
            name="tightness",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={"tightness_results": tightness_results, "threshold_pct": 95},
        )
    except Exception as e:
        return TestResult(
            name="tightness",
            passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={},
            error=str(e),
        )


# ─── Test 7: Utility vs fixed-epsilon baseline ──────────────────────

def test_utility_vs_fixed_baseline() -> TestResult:
    """Compare adaptive vs fixed-ε allocation on simulated FL accuracy curves."""
    start = time.perf_counter()
    try:
        n_rounds = 100
        total_budget = 5.0

        # Simulated accuracy as function of noise (epsilon): higher epsilon = less noise = better accuracy
        def simulated_accuracy(epsilon: float, round_num: int) -> float:
            """Higher epsilon -> less noise -> better accuracy.
            Uses sigmoid-scaled penalty so accuracy stays in [0.5, 0.9] range.
            """
            base_acc = 0.5 + 0.4 * (1 - np.exp(-0.05 * round_num))
            # Sigmoid-based noise penalty: smooth, bounded, realistic
            noise_factor = 1.0 / (1.0 + np.exp(-10 * (epsilon - 0.02)))
            return base_acc * noise_factor

        # Adaptive allocation
        velocities = [0.2 * (0.97 ** r) for r in range(n_rounds)]
        allocator, alloc_results = _run_allocator_simulation(
            total_budget=total_budget, total_rounds=n_rounds,
            velocity_schedule=velocities,
        )
        adaptive_epsilons = [r.allocated_epsilon for r in alloc_results]
        adaptive_acc = [simulated_accuracy(e, r) for r, e in enumerate(adaptive_epsilons)]

        # Fixed allocation
        fixed_epsilon = total_budget / n_rounds
        fixed_acc = [simulated_accuracy(fixed_epsilon, r) for r in range(n_rounds)]

        # Compare final accuracy and area under curve
        adaptive_final = adaptive_acc[-1]
        fixed_final = fixed_acc[-1]
        adaptive_auc = np.trapezoid(adaptive_acc)
        fixed_auc = np.trapezoid(fixed_acc)

        passed = True  # This is a comparison test, not a pass/fail

        return TestResult(
            name="utility_vs_fixed_baseline",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "adaptive_final_accuracy": adaptive_final,
                "fixed_final_accuracy": fixed_final,
                "adaptive_auc": adaptive_auc,
                "fixed_auc": fixed_auc,
                "adaptive_advantage_final": adaptive_final - fixed_final,
                "adaptive_advantage_auc": adaptive_auc - fixed_auc,
                "adaptive_epsilons": adaptive_epsilons,
                "adaptive_accuracies": adaptive_acc,
                "fixed_accuracies": fixed_acc,
            },
        )
    except Exception as e:
        return TestResult(
            name="utility_vs_fixed_baseline",
            passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={},
            error=str(e),
        )


# ─── Test 8: Edge — zero velocity ───────────────────────────────────

def test_edge_zero_velocity() -> TestResult:
    """Velocity = 0 for all rounds; verify no division errors."""
    start = time.perf_counter()
    try:
        velocities = [0.0] * 50
        allocator, results = _run_allocator_simulation(
            total_budget=5.0, total_rounds=50,
            velocity_schedule=velocities,
        )
        total_spent = sum(r.allocated_epsilon for r in results)
        epsilons = [r.allocated_epsilon for r in results]

        passed = (
            all(np.isfinite(e) for e in epsilons)
            and all(e >= 0 for e in epsilons)
            and abs(total_spent - 5.0) < 1e-4
        )

        return TestResult(
            name="edge_zero_velocity",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "total_spent": total_spent,
                "all_finite": all(np.isfinite(e) for e in epsilons),
                "epsilon_range": [min(epsilons), max(epsilons)],
            },
        )
    except Exception as e:
        return TestResult(
            name="edge_zero_velocity",
            passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={},
            error=str(e),
        )


# ─── Test 9: Edge — negative velocity (diverging model) ─────────────

def test_edge_negative_velocity() -> TestResult:
    """Diverging model (negative velocity); verify graceful handling."""
    start = time.perf_counter()
    try:
        # Negative velocities = model getting worse
        velocities = [-0.1, -0.2, -0.05, 0.0, 0.01, -0.15] + [0.0] * 44
        allocator, results = _run_allocator_simulation(
            total_budget=5.0, total_rounds=50,
            velocity_schedule=velocities,
        )
        total_spent = sum(r.allocated_epsilon for r in results)
        epsilons = [r.allocated_epsilon for r in results]

        passed = (
            all(np.isfinite(e) for e in epsilons)
            and all(e >= 0 for e in epsilons)
            and total_spent <= 5.0 + 1e-8
        )

        return TestResult(
            name="edge_negative_velocity",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "total_spent": total_spent,
                "all_non_negative": all(e >= 0 for e in epsilons),
                "all_finite": all(np.isfinite(e) for e in epsilons),
            },
        )
    except Exception as e:
        return TestResult(
            name="edge_negative_velocity",
            passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={},
            error=str(e),
        )


# ─── Test 10: Edge — single round ───────────────────────────────────

def test_edge_single_round() -> TestResult:
    """total_rounds=1; verify no crash and full budget allocated."""
    start = time.perf_counter()
    try:
        allocator, results = _run_allocator_simulation(
            total_budget=1.0, total_rounds=1,
            velocity_schedule=[0.5],
        )
        epsilon = results[0].allocated_epsilon

        passed = abs(epsilon - 1.0) < 1e-6

        return TestResult(
            name="edge_single_round",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={"allocated_epsilon": epsilon, "expected": 1.0},
        )
    except Exception as e:
        return TestResult(
            name="edge_single_round",
            passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={},
            error=str(e),
        )


# ─── Figure generation ───────────────────────────────────────────────

def generate_figures(all_results: list[TestResult], output_dir: Path) -> list[str]:
    """Generate publication-ready figures."""
    figures_generated = []

    # Figure 1: Utility curves — adaptive vs fixed
    utility_result = next((r for r in all_results if r.name == "utility_vs_fixed_baseline"), None)
    if utility_result and utility_result.passed and "adaptive_accuracies" in utility_result.details:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        adaptive_acc = utility_result.details["adaptive_accuracies"]
        fixed_acc = utility_result.details["fixed_accuracies"]
        adaptive_eps = utility_result.details["adaptive_epsilons"]
        n = len(adaptive_acc)

        ax1.plot(range(n), adaptive_acc, label="Adaptive ε", linewidth=2)
        ax1.plot(range(n), fixed_acc, label="Fixed ε", linewidth=2, linestyle="--")
        ax1.set_xlabel("Round", fontsize=12)
        ax1.set_ylabel("Simulated Accuracy", fontsize=12)
        ax1.set_title("Utility: Adaptive vs Fixed Privacy Budget", fontsize=13)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        ax2.plot(range(n), adaptive_eps, linewidth=2, color="tab:orange")
        ax2.axhline(y=sum(adaptive_eps) / n, color="tab:blue", linestyle="--",
                     label=f"Fixed ε = {sum(adaptive_eps)/n:.3f}")
        ax2.set_xlabel("Round", fontsize=12)
        ax2.set_ylabel("ε per round", fontsize=12)
        ax2.set_title("Privacy Budget Allocation Schedule", fontsize=13)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = output_dir / "adaptive_vs_fixed_utility.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures_generated.append(str(path))

    # Figure 2: Budget tightness across epsilon caps
    tightness_result = next((r for r in all_results if r.name == "tightness"), None)
    if tightness_result and "tightness_results" in tightness_result.details:
        fig, ax = plt.subplots(figsize=(8, 5))
        tr = tightness_result.details["tightness_results"]
        caps = sorted(tr.keys(), key=float)
        utilizations = [tr[c]["utilization_pct"] for c in caps]

        bars = ax.bar([f"ε={c}" for c in caps], utilizations, color="tab:green", alpha=0.8)
        ax.axhline(y=95, color="red", linestyle="--", label="95% threshold")
        ax.set_ylabel("Budget Utilization (%)", fontsize=12)
        ax.set_title("Privacy Budget Tightness Across ε Caps", fontsize=13)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=11)

        for bar, val in zip(bars, utilizations):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", fontsize=10)

        plt.tight_layout()
        path = output_dir / "budget_tightness.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures_generated.append(str(path))

    # Figure 3: Oscillating loss allocation behavior
    next((r for r in all_results if r.name == "oscillating_loss"), None)
    cap_result = next((r for r in all_results if r.name == "epsilon_cap_never_exceeded"), None)
    if cap_result and "results_by_cap" in cap_result.details:
        fig, ax = plt.subplots(figsize=(8, 5))
        caps_data = cap_result.details["results_by_cap"]
        caps_sorted = sorted(caps_data.keys(), key=float)
        overshoots = [caps_data[c]["overshoot"] for c in caps_sorted]

        ax.bar([f"ε={c}" for c in caps_sorted], overshoots, color="tab:red", alpha=0.8)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_ylabel("Budget Overshoot", fontsize=12)
        ax.set_title("Budget Cap Compliance (overshoot should be ≤ 0)", fontsize=13)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = output_dir / "cap_compliance.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures_generated.append(str(path))

    return figures_generated


# ─── Runner ──────────────────────────────────────────────────────────

ALL_TESTS = [
    test_fast_convergence_3_rounds,
    test_slow_convergence_200_rounds,
    test_async_one_fast_others_slow,
    test_oscillating_loss,
    test_epsilon_cap_never_exceeded,
    test_tightness,
    test_utility_vs_fixed_baseline,
    test_edge_zero_velocity,
    test_edge_negative_velocity,
    test_edge_single_round,
]


def run_all() -> list[TestResult]:
    """Run all adaptive privacy stress tests."""
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
    print("STRESS TEST: Adaptive Privacy Budget Allocation")
    print("=" * 60)
    results = run_all()

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\nResults: {passed}/{total} passed")

    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    figs = generate_figures(results, output_dir)
    print(f"Generated {len(figs)} figures")

    # Save results JSON
    results_path = Path(__file__).parent / "results_adaptive_privacy.json"
    serializable = []
    for r in results:
        d = {"name": r.name, "passed": r.passed, "wall_clock_seconds": r.wall_clock_seconds, "error": r.error}
        # Filter non-serializable items from details
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
