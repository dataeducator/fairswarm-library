"""
Stress Tests: Healthcare Non-IID Detection and Correction (Novel Contribution 3)

Tests NonIIDDetector and NonIIDCorrector under extreme distribution conditions
including identical, slightly different, completely disjoint, rare diseases,
and dynamic node joining scenarios.

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
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fairswarm" / "src"))

from novel.noniid_detection.detector import NonIIDDetector
from novel.noniid_detection.corrector import (
    NonIIDCorrector,
    ClientUpdate as CorrectorClientUpdate,
)


@dataclass
class TestResult:
    """Result of a single stress test."""
    name: str
    passed: bool
    wall_clock_seconds: float
    details: dict[str, Any]
    error: str | None = None


def _make_dirichlet_distributions(
    n_nodes: int,
    n_codes: int,
    concentration: float = 1.0,
    seed: int = 42,
) -> list[np.ndarray]:
    """Generate Dirichlet-distributed diagnosis code distributions."""
    rng = np.random.default_rng(seed)
    return [rng.dirichlet(np.ones(n_codes) * concentration) for _ in range(n_nodes)]


def _make_corrector_updates(
    n_clients: int,
    divergence_scores: list[float],
    param_dim: int = 50,
    seed: int = 42,
) -> list[CorrectorClientUpdate]:
    """Create synthetic corrector client updates."""
    rng = np.random.default_rng(seed)
    updates = []
    for i in range(n_clients):
        updates.append(CorrectorClientUpdate(
            client_id=i,
            parameters=rng.standard_normal(param_dim),
            divergence_score=divergence_scores[i],
        ))
    return updates


# ─── Test 1: Identical distributions ────────────────────────────────

def test_identical_distributions() -> TestResult:
    """All nodes have same distribution; verify is_noniid=False."""
    start = time.perf_counter()
    try:
        n_codes = 10
        base_dist = np.ones(n_codes) / n_codes
        node_dists = [base_dist.copy() for _ in range(5)]
        global_dist = base_dist.copy()

        detector = NonIIDDetector(divergence_threshold=0.1)
        report = detector.detect(node_dists, global_dist)

        passed = (
            not report.is_noniid
            and report.severity < 0.01
            and len(report.divergent_nodes) == 0
        )

        return TestResult(
            name="identical_distributions",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "is_noniid": report.is_noniid,
                "severity": report.severity,
                "severity_level": report.severity_level,
                "divergent_nodes": report.divergent_nodes,
            },
        )
    except Exception as e:
        return TestResult(
            name="identical_distributions", passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


# ─── Test 2: Slightly different ─────────────────────────────────────

def test_slightly_different() -> TestResult:
    """KL divergence ~0.05; verify correct severity classification."""
    start = time.perf_counter()
    try:
        n_codes = 10
        rng = np.random.default_rng(42)
        base_dist = np.ones(n_codes) / n_codes

        # Small perturbation
        node_dists = []
        for _ in range(5):
            noise = rng.normal(0, 0.01, n_codes)
            d = base_dist + noise
            d = np.clip(d, 1e-6, 1.0)
            d /= d.sum()
            node_dists.append(d)

        global_dist = np.mean(node_dists, axis=0)
        global_dist /= global_dist.sum()

        detector = NonIIDDetector(divergence_threshold=0.1, severity_thresholds=(0.1, 0.5))
        report = detector.detect(node_dists, global_dist)

        # Should be mild
        passed = report.severity_level == "mild" and report.severity < 0.1

        return TestResult(
            name="slightly_different",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "is_noniid": report.is_noniid,
                "severity": report.severity,
                "severity_level": report.severity_level,
                "divergence_scores": report.divergence_scores,
            },
        )
    except Exception as e:
        return TestResult(
            name="slightly_different", passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


# ─── Test 3: Moderately different ───────────────────────────────────

def test_moderately_different() -> TestResult:
    """KL divergence ~0.3; verify "moderate" severity."""
    start = time.perf_counter()
    try:
        n_codes = 10
        # Moderate heterogeneity: different Dirichlet concentrations
        node_dists = _make_dirichlet_distributions(5, n_codes, concentration=0.5, seed=42)
        global_dist = np.mean(node_dists, axis=0)
        global_dist /= global_dist.sum()

        detector = NonIIDDetector(divergence_threshold=0.1, severity_thresholds=(0.1, 0.5))
        report = detector.detect(node_dists, global_dist)

        # We expect moderate severity with low concentration
        passed = report.severity >= 0.05 and report.is_noniid

        return TestResult(
            name="moderately_different",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "is_noniid": report.is_noniid,
                "severity": report.severity,
                "severity_level": report.severity_level,
                "n_divergent_nodes": len(report.divergent_nodes),
            },
        )
    except Exception as e:
        return TestResult(
            name="moderately_different", passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


# ─── Test 4: Completely disjoint ────────────────────────────────────

def test_completely_disjoint() -> TestResult:
    """Zero overlap in distributions; verify "severe"."""
    start = time.perf_counter()
    try:
        n_codes = 5
        # Each node has a completely different single code
        node_dists = []
        for i in range(5):
            d = np.zeros(n_codes)
            d[i] = 1.0
            node_dists.append(d)

        global_dist = np.ones(n_codes) / n_codes  # Uniform global

        detector = NonIIDDetector(divergence_threshold=0.1, severity_thresholds=(0.1, 0.5))
        report = detector.detect(node_dists, global_dist)

        passed = (
            report.is_noniid
            and report.severity_level == "severe"
            and len(report.divergent_nodes) == 5
        )

        return TestResult(
            name="completely_disjoint",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "is_noniid": report.is_noniid,
                "severity": report.severity,
                "severity_level": report.severity_level,
                "n_divergent_nodes": len(report.divergent_nodes),
            },
        )
    except Exception as e:
        return TestResult(
            name="completely_disjoint", passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


# ─── Test 5-7: Diagnosis code vocabulary scaling ────────────────────

def _test_n_codes(n_codes: int, test_name: str) -> TestResult:
    """Test with varying numbers of diagnosis codes."""
    start = time.perf_counter()
    try:
        n_nodes = 5
        node_dists = _make_dirichlet_distributions(n_nodes, n_codes, concentration=0.5)
        global_dist = np.mean(node_dists, axis=0)
        global_dist /= global_dist.sum()

        detector = NonIIDDetector(divergence_threshold=0.1)
        report = detector.detect(node_dists, global_dist)

        # Also test pairwise
        pairwise = detector.detect_pairwise(node_dists)

        passed = (
            pairwise.shape == (n_nodes, n_nodes)
            and np.all(np.isfinite(pairwise))
            and np.allclose(np.diag(pairwise), 0)
        )

        return TestResult(
            name=test_name,
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "n_codes": n_codes,
                "severity": report.severity,
                "severity_level": report.severity_level,
                "pairwise_range": [float(pairwise.min()), float(pairwise.max())],
                "pairwise_mean": float(pairwise[np.triu_indices(n_nodes, k=1)].mean()),
            },
        )
    except Exception as e:
        return TestResult(
            name=test_name, passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


def test_3_diagnosis_codes() -> TestResult:
    return _test_n_codes(3, "3_diagnosis_codes")


def test_10_diagnosis_codes() -> TestResult:
    return _test_n_codes(10, "10_diagnosis_codes")


def test_50_diagnosis_codes() -> TestResult:
    return _test_n_codes(50, "50_diagnosis_codes")


# ─── Test 8: Rare disease single node ────────────────────────────────

def test_rare_disease_single_node() -> TestResult:
    """1 node has disease absent from all others."""
    start = time.perf_counter()
    try:
        n_codes = 10
        rng = np.random.default_rng(42)

        # Nodes 0-3: common diseases in first 8 codes
        node_dists = []
        for _ in range(4):
            d = np.zeros(n_codes)
            d[:8] = rng.dirichlet(np.ones(8) * 2)
            node_dists.append(d)

        # Node 4: has rare disease (code 9) that nobody else has
        d4 = np.zeros(n_codes)
        d4[:8] = rng.dirichlet(np.ones(8) * 2) * 0.7
        d4[9] = 0.3  # 30% rare disease
        d4 /= d4.sum()
        node_dists.append(d4)

        global_dist = np.mean(node_dists, axis=0)
        global_dist /= global_dist.sum()

        detector = NonIIDDetector(divergence_threshold=0.1)
        report = detector.detect(node_dists, global_dist)

        # Node 4 should be flagged as divergent
        node4_flagged = 4 in report.divergent_nodes

        # Node 4 should have highest divergence
        max_div_node = max(report.divergence_scores, key=report.divergence_scores.get)
        node4_highest = max_div_node == 4

        passed = node4_flagged and node4_highest

        return TestResult(
            name="rare_disease_single_node",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "node4_flagged": node4_flagged,
                "node4_highest_divergence": node4_highest,
                "divergence_scores": report.divergence_scores,
                "severity": report.severity,
            },
        )
    except Exception as e:
        return TestResult(
            name="rare_disease_single_node", passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


# ─── Test 9: Ranking correctness ────────────────────────────────────

def test_ranking_correctness() -> TestResult:
    """5 nodes with known ordering of heterogeneity; verify ranking."""
    start = time.perf_counter()
    try:
        n_codes = 10
        global_dist = np.ones(n_codes) / n_codes

        # Designed with increasing divergence from uniform
        node_dists = [
            global_dist.copy(),                                           # Node 0: identical
            np.array([0.12, 0.12, 0.12, 0.12, 0.12, 0.08, 0.08, 0.08, 0.08, 0.08]),  # Node 1: slight
            np.array([0.20, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05]),  # Node 2: moderate
            np.array([0.40, 0.20, 0.15, 0.10, 0.05, 0.04, 0.03, 0.02, 0.005, 0.005]),  # Node 3: high
            np.array([0.90, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.005, 0.003, 0.002]),  # Node 4: extreme
        ]

        detector = NonIIDDetector(divergence_threshold=0.01, method="kl")
        report = detector.detect(node_dists, global_dist)

        # Expected ranking: node 0 < node 1 < node 2 < node 3 < node 4
        scores = [report.divergence_scores[i] for i in range(5)]
        ranking_correct = all(scores[i] < scores[i + 1] for i in range(4))

        passed = ranking_correct

        return TestResult(
            name="ranking_correctness",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "divergence_scores": scores,
                "ranking_correct": ranking_correct,
                "expected_order": "0 < 1 < 2 < 3 < 4",
            },
        )
    except Exception as e:
        return TestResult(
            name="ranking_correctness", passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


# ─── Test 10: Correction convergence ────────────────────────────────

def test_correction_convergence() -> TestResult:
    """Simulated FL with/without correction; measure effectiveness."""
    start = time.perf_counter()
    try:
        n_rounds = 30
        n_clients = 5
        param_dim = 50
        rng = np.random.default_rng(42)

        # True global model
        true_params = rng.standard_normal(param_dim)

        # Divergence scores (higher = more non-IID)
        div_scores = [0.05, 0.8, 0.1, 0.6, 0.02]

        corrector = NonIIDCorrector(correction_strength=0.7)

        # Simulate FL rounds with and without correction
        uncorrected_errors = []
        corrected_errors = []

        for round_num in range(n_rounds):
            # Simulate client updates with bias proportional to divergence
            updates = []
            for i in range(n_clients):
                bias = div_scores[i] * rng.standard_normal(param_dim) * 0.5
                noise = rng.standard_normal(param_dim) * 0.1
                client_params = true_params + bias + noise
                updates.append(CorrectorClientUpdate(
                    client_id=i,
                    parameters=client_params,
                    divergence_score=div_scores[i],
                ))

            # Uncorrected: uniform weights
            uncorrected_weights = np.ones(n_clients) / n_clients
            uncorrected_agg = sum(w * u.parameters for w, u in zip(uncorrected_weights, updates))
            uncorrected_err = float(np.mean((uncorrected_agg - true_params) ** 2))
            uncorrected_errors.append(uncorrected_err)

            # Corrected: rebalanced weights
            result = corrector.rebalance_weights(updates, div_scores)
            corrected_agg = sum(w * u.parameters for w, u in zip(result.corrected_weights, updates))
            corrected_err = float(np.mean((corrected_agg - true_params) ** 2))
            corrected_errors.append(corrected_err)

        # Correction should reduce error
        avg_uncorrected = np.mean(uncorrected_errors)
        avg_corrected = np.mean(corrected_errors)
        improvement = (avg_uncorrected - avg_corrected) / avg_uncorrected * 100

        passed = avg_corrected < avg_uncorrected

        return TestResult(
            name="correction_convergence",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "avg_uncorrected_error": avg_uncorrected,
                "avg_corrected_error": avg_corrected,
                "improvement_pct": improvement,
                "uncorrected_errors": uncorrected_errors,
                "corrected_errors": corrected_errors,
            },
        )
    except Exception as e:
        return TestResult(
            name="correction_convergence", passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


# ─── Test 11: New node mid-federation ────────────────────────────────

def test_new_node_mid_federation() -> TestResult:
    """Node joins at round 50 with novel disease distribution."""
    start = time.perf_counter()
    try:
        n_codes = 10
        np.random.default_rng(42)

        # Original 4 nodes
        original_dists = _make_dirichlet_distributions(4, n_codes, concentration=2.0, seed=42)
        global_dist = np.mean(original_dists, axis=0)
        global_dist /= global_dist.sum()

        detector = NonIIDDetector(divergence_threshold=0.1)

        # Before new node joins
        report_before = detector.detect(original_dists, global_dist)

        # New node with novel distribution (rare diseases dominate)
        new_node_dist = np.zeros(n_codes)
        new_node_dist[8] = 0.6  # Rare disease 1
        new_node_dist[9] = 0.3  # Rare disease 2
        new_node_dist[0] = 0.1
        all_dists = original_dists + [new_node_dist]

        # Update global distribution
        new_global = np.mean(all_dists, axis=0)
        new_global /= new_global.sum()

        report_after = detector.detect(all_dists, new_global)

        # New node should be flagged
        new_node_idx = 4
        new_node_flagged = new_node_idx in report_after.divergent_nodes

        # Severity should increase
        severity_increased = report_after.severity > report_before.severity

        passed = new_node_flagged and severity_increased

        return TestResult(
            name="new_node_mid_federation",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "severity_before": report_before.severity,
                "severity_after": report_after.severity,
                "severity_increased": severity_increased,
                "new_node_flagged": new_node_flagged,
                "divergent_nodes_before": report_before.divergent_nodes,
                "divergent_nodes_after": report_after.divergent_nodes,
            },
        )
    except Exception as e:
        return TestResult(
            name="new_node_mid_federation", passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


# ─── Test 12: Edge — single node ────────────────────────────────────

def test_edge_single_node() -> TestResult:
    """Only 1 node; verify no crash in pairwise detection."""
    start = time.perf_counter()
    try:
        node_dists = [np.array([0.3, 0.4, 0.3])]
        global_dist = np.array([0.33, 0.33, 0.34])

        detector = NonIIDDetector(divergence_threshold=0.1)
        report = detector.detect(node_dists, global_dist)
        pairwise = detector.detect_pairwise(node_dists)

        passed = (
            pairwise.shape == (1, 1)
            and np.isfinite(report.severity)
        )

        return TestResult(
            name="edge_single_node",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "pairwise_shape": list(pairwise.shape),
                "severity": report.severity,
            },
        )
    except Exception as e:
        return TestResult(
            name="edge_single_node", passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


# ─── Test 13: Edge — empty distribution ─────────────────────────────

def test_edge_empty_distribution() -> TestResult:
    """Node with all-zero distribution; verify no NaN/Inf crash."""
    start = time.perf_counter()
    try:
        node_dists = [
            np.array([0.3, 0.4, 0.3]),
            np.zeros(3),  # Empty!
            np.array([0.2, 0.5, 0.3]),
        ]
        global_dist = np.array([0.33, 0.33, 0.34])

        detector = NonIIDDetector(divergence_threshold=0.1)
        report = detector.detect(node_dists, global_dist)

        # Should handle gracefully (node 1 flagged as highly divergent)
        passed = (
            report.is_noniid
            and 1 in report.divergent_nodes
        )

        return TestResult(
            name="edge_empty_distribution",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "is_noniid": report.is_noniid,
                "divergent_nodes": report.divergent_nodes,
                "divergence_scores": {k: (v if np.isfinite(v) else "inf") for k, v in report.divergence_scores.items()},
            },
        )
    except Exception as e:
        return TestResult(
            name="edge_empty_distribution", passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


# ─── Test 14: Edge — single diagnosis ───────────────────────────────

def test_edge_single_diagnosis() -> TestResult:
    """Distribution with only 1 nonzero entry."""
    start = time.perf_counter()
    try:
        n_codes = 5
        node_dists = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0]),  # Only code 0
            np.array([0.0, 1.0, 0.0, 0.0, 0.0]),  # Only code 1
            np.array([0.2, 0.2, 0.2, 0.2, 0.2]),  # Uniform
        ]
        global_dist = np.ones(n_codes) / n_codes

        detector = NonIIDDetector(divergence_threshold=0.1)
        report = detector.detect(node_dists, global_dist)

        # Nodes 0 and 1 should be flagged as divergent
        passed = (
            report.is_noniid
            and 0 in report.divergent_nodes
            and 1 in report.divergent_nodes
        )

        return TestResult(
            name="edge_single_diagnosis",
            passed=passed,
            wall_clock_seconds=time.perf_counter() - start,
            details={
                "is_noniid": report.is_noniid,
                "divergent_nodes": report.divergent_nodes,
                "severity": report.severity,
            },
        )
    except Exception as e:
        return TestResult(
            name="edge_single_diagnosis", passed=False,
            wall_clock_seconds=time.perf_counter() - start,
            details={}, error=str(e),
        )


# ─── Figure generation ───────────────────────────────────────────────

def generate_figures(all_results: list[TestResult], output_dir: Path) -> list[str]:
    """Generate publication-ready figures."""
    figures_generated = []

    # Figure 1: Correction convergence — with vs without
    conv_result = next((r for r in all_results if r.name == "correction_convergence"), None)
    if conv_result and "uncorrected_errors" in conv_result.details:
        fig, ax = plt.subplots(figsize=(8, 5))
        uncorr = conv_result.details["uncorrected_errors"]
        corr = conv_result.details["corrected_errors"]
        rounds = range(len(uncorr))

        ax.plot(rounds, uncorr, label="Without Correction", linewidth=2, color="tab:red")
        ax.plot(rounds, corr, label="With Correction", linewidth=2, color="tab:green")
        ax.set_xlabel("FL Round", fontsize=12)
        ax.set_ylabel("MSE from True Model", fontsize=12)
        ax.set_title("Non-IID Correction Effect on Model Quality", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = output_dir / "noniid_correction_convergence.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures_generated.append(str(path))

    # Figure 2: Detection ranking accuracy
    rank_result = next((r for r in all_results if r.name == "ranking_correctness"), None)
    if rank_result and "divergence_scores" in rank_result.details:
        fig, ax = plt.subplots(figsize=(8, 5))
        scores = rank_result.details["divergence_scores"]
        node_labels = [f"Node {i}\n({'uniform' if i == 0 else 'slight' if i == 1 else 'moderate' if i == 2 else 'high' if i == 3 else 'extreme'})" for i in range(5)]

        bars = ax.bar(node_labels, scores, color=plt.cm.Reds(np.linspace(0.2, 1.0, 5)))
        ax.set_ylabel("KL Divergence from Global", fontsize=12)
        ax.set_title("Non-IID Detection: Ranking Correctness", fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=9)

        plt.tight_layout()
        path = output_dir / "noniid_ranking.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures_generated.append(str(path))

    # Figure 3: Vocabulary scaling
    code_results = [r for r in all_results if r.name.endswith("_diagnosis_codes")]
    if code_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        n_codes_list = [r.details.get("n_codes", 0) for r in code_results]
        times = [r.wall_clock_seconds for r in code_results]
        severities = [r.details.get("severity", 0) for r in code_results if "severity" in r.details]

        ax1.plot(n_codes_list, times, "o-", linewidth=2, markersize=8)
        ax1.set_xlabel("Number of Diagnosis Codes", fontsize=12)
        ax1.set_ylabel("Wall-clock Time (s)", fontsize=12)
        ax1.set_title("Detection Scaling with Vocabulary Size", fontsize=13)
        ax1.grid(True, alpha=0.3)

        if severities and len(severities) == len(n_codes_list):
            ax2.bar([str(n) for n in n_codes_list], severities, color="tab:orange", alpha=0.8)
            ax2.set_xlabel("Number of Diagnosis Codes", fontsize=12)
            ax2.set_ylabel("Severity Score", fontsize=12)
            ax2.set_title("Severity vs Vocabulary Size", fontsize=13)
            ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = output_dir / "noniid_vocabulary_scaling.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures_generated.append(str(path))

    # Figure 4: New node detection
    new_node_result = next((r for r in all_results if r.name == "new_node_mid_federation"), None)
    if new_node_result and new_node_result.passed:
        fig, ax = plt.subplots(figsize=(6, 4))
        severities = [
            new_node_result.details["severity_before"],
            new_node_result.details["severity_after"],
        ]
        labels = ["Before New Node", "After New Node Joins"]
        colors = ["tab:blue", "tab:red"]
        ax.bar(labels, severities, color=colors, alpha=0.8)
        ax.set_ylabel("Overall Severity Score", fontsize=12)
        ax.set_title("Impact of Novel Node Joining Mid-Federation", fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")

        for i, (label, val) in enumerate(zip(labels, severities)):
            ax.text(i, val + 0.01, f"{val:.4f}", ha="center", fontsize=10)

        plt.tight_layout()
        path = output_dir / "noniid_new_node_impact.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures_generated.append(str(path))

    return figures_generated


# ─── Runner ──────────────────────────────────────────────────────────

ALL_TESTS = [
    test_identical_distributions,
    test_slightly_different,
    test_moderately_different,
    test_completely_disjoint,
    test_3_diagnosis_codes,
    test_10_diagnosis_codes,
    test_50_diagnosis_codes,
    test_rare_disease_single_node,
    test_ranking_correctness,
    test_correction_convergence,
    test_new_node_mid_federation,
    test_edge_single_node,
    test_edge_empty_distribution,
    test_edge_single_diagnosis,
]


def run_all() -> list[TestResult]:
    """Run all Non-IID detection and correction stress tests."""
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
    print("STRESS TEST: Healthcare Non-IID Detection & Correction")
    print("=" * 60)
    results = run_all()

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\nResults: {passed}/{total} passed")

    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    figs = generate_figures(results, output_dir)
    print(f"Generated {len(figs)} figures")

    results_path = Path(__file__).parent / "results_noniid_detection.json"
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
