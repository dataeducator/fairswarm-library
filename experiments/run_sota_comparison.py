"""
SOTA Comparison Experiment for FairSwarm JMLR Paper.

Compares FairSwarm against 8 baselines across multiple scales:
- Random Selection
- FedAvg (all clients)
- Greedy (by dataset size)
- FedFDP (Ling et al., 2024)
- DivFL (Balakrishnan et al., ICLR 2022)
- Oort (Lai et al., OSDI 2021)
- Power-of-Choice (Cho et al., 2022)
- SubTrunc (Wang et al., 2024)

Tests at scales: n=[20, 50, 100], k=[4, 8], m=n/2

Extended Metrics:
- DemDiv (KL divergence from target)
- Client Dissimilarity (std of per-client fitness proxy)
- AUC-ROC (proxy via normalized accuracy)
- Wall-clock time (seconds)

Statistical Analysis:
- Multiple trials per configuration (default 10)
- 95% confidence intervals via t-distribution
- Welch's t-test p-values comparing each baseline to FairSwarm
- Cohen's d effect sizes

Output:
- JSON results file for paper tables
- Console summary with key findings

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
Institution: Meharry Medical College

Usage:
    python run_sota_comparison.py
    python run_sota_comparison.py --n_clients 50 --k 4 --n_trials 5
    python run_sota_comparison.py --quick  # Fast sanity check (2 trials, n=20 only)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup: ensure fairswarm and experiments are importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_FAIRSWARM_SRC = _PROJECT_ROOT / "fairswarm" / "src"

# Add fairswarm source to path so `import fairswarm` works
if str(_FAIRSWARM_SRC) not in sys.path:
    sys.path.insert(0, str(_FAIRSWARM_SRC))

# Add experiments directory so `from statistics_utils import ...` works
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ---------------------------------------------------------------------------
# FairSwarm imports
# ---------------------------------------------------------------------------
from fairswarm import FairSwarm, FairSwarmConfig  # noqa: E402
from fairswarm.core.client import Client, create_synthetic_clients  # noqa: E402
from fairswarm.demographics.distribution import DemographicDistribution  # noqa: E402
from fairswarm.demographics.divergence import kl_divergence  # noqa: E402
from fairswarm.fitness.fairness import (  # noqa: E402
    DemographicFitness,
    compute_coalition_demographics,
)
from fairswarm.fitness.equity import client_dissimilarity  # noqa: E402

# ---------------------------------------------------------------------------
# Baseline imports
# ---------------------------------------------------------------------------
from baselines.selection_baselines import (  # noqa: E402
    DivFL,
    Oort,
    PowerOfChoice,
    SelectionBaseline,
    SubTrunc,
)
from baselines.grey_wolf_optimizer import GreyWolfOptimizer  # noqa: E402
from baselines.random_selection import RandomSelectionBaseline, RandomSelectionConfig  # noqa: E402
from baselines.fedavg import FedAvgBaseline, FedAvgConfig  # noqa: E402
from baselines.greedy import GreedyBaseline, GreedyConfig, GreedyCriterion  # noqa: E402
from baselines.fair_dpfl_scs import FairDPFL_SCS, FairDPFLConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Statistics utilities
# ---------------------------------------------------------------------------
from statistics_utils import compare_means, get_git_hash, mean_ci  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SOTAComparisonConfig:
    """
    Configuration for the SOTA comparison experiment.

    Attributes:
        n_clients_list: Client pool sizes to test.
        k_list: Numbers of demographic groups to test.
        n_trials: Independent trials per (n, k) configuration.
        n_iterations_fairswarm: FairSwarm PSO iterations.
        n_iterations_fairdpfl: FedFDP rounds.
        confidence: Confidence level for CIs and tests.
        output_dir: Directory for JSON result files.
        seed: Master random seed.
    """

    n_clients_list: List[int] = field(default_factory=lambda: [20, 50, 100])
    k_list: List[int] = field(default_factory=lambda: [4, 8])
    n_trials: int = 10
    n_iterations_fairswarm: int = 100
    n_iterations_fairdpfl: int = 50
    confidence: float = 0.95
    output_dir: str = "results"
    seed: int = 42


# =============================================================================
# Per-Trial Result Container
# =============================================================================


@dataclass
class MethodTrialResult:
    """
    Result for a single method on a single trial.

    Attributes:
        method: Algorithm name.
        coalition: Selected client indices.
        dem_div: KL divergence from target (Definition 2).
        client_dissim: Std of per-client accuracy proxy.
        auc_proxy: Proxy AUC-ROC from normalised dataset coverage.
        wall_time_s: Wall-clock time in seconds.
    """

    method: str
    coalition: List[int]
    dem_div: float
    client_dissim: float
    auc_proxy: float
    wall_time_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "coalition_size": len(self.coalition),
            "dem_div": self.dem_div,
            "client_dissim": self.client_dissim,
            "auc_proxy": self.auc_proxy,
            "wall_time_s": self.wall_time_s,
        }


# =============================================================================
# Metric Computation Helpers
# =============================================================================


def _compute_dem_div(
    coalition: List[int],
    clients: List[Client],
    target: DemographicDistribution,
) -> float:
    """
    Compute DemDiv(S) = D_KL(delta_S || delta*).

    Args:
        coalition: Selected client indices.
        clients: Full client pool.
        target: Target demographic distribution.

    Returns:
        KL divergence (non-negative).
    """
    if not coalition:
        return float("inf")
    coalition_demo = compute_coalition_demographics(coalition, clients)
    return float(kl_divergence(coalition_demo, target.as_array()))


def _compute_client_dissim(
    coalition: List[int],
    clients: List[Client],
) -> float:
    """
    Compute client dissimilarity CD(S) = std({acc_proxy_i : i in S}).

    Uses normalised dataset size as the per-client accuracy proxy,
    matching the proxy in ClientDissimilarityFitness.

    Args:
        coalition: Selected client indices.
        clients: Full client pool.

    Returns:
        Standard deviation of per-client accuracy proxy.
    """
    if len(coalition) < 2:
        return 0.0
    max_size = max(clients[i].dataset_size for i in coalition)
    if max_size <= 0:
        return 0.0
    per_client_acc = [clients[i].dataset_size / max_size for i in coalition]
    return client_dissimilarity(per_client_acc)


def _compute_auc_proxy(
    coalition: List[int],
    clients: List[Client],
) -> float:
    """
    Compute a proxy AUC-ROC based on normalised data coverage.

    The proxy models the well-established empirical relationship
    between dataset coverage and model performance:
        auc_proxy = 0.5 + 0.4 * (coalition_data / total_data)

    This maps the range [0, 1] of data fraction to [0.5, 0.9] AUC,
    reflecting that a random classifier achieves AUC 0.5 and more
    data generally improves performance with diminishing returns.

    Args:
        coalition: Selected client indices.
        clients: Full client pool.

    Returns:
        Proxy AUC-ROC in [0.5, 0.9].
    """
    if not coalition:
        return 0.5
    total_data = sum(c.dataset_size for c in clients)
    if total_data <= 0:
        return 0.5
    coalition_data = sum(clients[i].dataset_size for i in coalition)
    fraction = coalition_data / total_data
    return 0.5 + 0.4 * fraction


# =============================================================================
# Baseline Runners
# =============================================================================


def run_fairswarm_trial(
    clients: List[Client],
    coalition_size: int,
    target: DemographicDistribution,
    n_iterations: int,
    seed: int,
) -> MethodTrialResult:
    """
    Run a single FairSwarm trial.

    Args:
        clients: Client pool.
        coalition_size: Target m.
        target: Target demographic distribution.
        n_iterations: PSO iterations.
        seed: Trial-specific seed.

    Returns:
        MethodTrialResult for FairSwarm.
    """
    config = FairSwarmConfig(
        swarm_size=30,
        max_iterations=n_iterations,
        coalition_size=coalition_size,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
        fairness_coefficient=0.5,
        fairness_weight=0.3,
        adaptive_fairness=True,
        seed=seed,
    )
    fitness_fn = DemographicFitness(target_distribution=target)
    optimizer = FairSwarm(
        clients=clients,
        coalition_size=coalition_size,
        config=config,
        target_distribution=target,
        seed=seed,
    )

    t0 = time.perf_counter()
    result = optimizer.optimize(fitness_fn=fitness_fn, n_iterations=n_iterations)
    wall_time = time.perf_counter() - t0

    coalition = result.coalition
    return MethodTrialResult(
        method="FairSwarm",
        coalition=coalition,
        dem_div=_compute_dem_div(coalition, clients, target),
        client_dissim=_compute_client_dissim(coalition, clients),
        auc_proxy=_compute_auc_proxy(coalition, clients),
        wall_time_s=wall_time,
    )


def run_random_trial(
    clients: List[Client],
    coalition_size: int,
    target: DemographicDistribution,
    seed: int,
) -> MethodTrialResult:
    """Run a single Random Selection trial (best of 100 random draws)."""
    fitness_fn = DemographicFitness(target_distribution=target)
    config = RandomSelectionConfig(
        coalition_size=coalition_size,
        n_iterations=100,
        seed=seed,
    )
    baseline = RandomSelectionBaseline(clients=clients, config=config)

    t0 = time.perf_counter()
    result = baseline.run(fitness_fn=fitness_fn, target_distribution=target.as_array())
    wall_time = time.perf_counter() - t0

    coalition = result.coalition
    return MethodTrialResult(
        method="Random",
        coalition=coalition,
        dem_div=_compute_dem_div(coalition, clients, target),
        client_dissim=_compute_client_dissim(coalition, clients),
        auc_proxy=_compute_auc_proxy(coalition, clients),
        wall_time_s=wall_time,
    )


def run_fedavg_trial(
    clients: List[Client],
    coalition_size: int,
    target: DemographicDistribution,
    seed: int,
) -> MethodTrialResult:
    """
    Run a single FedAvg trial.

    FedAvg uses ALL clients (no selection), so the coalition is the
    full client pool. coalition_size is ignored for selection but we
    report metrics on the full pool.
    """
    fitness_fn = DemographicFitness(target_distribution=target)
    config = FedAvgConfig(n_rounds=10, participation_rate=1.0, seed=seed)
    baseline = FedAvgBaseline(clients=clients, config=config)

    t0 = time.perf_counter()
    result = baseline.run(fitness_fn=fitness_fn, target_distribution=target.as_array())
    wall_time = time.perf_counter() - t0

    coalition = result.coalition
    return MethodTrialResult(
        method="FedAvg",
        coalition=coalition,
        dem_div=_compute_dem_div(coalition, clients, target),
        client_dissim=_compute_client_dissim(coalition, clients),
        auc_proxy=_compute_auc_proxy(coalition, clients),
        wall_time_s=wall_time,
    )


def run_greedy_trial(
    clients: List[Client],
    coalition_size: int,
    target: DemographicDistribution,
    seed: int,
) -> MethodTrialResult:
    """Run a single Greedy (fairness criterion) trial."""
    config = GreedyConfig(
        criterion=GreedyCriterion.FAIRNESS,
        coalition_size=coalition_size,
        seed=seed,
    )
    baseline = GreedyBaseline(
        clients=clients,
        target_distribution=target,
        config=config,
    )

    t0 = time.perf_counter()
    result = baseline.run()
    wall_time = time.perf_counter() - t0

    coalition = result.coalition
    return MethodTrialResult(
        method="Greedy",
        coalition=coalition,
        dem_div=_compute_dem_div(coalition, clients, target),
        client_dissim=_compute_client_dissim(coalition, clients),
        auc_proxy=_compute_auc_proxy(coalition, clients),
        wall_time_s=wall_time,
    )


def run_fairdpfl_trial(
    clients: List[Client],
    coalition_size: int,
    target: DemographicDistribution,
    n_rounds: int,
    seed: int,
) -> MethodTrialResult:
    """Run a single FedFDP trial."""
    fitness_fn = DemographicFitness(target_distribution=target)
    config = FairDPFLConfig(
        coalition_size=coalition_size,
        n_rounds=n_rounds,
        fairness_threshold=0.05,
        seed=seed,
    )
    baseline = FairDPFL_SCS(clients=clients, config=config)

    t0 = time.perf_counter()
    result = baseline.run(fitness_fn=fitness_fn, target_distribution=target.as_array())
    wall_time = time.perf_counter() - t0

    coalition = result.coalition
    return MethodTrialResult(
        method="FedFDP",
        coalition=coalition,
        dem_div=_compute_dem_div(coalition, clients, target),
        client_dissim=_compute_client_dissim(coalition, clients),
        auc_proxy=_compute_auc_proxy(coalition, clients),
        wall_time_s=wall_time,
    )


def _run_selection_baseline_trial(
    baseline: SelectionBaseline,
    method_name: str,
    clients: List[Client],
    coalition_size: int,
    target: DemographicDistribution,
) -> MethodTrialResult:
    """
    Run a single trial for a SelectionBaseline (DivFL, Oort, PoC, SubTrunc).

    Args:
        baseline: Instantiated SelectionBaseline.
        method_name: Display name for the method.
        clients: Client pool.
        coalition_size: Target coalition size m.
        target: Target demographic distribution.

    Returns:
        MethodTrialResult.
    """
    t0 = time.perf_counter()
    coalition = baseline.select(clients, coalition_size, target)
    wall_time = time.perf_counter() - t0

    return MethodTrialResult(
        method=method_name,
        coalition=coalition,
        dem_div=_compute_dem_div(coalition, clients, target),
        client_dissim=_compute_client_dissim(coalition, clients),
        auc_proxy=_compute_auc_proxy(coalition, clients),
        wall_time_s=wall_time,
    )


# =============================================================================
# Aggregation & Statistical Analysis
# =============================================================================


@dataclass
class AggregatedMethodResult:
    """
    Aggregated results for a single method across all trials.

    Attributes:
        method: Algorithm name.
        n_trials: Number of trials.
        dem_div: Dict with mean, ci_lower, ci_upper, std.
        client_dissim: Dict with mean, ci_lower, ci_upper, std.
        auc_proxy: Dict with mean, ci_lower, ci_upper, std.
        wall_time_s: Dict with mean, ci_lower, ci_upper, std.
        vs_fairswarm: Dict of p-values and Cohen's d for each metric
            comparing this method to FairSwarm (None for FairSwarm itself).
    """

    method: str
    n_trials: int
    dem_div: Dict[str, float]
    client_dissim: Dict[str, float]
    auc_proxy: Dict[str, float]
    wall_time_s: Dict[str, float]
    vs_fairswarm: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "method": self.method,
            "n_trials": self.n_trials,
            "dem_div": self.dem_div,
            "client_dissim": self.client_dissim,
            "auc_proxy": self.auc_proxy,
            "wall_time_s": self.wall_time_s,
        }
        if self.vs_fairswarm is not None:
            result["vs_fairswarm"] = self.vs_fairswarm
        return result


def _summarize_metric(
    values: List[float],
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Compute mean, CI, and std for a list of metric values.

    Args:
        values: Per-trial metric values.
        confidence: Confidence level.

    Returns:
        Dict with keys: mean, ci_lower, ci_upper, std, median, min, max.
    """
    ci = mean_ci(values, confidence=confidence)
    arr = np.asarray(values)
    return {
        "mean": ci.mean,
        "ci_lower": ci.lower,
        "ci_upper": ci.upper,
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _welch_t_test(
    group1: List[float],
    group2: List[float],
) -> Dict[str, float]:
    """
    Welch's t-test comparing two groups.

    Args:
        group1: First group (typically FairSwarm).
        group2: Second group (baseline).

    Returns:
        Dict with t_statistic, p_value, cohens_d.
    """
    result = compare_means(group1, group2)
    return {
        "t_statistic": result["t_statistic"],
        "p_value": result["p_value"],
        "cohens_d": result["cohens_d"],
        "significant_at_005": result["p_value"] < 0.05,
    }


def aggregate_results(
    all_trials: Dict[str, List[MethodTrialResult]],
    confidence: float = 0.95,
) -> Dict[str, AggregatedMethodResult]:
    """
    Aggregate per-trial results into summary statistics with
    Welch's t-test comparisons against FairSwarm.

    Args:
        all_trials: Dict mapping method name to list of MethodTrialResult.
        confidence: Confidence level for CIs.

    Returns:
        Dict mapping method name to AggregatedMethodResult.
    """
    aggregated: Dict[str, AggregatedMethodResult] = {}

    # Extract FairSwarm metric arrays for comparison
    fs_trials = all_trials.get("FairSwarm", [])
    fs_dem_div = [t.dem_div for t in fs_trials]
    fs_dissim = [t.client_dissim for t in fs_trials]
    fs_auc = [t.auc_proxy for t in fs_trials]
    fs_time = [t.wall_time_s for t in fs_trials]

    for method_name, trials in all_trials.items():
        n = len(trials)
        dem_divs = [t.dem_div for t in trials]
        dissims = [t.client_dissim for t in trials]
        aucs = [t.auc_proxy for t in trials]
        times = [t.wall_time_s for t in trials]

        # Statistical comparison vs FairSwarm
        vs_fairswarm: Optional[Dict[str, Any]] = None
        if method_name != "FairSwarm" and len(fs_trials) >= 2 and n >= 2:
            vs_fairswarm = {
                "dem_div": _welch_t_test(fs_dem_div, dem_divs),
                "client_dissim": _welch_t_test(fs_dissim, dissims),
                "auc_proxy": _welch_t_test(fs_auc, aucs),
                "wall_time_s": _welch_t_test(fs_time, times),
            }

        aggregated[method_name] = AggregatedMethodResult(
            method=method_name,
            n_trials=n,
            dem_div=_summarize_metric(dem_divs, confidence),
            client_dissim=_summarize_metric(dissims, confidence),
            auc_proxy=_summarize_metric(aucs, confidence),
            wall_time_s=_summarize_metric(times, confidence),
            vs_fairswarm=vs_fairswarm,
        )

    return aggregated


# =============================================================================
# Main Experiment Loop
# =============================================================================


def run_experiment_config(
    n_clients: int,
    k: int,
    exp_config: SOTAComparisonConfig,
) -> Dict[str, Any]:
    """
    Run all methods across all trials for a single (n, k) configuration.

    Args:
        n_clients: Number of clients in the pool.
        k: Number of demographic groups.
        exp_config: Experiment configuration.

    Returns:
        Dict with aggregated results and metadata for this config.
    """
    coalition_size = n_clients // 2
    target = DemographicDistribution.uniform(k)
    master_rng = np.random.default_rng(exp_config.seed)

    logger.info(
        "=" * 70 + "\n"
        f"  Configuration: n={n_clients}, k={k}, m={coalition_size}, "
        f"trials={exp_config.n_trials}\n" + "=" * 70
    )

    # Pre-generate per-trial seeds for reproducibility
    trial_seeds = master_rng.integers(0, 2**31, size=exp_config.n_trials).tolist()

    # Accumulate per-method results
    all_trials: Dict[str, List[MethodTrialResult]] = {
        "FairSwarm": [],
        "Random": [],
        "FedAvg": [],
        "Greedy": [],
        "FedFDP": [],
        "DivFL": [],
        "Oort": [],
        "Power-of-Choice": [],
        "SubTrunc": [],
        "GWO": [],
    }

    for trial_idx in range(exp_config.n_trials):
        trial_seed = trial_seeds[trial_idx]
        logger.info(
            f"  Trial {trial_idx + 1}/{exp_config.n_trials} (seed={trial_seed})"
        )

        # Generate clients for this trial
        clients = create_synthetic_clients(
            n_clients=n_clients,
            n_demographic_groups=k,
            seed=trial_seed,
        )

        # --- FairSwarm ---
        try:
            result = run_fairswarm_trial(
                clients,
                coalition_size,
                target,
                n_iterations=exp_config.n_iterations_fairswarm,
                seed=trial_seed,
            )
            all_trials["FairSwarm"].append(result)
            logger.info(
                f"    FairSwarm:      DemDiv={result.dem_div:.4f}  "
                f"CD={result.client_dissim:.4f}  AUC={result.auc_proxy:.4f}  "
                f"t={result.wall_time_s:.2f}s"
            )
        except Exception as e:
            logger.warning(f"    FairSwarm FAILED: {e}")

        # --- Random ---
        try:
            result = run_random_trial(clients, coalition_size, target, seed=trial_seed)
            all_trials["Random"].append(result)
            logger.info(
                f"    Random:         DemDiv={result.dem_div:.4f}  "
                f"CD={result.client_dissim:.4f}  AUC={result.auc_proxy:.4f}  "
                f"t={result.wall_time_s:.2f}s"
            )
        except Exception as e:
            logger.warning(f"    Random FAILED: {e}")

        # --- FedAvg ---
        try:
            result = run_fedavg_trial(clients, coalition_size, target, seed=trial_seed)
            all_trials["FedAvg"].append(result)
            logger.info(
                f"    FedAvg:         DemDiv={result.dem_div:.4f}  "
                f"CD={result.client_dissim:.4f}  AUC={result.auc_proxy:.4f}  "
                f"t={result.wall_time_s:.2f}s"
            )
        except Exception as e:
            logger.warning(f"    FedAvg FAILED: {e}")

        # --- Greedy ---
        try:
            result = run_greedy_trial(clients, coalition_size, target, seed=trial_seed)
            all_trials["Greedy"].append(result)
            logger.info(
                f"    Greedy:         DemDiv={result.dem_div:.4f}  "
                f"CD={result.client_dissim:.4f}  AUC={result.auc_proxy:.4f}  "
                f"t={result.wall_time_s:.2f}s"
            )
        except Exception as e:
            logger.warning(f"    Greedy FAILED: {e}")

        # --- FedFDP ---
        try:
            result = run_fairdpfl_trial(
                clients,
                coalition_size,
                target,
                n_rounds=exp_config.n_iterations_fairdpfl,
                seed=trial_seed,
            )
            all_trials["FedFDP"].append(result)
            logger.info(
                f"    FedFDP:   DemDiv={result.dem_div:.4f}  "
                f"CD={result.client_dissim:.4f}  AUC={result.auc_proxy:.4f}  "
                f"t={result.wall_time_s:.2f}s"
            )
        except Exception as e:
            logger.warning(f"    FedFDP FAILED: {e}")

        # --- SOTA Baselines from selection_baselines ---
        # DivFL
        try:
            bl = DivFL(seed=trial_seed)
            result = _run_selection_baseline_trial(
                bl,
                "DivFL",
                clients,
                coalition_size,
                target,
            )
            all_trials["DivFL"].append(result)
            logger.info(
                f"    DivFL:          DemDiv={result.dem_div:.4f}  "
                f"CD={result.client_dissim:.4f}  AUC={result.auc_proxy:.4f}  "
                f"t={result.wall_time_s:.2f}s"
            )
        except Exception as e:
            logger.warning(f"    DivFL FAILED: {e}")

        # Oort
        try:
            bl = Oort(seed=trial_seed)
            result = _run_selection_baseline_trial(
                bl,
                "Oort",
                clients,
                coalition_size,
                target,
            )
            all_trials["Oort"].append(result)
            logger.info(
                f"    Oort:           DemDiv={result.dem_div:.4f}  "
                f"CD={result.client_dissim:.4f}  AUC={result.auc_proxy:.4f}  "
                f"t={result.wall_time_s:.2f}s"
            )
        except Exception as e:
            logger.warning(f"    Oort FAILED: {e}")

        # Power-of-Choice
        try:
            bl = PowerOfChoice(seed=trial_seed)
            result = _run_selection_baseline_trial(
                bl,
                "Power-of-Choice",
                clients,
                coalition_size,
                target,
            )
            all_trials["Power-of-Choice"].append(result)
            logger.info(
                f"    Power-of-Choice:DemDiv={result.dem_div:.4f}  "
                f"CD={result.client_dissim:.4f}  AUC={result.auc_proxy:.4f}  "
                f"t={result.wall_time_s:.2f}s"
            )
        except Exception as e:
            logger.warning(f"    Power-of-Choice FAILED: {e}")

        # SubTrunc
        try:
            bl = SubTrunc(seed=trial_seed)
            result = _run_selection_baseline_trial(
                bl,
                "SubTrunc",
                clients,
                coalition_size,
                target,
            )
            all_trials["SubTrunc"].append(result)
            logger.info(
                f"    SubTrunc:       DemDiv={result.dem_div:.4f}  "
                f"CD={result.client_dissim:.4f}  AUC={result.auc_proxy:.4f}  "
                f"t={result.wall_time_s:.2f}s"
            )
        except Exception as e:
            logger.warning(f"    SubTrunc FAILED: {e}")

        # Grey Wolf Optimizer (GWO)
        try:
            bl = GreyWolfOptimizer(seed=trial_seed)
            result = _run_selection_baseline_trial(
                bl,
                "GWO",
                clients,
                coalition_size,
                target,
            )
            all_trials["GWO"].append(result)
            logger.info(
                f"    GWO:            DemDiv={result.dem_div:.4f}  "
                f"CD={result.client_dissim:.4f}  AUC={result.auc_proxy:.4f}  "
                f"t={result.wall_time_s:.2f}s"
            )
        except Exception as e:
            logger.warning(f"    GWO FAILED: {e}")

    # Aggregate across trials
    aggregated = aggregate_results(all_trials, confidence=exp_config.confidence)

    return {
        "config": {
            "n_clients": n_clients,
            "k": k,
            "coalition_size": coalition_size,
            "n_trials": exp_config.n_trials,
            "n_iterations_fairswarm": exp_config.n_iterations_fairswarm,
            "n_iterations_fairdpfl": exp_config.n_iterations_fairdpfl,
            "confidence": exp_config.confidence,
            "seed": exp_config.seed,
        },
        "methods": {name: agg.to_dict() for name, agg in aggregated.items()},
    }


# =============================================================================
# Console Summary
# =============================================================================


def print_summary(all_results: List[Dict[str, Any]]) -> None:
    """
    Print a human-readable summary of experiment results.

    Args:
        all_results: List of per-configuration result dicts.
    """
    print("\n" + "=" * 90)
    print("  SOTA COMPARISON RESULTS  --  FairSwarm JMLR Paper")
    print("=" * 90)

    for cfg_result in all_results:
        cfg = cfg_result["config"]
        methods = cfg_result["methods"]

        print(
            f"\n--- n={cfg['n_clients']}, k={cfg['k']}, "
            f"m={cfg['coalition_size']}, trials={cfg['n_trials']} ---\n"
        )

        # Header
        print(
            f"  {'Method':<20s}  {'DemDiv (KL)':>14s}  {'Client Dissim':>14s}  "
            f"{'AUC Proxy':>14s}  {'Time (s)':>12s}"
        )
        print("  " + "-" * 78)

        # Sort methods: FairSwarm first, then alphabetically
        method_names = sorted(methods.keys(), key=lambda x: (x != "FairSwarm", x))

        for name in method_names:
            m = methods[name]
            dd = m["dem_div"]
            cd = m["client_dissim"]
            auc = m["auc_proxy"]
            wt = m["wall_time_s"]

            # Format: mean +/- (half-width of CI)
            dd_hw = (dd["ci_upper"] - dd["ci_lower"]) / 2
            cd_hw = (cd["ci_upper"] - cd["ci_lower"]) / 2
            auc_hw = (auc["ci_upper"] - auc["ci_lower"]) / 2
            wt_hw = (wt["ci_upper"] - wt["ci_lower"]) / 2

            marker = " *" if name == "FairSwarm" else "  "
            print(
                f"{marker}{name:<20s}  "
                f"{dd['mean']:>6.4f}+/-{dd_hw:.4f}  "
                f"{cd['mean']:>6.4f}+/-{cd_hw:.4f}  "
                f"{auc['mean']:>6.4f}+/-{auc_hw:.4f}  "
                f"{wt['mean']:>5.3f}+/-{wt_hw:.3f}"
            )

        # Print significance tests vs FairSwarm
        print("\n  Welch's t-test vs FairSwarm (DemDiv, lower is better):")
        for name in method_names:
            if name == "FairSwarm":
                continue
            m = methods[name]
            vs = m.get("vs_fairswarm")
            if vs and "dem_div" in vs:
                dd_test = vs["dem_div"]
                sig = (
                    "***"
                    if dd_test["p_value"] < 0.001
                    else (
                        "**"
                        if dd_test["p_value"] < 0.01
                        else ("*" if dd_test["p_value"] < 0.05 else "n.s.")
                    )
                )
                print(
                    f"    {name:<20s}  p={dd_test['p_value']:.4f}  "
                    f"d={dd_test['cohens_d']:.3f}  {sig}"
                )

    print("\n" + "=" * 90)
    print("  Legend: * p<0.05, ** p<0.01, *** p<0.001, n.s. not significant")
    print("  Negative Cohen's d means FairSwarm has LOWER DemDiv (better fairness).")
    print("=" * 90 + "\n")


# =============================================================================
# Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SOTA Comparison Experiment for FairSwarm JMLR Paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_sota_comparison.py                        # Full experiment
    python run_sota_comparison.py --n_clients 50 --k 4   # Single config
    python run_sota_comparison.py --quick                 # Sanity check
    python run_sota_comparison.py --n_trials 30           # More trials
        """,
    )

    parser.add_argument(
        "--n_clients",
        type=int,
        nargs="+",
        default=None,
        help="Client pool size(s). Default: [20, 50, 100].",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=None,
        help="Number(s) of demographic groups. Default: [4, 8].",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="Number of independent trials per config. Default: 10.",
    )
    parser.add_argument(
        "--n_iter_fs",
        type=int,
        default=None,
        help="FairSwarm PSO iterations. Default: 100.",
    )
    parser.add_argument(
        "--n_iter_dpfl",
        type=int,
        default=None,
        help="FedFDP rounds. Default: 50.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed. Default: 42.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for JSON results. Default: results/.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick sanity check: 2 trials, n=20 only, k=4 only.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build experiment config
    exp_config = SOTAComparisonConfig(seed=args.seed)

    if args.quick:
        exp_config.n_clients_list = [20]
        exp_config.k_list = [4]
        exp_config.n_trials = 2
        exp_config.n_iterations_fairswarm = 50
        exp_config.n_iterations_fairdpfl = 20
        logger.info("QUICK MODE: 2 trials, n=20, k=4")
    else:
        if args.n_clients is not None:
            exp_config.n_clients_list = args.n_clients
        if args.k is not None:
            exp_config.k_list = args.k
        if args.n_trials is not None:
            exp_config.n_trials = args.n_trials
        if args.n_iter_fs is not None:
            exp_config.n_iterations_fairswarm = args.n_iter_fs
        if args.n_iter_dpfl is not None:
            exp_config.n_iterations_fairdpfl = args.n_iter_dpfl

    exp_config.output_dir = args.output_dir

    # Log experiment plan
    total_configs = len(exp_config.n_clients_list) * len(exp_config.k_list)
    total_trials = total_configs * exp_config.n_trials
    n_methods = 9  # FairSwarm + 8 baselines
    logger.info(
        f"SOTA Comparison Experiment\n"
        f"  n_clients: {exp_config.n_clients_list}\n"
        f"  k groups:  {exp_config.k_list}\n"
        f"  trials:    {exp_config.n_trials} per config\n"
        f"  configs:   {total_configs}\n"
        f"  methods:   {n_methods}\n"
        f"  total runs: {total_trials * n_methods}\n"
        f"  seed:      {exp_config.seed}\n"
    )

    # Run all configurations
    all_results: List[Dict[str, Any]] = []
    experiment_start = time.perf_counter()

    for n_clients in exp_config.n_clients_list:
        for k in exp_config.k_list:
            cfg_result = run_experiment_config(n_clients, k, exp_config)
            all_results.append(cfg_result)

    total_time = time.perf_counter() - experiment_start

    # Print console summary
    print_summary(all_results)
    logger.info(f"Total experiment time: {total_time:.1f}s")

    # Save JSON results
    output_dir = Path(_SCRIPT_DIR) / exp_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"sota_comparison_{timestamp}.json"

    output_data = {
        "experiment": "sota_comparison",
        "timestamp": timestamp,
        "total_time_s": total_time,
        "experiment_config": {
            "n_clients_list": exp_config.n_clients_list,
            "k_list": exp_config.k_list,
            "n_trials": exp_config.n_trials,
            "n_iterations_fairswarm": exp_config.n_iterations_fairswarm,
            "n_iterations_fairdpfl": exp_config.n_iterations_fairdpfl,
            "confidence": exp_config.confidence,
            "seed": exp_config.seed,
        },
        "results": all_results,
        "code_version": get_git_hash(),
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info(f"Results saved to: {output_file}")

    # Also save a compact table-ready version for the paper
    table_file = output_dir / f"sota_table_{timestamp}.json"
    table_data = _build_paper_table(all_results)
    with open(table_file, "w") as f:
        json.dump(table_data, f, indent=2, default=str)
    logger.info(f"Paper table saved to: {table_file}")


def _build_paper_table(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a compact table structure suitable for direct inclusion
    in a LaTeX table generator.

    Each row is: (config, method, DemDiv mean+/-CI, CD mean+/-CI, AUC mean+/-CI, Time).

    Args:
        all_results: Full experiment results.

    Returns:
        Dict with 'columns' and 'rows' keys.
    """
    columns = [
        "n",
        "k",
        "m",
        "method",
        "dem_div_mean",
        "dem_div_ci",
        "client_dissim_mean",
        "client_dissim_ci",
        "auc_proxy_mean",
        "auc_proxy_ci",
        "wall_time_mean",
        "p_value_dem_div",
    ]

    rows: List[Dict[str, Any]] = []

    for cfg_result in all_results:
        cfg = cfg_result["config"]
        methods = cfg_result["methods"]

        method_order = [
            "FairSwarm",
            "Random",
            "FedAvg",
            "Greedy",
            "FedFDP",
            "DivFL",
            "Oort",
            "Power-of-Choice",
            "SubTrunc",
        ]

        for method_name in method_order:
            if method_name not in methods:
                continue
            m = methods[method_name]

            # Extract p-value for DemDiv vs FairSwarm
            p_val_dd = None
            if m.get("vs_fairswarm") and "dem_div" in m["vs_fairswarm"]:
                p_val_dd = m["vs_fairswarm"]["dem_div"]["p_value"]

            rows.append(
                {
                    "n": cfg["n_clients"],
                    "k": cfg["k"],
                    "m": cfg["coalition_size"],
                    "method": method_name,
                    "dem_div_mean": round(m["dem_div"]["mean"], 4),
                    "dem_div_ci": f"[{m['dem_div']['ci_lower']:.4f}, {m['dem_div']['ci_upper']:.4f}]",
                    "client_dissim_mean": round(m["client_dissim"]["mean"], 4),
                    "client_dissim_ci": f"[{m['client_dissim']['ci_lower']:.4f}, {m['client_dissim']['ci_upper']:.4f}]",
                    "auc_proxy_mean": round(m["auc_proxy"]["mean"], 4),
                    "auc_proxy_ci": f"[{m['auc_proxy']['ci_lower']:.4f}, {m['auc_proxy']['ci_upper']:.4f}]",
                    "wall_time_mean": round(m["wall_time_s"]["mean"], 4),
                    "p_value_dem_div": round(p_val_dd, 6)
                    if p_val_dd is not None
                    else None,
                }
            )

    return {"columns": columns, "rows": rows}


if __name__ == "__main__":
    main()
