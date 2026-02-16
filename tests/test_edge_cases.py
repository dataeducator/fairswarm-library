"""
Edge Case Stress Tests for the FairSwarm Pipeline.

Injects missing values, outliers, duplicate records, swapped column types,
and empty datasets at every layer of the pipeline. Documents which steps
break, which silently produce wrong results, and which handle errors correctly.

Failures ranked by severity:
  SILENT_WRONG > CRASH > HANDLED

Author: Tenicka Norwood
"""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pytest

# ── imports under test ───────────────────────────────────────────────────
from fairswarm.core.client import Client, create_synthetic_clients
from fairswarm.core.config import FairSwarmConfig
from fairswarm.demographics.distribution import (
    DemographicDistribution,
    combine_distributions,
)
from fairswarm.demographics.divergence import (
    coalition_demographic_divergence,
    kl_divergence,
)
from fairswarm.fitness.fairness import DemographicFitness
from fairswarm.algorithms.fairswarm import FairSwarm

# ── helpers ──────────────────────────────────────────────────────────────

@dataclass
class EdgeCaseResult:
    """Captures the outcome of one edge-case probe."""
    name: str
    category: str          # NaN, Inf, Empty, Outlier, TypeSwap, Duplicate, Boundary
    component: str         # e.g. "DemographicDistribution", "kl_divergence"
    outcome: str           # HANDLED, CRASH, SILENT_WRONG
    detail: str            # what happened
    severity: int          # 3=SILENT_WRONG, 2=CRASH, 1=HANDLED

    def __repr__(self) -> str:
        tag = {3: "SILENT_WRONG", 2: "CRASH", 1: "HANDLED"}[self.severity]
        return f"[{tag}] {self.component}::{self.name} - {self.detail}"


RESULTS: List[EdgeCaseResult] = []


def record(name: str, category: str, component: str,
           outcome: str, detail: str, severity: int) -> None:
    r = EdgeCaseResult(name, category, component, outcome, detail, severity)
    RESULTS.append(r)


# =========================================================================
# 1.  DemographicDistribution edge cases
# =========================================================================

class TestDemographicDistributionEdgeCases:
    """Probe DemographicDistribution with pathological inputs."""

    # ── NaN injection ────────────────────────────────────────────────
    def test_nan_in_values(self):
        """NaN in probability vector."""
        try:
            d = DemographicDistribution(values=np.array([0.5, float("nan"), 0.5]))
            # If we get here without error, check if it's flagged
            if np.any(np.isnan(d.values)):
                record("nan_in_values", "NaN", "DemographicDistribution",
                       "SILENT_WRONG", "NaN accepted without error", 3)
                pytest.fail("NaN silently accepted in DemographicDistribution")
            else:
                record("nan_in_values", "NaN", "DemographicDistribution",
                       "HANDLED", "NaN was cleaned or rejected", 1)
        except (ValueError, TypeError) as e:
            record("nan_in_values", "NaN", "DemographicDistribution",
                   "HANDLED", f"Rejected: {e}", 1)

    def test_all_nan_values(self):
        """All-NaN probability vector."""
        try:
            d = DemographicDistribution(values=np.array([float("nan")] * 4))
            record("all_nan_values", "NaN", "DemographicDistribution",
                   "SILENT_WRONG", "All-NaN accepted", 3)
            pytest.fail("All-NaN silently accepted")
        except (ValueError, TypeError):
            record("all_nan_values", "NaN", "DemographicDistribution",
                   "HANDLED", "Rejected", 1)

    # ── Inf injection ────────────────────────────────────────────────
    def test_inf_in_values(self):
        """Inf in probability vector."""
        try:
            d = DemographicDistribution(values=np.array([0.5, float("inf"), 0.5]))
            record("inf_in_values", "Inf", "DemographicDistribution",
                   "SILENT_WRONG", "Inf accepted", 3)
            pytest.fail("Inf silently accepted")
        except (ValueError, TypeError, OverflowError):
            record("inf_in_values", "Inf", "DemographicDistribution",
                   "HANDLED", "Rejected", 1)

    def test_neg_inf_in_values(self):
        """Negative Inf in probability vector."""
        try:
            d = DemographicDistribution(values=np.array([0.5, float("-inf"), 0.5]))
            record("neg_inf_values", "Inf", "DemographicDistribution",
                   "SILENT_WRONG", "-Inf accepted", 3)
            pytest.fail("-Inf silently accepted")
        except (ValueError, TypeError):
            record("neg_inf_values", "Inf", "DemographicDistribution",
                   "HANDLED", "Rejected", 1)

    # ── Negative values ──────────────────────────────────────────────
    def test_negative_values_summing_to_one(self):
        """Negative probs that sum to 1 (e.g. [-0.5, 1.0, 0.5])."""
        try:
            d = DemographicDistribution(values=np.array([-0.5, 1.0, 0.5]))
            record("neg_sum_one", "Outlier", "DemographicDistribution",
                   "SILENT_WRONG", "Negative prob accepted because sum=1", 3)
            pytest.fail("Negative values silently accepted")
        except ValueError:
            record("neg_sum_one", "Outlier", "DemographicDistribution",
                   "HANDLED", "Rejected", 1)

    # ── Empty array ──────────────────────────────────────────────────
    def test_empty_values(self):
        """Zero-length probability vector."""
        try:
            d = DemographicDistribution(values=np.array([]))
            record("empty_values", "Empty", "DemographicDistribution",
                   "SILENT_WRONG", "Empty array accepted", 3)
            pytest.fail("Empty array silently accepted")
        except (ValueError, IndexError):
            record("empty_values", "Empty", "DemographicDistribution",
                   "HANDLED", "Rejected", 1)

    # ── Single element ───────────────────────────────────────────────
    def test_single_element(self):
        """Single-group distribution [1.0]."""
        try:
            d = DemographicDistribution(values=np.array([1.0]))
            assert d.values[0] == 1.0
            record("single_element", "Boundary", "DemographicDistribution",
                   "HANDLED", "Single-group distribution works", 1)
        except Exception as e:
            record("single_element", "Boundary", "DemographicDistribution",
                   "CRASH", f"Unexpected error: {e}", 2)

    # ── Very large array ─────────────────────────────────────────────
    def test_very_large_k(self):
        """k=10000 demographic groups."""
        try:
            vals = np.ones(10000) / 10000
            d = DemographicDistribution(values=vals)
            assert len(d.values) == 10000
            record("very_large_k", "Boundary", "DemographicDistribution",
                   "HANDLED", "10000-group distribution works", 1)
        except Exception as e:
            record("very_large_k", "Boundary", "DemographicDistribution",
                   "CRASH", f"Error: {e}", 2)

    # ── All zeros ────────────────────────────────────────────────────
    def test_all_zeros(self):
        """All-zero probability vector."""
        try:
            d = DemographicDistribution(values=np.array([0.0, 0.0, 0.0, 0.0]))
            record("all_zeros", "Empty", "DemographicDistribution",
                   "SILENT_WRONG", "All-zero accepted (sum != 1)", 3)
            pytest.fail("All-zero silently accepted")
        except ValueError:
            record("all_zeros", "Empty", "DemographicDistribution",
                   "HANDLED", "Rejected", 1)

    # ── Label mismatch ───────────────────────────────────────────────
    def test_label_length_mismatch(self):
        """Labels don't match values length."""
        try:
            d = DemographicDistribution(
                values=np.array([0.5, 0.5]),
                labels=("a", "b", "c"),
            )
            record("label_mismatch", "TypeSwap", "DemographicDistribution",
                   "SILENT_WRONG", "Label/value mismatch accepted", 3)
            pytest.fail("Label mismatch silently accepted")
        except ValueError:
            record("label_mismatch", "TypeSwap", "DemographicDistribution",
                   "HANDLED", "Rejected", 1)


# =========================================================================
# 2.  KL Divergence edge cases
# =========================================================================

class TestKLDivergenceEdgeCases:
    """Probe kl_divergence with pathological inputs."""

    def test_nan_in_p(self):
        """NaN in P distribution."""
        try:
            result = kl_divergence(
                np.array([0.5, float("nan"), 0.5]),
                np.array([0.33, 0.34, 0.33]),
            )
            if math.isnan(result) or math.isinf(result):
                record("nan_in_p", "NaN", "kl_divergence",
                       "SILENT_WRONG", f"Returned {result} (NaN/Inf)", 3)
                pytest.fail(f"kl_divergence returned {result}")
            else:
                record("nan_in_p", "NaN", "kl_divergence",
                       "SILENT_WRONG", f"Returned numeric {result} from NaN input", 3)
                pytest.fail(f"kl_divergence returned {result} from NaN input")
        except (ValueError, TypeError):
            record("nan_in_p", "NaN", "kl_divergence",
                   "HANDLED", "Rejected NaN input", 1)

    def test_zero_in_q(self):
        """Zero in Q (should be smoothed, not crash)."""
        try:
            result = kl_divergence(
                np.array([0.5, 0.3, 0.2]),
                np.array([0.5, 0.0, 0.5]),
            )
            if math.isnan(result) or math.isinf(result):
                record("zero_in_q", "Outlier", "kl_divergence",
                       "SILENT_WRONG", f"Returned {result}", 3)
                pytest.fail(f"Zero in Q produced {result}")
            else:
                record("zero_in_q", "Outlier", "kl_divergence",
                       "HANDLED", f"Smoothed, returned {result:.6f}", 1)
        except Exception as e:
            record("zero_in_q", "Outlier", "kl_divergence",
                   "CRASH", f"Error: {e}", 2)

    def test_empty_arrays(self):
        """Empty arrays as input."""
        try:
            result = kl_divergence(np.array([]), np.array([]))
            record("empty_arrays", "Empty", "kl_divergence",
                   "SILENT_WRONG", f"Returned {result} for empty arrays", 3)
            pytest.fail("Empty arrays accepted")
        except (ValueError, IndexError):
            record("empty_arrays", "Empty", "kl_divergence",
                   "HANDLED", "Rejected", 1)

    def test_length_mismatch(self):
        """P and Q have different lengths."""
        try:
            result = kl_divergence(
                np.array([0.5, 0.5]),
                np.array([0.33, 0.34, 0.33]),
            )
            record("length_mismatch", "TypeSwap", "kl_divergence",
                   "SILENT_WRONG", f"Different-length arrays accepted, got {result}", 3)
            pytest.fail("Length mismatch accepted")
        except ValueError:
            record("length_mismatch", "TypeSwap", "kl_divergence",
                   "HANDLED", "Rejected", 1)

    def test_identical_distributions(self):
        """KL(P||P) should be exactly 0."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        result = kl_divergence(p, p.copy())
        if abs(result) > 1e-6:
            record("identical", "Boundary", "kl_divergence",
                   "SILENT_WRONG", f"KL(P||P)={result}, expected 0", 3)
            pytest.fail(f"KL(P||P) = {result}")
        else:
            record("identical", "Boundary", "kl_divergence",
                   "HANDLED", f"KL(P||P)={result:.2e}, correct", 1)

    def test_very_small_probabilities(self):
        """Near-zero probabilities (underflow risk)."""
        try:
            p = np.array([1e-15, 1e-15, 1.0 - 2e-15])
            q = np.array([0.33, 0.34, 0.33])
            result = kl_divergence(p, q)
            if math.isnan(result) or math.isinf(result):
                record("tiny_probs", "Outlier", "kl_divergence",
                       "SILENT_WRONG", f"Returned {result}", 3)
                pytest.fail(f"Tiny probs produced {result}")
            else:
                record("tiny_probs", "Outlier", "kl_divergence",
                       "HANDLED", f"Returned {result:.6f}", 1)
        except Exception as e:
            record("tiny_probs", "Outlier", "kl_divergence",
                   "CRASH", f"Error: {e}", 2)


# =========================================================================
# 3.  Client creation edge cases
# =========================================================================

class TestClientEdgeCases:
    """Probe Client with pathological inputs."""

    def test_nan_demographics(self):
        """Client with NaN in demographics."""
        try:
            c = Client(
                id="test",
                demographics=np.array([0.5, float("nan"), 0.5]),
                dataset_size=100,
            )
            record("nan_client_demo", "NaN", "Client",
                   "SILENT_WRONG", "NaN demographics accepted", 3)
            pytest.fail("NaN demographics accepted in Client")
        except (ValueError, TypeError):
            record("nan_client_demo", "NaN", "Client",
                   "HANDLED", "Rejected", 1)

    def test_zero_dataset_size(self):
        """Client with zero dataset_size."""
        try:
            c = Client(
                id="test",
                demographics=np.array([0.5, 0.5]),
                dataset_size=0,
            )
            record("zero_dataset", "Boundary", "Client",
                   "SILENT_WRONG", "Zero dataset_size accepted", 3)
            pytest.fail("Zero dataset_size accepted")
        except ValueError:
            record("zero_dataset", "Boundary", "Client",
                   "HANDLED", "Rejected", 1)

    def test_negative_dataset_size(self):
        """Client with negative dataset_size."""
        try:
            c = Client(
                id="test",
                demographics=np.array([0.5, 0.5]),
                dataset_size=-100,
            )
            record("neg_dataset", "Outlier", "Client",
                   "SILENT_WRONG", "Negative dataset_size accepted", 3)
            pytest.fail("Negative dataset_size accepted")
        except ValueError:
            record("neg_dataset", "Outlier", "Client",
                   "HANDLED", "Rejected", 1)

    def test_negative_communication_cost(self):
        """Client with communication_cost < 0."""
        try:
            c = Client(
                id="test",
                demographics=np.array([0.5, 0.5]),
                dataset_size=100,
                communication_cost=-0.5,
            )
            record("neg_comm_cost", "Outlier", "Client",
                   "SILENT_WRONG", "Negative comm cost accepted", 3)
            pytest.fail("Negative communication_cost accepted")
        except ValueError:
            record("neg_comm_cost", "Outlier", "Client",
                   "HANDLED", "Rejected", 1)

    def test_communication_cost_above_one(self):
        """Client with communication_cost > 1."""
        try:
            c = Client(
                id="test",
                demographics=np.array([0.5, 0.5]),
                dataset_size=100,
                communication_cost=1.5,
            )
            record("high_comm_cost", "Outlier", "Client",
                   "SILENT_WRONG", "Comm cost > 1 accepted", 3)
            pytest.fail("Comm cost > 1 accepted")
        except ValueError:
            record("high_comm_cost", "Outlier", "Client",
                   "HANDLED", "Rejected", 1)

    def test_duplicate_client_ids(self):
        """Multiple clients with same ID (no dedup)."""
        c1 = Client(id="dup", demographics=np.array([0.5, 0.5]), dataset_size=100)
        c2 = Client(id="dup", demographics=np.array([0.3, 0.7]), dataset_size=200)
        clients = [c1, c2]
        # This isn't validated at Client level; check if FairSwarm catches it
        record("dup_ids", "Duplicate", "Client",
               "HANDLED", "Duplicate IDs allowed (no id uniqueness constraint)", 1)


# =========================================================================
# 4.  Coalition divergence edge cases
# =========================================================================

class TestCoalitionDivergenceEdgeCases:
    """Probe coalition_demographic_divergence with edge cases."""

    def _make_clients(self, n: int = 5, k: int = 4) -> list:
        return create_synthetic_clients(n_clients=n, n_demographic_groups=k, seed=42)

    def test_empty_coalition(self):
        """Empty coalition indices."""
        clients = self._make_clients()
        target = DemographicDistribution(values=np.array([0.25, 0.25, 0.25, 0.25]))
        try:
            result = coalition_demographic_divergence(
                [c.demographics for c in clients],
                [],
                target,
            )
            record("empty_coalition", "Empty", "coalition_divergence",
                   "SILENT_WRONG", f"Empty coalition returned {result}", 3)
            pytest.fail("Empty coalition should raise error")
        except (ValueError, IndexError):
            record("empty_coalition", "Empty", "coalition_divergence",
                   "HANDLED", "Rejected empty coalition", 1)

    def test_out_of_bounds_index(self):
        """Coalition index exceeding client count."""
        clients = self._make_clients(5)
        target = DemographicDistribution(values=np.array([0.25, 0.25, 0.25, 0.25]))
        try:
            result = coalition_demographic_divergence(
                [c.demographics for c in clients],
                [0, 1, 99],
                target,
            )
            record("oob_index", "Boundary", "coalition_divergence",
                   "SILENT_WRONG", f"OOB index accepted, got {result}", 3)
            pytest.fail("Out-of-bounds index accepted")
        except (ValueError, IndexError):
            record("oob_index", "Boundary", "coalition_divergence",
                   "HANDLED", "Rejected OOB", 1)

    def test_negative_index(self):
        """Negative coalition index (Python wraps to end)."""
        clients = self._make_clients(5)
        target = DemographicDistribution(values=np.array([0.25, 0.25, 0.25, 0.25]))
        try:
            result = coalition_demographic_divergence(
                [c.demographics for c in clients],
                [0, -1],
                target,
            )
            # Python indexing wraps -1 to last element — may be "correct" but unexpected
            record("neg_index", "Outlier", "coalition_divergence",
                   "SILENT_WRONG", f"Negative index silently wrapped, got {result:.6f}", 3)
        except (ValueError, IndexError):
            record("neg_index", "Outlier", "coalition_divergence",
                   "HANDLED", "Rejected negative index", 1)

    def test_duplicate_indices(self):
        """Same client selected twice."""
        clients = self._make_clients(5)
        target = DemographicDistribution(values=np.array([0.25, 0.25, 0.25, 0.25]))
        result_dup = coalition_demographic_divergence(
            [c.demographics for c in clients],
            [0, 0, 0],
            target,
        )
        result_single = coalition_demographic_divergence(
            [c.demographics for c in clients],
            [0],
            target,
        )
        if abs(result_dup - result_single) < 1e-10:
            record("dup_indices", "Duplicate", "coalition_divergence",
                   "HANDLED", "Duplicates produce same result as single (correct for mean)", 1)
        else:
            record("dup_indices", "Duplicate", "coalition_divergence",
                   "SILENT_WRONG",
                   f"Dup={result_dup:.6f} vs single={result_single:.6f} — "
                   "duplicates change the result", 3)


# =========================================================================
# 5.  FairSwarm algorithm edge cases
# =========================================================================

class TestFairSwarmEdgeCases:
    """Probe FairSwarm optimizer with edge inputs."""

    def _make_target(self, k: int = 4) -> DemographicDistribution:
        return DemographicDistribution(
            values=np.ones(k) / k,
            labels=tuple(f"group_{i}" for i in range(k)),
        )

    def _make_fitness(self, k: int = 4) -> DemographicFitness:
        return DemographicFitness(target_distribution=self._make_target(k))

    # ── Empty client list ────────────────────────────────────────────
    def test_empty_clients(self):
        """No clients at all."""
        try:
            fs = FairSwarm(
                clients=[],
                coalition_size=3,
                config=FairSwarmConfig(),
                target_distribution=self._make_target(),
                seed=42,
            )
            record("empty_clients", "Empty", "FairSwarm",
                   "SILENT_WRONG", "Empty client list accepted", 3)
            pytest.fail("Empty clients should error")
        except (ValueError, IndexError, RuntimeError) as e:
            record("empty_clients", "Empty", "FairSwarm",
                   "HANDLED", f"Rejected: {type(e).__name__}", 1)

    # ── Coalition larger than clients ────────────────────────────────
    def test_coalition_larger_than_n(self):
        """coalition_size > len(clients)."""
        clients = create_synthetic_clients(3, 4, seed=42)
        try:
            fs = FairSwarm(
                clients=clients,
                coalition_size=10,
                config=FairSwarmConfig(),
                target_distribution=self._make_target(),
                seed=42,
            )
            record("coalition_gt_n", "Boundary", "FairSwarm",
                   "SILENT_WRONG", "coalition_size > n_clients accepted", 3)
            pytest.fail("coalition > n should error")
        except (ValueError, RuntimeError):
            record("coalition_gt_n", "Boundary", "FairSwarm",
                   "HANDLED", "Rejected", 1)

    # ── Coalition = all clients ──────────────────────────────────────
    def test_coalition_equals_n(self):
        """coalition_size == len(clients) — trivial solution."""
        clients = create_synthetic_clients(5, 4, seed=42)
        try:
            fs = FairSwarm(
                clients=clients,
                coalition_size=5,
                config=FairSwarmConfig(max_iterations=10),
                target_distribution=self._make_target(),
                seed=42,
            )
            result = fs.optimize(
                fitness_fn=self._make_fitness(),
                n_iterations=10,
                verbose=False,
            )
            assert len(result.coalition) == 5
            record("coalition_eq_n", "Boundary", "FairSwarm",
                   "HANDLED", "Trivial selection works", 1)
        except Exception as e:
            record("coalition_eq_n", "Boundary", "FairSwarm",
                   "CRASH", f"Error: {e}", 2)

    # ── Single client ────────────────────────────────────────────────
    def test_single_client(self):
        """Only one client, coalition_size=1."""
        clients = create_synthetic_clients(1, 4, seed=42)
        try:
            fs = FairSwarm(
                clients=clients,
                coalition_size=1,
                config=FairSwarmConfig(max_iterations=5),
                target_distribution=self._make_target(),
                seed=42,
            )
            result = fs.optimize(
                fitness_fn=self._make_fitness(),
                n_iterations=5,
                verbose=False,
            )
            assert len(result.coalition) == 1
            record("single_client", "Boundary", "FairSwarm",
                   "HANDLED", "Single client works", 1)
        except Exception as e:
            record("single_client", "Boundary", "FairSwarm",
                   "CRASH", f"Error: {e}", 2)

    # ── Mismatched demographic dimensions ────────────────────────────
    def test_mismatched_k(self):
        """Clients with different numbers of demographic groups."""
        c1 = Client(id="c1", demographics=np.array([0.5, 0.5]), dataset_size=100)
        c2 = Client(id="c2", demographics=np.array([0.33, 0.34, 0.33]), dataset_size=100)
        try:
            fs = FairSwarm(
                clients=[c1, c2],
                coalition_size=1,
                config=FairSwarmConfig(),
                target_distribution=self._make_target(2),
                seed=42,
            )
            result = fs.optimize(
                fitness_fn=DemographicFitness(
                    target_distribution=self._make_target(2)
                ),
                n_iterations=5,
                verbose=False,
            )
            record("mismatched_k", "TypeSwap", "FairSwarm",
                   "SILENT_WRONG",
                   "Clients with different k accepted — dimension mismatch undetected", 3)
            pytest.fail("Mismatched demographics should error")
        except (ValueError, IndexError, RuntimeError) as e:
            record("mismatched_k", "TypeSwap", "FairSwarm",
                   "HANDLED", f"Rejected: {e}", 1)

    # ── Zero iterations ──────────────────────────────────────────────
    def test_zero_iterations(self):
        """n_iterations=0."""
        clients = create_synthetic_clients(10, 4, seed=42)
        try:
            fs = FairSwarm(
                clients=clients,
                coalition_size=3,
                config=FairSwarmConfig(max_iterations=0),
                target_distribution=self._make_target(),
                seed=42,
            )
            result = fs.optimize(
                fitness_fn=self._make_fitness(),
                n_iterations=0,
                verbose=False,
            )
            # Should still return initial best
            if result.coalition is not None and len(result.coalition) == 3:
                record("zero_iter", "Boundary", "FairSwarm",
                       "HANDLED", "Zero iters returns initial best", 1)
            else:
                record("zero_iter", "Boundary", "FairSwarm",
                       "SILENT_WRONG", f"Zero iters returned {result.coalition}", 3)
                pytest.fail("Zero iterations produced bad result")
        except Exception as e:
            record("zero_iter", "Boundary", "FairSwarm",
                   "CRASH", f"Error: {e}", 2)

    # ── Extreme config values ────────────────────────────────────────
    def test_extreme_inertia(self):
        """Inertia weight at boundary (0.999)."""
        clients = create_synthetic_clients(10, 4, seed=42)
        try:
            config = FairSwarmConfig(
                inertia_weight=0.999,
                cognitive_coefficient=0.001,
                social_coefficient=0.001,
                max_iterations=10,
            )
            fs = FairSwarm(
                clients=clients,
                coalition_size=3,
                config=config,
                target_distribution=self._make_target(),
                seed=42,
            )
            result = fs.optimize(
                fitness_fn=self._make_fitness(),
                n_iterations=10,
                verbose=False,
            )
            assert result.coalition is not None
            record("extreme_inertia", "Boundary", "FairSwarm",
                   "HANDLED", "Extreme inertia runs without crash", 1)
        except Exception as e:
            record("extreme_inertia", "Boundary", "FairSwarm",
                   "CRASH", f"Error: {e}", 2)


# =========================================================================
# 6.  combine_distributions edge cases
# =========================================================================

class TestCombineDistributionsEdgeCases:
    """Probe combine_distributions with edge inputs."""

    def test_empty_list(self):
        """Combining zero distributions."""
        try:
            result = combine_distributions([])
            record("combine_empty", "Empty", "combine_distributions",
                   "SILENT_WRONG", "Empty list returned a result", 3)
            pytest.fail("Combining empty list should error")
        except (ValueError, IndexError):
            record("combine_empty", "Empty", "combine_distributions",
                   "HANDLED", "Rejected", 1)

    def test_single_distribution(self):
        """Combining one distribution returns itself."""
        d = DemographicDistribution(values=np.array([0.3, 0.7]))
        result = combine_distributions([d])
        assert np.allclose(result.values, d.values)
        record("combine_single", "Boundary", "combine_distributions",
               "HANDLED", "Single distribution returned unchanged", 1)

    def test_mismatched_group_counts(self):
        """Distributions with different number of groups."""
        d1 = DemographicDistribution(values=np.array([0.5, 0.5]))
        d2 = DemographicDistribution(values=np.array([0.33, 0.34, 0.33]))
        try:
            result = combine_distributions([d1, d2])
            record("combine_mismatch", "TypeSwap", "combine_distributions",
                   "SILENT_WRONG", "Mismatched groups combined", 3)
            pytest.fail("Mismatched groups should error")
        except ValueError:
            record("combine_mismatch", "TypeSwap", "combine_distributions",
                   "HANDLED", "Rejected mismatched groups", 1)

    def test_nan_weights(self):
        """NaN in weight vector."""
        d1 = DemographicDistribution(values=np.array([0.5, 0.5]))
        d2 = DemographicDistribution(values=np.array([0.3, 0.7]))
        try:
            result = combine_distributions(
                [d1, d2], weights=np.array([float("nan"), 0.5])
            )
            if np.any(np.isnan(result.values)):
                record("combine_nan_wt", "NaN", "combine_distributions",
                       "SILENT_WRONG", "NaN weights produced NaN output", 3)
                pytest.fail("NaN weights produced NaN")
            else:
                record("combine_nan_wt", "NaN", "combine_distributions",
                       "SILENT_WRONG",
                       f"NaN weight silently accepted, result={result.values}", 3)
                pytest.fail("NaN weight silently accepted")
        except (ValueError, TypeError):
            record("combine_nan_wt", "NaN", "combine_distributions",
                   "HANDLED", "Rejected NaN weights", 1)


# =========================================================================
# 7.  FairSwarmConfig edge cases
# =========================================================================

class TestConfigEdgeCases:
    """Probe FairSwarmConfig validation."""

    def test_negative_inertia(self):
        """Inertia < 0."""
        try:
            c = FairSwarmConfig(inertia=-0.5)
            record("neg_inertia", "Outlier", "FairSwarmConfig",
                   "SILENT_WRONG", "Negative inertia accepted", 3)
            pytest.fail("Negative inertia should error")
        except ValueError:
            record("neg_inertia", "Outlier", "FairSwarmConfig",
                   "HANDLED", "Rejected", 1)

    def test_inertia_above_one(self):
        """Inertia > 1 (violates convergence)."""
        try:
            c = FairSwarmConfig(inertia=1.5)
            record("high_inertia", "Outlier", "FairSwarmConfig",
                   "SILENT_WRONG", "Inertia > 1 accepted (violates theorem)", 3)
            pytest.fail("Inertia > 1 should error")
        except ValueError:
            record("high_inertia", "Outlier", "FairSwarmConfig",
                   "HANDLED", "Rejected", 1)

    def test_zero_swarm_size(self):
        """Swarm with 0 particles."""
        try:
            c = FairSwarmConfig(swarm_size=0)
            record("zero_swarm", "Boundary", "FairSwarmConfig",
                   "SILENT_WRONG", "Zero swarm_size accepted", 3)
            pytest.fail("Zero swarm should error")
        except ValueError:
            record("zero_swarm", "Boundary", "FairSwarmConfig",
                   "HANDLED", "Rejected", 1)

    def test_negative_epsilon_fair(self):
        """epsilon_fair < 0."""
        try:
            c = FairSwarmConfig(epsilon_fair=-0.1)
            record("neg_eps_fair", "Outlier", "FairSwarmConfig",
                   "SILENT_WRONG", "Negative epsilon_fair accepted", 3)
            pytest.fail("Negative epsilon_fair should error")
        except ValueError:
            record("neg_eps_fair", "Outlier", "FairSwarmConfig",
                   "HANDLED", "Rejected", 1)


# =========================================================================
# 8.  DemographicFitness edge cases
# =========================================================================

class TestDemographicFitnessEdgeCases:
    """Probe DemographicFitness evaluation."""

    def _make_clients(self, n=10, k=4):
        return create_synthetic_clients(n, k, seed=42)

    def test_empty_coalition_fitness(self):
        """Evaluate fitness on empty coalition."""
        clients = self._make_clients()
        target = DemographicDistribution(values=np.array([0.25, 0.25, 0.25, 0.25]))
        fitness = DemographicFitness(target_distribution=target)
        try:
            result = fitness.evaluate([], clients)
            if result.value == float("-inf") or result.value < -1e10:
                record("empty_fitness", "Empty", "DemographicFitness",
                       "HANDLED", f"Empty coalition returns -inf fitness", 1)
            else:
                record("empty_fitness", "Empty", "DemographicFitness",
                       "SILENT_WRONG", f"Empty coalition fitness = {result.value}", 3)
                pytest.fail("Empty coalition should return -inf or error")
        except (ValueError, IndexError):
            record("empty_fitness", "Empty", "DemographicFitness",
                   "HANDLED", "Rejected empty coalition", 1)

    def test_single_client_fitness(self):
        """Fitness evaluation on single-client coalition."""
        clients = self._make_clients()
        target = DemographicDistribution(values=np.array([0.25, 0.25, 0.25, 0.25]))
        fitness = DemographicFitness(target_distribution=target)
        try:
            result = fitness.evaluate([0], clients)
            assert not math.isnan(result.value)
            assert not math.isinf(result.value)
            record("single_fitness", "Boundary", "DemographicFitness",
                   "HANDLED", f"Single client fitness = {result.value:.4f}", 1)
        except Exception as e:
            record("single_fitness", "Boundary", "DemographicFitness",
                   "CRASH", f"Error: {e}", 2)


# =========================================================================
# Summary fixture — prints ranked results after all tests
# =========================================================================

@pytest.fixture(autouse=True, scope="session")
def print_summary(request):
    """Print ranked summary after all edge case tests complete."""
    yield
    if RESULTS:
        print("\n" + "=" * 72)
        print("EDGE CASE STRESS TEST SUMMARY")
        print("=" * 72)
        # Sort by severity descending
        sorted_results = sorted(RESULTS, key=lambda r: (-r.severity, r.component))
        for r in sorted_results:
            print(r)
        print("=" * 72)
        silent = sum(1 for r in RESULTS if r.severity == 3)
        crash = sum(1 for r in RESULTS if r.severity == 2)
        handled = sum(1 for r in RESULTS if r.severity == 1)
        print(f"Total: {len(RESULTS)} tests | "
              f"SILENT_WRONG: {silent} | CRASH: {crash} | HANDLED: {handled}")
        print("=" * 72)
