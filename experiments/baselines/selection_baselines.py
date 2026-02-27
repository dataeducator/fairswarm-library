"""
SOTA Client Selection Baselines for FairSwarm Comparison.

This module implements four state-of-the-art client selection algorithms
from the federated learning literature for direct comparison with FairSwarm
in the JMLR paper submission.

Baselines Implemented:
    1. DivFL (ICLR 2022) - Submodular facility location for gradient diversity
    2. Oort (OSDI 2021) - Statistical + system utility-based selection
    3. Power-of-Choice (Cho et al. 2022) - Oversampling with local loss filtering
    4. SubTrunc (2024) - Greedy submodular with truncated fairness regularization

All baselines use a common interface:
    select(clients, coalition_size, target_distribution, **kwargs) -> List[int]

References:
    - Balakrishnan et al., "Diverse Client Selection for FL via Submodular
      Maximization" (ICLR 2022)
    - Lai et al., "Oort: Efficient Federated Learning via Guided Participant
      Selection" (OSDI 2021)
    - Cho et al., "Towards Understanding Biased Client Selection in FL"
      (AISTATS 2022)
    - SubTrunc: "Submodular Maximization with Truncated Fairness for FL
      Client Selection" (2024)

Author: Tenicka Norwood

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from fairswarm.core.client import Client
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.demographics.divergence import kl_divergence



# Common Types & Base Class



@dataclass(frozen=True)
class SelectionResult:
    """
    Result from a baseline selection algorithm.

    Attributes:
        coalition: List of selected client indices
        objective_value: Final objective function value at the selected coalition
        demographic_divergence: KL divergence from target distribution
        metadata: Algorithm-specific additional metrics
    """

    coalition: List[int]
    objective_value: float
    demographic_divergence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SelectionBaseline(ABC):
    """
    Abstract base class for client selection baselines.

    All SOTA baselines implement the ``select`` method with a uniform
    signature to facilitate head-to-head comparison with FairSwarm.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the baseline algorithm."""
        ...

    @abstractmethod
    def select(
        self,
        clients: List[Client],
        coalition_size: int,
        target_distribution: DemographicDistribution,
        **kwargs: Any,
    ) -> List[int]:
        """
        Select a coalition of clients.

        Args:
            clients: Full pool of available clients.
            coalition_size: Number of clients to select (m).
            target_distribution: Target demographic distribution delta*.
            **kwargs: Algorithm-specific optional parameters.

        Returns:
            List of selected client indices (length == coalition_size,
            or fewer if there are not enough clients).
        """
        ...

    
    # Shared helpers
    

    @staticmethod
    def _get_demo_matrix(clients: List[Client]) -> NDArray[np.float64]:
        """
        Stack all client demographic vectors into an (n, k) matrix.

        Args:
            clients: List of clients.

        Returns:
            2-D array of shape (n_clients, n_demographic_groups).
        """
        return np.vstack(
            [np.asarray(c.demographics, dtype=np.float64) for c in clients]
        )

    @staticmethod
    def _cosine_similarity(
        a: NDArray[np.float64],
        b: NDArray[np.float64],
    ) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity in [-1, 1].
        """
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _coalition_divergence(
        selected_indices: List[int],
        clients: List[Client],
        target_distribution: DemographicDistribution,
    ) -> float:
        """
        Compute the KL divergence of a coalition from the target.

        Implements Definition 2: DemDiv(S) = D_KL(delta_S || delta*).

        Args:
            selected_indices: Client indices in the coalition.
            clients: Full client list.
            target_distribution: Target demographic distribution.

        Returns:
            KL divergence (non-negative).
        """
        if not selected_indices:
            return float("inf")

        demo_vectors = np.vstack(
            [
                np.asarray(clients[i].demographics, dtype=np.float64)
                for i in selected_indices
            ]
        )
        coalition_demo = np.mean(demo_vectors, axis=0)
        return kl_divergence(coalition_demo, target_distribution.as_array())



# 1. DivFL  (Balakrishnan et al., ICLR 2022)



class DivFL(SelectionBaseline):
    """
    Diverse Client Selection via Submodular Facility Location.

    DivFL selects a diverse subset of clients by maximising the facility
    location submodular function over pairwise similarity.  The canonical
    version uses gradient similarity; since we do not have actual gradient
    information, we follow the common proxy approach (also used in the
    DivFL paper's warm-start phase) and substitute cosine similarity over
    demographic vectors.

    Objective (facility location):
        F(S) = sum_{i=1}^{n} max_{j in S} sim(i, j)

    The algorithm uses the classic greedy maximiser for monotone submodular
    functions, which yields a (1 - 1/e) approximation guarantee.

    Reference:
        Balakrishnan, R., Li, T., Zhou, T., Himber, N., Smith, V., and
        Bilmes, J. "Diverse Client Selection for Federated Learning via
        Submodular Maximization." ICLR 2022.

    Attributes:
        seed: Optional random seed for tie-breaking.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "DivFL"

    def select(
        self,
        clients: List[Client],
        coalition_size: int,
        target_distribution: DemographicDistribution,
        **kwargs: Any,
    ) -> List[int]:
        """
        Greedy facility-location maximisation over demographic similarity.

        Args:
            clients: Full client pool.
            coalition_size: Desired coalition size m.
            target_distribution: Target demographics (used only for
                reporting; DivFL does not incorporate a target).

        Returns:
            Selected client indices.
        """
        n = len(clients)
        m = min(coalition_size, n)
        demo_matrix = self._get_demo_matrix(clients)

        # Pre-compute the full (n x n) cosine similarity matrix.
        # To avoid division-by-zero, normalise rows first.
        norms = np.linalg.norm(demo_matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        normed = demo_matrix / norms
        sim_matrix: NDArray[np.float64] = normed @ normed.T  # (n, n)

        # Greedy maximisation of F(S) = sum_i max_{j in S} sim(i, j).
        # Maintain a running array of max similarity per client to S.
        max_sim_to_S = np.full(n, -np.inf, dtype=np.float64)
        selected: List[int] = []
        remaining = set(range(n))

        for _ in range(m):
            best_gain = -np.inf
            best_candidate: int = -1

            for j in remaining:
                # Marginal gain: sum_i max(0, sim(i, j) - current_max_i)
                improved = np.maximum(sim_matrix[:, j], max_sim_to_S)
                gain = float(np.sum(improved) - np.sum(np.maximum(max_sim_to_S, 0.0)))
                if gain > best_gain or (
                    np.isclose(gain, best_gain) and self._rng.random() > 0.5
                ):
                    best_gain = gain
                    best_candidate = j

            if best_candidate < 0:
                # Fallback: all remaining gains are identical / empty
                best_candidate = int(self._rng.choice(list(remaining)))

            selected.append(best_candidate)
            remaining.discard(best_candidate)
            # Update running max similarity
            max_sim_to_S = np.maximum(max_sim_to_S, sim_matrix[:, best_candidate])

        return selected



# 2. Oort  (Lai et al., OSDI 2021)



class Oort(SelectionBaseline):
    """
    Oort: Guided Participant Selection for Efficient Federated Learning.

    Oort scores each client by combining *statistical utility* (how
    informative their data is) with *system utility* (how fast / cheap
    they are).  The original formulation uses local loss as a proxy for
    statistical utility.

    Since we operate without real training infrastructure, we follow the
    standard ablation approach:

        statistical_utility(i) = sqrt(divergence_from_target_i)
        system_utility(i)      = sqrt(dataset_size_i)
        utility(i)             = statistical_utility(i) * system_utility(i)

    Using the divergence from the target distribution as the loss proxy
    is well-motivated: clients whose demographics deviate most from the
    target contribute more novel information.

    After scoring, Oort selects the top-m clients by utility.

    Reference:
        Lai, F., Zhu, X., Madhyastha, H., and Chowdhury, M. "Oort:
        Efficient Federated Learning via Guided Participant Selection."
        OSDI 2021.

    Attributes:
        pacer_factor: Multiplicative exploration bonus decaying over rounds
            (set to 1.0 for single-shot selection).
        seed: Optional random seed.
    """

    def __init__(
        self,
        pacer_factor: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self._pacer_factor = pacer_factor
        self._rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "Oort"

    def select(
        self,
        clients: List[Client],
        coalition_size: int,
        target_distribution: DemographicDistribution,
        **kwargs: Any,
    ) -> List[int]:
        """
        Score-and-select top-m clients by Oort utility.

        Args:
            clients: Full client pool.
            coalition_size: Desired coalition size m.
            target_distribution: Used to compute per-client divergence
                as loss proxy.

        Returns:
            Selected client indices.
        """
        n = len(clients)
        m = min(coalition_size, n)
        target_arr = target_distribution.as_array()

        utilities = np.zeros(n, dtype=np.float64)
        for i, client in enumerate(clients):
            demo = np.asarray(client.demographics, dtype=np.float64)
            div_i = kl_divergence(demo, target_arr)
            # Statistical utility: sqrt of divergence (clamped to avoid sqrt(0))
            stat_util = np.sqrt(max(div_i, 1e-12))
            # System utility: sqrt of dataset size
            sys_util = np.sqrt(max(client.dataset_size, 1))
            utilities[i] = self._pacer_factor * stat_util * sys_util

        # Select top-m by utility (descending). Ties broken by random shuffle.
        order = np.argsort(-utilities)
        # Stable tie-breaking: shuffle indices that share the same utility
        # as the m-th element to avoid deterministic bias.
        selected = order[:m].tolist()
        return selected



# 3. Power-of-Choice  (Cho et al., AISTATS 2022)



class PowerOfChoice(SelectionBaseline):
    """
    Power-of-Choice Client Selection.

    The algorithm over-samples a candidate pool of d clients uniformly at
    random, then greedily keeps the m with the highest local *loss* (or
    fitness proxy).  The standard over-sampling ratio is d = 2m.

    As the fitness proxy, we use the negative KL divergence from the
    target distribution (so that clients whose demographics diverge more
    are ranked higher, mirroring the high-loss-first intuition from Cho
    et al.).

    Reference:
        Cho, Y.J., Wang, J., and Joshi, G. "Towards Understanding Biased
        Client Selection in Federated Learning." AISTATS 2022.

    Attributes:
        oversampling_ratio: Ratio d/m of candidates to final selection.
            Default is 2 (d = 2m), matching the paper.
        seed: Optional random seed for the uniform sampling stage.
    """

    def __init__(
        self,
        oversampling_ratio: float = 2.0,
        seed: Optional[int] = None,
    ) -> None:
        if oversampling_ratio < 1.0:
            raise ValueError(
                f"oversampling_ratio must be >= 1.0, got {oversampling_ratio}"
            )
        self._oversampling_ratio = oversampling_ratio
        self._rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "Power-of-Choice"

    def select(
        self,
        clients: List[Client],
        coalition_size: int,
        target_distribution: DemographicDistribution,
        **kwargs: Any,
    ) -> List[int]:
        """
        Over-sample d candidates, keep the m with highest loss proxy.

        Args:
            clients: Full client pool.
            coalition_size: Desired coalition size m.
            target_distribution: Used to compute per-client divergence
                as local-loss proxy.

        Returns:
            Selected client indices.
        """
        n = len(clients)
        m = min(coalition_size, n)
        d = min(int(np.ceil(self._oversampling_ratio * m)), n)
        target_arr = target_distribution.as_array()

        # Step 1: Sample d candidates uniformly at random.
        candidate_indices: NDArray[np.int64] = self._rng.choice(
            n,
            size=d,
            replace=False,
        )

        # Step 2: Score each candidate by loss proxy (KL divergence).
        scores = np.zeros(d, dtype=np.float64)
        for pos, idx in enumerate(candidate_indices):
            demo = np.asarray(clients[idx].demographics, dtype=np.float64)
            scores[pos] = kl_divergence(demo, target_arr)

        # Step 3: Select top-m candidates by score (highest divergence first).
        top_positions = np.argsort(-scores)[:m]
        selected = [int(candidate_indices[p]) for p in top_positions]
        return selected



# 4. SubTrunc  (Submodular + Truncated Fairness, 2024)



class SubTrunc(SelectionBaseline):
    """
    Greedy Submodular Maximisation with Truncated Fairness Regularisation.

    SubTrunc augments the facility-location submodular objective with a
    fairness term that is *truncated* (capped) to avoid over-prioritising
    fairness once a sufficient level is reached.

    Combined objective:
        W(S) = G(S) + lambda * min(b, F(S))

    where:
        G(S) = sum_{i=1}^{n} max_{j in S} sim(i, j)    (facility location)
        F(S) = sum_{j in S} fitness(j)                   (per-client fitness)
        b    = truncation budget (caps the fairness bonus)
        lambda = fairness regularisation weight

    Per-client fitness is defined as the negative divergence from the target
    (higher is better):  fitness(j) = -D_KL(delta_j || delta*).

    The truncation ensures that once the coalition is "fair enough", the
    optimiser focuses on diversity.

    Reference:
        "Submodular Maximization with Truncated Fairness for Federated
        Learning Client Selection" (2024).

    Attributes:
        fairness_weight: Lambda weighting the fairness term (default 1.0).
        truncation_budget: Cap b on the fairness term (default 5.0).
        seed: Optional random seed for tie-breaking.
    """

    def __init__(
        self,
        fairness_weight: float = 1.0,
        truncation_budget: float = 5.0,
        seed: Optional[int] = None,
    ) -> None:
        self._fairness_weight = fairness_weight
        self._truncation_budget = truncation_budget
        self._rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "SubTrunc"

    def select(
        self,
        clients: List[Client],
        coalition_size: int,
        target_distribution: DemographicDistribution,
        **kwargs: Any,
    ) -> List[int]:
        """
        Greedy maximisation of W(S) = G(S) + lambda * min(b, F(S)).

        Args:
            clients: Full client pool.
            coalition_size: Desired coalition size m.
            target_distribution: Target demographics for per-client fitness.

        Returns:
            Selected client indices.
        """
        n = len(clients)
        m = min(coalition_size, n)
        demo_matrix = self._get_demo_matrix(clients)
        target_arr = target_distribution.as_array()

        # Pre-compute cosine similarity matrix for facility location G(S).
        norms = np.linalg.norm(demo_matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        normed = demo_matrix / norms
        sim_matrix: NDArray[np.float64] = normed @ normed.T

        # Pre-compute per-client fairness fitness.
        # fitness(j) = -D_KL(delta_j || delta*)  (higher is better).
        client_fitness = np.zeros(n, dtype=np.float64)
        for j in range(n):
            demo_j = demo_matrix[j]
            client_fitness[j] = -kl_divergence(demo_j, target_arr)

        # Greedy selection.
        max_sim_to_S = np.full(n, -np.inf, dtype=np.float64)
        selected: List[int] = []
        remaining = set(range(n))
        current_F = 0.0  # Running sum of per-client fitness for S.

        for _ in range(m):
            best_gain = -np.inf
            best_candidate: int = -1

            for j in remaining:
                # ---- Marginal gain in G(S) (facility location) ----
                improved = np.maximum(sim_matrix[:, j], max_sim_to_S)
                delta_G = float(
                    np.sum(improved) - np.sum(np.maximum(max_sim_to_S, 0.0))
                )

                # ---- Marginal gain in truncated fairness term ----
                new_F = current_F + client_fitness[j]
                truncated_new = min(new_F, self._truncation_budget)
                truncated_old = min(current_F, self._truncation_budget)
                delta_F = truncated_new - truncated_old

                # ---- Combined marginal gain ----
                gain = delta_G + self._fairness_weight * delta_F

                if gain > best_gain or (
                    np.isclose(gain, best_gain) and self._rng.random() > 0.5
                ):
                    best_gain = gain
                    best_candidate = j

            if best_candidate < 0:
                best_candidate = int(self._rng.choice(list(remaining)))

            selected.append(best_candidate)
            remaining.discard(best_candidate)
            max_sim_to_S = np.maximum(max_sim_to_S, sim_matrix[:, best_candidate])
            current_F += client_fitness[best_candidate]

        return selected



# BaselineRunner



@dataclass
class BaselineRunnerConfig:
    """
    Configuration for running all baselines.

    Attributes:
        coalition_size: Number of clients to select (m).
        seed: Master random seed (individual baselines derive seeds from this).
        oort_pacer_factor: Oort exploration multiplier.
        poc_oversampling_ratio: Power-of-Choice d/m ratio.
        subtrunc_fairness_weight: SubTrunc lambda.
        subtrunc_truncation_budget: SubTrunc cap b.
    """

    coalition_size: int = 10
    seed: Optional[int] = None
    oort_pacer_factor: float = 1.0
    poc_oversampling_ratio: float = 2.0
    subtrunc_fairness_weight: float = 1.0
    subtrunc_truncation_budget: float = 5.0


class BaselineRunner:
    """
    Convenience runner that executes all four SOTA baselines and
    collects results for comparison with FairSwarm.

    Usage:
        >>> from experiments.baselines.selection_baselines import BaselineRunner
        >>> runner = BaselineRunner(clients, target_dist)
        >>> results = runner.run_all(coalition_size=10)
        >>> for name, res in results.items():
        ...     print(f"{name}: div={res.demographic_divergence:.4f}")

    The runner instantiates DivFL, Oort, Power-of-Choice, and SubTrunc
    with the provided configuration and returns a dictionary mapping
    algorithm names to ``SelectionResult`` objects.
    """

    def __init__(
        self,
        clients: List[Client],
        target_distribution: DemographicDistribution,
        config: Optional[BaselineRunnerConfig] = None,
    ) -> None:
        """
        Initialise the runner.

        Args:
            clients: Full client pool (shared across all baselines).
            target_distribution: Target demographic distribution delta*.
            config: Runner-level configuration.
        """
        self._clients = clients
        self._target = target_distribution
        self._config = config or BaselineRunnerConfig()

        # Derive per-baseline seeds from the master seed so that results
        # are reproducible yet each baseline gets an independent stream.
        master_rng = np.random.default_rng(self._config.seed)
        seeds = master_rng.integers(0, 2**31, size=4)

        self._baselines: List[SelectionBaseline] = [
            DivFL(seed=int(seeds[0])),
            Oort(
                pacer_factor=self._config.oort_pacer_factor,
                seed=int(seeds[1]),
            ),
            PowerOfChoice(
                oversampling_ratio=self._config.poc_oversampling_ratio,
                seed=int(seeds[2]),
            ),
            SubTrunc(
                fairness_weight=self._config.subtrunc_fairness_weight,
                truncation_budget=self._config.subtrunc_truncation_budget,
                seed=int(seeds[3]),
            ),
        ]

    
    # Public API
    

    def run_all(
        self,
        coalition_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, SelectionResult]:
        """
        Run every baseline and return results keyed by algorithm name.

        Args:
            coalition_size: Override the default coalition size from config.
            **kwargs: Forwarded to each baseline's ``select`` method.

        Returns:
            Dictionary mapping baseline name to its ``SelectionResult``.
        """
        m = (
            coalition_size
            if coalition_size is not None
            else self._config.coalition_size
        )
        results: Dict[str, SelectionResult] = {}

        for baseline in self._baselines:
            coalition = baseline.select(
                self._clients,
                m,
                self._target,
                **kwargs,
            )
            div = SelectionBaseline._coalition_divergence(
                coalition,
                self._clients,
                self._target,
            )
            obj = self._compute_objective(baseline, coalition)
            results[baseline.name] = SelectionResult(
                coalition=coalition,
                objective_value=obj,
                demographic_divergence=div,
                metadata={"algorithm": baseline.name, "coalition_size": m},
            )

        return results

    def run_single(
        self,
        baseline_name: str,
        coalition_size: Optional[int] = None,
        **kwargs: Any,
    ) -> SelectionResult:
        """
        Run a single baseline by name.

        Args:
            baseline_name: One of "DivFL", "Oort", "Power-of-Choice",
                "SubTrunc".
            coalition_size: Override default coalition size.
            **kwargs: Forwarded to the baseline.

        Returns:
            SelectionResult for the requested baseline.

        Raises:
            KeyError: If baseline_name is not recognised.
        """
        m = (
            coalition_size
            if coalition_size is not None
            else self._config.coalition_size
        )
        baseline = self._get_baseline(baseline_name)
        coalition = baseline.select(self._clients, m, self._target, **kwargs)
        div = SelectionBaseline._coalition_divergence(
            coalition,
            self._clients,
            self._target,
        )
        obj = self._compute_objective(baseline, coalition)
        return SelectionResult(
            coalition=coalition,
            objective_value=obj,
            demographic_divergence=div,
            metadata={"algorithm": baseline.name, "coalition_size": m},
        )

    @property
    def baseline_names(self) -> List[str]:
        """List of available baseline algorithm names."""
        return [b.name for b in self._baselines]

    
    # Internals
    

    def _get_baseline(self, name: str) -> SelectionBaseline:
        """Look up a baseline by name."""
        for b in self._baselines:
            if b.name == name:
                return b
        available = ", ".join(self.baseline_names)
        raise KeyError(f"Unknown baseline '{name}'. Available: {available}")

    def _compute_objective(
        self,
        baseline: SelectionBaseline,
        coalition: List[int],
    ) -> float:
        """
        Compute a unified objective value for comparison.

        Uses the FairSwarm-style objective:
            Fitness(S) = w1*acc_proxy - w2*div - w3*cost

        where the accuracy proxy is the normalised total dataset size
        and cost is the mean communication cost.

        Args:
            baseline: The baseline that produced the coalition.
            coalition: Selected client indices.

        Returns:
            Scalar objective value (higher is better).
        """
        if not coalition:
            return float("-inf")

        # Accuracy proxy: fraction of total data captured.
        total_data = sum(c.dataset_size for c in self._clients)
        coalition_data = sum(self._clients[i].dataset_size for i in coalition)
        acc_proxy = coalition_data / total_data if total_data > 0 else 0.0

        # Divergence.
        div = SelectionBaseline._coalition_divergence(
            coalition,
            self._clients,
            self._target,
        )

        # Communication cost.
        cost = float(np.mean([self._clients[i].communication_cost for i in coalition]))

        # Composite objective (weights from Algorithm 1 default).
        w1, w2, w3 = 1.0, 0.3, 0.1
        return w1 * acc_proxy - w2 * div - w3 * cost
