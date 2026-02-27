"""
Grey Wolf Optimizer (GWO) Baseline for FairSwarm Comparison.

Implements the Grey Wolf Optimizer (Mirjalili et al., 2014) adapted for
client selection in federated learning. Used as a swarm-intelligence
baseline to demonstrate FairSwarm's advantage from the fairness gradient.

GWO simulates grey wolf social hierarchy:
    - Alpha (α): best solution
    - Beta (β): second-best solution
    - Delta (δ): third-best solution
    - Omega (ω): remaining wolves, guided by top 3

Key difference from FairSwarm: GWO has NO fairness gradient term.
It optimizes the same composite fitness function but relies solely on
the wolf hierarchy dynamics, without explicit demographic guidance.

Reference:
    Mirjalili, S., Mirjalili, S.M., & Lewis, A. (2014).
    Grey Wolf Optimizer. Advances in Engineering Software, 69, 46-61.

    Khan, H.U. & Goodridge, W. (2024).
    A Comparative Analysis of Swarm Intelligence Algorithms for
    Cybersecurity-Focused Federated Learning. arXiv:2411.18877.

Author: Tenicka Norwood
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray

from fairswarm.core.client import Client
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.demographics.divergence import kl_divergence

from baselines.selection_baselines import SelectionBaseline


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sigmoid function for bounding positions to [0, 1]."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class GreyWolfOptimizer(SelectionBaseline):
    """
    Grey Wolf Optimizer for federated learning client selection.

    Adapts GWO to the coalition selection problem using the same
    position encoding as FairSwarm: each wolf's position is a vector
    in [0,1]^n representing client selection probabilities, decoded
    via SelectTop(position, m) to a discrete coalition.

    The composite fitness function is:
        Fitness(S) = w1 * accuracy_proxy - w2 * DemDiv(S, δ*) - w3 * cost

    Unlike FairSwarm, GWO has no fairness-aware velocity term (c₃·∇_fair).
    This makes it the ideal control for measuring the fairness gradient's
    contribution.

    Attributes:
        n_wolves: Number of wolves in the pack (population size).
        n_iterations: Number of optimization iterations.
        w_accuracy: Weight for accuracy component in fitness.
        w_fairness: Weight for fairness (divergence) component.
        w_cost: Weight for communication cost component.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_wolves: int = 30,
        n_iterations: int = 100,
        w_accuracy: float = 0.5,
        w_fairness: float = 0.3,
        w_cost: float = 0.2,
        seed: Optional[int] = None,
    ) -> None:
        self._n_wolves = n_wolves
        self._n_iterations = n_iterations
        self._w_accuracy = w_accuracy
        self._w_fairness = w_fairness
        self._w_cost = w_cost
        self._rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "Grey Wolf Optimizer"

    def _evaluate_fitness(
        self,
        position: NDArray[np.float64],
        clients: List[Client],
        coalition_size: int,
        target: DemographicDistribution,
        demo_matrix: NDArray[np.float64],
        target_arr: NDArray[np.float64],
    ) -> tuple[float, List[int]]:
        """
        Evaluate composite fitness for a wolf position.

        Args:
            position: Wolf position in [0,1]^n.
            clients: Client pool.
            coalition_size: Target size m.
            target: Target demographic distribution.
            demo_matrix: Pre-computed (n, k) demographic matrix.
            target_arr: Pre-computed target array.

        Returns:
            Tuple of (fitness_value, coalition_indices).
        """
        # Decode: select top-m clients by position value
        n = len(clients)
        m = min(coalition_size, n)
        indices = np.argsort(position)[-m:]
        coalition = sorted(indices.tolist())

        # Accuracy proxy: weighted mean of dataset sizes (normalized)
        sizes = np.array([clients[i].dataset_size for i in coalition], dtype=np.float64)
        max_size = max(c.dataset_size for c in clients)
        accuracy = float(np.mean(sizes) / max_size) if max_size > 0 else 0.0

        # Demographic divergence
        coalition_demo = np.mean(demo_matrix[coalition], axis=0)
        divergence = kl_divergence(coalition_demo, target_arr)

        # Communication cost proxy: mean of communication costs
        costs = np.array(
            [clients[i].communication_cost for i in coalition], dtype=np.float64
        )
        cost = float(np.mean(costs))

        fitness = (
            self._w_accuracy * accuracy
            - self._w_fairness * divergence
            - self._w_cost * cost
        )
        return fitness, coalition

    def select(
        self,
        clients: List[Client],
        coalition_size: int,
        target_distribution: DemographicDistribution,
        **kwargs: Any,
    ) -> List[int]:
        """
        Select a coalition using Grey Wolf Optimizer.

        Implements the standard GWO algorithm (Mirjalili et al., 2014):
        1. Initialize wolf pack with random positions
        2. Identify alpha, beta, delta (top 3 wolves)
        3. Update omega wolves toward weighted average of top 3
        4. Linearly decrease exploration parameter a from 2 to 0

        Args:
            clients: Full client pool.
            coalition_size: Number of clients to select (m).
            target_distribution: Target demographic distribution δ*.

        Returns:
            List of selected client indices.
        """
        n = len(clients)
        m = min(coalition_size, n)
        if m >= n:
            return list(range(n))

        # Pre-compute demographics
        demo_matrix = self._get_demo_matrix(clients)
        target_arr = target_distribution.as_array()

        # Initialize wolf pack
        positions = self._rng.uniform(0, 1, size=(self._n_wolves, n))
        fitnesses = np.full(self._n_wolves, -np.inf)

        # Evaluate initial positions
        best_coalitions: List[List[int]] = [[] for _ in range(self._n_wolves)]
        for i in range(self._n_wolves):
            fitnesses[i], best_coalitions[i] = self._evaluate_fitness(
                positions[i], clients, m, target_distribution, demo_matrix, target_arr
            )

        # Identify alpha, beta, delta
        sorted_idx = np.argsort(fitnesses)[::-1]
        alpha_pos = positions[sorted_idx[0]].copy()
        alpha_fit = fitnesses[sorted_idx[0]]
        alpha_coalition = best_coalitions[sorted_idx[0]]

        beta_pos = positions[sorted_idx[1]].copy()
        delta_pos = (
            positions[sorted_idx[2]].copy() if self._n_wolves > 2 else alpha_pos.copy()
        )

        # Main GWO loop
        for t in range(self._n_iterations):
            # Linearly decrease a from 2 to 0
            a = 2.0 - 2.0 * (t / self._n_iterations)

            for i in range(self._n_wolves):
                # Generate random vectors
                r1 = self._rng.uniform(0, 1, n)
                r2 = self._rng.uniform(0, 1, n)

                # Coefficient vectors
                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r2

                r1 = self._rng.uniform(0, 1, n)
                r2 = self._rng.uniform(0, 1, n)
                A2 = 2.0 * a * r1 - a
                C2 = 2.0 * r2

                r1 = self._rng.uniform(0, 1, n)
                r2 = self._rng.uniform(0, 1, n)
                A3 = 2.0 * a * r1 - a
                C3 = 2.0 * r2

                # Encircling prey: compute distance from alpha, beta, delta
                D_alpha = np.abs(C1 * alpha_pos - positions[i])
                D_beta = np.abs(C2 * beta_pos - positions[i])
                D_delta = np.abs(C3 * delta_pos - positions[i])

                # Update position toward the three leaders
                X1 = alpha_pos - A1 * D_alpha
                X2 = beta_pos - A2 * D_beta
                X3 = delta_pos - A3 * D_delta

                # New position is average of three guided positions
                positions[i] = (X1 + X2 + X3) / 3.0

                # Bound to [0, 1] via sigmoid
                positions[i] = _sigmoid(positions[i])

                # Evaluate new position
                fit, coal = self._evaluate_fitness(
                    positions[i],
                    clients,
                    m,
                    target_distribution,
                    demo_matrix,
                    target_arr,
                )
                fitnesses[i] = fit
                best_coalitions[i] = coal

            # Update alpha, beta, delta
            sorted_idx = np.argsort(fitnesses)[::-1]
            if fitnesses[sorted_idx[0]] > alpha_fit:
                alpha_pos = positions[sorted_idx[0]].copy()
                alpha_fit = fitnesses[sorted_idx[0]]
                alpha_coalition = best_coalitions[sorted_idx[0]]
            beta_pos = positions[sorted_idx[1]].copy()
            delta_pos = (
                positions[sorted_idx[2]].copy()
                if self._n_wolves > 2
                else alpha_pos.copy()
            )

        return alpha_coalition
