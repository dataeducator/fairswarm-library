"""
Fairness-based fitness functions for FairSwarm.

This module implements the demographic fairness component of the
FairSwarm fitness function, based on Definition 2 in CLAUDE.md.

Definition 2 (Demographic Divergence):
    DemDiv(S) = D_KL(δ_S || δ*)

    where:
    - δ_S = (1/|S|) Σ_{i∈S} δ_i is the coalition's demographic distribution
    - δ* is the target demographic distribution (e.g., US Census 2020)
    - D_KL is the Kullback-Leibler divergence

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.demographics.divergence import kl_divergence
from fairswarm.fitness.base import FitnessFunction, FitnessResult
from fairswarm.types import Coalition

if TYPE_CHECKING:
    from fairswarm.core.client import Client


@dataclass
class FairnessGradient:
    """
    Result of fairness gradient computation.

    Attributes:
        gradient: The fairness gradient vector
        divergence: Current demographic divergence
        coalition_distribution: Coalition's demographic distribution
    """

    gradient: NDArray[np.float64]
    divergence: float
    coalition_distribution: NDArray[np.float64]


def compute_coalition_demographics(
    coalition: Coalition,
    clients: list[Client],
) -> NDArray[np.float64]:
    """
    Compute the aggregate demographic distribution of a coalition.

    Implements: δ_S = (1/|S|) Σ_{i∈S} δ_i

    Args:
        coalition: List of client indices
        clients: List of all clients

    Returns:
        Aggregated demographic distribution vector

    Raises:
        ValueError: If coalition is empty

    Example:
        >>> # Two clients with different demographics
        >>> demographics = compute_coalition_demographics([0, 2], clients)
        >>> print(demographics)  # Average of clients 0 and 2
    """
    if not coalition:
        raise ValueError("Coalition cannot be empty")

    # Validate all indices before processing
    for idx in coalition:
        if not (0 <= idx < len(clients)):
            raise ValueError(
                f"Coalition index {idx} is out of range for "
                f"{len(clients)} clients (valid range: 0 to {len(clients) - 1})"
            )

    # Get demographic vectors for coalition members
    demo_vectors = []
    for idx in coalition:
        client = clients[idx]
        demo_vectors.append(np.asarray(client.demographics))

    # Compute average: δ_S = (1/|S|) Σ_{i∈S} δ_i
    stacked = np.vstack(demo_vectors)
    result: NDArray[np.float64] = np.mean(stacked, axis=0)
    return result


def compute_fairness_gradient(
    position: NDArray[np.float64],
    clients: list[Client],
    target_distribution: DemographicDistribution,
    coalition_size: int,
    eps: float = 1e-10,
    use_kl_gradient: bool = True,
    group_performance: NDArray[np.float64] | None = None,
    outcome_weight: float = 0.3,
) -> FairnessGradient:
    """
    Compute the fairness gradient for position update.

    The fairness gradient guides particles toward positions that
    reduce demographic divergence. This is the novel contribution
    of the FairSwarm algorithm.

    Mathematical Derivation (from CLAUDE.md Algorithm 1):
        ∇_fair[i] = -∂DemDiv/∂x[i] ≈ (δ* - δ_current) · δ[i]

        where:
        - δ* is the target demographic distribution
        - δ_current is the current weighted coalition demographics
        - δ[i] is client i's demographic distribution

        The gradient is positive when selecting client i would
        move the coalition demographics closer to the target.

    Outcome-Aware Extension:
        When group_performance is provided (per-group AUC or accuracy),
        the gradient is augmented with a performance correction term:

        ∇_outcome[i] = Σ_g performance_gap[g] · δ[i][g]

        where performance_gap[g] = max_performance - performance[g].
        This biases selection toward clients whose data helps
        underperforming demographic groups.

    Implementation:
        When use_kl_gradient=True (default), we compute the proper
        gradient of KL divergence using the chain rule.

        When use_kl_gradient=False, we use the simpler approximation
        from the algorithm specification.

    Args:
        position: Current position vector (selection probabilities)
        clients: List of all clients
        target_distribution: Target demographic distribution δ*
        coalition_size: Target coalition size m
        eps: Small value for numerical stability
        use_kl_gradient: If True, compute proper KL derivative; else use
            the simplified (δ* - δ_current) · δ[i] approximation
        group_performance: Optional per-group performance array (e.g., AUC).
            When provided, enables outcome-aware gradient correction.
        outcome_weight: Weight for the outcome-aware correction term.
            Controls balance between demographic and outcome fairness.

    Returns:
        FairnessGradient with gradient vector and diagnostics

    Algorithm Reference:
        Used in Algorithm 1 for:
        v_fairness ← c₃ · ∇_fair

    Example:
        >>> gradient = compute_fairness_gradient(
        ...     position=particle.position,
        ...     clients=clients,
        ...     target_distribution=CensusTarget.US_2020.as_distribution(),
        ...     coalition_size=10,
        ... )
        >>> particle.velocity += config.fairness_coeff * gradient.gradient
    """
    n_clients = len(clients)
    target = target_distribution.as_array()

    # Normalize position to get soft selection weights
    # Higher position values = higher selection probability
    position_sum = np.sum(position) + eps
    weights = position / position_sum

    # Compute weighted coalition demographics (soft selection)
    # This is a differentiable approximation of hard selection
    demo_matrix = np.vstack([np.asarray(c.demographics) for c in clients])

    # Weighted average: δ_soft = Σ w_i δ_i
    coalition_demo = np.sum(weights[:, np.newaxis] * demo_matrix, axis=0)

    # Ensure coalition_demo is a valid probability distribution
    coalition_demo = np.clip(coalition_demo, eps, 1.0)
    coalition_demo = coalition_demo / (coalition_demo.sum() + eps)

    # Compute current divergence
    divergence = kl_divergence(coalition_demo, target)

    # Compute gradient for each client
    gradient = np.zeros(n_clients)

    if use_kl_gradient:
        # Proper KL divergence gradient using chain rule
        # D_KL(p || q) = Σ p_k * log(p_k / q_k)
        # ∂D_KL/∂p_k = log(p_k / q_k) + 1
        #
        # For the chain rule through the weighted average:
        # ∂D_KL/∂position_i = Σ_k (∂D_KL/∂coalition_demo_k) * (∂coalition_demo_k/∂position_i)
        #
        # ∂coalition_demo_k/∂position_i for w_j = pos_j / Σ pos_k:
        # = (demo_ik - coalition_demo_k) / position_sum

        # KL gradient w.r.t. coalition demographics
        target_safe = np.clip(target, eps, 1.0)
        kl_grad_demo = np.log(coalition_demo / target_safe) + 1

        for i in range(n_clients):
            client_demo = demo_matrix[i]
            # Derivative of weighted average w.r.t. position_i:
            # ∂coalition_demo_k/∂position_i = (demo_ik - coalition_demo_k) / position_sum
            # Derivation: w_j = pos_j/S, ∂w_i/∂pos_i = (1-w_i)/S,
            # ∂w_j/∂pos_i = -w_j/S (j≠i), chain rule gives (demo_i - coalition) / S
            d_coalition_d_pos = (client_demo - coalition_demo) / position_sum

            # Chain rule: gradient of divergence w.r.t. position_i
            # We want to REDUCE divergence, so negate
            gradient[i] = -np.dot(kl_grad_demo, d_coalition_d_pos)
    else:
        # Simplified approximation from CLAUDE.md Algorithm 1:
        # ∇_fair[i] = (δ* - δ_current) · δ[i]
        #
        # Intuition: The gap (δ* - δ_current) shows which demographics
        # we need more of. The dot product with δ[i] measures how well
        # client i can fill that gap.
        gap = target - coalition_demo  # What demographics we need more of

        for i in range(n_clients):
            client_demo = demo_matrix[i]
            # Positive gradient if client helps fill the demographic gap
            gradient[i] = np.dot(gap, client_demo)

    # Outcome-aware correction: bias toward clients that help
    # underperforming demographic groups (novel extension)
    if group_performance is not None and len(group_performance) > 1:
        n_groups = min(len(group_performance), demo_matrix.shape[1])
        perf = group_performance[:n_groups]
        # Performance gap: how far each group is from the best
        perf_gap = np.max(perf) - perf  # Higher gap = needs more help
        perf_gap = perf_gap / (np.sum(perf_gap) + eps)  # Normalize

        # For each client, compute how much their data helps underperforming groups
        outcome_correction = np.zeros(n_clients)
        for i in range(n_clients):
            client_demo = demo_matrix[i, :n_groups]
            # Clients with data from underperforming groups get positive correction
            outcome_correction[i] = np.dot(perf_gap, client_demo)

        # Normalize correction to same scale as fairness gradient
        corr_norm = np.linalg.norm(outcome_correction)
        fair_norm = np.linalg.norm(gradient)
        if corr_norm > eps and fair_norm > eps:
            outcome_correction = outcome_correction * (fair_norm / corr_norm)

        gradient = gradient + outcome_weight * outcome_correction

    # Handle numerical issues: replace NaN/inf with zeros
    gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip gradient norm to prevent extreme velocity updates,
    # but preserve magnitude so the restoring force is proportional
    # to divergence (required by Theorem 2's drift analysis).
    grad_norm = np.linalg.norm(gradient)
    max_grad_norm = 10.0
    if grad_norm > max_grad_norm:
        gradient = gradient * (max_grad_norm / grad_norm)

    return FairnessGradient(
        gradient=gradient,
        divergence=divergence,
        coalition_distribution=coalition_demo,
    )


class DemographicFitness(FitnessFunction):
    """
    Fitness function based on demographic fairness.

    Evaluates coalitions based on how well their demographic
    distribution matches a target distribution.

    Fitness = -DemDiv(S) = -D_KL(δ_S || δ*)

    Higher fitness (less divergence) is better.

    Definition 2 Reference:
        DemDiv(S) = D_KL(δ_S || δ*)
        where δ_S = (1/|S|) Σ_{i∈S} δ_i

    Theorem 2 Connection:
        With appropriate hyperparameters, FairSwarm guarantees:
        DemDiv(S*) ≤ ε with probability ≥ 1 - δ

    Attributes:
        target_distribution: The target demographic distribution δ*
        divergence_weight: Weight for the divergence penalty (λ)

    Example:
        >>> from fairswarm.demographics import CensusTarget
        >>> fitness = DemographicFitness(
        ...     target_distribution=CensusTarget.US_2020.as_distribution(),
        ...     divergence_weight=1.0,
        ... )
        >>> result = fitness.evaluate(coalition, clients)
        >>> print(f"Divergence: {result.components['divergence']:.4f}")
    """

    def __init__(
        self,
        target_distribution: DemographicDistribution,
        divergence_weight: float = 1.0,
    ):
        """
        Initialize DemographicFitness.

        Args:
            target_distribution: Target demographic distribution δ*
            divergence_weight: Weight for divergence in fitness (λ)
        """
        self.target_distribution = target_distribution
        self.divergence_weight = divergence_weight

    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> FitnessResult:
        """
        Evaluate fitness based on demographic divergence.

        Fitness = -λ · DemDiv(S)

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            FitnessResult with divergence-based fitness
        """
        if not coalition:
            return FitnessResult(
                value=float("-inf"),
                components={"divergence": float("inf")},
                coalition=coalition,
                metadata={"error": "Empty coalition"},
            )

        # Compute coalition demographics: δ_S = (1/|S|) Σ_{i∈S} δ_i
        coalition_demo = compute_coalition_demographics(coalition, clients)

        # Compute divergence: DemDiv(S) = D_KL(δ_S || δ*)
        target = self.target_distribution.as_array()
        divergence = kl_divergence(coalition_demo, target)

        # Fitness = -λ · DemDiv(S)
        fitness = -self.divergence_weight * divergence

        return FitnessResult(
            value=fitness,
            components={
                "divergence": divergence,
                "divergence_penalty": -self.divergence_weight * divergence,
            },
            coalition=coalition,
            metadata={
                "coalition_demographics": coalition_demo.tolist(),
                "target_demographics": target.tolist(),
            },
        )

    def compute_gradient(
        self,
        position: NDArray[np.float64],
        clients: list[Client],
        coalition_size: int,
    ) -> NDArray[np.float64]:
        """
        Compute fairness gradient for velocity update.

        Args:
            position: Current particle position
            clients: List of all clients
            coalition_size: Target coalition size

        Returns:
            Gradient vector guiding toward fairer positions
        """
        result = compute_fairness_gradient(
            position=position,
            clients=clients,
            target_distribution=self.target_distribution,
            coalition_size=coalition_size,
        )
        return result.gradient

    def get_config(self) -> dict[str, Any]:
        """Get configuration for reproducibility."""
        return {
            "class": self.__class__.__name__,
            "target_distribution": (
                self.target_distribution.as_dict()
                if self.target_distribution.labels
                else self.target_distribution.values.tolist()
            ),
            "divergence_weight": self.divergence_weight,
        }


class AccuracyFairnessFitness(FitnessFunction):
    """
    Combined accuracy and fairness fitness.

    Fitness = ValAcc(S) - λ · DemDiv(S)

    This is the primary fitness function for FairSwarm, balancing
    model performance with demographic fairness.

    Attributes:
        target_distribution: Target demographic distribution
        fairness_weight: Weight for fairness penalty (λ)
        accuracy_fn: Function to compute validation accuracy

    Example:
        >>> def get_accuracy(coalition, clients):
        ...     # Simulate federated training and return validation accuracy
        ...     return 0.85
        ...
        >>> fitness = AccuracyFairnessFitness(
        ...     target_distribution=CensusTarget.US_2020.as_distribution(),
        ...     accuracy_fn=get_accuracy,
        ...     fairness_weight=0.3,
        ... )
    """

    def __init__(
        self,
        target_distribution: DemographicDistribution,
        accuracy_fn: Any | None = None,
        fairness_weight: float = 0.3,
    ):
        """
        Initialize AccuracyFairnessFitness.

        Args:
            target_distribution: Target demographic distribution
            accuracy_fn: Callable (coalition, clients) -> accuracy
            fairness_weight: Weight for fairness penalty (λ)
        """
        self.target_distribution = target_distribution
        self.accuracy_fn = accuracy_fn
        self.fairness_weight = fairness_weight

    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> FitnessResult:
        """
        Evaluate combined accuracy and fairness fitness.

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            FitnessResult with accuracy and fairness components
        """
        if not coalition:
            return FitnessResult(
                value=float("-inf"),
                components={
                    "accuracy": 0.0,
                    "divergence": float("inf"),
                },
                coalition=coalition,
            )

        # Compute accuracy (or use default if no function provided)
        if self.accuracy_fn is not None:
            accuracy = self.accuracy_fn(coalition, clients)
        else:
            # Default: use normalized dataset size as accuracy proxy
            # Larger datasets generally correlate with better model performance
            dataset_sizes = [
                clients[i].dataset_size for i in coalition if 0 <= i < len(clients)
            ]
            if dataset_sizes:
                # Normalize to [0, 1] range based on total possible data
                max_possible = sum(c.dataset_size for c in clients)
                accuracy = (
                    sum(dataset_sizes) / max_possible if max_possible > 0 else 0.0
                )
            else:
                accuracy = 0.0

        # Compute fairness (demographic divergence)
        coalition_demo = compute_coalition_demographics(coalition, clients)
        target = self.target_distribution.as_array()
        divergence = kl_divergence(coalition_demo, target)

        # Combined fitness: ValAcc(S) - λ · DemDiv(S)
        fitness = accuracy - self.fairness_weight * divergence

        return FitnessResult(
            value=fitness,
            components={
                "accuracy": accuracy,
                "divergence": divergence,
                "fairness_penalty": -self.fairness_weight * divergence,
            },
            coalition=coalition,
            metadata={
                "coalition_demographics": coalition_demo.tolist(),
            },
        )

    def compute_gradient(
        self,
        position: NDArray[np.float64],
        clients: list[Client],
        coalition_size: int,
    ) -> NDArray[np.float64]:
        """
        Compute gradient for fairness-aware velocity update.

        Note: Currently only considers fairness gradient.
        Accuracy gradient would require differentiable training.

        Args:
            position: Current particle position
            clients: List of all clients
            coalition_size: Target coalition size

        Returns:
            Fairness gradient vector
        """
        result = compute_fairness_gradient(
            position=position,
            clients=clients,
            target_distribution=self.target_distribution,
            coalition_size=coalition_size,
        )
        return result.gradient

    def get_config(self) -> dict[str, Any]:
        """Get configuration for reproducibility."""
        return {
            "class": self.__class__.__name__,
            "target_distribution": (
                self.target_distribution.as_dict()
                if self.target_distribution.labels
                else self.target_distribution.values.tolist()
            ),
            "fairness_weight": self.fairness_weight,
            "has_accuracy_fn": self.accuracy_fn is not None,
        }
