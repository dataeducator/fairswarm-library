"""
Extended fairness and equity metrics for FairSwarm.

Implements additional fairness metrics beyond demographic divergence
to match current SOTA (FlexFair, SubTrunc, LongFed, FairFML).

Metrics:
    - client_dissimilarity: Performance variance across clients
    - equalized_odds_gap: Max gap in TPR/FPR across groups
    - demographic_parity_difference: Gap in positive prediction rates
    - equal_opportunity_difference: Gap in true positive rates

Mathematical Foundation:
    These metrics complement Definition 2 (DemDiv) by measuring
    outcome-level fairness rather than input-level representation.

    While DemDiv(S) = D_KL(delta_S || delta*) measures coalition
    composition, these metrics measure model behavior:

    - Client Dissimilarity (SubTrunc 2024, UnionFL):
        CD(S) = std({acc_i : i in S})
        Lower values indicate more equitable performance.

    - Equalized Odds Gap (Hardt et al. 2016, FlexFair 2025):
        EOG = max_k |TPR_k - TPR_ref| + max_k |FPR_k - FPR_ref|
        Measures worst-case disparity in error rates.

    - Equal Opportunity Difference (Hardt et al. 2016, LongFed 2025):
        EOD = max_{k1, k2} |TPR_k1 - TPR_k2|
        Measures max gap in true positive rates across groups.

    - Demographic Parity Difference (Dwork et al. 2012, FairFML):
        DPD = max_{k1, k2} |P(Y_hat=1 | G=k1) - P(Y_hat=1 | G=k2)|
        Measures max gap in positive prediction rates.

References:
    - FlexFair: Flexible Fairness in Federated Learning,
      Nature Communications (2025)
    - SubTrunc: Subgroup-Aware Truncation for Fair FL,
      ICML (2024)
    - LongFed: Long-Term Fairness in Federated Learning,
      NeurIPS (2025)
    - FairFML: Fair Federated Machine Learning,
      AAAI (2024)

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from fairswarm.fitness.base import FitnessFunction, FitnessResult
from fairswarm.types import Coalition

if TYPE_CHECKING:
    from fairswarm.core.client import Client


__all__ = [
    "client_dissimilarity",
    "equalized_odds_gap",
    "equal_opportunity_difference",
    "demographic_parity_difference",
    "ClientDissimilarityFitness",
]


# =============================================================================
# Standalone Metric Functions
# =============================================================================


def client_dissimilarity(
    per_client_accuracies: list[float],
) -> float:
    """
    Compute client dissimilarity as the standard deviation of per-client accuracies.

    This metric captures the variance of model performance across clients in
    a coalition. Lower values indicate more equitable performance across all
    participating institutions.

    Mathematical Definition (SubTrunc 2024):
        CD(S) = std({acc_i : i in S})
             = sqrt( (1/|S|) * sum_{i in S} (acc_i - mean_acc)^2 )

    This is the metric optimized by SubTrunc and UnionFL for achieving
    uniform performance across federated learning participants.

    Args:
        per_client_accuracies: List of accuracy values (or any performance
            metric) for each client in the coalition. Each value should
            be in [0, 1] for standard accuracy, but the function accepts
            any numeric range.

    Returns:
        Standard deviation of per-client accuracies (non-negative).
        Returns 0.0 if fewer than 2 clients are provided.

    Raises:
        ValueError: If per_client_accuracies is empty.

    Example:
        >>> # Equitable coalition: all clients perform similarly
        >>> client_dissimilarity([0.85, 0.84, 0.86, 0.85])
        0.0070...
        >>> # Inequitable coalition: large performance gap
        >>> client_dissimilarity([0.95, 0.60, 0.88, 0.72])
        0.1317...

    Research Reference:
        SubTrunc (ICML 2024) minimizes this metric to ensure no client
        is disproportionately harmed by the federated model.
    """
    if not per_client_accuracies:
        raise ValueError(
            "per_client_accuracies must be non-empty. "
            "Provide at least one client accuracy value."
        )

    if len(per_client_accuracies) < 2:
        return 0.0

    accuracies = np.asarray(per_client_accuracies, dtype=np.float64)
    return float(np.std(accuracies))


def equalized_odds_gap(
    group_tpr: NDArray[np.float64],
    group_fpr: NDArray[np.float64],
) -> float:
    """
    Compute equalized odds gap across demographic groups.

    Equalized odds requires that the true positive rate (TPR) and
    false positive rate (FPR) are equal across all demographic groups.
    The gap measures the worst-case deviation.

    Mathematical Definition (Hardt et al. 2016, FlexFair 2025):
        EOG = max_{k1, k2} |TPR_{k1} - TPR_{k2}|
            + max_{k1, k2} |FPR_{k1} - FPR_{k2}|

    This is the sum of the maximum TPR gap and maximum FPR gap
    across all pairs of demographic groups. A value of 0 indicates
    perfect equalized odds.

    Args:
        group_tpr: Array of true positive rates, one per demographic group.
            Shape: (k,) where k is the number of groups.
            Each value should be in [0, 1].
        group_fpr: Array of false positive rates, one per demographic group.
            Shape: (k,) where k is the number of groups.
            Each value should be in [0, 1].

    Returns:
        Equalized odds gap (non-negative). Range: [0, 2].
        0.0 means perfect equalized odds across all groups.

    Raises:
        ValueError: If group_tpr and group_fpr have different lengths
            or contain fewer than 2 groups.

    Example:
        >>> import numpy as np
        >>> # Three demographic groups with different TPR/FPR
        >>> tpr = np.array([0.90, 0.85, 0.70])
        >>> fpr = np.array([0.10, 0.12, 0.20])
        >>> equalized_odds_gap(tpr, fpr)
        0.3

    Research Reference:
        FlexFair (Nature Communications 2025) uses this metric as a
        primary fairness constraint in federated learning optimization.
        Hardt et al. (2016) originally defined equalized odds.
    """
    group_tpr = np.asarray(group_tpr, dtype=np.float64)
    group_fpr = np.asarray(group_fpr, dtype=np.float64)

    if len(group_tpr) != len(group_fpr):
        raise ValueError(
            f"group_tpr and group_fpr must have the same length. "
            f"Got {len(group_tpr)} and {len(group_fpr)}."
        )

    if len(group_tpr) < 2:
        raise ValueError(
            "At least 2 demographic groups are required to compute "
            f"equalized odds gap. Got {len(group_tpr)}."
        )

    # Maximum gap in TPR across all pairs of groups
    tpr_gap = float(np.max(group_tpr) - np.min(group_tpr))

    # Maximum gap in FPR across all pairs of groups
    fpr_gap = float(np.max(group_fpr) - np.min(group_fpr))

    return tpr_gap + fpr_gap


def equal_opportunity_difference(
    group_tpr: NDArray[np.float64],
) -> float:
    """
    Compute equal opportunity difference across demographic groups.

    Equal opportunity requires that the true positive rate (recall)
    is equal across all demographic groups. This metric measures
    the maximum gap, focusing on harm to the positive class.

    Mathematical Definition (Hardt et al. 2016, LongFed 2025):
        EOD = max_{k1, k2} |TPR_{k1} - TPR_{k2}|
            = max(group_tpr) - min(group_tpr)

    This captures the worst-case disparity in the model's ability
    to correctly identify positive cases across demographic groups.
    In healthcare, this directly measures whether the model is
    equally sensitive to detecting conditions across populations.

    Args:
        group_tpr: Array of true positive rates (recall), one per
            demographic group. Shape: (k,) where k is the number
            of groups. Each value should be in [0, 1].

    Returns:
        Equal opportunity difference (non-negative). Range: [0, 1].
        0.0 means equal recall across all groups.

    Raises:
        ValueError: If group_tpr contains fewer than 2 groups.

    Example:
        >>> import numpy as np
        >>> # Model has higher recall for group 0 than group 2
        >>> tpr = np.array([0.92, 0.88, 0.75])
        >>> equal_opportunity_difference(tpr)
        0.17

    Research Reference:
        LongFed (NeurIPS 2025) tracks this metric over long-horizon
        federated training to ensure fairness does not degrade.
        In clinical settings (SwarmClinical), this measures whether
        mortality prediction is equally sensitive across patient groups.
    """
    group_tpr = np.asarray(group_tpr, dtype=np.float64)

    if len(group_tpr) < 2:
        raise ValueError(
            "At least 2 demographic groups are required to compute "
            f"equal opportunity difference. Got {len(group_tpr)}."
        )

    return float(np.max(group_tpr) - np.min(group_tpr))


def demographic_parity_difference(
    group_positive_rates: NDArray[np.float64],
) -> float:
    """
    Compute demographic parity difference across groups.

    Demographic parity requires that the positive prediction rate
    is equal across all demographic groups, regardless of the
    true label distribution.

    Mathematical Definition (Dwork et al. 2012, FairFML):
        DPD = max_{k1, k2} |P(Y_hat=1 | G=k1) - P(Y_hat=1 | G=k2)|
            = max(group_positive_rates) - min(group_positive_rates)

    This is the simplest group fairness metric, measuring whether
    the model's predictions are independent of group membership.

    Note:
        Demographic parity can conflict with accuracy when base rates
        differ across groups. In healthcare, disease prevalence often
        varies by demographic group, so this metric should be used
        alongside equalized odds metrics, not in isolation.

    Args:
        group_positive_rates: Array of positive prediction rates
            P(Y_hat=1 | G=k) for each demographic group k.
            Shape: (k,) where k is the number of groups.
            Each value should be in [0, 1].

    Returns:
        Demographic parity difference (non-negative). Range: [0, 1].
        0.0 means equal positive prediction rates across all groups.

    Raises:
        ValueError: If group_positive_rates contains fewer than 2 groups.

    Example:
        >>> import numpy as np
        >>> # Model predicts positive more often for group 0
        >>> rates = np.array([0.45, 0.30, 0.32])
        >>> demographic_parity_difference(rates)
        0.15

    Research Reference:
        Dwork et al. (2012) introduced demographic parity as a
        formalization of anti-classification. FairFML (AAAI 2024)
        uses this in federated settings with privacy constraints.
    """
    group_positive_rates = np.asarray(group_positive_rates, dtype=np.float64)

    if len(group_positive_rates) < 2:
        raise ValueError(
            "At least 2 demographic groups are required to compute "
            f"demographic parity difference. Got {len(group_positive_rates)}."
        )

    return float(np.max(group_positive_rates) - np.min(group_positive_rates))


# =============================================================================
# FitnessFunction Subclass
# =============================================================================


class ClientDissimilarityFitness(FitnessFunction):
    """
    Fitness function based on client performance dissimilarity.

    Evaluates coalitions by measuring how uniformly a model would
    perform across coalition members. Lower dissimilarity (higher
    fitness) indicates more equitable performance distribution.

    Fitness = -CD(S) = -std({acc_i : i in S})

    Higher fitness (less dissimilarity) is better.

    This addresses a key limitation of demographic divergence alone:
    a coalition can have perfect demographic representation but still
    produce a model that performs poorly on certain clients' data.
    Client dissimilarity directly measures outcome equity.

    Mathematical Foundation (SubTrunc 2024):
        CD(S) = std({acc_i : i in S})

        The gradient guides particles toward coalitions where
        including a client would reduce the performance variance.

    Theorem 2 Connection:
        While Theorem 2 bounds demographic divergence (input fairness),
        client dissimilarity bounds outcome fairness. Together they
        provide a more complete fairness guarantee.

    Attributes:
        accuracy_fn: Function that returns per-client accuracies for
            a given coalition. Signature: (coalition, clients) -> List[float]
        dissimilarity_weight: Weight for the dissimilarity penalty

    Example:
        >>> def get_per_client_acc(coalition, clients):
        ...     # Simulate per-client evaluation
        ...     return [0.85 + 0.01 * i for i in range(len(coalition))]
        ...
        >>> fitness = ClientDissimilarityFitness(
        ...     accuracy_fn=get_per_client_acc,
        ...     dissimilarity_weight=1.0,
        ... )
        >>> result = fitness.evaluate(coalition, clients)
        >>> print(f"Dissimilarity: {result.components['dissimilarity']:.4f}")
    """

    def __init__(
        self,
        accuracy_fn: Callable[[Coalition, list[Client]], list[float]] | None = None,
        dissimilarity_weight: float = 1.0,
    ):
        """
        Initialize ClientDissimilarityFitness.

        Args:
            accuracy_fn: Callable (coalition, clients) -> List[float]
                that returns per-client accuracy values for the given
                coalition. If None, uses dataset_size-based proxy where
                larger datasets are assumed to yield better local accuracy.
            dissimilarity_weight: Weight for dissimilarity in fitness.
                Higher values penalize performance variance more strongly.
        """
        self.accuracy_fn = accuracy_fn
        self.dissimilarity_weight = dissimilarity_weight

    def _get_per_client_accuracies(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> list[float]:
        """
        Get per-client accuracy values for a coalition.

        If no accuracy_fn is provided, uses a dataset-size-based proxy:
        accuracy_i = dataset_size_i / max_dataset_size across all clients.

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            List of accuracy values, one per coalition member
        """
        if self.accuracy_fn is not None:
            return self.accuracy_fn(coalition, clients)

        # Proxy: normalized dataset size as accuracy estimate
        # Clients with larger datasets tend to have better local models
        max_size = max(
            (clients[i].dataset_size for i in coalition if 0 <= i < len(clients)),
            default=1,
        )
        if max_size <= 0:
            max_size = 1

        accuracies: list[float] = []
        for idx in coalition:
            if 0 <= idx < len(clients):
                accuracies.append(clients[idx].dataset_size / max_size)

        return accuracies

    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> FitnessResult:
        """
        Evaluate fitness based on client performance dissimilarity.

        Fitness = -w * CD(S) = -w * std({acc_i : i in S})

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            FitnessResult with dissimilarity-based fitness
        """
        if not coalition:
            return FitnessResult(
                value=float("-inf"),
                components={"dissimilarity": float("inf")},
                coalition=coalition,
                metadata={"error": "Empty coalition"},
            )

        # Get per-client accuracies
        per_client_acc = self._get_per_client_accuracies(coalition, clients)

        if len(per_client_acc) < 2:
            # Single client: no dissimilarity possible
            return FitnessResult(
                value=0.0,
                components={
                    "dissimilarity": 0.0,
                    "dissimilarity_penalty": 0.0,
                    "mean_accuracy": per_client_acc[0] if per_client_acc else 0.0,
                },
                coalition=coalition,
            )

        # Compute client dissimilarity: CD(S) = std({acc_i})
        dissimilarity = client_dissimilarity(per_client_acc)
        mean_acc = float(np.mean(per_client_acc))

        # Fitness = -w * CD(S)
        fitness = -self.dissimilarity_weight * dissimilarity

        return FitnessResult(
            value=fitness,
            components={
                "dissimilarity": dissimilarity,
                "dissimilarity_penalty": -self.dissimilarity_weight * dissimilarity,
                "mean_accuracy": mean_acc,
                "min_accuracy": float(np.min(per_client_acc)),
                "max_accuracy": float(np.max(per_client_acc)),
            },
            coalition=coalition,
            metadata={
                "per_client_accuracies": per_client_acc,
                "n_clients_evaluated": len(per_client_acc),
            },
        )

    def compute_gradient(
        self,
        position: NDArray[np.float64],
        clients: list[Client],
        coalition_size: int,
    ) -> NDArray[np.float64]:
        """
        Compute gradient for equity-aware velocity update.

        The gradient estimates how including each client would affect
        the performance variance of the coalition. Clients whose
        dataset sizes are closer to the coalition mean receive higher
        gradient values (they would reduce variance).

        Derivation:
            For the proxy case (dataset-size-based accuracy):
            Let s_i = dataset_size_i. The dissimilarity is std(s_selected).
            The gradient approximation for client i is:
                grad[i] = -(s_i - mean_s)^2
            Normalized and negated so that clients reducing variance
            get positive gradient values.

        Args:
            position: Current particle position (selection probabilities)
            clients: List of all clients
            coalition_size: Target coalition size m

        Returns:
            Gradient vector of same dimension as position.
            Positive values for clients that would reduce dissimilarity.
        """
        n_clients = len(clients)
        eps = 1e-10

        # Compute weighted mean dataset size using position as soft weights
        position_sum = np.sum(position) + eps
        weights = position / position_sum

        sizes = np.array(
            [c.dataset_size for c in clients],
            dtype=np.float64,
        )

        # Normalize sizes to [0, 1]
        max_size = np.max(sizes)
        if max_size > 0:
            normalized_sizes = sizes / max_size
        else:
            normalized_sizes = np.zeros(n_clients)

        # Weighted mean accuracy (using size proxy)
        weighted_mean = np.sum(weights * normalized_sizes)

        # Gradient: clients closer to the mean reduce variance
        # grad[i] = -(normalized_size_i - weighted_mean)^2
        # Negated because we want to REDUCE dissimilarity
        deviations = normalized_sizes - weighted_mean
        gradient = -(deviations**2)

        # Shift so the client with smallest deviation has highest gradient
        gradient = gradient - np.min(gradient)

        # Normalize gradient to unit norm
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > eps:
            gradient = gradient / grad_norm
        else:
            gradient = np.zeros(n_clients)

        return gradient

    def get_config(self) -> dict[str, Any]:
        """Get configuration for reproducibility."""
        return {
            "class": self.__class__.__name__,
            "dissimilarity_weight": self.dissimilarity_weight,
            "has_accuracy_fn": self.accuracy_fn is not None,
        }
