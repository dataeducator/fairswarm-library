"""
Federated Learning Client representation.

This module defines the Client dataclass representing a participant
in the federated learning coalition (e.g., a hospital).

Research Foundation:
    Corresponds to c_i in C = {c_1, ..., c_n} from Definition 1 in CLAUDE.md.
    Each client has:
    - Local dataset D_i
    - Demographic distribution δ_i ∈ Δ^(k-1)

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fairswarm.types import (
    ClientId,
    Demographics,
    DemographicVector,
    validate_demographic_vector,
)


@dataclass(frozen=True)
class Client:
    """
    Represents a federated learning client (e.g., a hospital).

    Each client has demographic information about their patient population
    and metadata about their dataset. The Client class is immutable (frozen)
    to ensure consistency during optimization.

    Attributes:
        id: Unique identifier for this client
        demographics: Probability distribution over demographic groups.
            Accepts a numpy array or a Demographics object (auto-normalized).
        dataset_size: Number of samples in the local dataset
        num_samples: Alias for dataset_size (if both given, num_samples wins)
        communication_cost: Relative cost to communicate with this client (0-1)
        data_quality: Quality score for this client's data (0-1)
        metadata: Optional additional information

    Mathematical Notation (from CLAUDE.md):
        - id → c_i
        - demographics → δ_i ∈ Δ^(k-1) (probability simplex)
        - dataset_size → |D_i|

    Example:
        >>> from fairswarm import Client
        >>> import numpy as np
        >>>
        >>> hospital = Client(
        ...     id="hospital_atlanta_01",
        ...     demographics=np.array([0.35, 0.45, 0.15, 0.05]),
        ...     dataset_size=5000,
        ...     communication_cost=0.2,
        ...     metadata={"region": "southeast", "ehr_system": "epic"}
        ... )
        >>> print(f"Hospital has {hospital.dataset_size} patients")

    Security Note (Ghosh Framework):
        - Client IDs should not contain PHI
        - Demographics should be aggregated, not individual-level
        - All client data is validated at construction time
    """

    id: ClientId
    demographics: DemographicVector
    dataset_size: int = 1000
    num_samples: int | None = None
    communication_cost: float = 0.5
    data_quality: float = 1.0
    metadata: dict[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        """Validate client data at construction time."""
        # Handle Demographics object input (convert to array)
        if isinstance(self.demographics, Demographics):
            object.__setattr__(self, "demographics", self.demographics.to_array())

        # Handle num_samples alias for dataset_size
        if self.num_samples is not None:
            object.__setattr__(self, "dataset_size", self.num_samples)

        # Validate demographics is a proper probability distribution
        # DemographicDistribution objects self-validate, so skip for those
        from fairswarm.demographics.distribution import DemographicDistribution
        if isinstance(self.demographics, DemographicDistribution):
            pass  # Already validated in DemographicDistribution.__post_init__
        elif not validate_demographic_vector(self.demographics):
            raise ValueError(
                f"Client {self.id}: demographics must be a valid probability "
                f"distribution (non-negative, sum to 1). "
                f"Got sum={np.sum(self.demographics):.4f}"
            )

        # Validate dataset_size is positive
        if self.dataset_size <= 0:
            raise ValueError(
                f"Client {self.id}: dataset_size must be positive, "
                f"got {self.dataset_size}"
            )

        # Validate communication_cost is in [0, 1]
        if not 0 <= self.communication_cost <= 1:
            raise ValueError(
                f"Client {self.id}: communication_cost must be in [0, 1], "
                f"got {self.communication_cost}"
            )

    @property
    def n_demographic_groups(self) -> int:
        """Number of demographic groups tracked by this client."""
        return len(self.demographics)

    def demographic_contribution(self, group_index: int) -> float:
        """
        Get the proportion of this client's population in a demographic group.

        Args:
            group_index: Index of the demographic group

        Returns:
            Proportion in [0, 1]

        Raises:
            IndexError: If group_index is out of range
        """
        if group_index < 0 or group_index >= self.n_demographic_groups:
            raise IndexError(
                f"group_index {group_index} out of range for "
                f"{self.n_demographic_groups} groups"
            )
        return float(self.demographics[group_index])

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        demographic_keys: list[str] | None = None,
    ) -> Client:
        """
        Create a Client from a dictionary.

        This factory method is useful when loading client data from
        configuration files or databases.

        Args:
            data: Dictionary with client attributes
            demographic_keys: Optional list of keys to extract demographics from.
                If None, expects a 'demographics' key with array-like value.

        Returns:
            New Client instance

        Example:
            >>> data = {
            ...     "id": "hospital_01",
            ...     "demographics": [0.6, 0.2, 0.15, 0.05],
            ...     "dataset_size": 3000,
            ... }
            >>> client = Client.from_dict(data)
        """
        # Extract demographics
        if demographic_keys:
            demographics = np.array(
                [data[k] for k in demographic_keys],
                dtype=np.float64,
            )
        else:
            demographics = np.asarray(data["demographics"], dtype=np.float64)

        return cls(
            id=ClientId(data["id"]),
            demographics=demographics,
            dataset_size=data.get("dataset_size", 1000),
            communication_cost=data.get("communication_cost", 0.5),
            metadata=data.get("metadata"),
        )


def create_synthetic_clients(
    n_clients: int,
    n_demographic_groups: int = 4,
    seed: int | None = None,
) -> list[Client]:
    """
    Create synthetic clients for testing and development.

    This function generates clients with random demographic distributions
    following a Dirichlet distribution (which naturally produces valid
    probability distributions).

    Args:
        n_clients: Number of clients to generate
        n_demographic_groups: Number of demographic groups
        seed: Random seed for reproducibility

    Returns:
        List of synthetic Client instances

    Example:
        >>> clients = create_synthetic_clients(20, seed=42)
        >>> len(clients)
        20
        >>> all(c.n_demographic_groups == 4 for c in clients)
        True

    Security Note:
        These are synthetic clients for development only.
        Never use real patient data without proper authorization.
    """
    rng = np.random.default_rng(seed)

    clients = []
    for i in range(n_clients):
        # Dirichlet distribution produces valid probability distributions
        # Alpha > 1 produces more uniform distributions
        # Alpha < 1 produces more concentrated distributions
        alpha = np.ones(n_demographic_groups) * 2.0
        demographics = rng.dirichlet(alpha)

        # Vary dataset sizes (log-normal distribution for realistic variation)
        dataset_size = int(rng.lognormal(mean=7.0, sigma=0.8))  # ~1000-5000
        dataset_size = max(100, min(dataset_size, 50000))

        # Communication cost correlated with dataset size (larger = more cost)
        comm_cost = 0.3 + 0.4 * (dataset_size / 50000) + rng.uniform(-0.1, 0.1)
        comm_cost = max(0.1, min(0.9, comm_cost))

        client = Client(
            id=ClientId(f"synthetic_client_{i:03d}"),
            demographics=demographics,
            dataset_size=dataset_size,
            communication_cost=comm_cost,
            metadata={"synthetic": True, "seed": seed},
        )
        clients.append(client)

    return clients
