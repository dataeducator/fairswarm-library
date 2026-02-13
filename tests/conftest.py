"""
Pytest fixtures for FairSwarm tests.

This module provides reusable fixtures for testing the FairSwarm library,
including synthetic clients, configurations, and demographic distributions.

Security Note:
    All test data is synthetic. Never use real patient data in tests.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from fairswarm.core.client import Client, create_synthetic_clients
from fairswarm.core.config import FairSwarmConfig
from fairswarm.types import ClientId, DemographicVector


# =============================================================================
# Demographic Distribution Fixtures
# =============================================================================


@pytest.fixture
def us_census_demographics() -> DemographicVector:
    """
    Approximate US 2020 Census demographic distribution.

    Groups: [White, Black, Hispanic, Asian, Other]

    Returns:
        Normalized demographic vector
    """
    return np.array([0.576, 0.124, 0.187, 0.061, 0.052], dtype=np.float64)


@pytest.fixture
def uniform_demographics() -> DemographicVector:
    """
    Uniform distribution across 4 demographic groups.

    Returns:
        Array of [0.25, 0.25, 0.25, 0.25]
    """
    return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)


@pytest.fixture
def skewed_demographics() -> DemographicVector:
    """
    Highly skewed demographic distribution for testing fairness.

    Returns:
        Array with 80% in first group
    """
    return np.array([0.80, 0.10, 0.07, 0.03], dtype=np.float64)


# =============================================================================
# Client Fixtures
# =============================================================================


@pytest.fixture
def single_client(uniform_demographics: DemographicVector) -> Client:
    """
    A single client for unit testing.

    Returns:
        Client with uniform demographics
    """
    return Client(
        id=ClientId("test_client_001"),
        demographics=uniform_demographics,
        dataset_size=1000,
        communication_cost=0.5,
        metadata={"test": True},
    )


@pytest.fixture
def small_client_pool() -> List[Client]:
    """
    Small pool of 5 synthetic clients for fast tests.

    Returns:
        List of 5 clients with varied demographics
    """
    return create_synthetic_clients(n_clients=5, seed=42)


@pytest.fixture
def medium_client_pool() -> List[Client]:
    """
    Medium pool of 20 synthetic clients for integration tests.

    Returns:
        List of 20 clients
    """
    return create_synthetic_clients(n_clients=20, seed=42)


@pytest.fixture
def large_client_pool() -> List[Client]:
    """
    Large pool of 100 synthetic clients for stress tests.

    Returns:
        List of 100 clients
    """
    return create_synthetic_clients(n_clients=100, seed=42)


@pytest.fixture
def diverse_client_pool() -> List[Client]:
    """
    Client pool with intentionally diverse demographics.

    Creates clients spanning different demographic compositions
    to test fairness optimization.

    Returns:
        List of 10 clients with varied demographics
    """
    np.random.seed(123)
    clients = []

    # Create clients with different demographic profiles
    profiles = [
        np.array([0.8, 0.1, 0.05, 0.05]),  # Mostly group 1
        np.array([0.1, 0.8, 0.05, 0.05]),  # Mostly group 2
        np.array([0.1, 0.1, 0.7, 0.1]),  # Mostly group 3
        np.array([0.1, 0.1, 0.1, 0.7]),  # Mostly group 4
        np.array([0.25, 0.25, 0.25, 0.25]),  # Uniform
        np.array([0.4, 0.3, 0.2, 0.1]),  # Gradual
        np.array([0.3, 0.3, 0.2, 0.2]),  # Two groups
        np.array([0.5, 0.2, 0.2, 0.1]),  # Half in one
        np.array([0.35, 0.35, 0.15, 0.15]),  # Two equal
        np.array([0.4, 0.2, 0.25, 0.15]),  # Mixed
    ]

    for i, demo in enumerate(profiles):
        client = Client(
            id=ClientId(f"diverse_client_{i:03d}"),
            demographics=demo,
            dataset_size=1000 + i * 100,
            communication_cost=0.3 + i * 0.05,
        )
        clients.append(client)

    return clients


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> FairSwarmConfig:
    """
    Default FairSwarmConfig for testing.

    Returns:
        Configuration with default values
    """
    return FairSwarmConfig(seed=42)


@pytest.fixture
def fast_config() -> FairSwarmConfig:
    """
    Fast configuration for quick tests.

    Returns:
        Configuration optimized for speed
    """
    return FairSwarmConfig(
        swarm_size=10,
        max_iterations=20,
        patience=3,
        seed=42,
    )


@pytest.fixture
def convergent_config() -> FairSwarmConfig:
    """
    Configuration guaranteed to satisfy Theorem 1 convergence.

    Returns:
        Configuration with ω + (c₁+c₂)/2 < 1
    """
    return FairSwarmConfig(
        inertia=0.4,
        cognitive=0.5,
        social=0.5,
        seed=42,
    )


@pytest.fixture
def fair_config() -> FairSwarmConfig:
    """
    Configuration emphasizing fairness.

    Returns:
        Configuration with high fairness weight
    """
    return FairSwarmConfig(
        fairness_coefficient=0.8,
        fairness_weight=0.5,
        adaptive_fairness=True,
        epsilon_fair=0.03,  # Strict fairness target
        seed=42,
    )


# =============================================================================
# Random State Fixtures
# =============================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """
    Seeded random number generator for reproducible tests.

    Returns:
        NumPy Generator with fixed seed
    """
    return np.random.default_rng(42)


# =============================================================================
# Hypothesis Configuration
# =============================================================================

# Configure Hypothesis settings for the test suite
from hypothesis import settings, Phase, Verbosity

# Register a profile for CI (faster, fewer examples)
settings.register_profile(
    "ci",
    max_examples=20,
    deadline=None,
    suppress_health_check=[],
)

# Register a profile for development (more thorough)
settings.register_profile(
    "dev",
    max_examples=50,
    deadline=None,
)

# Register a profile for thorough testing
settings.register_profile(
    "thorough",
    max_examples=200,
    deadline=None,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
)


# =============================================================================
# Hypothesis Strategies (for property-based testing)
# =============================================================================


def generate_random_clients(
    n_clients: int,
    n_demographic_groups: int = 4,
    seed: int = 42,
) -> List[Client]:
    """
    Generate random clients for hypothesis testing.

    This is a standalone function (not a fixture) for use with
    Hypothesis strategies.

    Args:
        n_clients: Number of clients to generate
        n_demographic_groups: Number of demographic groups
        seed: Random seed

    Returns:
        List of synthetic clients
    """
    return create_synthetic_clients(
        n_clients=n_clients,
        n_demographic_groups=n_demographic_groups,
        seed=seed,
    )


def generate_target_distribution(
    n_groups: int = 4,
    seed: int = 42,
) -> DemographicVector:
    """
    Generate a random target demographic distribution.

    Args:
        n_groups: Number of demographic groups
        seed: Random seed

    Returns:
        Valid demographic distribution
    """
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(np.ones(n_groups) * 2)
    return raw.astype(np.float64)


# =============================================================================
# Pytest Markers for Slow Tests
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "theorem1: tests for Theorem 1 (Convergence)"
    )
    config.addinivalue_line(
        "markers", "theorem2: tests for Theorem 2 (ε-Fairness)"
    )
    config.addinivalue_line(
        "markers", "theorem3: tests for Theorem 3 (Approximation)"
    )
