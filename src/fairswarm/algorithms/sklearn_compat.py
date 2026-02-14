"""
Scikit-learn compatible wrapper for FairSwarm.

This module provides sklearn-compatible interfaces for FairSwarm,
enabling integration with sklearn's cross-validation, grid search,
and pipeline APIs.

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from fairswarm.algorithms.fairswarm import FairSwarm
from fairswarm.algorithms.result import OptimizationResult
from fairswarm.core.config import FairSwarmConfig
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.fitness.fairness import DemographicFitness

if TYPE_CHECKING:
    from fairswarm.core.client import Client
    from fairswarm.fitness.base import FitnessFunction


class FairSwarmSelector:
    """
    Scikit-learn compatible wrapper for FairSwarm coalition selection.

    This class follows sklearn conventions with fit/transform pattern,
    enabling use with sklearn's model selection and pipeline APIs.

    Parameters
    ----------
    coalition_size : int, default=10
        Number of clients to select in the coalition.

    swarm_size : int, default=30
        Number of particles in the PSO swarm.

    max_iterations : int, default=100
        Maximum optimization iterations.

    inertia : float, default=0.7
        PSO inertia weight (must be in (0, 1)).

    cognitive : float, default=1.5
        PSO cognitive coefficient.

    social : float, default=1.5
        PSO social coefficient.

    fairness_coefficient : float, default=0.5
        Fairness gradient coefficient.

    fairness_weight : float, default=0.3
        Weight of fairness term in fitness function.

    epsilon_fair : float, default=0.05
        Target epsilon for epsilon-fairness.

    adaptive_fairness : bool, default=True
        Whether to adapt fairness weight dynamically.

    random_state : int or None, default=None
        Random seed for reproducibility.

    verbose : bool, default=False
        Print progress information.

    Attributes
    ----------
    result_ : OptimizationResult
        The optimization result from the last fit.

    selected_indices_ : List[int]
        Indices of selected clients from the last fit.

    fitness_ : float
        Final fitness value from the last fit.

    is_fitted_ : bool
        Whether the selector has been fitted.

    Examples
    --------
    >>> from fairswarm.algorithms import FairSwarmSelector
    >>> from sklearn.model_selection import GridSearchCV
    >>>
    >>> selector = FairSwarmSelector(coalition_size=10)
    >>> selector.fit(clients, fitness_fn=fitness)
    >>> selected = selector.transform(clients)
    >>>
    >>> # Use with GridSearchCV
    >>> param_grid = {
    ...     'fairness_weight': [0.1, 0.3, 0.5],
    ...     'swarm_size': [20, 30, 50],
    ... }
    >>> grid_search = GridSearchCV(selector, param_grid)
    """

    def __init__(
        self,
        coalition_size: int = 10,
        swarm_size: int = 30,
        max_iterations: int = 100,
        inertia: float = 0.7,
        cognitive: float = 1.5,
        social: float = 1.5,
        fairness_coefficient: float = 0.5,
        fairness_weight: float = 0.3,
        epsilon_fair: float = 0.05,
        adaptive_fairness: bool = True,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.coalition_size = coalition_size
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.fairness_coefficient = fairness_coefficient
        self.fairness_weight = fairness_weight
        self.epsilon_fair = epsilon_fair
        self.adaptive_fairness = adaptive_fairness
        self.random_state = random_state
        self.verbose = verbose

        # Attributes set during fit
        self.result_: OptimizationResult | None = None
        self.selected_indices_: list[int] | None = None
        self.fitness_: float | None = None
        self.is_fitted_: bool = False

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters of nested objects.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "coalition_size": self.coalition_size,
            "swarm_size": self.swarm_size,
            "max_iterations": self.max_iterations,
            "inertia": self.inertia,
            "cognitive": self.cognitive,
            "social": self.social,
            "fairness_coefficient": self.fairness_coefficient,
            "fairness_weight": self.fairness_weight,
            "epsilon_fair": self.epsilon_fair,
            "adaptive_fairness": self.adaptive_fairness,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **params: Any) -> FairSwarmSelector:
        """
        Set parameters for this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : FairSwarmSelector
            Estimator instance.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def _build_config(self) -> FairSwarmConfig:
        """Build FairSwarmConfig from current parameters."""
        return FairSwarmConfig(
            swarm_size=self.swarm_size,
            max_iterations=self.max_iterations,
            coalition_size=self.coalition_size,
            inertia=self.inertia,
            cognitive=self.cognitive,
            social=self.social,
            fairness_coefficient=self.fairness_coefficient,
            fairness_weight=self.fairness_weight,
            epsilon_fair=self.epsilon_fair,
            adaptive_fairness=self.adaptive_fairness,
            seed=self.random_state,
        )

    def fit(
        self,
        clients: list[Client],
        fitness_fn: FitnessFunction | None = None,
        target_distribution: DemographicDistribution | None = None,
        **fit_params: Any,
    ) -> FairSwarmSelector:
        """
        Fit the selector by running FairSwarm optimization.

        Parameters
        ----------
        clients : List[Client]
            List of federated learning clients.

        fitness_fn : FitnessFunction, optional
            Custom fitness function. If None, uses DemographicFitness
            with the target_distribution.

        target_distribution : DemographicDistribution, optional
            Target demographic distribution for fairness. Required if
            fitness_fn is None.

        **fit_params : dict
            Additional parameters for the optimizer.

        Returns
        -------
        self : FairSwarmSelector
            Fitted selector.

        Raises
        ------
        ValueError
            If neither fitness_fn nor target_distribution is provided.
        """
        # Validate input
        if fitness_fn is None and target_distribution is None:
            raise ValueError(
                "Either fitness_fn or target_distribution must be provided"
            )

        # Build config and optimizer
        config = self._build_config()

        optimizer = FairSwarm(
            clients=clients,
            coalition_size=self.coalition_size,
            config=config,
            target_distribution=target_distribution,
            seed=self.random_state,
        )

        # Build default fitness function if not provided
        if fitness_fn is None and target_distribution is not None:
            fitness_fn = DemographicFitness(
                target_distribution=target_distribution,
                divergence_weight=self.fairness_weight,
            )

        # Run optimization
        assert fitness_fn is not None, (
            "fitness_fn must be set (either provided or built from target_distribution)"
        )
        self.result_ = optimizer.optimize(
            fitness_fn=fitness_fn,
            n_iterations=self.max_iterations,
            verbose=self.verbose,
        )

        # Store results
        self.selected_indices_ = list(self.result_.coalition)
        self.fitness_ = self.result_.fitness
        self.is_fitted_ = True

        return self

    def transform(
        self,
        clients: list[Client],
    ) -> list[Client]:
        """
        Select clients based on the fitted coalition.

        Parameters
        ----------
        clients : List[Client]
            List of all clients.

        Returns
        -------
        selected_clients : List[Client]
            List of selected clients.

        Raises
        ------
        RuntimeError
            If selector has not been fitted.
        """
        if not self.is_fitted_ or self.selected_indices_ is None:
            raise RuntimeError(
                "FairSwarmSelector has not been fitted. Call fit() first."
            )

        return [clients[i] for i in self.selected_indices_ if i < len(clients)]

    def fit_transform(
        self,
        clients: list[Client],
        fitness_fn: FitnessFunction | None = None,
        target_distribution: DemographicDistribution | None = None,
        **fit_params: Any,
    ) -> list[Client]:
        """
        Fit the selector and return selected clients.

        Parameters
        ----------
        clients : List[Client]
            List of federated learning clients.

        fitness_fn : FitnessFunction, optional
            Custom fitness function.

        target_distribution : DemographicDistribution, optional
            Target demographic distribution.

        **fit_params : dict
            Additional parameters for the optimizer.

        Returns
        -------
        selected_clients : List[Client]
            List of selected clients.
        """
        self.fit(
            clients=clients,
            fitness_fn=fitness_fn,
            target_distribution=target_distribution,
            **fit_params,
        )
        return self.transform(clients)

    def get_selection_mask(self, n_clients: int) -> NDArray[np.bool_]:
        """
        Get a boolean mask indicating selected clients.

        Parameters
        ----------
        n_clients : int
            Total number of clients.

        Returns
        -------
        mask : ndarray of shape (n_clients,)
            Boolean mask where True indicates selection.
        """
        if not self.is_fitted_ or self.selected_indices_ is None:
            raise RuntimeError("Selector has not been fitted")

        mask = np.zeros(n_clients, dtype=bool)
        for idx in self.selected_indices_:
            if idx < n_clients:
                mask[idx] = True
        return mask

    def score(
        self,
        clients: list[Client],
        fitness_fn: FitnessFunction | None = None,
        target_distribution: DemographicDistribution | None = None,
    ) -> float:
        """
        Return the fitness score of the selected coalition.

        This method is compatible with sklearn's cross-validation
        and grid search APIs.

        Parameters
        ----------
        clients : List[Client]
            List of all clients.

        fitness_fn : FitnessFunction, optional
            Fitness function for evaluation.

        target_distribution : DemographicDistribution, optional
            Target distribution for default fitness function.

        Returns
        -------
        score : float
            Fitness score of the coalition.
        """
        if not self.is_fitted_ or self.selected_indices_ is None:
            raise RuntimeError("Selector has not been fitted")

        # If we have a stored result, use its fitness
        if self.fitness_ is not None:
            return self.fitness_

        # Otherwise, re-evaluate
        if fitness_fn is None and target_distribution is not None:
            fitness_fn = DemographicFitness(
                target_distribution=target_distribution,
                divergence_weight=self.fairness_weight,
            )

        if fitness_fn is not None:
            result = fitness_fn.evaluate(self.selected_indices_, clients)
            return result.value

        return 0.0

    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self.is_fitted_ else "not fitted"
        return (
            f"FairSwarmSelector(coalition_size={self.coalition_size}, "
            f"swarm_size={self.swarm_size}, status={status})"
        )


__all__ = [
    "FairSwarmSelector",
]
