"""
Experiment: Real Federated Learning with FairSwarm Coalition Selection.

This script implements an end-to-end federated learning experiment with
actual model training (logistic regression via sklearn), replacing mock
fitness functions with a FederatedFitness class that trains and evaluates
coalitions in a realistic FL simulation.

Experimental Setup:
    1. Generate synthetic classification data partitioned across n clients
       with non-IID demographic distributions
    2. Implement FederatedFitness (extends FitnessFunction) that performs:
       - Local training on each client's data partition
       - FedAvg-style weighted model aggregation
       - Validation AUC-ROC on a held-out global test set
    3. Run FairSwarm with FederatedFitness for coalition selection
    4. Compare against baselines: random, greedy (by dataset size),
       all-clients FedAvg, standard PSO (c3=0)
    5. Sweep over n_clients in [20, 50, 100] and k in [4, 8]
    6. Report: validation AUC-ROC, demographic divergence,
       convergence iterations, wall-clock time
    7. Output results as JSON to results/real_fl/

Mathematical Foundation:
    Fitness(S) = w_1 * AUC(S) - w_2 * DemDiv(S) - w_3 * Cost(S)
    where AUC(S) is the actual validation AUC-ROC from federated training
    on coalition S, matching Algorithm 1 in the paper.

Author: Tenicka Norwood

Usage:
    python experiments/run_real_fl.py
    python experiments/run_real_fl.py --parallel
    python experiments/run_real_fl.py --n_trials 10 --seed 42
    python experiments/run_real_fl.py --n_clients 50 --k 4 --parallel
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fairswarm import Client, FairSwarm, FairSwarmConfig
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.demographics.divergence import kl_divergence
from fairswarm.fitness.base import FitnessFunction, FitnessResult
from fairswarm.fitness.equity import equalized_odds_gap
from fairswarm.fitness.fairness import (
    compute_coalition_demographics,
    compute_fairness_gradient,
)
from fairswarm.types import ClientId, Coalition

from baselines.fair_dpfl_scs import FairDPFL_SCS, FairDPFLConfig
from baselines.fairfed import FairFedBaseline, FairFedConfig
from baselines.grey_wolf_optimizer import GreyWolfOptimizer
from baselines.qffl import QFFLBaseline, QFFLConfig
from statistics_utils import compare_means, get_git_hash, mean_ci, statistical_summary

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)



# Parallelization Utilities



def get_n_workers() -> int:
    """Get optimal number of workers (leave 2 cores for system)."""
    return max(2, (os.cpu_count() or 4) - 2)



# Data Generation and Partitioning



@dataclass(frozen=True)
class ClientData:
    """
    Holds a single client's local data partition.

    Attributes:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (n_samples,)
        demographic_dist: Probability distribution over k demographic groups
        client_id: Unique string identifier
    """

    X_train: NDArray[np.float64]
    y_train: NDArray[np.int64]
    demographic_dist: NDArray[np.float64]
    client_id: str


@dataclass
class FederatedDataset:
    """
    Complete federated dataset with global test set and per-client partitions.

    Attributes:
        client_data: Per-client training data
        X_test: Global held-out test features
        y_test: Global held-out test labels
        groups_test: Demographic group assignment for each test sample
        n_features: Dimensionality of feature space
        n_classes: Number of target classes
        target_distribution: Target demographic distribution
    """

    client_data: List[ClientData]
    X_test: NDArray[np.float64]
    y_test: NDArray[np.int64]
    groups_test: NDArray[np.int64]
    n_features: int
    n_classes: int
    target_distribution: NDArray[np.float64]


def generate_federated_dataset(
    n_clients: int,
    n_demographic_groups: int,
    n_features: int = 20,
    n_informative: int = 12,
    n_samples_total: int = 20000,
    test_fraction: float = 0.2,
    non_iid_alpha: float = 0.5,
    seed: int = 42,
) -> FederatedDataset:
    """
    Generate a synthetic federated classification dataset.

    Creates a binary classification task and partitions it across n_clients
    with non-IID demographic distributions using a Dirichlet allocation
    scheme. Each client gets data that is skewed toward certain demographic
    groups, simulating realistic hospital-level demographic variation.

    The Dirichlet parameter alpha controls non-IID-ness:
        - alpha -> 0: highly non-IID (each client has mostly one group)
        - alpha -> inf: IID (each client mirrors the population)

    Args:
        n_clients: Number of federated learning clients
        n_demographic_groups: Number of demographic groups (k)
        n_features: Total feature dimensionality
        n_informative: Number of informative features
        n_samples_total: Total samples before partitioning
        test_fraction: Fraction reserved for global test set
        non_iid_alpha: Dirichlet concentration parameter
        seed: Random seed for reproducibility

    Returns:
        FederatedDataset with per-client partitions and global test set
    """
    rng = np.random.default_rng(seed)

    # Generate base classification dataset
    X, y = make_classification(
        n_samples=n_samples_total,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(0, n_features - n_informative - 2),
        n_clusters_per_class=2,
        flip_y=0.05,
        class_sep=1.0,
        random_state=seed,
    )

    # Assign each sample to a demographic group
    # Use feature-correlated assignment for realism: certain feature
    # patterns correlate with demographic group membership
    group_features = X[:, :n_demographic_groups]
    group_logits = group_features + rng.normal(0, 0.3, group_features.shape)
    group_probs = np.exp(group_logits) / np.exp(group_logits).sum(axis=1, keepdims=True)
    sample_groups = np.array(
        [rng.choice(n_demographic_groups, p=p) for p in group_probs]
    )

    # Global train/test split (stratified by label)
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X,
        y,
        sample_groups,
        test_size=test_fraction,
        random_state=seed,
        stratify=y,
    )

    # Standardize features using training data statistics
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Partition training data across clients using Dirichlet allocation
    # Each client gets a non-IID mixture of demographic groups
    n_train = len(y_train)

    # Dirichlet allocation: each client's share of each demographic group
    client_group_proportions = rng.dirichlet(
        alpha=np.ones(n_clients) * non_iid_alpha,
        size=n_demographic_groups,
    )  # shape: (k, n_clients)

    # For each demographic group, distribute samples across clients
    client_indices: List[List[int]] = [[] for _ in range(n_clients)]

    for group_id in range(n_demographic_groups):
        group_mask = groups_train == group_id
        group_indices = np.where(group_mask)[0]

        if len(group_indices) == 0:
            continue

        # Allocate group samples to clients according to Dirichlet proportions
        proportions = client_group_proportions[group_id]
        proportions = proportions / proportions.sum()

        # Multinomial assignment
        counts = rng.multinomial(len(group_indices), proportions)

        # Shuffle group indices before distributing
        shuffled = rng.permutation(group_indices)
        offset = 0
        for client_id in range(n_clients):
            n_assign = counts[client_id]
            client_indices[client_id].extend(
                shuffled[offset : offset + n_assign].tolist()
            )
            offset += n_assign

    # Ensure every client has at least min_samples data points
    min_samples = 20
    for client_id in range(n_clients):
        if len(client_indices[client_id]) < min_samples:
            # Supplement with random samples from the training set
            shortfall = min_samples - len(client_indices[client_id])
            extra = rng.choice(n_train, size=shortfall, replace=False).tolist()
            client_indices[client_id].extend(extra)

    # Build ClientData objects
    client_data_list: List[ClientData] = []
    for client_id in range(n_clients):
        idx = np.array(client_indices[client_id])
        X_local = X_train[idx]
        y_local = y_train[idx]

        # Compute actual demographic distribution from local data
        local_groups = groups_train[idx]
        demo_counts = np.bincount(local_groups, minlength=n_demographic_groups).astype(
            np.float64
        )
        demo_dist = demo_counts / demo_counts.sum()

        client_data_list.append(
            ClientData(
                X_train=X_local,
                y_train=y_local,
                demographic_dist=demo_dist,
                client_id=f"client_{client_id:03d}",
            )
        )

    # Target distribution: uniform across demographic groups
    # (representing an ideal balanced population)
    target_distribution = np.ones(n_demographic_groups) / n_demographic_groups

    return FederatedDataset(
        client_data=client_data_list,
        X_test=X_test,
        y_test=y_test,
        groups_test=groups_test,
        n_features=n_features,
        n_classes=2,
        target_distribution=target_distribution,
    )


def build_fairswarm_clients(
    fed_dataset: FederatedDataset,
) -> List[Client]:
    """
    Build fairswarm Client objects from the federated dataset.

    Each Client stores the demographic distribution as its demographics
    attribute, and the local dataset size. The actual training data is
    stored separately in the FederatedDataset.

    Args:
        fed_dataset: The federated dataset with per-client partitions

    Returns:
        List of Client objects compatible with the fairswarm library
    """
    clients: List[Client] = []
    for cd in fed_dataset.client_data:
        # Communication cost proportional to dataset size (normalized)
        max_size = max(len(d.y_train) for d in fed_dataset.client_data)
        comm_cost = min(0.9, max(0.1, len(cd.y_train) / max_size))

        client = Client(
            id=ClientId(cd.client_id),
            demographics=cd.demographic_dist,
            dataset_size=len(cd.y_train),
            communication_cost=comm_cost,
        )
        clients.append(client)

    return clients



# Federated Training Utilities



def train_local_model(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    n_features: int,
    global_weights: Optional[NDArray[np.float64]] = None,
    global_intercept: Optional[NDArray[np.float64]] = None,
    local_epochs: int = 5,
    learning_rate: float = 0.01,
    seed: int = 42,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """
    Train a logistic regression model on local data using warm-started SGD.

    Implements the local training step of FedAvg: each client runs
    several epochs of SGD starting from the global model parameters.

    Args:
        X: Local training features
        y: Local training labels
        n_features: Feature dimensionality (for initialization)
        global_weights: Current global model weights (warm start)
        global_intercept: Current global model intercept (warm start)
        local_epochs: Number of local SGD epochs
        learning_rate: Local learning rate
        seed: Random seed

    Returns:
        Tuple of (updated_weights, updated_intercept, n_samples)
    """
    # Sanitize inputs: clip extreme values to prevent weight explosion
    feature_clip = 1e6
    if np.any(~np.isfinite(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=feature_clip, neginf=-feature_clip)
    X = np.clip(X, -feature_clip, feature_clip)

    # Use SGDClassifier for epoch-based training with warm start
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="constant",
        eta0=learning_rate,
        max_iter=local_epochs,
        random_state=seed,
        warm_start=True,
        shuffle=True,
    )

    # Initialize with global model parameters if provided
    if global_weights is not None and global_intercept is not None:
        # Sanitize global weights: NaN/Inf from a corrupted aggregation
        # round must not propagate into SGD (which would throw ValueError)
        gw = global_weights.copy()
        gi = global_intercept.copy()
        if not (np.all(np.isfinite(gw)) and np.all(np.isfinite(gi))):
            gw = np.nan_to_num(gw, nan=0.0, posinf=0.0, neginf=0.0)
            gi = np.nan_to_num(gi, nan=0.0, posinf=0.0, neginf=0.0)

        # Partial fit initializes the model structure
        model.partial_fit(X[:1], y[:1], classes=np.array([0, 1]))
        model.coef_ = gw.reshape(1, -1)
        model.intercept_ = gi.reshape(-1)

    # Handle edge case: single-class local data
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        # Cannot train meaningfully; return global params or zeros
        if global_weights is not None:
            return global_weights.copy(), global_intercept.copy(), len(y)
        return np.zeros(n_features), np.zeros(1), len(y)

    # Train locally
    model.partial_fit(X, y, classes=np.array([0, 1]))

    weights = model.coef_.flatten()
    intercept = model.intercept_.flatten()

    # Guard: if training produced NaN/Inf, fall back to global or zeros
    if np.any(~np.isfinite(weights)) or np.any(~np.isfinite(intercept)):
        if global_weights is not None and np.all(np.isfinite(global_weights)):
            return global_weights.copy(), global_intercept.copy(), len(y)
        return np.zeros(n_features), np.zeros(1), len(y)

    return weights, intercept, len(y)


def federated_aggregate(
    client_updates: List[Tuple[NDArray[np.float64], NDArray[np.float64], int]],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    FedAvg-style weighted aggregation of client model updates.

    Implements McMahan et al. (2017): global model is the weighted
    average of local models, weighted by each client's dataset size.

    Args:
        client_updates: List of (weights, intercept, n_samples) from each client

    Returns:
        Tuple of (aggregated_weights, aggregated_intercept)

    Raises:
        ValueError: If no updates provided
    """
    if not client_updates:
        raise ValueError("No client updates to aggregate")

    # Filter out updates with NaN/Inf weights (defensive: one bad hospital
    # must not poison the global model)
    clean_updates = [
        (w, b, n) for w, b, n in client_updates
        if np.all(np.isfinite(w)) and np.all(np.isfinite(b))
    ]
    if not clean_updates:
        raise ValueError(
            "No valid client updates to aggregate "
            f"({len(client_updates)} updates contained NaN/Inf)"
        )

    total_samples = sum(n for _, _, n in clean_updates)
    if total_samples == 0:
        raise ValueError("Total sample count is zero")

    # Weighted average by dataset size
    agg_weights = np.zeros_like(clean_updates[0][0])
    agg_intercept = np.zeros_like(clean_updates[0][1])

    for weights, intercept, n_samples in clean_updates:
        proportion = n_samples / total_samples
        agg_weights += proportion * weights
        agg_intercept += proportion * intercept

    return agg_weights, agg_intercept


def evaluate_global_model(
    weights: NDArray[np.float64],
    intercept: NDArray[np.float64],
    X_test: NDArray[np.float64],
    y_test: NDArray[np.int64],
) -> float:
    """
    Evaluate a global model on the held-out test set using AUC-ROC.

    Args:
        weights: Model weight vector
        intercept: Model intercept
        X_test: Test features
        y_test: Test labels

    Returns:
        AUC-ROC score in [0, 1]. Returns 0.5 on failure.
    """
    try:
        # Compute predicted probabilities via sigmoid
        logits = X_test @ weights.reshape(-1, 1) + intercept[0]
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        probs = probs.flatten()

        # Ensure probabilities are valid
        probs = np.clip(probs, 1e-8, 1.0 - 1e-8)

        if len(np.unique(y_test)) < 2:
            return 0.5

        return float(roc_auc_score(y_test, probs))
    except Exception:
        return 0.5



# FederatedFitness: Extends FitnessFunction with Real Training



class FederatedFitness(FitnessFunction):
    """
    Fitness function that performs actual federated training and evaluation.

    This is the core contribution of this experiment: replacing mock fitness
    with a real FL training loop. For a given coalition S, the fitness is:

        Fitness(S) = w_1 * AUC(S) - w_2 * DemDiv(S) - w_3 * Cost(S)

    where AUC(S) is obtained by:
        1. Each client in S trains locally from the current global model
        2. Local updates are aggregated via FedAvg weighted averaging
        3. The aggregated model is evaluated on the global test set

    This matches the Fitness subroutine in Algorithm 1 of the paper.

    Attributes:
        fed_dataset: The complete federated dataset
        target_distribution: Target demographic distribution
        n_fl_rounds: Number of FL rounds per fitness evaluation
        local_epochs: Local training epochs per round
        weight_accuracy: w_1 in fitness function
        weight_fairness: w_2 in fitness function
        weight_cost: w_3 in fitness function

    Algorithm Reference:
        Algorithm 1, Fitness subroutine:
            Fitness(S) = w_1 * acc - w_2 * div - w_3 * cost

    Example:
        >>> fitness = FederatedFitness(
        ...     fed_dataset=dataset,
        ...     target_distribution=target,
        ...     n_fl_rounds=3,
        ... )
        >>> result = fitness.evaluate(coalition=[0, 2, 5], clients=clients)
        >>> print(f"AUC-ROC: {result.components['accuracy']:.4f}")
    """

    def __init__(
        self,
        fed_dataset: FederatedDataset,
        target_distribution: DemographicDistribution,
        n_fl_rounds: int = 3,
        local_epochs: int = 5,
        learning_rate: float = 0.01,
        weight_accuracy: float = 0.5,
        weight_fairness: float = 0.3,
        weight_cost: float = 0.2,
        weight_eqodds: float = 0.0,
        seed: int = 42,
    ):
        self.fed_dataset = fed_dataset
        self.target_distribution = target_distribution
        self.n_fl_rounds = n_fl_rounds
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.weight_accuracy = weight_accuracy
        self.weight_fairness = weight_fairness
        self.weight_cost = weight_cost
        self.weight_eqodds = weight_eqodds
        self.seed = seed

        # Cache: coalition tuple -> FitnessResult
        # Avoids redundant training for the same coalition within one PSO run
        self._cache: Dict[Tuple[int, ...], FitnessResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Track per-group performance for outcome-aware gradient (Item 3)
        self._last_group_performance: Optional[NDArray[np.float64]] = None

    def evaluate(
        self,
        coalition: Coalition,
        clients: List[Client],
    ) -> FitnessResult:
        """
        Evaluate a coalition by running actual federated training.

        Performs n_fl_rounds of federated learning on the coalition,
        then evaluates the aggregated model on the global test set.

        Args:
            coalition: List of client indices in the coalition
            clients: List of all available clients

        Returns:
            FitnessResult with AUC-ROC accuracy, demographic divergence,
            and communication cost components

        Algorithm Reference:
            Implements Algorithm 1 Fitness subroutine:
                acc = EvaluateCoalitionAccuracy(S)
                div = DemographicDivergence(S, delta_star)
                cost = CommunicationCost(S)
                Return w_1 * acc - w_2 * div - w_3 * cost
        """
        if not coalition:
            return FitnessResult(
                value=float("-inf"),
                components={"accuracy": 0.0, "divergence": float("inf"), "cost": 0.0},
                coalition=coalition,
                metadata={"error": "Empty coalition"},
            )

        # Check cache
        cache_key = tuple(sorted(coalition))
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        self._cache_misses += 1

        # --- Federated Training ---
        n_features = self.fed_dataset.n_features
        global_weights: Optional[NDArray[np.float64]] = None
        global_intercept: Optional[NDArray[np.float64]] = None

        for fl_round in range(self.n_fl_rounds):
            client_updates: List[
                Tuple[NDArray[np.float64], NDArray[np.float64], int]
            ] = []

            for client_idx in coalition:
                if client_idx < 0 or client_idx >= len(self.fed_dataset.client_data):
                    continue

                cd = self.fed_dataset.client_data[client_idx]

                local_weights, local_intercept, n_local = train_local_model(
                    X=cd.X_train,
                    y=cd.y_train,
                    n_features=n_features,
                    global_weights=global_weights,
                    global_intercept=global_intercept,
                    local_epochs=self.local_epochs,
                    learning_rate=self.learning_rate,
                    seed=self.seed + fl_round * 1000 + client_idx,
                )
                client_updates.append((local_weights, local_intercept, n_local))

            if not client_updates:
                return FitnessResult(
                    value=float("-inf"),
                    components={
                        "accuracy": 0.0,
                        "divergence": float("inf"),
                        "cost": 0.0,
                    },
                    coalition=coalition,
                    metadata={"error": "No valid client updates"},
                )

            # FedAvg aggregation
            global_weights, global_intercept = federated_aggregate(client_updates)

        # --- Evaluation ---
        # Accuracy: AUC-ROC on global test set
        accuracy = evaluate_global_model(
            weights=global_weights,
            intercept=global_intercept,
            X_test=self.fed_dataset.X_test,
            y_test=self.fed_dataset.y_test,
        )

        # Equalized odds: per-group TPR/FPR on test set
        # CIA-Integrity: group labels are metadata, not used in training
        eq_odds = 0.0
        group_tpr_list: List[float] = []
        group_fpr_list: List[float] = []
        group_auc_list: List[float] = []
        try:
            eq_odds, group_tpr_list, group_fpr_list, group_auc_list = (
                self._compute_per_group_metrics(global_weights, global_intercept)
            )
        except Exception:
            pass  # Fall back to 0.0 if groups too small

        # Store for outcome-aware gradient (Item 3)
        if group_auc_list:
            self._last_group_performance = np.array(group_auc_list)

        # Demographic divergence: DemDiv(S) = D_KL(delta_S || delta_star)
        coalition_demo = compute_coalition_demographics(coalition, clients)
        target = self.target_distribution.as_array()
        divergence = kl_divergence(coalition_demo, target)

        # Communication cost: normalized total data transferred
        total_data = sum(
            clients[i].dataset_size for i in coalition if 0 <= i < len(clients)
        )
        max_possible = sum(c.dataset_size for c in clients)
        cost = total_data / max_possible if max_possible > 0 else 0.0

        # Combined fitness: w_1*acc - w_2*div - w_3*cost - w_4*eqodds
        fitness_value = (
            self.weight_accuracy * accuracy
            - self.weight_fairness * divergence
            - self.weight_cost * cost
            - self.weight_eqodds * eq_odds
        )

        result = FitnessResult(
            value=fitness_value,
            components={
                "accuracy": accuracy,
                "divergence": divergence,
                "divergence_penalty": -self.weight_fairness * divergence,
                "equalized_odds_gap": eq_odds,
                "eqodds_penalty": -self.weight_eqodds * eq_odds,
                "cost": cost,
                "cost_penalty": -self.weight_cost * cost,
            },
            coalition=coalition,
            metadata={
                "auc_roc": accuracy,
                "equalized_odds_gap": eq_odds,
                "group_tpr": group_tpr_list,
                "group_fpr": group_fpr_list,
                "group_auc": group_auc_list,
                "n_fl_rounds": self.n_fl_rounds,
                "coalition_demographics": coalition_demo.tolist(),
                "target_demographics": target.tolist(),
                "y_pred_proba": self._compute_probs(global_weights, global_intercept) if hasattr(self, '_save_roc_data') else None,
                "y_true": self.fed_dataset.y_test.tolist() if hasattr(self, '_save_roc_data') else None,
                "groups_test": self.fed_dataset.groups_test.tolist() if hasattr(self, '_save_roc_data') else None,
            },
        )

        # Update cache
        self._cache[cache_key] = result
        return result

    def _compute_probs(
        self,
        weights: NDArray[np.float64],
        intercept: NDArray[np.float64],
    ) -> List[float]:
        """Compute prediction probabilities for ROC curve generation."""
        X_test = self.fed_dataset.X_test
        logits = X_test @ weights.reshape(-1, 1) + intercept[0]
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500))).flatten()
        return probs.tolist()

    def _compute_per_group_metrics(
        self,
        weights: NDArray[np.float64],
        intercept: NDArray[np.float64],
    ) -> Tuple[float, List[float], List[float], List[float]]:
        """
        Compute per-demographic-group TPR, FPR, and AUC on the test set.

        CIA-Integrity: group labels are test-set metadata only, never
        used during model training. This prevents information leakage.

        Returns:
            Tuple of (equalized_odds_gap, group_tpr_list, group_fpr_list, group_auc_list)

        Raises:
            ValueError: If fewer than 2 groups have sufficient samples
        """
        X_test = self.fed_dataset.X_test
        y_test = self.fed_dataset.y_test
        groups_test = self.fed_dataset.groups_test

        # Predicted probabilities
        logits = X_test @ weights.reshape(-1, 1) + intercept[0]
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500))).flatten()
        preds = (probs >= 0.5).astype(int)

        unique_groups = np.unique(groups_test)
        group_tpr_list: List[float] = []
        group_fpr_list: List[float] = []
        group_auc_list: List[float] = []

        for g in unique_groups:
            mask = groups_test == g
            y_g = y_test[mask]
            preds_g = preds[mask]
            probs_g = probs[mask]

            # Need both classes present for TPR/FPR
            if len(np.unique(y_g)) < 2 or np.sum(mask) < 10:
                continue

            # TPR = TP / (TP + FN)
            positives = y_g == 1
            tp = np.sum((preds_g == 1) & positives)
            fn = np.sum((preds_g == 0) & positives)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # FPR = FP / (FP + TN)
            negatives = y_g == 0
            fp = np.sum((preds_g == 1) & negatives)
            tn = np.sum((preds_g == 0) & negatives)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            # Per-group AUC
            try:
                g_auc = float(roc_auc_score(y_g, probs_g))
            except Exception:
                g_auc = 0.5

            group_tpr_list.append(float(tpr))
            group_fpr_list.append(float(fpr))
            group_auc_list.append(g_auc)

        if len(group_tpr_list) < 2:
            return 0.0, group_tpr_list, group_fpr_list, group_auc_list

        # Equalized odds gap: max TPR gap + max FPR gap
        tpr_arr = np.array(group_tpr_list)
        fpr_arr = np.array(group_fpr_list)
        eq_odds = equalized_odds_gap(tpr_arr, fpr_arr)

        return float(eq_odds), group_tpr_list, group_fpr_list, group_auc_list

    def compute_gradient(
        self,
        position: NDArray[np.float64],
        clients: List[Client],
        coalition_size: int,
    ) -> NDArray[np.float64]:
        """
        Compute fairness gradient for velocity update.

        Combines demographic fairness gradient with outcome-aware
        performance correction when group performance data is available.

        Args:
            position: Current particle position
            clients: List of all clients
            coalition_size: Target coalition size

        Returns:
            Fairness gradient vector

        Algorithm Reference:
            v_fairness = c_3 * nabla_fair
        """
        result = compute_fairness_gradient(
            position=position,
            clients=clients,
            target_distribution=self.target_distribution,
            coalition_size=coalition_size,
            group_performance=self._last_group_performance,
        )
        return result.gradient

    def clear_cache(self) -> None:
        """Clear the evaluation cache (call between independent experiments)."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cache_stats(self) -> Dict[str, int]:
        """Return cache hit/miss statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for reproducibility."""
        return {
            "class": self.__class__.__name__,
            "n_fl_rounds": self.n_fl_rounds,
            "local_epochs": self.local_epochs,
            "learning_rate": self.learning_rate,
            "weight_accuracy": self.weight_accuracy,
            "weight_fairness": self.weight_fairness,
            "weight_cost": self.weight_cost,
            "weight_eqodds": self.weight_eqodds,
            "seed": self.seed,
        }



# Experiment Configuration



@dataclass
class RealFLExperimentConfig:
    """
    Configuration for the real federated learning experiment.

    Attributes:
        n_clients_values: Client counts to sweep over
        k_values: Demographic group counts to sweep over
        coalition_fraction: Fraction of clients selected per coalition
        n_fl_rounds: FL training rounds per fitness evaluation
        local_epochs: Local SGD epochs per FL round
        n_trials: Independent trials per configuration
        n_iterations: PSO iterations for FairSwarm
        swarm_size: Number of particles in the swarm
        non_iid_alpha: Dirichlet parameter for non-IID partitioning
        n_samples_total: Total dataset size
        output_dir: Directory for JSON results
        seed: Master random seed
    """

    n_clients_values: List[int] = field(default_factory=lambda: [20, 50, 100])
    k_values: List[int] = field(default_factory=lambda: [4, 8])
    coalition_fraction: float = 0.3
    n_fl_rounds: int = 3
    local_epochs: int = 5
    learning_rate: float = 0.01
    n_trials: int = 5
    n_iterations: int = 50
    swarm_size: int = 20
    non_iid_alpha: float = 0.5  # Current alpha (set per-scenario when sweeping)
    non_iid_alpha_values: List[float] = field(default_factory=lambda: [0.5])
    n_samples_total: int = 20000
    n_features: int = 20
    output_dir: str = "results/real_fl"
    seed: int = 42



# Single Trial Execution



@dataclass
class TrialResult:
    """Result from a single trial of one algorithm on one configuration."""

    algorithm: str
    n_clients: int
    k: int
    coalition_size: int
    trial_idx: int

    # Primary metrics
    auc_roc: float
    demographic_divergence: float
    equalized_odds_gap: float
    convergence_iterations: int
    wall_clock_seconds: float

    # Coalition details
    coalition: List[int]
    coalition_demographics: List[float]
    target_demographics: List[float]

    # Per-group metrics
    group_tpr: List[float] = field(default_factory=list)
    group_fpr: List[float] = field(default_factory=list)
    group_auc: List[float] = field(default_factory=list)

    # Additional
    fitness: float = 0.0
    non_iid_alpha: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


def run_fairswarm_trial(
    clients: List[Client],
    fed_dataset: FederatedDataset,
    target_dist: DemographicDistribution,
    coalition_size: int,
    config: RealFLExperimentConfig,
    fairness_coefficient: float = 0.5,
    trial_seed: int = 42,
    label: str = "FairSwarm",
) -> TrialResult:
    """
    Run a single FairSwarm trial with real federated training.

    Args:
        clients: List of fairswarm Client objects
        fed_dataset: Federated dataset with client partitions
        target_dist: Target demographic distribution
        coalition_size: Number of clients to select
        config: Experiment configuration
        fairness_coefficient: c_3 value (0 = standard PSO, >0 = FairSwarm)
        trial_seed: Random seed for this trial
        label: Algorithm label for result tracking

    Returns:
        TrialResult with metrics from this trial
    """
    start_time = time.time()

    # Build fitness function with real training
    fitness_fn = FederatedFitness(
        fed_dataset=fed_dataset,
        target_distribution=target_dist,
        n_fl_rounds=config.n_fl_rounds,
        local_epochs=config.local_epochs,
        learning_rate=config.learning_rate,
        seed=trial_seed,
    )

    # Configure FairSwarm
    fs_config = FairSwarmConfig(
        swarm_size=config.swarm_size,
        max_iterations=config.n_iterations,
        coalition_size=coalition_size,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
        fairness_coefficient=fairness_coefficient,
        fairness_weight=0.3,
        adaptive_fairness=(fairness_coefficient > 0),
        velocity_max=4.0,
        convergence_threshold=1e-6,
        patience=10,
    )

    optimizer = FairSwarm(
        clients=clients,
        coalition_size=coalition_size,
        config=fs_config,
        target_distribution=target_dist,
        seed=trial_seed,
    )

    result = optimizer.optimize(
        fitness_fn=fitness_fn,
        n_iterations=config.n_iterations,
        verbose=False,
    )

    wall_time = time.time() - start_time

    # Extract metrics
    coalition = result.coalition
    final_eval = fitness_fn.evaluate(coalition, clients)
    auc_roc = final_eval.components.get("accuracy", 0.5)
    divergence = final_eval.components.get("divergence", float("inf"))
    eq_odds = final_eval.components.get("equalized_odds_gap", 0.0)

    coalition_demo = compute_coalition_demographics(coalition, clients)
    convergence_iter = (
        result.convergence.convergence_iteration
        if result.convergence and result.convergence.convergence_iteration is not None
        else config.n_iterations
    )

    return TrialResult(
        algorithm=label,
        n_clients=len(clients),
        k=len(target_dist),
        coalition_size=coalition_size,
        trial_idx=0,
        auc_roc=auc_roc,
        demographic_divergence=divergence,
        equalized_odds_gap=eq_odds,
        convergence_iterations=convergence_iter,
        wall_clock_seconds=wall_time,
        coalition=coalition,
        coalition_demographics=coalition_demo.tolist(),
        target_demographics=target_dist.as_array().tolist(),
        group_tpr=final_eval.metadata.get("group_tpr", []),
        group_fpr=final_eval.metadata.get("group_fpr", []),
        group_auc=final_eval.metadata.get("group_auc", []),
        fitness=result.fitness,
        metadata={
            "cache_stats": fitness_fn.get_cache_stats(),
            "converged": result.convergence.converged if result.convergence else False,
        },
    )


def run_random_baseline_trial(
    clients: List[Client],
    fed_dataset: FederatedDataset,
    target_dist: DemographicDistribution,
    coalition_size: int,
    config: RealFLExperimentConfig,
    n_random_draws: int = 50,
    trial_seed: int = 42,
) -> TrialResult:
    """
    Run random selection baseline with real federated training.

    Draws n_random_draws random coalitions, trains each, and
    keeps the best by fitness value.

    Args:
        clients: List of fairswarm Client objects
        fed_dataset: Federated dataset
        target_dist: Target demographic distribution
        coalition_size: Number of clients to select
        config: Experiment configuration
        n_random_draws: Number of random coalitions to try
        trial_seed: Random seed

    Returns:
        TrialResult with best random coalition's metrics
    """
    start_time = time.time()
    rng = np.random.default_rng(trial_seed)
    n_clients = len(clients)

    fitness_fn = FederatedFitness(
        fed_dataset=fed_dataset,
        target_distribution=target_dist,
        n_fl_rounds=config.n_fl_rounds,
        local_epochs=config.local_epochs,
        learning_rate=config.learning_rate,
        seed=trial_seed,
    )

    best_fitness = float("-inf")
    best_coalition: List[int] = []
    best_result: Optional[FitnessResult] = None

    for _ in range(n_random_draws):
        coalition = rng.choice(n_clients, size=coalition_size, replace=False).tolist()
        result = fitness_fn.evaluate(coalition, clients)

        if result.value > best_fitness:
            best_fitness = result.value
            best_coalition = coalition
            best_result = result

    wall_time = time.time() - start_time

    auc_roc = best_result.components.get("accuracy", 0.5) if best_result else 0.5
    divergence = (
        best_result.components.get("divergence", float("inf"))
        if best_result
        else float("inf")
    )
    eq_odds = (
        best_result.components.get("equalized_odds_gap", 0.0) if best_result else 0.0
    )
    coalition_demo = (
        compute_coalition_demographics(best_coalition, clients)
        if best_coalition
        else np.zeros(len(target_dist))
    )

    return TrialResult(
        algorithm="Random",
        n_clients=n_clients,
        k=len(target_dist),
        coalition_size=coalition_size,
        trial_idx=0,
        auc_roc=auc_roc,
        demographic_divergence=divergence,
        equalized_odds_gap=eq_odds,
        convergence_iterations=n_random_draws,
        wall_clock_seconds=wall_time,
        coalition=best_coalition,
        coalition_demographics=coalition_demo.tolist(),
        target_demographics=target_dist.as_array().tolist(),
        group_tpr=best_result.metadata.get("group_tpr", []) if best_result else [],
        group_fpr=best_result.metadata.get("group_fpr", []) if best_result else [],
        group_auc=best_result.metadata.get("group_auc", []) if best_result else [],
        fitness=best_fitness,
        metadata={"n_random_draws": n_random_draws},
    )


def run_greedy_size_baseline_trial(
    clients: List[Client],
    fed_dataset: FederatedDataset,
    target_dist: DemographicDistribution,
    coalition_size: int,
    config: RealFLExperimentConfig,
    trial_seed: int = 42,
) -> TrialResult:
    """
    Run greedy-by-dataset-size baseline with real federated training.

    Selects the top-k clients by dataset size, then evaluates via
    real FL training. This is a strong accuracy baseline because
    larger datasets generally improve model quality.

    Args:
        clients: List of fairswarm Client objects
        fed_dataset: Federated dataset
        target_dist: Target demographic distribution
        coalition_size: Number of clients to select
        config: Experiment configuration
        trial_seed: Random seed

    Returns:
        TrialResult with greedy coalition's metrics
    """
    start_time = time.time()

    # Select top-k by dataset size
    size_ranking = sorted(
        range(len(clients)),
        key=lambda i: clients[i].dataset_size,
        reverse=True,
    )
    coalition = size_ranking[:coalition_size]

    fitness_fn = FederatedFitness(
        fed_dataset=fed_dataset,
        target_distribution=target_dist,
        n_fl_rounds=config.n_fl_rounds,
        local_epochs=config.local_epochs,
        learning_rate=config.learning_rate,
        seed=trial_seed,
    )

    result = fitness_fn.evaluate(coalition, clients)
    wall_time = time.time() - start_time

    auc_roc = result.components.get("accuracy", 0.5)
    divergence = result.components.get("divergence", float("inf"))
    eq_odds = result.components.get("equalized_odds_gap", 0.0)
    coalition_demo = compute_coalition_demographics(coalition, clients)

    return TrialResult(
        algorithm="Greedy (size)",
        n_clients=len(clients),
        k=len(target_dist),
        coalition_size=coalition_size,
        trial_idx=0,
        auc_roc=auc_roc,
        demographic_divergence=divergence,
        equalized_odds_gap=eq_odds,
        convergence_iterations=1,
        wall_clock_seconds=wall_time,
        coalition=coalition,
        coalition_demographics=coalition_demo.tolist(),
        target_demographics=target_dist.as_array().tolist(),
        group_tpr=result.metadata.get("group_tpr", []),
        group_fpr=result.metadata.get("group_fpr", []),
        group_auc=result.metadata.get("group_auc", []),
        fitness=result.value,
        metadata={"criterion": "dataset_size"},
    )


def run_fedavg_all_clients_trial(
    clients: List[Client],
    fed_dataset: FederatedDataset,
    target_dist: DemographicDistribution,
    coalition_size: int,
    config: RealFLExperimentConfig,
    trial_seed: int = 42,
) -> TrialResult:
    """
    Run all-clients FedAvg baseline with real federated training.

    All clients participate (no selection). This provides an upper
    bound for accuracy and a baseline for demographic divergence.

    Args:
        clients: List of fairswarm Client objects
        fed_dataset: Federated dataset
        target_dist: Target demographic distribution
        coalition_size: Coalition size (for metadata; all clients used)
        config: Experiment configuration
        trial_seed: Random seed

    Returns:
        TrialResult with all-clients FedAvg metrics
    """
    start_time = time.time()
    n_clients = len(clients)
    coalition = list(range(n_clients))

    fitness_fn = FederatedFitness(
        fed_dataset=fed_dataset,
        target_distribution=target_dist,
        n_fl_rounds=config.n_fl_rounds,
        local_epochs=config.local_epochs,
        learning_rate=config.learning_rate,
        seed=trial_seed,
    )

    result = fitness_fn.evaluate(coalition, clients)
    wall_time = time.time() - start_time

    auc_roc = result.components.get("accuracy", 0.5)
    divergence = result.components.get("divergence", float("inf"))
    eq_odds = result.components.get("equalized_odds_gap", 0.0)
    coalition_demo = compute_coalition_demographics(coalition, clients)

    return TrialResult(
        algorithm="FedAvg (all)",
        n_clients=n_clients,
        k=len(target_dist),
        coalition_size=n_clients,
        trial_idx=0,
        auc_roc=auc_roc,
        demographic_divergence=divergence,
        equalized_odds_gap=eq_odds,
        convergence_iterations=config.n_fl_rounds,
        wall_clock_seconds=wall_time,
        coalition=coalition,
        coalition_demographics=coalition_demo.tolist(),
        target_demographics=target_dist.as_array().tolist(),
        group_tpr=result.metadata.get("group_tpr", []),
        group_fpr=result.metadata.get("group_fpr", []),
        group_auc=result.metadata.get("group_auc", []),
        fitness=result.value,
        metadata={"all_clients": True},
    )


def run_fairdpfl_baseline_trial(
    clients: List[Client],
    fed_dataset: FederatedDataset,
    target_dist: DemographicDistribution,
    coalition_size: int,
    config: RealFLExperimentConfig,
    trial_seed: int = 42,
) -> TrialResult:
    """
    Run FedFDP baseline with real federated training evaluation.

    Uses Lagrangian relaxation-based client selection (Ling et al. 2024),
    then evaluates the selected coalition via real FL training for
    fair head-to-head comparison.

    Args:
        clients: List of fairswarm Client objects
        fed_dataset: Federated dataset
        target_dist: Target demographic distribution
        coalition_size: Number of clients to select
        config: Experiment configuration
        trial_seed: Random seed

    Returns:
        TrialResult with FedFDP coalition's metrics
    """
    start_time = time.time()

    # Build fitness function for evaluation
    fitness_fn = FederatedFitness(
        fed_dataset=fed_dataset,
        target_distribution=target_dist,
        n_fl_rounds=config.n_fl_rounds,
        local_epochs=config.local_epochs,
        learning_rate=config.learning_rate,
        seed=trial_seed,
    )

    # Run FedFDP to select coalition
    dpfl_config = FairDPFLConfig(
        coalition_size=coalition_size,
        n_rounds=config.n_iterations,
        fairness_threshold=0.05,
        privacy_budget=4.0,
        seed=trial_seed,
    )
    dpfl = FairDPFL_SCS(clients=clients, config=dpfl_config)
    dpfl_result = dpfl.run(
        fitness_fn=fitness_fn,
        target_distribution=target_dist.as_array(),
    )

    # Evaluate the selected coalition with real FL training
    coalition = dpfl_result.coalition
    result = fitness_fn.evaluate(coalition, clients)
    wall_time = time.time() - start_time

    auc_roc = result.components.get("accuracy", 0.5)
    divergence = result.components.get("divergence", float("inf"))
    eq_odds = result.components.get("equalized_odds_gap", 0.0)
    coalition_demo = compute_coalition_demographics(coalition, clients)

    return TrialResult(
        algorithm="FedFDP",
        n_clients=len(clients),
        k=len(target_dist),
        coalition_size=coalition_size,
        trial_idx=0,
        auc_roc=auc_roc,
        demographic_divergence=divergence,
        equalized_odds_gap=eq_odds,
        convergence_iterations=dpfl_result.convergence_round or config.n_iterations,
        wall_clock_seconds=wall_time,
        coalition=coalition,
        coalition_demographics=coalition_demo.tolist(),
        target_demographics=target_dist.as_array().tolist(),
        group_tpr=result.metadata.get("group_tpr", []),
        group_fpr=result.metadata.get("group_fpr", []),
        group_auc=result.metadata.get("group_auc", []),
        fitness=result.value,
        metadata={
            "privacy_spent": dpfl_result.privacy_spent,
            "final_lambda": dpfl_result.metrics.get("final_lambda", 0.0),
        },
    )


def run_gwo_baseline_trial(
    clients: List[Client],
    fed_dataset: FederatedDataset,
    target_dist: DemographicDistribution,
    coalition_size: int,
    config: RealFLExperimentConfig,
    trial_seed: int = 42,
) -> TrialResult:
    """
    Run Grey Wolf Optimizer baseline with real federated training.

    GWO selects the coalition using wolf hierarchy dynamics (no fairness
    gradient), then trains a federated model on the selected coalition.

    Args:
        clients: List of fairswarm Client objects
        fed_dataset: Federated dataset
        target_dist: Target demographic distribution
        coalition_size: Number of clients to select
        config: Experiment configuration
        trial_seed: Random seed

    Returns:
        TrialResult with GWO coalition's metrics
    """
    start_time = time.time()
    n_clients = len(clients)

    # Use GWO to select coalition (no fairness gradient)
    gwo = GreyWolfOptimizer(
        n_wolves=config.swarm_size,
        n_iterations=config.n_iterations,
        seed=trial_seed,
    )
    coalition = gwo.select(clients, coalition_size, target_dist)

    # Train federated model on GWO-selected coalition
    fitness_fn = FederatedFitness(
        fed_dataset=fed_dataset,
        target_distribution=target_dist,
        n_fl_rounds=config.n_fl_rounds,
        local_epochs=config.local_epochs,
        learning_rate=config.learning_rate,
        seed=trial_seed,
    )
    result = fitness_fn.evaluate(coalition, clients)
    wall_time = time.time() - start_time

    auc_roc = result.components.get("accuracy", 0.5)
    divergence = result.components.get("divergence", float("inf"))
    eq_odds = result.components.get("equalized_odds_gap", 0.0)
    coalition_demo = compute_coalition_demographics(coalition, clients)

    return TrialResult(
        algorithm="GWO",
        n_clients=n_clients,
        k=len(target_dist),
        coalition_size=coalition_size,
        trial_idx=0,
        auc_roc=auc_roc,
        demographic_divergence=divergence,
        equalized_odds_gap=eq_odds,
        convergence_iterations=config.n_iterations,
        wall_clock_seconds=wall_time,
        coalition=coalition,
        coalition_demographics=coalition_demo.tolist(),
        target_demographics=target_dist.as_array().tolist(),
        group_tpr=result.metadata.get("group_tpr", []),
        group_fpr=result.metadata.get("group_fpr", []),
        group_auc=result.metadata.get("group_auc", []),
        fitness=result.value,
        metadata={"gwo_wolves": config.swarm_size},
    )


def run_fairfed_baseline_trial(
    clients: List[Client],
    fed_dataset: FederatedDataset,
    target_dist: DemographicDistribution,
    coalition_size: int,
    config: RealFLExperimentConfig,
    trial_seed: int = 42,
) -> TrialResult:
    """
    Run FairFed baseline (Ezzeldin et al., AAAI 2023) with real FL evaluation.

    FairFed adjusts aggregation weights based on demographic parity gaps,
    then selects top-k clients by adjusted weight. The selected coalition
    is evaluated via real FL training for fair head-to-head comparison.

    Args:
        clients: List of fairswarm Client objects
        fed_dataset: Federated dataset
        target_dist: Target demographic distribution
        coalition_size: Number of clients to select
        config: Experiment configuration
        trial_seed: Random seed

    Returns:
        TrialResult with FairFed coalition's metrics
    """
    start_time = time.time()

    fitness_fn = FederatedFitness(
        fed_dataset=fed_dataset,
        target_distribution=target_dist,
        n_fl_rounds=config.n_fl_rounds,
        local_epochs=config.local_epochs,
        learning_rate=config.learning_rate,
        seed=trial_seed,
    )

    fairfed_config = FairFedConfig(
        coalition_size=coalition_size,
        n_rounds=config.n_iterations,
        beta=1.0,
        seed=trial_seed,
    )
    fairfed = FairFedBaseline(clients=clients, config=fairfed_config)
    fairfed_result = fairfed.run(
        fitness_fn=fitness_fn,
        target_distribution=target_dist.as_array(),
    )

    coalition = fairfed_result.coalition
    result = fitness_fn.evaluate(coalition, clients)
    wall_time = time.time() - start_time

    auc_roc = result.components.get("accuracy", 0.5)
    divergence = result.components.get("divergence", float("inf"))
    eq_odds = result.components.get("equalized_odds_gap", 0.0)
    coalition_demo = compute_coalition_demographics(coalition, clients)

    return TrialResult(
        algorithm="FairFed",
        n_clients=len(clients),
        k=len(target_dist),
        coalition_size=coalition_size,
        trial_idx=0,
        auc_roc=auc_roc,
        demographic_divergence=divergence,
        equalized_odds_gap=eq_odds,
        convergence_iterations=fairfed_result.convergence_round or config.n_iterations,
        wall_clock_seconds=wall_time,
        coalition=coalition,
        coalition_demographics=coalition_demo.tolist(),
        target_demographics=target_dist.as_array().tolist(),
        group_tpr=result.metadata.get("group_tpr", []),
        group_fpr=result.metadata.get("group_fpr", []),
        group_auc=result.metadata.get("group_auc", []),
        fitness=result.value,
        metadata={"beta": fairfed_config.beta},
    )


def run_qffl_baseline_trial(
    clients: List[Client],
    fed_dataset: FederatedDataset,
    target_dist: DemographicDistribution,
    coalition_size: int,
    config: RealFLExperimentConfig,
    trial_seed: int = 42,
) -> TrialResult:
    """
    Run q-FFL baseline (Li et al., ICLR 2020) with real FL evaluation.

    q-FFL reweights clients based on local loss (higher loss = more weight),
    equalizing performance across clients. Unlike FairSwarm's demographic
    fairness, q-FFL targets performance equity.

    Args:
        clients: List of fairswarm Client objects
        fed_dataset: Federated dataset
        target_dist: Target demographic distribution
        coalition_size: Number of clients to select
        config: Experiment configuration
        trial_seed: Random seed

    Returns:
        TrialResult with q-FFL coalition's metrics
    """
    start_time = time.time()

    fitness_fn = FederatedFitness(
        fed_dataset=fed_dataset,
        target_distribution=target_dist,
        n_fl_rounds=config.n_fl_rounds,
        local_epochs=config.local_epochs,
        learning_rate=config.learning_rate,
        seed=trial_seed,
    )

    qffl_config = QFFLConfig(
        n_rounds=config.n_iterations,
        local_epochs=config.local_epochs,
        learning_rate=config.learning_rate,
        q=5.0,
        participation_rate=config.coalition_fraction,
        seed=trial_seed,
    )
    qffl = QFFLBaseline(clients=clients, config=qffl_config)
    qffl_result = qffl.run(
        fitness_fn=fitness_fn,
        target_distribution=target_dist.as_array(),
    )

    coalition = qffl_result.coalition
    result = fitness_fn.evaluate(coalition, clients)
    wall_time = time.time() - start_time

    auc_roc = result.components.get("accuracy", 0.5)
    divergence = result.components.get("divergence", float("inf"))
    eq_odds = result.components.get("equalized_odds_gap", 0.0)
    coalition_demo = compute_coalition_demographics(coalition, clients)

    return TrialResult(
        algorithm="q-FFL",
        n_clients=len(clients),
        k=len(target_dist),
        coalition_size=coalition_size,
        trial_idx=0,
        auc_roc=auc_roc,
        demographic_divergence=divergence,
        equalized_odds_gap=eq_odds,
        convergence_iterations=qffl_result.convergence_round or config.n_iterations,
        wall_clock_seconds=wall_time,
        coalition=coalition,
        coalition_demographics=coalition_demo.tolist(),
        target_demographics=target_dist.as_array().tolist(),
        group_tpr=result.metadata.get("group_tpr", []),
        group_fpr=result.metadata.get("group_fpr", []),
        group_auc=result.metadata.get("group_auc", []),
        fitness=result.value,
        metadata={"q": qffl_config.q},
    )



# Full Experiment Runner



def _run_single_scenario_trial(args: Dict[str, Any]) -> List[TrialResult]:
    """
    Worker function for parallel execution of a single (n, k, trial) scenario.

    Runs all algorithms (FairSwarm, Standard PSO, Random, Greedy, FedAvg-all)
    on the same data partition for a single trial.

    Must be a top-level function for pickling in ProcessPoolExecutor.

    Args:
        args: Dictionary with all parameters needed to reconstruct
              the scenario and run all algorithms

    Returns:
        List of TrialResult, one per algorithm
    """
    n_clients = args["n_clients"]
    k = args["k"]
    trial_idx = args["trial_idx"]
    trial_seed = args["trial_seed"]
    coalition_size = args["coalition_size"]

    # Rebuild config (not picklable as dataclass across processes)
    config = RealFLExperimentConfig(
        n_clients_values=[n_clients],
        k_values=[k],
        coalition_fraction=args["coalition_fraction"],
        n_fl_rounds=args["n_fl_rounds"],
        local_epochs=args["local_epochs"],
        learning_rate=args["learning_rate"],
        n_trials=1,
        n_iterations=args["n_iterations"],
        swarm_size=args["swarm_size"],
        non_iid_alpha=args["non_iid_alpha"],
        n_samples_total=args["n_samples_total"],
        n_features=args["n_features"],
        seed=trial_seed,
    )

    # Generate data for this trial
    fed_dataset = generate_federated_dataset(
        n_clients=n_clients,
        n_demographic_groups=k,
        n_features=config.n_features,
        n_samples_total=config.n_samples_total,
        non_iid_alpha=config.non_iid_alpha,
        seed=trial_seed,
    )

    clients = build_fairswarm_clients(fed_dataset)
    target_dist = DemographicDistribution(
        values=fed_dataset.target_distribution,
    )

    results: List[TrialResult] = []

    # 1. FairSwarm (with fairness gradient)
    try:
        r = run_fairswarm_trial(
            clients=clients,
            fed_dataset=fed_dataset,
            target_dist=target_dist,
            coalition_size=coalition_size,
            config=config,
            fairness_coefficient=0.5,
            trial_seed=trial_seed,
            label="FairSwarm",
        )
        r.trial_idx = trial_idx
        results.append(r)
    except Exception as e:
        logger.error(f"FairSwarm failed (n={n_clients}, k={k}, trial={trial_idx}): {e}")

    # 2. Standard PSO (c3=0, no fairness gradient)
    try:
        r = run_fairswarm_trial(
            clients=clients,
            fed_dataset=fed_dataset,
            target_dist=target_dist,
            coalition_size=coalition_size,
            config=config,
            fairness_coefficient=0.0,
            trial_seed=trial_seed + 1,
            label="Standard PSO",
        )
        r.trial_idx = trial_idx
        results.append(r)
    except Exception as e:
        logger.error(
            f"Standard PSO failed (n={n_clients}, k={k}, trial={trial_idx}): {e}"
        )

    # 3. Random selection
    try:
        r = run_random_baseline_trial(
            clients=clients,
            fed_dataset=fed_dataset,
            target_dist=target_dist,
            coalition_size=coalition_size,
            config=config,
            trial_seed=trial_seed + 2,
        )
        r.trial_idx = trial_idx
        results.append(r)
    except Exception as e:
        logger.error(
            f"Random baseline failed (n={n_clients}, k={k}, trial={trial_idx}): {e}"
        )

    # 4. Greedy by dataset size
    try:
        r = run_greedy_size_baseline_trial(
            clients=clients,
            fed_dataset=fed_dataset,
            target_dist=target_dist,
            coalition_size=coalition_size,
            config=config,
            trial_seed=trial_seed + 3,
        )
        r.trial_idx = trial_idx
        results.append(r)
    except Exception as e:
        logger.error(
            f"Greedy baseline failed (n={n_clients}, k={k}, trial={trial_idx}): {e}"
        )

    # 5. FedAvg all clients
    try:
        r = run_fedavg_all_clients_trial(
            clients=clients,
            fed_dataset=fed_dataset,
            target_dist=target_dist,
            coalition_size=coalition_size,
            config=config,
            trial_seed=trial_seed + 4,
        )
        r.trial_idx = trial_idx
        results.append(r)
    except Exception as e:
        logger.error(
            f"FedAvg-all baseline failed (n={n_clients}, k={k}, trial={trial_idx}): {e}"
        )

    # 6. FedFDP baseline (head-to-head SOTA comparison)
    try:
        r = run_fairdpfl_baseline_trial(
            clients=clients,
            fed_dataset=fed_dataset,
            target_dist=target_dist,
            coalition_size=coalition_size,
            config=config,
            trial_seed=trial_seed + 5,
        )
        r.trial_idx = trial_idx
        results.append(r)
    except Exception as e:
        logger.error(
            f"FedFDP baseline failed (n={n_clients}, k={k}, trial={trial_idx}): {e}"
        )

    # 7. Grey Wolf Optimizer (swarm intelligence baseline, no fairness gradient)
    try:
        r = run_gwo_baseline_trial(
            clients=clients,
            fed_dataset=fed_dataset,
            target_dist=target_dist,
            coalition_size=coalition_size,
            config=config,
            trial_seed=trial_seed + 6,
        )
        r.trial_idx = trial_idx
        results.append(r)
    except Exception as e:
        logger.error(
            f"GWO baseline failed (n={n_clients}, k={k}, trial={trial_idx}): {e}"
        )

    # 8. FairFed (aggregation-level fairness, Ezzeldin et al. AAAI 2023)
    try:
        r = run_fairfed_baseline_trial(
            clients=clients,
            fed_dataset=fed_dataset,
            target_dist=target_dist,
            coalition_size=coalition_size,
            config=config,
            trial_seed=trial_seed + 7,
        )
        r.trial_idx = trial_idx
        results.append(r)
    except Exception as e:
        logger.error(
            f"FairFed baseline failed (n={n_clients}, k={k}, trial={trial_idx}): {e}"
        )

    # 9. q-FFL (loss-based fairness, Li et al. ICLR 2020)
    try:
        r = run_qffl_baseline_trial(
            clients=clients,
            fed_dataset=fed_dataset,
            target_dist=target_dist,
            coalition_size=coalition_size,
            config=config,
            trial_seed=trial_seed + 8,
        )
        r.trial_idx = trial_idx
        results.append(r)
    except Exception as e:
        logger.error(
            f"q-FFL baseline failed (n={n_clients}, k={k}, trial={trial_idx}): {e}"
        )

    # Tag all results with the non-IID alpha used
    alpha_val = args.get("non_iid_alpha", 0.5)
    for r in results:
        r.non_iid_alpha = alpha_val

    return results


def run_experiment_sequential(
    exp_config: RealFLExperimentConfig,
) -> Dict[str, Any]:
    """
    Run the full experiment sequentially (single process).

    Args:
        exp_config: Experiment configuration

    Returns:
        Complete results dictionary
    """
    rng = np.random.default_rng(exp_config.seed)
    all_results: List[TrialResult] = []
    total_start = time.time()

    total_scenarios = (
        len(exp_config.n_clients_values)
        * len(exp_config.k_values)
        * len(exp_config.non_iid_alpha_values)
        * exp_config.n_trials
    )
    scenario_idx = 0

    for alpha in exp_config.non_iid_alpha_values:
        for n_clients in exp_config.n_clients_values:
            for k in exp_config.k_values:
                coalition_size = max(3, int(n_clients * exp_config.coalition_fraction))

                for trial_idx in range(exp_config.n_trials):
                    scenario_idx += 1
                    trial_seed = int(rng.integers(0, 2**31))

                    logger.info(
                        f"Scenario {scenario_idx}/{total_scenarios}: "
                        f"n={n_clients}, k={k}, alpha={alpha}, "
                        f"trial={trial_idx + 1}/{exp_config.n_trials}"
                    )

                    task_args = {
                        "n_clients": n_clients,
                        "k": k,
                        "trial_idx": trial_idx,
                        "trial_seed": trial_seed,
                        "coalition_size": coalition_size,
                        "coalition_fraction": exp_config.coalition_fraction,
                        "n_fl_rounds": exp_config.n_fl_rounds,
                        "local_epochs": exp_config.local_epochs,
                        "learning_rate": exp_config.learning_rate,
                        "n_iterations": exp_config.n_iterations,
                        "swarm_size": exp_config.swarm_size,
                        "non_iid_alpha": alpha,
                        "n_samples_total": exp_config.n_samples_total,
                        "n_features": exp_config.n_features,
                    }

                    trial_results = _run_single_scenario_trial(task_args)
                    all_results.extend(trial_results)

    total_time = time.time() - total_start
    analysis = analyze_results(all_results, exp_config)

    return {
        "config": asdict(exp_config),
        "analysis": analysis,
        "n_total_trials": len(all_results),
        "execution_time_seconds": total_time,
        "execution_mode": "sequential",
        "timestamp": datetime.now().isoformat(),
    }


def run_experiment_parallel(
    exp_config: RealFLExperimentConfig,
) -> Dict[str, Any]:
    """
    Run the full experiment in parallel across CPU cores.

    Each (n_clients, k, trial) scenario runs as an independent task
    in a ProcessPoolExecutor. All algorithms are run within the same
    task to share the generated dataset.

    Args:
        exp_config: Experiment configuration

    Returns:
        Complete results dictionary
    """
    n_workers = get_n_workers()
    rng = np.random.default_rng(exp_config.seed)
    total_start = time.time()

    # Build task list
    tasks: List[Dict[str, Any]] = []

    for alpha in exp_config.non_iid_alpha_values:
        for n_clients in exp_config.n_clients_values:
            for k in exp_config.k_values:
                coalition_size = max(3, int(n_clients * exp_config.coalition_fraction))

                for trial_idx in range(exp_config.n_trials):
                    trial_seed = int(rng.integers(0, 2**31))

                    tasks.append(
                        {
                            "n_clients": n_clients,
                            "k": k,
                            "trial_idx": trial_idx,
                            "trial_seed": trial_seed,
                            "coalition_size": coalition_size,
                            "coalition_fraction": exp_config.coalition_fraction,
                            "n_fl_rounds": exp_config.n_fl_rounds,
                            "local_epochs": exp_config.local_epochs,
                            "learning_rate": exp_config.learning_rate,
                            "n_iterations": exp_config.n_iterations,
                            "swarm_size": exp_config.swarm_size,
                            "non_iid_alpha": alpha,
                            "n_samples_total": exp_config.n_samples_total,
                            "n_features": exp_config.n_features,
                        }
                    )

    n_tasks = len(tasks)
    logger.info(
        f"Running {n_tasks} scenarios across {n_workers} workers "
        f"(9 algorithms per scenario)"
    )

    all_results: List[TrialResult] = []
    completed = 0

    print(f"\nRunning {n_tasks} scenarios across {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(_run_single_scenario_trial, task): idx
            for idx, task in enumerate(tasks)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                trial_results = future.result()
                all_results.extend(trial_results)
                completed += 1

                elapsed = time.time() - total_start
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (n_tasks - completed) / rate if rate > 0 else 0
                print(
                    f"\r  Progress: {completed}/{n_tasks} ({100 * completed / n_tasks:.0f}%) "
                    f"- {rate:.1f} scenarios/sec - ETA: {eta:.0f}s",
                    end="",
                    flush=True,
                )

            except Exception as e:
                logger.error(f"Scenario {idx} failed: {e}")
                completed += 1

    total_time = time.time() - total_start
    print(f"\n  Completed in {total_time:.1f}s")

    analysis = analyze_results(all_results, exp_config)

    return {
        "config": asdict(exp_config),
        "analysis": analysis,
        "n_total_trials": len(all_results),
        "execution_time_seconds": total_time,
        "execution_mode": "parallel",
        "n_workers": n_workers,
        "timestamp": datetime.now().isoformat(),
    }



# Results Analysis



def analyze_results(
    all_results: List[TrialResult],
    config: RealFLExperimentConfig,
) -> Dict[str, Any]:
    """
    Analyze experiment results with confidence intervals for publication.

    Groups results by (n_clients, k, algorithm) and computes:
    - Mean and 95% CI for AUC-ROC, demographic divergence, wall-clock time
    - Statistical comparisons (FairSwarm vs each baseline)
    - Summary table suitable for JMLR publication

    Args:
        all_results: All trial results across all scenarios and algorithms
        config: Experiment configuration

    Returns:
        Analysis dictionary with per-scenario and aggregate statistics
    """
    analysis: Dict[str, Any] = {
        "per_scenario": {},
        "aggregate": {},
        "statistical_comparisons": {},
    }

    algorithms = [
        "FairSwarm",
        "Standard PSO",
        "Random",
        "Greedy (size)",
        "FedAvg (all)",
        "FedFDP",
        "GWO",
        "FairFed",
        "q-FFL",
    ]

    for n_clients in config.n_clients_values:
        for k in config.k_values:
            scenario_key = f"n{n_clients}_k{k}"
            scenario_results: Dict[str, List[TrialResult]] = {
                alg: [] for alg in algorithms
            }

            # Collect results for this scenario
            for r in all_results:
                if (
                    r.n_clients == n_clients
                    and r.k == k
                    and r.algorithm in scenario_results
                ):
                    scenario_results[r.algorithm].append(r)

            scenario_analysis: Dict[str, Any] = {}

            for alg, results in scenario_results.items():
                if not results:
                    continue

                auc_values = [r.auc_roc for r in results]
                div_values = [r.demographic_divergence for r in results]
                eqodds_values = [r.equalized_odds_gap for r in results]
                time_values = [r.wall_clock_seconds for r in results]
                conv_values = [r.convergence_iterations for r in results]

                auc_ci = mean_ci(auc_values)
                div_ci = mean_ci(div_values)
                eqodds_ci = mean_ci(eqodds_values)
                time_ci = mean_ci(time_values)
                conv_ci = mean_ci(conv_values)

                scenario_analysis[alg] = {
                    "n_trials": len(results),
                    "auc_roc": {
                        "mean": auc_ci.mean,
                        "ci": auc_ci.to_dict(),
                    },
                    "demographic_divergence": {
                        "mean": div_ci.mean,
                        "ci": div_ci.to_dict(),
                    },
                    "equalized_odds_gap": {
                        "mean": eqodds_ci.mean,
                        "ci": eqodds_ci.to_dict(),
                    },
                    "wall_clock_seconds": {
                        "mean": time_ci.mean,
                        "ci": time_ci.to_dict(),
                    },
                    "convergence_iterations": {
                        "mean": conv_ci.mean,
                        "ci": conv_ci.to_dict(),
                    },
                    "coalition_size": results[0].coalition_size,
                }

            analysis["per_scenario"][scenario_key] = scenario_analysis

            # Statistical comparisons: FairSwarm vs each baseline
            fairswarm_results = scenario_results.get("FairSwarm", [])
            comparisons: Dict[str, Any] = {}

            baselines = [
                "Standard PSO",
                "Random",
                "Greedy (size)",
                "FedAvg (all)",
                "FedFDP",
                "GWO",
                "FairFed",
                "q-FFL",
            ]
            n_baselines = len(baselines)  # Bonferroni correction factor

            if len(fairswarm_results) >= 2:
                for baseline_name in baselines:
                    baseline_results = scenario_results.get(baseline_name, [])
                    if len(baseline_results) < 2:
                        continue

                    # Compare AUC-ROC (Bonferroni-corrected for multiple comparisons)
                    fs_auc = [r.auc_roc for r in fairswarm_results]
                    bl_auc = [r.auc_roc for r in baseline_results]
                    auc_comparison = compare_means(
                        fs_auc, bl_auc, n_comparisons=n_baselines
                    )

                    # Compare divergence (lower is better, Bonferroni-corrected)
                    fs_div = [r.demographic_divergence for r in fairswarm_results]
                    bl_div = [r.demographic_divergence for r in baseline_results]
                    div_comparison = compare_means(
                        bl_div, fs_div, n_comparisons=n_baselines
                    )

                    # Compare equalized odds gap (lower is better, Bonferroni-corrected)
                    fs_eqodds = [r.equalized_odds_gap for r in fairswarm_results]
                    bl_eqodds = [r.equalized_odds_gap for r in baseline_results]
                    eqodds_comparison = compare_means(
                        bl_eqodds, fs_eqodds, n_comparisons=n_baselines
                    )

                    comparisons[baseline_name] = {
                        "auc_roc": auc_comparison,
                        "demographic_divergence": div_comparison,
                        "equalized_odds_gap": eqodds_comparison,
                    }

            analysis["statistical_comparisons"][scenario_key] = comparisons

    # Aggregate statistics across all scenarios
    for alg in algorithms:
        alg_results = [r for r in all_results if r.algorithm == alg]
        if not alg_results:
            continue

        analysis["aggregate"][alg] = {
            "n_total_trials": len(alg_results),
            "auc_roc": statistical_summary(
                [r.auc_roc for r in alg_results], name=f"{alg}_auc_roc"
            ),
            "demographic_divergence": statistical_summary(
                [r.demographic_divergence for r in alg_results],
                name=f"{alg}_divergence",
            ),
            "equalized_odds_gap": statistical_summary(
                [r.equalized_odds_gap for r in alg_results],
                name=f"{alg}_eqodds",
            ),
            "wall_clock_seconds": statistical_summary(
                [r.wall_clock_seconds for r in alg_results],
                name=f"{alg}_time",
            ),
        }

    # Build summary
    analysis["summary"] = _build_summary(analysis, config)

    return analysis


def _build_summary(analysis: Dict[str, Any], config: RealFLExperimentConfig) -> str:
    """
    Build a human-readable summary of the experiment results.

    Args:
        analysis: The full analysis dictionary
        config: Experiment configuration

    Returns:
        Multi-line summary string
    """
    lines = []
    lines.append("Real Federated Learning Experiment Summary")
    lines.append("=" * 60)

    agg = analysis.get("aggregate", {})
    fs_agg = agg.get("FairSwarm", {})
    pso_agg = agg.get("Standard PSO", {})

    if fs_agg:
        fs_auc = fs_agg.get("auc_roc", {})
        fs_div = fs_agg.get("demographic_divergence", {})
        lines.append(
            f"FairSwarm:      AUC-ROC = {fs_auc.get('mean', 0):.4f} "
            f"[{fs_auc.get('mean_ci', {}).get('ci_lower', 0):.4f}, "
            f"{fs_auc.get('mean_ci', {}).get('ci_upper', 0):.4f}], "
            f"DemDiv = {fs_div.get('mean', 0):.4f}"
        )

    if pso_agg:
        pso_auc = pso_agg.get("auc_roc", {})
        pso_div = pso_agg.get("demographic_divergence", {})
        lines.append(
            f"Standard PSO:   AUC-ROC = {pso_auc.get('mean', 0):.4f} "
            f"[{pso_auc.get('mean_ci', {}).get('ci_lower', 0):.4f}, "
            f"{pso_auc.get('mean_ci', {}).get('ci_upper', 0):.4f}], "
            f"DemDiv = {pso_div.get('mean', 0):.4f}"
        )

    for baseline in ["Random", "Greedy (size)", "FedAvg (all)", "FedFDP", "GWO", "FairFed", "q-FFL"]:
        bl_agg = agg.get(baseline, {})
        if bl_agg:
            bl_auc = bl_agg.get("auc_roc", {})
            bl_div = bl_agg.get("demographic_divergence", {})
            label = f"{baseline}:".ljust(16)
            lines.append(
                f"{label}AUC-ROC = {bl_auc.get('mean', 0):.4f} "
                f"[{bl_auc.get('mean_ci', {}).get('ci_lower', 0):.4f}, "
                f"{bl_auc.get('mean_ci', {}).get('ci_upper', 0):.4f}], "
                f"DemDiv = {bl_div.get('mean', 0):.4f}"
            )

    # FairSwarm vs Standard PSO fairness comparison
    if fs_agg and pso_agg:
        fs_div_mean = fs_agg.get("demographic_divergence", {}).get("mean", 0)
        pso_div_mean = pso_agg.get("demographic_divergence", {}).get("mean", 0)
        if pso_div_mean > 0:
            improvement = (pso_div_mean - fs_div_mean) / pso_div_mean * 100
            lines.append(
                f"\nFairSwarm fairness gradient reduces demographic divergence "
                f"by {improvement:.1f}% vs Standard PSO"
            )

    lines.append("=" * 60)
    return "\n".join(lines)



# Result Saving



def save_results(results: Dict[str, Any], output_dir: str) -> Path:
    """
    Save experiment results as JSON to the output directory.

    Creates the output directory if it does not exist. File is
    timestamped for provenance tracking.

    Args:
        results: Complete results dictionary
        output_dir: Target directory path

    Returns:
        Path to the saved JSON file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results["code_version"] = get_git_hash()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"real_fl_results_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {filename}")
    return filename



# Main Entry Point



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Real Federated Learning Experiment with FairSwarm Coalition Selection. "
            "Trains logistic regression models on synthetic classification data "
            "partitioned across n clients with non-IID demographics."
        ),
    )

    # Client and data configuration
    parser.add_argument(
        "--n_clients",
        type=int,
        nargs="+",
        default=None,
        help="Client counts to sweep (default: 20 50 100)",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=None,
        help="Demographic group counts to sweep (default: 4 8)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=20000,
        help="Total dataset size before partitioning (default: 20000)",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=20,
        help="Feature dimensionality (default: 20)",
    )
    parser.add_argument(
        "--non_iid_alpha",
        type=float,
        nargs="+",
        default=[0.5],
        help="Dirichlet concentration(s) for non-IID partitioning (default: 0.5). "
        "Pass multiple values to sweep heterogeneity levels, e.g. --non_iid_alpha 0.1 0.5 1.0",
    )

    # FL configuration
    parser.add_argument(
        "--n_fl_rounds",
        type=int,
        default=3,
        help="FL training rounds per fitness evaluation (default: 3)",
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=5,
        help="Local SGD epochs per FL round (default: 5)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Local learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--coalition_fraction",
        type=float,
        default=0.3,
        help="Fraction of clients per coalition (default: 0.3)",
    )

    # PSO configuration
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=50,
        help="PSO iterations for FairSwarm (default: 50)",
    )
    parser.add_argument(
        "--swarm_size",
        type=int,
        default=20,
        help="Number of PSO particles (default: 20)",
    )

    # Experiment configuration
    parser.add_argument(
        "--n_trials",
        type=int,
        default=5,
        help="Independent trials per (n, k) configuration (default: 5)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/real_fl",
        help="Output directory for JSON results (default: results/real_fl)",
    )

    # Execution mode
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run scenarios in parallel across CPU cores (recommended)",
    )

    args = parser.parse_args()

    # Build configuration
    config = RealFLExperimentConfig(
        n_clients_values=args.n_clients if args.n_clients else [20, 50, 100],
        k_values=args.k if args.k else [4, 8],
        coalition_fraction=args.coalition_fraction,
        n_fl_rounds=args.n_fl_rounds,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        n_trials=args.n_trials,
        n_iterations=args.n_iterations,
        swarm_size=args.swarm_size,
        non_iid_alpha=args.non_iid_alpha[0],
        non_iid_alpha_values=args.non_iid_alpha,
        n_samples_total=args.n_samples,
        n_features=args.n_features,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # Print experiment summary
    total_scenarios = (
        len(config.n_clients_values) * len(config.k_values)
        * len(config.non_iid_alpha_values) * config.n_trials
    )
    algorithms_per_scenario = 9
    total_trial_runs = total_scenarios * algorithms_per_scenario

    print("\n" + "=" * 70)
    print("REAL FEDERATED LEARNING EXPERIMENT")
    print("FairSwarm Coalition Selection with Actual Model Training")
    print("=" * 70)
    print(f"  n_clients:       {config.n_clients_values}")
    print(f"  k (demographics): {config.k_values}")
    print(f"  coalition_fraction: {config.coalition_fraction}")
    print(f"  FL rounds/eval:  {config.n_fl_rounds}")
    print(f"  Local epochs:    {config.local_epochs}")
    print(f"  PSO iterations:  {config.n_iterations}")
    print(f"  Swarm size:      {config.swarm_size}")
    print(f"  Trials per (n,k): {config.n_trials}")
    print(f"  Total scenarios: {total_scenarios}")
    print(f"  Algorithms:      {algorithms_per_scenario}")
    print(f"  Total runs:      {total_trial_runs}")
    print(f"  Dataset size:    {config.n_samples_total} samples")
    print(f"  Non-IID alpha:   {config.non_iid_alpha_values}")
    print(f"  Seed:            {config.seed}")
    print(f"  Output:          {config.output_dir}")

    if args.parallel:
        n_workers = get_n_workers()
        print(f"\n  Mode:            PARALLEL ({n_workers} workers)")
        print("=" * 70 + "\n")
        results = run_experiment_parallel(config)
    else:
        print("\n  Mode:            SEQUENTIAL")
        print("=" * 70 + "\n")
        results = run_experiment_sequential(config)

    # Print summary
    print("\n" + results["analysis"]["summary"])

    if "execution_time_seconds" in results:
        print(f"\nTotal execution time: {results['execution_time_seconds']:.1f}s")

    # Save results
    output_path = save_results(results, config.output_dir)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
