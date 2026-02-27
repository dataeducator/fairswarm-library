"""
Baseline algorithms for experimental comparison with FairSwarm.

Implements the baselines from Section 4.2 of the paper:
    - Random selection (lower bound)
    - FedAvg with all clients (no selection)
    - Greedy selection (myopic comparison)
    - Standard PSO without fairness term (ablation)
    - FedFDP (Ling et al., 2024, state-of-the-art competitor)
    - Grey Wolf Optimizer (alternative swarm algorithm)
"""

from baselines.fedavg import FedAvgBaseline, FedAvgConfig
from baselines.fair_dpfl_scs import FairDPFL_SCS, FairDPFLConfig
from baselines.greedy import GreedyBaseline, GreedyConfig, GreedyCriterion
from baselines.greedy_selection import GreedySelection
from baselines.random_selection import RandomSelectionBaseline, RandomSelectionConfig
from baselines.standard_pso import StandardPSO

# Backward-compatible aliases
RandomSelection = RandomSelectionBaseline

__all__ = [
    "FedAvgBaseline",
    "FedAvgConfig",
    "FairDPFL_SCS",
    "FairDPFLConfig",
    "GreedyBaseline",
    "GreedyConfig",
    "GreedyCriterion",
    "GreedySelection",
    "RandomSelection",
    "RandomSelectionBaseline",
    "RandomSelectionConfig",
    "StandardPSO",
]
