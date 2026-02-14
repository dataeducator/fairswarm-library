"""
Baseline algorithms for experimental comparison with FairSwarm.

Implements the baselines from Section 4.2 of CLAUDE.md:
    - Random selection (lower bound)
    - FedAvg with all clients (no selection)
    - Greedy selection (myopic comparison)
    - Standard PSO without fairness term (ablation)
    - FairDPFL-SCS (2024 state-of-the-art competitor)
    - Grey Wolf Optimizer (alternative swarm algorithm)
"""

from experiments.baselines.random_selection import RandomSelection
from experiments.baselines.greedy_selection import GreedySelection
from experiments.baselines.standard_pso import StandardPSO

__all__ = [
    "RandomSelection",
    "GreedySelection",
    "StandardPSO",
]
