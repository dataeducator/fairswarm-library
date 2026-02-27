"""
Parallel Experiment Runner for FairSwarm.

Provides utilities for running experiments in parallel across multiple CPU cores.
Optimized for systems without GPU acceleration.

Author: Tenicka Norwood
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypeVar

import numpy as np


# Get optimal worker count (leave 2 cores for system)
def get_optimal_workers() -> int:
    """Get optimal number of worker processes."""
    cpu_count = os.cpu_count() or 4
    # Use all but 2 cores, minimum 2 workers
    return max(2, cpu_count - 2)


@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""

    n_workers: int = None  # None = auto-detect
    chunk_size: int = 1  # Tasks per worker batch
    timeout: Optional[float] = None  # Timeout per task in seconds
    show_progress: bool = True

    def __post_init__(self):
        if self.n_workers is None:
            self.n_workers = get_optimal_workers()


T = TypeVar("T")
R = TypeVar("R")


def run_parallel(
    func: Callable[[T], R],
    tasks: List[T],
    config: Optional[ParallelConfig] = None,
    desc: str = "Running",
) -> List[R]:
    """
    Run a function in parallel across multiple tasks.

    Args:
        func: Function to execute (must be picklable)
        tasks: List of task arguments
        config: Parallel execution configuration
        desc: Description for progress display

    Returns:
        List of results in same order as tasks
    """
    if config is None:
        config = ParallelConfig()

    n_tasks = len(tasks)
    if n_tasks == 0:
        return []

    # For very few tasks, run sequentially
    if n_tasks <= 2:
        return [func(task) for task in tasks]

    results = [None] * n_tasks
    completed = 0
    start_time = time.time()

    if config.show_progress:
        print(f"\n{desc}: {n_tasks} tasks across {config.n_workers} workers")

    with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
        # Submit all tasks with their indices
        future_to_idx = {
            executor.submit(func, task): idx for idx, task in enumerate(tasks)
        }

        # Collect results as they complete
        for future in as_completed(future_to_idx, timeout=config.timeout):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
                completed += 1

                if config.show_progress:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (n_tasks - completed) / rate if rate > 0 else 0
                    print(
                        f"\r  Progress: {completed}/{n_tasks} "
                        f"({100 * completed / n_tasks:.0f}%) "
                        f"- {rate:.1f} tasks/sec "
                        f"- ETA: {eta:.0f}s",
                        end="",
                        flush=True,
                    )

            except Exception as e:
                print(f"\n  Task {idx} failed: {e}")
                results[idx] = None

    if config.show_progress:
        total_time = time.time() - start_time
        print(
            f"\n  Completed in {total_time:.1f}s ({n_tasks / total_time:.1f} tasks/sec)"
        )

    return results


def run_parallel_with_seeds(
    func: Callable[[Any, int], R],
    base_args: Any,
    n_trials: int,
    base_seed: int = 42,
    config: Optional[ParallelConfig] = None,
    desc: str = "Running trials",
) -> List[R]:
    """
    Run trials in parallel with different random seeds.

    Args:
        func: Function taking (args, seed) -> result
        base_args: Arguments to pass to each trial
        n_trials: Number of trials to run
        base_seed: Base seed for reproducibility
        config: Parallel configuration
        desc: Description for progress

    Returns:
        List of results from all trials
    """
    # Generate deterministic seeds
    rng = np.random.default_rng(base_seed)
    seeds = rng.integers(0, 2**31, size=n_trials).tolist()

    # Create task tuples
    tasks = [(base_args, seed) for seed in seeds]

    # Wrapper to unpack tuple
    def wrapper(task_tuple):
        args, seed = task_tuple
        return func(args, seed)

    return run_parallel(wrapper, tasks, config, desc)


class ProgressTracker:
    """Thread-safe progress tracker for parallel experiments."""

    def __init__(self, total: int, desc: str = "Progress"):
        self.total = total
        self.desc = desc
        self.completed = mp.Value("i", 0)
        self.start_time = time.time()
        self._lock = mp.Lock()

    def update(self, n: int = 1):
        """Update progress by n items."""
        with self._lock:
            self.completed.value += n
            self._print_progress()

    def _print_progress(self):
        """Print current progress."""
        completed = self.completed.value
        elapsed = time.time() - self.start_time
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (self.total - completed) / rate if rate > 0 else 0

        print(
            f"\r{self.desc}: {completed}/{self.total} "
            f"({100 * completed / self.total:.0f}%) "
            f"- ETA: {eta:.0f}s",
            end="",
            flush=True,
        )


def estimate_time(
    n_tasks: int,
    time_per_task: float,
    n_workers: Optional[int] = None,
) -> Dict[str, float]:
    """
    Estimate parallel execution time.

    Args:
        n_tasks: Number of tasks
        time_per_task: Estimated seconds per task
        n_workers: Number of workers (None = auto)

    Returns:
        Dictionary with time estimates
    """
    if n_workers is None:
        n_workers = get_optimal_workers()

    sequential_time = n_tasks * time_per_task
    parallel_time = sequential_time / n_workers
    # Add 10% overhead for process management
    parallel_time *= 1.1

    return {
        "sequential_seconds": sequential_time,
        "parallel_seconds": parallel_time,
        "speedup": sequential_time / parallel_time,
        "n_workers": n_workers,
        "tasks_per_worker": n_tasks / n_workers,
    }


if __name__ == "__main__":
    # Demo / test
    print(f"System has {os.cpu_count()} CPUs")
    print(f"Using {get_optimal_workers()} workers")

    # Simple test
    def square(x):
        time.sleep(0.1)  # Simulate work
        return x**2

    tasks = list(range(20))

    print("\nSequential:")
    start = time.time()
    seq_results = [square(x) for x in tasks]
    seq_time = time.time() - start
    print(f"  Time: {seq_time:.2f}s")

    print("\nParallel:")
    start = time.time()
    par_results = run_parallel(square, tasks, desc="Squaring numbers")
    par_time = time.time() - start

    print(f"\nSpeedup: {seq_time / par_time:.1f}x")
    print(f"Results match: {seq_results == par_results}")
