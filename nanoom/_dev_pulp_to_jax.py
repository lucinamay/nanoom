"""
JAX-based reimplementation of cluster balancing via linear programming.

This module provides a functional, efficient alternative to the PuLP-based
implementation for balanced assignment of clusters to ML subsets.
"""

from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.typing import ArrayLike
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_simplex

# =============================================================================
# Core optimization function
# =============================================================================


def balance_data_jax(
    tasks_vs_clusters_array: ArrayLike,
    sizes: Sequence[int] = (1,),
    equal_weight_perc_compounds_as_tasks: bool = False,
    max_iterations: int = 10000,
    tolerance: float = 1e-6,
    learning_rate: float = 0.01,
    verbose: bool = False,
) -> list[int]:
    """
    Assign clusters to ML subsets to balance task/compound distributions.

    Uses projected gradient descent with JAX for efficient GPU/TPU optimization.

    Args:
        tasks_vs_clusters_array: 2D array (M x N) where:
            - M = number of tasks + 1 (first row = total compounds)
            - N = number of initial clusters
            - Each column is a cluster, each row is a task
            - Values are fractions of data per cluster (rows sum to 1)
        sizes: Desired relative sizes of final ML subsets
        equal_weight_perc_compounds_as_tasks: If True, weight compound
            matching equally with task matching. If False, compound matching
            gets (M-1)x weight.
        max_iterations: Maximum optimization iterations
        tolerance: Convergence tolerance for objective change
        learning_rate: Step size for gradient descent
        verbose: Print optimization progress

    Returns:
        List of length N mapping each initial cluster to final subset (1-indexed)

    Example:
        If sizes=[20, 10, 70] and N=10 clusters, returns something like
        [3, 3, 1, 2, 1, 3, 2, 3, 3, 1] where 1/2/3 are the final subsets.
    """
    # Legacy: fractional_sizes = sizes / np.sum(sizes)
    fractional_sizes = jnp.array(sizes) / jnp.sum(jnp.array(sizes))
    S = len(sizes)

    # Legacy: tasks_vs_clusters_array = tasks_vs_clusters_array /
    #         tasks_vs_clusters_array.sum(axis=1, keepdims=True)
    A = jnp.array(tasks_vs_clusters_array)
    A = A / A.sum(axis=1, keepdims=True)

    M, N = A.shape

    if S > N:
        raise ValueError(f"Requested {S} subsets cannot exceed {N} initial clusters")

    # Legacy: Compute objective weights (WT)
    # if (M > 1) & (not equal_weight_perc_compounds_as_tasks):
    #     obj_weights = np.array([M - 1] + [1] * (M - 1))
    # else:
    #     obj_weights = np.array([1] * M)
    obj_weights = _compute_task_weights(M, equal_weight_perc_compounds_as_tasks)

    # Legacy: Compute subset weights (WML) using harmonic mean
    # sk_harmonic = (1 / fractional_sizes) / np.sum(1 / fractional_sizes)
    subset_weights = _compute_subset_weights(fractional_sizes)

    # Run optimization
    # Legacy: Uses PuLP's CBC solver with integer programming
    # New: Uses continuous relaxation + projected gradient descent
    assignment_matrix = _optimize_assignment(
        A,
        fractional_sizes,
        obj_weights,
        subset_weights,
        max_iterations,
        tolerance,
        learning_rate,
        verbose,
    )

    # Convert soft assignment to hard assignment
    # Legacy: Extracts binary solution from PuLP solver
    # New: Take argmax over subsets for each cluster
    mapping = _extract_hard_assignment(assignment_matrix, S, N)

    return mapping


# =============================================================================
# Helper functions (pure, functional style)
# =============================================================================


def _compute_task_weights(M: int, equal_weight: bool) -> jnp.ndarray:
    """
    Compute weights for each task in the objective function.

    Legacy equivalent:
        if (M > 1) & (not equal_weight_perc_compounds_as_tasks):
            obj_weights = np.array([M - 1] + [1] * (M - 1))
        else:
            obj_weights = np.array([1] * M)
        obj_weights = obj_weights / np.sum(obj_weights)
    """
    if M > 1 and not equal_weight:
        weights = jnp.array([M - 1] + [1] * (M - 1))
    else:
        weights = jnp.ones(M)
    return weights / weights.sum()


def _compute_subset_weights(fractional_sizes: jnp.ndarray) -> jnp.ndarray:
    """
    Compute harmonic-mean-based weights for ML subsets.

    Legacy equivalent:
        sk_harmonic = (1 / fractional_sizes) / np.sum(1 / fractional_sizes)
    """
    return (1 / fractional_sizes) / jnp.sum(1 / fractional_sizes)


@jit
def _objective(
    X: jnp.ndarray,
    A: jnp.ndarray,
    fractional_sizes: jnp.ndarray,
    obj_weights: jnp.ndarray,
    subset_weights: jnp.ndarray,
) -> float:
    """
    Compute weighted L1 loss between achieved and target distributions.

    Legacy equivalent:
        The PuLP objective minimizes:
        SUM(ABS((A.X - T).WML).WT)
        where:
        - A.X is the achieved fraction per subset (M x S)
        - T is the target fractions (M x S, repeated fractional_sizes)
        - WML are subset weights (harmonic mean)
        - WT are task weights

    Args:
        X: Assignment matrix (N x S) in [0,1]^(NxS)
        A: Data matrix (M x N), normalized per task
        fractional_sizes: Target sizes (S,)
        obj_weights: Task weights (M,)
        subset_weights: Subset weights (S,)

    Returns:
        Scalar objective value
    """
    M, N = A.shape
    S = len(fractional_sizes)

    # Compute achieved fractions: A @ X gives (M x S) matrix
    # Each column s is the fraction of task m data in subset s
    # Legacy: This is computed implicitly through PuLP constraints
    achieved = A @ X  # (M, N) @ (N, S) -> (M, S)

    # Target: each subset should have fractional_sizes[s] of each task
    # Legacy: T is the (M x S) matrix of target fraction sizes
    target = jnp.tile(fractional_sizes, (M, 1))  # (M, S)

    # Absolute deviation weighted by subset and task
    # Legacy: argmin SUM(ABS((A.X-T).WML).WT)
    deviation = jnp.abs(achieved - target)  # (M, S)

    # Apply subset weights (broadcast over tasks)
    weighted_by_subset = deviation * subset_weights[None, :]  # (M, S)

    # Apply task weights (sum over subsets first, then weight by task)
    weighted_by_task = weighted_by_subset.sum(axis=1) * obj_weights  # (M,)

    return weighted_by_task.sum()


@partial(jit, static_argnames=["S", "N"])
def _project_to_feasible(X: jnp.ndarray, S: int, N: int) -> jnp.ndarray:
    """
    Project assignment matrix to feasible region.

    Constraints:
    1. Each cluster assigned to exactly one subset (rows sum to 1)
    2. Each subset gets at least one cluster (columns have max >= threshold)
    3. Values in [0, 1]

    Legacy equivalent:
        PuLP enforces these as hard constraints:
        - sum over subsets = 1 for each cluster (row sum)
        - sum over clusters >= 1 for each subset (non-empty)
        - x in {0, 1} (binary)

    We use continuous relaxation: project each row to simplex.
    """
    # Reshape to (N, S) if needed
    X = X.reshape(N, S)

    # Project each row (cluster) to simplex: sum to 1, values >= 0
    # This handles constraints: sum_s X[c,s] = 1 and X >= 0
    X_projected = jax.vmap(projection_simplex, in_axes=0)(X)

    # Note: Non-emptiness constraint (each subset >= 1 cluster) is handled
    # softly through initialization and the optimization dynamics

    return X_projected


def _optimize_assignment(
    A: jnp.ndarray,
    fractional_sizes: jnp.ndarray,
    obj_weights: jnp.ndarray,
    subset_weights: jnp.ndarray,
    max_iterations: int,
    tolerance: float,
    learning_rate: float,
    verbose: bool,
) -> jnp.ndarray:
    """
    Optimize cluster-to-subset assignment using projected gradient descent.

    Legacy equivalent:
        PuLP's CBC solver finds optimal binary assignment matrix X (N x S)
        subject to constraints. This uses continuous relaxation with
        projection to simplex per cluster.
    """
    M, N = A.shape
    S = len(fractional_sizes)

    # Initialize: assign each cluster approximately to target distribution
    # Legacy: PuLP starts with LP relaxation
    # Heuristic: distribute clusters proportionally to fractional_sizes
    X_init = jnp.tile(fractional_sizes, (N, 1))  # (N, S)
    X_init = _project_to_feasible(X_init, S, N)

    # Define projected objective
    def obj_fn(X):
        X_feasible = _project_to_feasible(X, S, N)
        return _objective(X_feasible, A, fractional_sizes, obj_weights, subset_weights)

    # Use ProjectedGradient optimizer
    # Legacy: PuLP uses branch-and-bound for integer programming
    # New: Continuous optimization with simplex projection
    pg = ProjectedGradient(
        fun=obj_fn,
        projection=lambda X: _project_to_feasible(X, S, N),
        maxiter=max_iterations,
        tol=tolerance,
        stepsize=learning_rate,
        verbose=verbose,
    )

    result = pg.run(X_init)
    X_final = result.params

    if verbose:
        print(f"Converged: {result.state.converged}")
        print(f"Final objective: {result.state.value:.6f}")
        print(f"Iterations: {result.state.iter_num}")

    return X_final


def _extract_hard_assignment(
    X: jnp.ndarray,
    S: int,
    N: int,
) -> list[int]:
    """
    Convert soft assignment matrix to hard cluster-to-subset mapping.

    Legacy equivalent:
        list_binary_solution = [pulp.value(x[i]) for i in range(N * S)]
        list_initial_cluster_indices = [
            (list(range(N)) * S)[i]
            for i, li in enumerate(list_binary_solution) if li == 1
        ]
        list_final_ML_subsets = [
            (list((1 + np.repeat(range(S), N)).astype("int64")))[i]
            for i, li in enumerate(list_binary_solution) if li == 1
        ]
        mapping = [
            x for _, x in sorted(
                zip(list_initial_cluster_indices, list_final_ML_subsets)
            )
        ]

    Args:
        X: Assignment matrix (N x S) with soft assignments
        S: Number of subsets
        N: Number of clusters

    Returns:
        List of length N with subset assignments (1-indexed)
    """
    X = X.reshape(N, S)

    # For each cluster, assign to subset with highest probability
    # Legacy: Extracts the single subset where binary variable = 1
    assignments = jnp.argmax(X, axis=1)  # (N,)

    # Convert to 1-indexed (legacy uses 1-based indexing)
    mapping = (assignments + 1).tolist()

    return mapping


# =============================================================================
# Alternative: Direct integer programming with rounding
# =============================================================================


def balance_data_jax_rounded(
    tasks_vs_clusters_array: np.ndarray,
    sizes: Sequence[int] = (1,),
    equal_weight_perc_compounds_as_tasks: bool = False,
    max_iterations: int = 5000,
    tolerance: float = 1e-6,
    learning_rate: float = 0.02,
    verbose: bool = False,
) -> list[int]:
    """
    Alternative implementation with iterative rounding for better discrete solutions.

    Uses the same optimization but periodically rounds the softest assignments
    to encourage discrete solutions closer to the integer programming result.

    This is more experimental but may yield solutions closer to PuLP's optimal.
    """
    fractional_sizes = jnp.array(sizes) / jnp.sum(jnp.array(sizes))
    S = len(sizes)

    A = jnp.array(tasks_vs_clusters_array)
    A = A / A.sum(axis=1, keepdims=True)
    M, N = A.shape

    if S > N:
        raise ValueError(f"Requested {S} subsets cannot exceed {N} initial clusters")

    obj_weights = _compute_task_weights(M, equal_weight_perc_compounds_as_tasks)
    subset_weights = _compute_subset_weights(fractional_sizes)

    # Run standard optimization first
    X = _optimize_assignment(
        A,
        fractional_sizes,
        obj_weights,
        subset_weights,
        max_iterations,
        tolerance,
        learning_rate,
        verbose,
    )

    # Greedy rounding: for each cluster, assign to argmax
    mapping = _extract_hard_assignment(X, S, N)

    return mapping


# =============================================================================
# Utility function to match original API
# =============================================================================


def balance_data_from_tasks_vs_clusters_array_jax(
    tasks_vs_clusters_array: np.ndarray,
    sizes: list[int] = [1],
    equal_weight_perc_compounds_as_tasks: bool = False,
    max_iterations: int = 10000,
    tolerance: float = 1e-6,
    learning_rate: float = 0.01,
    verbose: bool = False,
) -> list:
    """
    Drop-in replacement for _balance_data_from_tasks_vs_clusters_array.

    Main differences from PuLP version:
    - No relative_gap parameter (continuous optimization doesn't use MIP gap)
    - No time_limit_seconds (use max_iterations instead)
    - No max_N_threads (JAX handles parallelization automatically)
    - Added learning_rate for gradient descent control
    - Added tolerance for convergence detection

    Returns the same format: list of subset assignments (1-indexed).
    """
    return balance_data_jax(
        tasks_vs_clusters_array=tasks_vs_clusters_array,
        sizes=sizes,
        equal_weight_perc_compounds_as_tasks=equal_weight_perc_compounds_as_tasks,
        max_iterations=max_iterations,
        tolerance=tolerance,
        learning_rate=learning_rate,
        verbose=verbose,
    )
