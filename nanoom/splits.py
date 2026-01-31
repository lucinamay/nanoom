"""
balancing, splitting

"""

import json
import logging as lg
import os
from datetime import datetime
from typing import Sequence

import jax.numpy as jnp
import polars as pl
import pulp  # https://coin-or.github.io/pulp/ for docs
from jax.typing import ArrayLike
from sklearn.model_selection import StratifiedGroupKFold

# === balancing ===


def __ensure_base0(clusters: Sequence[int]) -> Sequence[int]:
    while 0 not in clusters:  # ensure cluster numbers are base 0
        clusters = list(jnp.array(clusters) - 1)
    return clusters


def _datapoints_vs_tasks_array(
    df: pl.DataFrame, datapoint_id_col: str, task_name_col: str, task_val_col: str
) -> pl.DataFrame:
    return df.group_by(datapoint_id_col).agg(
        pl.struct([task_name_col, task_val_col]).alias("tasks")
    )


def tasks_vs_clusters_array_polars(
    df: pl.DataFrame, task_cols: str, cluster_col: str
) -> pl.DataFrame | jnp.ndarray:
    """
    Create a cross-tabulation 2D numpy array counting the # data points per task, per cluster

    Args:
        df (pl.DataFrame): dataframe with task columns and cluster column
        task_cols (str): name of the column containing tasks
        cluster_col (str): name of the column containing clusters
    Returns:
        pl.DataFrame.to_numpy(): 2D array of shape (num_tasks+1, num_clusters)

    Comment:
        In the returned array
        - each column is a unique initial cluster
        - each row is a unique task
        (except the first row, which is the total # objects in the cluster)
        This is the format requrired by the balancing algorithm
        see: https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/660581be9138d231618d6047/original/readme-md.md

    """
    # @TODO: figure out how to do it nice and polars-y

    raise NotImplementedError


def _task_vs_clusters_array_jax(
    datapoints_tasks_array: jnp.ndarray, clusters: jnp.ndarray
) -> jnp.ndarray:
    """
    Create a cross-tabulation 2D numpy array counting the # data points per task, per cluster

    Args:
        tasks (2d-Array): list of task indices per data point (e.g. one-hot encoding matrix for multilabels)
        clusters (list): list of cluster indices per data point (e.g. clusters per compound)
    Returns:
        jnp.ndarray: 2D array of shape (num_tasks+1, num_clusters)

    Comment:
        In the returned array
        - each column is a unique initial cluster
        - each row is a unique task (except the first row, which is the
          total # objects in the cluster)
        This is the format requrired by the balancing algorithm
        see: https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/660581be9138d231618d6047/original/readme-md.md

    """

    task_vs_clusters = jnp.zeros((len(datapoints_tasks_array) + 1, len(clusters)))
    # first row is total number of items per cluster
    task_vs_clusters.at[0, :].set(
        jnp.array([jnp.sum(jnp.array(clusters) == i) for i in clusters.unique()])
    )
    # get the number of items per cluster that have non-null task values
    for i, task_list in enumerate(datapoints_tasks_array):
        for task in task_list:
            task_vs_clusters = task_vs_clusters.at[1 + i, clusters[i]].add(1)

    return task_vs_clusters


def _balance_data_from_tasks_vs_clusters_array(
    tasks_vs_clusters_array: jnp.ndarray,
    sizes: list[int] = [1],
    equal_weight_perc_compounds_as_tasks: bool = False,
    relative_gap: int = 0,
    time_limit_seconds: float = 60 * 60,
    max_N_threads: int = int((os.cpu_count() or 1) // 1.2) + 1,
    verbose: bool = False,
) -> list:
    """
    Linear programming function needed to balance the data while merging clusters
    taken from https://doi.org/10.26434/chemrxiv-2022-m8l33-v3

    Args:
        tasks_vs_clusters_array : 2D jnp.array
            - the cross-tabulation of the number of data points per cluster, per task.
            - columns represent unique clusters.
            - rows represent tasks, except the first row, which represents the number of records (or compounds).
            - Optionally, instead of the number of data points, the provided array may contain the *percentages*
                of data points _for the task across all clusters_ (i.e. each *row*, NOT column, may sum to 1).
            IMPORTANT: make sure the array has 2 dimensions, even if only balancing the number of data records,
                so there is only 1 row. This can be achieved by setting ndmin = 2 in the jnp.array function.
        sizes : list
            - list of the desired final sizes (will be normalised to fractions internally).
        equal_weight_perc_compounds_as_tasks : bool
            - if True, matching the % records will have the same weight as matching the % data of individual tasks.
            - if False, matching the % records will have a weight X times larger than the X tasks.
        relative_gap : float
            - the relative gap between the absolute optimal objective and the current one at which the solver
            stops and returns a solution. Can be very useful for cases where the exact solution requires
            far too long to be found to be of any practical use.
            - set to 0 to obtain the absolute optimal solution (if reached within the time_limit_seconds)
        time_limit_seconds : int
            - the time limit in seconds for the solver (by default set to 1 hour)
            - after this time, whatever solution is available is returned
        max_N_threads : int
            - the maximal number of threads to be used by the solver.
            - it is advisable to set this number as high as allowed by the available resources.

    Returns:
        List (of length equal to the number of columns of tasks_vs_clusters_array) of final cluster identifiers
            (integers, numbered from 1 to len(sizes)), mapping each unique initial cluster to its final cluster.

        - Example: if sizes == [20, 10, 70], the output will be a list like [3, 3, 1, 2, 1, 3...], where
        '1' represents the final cluster of relative size 20, '2' the one of relative size 10, and '3' the
        one of relative size 70.


    Note: from https://doi.org/10.26434/chemrxiv-2022-m8l33-v3, original code can be found at
    https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/660581be9138d231618d604c/original/balance-data-from-tasks-vs-clusters-array-pulp-py.py
    """
    # Calculate the fractions from sizes

    fractional_sizes = sizes / jnp.sum(sizes)
    S = len(sizes)

    # Normalise the data matrix
    tasks_vs_clusters_array = tasks_vs_clusters_array / tasks_vs_clusters_array.sum(
        axis=1, keepdims=True
    )

    # Find the number of tasks + compounds (M) and the number of initial clusters (N)
    M, N = tasks_vs_clusters_array.shape
    if S > N:
        errormessage = (
            "The requested number of new clusters to make ("
            + str(S)
            + ") cannot be larger than the initial number of clusters ("
            + str(N)
            + "). Please review."
        )
        raise ValueError(errormessage)

    if relative_gap < 0:
        errormessage = f"relative gap should be 0 or positive, is {relative_gap}"
        raise ValueError(errormessage)

    # Given matrix A (M x N) of fraction of data per cluster, assign each cluster to one of S final ML subsets,
    # so that the fraction of data per ML subset is closest to the corresponding fraction_size.
    # The weights on each ML subset (WML, S x 1) are calculated from fractional_sizes harmonic-mean-like.
    # The weights on each task (WT, M x 1) are calculated as requested by the user.
    # In the end: argmin SUM(ABS((A.X-T).WML).WT)
    # where X is the (N x S) binary solution matrix
    # where T is the (M x S) matrix of target fraction sizes (repeat of fractional_sizes)
    # constraint: assign one cluster to one and only one final ML subset
    # i.e. each row of X must sum to 1

    A = jnp.copy(tasks_vs_clusters_array)

    # Create WT = obj_weights
    if (M > 1) & (not equal_weight_perc_compounds_as_tasks):
        obj_weights = jnp.array([M - 1] + [1] * (M - 1))
    else:
        obj_weights = jnp.array([1] * M)

    obj_weights = obj_weights / jnp.sum(obj_weights)

    # Create WML
    sk_harmonic = (1 / fractional_sizes) / jnp.sum(1 / fractional_sizes)

    # Create the pulp model
    prob = pulp.LpProblem("Data_balancing", pulp.LpMinimize)

    # Create the pulp variables
    # x_names represent clusters, ML_subsets, and are binary variables
    x_names = ["x_" + str(i) for i in range(N * S)]
    x = [
        pulp.LpVariable(x_names[i], lowBound=0, upBound=1, cat="Integer")
        for i in range(N * S)
    ]
    # X_names represent tasks, ML_subsets, and are continuous positive variables
    X_names = ["X_" + str(i) for i in range(M * S)]
    X = [
        pulp.LpVariable(X_names[i], lowBound=0, cat="Continuous") for i in range(M * S)
    ]

    # Add the objective to the model

    obj = []
    coeff = []
    for m in range(S):
        for t in range(M):
            obj.append(X[m * M + t])
            coeff.append(sk_harmonic[m] * obj_weights[t])

    prob += pulp.LpAffineExpression([(obj[i], coeff[i]) for i in range(len(obj))])

    # Add the constraints to the model

    # Constraints forcing each cluster to be in one and only one ML_subset
    for c in range(N):
        prob += pulp.LpAffineExpression([(x[c + m * N], +1) for m in range(S)]) == 1

    # Constraints forcing each ML_subset to be non-empty
    for m in range(S):
        prob += (
            pulp.LpAffineExpression([(x[i], +1) for i in range(m * N, (m + 1) * N)])
            >= 1
        )

    # Constraints related to the ABS values handling, part 1 and 2
    for m in range(S):
        for t in range(M):
            cs = [c for c in range(N) if A[t, c] != 0]
            prob += (
                pulp.LpAffineExpression([(x[c + m * N], A[t, c]) for c in cs])
                - X[m * M + t]
                <= fractional_sizes[m]
            )
            prob += (
                pulp.LpAffineExpression([(x[c + m * N], A[t, c]) for c in cs])
                + X[m * M + t]
                >= fractional_sizes[m]
            )

    # Solve the model
    prob.solve(
        pulp.PULP_CBC_CMD(
            gapRel=relative_gap,
            timeLimit=time_limit_seconds,
            threads=max_N_threads,
            msg=verbose,
        )
    )

    # Extract the solution

    list_binary_solution = [pulp.value(x[i]) for i in range(N * S)]
    list_initial_cluster_indices = [
        (list(range(N)) * S)[i] for i, li in enumerate(list_binary_solution) if li == 1
    ]
    list_final_ML_subsets = [
        (list((1 + jnp.repeat(range(S), N)).astype("int64")))[i]
        for i, li in enumerate(list_binary_solution)
        if li == 1
    ]
    mapping = [
        x for _, x in sorted(zip(list_initial_cluster_indices, list_final_ML_subsets))
    ]

    return mapping


def balanced_split(
    data: pl.DataFrame,
    clusters: Sequence[int],
    to_split: str = "molecule",
):
    """Balanced splitting of data based on clusters and tasks

    Args:
        data (pl.DataFrame): input data, with structure/sequence information on the to_split choice
        to_split (str): what to split. Either "molecule" or "protein". Default is "molecule".

    Returns:
        dict: mapping of split names to lists of data point identifiers
    """
    match to_split:
        case "molecule":
            lg.info("grouping by cid and aggregating unique `target_id`s as tasks")
            datapoint_tasks = data.group_by("cid").agg(
                pl.col("target_id")
                .unique()
                .alias("tasks")  # @TODO: check if we should double-count double targets
            )
            ids: list[str] = datapoint_tasks["cid"].to_list()
            tasks: list[list[str]] = datapoint_tasks["tasks"].to_list()
            datapoints = Compounds().retrieve(
                ids=ids, with_mols=True, originals=True, source="papyrus"
            )
            # clusters = clustering_mols(
            #     datapoints["mol"].to_list(),
            #     clustering_sphere_exclusion_rdkit,
            # )

        # case "protein":
        #     lg.info("grouping by tid and aggregating unique `cid`s as tasks")
        #     datapoint_tasks = data.group_by("tid").agg(
        #         pl.col("cid")
        #         .unique()
        #         .alias("tasks")  # @TODO: check if we should double-count double targets
        #     )
        #     ids: list[str] = datapoint_tasks["tid"].to_list()
        #     tasks: list[list[str]] = datapoint_tasks["tasks"].to_list()
        #     datapoints = Proteins().retrieve("papyrus", original_ids=ids)
        #     clusters = prot_clustering(
        #         datapoints["sequence"].to_list(),
        #         sphere_exclusion_clustering,
        #         batch_descriptor=batch_esm,
        #     )
        #     pass
        case "molecule_protein":
            raise NotImplementedError
        case _:
            raise ValueError(f"Unknown datapoint type: {to_split}")

    tasks_vs_clusters = _task_vs_clusters_array(tasks, list(clusters))
    mappings: jnp.ndarray = jnp.array(
        _balance_data_from_tasks_vs_clusters_array(tasks_vs_clusters, sizes=[80, 20])
    )
    # @TODO: check if 6x 20 would be better
    jnp.save(TMP.base / f"{to_split}_split", mappings)
    return mappings


# ============== based on gbmt-split ==============


def _regression_binning(n_bins: int = 5):
    raise NotImplementedError


def globally_balanced_split(
    clusters: jnp.ndarray[int], tasks: jnp.ndarray, alias: str | None = None
):
    """adapted from https://github.com/sohviluukkonen/gbmt-splits/blob/main/gbmtsplits/split.py"""
    lg.warning(
        "if you want more than 1 split, you might want to change the seed of clustering"
    )
    tasks_vs_clusters_array = _tasks_vs_clusters_array()


def _sklearn_split(X, y, group_on, random_state: int, n_splits: int = 5) -> jnp.ndarray:
    """returns jnp.ndarray of shape group_on.shape[0], n_splits"""
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    split_idx = jnp.zeros((X.shape[0], n_splits))
    for k, (train_idx, test_idx) in enumerate(
        splitter.split(
            X=X,
            y=y,
            groups=group_on,
        )
    ):
        split_idx[test_idx, k] = 1
    return split_idx


def split(
    clusters,
    tasks,
    type: Literal["regression", "classification"] = "regression",
    method: str = "sklearn",
    k_folds: int = 5,
    random_seed: int = 0,
) -> jnp.ndarray():
    jnp.random.seed(random_seed)
    match method:
        case "sklearn":
            _sklearn_split()
    pass


# ========================================================


# def main():
#     configure_pystow_logging(logname="splitting", level="debug")
#     data = pl.read_parquet(CLEANDATA.join("papyrus") / "drug.parquet")[:10]
#     mappings = balanced_split(data, to_split="molecule")
#     print(mappings)


# if __name__ == "__main__":
#     main()
#     # print("Hello world")
