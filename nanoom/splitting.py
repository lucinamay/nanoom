"""
balancing, splitting

"""

import logging
import os
from typing import Literal, Sequence

import numpy as np
import polars as pl
import polars.selectors as cs
import pulp  # https://coin-or.github.io/pulp/ for docs
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedGroupKFold

lg = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


def _task_type(series=pl.Series) -> Literal["regression", "classification_onehot"]:
    # if float-numerical:
    if series.dtype.is_float():  # ty: ignore
        return "regression"
    elif series.dtype.is_int() and set(series.to_list()).difference({0, 1}) == set():  # ty: ignore
        return "classification_onehot"
    else:
        raise NotImplementedError("Other types not yet implemented")


def _task_vs_clusters_df(
    df: pl.DataFrame,
    x_col: str = "activity_id",
    task_cols: str | Sequence[str] = ["pchembl_value_mean"],
    cluster_col: str = "cluster",
    n_bins_for_regression: int | None = 5,
) -> pl.DataFrame:
    """
    Create a cross-tabulation 2D numpy array counting the # data points per task, per cluster

    Args:
        df (pl.DataFrame): input dataframe
        task_cols (Sequence[str]): columns representing tasks. should be a single column
        cluster_col (str): column representing clusters
    Returns:
        np.ndarray: 2D array of shape (num_tasks+1, num_clusters)

    Comment:
        In the returned array
        - each column is a unique initial cluster
        - each row is a unique task
        (except the first row, which is the total # objects in the cluster)
        This is the format requrired by the balancing algorithm
        see: https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/660581be9138d231618d6047/original/readme-md.md

    """
    # @TODO: figure out how to do it nice and polars-y
    task_cols = list(task_cols)
    df = df.lazy().select([x_col, cluster_col] + task_cols).collect()
    types = set(
        (
            _task_type(df.select(col).lazy().collect().get_column(col))
            for col in task_cols
        )
    )
    if len(types) > 1:
        raise NotImplementedError("No support for multiple types of tasks (yet?)")
    type = list(types)[0]
    match type:
        case "regression":
            if not n_bins_for_regression:
                lg.warning(
                    "`n_bins_for_regression` == None,but is regression task: using 5 bins"
                )
                n_bins_for_regression = 5
            df = df.with_columns(
                pl.col(col)
                .qcut(5, labels=[f"bin_{i}" for i in range(5)])
                .alias(f"{col}_binned")
                for col in task_cols
            ).drop(task_cols)
            to_pivot_on = cs.ends_with("_binned")
        case "classification_onehot":
            df = (
                df.unpivot(
                    on=task_cols,
                    index=x_col,
                    variable_name="class",
                    value_name="value",
                )
                .drop("value")
                .cast({"class": pl.Categorical})
            )
            to_pivot_on = "class"

    # now that the task is one long-column of possibilities, we pivot it to the
    # format required for the split-balancing script of Tricario et al.
    return (
        df.pivot(
            on=to_pivot_on,
            index=cluster_col,
            values=x_col,
            aggregate_function="len",
            sort_columns=True,
        )
        .join(
            df.group_by(cluster_col).agg(pl.len().alias("number")),
            on=cluster_col,
            how="left",
        )
        .sort(cluster_col)
        .select(
            # ensure the order is cluster first,
            pl.col(cluster_col),
            # then the number of datapoints per cluster,
            pl.col("number"),
            # then tasks
            pl.exclude("number", cluster_col),
        )
    )


def _balance_splits_from_tasks_vs_clusters_array(
    tasks_vs_clusters_array: np.ndarray,
    split_sizes: list[float] = [0.2, 0.2, 0.2, 0.2, 0.2],
    equal_weight_perc_compounds_as_tasks: bool = False,
    relative_gap: int = 0,
    time_limit_seconds: float = 60 * 60,
    n_jobs: int = int((os.cpu_count() or 1) // 1.2) + 1,
    verbose: bool = False,
) -> np.ndarray:
    """
    Linear programming function needed to balance the data while merging clusters
    taken from https://doi.org/10.26434/chemrxiv-2022-m8l33-v3

    Args:
        tasks_vs_clusters_array : 2D np.array
            - the cross-tabulation of the number of data points per cluster, per task.
            - columns represent unique clusters.
            - rows represent tasks, except the first row, which represents the number of records (or compounds).
            - Optionally, instead of the number of data points, the provided array may contain the *percentages*
                of data points _for the task across all clusters_ (i.e. each *row*, NOT column, may sum to 1).
            IMPORTANT: make sure the array has 2 dimensions, even if only balancing the number of data records,
                so there is only 1 row. This can be achieved by setting ndmin = 2 in the np.array function.
        split_sizes : list
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
        n_jobs : int
            - the maximal number of threads to be used by the solver.
            - it is advisable to set this number as high as allowed by the available resources.

    Returns:
        List (of length equal to the number of columns of tasks_vs_clusters_array) of final cluster identifiers
            (integers, numbered from 0 to len(sizes)), mapping each unique initial cluster to its final cluster.

        - Example: if split_sizes == [20, 10, 70], the output will be a list like [2, 2, 0, 1, 0, 2...], where
        '0' represents the final cluster of relative size 20, '1' the one of relative size 10, and '2' the
        one of relative size 70.


    Note: from https://doi.org/10.26434/chemrxiv-2022-m8l33-v3, original code can be found at
    https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/660581be9138d231618d604c/original/balance-data-from-tasks-vs-clusters-array-pulp-py.py
    """
    lg.debug(
        f"Splitting data with {n_jobs} jobs and a time limit of {time_limit_seconds} seconds"
    )
    # Calculate the fractions from sizes
    fractional_sizes = split_sizes / np.sum(split_sizes)
    S = len(split_sizes)

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

    A = np.copy(tasks_vs_clusters_array)

    # Create WT = obj_weights
    if (M > 1) & (not equal_weight_perc_compounds_as_tasks):
        obj_weights = np.array([M - 1] + [1] * (M - 1))
    else:
        obj_weights = np.array([1] * M)

    obj_weights = obj_weights / np.sum(obj_weights)

    # Create WML
    sk_harmonic = (1 / fractional_sizes) / np.sum(1 / fractional_sizes)

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
            threads=n_jobs,
            msg=verbose,
        )
    )

    # Extract the solution

    list_binary_solution = [pulp.value(x[i]) for i in range(N * S)]
    list_initial_cluster_indices = [
        (list(range(N)) * S)[i] for i, li in enumerate(list_binary_solution) if li == 1
    ]
    list_final_ML_subsets = [
        (list((1 + np.repeat(range(S), N)).astype("int64")))[i]
        for i, li in enumerate(list_binary_solution)
        if li == 1
    ]
    mapping = np.array(
        [x for _, x in sorted(zip(list_initial_cluster_indices, list_final_ML_subsets))]
    )

    return mapping - 1


def globally_balanced_split_polars(
    df: pl.DataFrame,
    split_sizes: Sequence[float] = [0.2, 0.2, 0.2, 0.2, 0.2],
    x_col: str = "activity_id",
    y_cols: str | Sequence[str] = "pchembl_value_mean",
    cluster_col: str = "cluster",
    n_bins_for_regression: int | None = 5,
    alias: str = "split",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """splits the data, returning a cluster -> split assignment mapping as a dictionary

    note: inspired by https://github.com/sohviluukkonen/gbmt-splits/blob/main/gbmtsplits/split.py
    implementation of the tricario et al split
    """
    lg.info(
        "if you want more than 1 split, you might want to change "
        "the seed of clustering for each run"
    )

    # rows: clusters, cols: tasks
    clusters_vs_tasks_df = _task_vs_clusters_df(
        df=df,
        x_col=x_col,
        task_cols=y_cols,
        cluster_col=cluster_col,
        n_bins_for_regression=n_bins_for_regression,
    ).to_numpy()

    # the column that is the explicit cluster numbers
    clusters = clusters_vs_tasks_df[:, 0]

    # the array that the next code requires, with tasks as rows and clusters as cols
    task_vs_clusters_array = clusters_vs_tasks_df[:, 1:].T

    split_assignments = _balance_splits_from_tasks_vs_clusters_array(
        task_vs_clusters_array,
        split_sizes=list(split_sizes),
        **kwargs,
    )
    return clusters, split_assignments
    # mapping = pl.DataFrame(
    #     [
    #         pl.Series(name=cluster_col, values=clusters),
    #         pl.Series(name=alias, values=split_assignments),
    #     ]
    # )
    # return df.lazy().join(mapping.lazy(), on=cluster_col, how="left").collect()


def sklearn_split(
    X: NDArray,
    y: NDArray,
    group_on: NDArray,
    random_state: int,
    n_splits: int = 5,
    n_bins_for_regression: int | None = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """returns jnp.ndarray of shape group_on.shape[0], n_splits"""
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    assert isinstance(group_on, np.ndarray), "group_on must be a np.ndarray"
    assert isinstance(X, np.ndarray), "X must be a np.ndarray"
    assert isinstance(y, np.ndarray), "y must be a np.ndarray"

    y = np.array(y)

    if y.dtype.kind in "fc":  # float or complex
        assert n_bins_for_regression is not None, (
            "n_bins_for_regression cannot be None for regression tasks"
        )
        n_bins = n_bins_for_regression
        y = np.searchsorted(
            np.quantile(y, np.linspace(0, 1, n_bins + 1))[1:-1], y, side="right"
        )

    split_idx = np.zeros((X.shape[0], n_splits))
    for k, (train_idx, test_idx) in enumerate(
        splitter.split(
            X=X,
            y=y,
            groups=group_on,
        )
    ):
        split_idx[test_idx, k] = 1
    # assert only one column per row is 1
    assert np.all(split_idx.sum(axis=1) == 1), RuntimeError(
        "Each row should be assigned to only one test split"
    )
    # now squish them by assigning np.nan to the 0s, and the column number to the 1s
    split_idx = np.argmax(split_idx, axis=1)
    return group_on, split_idx


def split(
    df,
    X_col: str,
    y_cols: Sequence[str] | str,
    cluster_col: str,
    n_splits: int,
    method: Literal["tricario", "sklearn"],
    *args,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Takes a regular dataframe with various X-y columns
    (these do not have to be unique X's) and turns them into
    a tuple of the original cluster values to test split indices
    Args:
        df: polars DataFrame
        X_col: str, name of the column representing X values (e.g. activity_id)
        y_cols: Sequence[str] | str, name(s) of the column(s) representing y values (e.g. pchembl_value_mean)
        cluster_col: str, name of the column representing cluster assignments
        n_splits: int, number of splits to create
        method: Literal["tricario", "sklearn"], method to use for splitting
    Returns:
        tuple[np.ndarray, np.ndarray]: (clusters, split assignments)
    0th array is the original cluster values
    1st array is the split assignments to test (of shape (num_clusters, n_splits
    """
    match method:
        case "tricario":
            return globally_balanced_split_polars(
                df=df,
                split_sizes=[1 / n_splits] * n_splits,
                x_col=X_col,
                y_cols=y_cols,
                cluster_col=cluster_col,
                alias="split",
                **kwargs,
            )
        case "sklearn":
            lg.warning("`method=sklearn` validity not tested yet!")
            y = df.lazy().select(y_cols).collect()
            # print(y[:10])
            # print(y.to_numpy()[:10])
            X = (
                df.select(X_col)
                .fill_null(strategy="forward")  # not returned anyway
                .lazy()
                .collect()[X_col]
                .to_numpy()
            )
            group_on = df.select(cluster_col).lazy().collect().to_numpy().squeeze()
            if not "random_state" in kwargs:
                kwargs["random_state"] = 0
            return sklearn_split(
                X=X,
                y=y.to_numpy()[:, 0],
                group_on=group_on,
                n_splits=n_splits,
                **kwargs,
            )
    raise NotImplementedError("general split function not yet implemented")
