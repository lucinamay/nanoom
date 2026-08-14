"""
balancing, splitting

"""

import logging
import os
from typing import Literal, Sequence

import numpy as np
import polars as pl
import pulp  # https://coin-or.github.io/pulp/ for docs
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

lg = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


SplitMethod = Literal["tricarico", "sklearn"]


def _task_type(
    series: pl.Series,
) -> Literal["regression", "classification", "string_classification"]:
    dtype = series.dtype
    if dtype.is_integer():
        return "classification"  # @TODO: add in ordinal classification support
    if dtype.is_float():
        values = series.drop_nulls().unique()
        if len(values) == 0:
            raise ValueError(f"{series.name} is all-null. please check input")
        if (values % 1 == 0).all():
            return "classification"
        return "regression"
    if dtype in (pl.Utf8, pl.Categorical):
        return "string_classification"
    raise NotImplementedError(f"Unsupported task dtype: {dtype}")


def _pseudo_tasks_long(
    df: pl.DataFrame,
    col: str,
    cluster_col: str,
    n_bins_for_regression: int,
    binning_approach: Literal["gbmt_splits", "qcut"] = "qcut",
) -> pl.DataFrame:
    """Converts one task column into stratification pseudo-tasks (class, string, or regression bin).

    binning_approach: "gbmt_splits" reproduces the original algorithm (bins unique values equally);
    "qcut" (default) balances row counts per bin (better for repeated values).
    """
    sub = df.select(cluster_col, col).drop_nulls(col)
    match _task_type(df.get_column(col)):
        case "regression":
            match binning_approach:
                case "gbmt_splits":
                    # this bins the unique values
                    # rather than quantile values:
                    # original behaviour
                    values = sub.get_column(col).unique().sort()
                    bins = np.array_split(np.arange(len(values)), n_bins_for_regression)
                    bin_of_rank = np.concatenate(
                        [np.full(len(rank_idx), i) for i, rank_idx in enumerate(bins)]
                    )
                    lookup = pl.DataFrame(
                        {
                            col: values,
                            "pseudo_task": [f"{col}_bin{i}" for i in bin_of_rank],
                        }
                    )
                    return sub.join(lookup, on=col, how="inner").select(
                        cluster_col, "pseudo_task"
                    )
                case "qcut":
                    return sub.select(
                        cluster_col,
                        pl.col(col)
                        .qcut(
                            n_bins_for_regression,
                            labels=[
                                f"{col}_bin{i}" for i in range(n_bins_for_regression)
                            ],
                            allow_duplicates=True,
                        )
                        .cast(pl.Utf8)
                        .alias("pseudo_task"),
                    )
        case "classification":
            return sub.select(
                cluster_col,
                pl.concat_str(
                    [pl.lit(f"{col}_"), pl.col(col).cast(pl.Int64).cast(pl.Utf8)]
                ).alias("pseudo_task"),
            )
        case "string_classification":
            return sub.select(
                cluster_col,
                pl.concat_str([pl.lit(f"{col}_"), pl.col(col)]).alias("pseudo_task"),
            )


def _task_vs_clusters_df(
    df: pl.DataFrame,
    task_cols: str | Sequence[str] = ["pchembl_value_mean"],
    cluster_col: str = "cluster",
    n_bins_for_regression: int | None = 5,
    binning_approach: Literal["gbmt_splits", "qcut"] = "qcut",
) -> pl.DataFrame:
    """Cross-tabulation of task counts per cluster. Format required by the LP balancer.

    Rows: tasks (or regression bins), Columns: clusters. First column is cluster ID,
    second is total count, rest are pseudo-task counts."""
    task_cols = [task_cols] if isinstance(task_cols, str) else list(task_cols)
    if not n_bins_for_regression:
        lg.warning(
            "`n_bins_for_regression` == None, using 5 bins for any regression tasks"
        )
        n_bins_for_regression = 5

    df = df.lazy().select([cluster_col] + task_cols).collect()

    # each task column is stratified into pseudo-tasks independently of the
    # others' types, then concatenated into one long table so the split-balancing
    # script of Tricario et al. can balance them jointly
    pseudo_tasks = pl.concat(
        [
            _pseudo_tasks_long(
                df, col, cluster_col, n_bins_for_regression, binning_approach
            )
            for col in task_cols
        ]
    )

    return (
        pseudo_tasks.pivot(  # to wide-format
            on="pseudo_task",
            index=cluster_col,
            values=cluster_col,
            aggregate_function="len",
            sort_columns=True,
        )
        .fill_null(0)
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
    time_limit_seconds: float = 5 * 60,  # og: 60m but no one has the time
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
            - the time limit in seconds for the solver (by default 5 minutes)
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

    # N clusters * S splits binary variables; proving exact optimality gets
    # expensive well before the time limit does
    if N > 1000:
        lg.warning(
            f"balancing {N} clusters into {S} splits = {N * S} binary variables. "
            "CBC may not prove optimality within `time_limit_seconds` "
            f"(currently {time_limit_seconds}s), and will return its best solution "
            "so far. Pass `relative_gap` (e.g. 0.01) to stop at a near-optimal "
            "solution quickly, or cluster more coarsely."
        )

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
    y_cols: str | Sequence[str] = "pchembl_value_mean",
    cluster_col: str = "cluster",
    n_bins_for_regression: int | None = 5,
    binning_approach: Literal["gbmt_splits", "qcut"] = "qcut",
    **kwargs,
) -> tuple[pl.Series, np.ndarray]:
    """splits the data, returning the unique clusters and the split each was assigned to

    note: reimplementation of https://github.com/sohviluukkonen/gbmt-splits/blob/main/gbmtsplits/split.py
    implementation of the tricarico et al split

    binning_approach: see `_task_vs_clusters_df`/`_pseudo_tasks_long`. Pass
    "gbmt_splits" to reproduce gbmt-splits' approach.
    """
    lg.info(
        "if you want more than 1 split, you might want to change "
        "the seed of clustering for each run"
    )

    # rows: clusters, cols: tasks
    clusters_vs_tasks_df = _task_vs_clusters_df(
        df=df,
        task_cols=y_cols,
        cluster_col=cluster_col,
        n_bins_for_regression=n_bins_for_regression,
        binning_approach=binning_approach,
    )

    # keep as a polars Series to maintain dtype: they have to stay joinable against df[cluster_col]
    clusters = clusters_vs_tasks_df.get_column(cluster_col)

    # the array that the next code requires, with tasks as rows and clusters as cols
    task_vs_clusters_array = clusters_vs_tasks_df.drop(cluster_col).to_numpy().T

    split_assignments = _balance_splits_from_tasks_vs_clusters_array(
        task_vs_clusters_array,
        split_sizes=list(split_sizes),
        **kwargs,
    )
    return clusters, split_assignments


def sklearn_split(
    X: NDArray,
    y: NDArray,
    group_on: NDArray | None,
    random_state: int,
    n_splits: int = 5,
    n_bins_for_regression: int | None = 5,
) -> np.ndarray:
    """Split index (0..n_splits-1) per row of X.

    `group_on=None` splits rows independently (`StratifiedKFold`); otherwise no
    group is allowed to span two splits (`StratifiedGroupKFold`).
    """
    splitter_cls = StratifiedKFold if group_on is None else StratifiedGroupKFold
    splitter = splitter_cls(n_splits=n_splits, shuffle=True, random_state=random_state)
    assert group_on is None or isinstance(group_on, np.ndarray), (
        "group_on must be a np.ndarray or None"
    )
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

    split_idx = np.full(X.shape[0], -1)
    for k, (_, test_idx) in enumerate(splitter.split(X=X, y=y, groups=group_on)):
        split_idx[test_idx] = k
    assert (split_idx >= 0).all(), "every row must be assigned a fold"
    return split_idx


def split(
    df: pl.DataFrame,
    y_cols: Sequence[str] | str,
    n_splits: int,
    method: SplitMethod,
    cluster_col: str | None = None,
    split_col: str = "split",
    **kwargs,
) -> pl.DataFrame:
    """Return `df` with `split_col` added, holding 0..n_splits-1 per row.

    Route to tricarico (multi-task LP balancing) or sklearn (StratifiedGroupKFold).

    `cluster_col` names a pre-computed cluster column (see `nanoom.cluster`); no
    cluster is allowed to span two splits. Pass `cluster_col=None` to split rows
    directly, with no leakage constraint. Equivalent of clustersize=1 (i.e. each row becomes its own cluster).

    The cluster column stays in the frame, so the per-cluster assignment is
    `out.group_by(cluster_col).agg(pl.col(split_col).first())`.
    """
    if split_col in df.columns:
        # on the tricarico path this would join to `split_right` and leave the stale
        # column in place, so the caller reads back the wrong assignment
        raise ValueError(
            f"df already has a {split_col!r} column; pass `split_col=` to name the "
            "output column something else"
        )
    grouped = cluster_col is not None
    if grouped:
        if cluster_col not in df.columns:
            raise ValueError(f"cluster_col {cluster_col!r} not in df: {df.columns}")
    else:
        cluster_col = "cluster"
        if cluster_col in df.columns:
            raise ValueError(
                "cluster_col=None adds a 'cluster' column of row indices, but df "
                "already has one. Pass `cluster_col='cluster'` to split on it, or "
                "rename it."
            )  # @TODO: fix to have a temporary column
        df = df.with_columns(pl.int_range(pl.len()).alias(cluster_col))

    match method:
        case "tricarico":
            clusters, split_assignments = globally_balanced_split_polars(
                df=df,
                split_sizes=[1 / n_splits] * n_splits,
                y_cols=y_cols,
                cluster_col=cluster_col,
                **kwargs,
            )
            # broadcast the per-cluster assignment back onto rows
            mapping = pl.DataFrame(
                {cluster_col: clusters, split_col: split_assignments}
            )
            return df.join(mapping, on=cluster_col, how="left", maintain_order="left")
        case "sklearn":
            y_data = df.lazy().select(y_cols).collect()
            if y_data.width > 1:
                raise ValueError(
                    f"sklearn_split supports only 1 task column, got {y_data.width}"
                )
            group_on = df.get_column(cluster_col).to_numpy() if grouped else None
            if "random_state" not in kwargs:
                kwargs["random_state"] = 0
            split_idx = sklearn_split(
                # sklearn's splitters only read X's row count, never its values
                X=np.zeros((df.height, 1)),
                y=y_data.to_numpy()[:, 0],
                group_on=group_on,
                n_splits=n_splits,
                **kwargs,
            )
            return df.with_columns(
                pl.Series(split_col, split_idx)
            )  # @TODO: check if laziness ensures same order
    raise NotImplementedError(f"unknown split method: {method}")
