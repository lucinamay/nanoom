"""Correctness tests for nanoom.splitting: type detection, cross-tab
construction, leakage-safety, and balance quality of the tricario (linear programming) and
sklearn split methods & tricario/luukonen parity check.
"""

import numpy as np
import polars as pl
import pytest

from nanoom.eval import check_no_group_overlap
from nanoom.splitting import (
    _task_type,
    _task_vs_clusters_df,
    globally_balanced_split_polars,
    sklearn_split,
    split,
)

# ILP calls in these tests use n_jobs=1 for solver determinism, and small
# cluster counts so CBC reaches the exact (relative_gap=0) optimum quickly:
# proving *exact* optimality on a larger/harder instance can take CBC a long
# time even though a near-optimal solution is found almost instantly.
SOLVE_KWARGS = dict(n_jobs=1, time_limit_seconds=20)


# --- Fixtures ---


def _clustered_df(n_clusters=10, per_cluster=8, seed=0):
    rng = np.random.default_rng(seed)
    n = n_clusters * per_cluster
    cluster = np.repeat(np.arange(n_clusters), per_cluster)
    return pl.DataFrame(
        {
            "x": np.arange(n),
            "cluster": cluster,
            "reg": rng.random(n) * 10,
            "cls": rng.integers(0, 3, n),
            "strcls": rng.choice(["a", "b"], n),
        }
    )


@pytest.fixture
def mixed_df():
    return _clustered_df()


# --- _task_type ---


def test_task_type_regression():
    s = pl.Series([0.1, 0.2, 0.35, 5.7, 8.9])
    assert _task_type(s) == "regression"


def test_task_type_classification_int_dtype():
    s = pl.Series([0, 1, 2, 1, 0, 2], dtype=pl.Int64)
    assert _task_type(s) == "classification"


def test_task_type_classification_float_integer_valued():
    # e.g. gbmt-splits' own fixture: Binary/Categorical load as Float64 (null
    # forces an upcast) but only ever contain integer-valued floats.
    s = pl.Series([0.0, 1.0, None, 1.0, 0.0], dtype=pl.Float64)
    assert _task_type(s) == "classification"


def test_task_type_string_classification():
    s = pl.Series(["a", "b", "a", None])
    assert _task_type(s) == "string_classification"


# --- _task_vs_clusters_df ---


def test_task_vs_clusters_df_single_regression_task(mixed_df):
    out = _task_vs_clusters_df(
        mixed_df, task_cols=["reg"], cluster_col="cluster", n_bins_for_regression=5
    )
    assert out["cluster"].to_list() == list(range(10))
    pseudo_cols = [c for c in out.columns if c not in ("cluster", "number")]
    assert len(pseudo_cols) == 5
    # every cluster's pseudo-task counts sum to its total row count
    assert (out.select(pseudo_cols).sum_horizontal() == out["number"]).all()


def test_task_vs_clusters_df_single_classification_task(mixed_df):
    out = _task_vs_clusters_df(mixed_df, task_cols=["cls"], cluster_col="cluster")
    expected = {f"cls_{c}" for c in mixed_df["cls"].unique().to_list()}
    pseudo_cols = {c for c in out.columns if c not in ("cluster", "number")}
    assert pseudo_cols == expected
    # cross-check counts against an independently-computed group_by
    independent = (
        mixed_df.group_by("cluster", "cls").agg(pl.len()).sort("cluster", "cls")
    )
    for row in independent.iter_rows(named=True):
        col = f"cls_{row['cls']}"
        got = out.filter(pl.col("cluster") == row["cluster"])[col].item()
        assert got == row["len"]


def test_task_vs_clusters_df_mixed_regression_and_classification(mixed_df):
    # headline new capability: previously raised NotImplementedError
    out = _task_vs_clusters_df(
        mixed_df, task_cols=["reg", "cls"], cluster_col="cluster"
    )
    cols = set(out.columns)
    assert {"reg_bin0", "reg_bin1", "reg_bin2", "reg_bin3", "reg_bin4"} <= cols
    assert {"cls_0", "cls_1", "cls_2"} <= cols
    # per-cluster totals unaffected by which/how-many tasks were requested
    totals_single = _task_vs_clusters_df(
        mixed_df, task_cols=["reg"], cluster_col="cluster"
    )["number"]
    assert (out["number"] == totals_single).all()


def test_task_vs_clusters_df_mixed_all_three_types():
    # real mixed-type data: Contineous (regression), Binary/Categorical
    # (classification), String (string classification) balanced together
    df = pl.read_csv("tests/gmbtsplit_test_data.csv").with_row_index("row")
    df = df.with_columns((pl.col("row") % 12).alias("cluster"))
    out = _task_vs_clusters_df(
        df,
        task_cols=["Contineous", "Binary", "Categorical", "String"],
        cluster_col="cluster",
    )
    assert out.shape[0] == 12
    cols = set(out.columns)
    assert {"Binary_0", "Binary_1"} <= cols
    assert {"Categorical_0", "Categorical_1", "Categorical_2"} <= cols
    assert any(c.startswith("String_") for c in cols)
    assert any(c.startswith("Contineous_bin") for c in cols)


def test_task_vs_clusters_df_preserves_cluster_col(mixed_df):
    # regression test: the old classification_onehot branch silently dropped
    # cluster_col during unpivot when it wasn't also x_col
    out = _task_vs_clusters_df(mixed_df, task_cols=["cls"], cluster_col="cluster")
    assert sorted(out["cluster"].to_list()) == sorted(
        mixed_df["cluster"].unique().to_list()
    )


def test_task_vs_clusters_df_regression_bins_row_count_balance():
    # "qcut" targets ~equal row count per bin; "gbmt_splits" bins the
    # *distinct* values instead, so a repeated value skews row counts across
    # bins even though the bin *count* itself is balanced. A mild 20/80 skew
    # (not so extreme it collapses qcut's bins - see the module docstring on
    # `_pseudo_tasks_long` for that degenerate case) demonstrates this cleanly.
    n = 100
    reg = np.concatenate([np.full(20, 1.0), np.linspace(2.0, 6.0, 80)])
    df = pl.DataFrame({"cluster": np.arange(n) % 10, "reg": reg})

    qcut_out = _task_vs_clusters_df(
        df, task_cols=["reg"], cluster_col="cluster", binning_approach="qcut"
    )
    gbmt_out = _task_vs_clusters_df(
        df, task_cols=["reg"], cluster_col="cluster", binning_approach="gbmt_splits"
    )
    qcut_bin_totals = qcut_out.select(pl.exclude("cluster", "number")).sum().row(0)
    gbmt_bin_totals = gbmt_out.select(pl.exclude("cluster", "number")).sum().row(0)

    assert sum(qcut_bin_totals) == sum(gbmt_bin_totals) == n
    assert len(qcut_bin_totals) == len(gbmt_bin_totals) == 5

    # qcut: all 5 bins get an equal 20-row share
    assert max(qcut_bin_totals) - min(qcut_bin_totals) == 0
    # gbmt_splits: the 20 duplicate rows all land in one value-rank bin,
    # skewing it well above the even 20-per-bin share
    assert max(gbmt_bin_totals) > 20


# --- leakage + balance ---


def test_globally_balanced_split_no_leakage_single_task(mixed_df):
    clusters, assignments = globally_balanced_split_polars(
        mixed_df,
        split_sizes=[0.34, 0.33, 0.33],
        y_cols=["reg"],
        cluster_col="cluster",
        **SOLVE_KWARGS,
    )
    assert len(set(clusters.tolist())) == len(clusters)  # no cluster appears twice
    row_cluster = mixed_df["cluster"].to_numpy()
    mapping = dict(zip(clusters.tolist(), assignments.tolist()))
    row_split = np.array([mapping[c] for c in row_cluster])
    check_no_group_overlap(row_cluster, row_split)


def test_globally_balanced_split_no_leakage_mixed_tasks(mixed_df):
    clusters, assignments = globally_balanced_split_polars(
        mixed_df,
        split_sizes=[0.34, 0.33, 0.33],
        y_cols=["reg", "cls", "strcls"],
        cluster_col="cluster",
        **SOLVE_KWARGS,
    )
    assert len(set(clusters.tolist())) == len(clusters)
    row_cluster = mixed_df["cluster"].to_numpy()
    mapping = dict(zip(clusters.tolist(), assignments.tolist()))
    row_split = np.array([mapping[c] for c in row_cluster])
    check_no_group_overlap(row_cluster, row_split)


def test_globally_balanced_split_balances_split_sizes(mixed_df):
    clusters, assignments = globally_balanced_split_polars(
        mixed_df,
        split_sizes=[0.5, 0.5],
        y_cols=["reg"],
        cluster_col="cluster",
        **SOLVE_KWARGS,
    )
    sizes = np.bincount(assignments) / len(assignments)
    assert np.allclose(sizes, [0.5, 0.5], atol=0.15)


def test_globally_balanced_split_balances_task_distribution():
    # deliberately skew class '1' toward low-numbered clusters; the balancer
    # should still spread it close to evenly across splits. 6 skewed clusters
    # (of 18 total) is chosen so an exactly-even 2/2/2 cluster split is
    # actually achievable - clusters are the atomic assignment unit, so a
    # coarser skew (e.g. 4 clusters over 3 splits) can't be balanced better
    # than 2/1/1 no matter how good the solver is.
    n_clusters, per_cluster = 18, 6
    cluster = np.repeat(np.arange(n_clusters), per_cluster)
    cls = np.where(cluster < 6, 1, 0)  # class 1 concentrated in clusters 0-5
    df = pl.DataFrame({"cluster": cluster, "cls": cls})
    clusters, assignments = globally_balanced_split_polars(
        df,
        split_sizes=[1 / 3, 1 / 3, 1 / 3],
        y_cols=["cls"],
        cluster_col="cluster",
        **SOLVE_KWARGS,
    )
    mapping = dict(zip(clusters.tolist(), assignments.tolist()))
    split_of_row = np.array([mapping[c] for c in cluster])
    class1_frac_per_split = [
        (cls[split_of_row == s] == 1).mean() for s in sorted(set(assignments.tolist()))
    ]
    assert np.allclose(class1_frac_per_split, 1 / 3, atol=0.05)


# --- dispatcher + guards ---


def test_split_dispatch_tricario(mixed_df):
    clusters, assignments = split(
        mixed_df,
        X_col="x",
        y_cols=["reg"],
        cluster_col="cluster",
        n_splits=3,
        method="tricario",
        **SOLVE_KWARGS,
    )
    assert len(clusters) == 10
    assert set(assignments.tolist()) <= {0, 1, 2}


def test_split_dispatch_sklearn(mixed_df):
    group_on, split_idx = split(
        mixed_df,
        X_col="x",
        y_cols="reg",
        cluster_col="cluster",
        n_splits=3,
        method="sklearn",
        random_state=0,
    )
    check_no_group_overlap(group_on, split_idx)


def test_sklearn_split_no_group_leakage(mixed_df):
    X = mixed_df["x"].to_numpy()
    y = mixed_df["reg"].to_numpy()
    group_on = mixed_df["cluster"].to_numpy()
    _, split_idx = sklearn_split(X, y, group_on, random_state=0, n_splits=3)
    check_no_group_overlap(group_on, split_idx)


def test_split_sizes_larger_than_clusters_raises(mixed_df):
    with pytest.raises(ValueError):
        globally_balanced_split_polars(
            mixed_df,
            split_sizes=[0.1] * 20,
            y_cols=["reg"],
            cluster_col="cluster",
            **SOLVE_KWARGS,
        )


def test_split_unknown_method_raises(mixed_df):
    with pytest.raises(NotImplementedError):
        split(
            mixed_df,
            X_col="x",
            y_cols=["reg"],
            cluster_col="cluster",
            n_splits=3,
            method="unknown",
        )


def test_single_oversized_cluster_does_not_crash():
    # known, documented limitation: nanoom has no overflow/repair handling
    # for a cluster larger than the smallest requested split (neither does
    # gbmt-splits). This characterizes today's behavior (doesn't crash)
    # without asserting good size balance.
    n_clusters, per_cluster = 5, 4
    cluster = np.concatenate(
        [np.zeros(80, dtype=int), np.repeat(np.arange(1, n_clusters), per_cluster)]
    )
    df = pl.DataFrame(
        {"cluster": cluster, "reg": np.random.default_rng(0).random(len(cluster)) * 10}
    )
    clusters, assignments = globally_balanced_split_polars(
        df,
        split_sizes=[1 / 3, 1 / 3, 1 / 3],
        y_cols=["reg"],
        cluster_col="cluster",
        **SOLVE_KWARGS,
    )
    assert len(clusters) == n_clusters
    row_cluster = df["cluster"].to_numpy()
    mapping = dict(zip(clusters.tolist(), assignments.tolist()))
    row_split = np.array([mapping[c] for c in row_cluster])
    check_no_group_overlap(row_cluster, row_split)  # leakage safety still holds


# --- equivalence with the real upstream gbmtsplits package ---


def test_tricario_matches_gbmtsplits_reference():
    pytest.importorskip("gbmtsplits")
    import pandas as pd
    from gbmtsplits.split import GloballyBalancedSplit

    n_clusters = 12
    sizes = [1 / 3, 1 / 3, 1 / 3]
    tasks = ["Contineous", "Binary", "Categorical", "String"]

    df_pl = pl.read_csv("tests/gmbtsplit_test_data.csv").with_row_index("row")
    df_pl = df_pl.with_columns((pl.col("row") % n_clusters).alias("cluster"))
    df_pd = pd.read_csv("tests/gmbtsplit_test_data.csv")
    cluster_of_row = np.arange(len(df_pd)) % n_clusters
    clusters_dict = {
        c: list(np.where(cluster_of_row == c)[0]) for c in range(n_clusters)
    }

    # explicit matching settings on both sides - the two libraries' defaults
    # differ (equal_weight_perc_compounds_as_tasks, gap type/value, and
    # nanoom's binning_approach defaults to "qcut" rather than gbmt-splits'
    # own bin-over-distinct-values behaviour), so relying on defaults would
    # compare different problems, not the same one
    nanoom_clusters, nanoom_assign = globally_balanced_split_polars(
        df_pl,
        split_sizes=sizes,
        y_cols=tasks,
        cluster_col="cluster",
        equal_weight_perc_compounds_as_tasks=True,
        binning_approach="gbmt_splits",
        relative_gap=0,
        n_jobs=1,
        time_limit_seconds=45,
    )
    nanoom_mapping = dict(zip(nanoom_clusters.tolist(), nanoom_assign.tolist()))

    splitter = GloballyBalancedSplit(
        sizes=sizes,
        clusters=clusters_dict,
        clustering_method=None,
        n_splits=1,
        equal_weight_perc_compounds_as_tasks=True,
        absolute_gap=0,
        time_limit_seconds=45,
        n_jobs=1,
        min_distance=False,
        stratify=True,
        stratify_reg_nbins=5,
    )
    out = splitter(data=df_pd, smiles_column="SMILES", tasks=tasks)
    out["cluster"] = cluster_of_row
    split_per_cluster = out.groupby("cluster")["Split"].nunique()
    assert (split_per_cluster == 1).all()  # gbmt-splits itself doesn't leak either
    gbmt_mapping = out.groupby("cluster")["Split"].first().astype(int).to_dict()

    # zero leakage on both sides - the one assertion that must always hold
    check_no_group_overlap(
        cluster_of_row, np.array([nanoom_mapping[c] for c in cluster_of_row])
    )
    check_no_group_overlap(
        cluster_of_row, np.array([gbmt_mapping[c] for c in cluster_of_row])
    )

    # not asserting bit-exact assignment equality: ILP ties are real (both
    # implementations solve the identical objective/constraints, but a tied
    # optimum can land on either symmetric solution - verified empirically:
    # 10/12 clusters get identical assignment, the other 2 are a symmetric
    # split-label swap). Instead assert equivalent *achieved balance*.
    df_pd["cluster"] = cluster_of_row

    def fractions(mapping):
        s = df_pd.assign(split=df_pd["cluster"].map(mapping))
        result = {}
        for t in tasks:
            sub = s.dropna(subset=[t])
            result[t] = (sub.groupby("split").size() / len(sub)).sort_index().to_numpy()
        return result

    nanoom_frac = fractions(nanoom_mapping)
    gbmt_frac = fractions(gbmt_mapping)
    for t in tasks:
        assert np.allclose(nanoom_frac[t], gbmt_frac[t], atol=0.03), (
            f"{t}: nanoom={nanoom_frac[t]} gbmt={gbmt_frac[t]}"
        )
        assert np.allclose(nanoom_frac[t], 1 / 3, atol=0.06)
