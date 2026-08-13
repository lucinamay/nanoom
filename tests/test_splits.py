"""Correctness tests for nanoom.splitting: type detection, cross-tab
construction, leakage-safety, and balance quality of the tricarico (linear programming) and
sklearn split methods & tricarico/luukkonen parity check.
"""

import numpy as np
import polars as pl
import pytest

from nanoom.eval import check_no_group_overlap, split_y_means
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


class TestTaskType:
    def test_regression(self):
        s = pl.Series([0.1, 0.2, 0.35, 5.7, 8.9])
        assert _task_type(s) == "regression"

    def test_classification_int_dtype(self):
        s = pl.Series([0, 1, 2, 1, 0, 2], dtype=pl.Int64)
        assert _task_type(s) == "classification"

    def test_classification_float_integer_valued(self):
        # e.g. gbmt-splits' own fixture: Binary/Categorical load as Float64 (null
        # forces an upcast) but only ever contain integer-valued floats.
        s = pl.Series([0.0, 1.0, None, 1.0, 0.0], dtype=pl.Float64)
        assert _task_type(s) == "classification"

    def test_string_classification(self):
        s = pl.Series(["a", "b", "a", None])
        assert _task_type(s) == "string_classification"


class TestTaskVsClustersDf:
    def test_single_regression_task(self, mixed_df: pl.DataFrame):
        out = _task_vs_clusters_df(
            mixed_df, task_cols=["reg"], cluster_col="cluster", n_bins_for_regression=5
        )
        assert out["cluster"].to_list() == list(range(10))
        pseudo_cols = [c for c in out.columns if c not in ("cluster", "number")]
        assert len(pseudo_cols) == 5
        # every cluster's pseudo-task counts sum to its total row count
        assert (out.select(pseudo_cols).sum_horizontal() == out["number"]).all()

    def test_single_classification_task(self, mixed_df: pl.DataFrame):
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

    def test_mixed_regression_and_classification(self, mixed_df: pl.DataFrame):
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

    def test_mixed_all_three_types(self):
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

    def test_preserves_cluster_col(self, mixed_df: pl.DataFrame):
        # regression test: the old classification_onehot branch silently dropped
        # cluster_col during unpivot when it wasn't also x_col
        out = _task_vs_clusters_df(mixed_df, task_cols=["cls"], cluster_col="cluster")
        assert sorted(out["cluster"].to_list()) == sorted(
            mixed_df["cluster"].unique().to_list()
        )

    def test_regression_bins_row_count_balance(self):
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


class TestGloballyBalancedSplit:
    def test_no_leakage_single_task(self, mixed_df: pl.DataFrame):
        clusters, assignments = globally_balanced_split_polars(
            mixed_df,
            split_sizes=[0.34, 0.33, 0.33],
            y_cols=["reg"],
            cluster_col="cluster",
            **SOLVE_KWARGS,
        )
        assert len(set(clusters.to_list())) == len(clusters)  # no cluster appears twice
        row_cluster = mixed_df.get_column("cluster").to_numpy()
        mapping = dict(zip(clusters.to_list(), assignments.tolist()))
        row_split = np.array([mapping[c] for c in row_cluster])
        check_no_group_overlap(
            row_cluster, row_split
        )  # @TODO: implement as true test (not own function)

    def test_no_leakage_mixed_tasks(self, mixed_df: pl.DataFrame):
        clusters, assignments = globally_balanced_split_polars(
            mixed_df,
            split_sizes=[0.34, 0.33, 0.33],
            y_cols=["reg", "cls", "strcls"],
            cluster_col="cluster",
            **SOLVE_KWARGS,
        )
        assert len(set(clusters.to_list())) == len(clusters)
        row_cluster = mixed_df["cluster"].to_numpy()
        mapping = dict(zip(clusters.to_list(), assignments.tolist()))
        row_split = np.array([mapping[c] for c in row_cluster])
        check_no_group_overlap(row_cluster, row_split)

    def test_balances_split_sizes(self, mixed_df: pl.DataFrame):
        clusters, assignments = globally_balanced_split_polars(
            mixed_df,
            split_sizes=[0.5, 0.5],
            y_cols=["reg"],
            cluster_col="cluster",
            **SOLVE_KWARGS,
        )
        sizes = np.bincount(assignments) / len(assignments)
        assert np.allclose(sizes, [0.5, 0.5], atol=0.15)

    def test_balances_task_distribution(self):
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
        mapping = dict(zip(clusters.to_list(), assignments.tolist()))
        split_of_row = np.array([mapping[c] for c in cluster])
        class1_frac_per_split = [
            (cls[split_of_row == s] == 1).mean()
            for s in sorted(set(assignments.tolist()))
        ]
        assert np.allclose(class1_frac_per_split, 1 / 3, atol=0.05)

    def test_single_oversized_cluster_does_not_crash(self):
        # known, documented limitation: nanoom has no overflow/repair handling
        # for a cluster larger than the smallest requested split (neither does
        # gbmt-splits). This characterizes today's behavior (doesn't crash)
        # without asserting good size balance.
        n_clusters, per_cluster = 5, 4
        cluster = np.concatenate(
            [np.zeros(80, dtype=int), np.repeat(np.arange(1, n_clusters), per_cluster)]
        )
        df = pl.DataFrame(
            {
                "cluster": cluster,
                "reg": np.random.default_rng(0).random(len(cluster)) * 10,
            }
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
        mapping = dict(zip(clusters.to_list(), assignments.tolist()))
        row_split = np.array([mapping[c] for c in row_cluster])
        check_no_group_overlap(row_cluster, row_split)  # leakage safety still holds

    def test_split_sizes_larger_than_clusters_raises(self, mixed_df: pl.DataFrame):
        with pytest.raises(ValueError):
            globally_balanced_split_polars(
                mixed_df,
                split_sizes=[0.1] * 20,
                y_cols=["reg"],
                cluster_col="cluster",
                **SOLVE_KWARGS,
            )


class TestSklearnSplit:
    def test_no_group_leakage(self, mixed_df: pl.DataFrame):
        X = mixed_df["x"].to_numpy()
        y = mixed_df["reg"].to_numpy()
        group_on = mixed_df["cluster"].to_numpy()
        split_idx = sklearn_split(X, y, group_on, random_state=0, n_splits=3)
        check_no_group_overlap(group_on, split_idx)

    def test_without_groups(self, mixed_df: pl.DataFrame):
        X = mixed_df["x"].to_numpy()
        y = mixed_df["reg"].to_numpy()
        split_idx = sklearn_split(X, y, None, random_state=0, n_splits=3)
        assert len(split_idx) == mixed_df.height
        assert set(split_idx.tolist()) == {0, 1, 2}


class TestSplitDispatcher:
    def test_dispatch_tricarico(self, mixed_df: pl.DataFrame):
        out = split(
            mixed_df,
            y_cols=["reg"],
            cluster_col="cluster",
            n_splits=3,
            method="tricarico",
            **SOLVE_KWARGS,
        )
        assert out.height == mixed_df.height
        assert set(out["split"].to_list()) <= {0, 1, 2}

    def test_dispatch_sklearn(self, mixed_df: pl.DataFrame):
        out = split(
            mixed_df,
            y_cols="reg",
            cluster_col="cluster",
            n_splits=3,
            method="sklearn",
            random_state=0,
        )
        check_no_group_overlap(out["cluster"], out["split"])

    @pytest.mark.parametrize(
        "method,kwargs",
        [("tricarico", SOLVE_KWARGS), ("sklearn", {"random_state": 0})],
    )
    def test_returns_row_aligned_for_both_methods(self, mixed_df, method, kwargs):
        """Same contract for both: one split per row, in the frame, auditable by nanoom.eval."""
        out = split(
            mixed_df,
            y_cols="reg",
            cluster_col="cluster",
            n_splits=3,
            method=method,
            **kwargs,
        )
        assert out.height == mixed_df.height
        assert out["split"].null_count() == 0
        assert set(out["split"].to_list()) == {0, 1, 2}
        check_no_group_overlap(out["cluster"], out["split"])

        got = split_y_means(out["reg"], out["split"])
        expected = [
            out.filter(pl.col("split") == s)["reg"].mean()
            for s in sorted(set(out["split"]))
        ]
        assert np.allclose(got, expected)
        # rows survive the join in their original order
        assert out["reg"].to_list() == mixed_df["reg"].to_list()

    @pytest.mark.parametrize(
        "method,kwargs",
        [("tricarico", SOLVE_KWARGS), ("sklearn", {"random_state": 0})],
    )
    def test_without_clusters(self, method, kwargs):
        df = _clustered_df(n_clusters=4, per_cluster=5).drop("cluster")
        out = split(df, y_cols="reg", n_splits=2, method=method, **kwargs)
        assert out.height == df.height
        # a row-index cluster column is added so grouping/eval work in both modes
        assert out["cluster"].to_list() == list(range(df.height))
        assert set(out["split"].to_list()) == {0, 1}
        check_no_group_overlap(out["cluster"].to_numpy(), out["split"].to_numpy())

    def test_tricarico_row_mode_matches_singleton_clusters(self):
        """cluster_col=None is exactly 'one cluster per row', not a separate algorithm."""
        df = _clustered_df(n_clusters=4, per_cluster=3).drop("cluster")
        implicit = split(
            df, y_cols="reg", n_splits=2, method="tricarico", **SOLVE_KWARGS
        )
        explicit = split(
            df.with_columns(pl.int_range(pl.len()).alias("row_id")),
            y_cols="reg",
            cluster_col="row_id",
            n_splits=2,
            method="tricarico",
            **SOLVE_KWARGS,
        )
        assert implicit["split"].to_list() == explicit["split"].to_list()

    def test_preserves_row_order_with_unsorted_clusters(self):
        """The tricarico path joins a per-cluster mapping back onto rows."""
        df = pl.DataFrame(
            {
                "cluster": ["c", "a", "c", "b", "a", "b"],
                "reg": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        out = split(
            df,
            y_cols="reg",
            cluster_col="cluster",
            n_splits=2,
            method="tricarico",
            **SOLVE_KWARGS,
        )
        assert out["cluster"].to_list() == df["cluster"].to_list()
        assert out["reg"].to_list() == df["reg"].to_list()
        # string cluster ids must survive the round-trip through the LP mapping
        assert out["split"].null_count() == 0

    def test_raises_rather_than_overwriting_existing_columns(
        self, mixed_df: pl.DataFrame
    ):
        with pytest.raises(ValueError, match="already has a 'split' column"):
            split(
                mixed_df.with_columns(pl.lit(0).alias("split")),
                y_cols="reg",
                cluster_col="cluster",
                n_splits=3,
                method="sklearn",
            )
        with pytest.raises(ValueError, match="already has one"):
            split(mixed_df, y_cols="reg", n_splits=3, method="sklearn")

    def test_raises_on_missing_cluster_col(self, mixed_df: pl.DataFrame):
        with pytest.raises(ValueError, match="not in df"):
            split(
                mixed_df, y_cols="reg", cluster_col="nope", n_splits=3, method="sklearn"
            )

    def test_unknown_method_raises(self, mixed_df: pl.DataFrame):
        with pytest.raises(NotImplementedError):
            split(
                mixed_df,
                y_cols=["reg"],
                cluster_col="cluster",
                n_splits=3,
                method="unknown",
            )


class TestGbmtSplitsEquivalence:
    def test_tricarico_matches_gbmtsplits_reference(self):
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
        nanoom_mapping = dict(zip(nanoom_clusters.to_list(), nanoom_assign.tolist()))

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
                result[t] = (
                    (sub.groupby("split").size() / len(sub)).sort_index().to_numpy()
                )
            return result

        nanoom_frac = fractions(nanoom_mapping)
        gbmt_frac = fractions(gbmt_mapping)
        for t in tasks:
            assert np.allclose(nanoom_frac[t], gbmt_frac[t], atol=0.03), (
                f"{t}: nanoom={nanoom_frac[t]} gbmt={gbmt_frac[t]}"
            )
            assert np.allclose(nanoom_frac[t], 1 / 3, atol=0.06)
