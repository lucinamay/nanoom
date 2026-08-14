import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import polars as pl

    return mo, pl


@app.cell
def _(mo):
    mo.md(r"""
    load the data from papyrus: the data will initially be oriented as MOL_PROT activity rows (values from separate sources are combined by default into one value, pchembl_mean)
    """)
    return


@app.cell
def _(pl):
    # standard papyrus data (the first 1000 rows)
    df = pl.read_csv("tests/test_data_papyrus_1000.csv", infer_schema_length=1000)
    df
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    there are 379 unique molecules and 501 unique proteins. only 1000 of the theoretical ~15000 mol-protein pairs are filled. however, for illustration purposes, we will pivot it to a mol vs protein-value array.

    (disclaimer: for illustration purposes, we assume every unique smiles is a unique molecule)
    """)
    return


@app.cell
def _(df):
    assert (
        df.pivot(
            values="pchembl_value_mean",
            index="smiles",
            on="target_id",
            aggregate_function="len",
        )
        .drop("smiles")
        .max()
        .max_horizontal()[0]
        == 1
    ), "unexpected double values!"
    mol_vs_protein_regression = df.pivot(
        values="pchembl_value_mean",
        index="smiles",
        on="target_id",
        sort_columns=True,
    )
    mol_vs_protein_regression
    return (mol_vs_protein_regression,)


@app.cell
def _(mol_vs_protein_regression, pl):
    # (we can also combine them into one array for easier accessing, but we won't. example:)
    mol_vs_protein_regression.select(
        pl.col("smiles"), pl.concat_arr(pl.exclude("smiles").alias("task"))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we can see, the many null values would make it impossible to 'balance' on the proteins as tasks: the data is just too sparse. This is also why a multitask prediction model (X: molecules, y: prediction per protein) would not train very well.

    Instead, we try to put the protein information into the X itself, and for y, have only a bioactivity prediction (in this case regression, but you could also do classification). This in fact just keeps the original data format, as follows:
    """)
    return


@app.cell
def _(df):
    pcm_data = df.select(
        ["activity_id", "smiles", "target_id", "pchembl_value_mean"]
    ).rename({"activity_id": "datapoint", "pchembl_value_mean": "y"})
    pcm_data
    return (pcm_data,)


@app.cell
def _(mo):
    mo.md(r"""
    As you can see from the distribution visualisation of y, not all values are equally represented. To ensure we have the same y distribution in our training and evaluation dataset, we can bin the y values as follows:
    """)
    return


@app.cell
def _(pcm_data, pl):
    binned_pcm = pcm_data.with_columns(
        y_binned=pl.col("y").qcut(5, labels=[f"bin_{i}" for i in range(5)])
    )
    binned_pcm
    return (binned_pcm,)


@app.cell
def _(binned_pcm, pl):
    # visual check
    binned_pcm.pivot(
        index="datapoint", on="y_binned", values="y", aggregate_function="len"
    ).cast({pl.UInt32: pl.Boolean})  # boolean for visualisation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    An alternative option is to not do a single regression task on each mol-prot pair, but instead to do a multitask with each readout (IC50, EC50 etc.) as a separate task. As the papyrus data has all mol-protein values, regardless of readout type, in one row, we need to separate the data and values on readout first. (However, I forgot to keep in the pchembl_value columns in the test data, so here we will just take the same pchembl_value_mean for all points.)
    """)
    return


@app.cell
def _(df, pl):
    def equalize_list_lengths(
        df: pl.DataFrame | pl.LazyFrame, cols: list[str] | None = None
    ) -> pl.DataFrame | pl.LazyFrame:
        """
        Equalize list lengths by exploding, grouping, and re-aggregating with padding.
        Implementation mirroring equalize_cell_size_in_row of https://github.com/OlivierBeq/Papyrus-scripts/blob/master/src/papyrus_scripts/preprocess.py
        """

        if not cols:
            cols = [name for name, dtype in df.select(pl.col(pl.List)).collect_schema()]
        max_list_len = pl.max_horizontal(pl.selectors.list().list.len())
        return df.with_columns(
            [
                pl.concat_list(
                    pl.col(c),
                    pl.col(c)
                    .list.last()
                    .repeat_by(max_list_len - pl.col(c).list.len() + 1),
                ).list.slice(0, max_list_len)
                for c in cols
            ]
        )

    def semicolon_to_flat(
        df: pl.DataFrame | pl.LazyFrame,
    ) -> pl.DataFrame | pl.LazyFrame:
        """makes a flat file with all single values.

        Note that for the returned DataFrame pchembl_value_n no longer denotes the
        number of datapoints within the row but rather the number of datapoints in the
        given combination of protein+molecule

        Note: takes about 1m24s on celsus for collect()
        """
        cols = [
            # "source",
            # "cid",
            # "aid",
            "type_ic50",
            "type_ec50",
            "type_kd",
            "type_ki",
            "type_other",
            # "relation",
            # "pchembl_value",
            # "all_years",  # has quite some nulls, which we catch
            # "all_doc_ids",
        ]
        return (
            df.with_columns(pl.col(cols).str.split(";").fill_null([None]))
            .pipe(equalize_list_lengths, cols)
            .explode(cols)
            .with_columns(pl.col("^type_.*$").cast(pl.Int8))
        )

    def task_readout_per_activity(
        flat_df, readout_fmt: str = "^type_.*$", group_by="activity_id"
    ):
        return (
            flat_df.select(pl.col(readout_fmt), group_by)
            .group_by(group_by)
            .agg(pl.col(readout_fmt).sum())
        )

    import polars.selectors as cs

    readout_values = semicolon_to_flat(df).with_columns(
        pl.col("^type_.*$")
        .fill_null(0)
        .replace(1, None)
        .fill_null(pl.col("pchembl_value_mean"))
        .replace(0, None)
    )
    readout_values.unpivot(
        on=cs.starts_with("type_"),
        index="activity_id",
        variable_name="readout",
        value_name="y",
    ).drop_nulls().cast({"readout": pl.Categorical})
    return


@app.cell
def _(mo):
    mo.md(r"""
    For this type of data, we could 1) balance the readouts, or 2) balance the readouts in bins. Judging from the distribution of the readouts (some only <5%), this will become difficult when also considering which clusters to use. We thus choose to only balance the readouts.
    """)
    return


@app.cell
def _(binned_pcm, pl):
    # simulate clusters by hashing the mol-part of the id into 100 clusters (N/10)
    binned_pcm_cluster = binned_pcm.with_columns(
        pl.col("datapoint")
        .map_elements(lambda x: hash(x.split("_")[0]) % 100)
        .alias("cluster")
    )
    binned_pcm_cluster
    return (binned_pcm_cluster,)


@app.cell
def _(binned_pcm_cluster, pl):
    def tasks_vs_clusters_array_polars(
        df: pl.DataFrame, task_cols: list[str], cluster_col: str
    ):
        """
        Create a cross-tabulation 2D numpy array counting the # data points per task, per cluster

        Args:
            df (pl.DataFrame): dataframe with task columns and cluster column
            task_cols (list[str]): names of the columns containing tasks
            cluster_col (str): name of the column containing clusters
        Returns:
            pl.DataFrame.to_numpy(): 2D array of shape (num_tasks+1, num_clusters)

        Comment:
            In the returned array
            - each column is a unique initial cluster
            - each row is a unique task
            (except the first row, which is the total # objects in the cluster)
            This is the format requrired by the balancing algorithm
        """
        # Get unique clusters sorted for column order
        pass

    tasks_vs_clusters_array_polars(
        binned_pcm_cluster,
        task_cols=[f"bin_{i}" for i in range(5)],
        cluster_col="cluster",
    )
    pre_transpose = (
        binned_pcm_cluster.pivot(
            on="y_binned",
            index="cluster",
            values="datapoint",
            aggregate_function="len",
            sort_columns=True,
        )
        .join(
            binned_pcm_cluster.group_by("cluster").agg(pl.len().alias("number")),
            on="cluster",
            how="left",
        )
        .sort("cluster")
        .select(pl.col("cluster"), pl.col("number"), pl.exclude("number", "cluster"))
    )
    pre_transpose
    return (pre_transpose,)


@app.cell
def _(pre_transpose):
    # from nanoom.splitting import _balance_splits_from_tasks_vs_clusters_array
    import numpy as np

    arr = np.array(pre_transpose.to_numpy())
    column_vals = arr[:, 0]
    pass_to_opt = arr[:, 1:]
    pass_to_opt.T  # -> this is what we pass to the function
    return (pass_to_opt,)


@app.cell
def _():
    from nanoom.splitting import _balance_splits_from_tasks_vs_clusters_array

    return


@app.cell
def _(pass_to_opt):
    input = pass_to_opt.T
    input
    return (input,)


@app.cell
def _(input):
    _balance_splits_from_tasks_vs_clusters_array(input, time_limit_seconds=20)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
