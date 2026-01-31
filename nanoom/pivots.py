import polars as pl


def task_readout_per_activity(flat_df):
    return (
        flat_df.select(pl.col("^type_.*$"), "activity_id")
        .group_by("activity_id")
        .agg(pl.col("^type_.*$").sum())
    )


def task_protein_per_mol(flat_df) -> pl.LazyFrame:
    # return (
    #     flat_data.select("target_id", "inchi")
    #     .collect()
    #     .pivot(
    #         on="target_id",
    #         index="inchi",
    #         values="target_id",
    #         aggregate_function="len",
    #     )
    # )
    return flat_df.select("target_id", "inchi").pivot(
        on="target_id",
        on_columns=flat_df.select("target_id").unique().collect().to_series(),
        index="inchi",
        values="target_id",
        aggregate_function="len",
    )
