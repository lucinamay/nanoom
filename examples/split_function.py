from pathlib import Path

import polars as pl

from nanoom.splitting import split

MAINDIR = Path(__file__).parent.parent
if __name__ == "__main__":
    df = pl.read_csv(
        MAINDIR / "tests/test_data_papyrus_1000.csv", infer_schema_length=1000
    ).with_columns(
        pl.col("smiles").map_elements(lambda x: hash(x) % 100).alias("cluster")
    )

    train, test = split(
        df=df,
        X_col="smiles",
        y_cols=["pchembl_value_mean"],
        cluster_col="cluster",
        method="sklearn",
        n_splits=3,
    )
