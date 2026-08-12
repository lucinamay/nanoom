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

    out = split(
        df=df,
        y_cols=["pchembl_value_mean"],
        cluster_col="cluster",
        method="sklearn",
        n_splits=3,
    )
    # fold 0 is the test set, the rest is training
    train = out.filter(pl.col("split") != 0)
    test = out.filter(pl.col("split") == 0)
    print(f"{train.height} train rows, {test.height} test rows")
