import polars as pl
import pytest


class TestSplitting:
    full_dataframe = pl.DataFrame(
        [
            {
                "id": 1,
                "pchembl_various_proteins": [2.5, None, 7],
                "activity": [1, None, 0],
            },
            {
                "id": 2,
                "pchembl_various_proteins": [None, 5.6, 7.2],
                "activity": [0, 1, 0],
            },
            {
                "id": 3,
                "pchembl_various_proteins": [5.6, 7.2, 4.5],
                "activity": [0, 0, 1],
            },
        ],
        schema=[("id", pl.Int64), ("value", pl.Array(pl.Float16, shape=3))],
    )
    print(full_dataframe)
