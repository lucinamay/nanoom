"""Smoke tests to verify package installation and basic functionality."""

import nanoom


def test_import():
    """Verify nanoom imports successfully."""
    assert hasattr(nanoom, "split")
    assert hasattr(nanoom, "cluster")


def test_cluster_basic():
    """Verify basic clustering works."""
    import polars as pl

    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": [1.0, 1.1, 5.0, 5.1, 9.0, 9.1],
        }
    )

    clusters = nanoom.cluster(df[["x", "y"]].to_numpy(), method="random", n_clusters=2)
    assert len(clusters) == 6
