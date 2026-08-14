# nanoom

Multimodal-multitask-balanced dataset splitting: cluster-based separation and task-stratification/balancing
across splits.

Design rationale is in [ARCHITECTURE.md](ARCHITECTURE.md), the task list in
[DEVELOPMENT.md](DEVELOPMENT.md), setup and conventions in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Install

```
pip install nanoom
```

Core dependencies: `polars`, `pulp`, and `scikit-learn`. 

Optional extras:

- `nanoom[chem]` - rdkit/bblean-based clustering methods for molecular data
- `nanoom[tutorial]` - marimo, for the notebooks in `examples/`
- `nanoom[plotting]` - reserved for future split-quality visualization (not yet implemented)

## Splitting

`nanoom.split` takes a dataframe with a pre-computed cluster column (see
[Clustering](#clustering) below) and returns a leakage-safe, task-balanced
train/val/test assignment.

```python
import nanoom

# balances by exactly solving a linear program (Tricarico et al., ported from
# https://github.com/sohviluukkonen/gbmt-splits); task columns may be a mix of
# regression, classification, and string columns, balanced jointly
out = nanoom.split(
    df,
    y_cols=["pchembl_value_mean", "target_id"],
    cluster_col="cluster",
    n_splits=3,
    method="tricarico",
)

# or: a lighter/faster fallback via sklearn's StratifiedGroupKFold
out = nanoom.split(
    df,
    y_cols="ic50",
    cluster_col="cluster",
    n_splits=3,
    method="sklearn",
)
```

Both methods return `df` with a `split` column added, holding `0..n_splits-1` per
row (rename it with `split_col=`). The cluster column stays in the frame, so the
per-cluster assignment is
`out.group_by("cluster").agg(pl.col("split").first())`.

### Splitting without clusters

`cluster_col` is optional. Passing `cluster_col=None` splits rows directly with no
leakage constraint — each row becomes its own cluster, so the LP balances rows and
sklearn falls back to `StratifiedKFold`:

```python
out = nanoom.split(df, y_cols="ic50", n_splits=3, method="tricarico")
```

A `cluster` column of row indices is added so grouping and the audit functions
below work the same in both modes. Note that the LP carries one binary variable per
cluster per split, so cluster-free mode on a large frame gets slow; nanoom warns
past 1000 clusters and suggests `relative_gap` for a fast near-optimal solve.

For regression task columns, `method="tricarico"` bins values before balancing
(`binning_approach=`, default `"qcut"`): `"qcut"` targets roughly equal n_datapoints per
bin; `"gbmt_splits"` instead bins the distinct values (matching gbmt-splits'
behaviour).
for repeated values this can skew row counts across bins. Neither option fully
resolves a very-repeated value, since those rows can't be split
across a bin boundary either way.

## Clustering

`nanoom.cluster` produces the `cluster_col` that `split()` expects, decoupled from
splitting itself so you can bring your own cluster assignment instead if you prefer.

```python
clusters = nanoom.cluster(descriptors, method="kmeans", n_clusters=20)
```

Supports `kmeans`, `dbscan`, `hdbscan`, `random`, and `hash_dummy` out of the box;
`sphere_exclusion`, `bitbirch`, `maxmin`, and `leader_picker` additionally require
`nanoom[chem]`. An `auto` mode that adaptively tunes cluster granularity to trade off
leakage-avoidance against task-balance quality is planned for a later iteration.

## Auditing a split

`nanoom.eval` answers "is my split actually leakage-free and balanced?":

```python
from nanoom.eval import check_no_group_overlap, check_distribution_y_similar, min_distances_splits

check_no_group_overlap(out["cluster"], out["split"])   # raises if any cluster spans >1 split
check_distribution_y_similar(out["ic50"], out["split"])  # raises if a split's y-mean drifts >10%
min_distances_splits(descriptors, out["split"], metric="euclidean")  # inter/intra-split distance stats
```

All three take per-row arrays and raise on a length mismatch rather than silently
scoring the wrong subset.
