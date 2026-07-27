# nanoom

Multimodal-multitask-balanced dataset splitting: cluster-based separation and task-stratification/balancing
across splits.

## Install

```
pip install nanoom
```

Core dependencies: `polars`, `pulp`, and `scikit-learn`. 

Optional extras:

- `nanoom[chem]` - rdkit/bblean-based clustering methods for molecular data
- `nanoom[tutorial]` - marimo, for the notebooks in `examples/`
- `nanoom[jax_cuda13]` - jax/CUDA backend for `nanoom.eval_jax`
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
clusters, split_assignment = nanoom.split(
    df,
    X_col="smiles",
    y_cols=["pchembl_value_mean", "target_id"],
    cluster_col="cluster",
    n_splits=3,
    method="tricario",
)

# or: a lighter/faster fallback via sklearn's StratifiedGroupKFold
clusters, split_assignment = nanoom.split(
    df,
    X_col="smiles",
    y_cols="ic50",
    cluster_col="cluster",
    n_splits=3,
    method="sklearn",
)
```

Both methods return `(clusters, split_assignment)`: the unique cluster values, and
which split each cluster was assigned to.

For regression task columns, `method="tricario"` bins values before balancing
(`regression_bins=`, default `"qcut"`): `"qcut"` targets roughly equal n_datapoints per
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

check_no_group_overlap(cluster_ids, split_assignment)      # raises if any cluster spans >1 split
check_distribution_y_similar(y, split_assignment)           # raises if a split's y-mean drifts >10%
min_distances_splits(descriptors, split_assignment, metric="euclidean")  # inter/intra-split distance stats
```
