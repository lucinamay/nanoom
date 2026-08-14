# Architecture

nanoom implementation: current state + decisions (for reference) + open questions

For usage see [README.md](README.md); for setup and conventions see
[CONTRIBUTING.md](CONTRIBUTING.md); for the ordered task list see
[DEVELOPMENT.md](DEVELOPMENT.md).

## current state

nanoom's main task: assign rows of a dataframe to train/val/test splits so that
(a) related rows don't leak across splits and (b) every task stays balanced across
them. modules split by task / context (mostly external package context):

| module          | owns                                                                       |
| --------------- | -------------------------------------------------------------------------- |
| `clustering.py` | `cluster()` — descriptors → cluster label per row. uses sklearn (kmeans/dbscan/hdbscan), `chem.py`, or the built-in random/hash dummies. |
| `chem.py`       | cheminformatics-specific (rdkit/bblean) cluster methods for bit-vector fingerprints. Imported lazily, only under the `chem` extra. |
| `splitting.py`  | `split()` — dataframe + task columns → the same dataframe with a `split` column. Both the LP balancer and the sklearn fallback. |
| `eval.py`       | leakage & balance reports/metrics of a performed split. |

The conceptual pipeline is `cluster()` → `split()` → `eval`, and each step is usable without
the others.

`split()` takes `y_cols` (any mix of regression, integer-classification, and string
columns), turns each into stratification pseudo-tasks, and balances them jointly.

## Decisions

** tricarico/luukkonen LP balancer is a port, not a dependency.** The algorithm is from Tricarico
et al.
([chemrxiv-2022-m8l33-v3](https://doi.org/10.26434/chemrxiv-2022-m8l33-v3)), and
a working implementation already exists in
[gbmt-splits](https://github.com/sohviluukkonen/gbmt-splits). We reimplemented
it to use it without its dependencies for non-chemistry use too.
`_balance_splits_from_tasks_vs_clusters_array` stays close to the
published source deliberately (so you can diff it).
`tests/test_splits.py` holds a parity check against gbmt-splits' own output.

**clustering and splitting are independent** `split()` takes an existing
`cluster_col`. Clustering is the part most specific to a dataset (scaffolds,
sequence identity, assay, time, etc.), so users must be able to use their own
clustering without touching the splitter. `nanoom.cluster` is a convenience, not
a required step.

**`qcut` is the default regression binning, `gbmt_splits` is kept for parity.**
Balancing a continuous column requires binning it. gbmt-splits bins the n_*distinct
values* into equal-count groups; nanoom defaults to `qcut`, which bins so each bin
holds roughly equal n_*rows*. 
for gbmt-splits in the case of many repeated valuesm 20 copies of one value land
in a single value-rank bin and skew it. qcut may be better, but neither approach fixes a
extreme case of repeated values.
we kept in `binning_approach="gbmt_splits"` for 'reproducing' gbmt-based results.

**pulp/CBC, and the time limit returns rather than raises.** pulp ships with the
solver CBC, so there's no need for installing separate solvers. Reaching *exact*
optimality is much slower than near-optimality, so `time_limit_seconds` (default
300, might be turned back to 60*60) returns the best solution found so far
(instead of failing) (c.f. `relative_gap` allows stopping early on purpose).
Past 1000 clusters/rows `split()` because "exact optimum" for that may not be realistic.

**`split()` returns the dataframe, per row.** 
Both methods return `df` with a `split` column, one value per row. 
The per-cluster mapping is recoverable as  
`out.group_by(cluster_col).agg(pl.col("split").first())`. 
(may change in later versions where we take an polars expression-based api / non-polars-only approach)

**`cluster_col=None` means one cluster per row.** 
Same algorithm, just without grouping. The LP's cross-tab just gets one column per
row, and sklearn drops from `StratifiedGroupKFold` to `StratifiedKFold`. A `cluster`
column of row indices is added so grouping and the audit functions behave identically
in both modes. The cost is that the LP is then `n_rows * n_splits` binary variables,
which is what the size warning above is for.

**`eval` raises on length mismatch.** 
Because its inputs all describe the same rows, mismatch == bug. 
They accept polars Series, numpy arrays or lists, because the natural call site is straight off `split()`'s output columns.

**rdkit and bblean are optionals in the `chem` extra.** They're heavy and only
needed for chemical fingerprint-based clustering.`clustering.py` imports
`chem.py` inside the relevant `match` arms, so the core install stays polars +
pulp + scikit-learn.


## Open questions

- **How should `cluster()` pick granularity?** Cluster count trades leakage-avoidance
  against achievable balance: too few clusters and the LP has no freedom, too many and
  clusters stop representing real similarity. Currently the caller decides. An `auto`
  mode that tunes this against a held-out balance/leakage score is task 1 in
  [DEVELOPMENT.md](DEVELOPMENT.md), and the right objective for it is not settled.
- **Should nanoom score cluster quality at all?** `clustering.evaluate()` is a
  declared stub and `_silhouette_score` is written but unused. Silhouette on
  fingerprints is weakly informative, and the metric that actually matters is
  downstream split quality, not cluster compactness.
- **Ordinal classification.** `_task_type` maps integer columns to plain
  classification, so an ordered readout (e.g. activity bands) loses its ordering when
  binned. Whether to add an explicit ordinal task type, or let callers pre-bin, is
  open.
- **Overflow handling.** A cluster larger than the smallest requested split cannot be
  placed without blowing that split's size target. nanoom does not detect or repair
  this (neither does gbmt-splits); `test_single_oversized_cluster_does_not_crash`
  characterises today's behaviour rather than endorsing it.
