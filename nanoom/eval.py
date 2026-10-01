from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def _numpy_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """of 2 2D matrices"""
    return np.linalg.norm(a[:, np.newaxis, :] - b[np.newaxis, :, :], axis=-1)


def _numpy_jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = a[:, np.newaxis, :]  # shape (N, 1, D)
    B = b[np.newaxis, :, :]  # shape (1, M, D)

    intersection = np.logical_and(A, B).sum(axis=-1)  # shape (N, M)
    union = np.logical_or(A, B).sum(axis=-1)  # shape (N, M)
    return 1 - intersection / union


def _distance_function(metric: str):
    match metric:
        case "euclidean":
            return _numpy_euclidean
        case "jaccard":
            return _numpy_jaccard
        case _:
            raise ValueError(f"Unknown metric: {metric}")


def _same_length_arrays(
    a: npt.ArrayLike, b: npt.ArrayLike, a_name: str, b_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """As numpy arrays (accepts polars Series / lists), raising if lengths differ."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b):
        raise ValueError(
            f"{a_name} has length {len(a)} but {b_name} has length {len(b)}; "
            + "both must be per-row and aligned"
        )
    return a, b


class SplitDistances(TypedDict):
    """Distance stats for one split: `ext_` against all other splits, `int_` within."""

    split: int
    ext_distance_min: float
    ext_distance_mean: float
    ext_distance_median: float
    ext_distance_std: float
    int_distance_min: float
    int_distance_mean: float
    int_distance_median: float
    int_distance_std: float


def min_distances_splits(
    descriptors: np.ndarray, splits: np.ndarray, metric: Literal["euclidean", "jaccard"]
) -> list[SplitDistances]:
    descriptors, splits = _same_length_arrays(
        descriptors, splits, "descriptors", "splits"
    )
    # loop ovebr the different values in the split
    results: list[SplitDistances] = []
    distance_function = _distance_function(metric)
    for j in sorted(set(splits)):
        split_descriptors = descriptors[np.where(splits == j)[0]]
        other_descriptors = descriptors[np.where(splits != j)[0]]
        distances = distance_function(split_descriptors, other_descriptors)
        intra_distances = distance_function(split_descriptors, split_descriptors)
        # remove diagonal
        intra_distances = intra_distances[~np.eye(intra_distances.shape[0], dtype=bool)]
        results.append(
            SplitDistances(
                split=int(j),
                ext_distance_min=float(distances.min()),
                ext_distance_mean=float(distances.mean()),
                ext_distance_median=float(np.median(distances)),
                ext_distance_std=float(distances.std()),
                int_distance_min=float(intra_distances.min()),
                int_distance_mean=float(intra_distances.mean()),
                int_distance_median=float(np.median(intra_distances)),
                int_distance_std=float(intra_distances.std()),
            )
        )
    return results


def split_y_means(y: npt.ArrayLike, splits: npt.ArrayLike) -> np.ndarray:
    """Mean of y per split, ordered by sorted split label.

    Indexed by position, not by the split label itself: labels are not guaranteed
    to be 0..n-1 (user may pass names, or a subset of folds)."""
    y, splits = _same_length_arrays(y, splits, "y", "splits")
    means = np.zeros(len(set(splits)))
    for i, split in enumerate(sorted(set(splits))):
        split_indices = np.where(splits == split)[0]
        means[i] = y[split_indices].mean()
    return means


def check_distribution_y_similar(y: npt.ArrayLike, splits: npt.ArrayLike) -> None:
    means = split_y_means(y, splits)
    overall_mean = np.asarray(y).mean()
    if not np.allclose(means, overall_mean, rtol=0.1):
        raise ValueError(
            f"Means of y in splits are not similar: {means}, overall mean: {overall_mean}"
        )


def check_no_group_overlap(group_by: npt.ArrayLike, splits: npt.ArrayLike) -> None:
    group_by, splits = _same_length_arrays(group_by, splits, "group_by", "splits")
    unique_groups = np.unique(group_by)
    group_split_counts = np.array(
        [np.unique(splits[group_by == g]).size for g in unique_groups]
    )
    if np.any(group_split_counts > 1):
        problematic = unique_groups[group_split_counts > 1]
        raise ValueError(f"Groups {problematic} have samples in multiple splits.")


def nearest_neighbour_distances(
    descriptors: np.ndarray, splits: np.ndarray, metric: Literal["euclidean", "jaccard"]
) -> np.ndarray:
    """Per row: distance to the nearest row in a *different* split.

    Row-level counterpart of `min_distances_splits` (which only returns summary
    stats); small values mark leakage candidates. Needs at least 2 splits.
    """
    descriptors, splits = _same_length_arrays(
        descriptors, splits, "descriptors", "splits"
    )
    if len(np.unique(splits)) < 2:
        raise ValueError("need >= 2 distinct splits to find a neighbour in another")
    distance_function = _distance_function(metric)
    nearest = np.empty(len(splits))
    for j in np.unique(splits):
        in_split = splits == j
        distances = distance_function(descriptors[in_split], descriptors[~in_split])
        nearest[in_split] = distances.min(axis=1)
    return nearest


def adversarial_auc(
    descriptors: np.ndarray, splits: np.ndarray, n_folds: int = 5, seed: int = 0
) -> dict[int, float]:
    """Per split label: cross-validated ROC-AUC of a classifier separating that split's
    rows from all other rows, using `descriptors`.

    ~0.5 means the split is indistinguishable from the rest; ~1.0 means it occupies
    a separate region of descriptor space.
    """
    descriptors, splits = _same_length_arrays(
        descriptors, splits, "descriptors", "splits"
    )
    if len(np.unique(splits)) < 2:
        raise ValueError("need >= 2 distinct splits to tell one from the rest")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    clf = RandomForestClassifier(random_state=seed)
    aucs: dict[int, float] = {}
    for j in np.unique(splits):
        is_split = splits == j
        proba = cross_val_predict(
            clf, descriptors, is_split, cv=cv, method="predict_proba"
        )
        aucs[int(j)] = float(roc_auc_score(is_split, proba[:, 1]))
    return aucs
