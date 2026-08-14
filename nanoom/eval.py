from typing import Literal

import numpy as np


def _numpy_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """of 2 2D matrices"""
    return np.linalg.norm(a[:, np.newaxis, :] - b[np.newaxis, :, :], axis=-1)


def _numpy_jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = a[:, np.newaxis, :]  # shape (N, 1, D)
    B = b[np.newaxis, :, :]  # shape (1, M, D)

    intersection = np.logical_and(A, B).sum(axis=-1)  # shape (N, M)
    union = np.logical_or(A, B).sum(axis=-1)  # shape (N, M)
    return 1 - intersection / union


def _same_length_arrays(
    a, b, a_name: str, b_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """As numpy arrays (accepts polars Series / lists), raising if lengths differ."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b):
        raise ValueError(
            f"{a_name} has length {len(a)} but {b_name} has length {len(b)}; "
            "both must be per-row and aligned"
        )
    return a, b


def min_distances_splits(
    descriptors: np.ndarray, splits: np.ndarray, metric: Literal["euclidean", "jaccard"]
) -> list[dict]:
    descriptors, splits = _same_length_arrays(
        descriptors, splits, "descriptors", "splits"
    )
    # loop ovebr the different values in the split
    results = []
    match metric:
        case "euclidean":
            distance_function = _numpy_euclidean
        case "jaccard":
            distance_function = _numpy_jaccard
        case _:
            raise ValueError(f"Unknown metric: {metric}")
    for j in sorted(set(splits)):
        res = {"split": j}
        split_descriptors = descriptors[np.where(splits == j)[0]]
        other_descriptors = descriptors[np.where(splits != j)[0]]
        distances = distance_function(split_descriptors, other_descriptors)
        res["ext_distance_min"] = distances.min()
        res["ext_distance_mean"] = distances.mean()
        res["ext_distance_median"] = np.median(distances)
        res["ext_distance_std"] = distances.std()
        intra_distances = distance_function(split_descriptors, split_descriptors)
        # remove diagonal
        intra_distances = intra_distances[~np.eye(intra_distances.shape[0], dtype=bool)]
        res["int_distance_min"] = intra_distances.min()
        res["int_distance_mean"] = intra_distances.mean()
        res["int_distance_median"] = np.median(intra_distances)
        res["int_distance_std"] = intra_distances.std()
        results.append(res)
    return results


def split_y_means(y, splits):
    """Mean of y per split, ordered by sorted split label.

    Indexed by position, not by the split label itself: labels are not guaranteed
    to be 0..n-1 (user may pass names, or a subset of folds)."""
    y, splits = _same_length_arrays(y, splits, "y", "splits")
    means = np.zeros(len(set(splits)))
    for i, split in enumerate(sorted(set(splits))):
        split_indices = np.where(splits == split)[0]
        means[i] = y[split_indices].mean()
    return means


def check_distribution_y_similar(y, splits):
    means = split_y_means(y, splits)
    overall_mean = np.asarray(y).mean()
    if not np.allclose(means, overall_mean, rtol=0.1):
        raise ValueError(
            f"Means of y in splits are not similar: {means}, overall mean: {overall_mean}"
        )


def check_no_group_overlap(group_by, splits):
    group_by, splits = _same_length_arrays(group_by, splits, "group_by", "splits")
    unique_groups = np.unique(group_by)
    group_split_counts = np.array(
        [np.unique(splits[group_by == g]).size for g in unique_groups]
    )
    if np.any(group_split_counts > 1):
        problematic = unique_groups[group_split_counts > 1]
        raise ValueError(f"Groups {problematic} have samples in multiple splits.")
