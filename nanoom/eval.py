from typing import Literal

import jax.numpy as jnp
import numpy as np


def _jax_euclidean(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """of 2 2D matrices"""
    return jnp.linalg.norm(a[:, jnp.newaxis, :] - b[jnp.newaxis, :, :], axis=-1)


def _jax_jaccard(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    A = a[:, jnp.newaxis, :]  # shape (N, 1, D)
    B = b[jnp.newaxis, :, :]  # shape (1, M, D)

    intersection = jnp.logical_and(A, B).sum(axis=-1)  # shape (N, M)
    union = jnp.logical_or(A, B).sum(axis=-1)  # shape (N, M)
    return 1 - intersection / union


def _numpy_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """of 2 2D matrices"""
    return np.linalg.norm(a[:, np.newaxis, :] - b[np.newaxis, :, :], axis=-1)


def _numpy_jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = a[:, np.newaxis, :]  # shape (N, 1, D)
    B = b[np.newaxis, :, :]  # shape (1, M, D)

    intersection = np.logical_and(A, B).sum(axis=-1)  # shape (N, M)
    union = np.logical_or(A, B).sum(axis=-1)  # shape (N, M)
    return 1 - intersection / union


def numpy_min_distances_splits(
    descriptors: np.ndarray, splits: np.ndarray, metric: Literal["euclidean", "jaccard"]
) -> list[dict]:
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

    def jax_min_distances_splits(
        descriptors: jnp.ndarray,
        splits: jnp.ndarray,
        metric: Literal["euclidean", "jaccard"],
    ) -> list[dict]:
        # loop over the different values in the split
        results = []
        match metric:
            case "euclidean":
                distance_function = _jax_euclidean
            case "jaccard":
                distance_function = _jax_jaccard
            case _:
                raise ValueError(f"Unknown metric: {metric}")
        for j in sorted(set(splits.tolist())):
            res = {"split": j}
            split_indices = jnp.where(splits == j)[0]
            other_indices = jnp.where(splits != j)[0]
            split_descriptors = descriptors[split_indices]
            other_descriptors = descriptors[other_indices]
            distances = distance_function(split_descriptors, other_descriptors)
            res["ext_distance_min"] = float(distances.min())
            res["ext_distance_mean"] = float(distances.mean())
            res["ext_distance_median"] = float(jnp.median(distances))
            res["ext_distance_std"] = float(distances.std())
            intra_distances = distance_function(split_descriptors, split_descriptors)
            # remove diagonal
            mask = ~jnp.eye(intra_distances.shape[0], dtype=bool)
            intra_distances = intra_distances[mask]
            res["int_distance_min"] = float(intra_distances.min())
            res["int_distance_mean"] = float(intra_distances.mean())
            res["int_distance_median"] = float(jnp.median(intra_distances))
            res["int_distance_std"] = float(intra_distances.std())
            results.append(res)
        return results


min_distances_splits = jax_min_distances_splits


def split_y_means(y, splits):
    means = np.zeros(len(set(splits)))
    for split in sorted(set(splits)):
        split_indices = np.where(splits == split)[0]
        split_y = y[split_indices]
        means[split] = split_y.mean()
    return means


def check_distribution_y_similar(y, splits):
    means = split_y_means(y, splits)
    overall_mean = y.mean()
    if not np.allclose(means, overall_mean, rtol=0.1):
        raise ValueError(
            f"Means of y in splits are not similar: {means}, overall mean: {overall_mean}"
        )


def check_no_group_overlap(group_by, splits):
    unique_groups = np.unique(group_by)
    group_split_counts = np.array(
        [np.unique(splits[group_by == g]).size for g in unique_groups]
    )
    if np.any(group_split_counts > 1):
        problematic = unique_groups[group_split_counts > 1]
        raise ValueError(f"Groups {problematic} have samples in multiple splits.")
