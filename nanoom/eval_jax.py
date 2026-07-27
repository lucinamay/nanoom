from typing import Literal

import jax.numpy as jnp


def euclidean(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """of 2 2D matrices"""
    return jnp.linalg.norm(a[:, jnp.newaxis, :] - b[jnp.newaxis, :, :], axis=-1)


def jaccard(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    A = a[:, jnp.newaxis, :]  # shape (N, 1, D)
    B = b[jnp.newaxis, :, :]  # shape (1, M, D)

    intersection = jnp.logical_and(A, B).sum(axis=-1)  # shape (N, M)
    union = jnp.logical_or(A, B).sum(axis=-1)  # shape (N, M)
    return 1 - intersection / union


def min_distances_splits(
    descriptors: jnp.ndarray,
    splits: jnp.ndarray,
    metric: Literal["euclidean", "jaccard"],
) -> list[dict]:
    # loop over the different values in the split
    results = []
    match metric:
        case "euclidean":
            distance_function = euclidean
        case "jaccard":
            distance_function = jaccard
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
