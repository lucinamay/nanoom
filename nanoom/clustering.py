"""clustering logic for various clustering"""

import logging as lg
from typing import Literal, Sequence

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import silhouette_score as sk_silhouette


def _auto_n_clusters(descriptors: np.ndarray) -> int:
    return len(descriptors) // 10 + 1


def random_clustering(
    descriptors: np.ndarray, n_clusters: int | None = None, seed: int = 0
) -> np.ndarray:
    n_clusters = n_clusters if n_clusters is not None else _auto_n_clusters(descriptors)
    return (
        np.ndarray(np.random.RandomState(seed=seed).permutation(len(descriptors)))
        % n_clusters
    )


def dummy_hash_clustering(
    descriptors: np.ndarray, n_clusters: int | None = None
) -> np.ndarray:
    n_clusters = n_clusters if n_clusters is not None else _auto_n_clusters(descriptors)
    return np.ndarray(descriptors) % n_clusters


def cluster(
    descriptors: np.ndarray,
    method: Literal[
        "kmeans",
        "kmeans_10pct",
        "dbscan",
        "hdbscan",
        "sphere_exclusion",
        "bitbirch",
        "maxmin",
        "leader_picker",
        "hash_dummy",
        "random",
    ],
    *args,
    **kwargs,
) -> np.ndarray:
    """Generic clustering function that selects clustering method based on kwargs.

    Args:
        descriptors (np.ndarray): array of descriptors to cluster
        *args: additional arguments to pass to the clustering function
        **kwargs: additional keyword arguments to pass to the clustering function

    Returns:
        np.ndarray: array of cluster indices per descriptor

    Notes on the methods:
        _kmeans_
    """
    # if descriptors are square matrix, assume its a distance matrix and convert to condensed form
    if len(descriptors.shape) == 2 and descriptors.shape[0] == descriptors.shape[1]:
        raise NotImplementedError(
            "Clustering from distance matrix not implemented yet."
        )

    match method:
        case "kmeans":
            return KMeans(*args, **kwargs).fit(descriptors).labels_
        case "kmeans_10pct":
            assert "n_clusters" not in kwargs, (
                "n_clusters should not be provided for kmeans_10pct"
            )
            return (
                KMeans(n_clusters=(descriptors.shape[0] // 10 + 1), *args, **kwargs)
                .fit(descriptors)
                .labels_
            )
        case "dbscan":
            if kwargs.get("metric", None) == "jaccard":
                descriptors = descriptors.astype(bool)  # prevent warning
            return DBSCAN(*args, **kwargs).fit(descriptors).labels_
        case "hdbscan":
            return HDBSCAN(*args, **kwargs).fit(descriptors).labels_
        case "sphere_exclusion":
            from nanoom.chem import _sphere_exclusion

            return _sphere_exclusion(descriptors, *args, **kwargs)
        case "bitbirch":
            from nanoom.chem import _bitbirch

            return _bitbirch(descriptors, *args, **kwargs)
        case "maxmin":
            from nanoom.chem import maxmin_clustering

            return maxmin_clustering(descriptors, *args, **kwargs)
        case "leader_picker":
            from nanoom.chem import leader_picker_clustering

            return leader_picker_clustering(descriptors, *args, **kwargs)
        case "hash_dummy":
            return dummy_hash_clustering(descriptors, *args, **kwargs)
        case "random":
            return random_clustering(descriptors, *args, **kwargs)
        case _:
            raise ValueError(f"Unknown clustering method: {method}")


def _silhouette_score(
    descriptors: np.ndarray, clusters: np.ndarray, **kwargs
) -> float | None:
    """Compute silhouette score if valid, otherwise return None."""
    unique_labels = np.unique(clusters)
    n_valid_clusters = len(unique_labels[unique_labels != -1])
    metric = "euclidean" if descriptors.max() > 1 else "jaccard"
    if n_valid_clusters >= 2 and len(clusters) > n_valid_clusters:
        if descriptors.shape[0] > 10000:
            lg.info(
                f"Computing silhouette score on a sample of 10,000 from {descriptors.shape[0]} datapoints"
            )
            np.random.seed(1)
            sample_indices = np.random.choice(
                descriptors.shape[0], size=10000, replace=False
            )
            return sk_silhouette(
                descriptors[sample_indices],
                clusters[sample_indices],
                metric=metric,
                **kwargs,
            )
        return sk_silhouette(descriptors, clusters, **kwargs)

    lg.warning(f"Only {n_valid_clusters} valid cluster(s) found, returning None")
    return None


def evaluate(
    descriptors: np.ndarray,
    clusters: np.ndarray,
    method: Literal["silhouette", "inertia", "calinski-harabasz", "davies-bouldin"],
    **kwargs,
):
    pass
