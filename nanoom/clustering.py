"""clustering logic for various clustering"""

import logging as lg
from typing import Literal

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import silhouette_score as sk_silhouette

BIT_CLUSTERING_METHODS = {
    "sphere_exclusion",
    "bitbirch",
    "maxmin",
    "leader_picker",
}


def _auto_n_clusters(descriptors: np.ndarray) -> int:
    n_clusters = len(descriptors) // 10 + 1
    lg.info(f"Auto setting n_clusters to {n_clusters}")
    return n_clusters


def random_clustering(
    descriptors: np.ndarray, n_clusters: int | None = None, seed: int = 0
) -> np.ndarray:
    n_clusters = n_clusters if n_clusters is not None else _auto_n_clusters(descriptors)
    return (
        np.asarray(np.random.RandomState(seed=seed).permutation(len(descriptors)))
        % n_clusters
    )


def dummy_hash_clustering(
    descriptors: np.ndarray, n_clusters: int | None = None
) -> np.ndarray:
    n_clusters = n_clusters if n_clusters is not None else _auto_n_clusters(descriptors)
    hashes = np.array([hash(row.tobytes()) for row in descriptors])
    return (hashes % n_clusters).astype(np.int32)


def cluster(
    descriptors: np.ndarray,
    method: Literal[
        "kmeans",
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

    Notes:
        - If descriptor is a bit vector, DBSCAN will use Jaccard metric by default.
        - If `n_clusters` is provided as a float between 0 and 1, it is interpreted
          as the fraction of the dataset size to use as the number of clusters (plus 1).
    """
    # if descriptors are square matrix, assume its a distance matrix and convert to condensed form
    if len(descriptors.shape) == 2 and descriptors.shape[0] == descriptors.shape[1]:
        lg.warning(
            "Descriptors appear to be a square distance matrix, but clustering methods expect feature vectors. Proceeding anyway."
        )
    n_clusters = kwargs.get("n_clusters", None)
    if n_clusters and n_clusters < 1 and n_clusters > 0:
        kwargs["n_clusters"] = int(descriptors.shape[0] // (1 / n_clusters) + 1)
        lg.info(
            f"Interpreting n_clusters={n_clusters} as fraction, setting n_clusters to {kwargs['n_clusters']}"
        )

    match method:
        case "kmeans":
            return KMeans(*args, **kwargs).fit(descriptors).labels_
        case "dbscan":
            if kwargs.get("metric", None) == "jaccard":
                descriptors = descriptors.astype(bool)  # prevent warning
            return DBSCAN(*args, **kwargs).fit(descriptors).labels_
        case "hdbscan":
            return (
                HDBSCAN(*args, copy=kwargs.pop("copy", False), **kwargs)
                .fit(descriptors)
                .labels_
            )
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
