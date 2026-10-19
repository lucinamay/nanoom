"""clustering logic for various clustering"""

import logging as lg
from typing import Literal

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

ClusteringMethod = Literal[
    "kmeans",
    "dbscan",
    "hdbscan",
    "sphere_exclusion",
    "bitbirch",
    "maxmin",
    "leader_picker",
    "hash_dummy",
    "random",
]

BIT_CLUSTERING_METHODS: set[ClusteringMethod] = {
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


def _is_binary_descriptor(array: np.ndarray) -> bool:
    return np.array_equal(array, array.astype(bool))


def dummy_hash_clustering(
    descriptors: np.ndarray, n_clusters: int | None = None
) -> np.ndarray:
    n_clusters = n_clusters if n_clusters is not None else _auto_n_clusters(descriptors)
    hashes = np.array([hash(row.tobytes()) for row in descriptors])
    return (hashes % n_clusters).astype(np.int32)


def cluster(
    descriptors: np.ndarray,
    method: ClusteringMethod,
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
        - identical descriptor rows are clustered once and share a label.
        - If descriptor is a bit vector, DBSCAN will use Jaccard metric by default.
        - If `n_clusters` is provided as a float between 0 and 1, it is interpreted
          as the fraction of the dataset size to use as the number of clusters (plus 1).
    """
    # if descriptors are square matrix, assume its a distance matrix and convert to condensed form
    if len(descriptors.shape) == 2 and descriptors.shape[0] == descriptors.shape[1]:
        lg.warning(
            "Descriptors appear to be a square distance matrix, but clustering methods expect feature vectors. Proceeding anyway."
        )
    descriptors, inverse = np.unique(descriptors, axis=0, return_inverse=True)
    n_clusters = kwargs.get("n_clusters", None)
    if n_clusters and n_clusters < 1 and n_clusters > 0:
        kwargs["n_clusters"] = int(descriptors.shape[0] // (1 / n_clusters) + 1)
        lg.info(
            f"Interpreting n_clusters={n_clusters} as fraction, setting n_clusters to {kwargs['n_clusters']}"
        )

    if method in BIT_CLUSTERING_METHODS and not _is_binary_descriptor(descriptors):
        raise ValueError(
            f"{method} requires binary descriptors, got dtype {descriptors.dtype} with"
            + " non-binary values. Pass descriptors.astype(bool) if you want to binarize"
        )

    match method:
        case "kmeans":
            labels = KMeans(*args, **kwargs).fit(descriptors).labels_
        case "dbscan":
            if "metric" not in kwargs and _is_binary_descriptor(descriptors):
                kwargs["metric"] = "jaccard"
            if kwargs.get("metric", None) == "jaccard":
                descriptors = descriptors.astype(bool)  # prevent warning
            labels = DBSCAN(*args, **kwargs).fit(descriptors).labels_
        case "hdbscan":
            labels = (
                HDBSCAN(*args, copy=kwargs.pop("copy", False), **kwargs)
                .fit(descriptors)
                .labels_
            )
        case "sphere_exclusion":
            from nanoom.chem import _sphere_exclusion

            labels = _sphere_exclusion(descriptors, *args, **kwargs)
        case "bitbirch":
            from nanoom.chem import _bitbirch

            labels = _bitbirch(descriptors, *args, **kwargs)
        case "maxmin":
            from nanoom.chem import maxmin_clustering

            labels = maxmin_clustering(descriptors, *args, **kwargs)
        case "leader_picker":
            from nanoom.chem import leader_picker_clustering

            labels = leader_picker_clustering(descriptors, *args, **kwargs)
        case "hash_dummy":
            labels = dummy_hash_clustering(descriptors, *args, **kwargs)
        case "random":
            labels = random_clustering(descriptors, *args, **kwargs)
        case _:
            raise ValueError(f"Unknown clustering method: {method}")
    assert labels is not None
    return labels[inverse.reshape(-1)]  # numpy 2.0.0 returns a 2-D inverse for axis=0


def merge_clusters(
    descriptors: np.ndarray,
    clusters: np.ndarray,
    *,
    threshold: float | None = None,
    n_clusters: int | None = None,
    linkage: Literal[
        "single", "complete", "average", "weighted", "centroid", "median", "ward"
    ] = "average",
    metric: str = "euclidean",
) -> np.ndarray:
    """Merge clusters by hierarchical clustering of their centroids (mean descriptor).

    Exactly one of `threshold` (merge clusters whose centroid linkage distance is
    <= threshold) or `n_clusters` (merge down to at most n_clusters) is required.
    `linkage` and `metric` go to `scipy.cluster.hierarchy.linkage`; "centroid",
    "median" and "ward" need metric="euclidean".

    Returns labels 0..k-1 per row; noise (-1, from dbscan / hdbscan) stays -1.
    """
    from scipy.cluster import hierarchy

    descriptors, clusters = np.asarray(descriptors), np.asarray(clusters)
    if len(descriptors) != len(clusters):
        raise ValueError(
            f"descriptors has length {len(descriptors)} but clusters has length "
            + f"{len(clusters)}; both must be per-row and aligned"
        )
    if (threshold is None) == (n_clusters is None):
        raise ValueError("pass exactly one of `threshold` or `n_clusters`")

    keep = clusters != -1
    ids, inverse = np.unique(clusters[keep], return_inverse=True)
    if len(ids) < 2:
        merged = np.ones(len(ids), dtype=int)
    else:
        # ponytail: centroids keep this O(k^2); per-row linkage would be O(n^2)
        centroids = np.zeros((len(ids), descriptors.shape[1]))
        np.add.at(centroids, inverse, descriptors[keep])
        centroids /= np.bincount(inverse)[:, None]
        z = hierarchy.linkage(centroids, method=linkage, metric=metric)
        merged = (
            hierarchy.fcluster(z, threshold, "distance")
            if threshold is not None
            else hierarchy.fcluster(z, n_clusters, "maxclust")
        )
    out = np.full(len(clusters), -1)
    out[keep] = merged[inverse] - 1  # fcluster labels 1..k
    return out


def evaluate(
    descriptors: np.ndarray,
    clusters: np.ndarray,
    method: Literal["silhouette", "calinski-harabasz", "davies-bouldin"] = "silhouette",
    metric: str | None = None,
    sample_size: int | None = 10_000,
    seed: int = 0,
) -> float:
    """Cluster quality score of `clusters` on `descriptors` (sklearn.metrics).

    silhouette: -1..1, higher is better. `metric` defaults to jaccard for binary
        descriptors, else euclidean; "precomputed" takes the n x n distance
        matrix. Scored on `sample_size` random rows (None: all rows).
    calinski-harabasz (higher is better), davies-bouldin (lower is better):
        euclidean on feature vectors; `metric`, `sample_size` are not used.

    Noise (-1) counts as one cluster; filter it out to score the clusters alone.
    Raises (sklearn) unless 2 <= number of clusters < number of rows.
    """
    match method:
        case "silhouette":
            if metric is None:
                metric = (
                    "jaccard" if _is_binary_descriptor(descriptors) else "euclidean"
                )
            if metric == "jaccard":
                descriptors = descriptors.astype(bool)  # prevent warning
            return float(
                silhouette_score(
                    descriptors,
                    clusters,
                    metric=metric,
                    sample_size=sample_size,
                    random_state=seed,
                )
            )
        case "calinski-harabasz":
            return float(calinski_harabasz_score(descriptors, clusters))
        case "davies-bouldin":
            return float(davies_bouldin_score(descriptors, clusters))
        case _:
            raise ValueError(f"Unknown evaluation method: {method}")
