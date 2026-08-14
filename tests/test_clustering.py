import numpy as np
import pytest

from nanoom.clustering import (
    BIT_CLUSTERING_METHODS,
    _auto_n_clusters,
    _silhouette_score,
    cluster,
    dummy_hash_clustering,
    random_clustering,
)


# --- Fixtures ---
@pytest.fixture
def sample_descriptors():
    np.random.seed(42)
    return np.random.rand(100, 5)


@pytest.fixture
def sample_bit_descriptors():
    np.random.seed(42)
    return np.random.randint(0, 2, (100, 10)).astype(bool)


@pytest.fixture
def sample_distance_matrix():
    np.random.seed(42)
    mat = np.random.rand(10, 10)
    return (mat + mat.T) / 2  # Symmetric matrix


# --- Tests for Helper Functions ---
def test_auto_n_clusters():
    descriptors = np.zeros((50, 5))
    assert _auto_n_clusters(descriptors) == 6  # 50//10 + 1 = 6


def test_auto_n_clusters_empty():
    descriptors = np.zeros((0, 5))
    assert _auto_n_clusters(descriptors) == 1  # 0//10 + 1 = 1


# --- Tests for Clustering Methods ---
def test_random_clustering(sample_descriptors):
    clusters = random_clustering(sample_descriptors, n_clusters=5, seed=42)
    assert len(clusters) == len(sample_descriptors)
    assert set(clusters).issubset({0, 1, 2, 3, 4})


def test_random_clustering_auto_n_clusters(sample_descriptors):
    clusters = random_clustering(sample_descriptors, seed=42)
    assert len(clusters) == len(sample_descriptors)
    assert max(clusters) >= 1  # At least 2 clusters (100//10 + 1 = 11)


def test_dummy_hash_clustering(sample_descriptors):
    clusters = dummy_hash_clustering(sample_descriptors, n_clusters=5)
    assert len(clusters) == len(sample_descriptors)
    assert max(clusters) <= 4


# --- Tests for Generic Cluster Function ---
def test_cluster_kmeans(sample_descriptors):
    clusters = cluster(sample_descriptors, method="kmeans", n_clusters=3)
    assert len(clusters) == len(sample_descriptors)
    assert max(clusters) == 2  # 0, 1, 2


def test_cluster_dbscan(sample_descriptors):
    clusters = cluster(sample_descriptors, method="dbscan", eps=0.5, min_samples=5)
    assert len(clusters) == len(sample_descriptors)


def test_cluster_dbscan_jaccard(sample_bit_descriptors):
    clusters = cluster(
        sample_bit_descriptors,
        method="dbscan",
        metric="jaccard",
        eps=0.5,
        min_samples=5,
    )
    assert len(clusters) == len(sample_bit_descriptors)


def test_cluster_hdbscan(sample_descriptors):
    clusters = cluster(sample_descriptors, method="hdbscan", min_cluster_size=5)
    assert len(clusters) == len(sample_descriptors)


def test_cluster_random(sample_descriptors):
    clusters = cluster(sample_descriptors, method="random", n_clusters=3, seed=42)
    assert len(clusters) == len(sample_descriptors)
    assert set(clusters).issubset({0, 1, 2})


def test_cluster_hash_dummy(sample_descriptors):
    clusters = cluster(sample_descriptors, method="hash_dummy", n_clusters=3)
    assert len(clusters) == len(sample_descriptors)
    assert max(clusters) <= 2


def test_cluster_invalid_method(sample_descriptors):
    with pytest.raises(ValueError, match="Unknown clustering method"):
        cluster(sample_descriptors, method="invalid_method")  # ty: ignore[invalid-argument-type]


def test_cluster_n_clusters_as_fraction(sample_descriptors):
    clusters = cluster(sample_descriptors, method="kmeans", n_clusters=0.1)
    assert len(clusters) == len(sample_descriptors)
    assert max(clusters) == 10  # 100//(1/0.1) + 1 = 101, but capped by data


def test_cluster_square_distance_matrix(sample_distance_matrix):
    clusters = cluster(sample_distance_matrix, method="kmeans", n_clusters=2)
    assert len(clusters) == len(sample_distance_matrix)


# --- Tests for Silhouette Score ---
def test_silhouette_score_valid(sample_descriptors):
    clusters = cluster(sample_descriptors, method="kmeans", n_clusters=3)
    score = _silhouette_score(sample_descriptors, clusters)
    assert score is not None
    assert -1 <= score <= 1


def test_silhouette_score_single_cluster(sample_descriptors):
    clusters = np.zeros(len(sample_descriptors), dtype=int)
    score = _silhouette_score(sample_descriptors, clusters)
    assert score is None


def test_silhouette_score_large_dataset():
    np.random.seed(42)
    descriptors = np.random.rand(15000, 5)
    clusters = np.random.randint(0, 3, size=15000)
    score = _silhouette_score(descriptors, clusters)
    assert score is not None


# --- Tests for BIT Clustering Methods (Mocked) ---
def test_cluster_bitbirch(sample_bit_descriptors):
    clusters = cluster(sample_bit_descriptors, method="bitbirch")
    assert len(clusters) == len(sample_bit_descriptors)
    assert len(np.unique(clusters)) > 1


def test_cluster_sphere_exclusion(sample_bit_descriptors):
    clusters = cluster(sample_bit_descriptors, method="sphere_exclusion")
    assert len(clusters) == len(sample_bit_descriptors)


@pytest.mark.parametrize("method", sorted(BIT_CLUSTERING_METHODS))
@pytest.mark.parametrize(
    "descriptors",
    [
        np.random.default_rng(42).random((20, 10)),  # floats
        np.random.default_rng(42).integers(0, 5, (20, 10)),  # counts
        np.packbits(
            np.random.default_rng(42).integers(0, 2, (20, 64), dtype=np.uint8), axis=1
        ),  # packed fingerprints (8bit->1byte(0-255)),
    ],
    ids=["floats", "counts", "packed"],
)
def test_bit_clustering_rejects_non_binary(descriptors, method):
    with pytest.raises(ValueError, match="binary descriptors"):
        cluster(descriptors, method=method)


@pytest.mark.parametrize("method", sorted(BIT_CLUSTERING_METHODS))
@pytest.mark.parametrize("dtype", [bool, np.uint8, np.int64, np.float64], ids=str)
def test_bit_clustering_accepts_binary(sample_bit_descriptors, method, dtype):
    clusters = cluster(sample_bit_descriptors.astype(dtype), method=method)
    assert len(clusters) == len(sample_bit_descriptors)
