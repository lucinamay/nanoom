import numpy as np
import pytest

from nanoom.clustering import (
    BIT_CLUSTERING_METHODS,
    ClusteringMethod,
    _auto_n_clusters,
    _silhouette_score,
    cluster,
    dummy_hash_clustering,
    random_clustering,
)


# --- Fixtures ---
@pytest.fixture
def sample_descriptors() -> np.ndarray:
    np.random.seed(42)
    return np.random.rand(100, 5)


@pytest.fixture
def sample_bit_descriptors() -> np.ndarray:
    np.random.seed(42)
    return np.random.randint(0, 2, (100, 10)).astype(bool).astype(int)


@pytest.fixture
def sample_distance_matrix() -> np.ndarray:
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
def test_random_clustering(sample_descriptors: np.ndarray):
    clusters = random_clustering(sample_descriptors, n_clusters=5, seed=42)
    assert len(clusters) == len(sample_descriptors)
    assert set(clusters).issubset({0, 1, 2, 3, 4})


def test_random_clustering_auto_n_clusters(sample_descriptors: np.ndarray):
    clusters = random_clustering(sample_descriptors, seed=42)
    assert len(clusters) == len(sample_descriptors)
    assert max(clusters) >= 1  # At least 2 clusters (100//10 + 1 = 11)


def test_dummy_hash_clustering(sample_descriptors: np.ndarray):
    clusters = dummy_hash_clustering(sample_descriptors, n_clusters=5)
    assert len(clusters) == len(sample_descriptors)
    assert max(clusters) <= 4


# --- Tests for Generic Cluster Function ---
def test_cluster_kmeans(sample_descriptors: np.ndarray):
    clusters = cluster(sample_descriptors, method="kmeans", n_clusters=3)
    assert len(clusters) == len(sample_descriptors)
    assert max(clusters) == 2  # 0, 1, 2


def test_cluster_dbscan(sample_descriptors: np.ndarray):
    clusters = cluster(sample_descriptors, method="dbscan", eps=0.5, min_samples=5)
    assert len(clusters) == len(sample_descriptors)


def test_cluster_dbscan_jaccard(sample_bit_descriptors: np.ndarray):
    clusters = cluster(
        sample_bit_descriptors,
        method="dbscan",
        metric="jaccard",
        eps=0.5,
        min_samples=5,
    )
    assert len(clusters) == len(sample_bit_descriptors)


def test_cluster_hdbscan(sample_descriptors: np.ndarray):
    clusters = cluster(sample_descriptors, method="hdbscan", min_cluster_size=5)
    assert len(clusters) == len(sample_descriptors)


def test_cluster_random(sample_descriptors: np.ndarray):
    clusters = cluster(sample_descriptors, method="random", n_clusters=3, seed=42)
    assert len(clusters) == len(sample_descriptors)
    assert set(clusters).issubset({0, 1, 2})


def test_cluster_hash_dummy(sample_descriptors: np.ndarray):
    clusters = cluster(sample_descriptors, method="hash_dummy", n_clusters=3)
    assert len(clusters) == len(sample_descriptors)
    assert max(clusters) <= 2


def test_cluster_invalid_method(sample_descriptors: np.ndarray):
    with pytest.raises(ValueError, match="Unknown clustering method"):
        cluster(sample_descriptors, method="invalid_method")  # ty: ignore[invalid-argument-type]


def test_cluster_n_clusters_as_fraction(sample_descriptors: np.ndarray):
    clusters = cluster(sample_descriptors, method="kmeans", n_clusters=0.1)
    assert len(clusters) == len(sample_descriptors)
    assert max(clusters) == 10  # 100//(1/0.1) + 1 = 101, but capped by data


def test_cluster_square_distance_matrix(sample_distance_matrix: np.ndarray):
    clusters = cluster(sample_distance_matrix, method="kmeans", n_clusters=2)
    assert len(clusters) == len(sample_distance_matrix)


# --- Tests for Silhouette Score ---
def test_silhouette_score_valid(sample_descriptors: np.ndarray):
    clusters = cluster(sample_descriptors, method="kmeans", n_clusters=3)
    score = _silhouette_score(sample_descriptors, clusters)
    assert score is not None
    assert -1 <= score <= 1


def test_silhouette_score_single_cluster(sample_descriptors: np.ndarray):
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
def test_cluster_bitbirch(sample_bit_descriptors: np.ndarray):
    clusters = cluster(sample_bit_descriptors, method="bitbirch")
    assert len(clusters) == len(sample_bit_descriptors)
    assert len(np.unique(clusters)) > 1


def test_cluster_sphere_exclusion(sample_bit_descriptors: np.ndarray):
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
def test_bit_clustering_rejects_non_binary(
    descriptors: np.ndarray, method: ClusteringMethod
):
    with pytest.raises(ValueError, match="binary descriptors"):
        cluster(descriptors, method=method)


@pytest.mark.parametrize("method", sorted(BIT_CLUSTERING_METHODS))
@pytest.mark.parametrize("dtype", [bool, np.uint8, np.int64, np.float64], ids=str)
def test_bit_clustering_accepts_binary(
    sample_bit_descriptors: np.ndarray, method: ClusteringMethod, dtype: type
):
    clusters = cluster(sample_bit_descriptors.astype(dtype), method=method)
    assert len(clusters) == len(sample_bit_descriptors)


# --- Tests for duplicate descriptor rows ---
def _n_split_duplicates(descriptors: np.ndarray, clusters: np.ndarray) -> int:
    """Number of distinct rows whose copies received more than one cluster label."""
    _, inverse = np.unique(descriptors, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    return sum(
        len(np.unique(clusters[inverse == row])) > 1 for row in range(inverse.max() + 1)
    )


def _shuffled_copies(base: np.ndarray, n_copies: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.tile(base, (n_copies, 1))[rng.permutation(n_copies * len(base))]


@pytest.mark.parametrize("method", ClusteringMethod.__args__)
def test_identical_rows_share_cluster_for_every_method(method: ClusteringMethod):
    base = np.random.default_rng(0).integers(0, 2, (30, 32))
    descriptors = _shuffled_copies(base, n_copies=3, seed=0)
    kwargs = {
        "kmeans": {"n_clusters": 5, "random_state": 0},
        "maxmin": {"n_clusters": 5, "seed": 0},
        "random": {"n_clusters": 5, "seed": 0},
    }.get(method, {})
    clusters = cluster(descriptors, method=method, **kwargs)
    assert len(clusters) == len(descriptors)
    assert _n_split_duplicates(descriptors, clusters) == 0
    np.testing.assert_array_equal(
        clusters, cluster(descriptors, method=method, **kwargs)
    )


def test_square_descriptors_are_deduplicated_too():
    """As many rows as bits (e.g. 2048 fingerprints of 2048 bits) is still a feature
    matrix: "random" labels each row it is given, so copies only share a label when
    the square input is deduplicated as well."""
    base = np.random.default_rng(0).integers(0, 2, (16, 48))
    descriptors = _shuffled_copies(base, n_copies=3, seed=0)
    assert descriptors.shape == (48, 48)
    clusters = cluster(descriptors, method="random", n_clusters=5, seed=0)
    assert _n_split_duplicates(descriptors, clusters) == 0


def test_bitbirch_identical_fingerprints_share_cluster():
    """Plain bblean.BitBirch on these rows gives 9 of 60 distinct fingerprints more
    than one label: BIRCH is order-dependent, so after node splits a later copy can be
    routed to a different leaf. Sparse noisy copies of 10 centres, each row 3x, shuffled.
    """
    rng = np.random.default_rng(0)
    centres = rng.random((10, 256)) < 0.1
    base = centres[rng.integers(10, size=60)] ^ (rng.random((60, 256)) < 0.03)
    descriptors = np.tile(base, (3, 1))[rng.permutation(180)]
    clusters = cluster(
        descriptors, method="bitbirch", threshold=0.5, branching_factor=5
    )
    assert _n_split_duplicates(descriptors, clusters) == 0
