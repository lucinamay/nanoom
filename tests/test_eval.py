"""Tests for nanoom.eval: the split-quality audit functions (leakage and
balance checks, inter/intra-split distance stats).
"""

from collections.abc import Callable

import numpy as np
import pytest

import nanoom.eval as ev


def test_check_no_group_overlap_passes_when_disjoint():
    group_by = np.array([1, 1, 2, 2, 3, 3])
    splits = np.array([0, 0, 1, 1, 2, 2])
    assert ev.check_no_group_overlap(group_by, splits) is None


def test_check_no_group_overlap_raises_on_leakage():
    group_by = np.array([1, 1, 2, 2])
    splits = np.array([0, 1, 1, 1])  # group 1 spans splits 0 and 1
    with pytest.raises(ValueError, match="multiple splits"):
        ev.check_no_group_overlap(group_by, splits)


def test_check_distribution_y_similar_passes_when_balanced():
    y = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    splits = np.array([0, 0, 0, 1, 1, 1])
    assert ev.check_distribution_y_similar(y, splits) is None


def test_check_distribution_y_similar_raises_when_skewed():
    y = np.array([1.0, 1.0, 1.0, 100.0, 100.0, 100.0])
    splits = np.array([0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="not similar"):
        ev.check_distribution_y_similar(y, splits)


def test_split_y_means_values():
    y = np.array([1.0, 3.0, 10.0, 20.0])
    splits = np.array([0, 0, 1, 1])
    means = ev.split_y_means(y, splits)
    assert np.allclose(means, [2.0, 15.0])


@pytest.mark.parametrize(
    "fn,args",
    [
        (ev.check_no_group_overlap, (np.arange(12), np.zeros(60, dtype=int))),
        (ev.check_distribution_y_similar, (np.arange(60.0), np.zeros(12, dtype=int))),
        (ev.split_y_means, (np.arange(60.0), np.zeros(12, dtype=int))),
        (ev.min_distances_splits, (np.zeros((60, 3)), np.zeros(12, dtype=int))),
    ],
)
def test_eval_raises_on_length_mismatch(
    fn: Callable[..., object], args: tuple[np.ndarray, ...]
):
    kwargs = {"metric": "euclidean"} if fn is ev.min_distances_splits else {}
    with pytest.raises(ValueError, match="must be per-row and aligned"):
        fn(*args, **kwargs)


def test_split_y_means_with_noncontiguous_split_labels():
    # labels are not guaranteed to be 0..n-1; means come back ordered by sorted label
    y = np.array([1.0, 3.0, 10.0, 20.0])
    splits = np.array([2, 2, 7, 7])
    assert np.allclose(ev.split_y_means(y, splits), [2.0, 15.0])


def test_min_distances_splits_euclidean():
    rng = np.random.default_rng(0)
    split_a = rng.normal(loc=0.0, scale=0.05, size=(10, 2))
    split_b = rng.normal(loc=10.0, scale=0.05, size=(10, 2))
    descriptors = np.concatenate([split_a, split_b])
    splits = np.array([0] * 10 + [1] * 10)
    results = ev.min_distances_splits(descriptors, splits, metric="euclidean")
    assert {r["split"] for r in results} == {0, 1}
    for r in results:
        # well-separated clusters: nearest neighbor across splits is much
        # farther than the typical nearest neighbor within the same split
        assert r["ext_distance_min"] > r["int_distance_mean"]


def test_min_distances_splits_jaccard():
    rng = np.random.default_rng(0)
    descriptors = rng.integers(0, 2, size=(20, 16)).astype(bool)
    splits = np.array([0] * 10 + [1] * 10)
    results = ev.min_distances_splits(descriptors, splits, metric="jaccard")
    assert {r["split"] for r in results} == {0, 1}
    for r in results:
        assert 0 <= r["ext_distance_min"] <= 1
        assert 0 <= r["int_distance_min"] <= 1
