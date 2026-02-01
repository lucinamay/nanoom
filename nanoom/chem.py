"""rdkit clustering"""

import logging as lg
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import bblean
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import ExplicitBitVect
from rdkit.SimDivFilters import rdSimDivPickers
from rich.progress import track


def _auto_n_clusters(X: Sequence) -> int:
    return len(X) // 10 + 1


def _smiles_to_murcko_scaffolds(smiles: Sequence[str]) -> list:
    """from logic of https://github.com/sohviluukkonen/gbmt-splits/blob/main/gbmtsplits/clustering.py"""
    return [MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smi)) for smi in smiles]


def _dissimilarity_cluster_assignment(fps: Sequence, centroids: Sequence) -> np.ndarray:
    """from logic of https://github.com/sohviluukkonen/gbmt-splits/blob/main/gbmtsplits/clustering.py"""
    clusters = np.empty(len(fps), dtype=int)
    for i, fp in enumerate(fps):
        similarities = [
            DataStructs.FingerprintSimilarity(fp, fps[j]) for j in centroids
        ]
        clusters[i] = np.argmax(similarities)
    return clusters


def murcko_scaffold_clustering(X: Sequence) -> np.ndarray:
    """from logic of https://github.com/sohviluukkonen/gbmt-splits/blob/main/gbmtsplits/clustering.py"""
    scaffolds = _smiles_to_murcko_scaffolds(X)
    scaffold_to_cluster = dict(zip(*enumerate(set(scaffolds))))
    return np.array([scaffold_to_cluster[scaf] for scaf in scaffolds])


def maxmin_clustering(
    X: Sequence, fps: Sequence, n_clusters: int | None = None, seed: int = 0
) -> np.ndarray:
    """from logic of https://github.com/sohviluukkonen/gbmt-splits/blob/main/gbmtsplits/clustering.py"""
    n_clusters = n_clusters if n_clusters is not None else _auto_n_clusters(X)
    centroids = rdSimDivPickers.MaxMinPicker().LazyBitVectorPick(
        fps, len(fps), n_clusters, seed=seed
    )
    return _dissimilarity_cluster_assignment(fps, centroids)


def leader_picker_clustering(
    X: Sequence, fps: Sequence, similarity_threshold: float = 0.7
) -> np.ndarray:
    """from logic of https://github.com/sohviluukkonen/gbmt-splits/blob/main/gbmtsplits/clustering.py"""
    centroids = rdSimDivPickers.LeaderPicker().LazyBitVectorPick(
        fps, len(fps), similarity_threshold
    )
    return _dissimilarity_cluster_assignment(fps, centroids)


def _sphere_exclusion_directed(
    descriptors: np.ndarray,
    threshold: float = 0.65,
):
    """Based on https://doi.org/10.1021/ci025554v"""
    # @TODO check if useful
    raise NotImplementedError


# ==================== own implementations ====================


def _ndarray_to_bitvects(array: np.ndarray) -> list[DataStructs.ExplicitBitVect]:
    lg.debug(f"Converting array of shape {array.shape} to BitVect")
    bitvect_list = []
    for row in array:
        bv = ExplicitBitVect(len(row))
        set_bits = np.flatnonzero(row)
        for bit_idx in set_bits:
            bv.SetBit(int(bit_idx))
        bitvect_list.append(bv)
    return bitvect_list


def _sphere_exclusion(
    descriptors: np.ndarray | list[DataStructs.ExplicitBitVect],
    threshold: float = 0.65,
    save_cluster_leaders_to: Path | None = None,
) -> np.ndarray:
    """
    threshold == minimum distance between cluster centroids
    Takes a list of ExplicitBitVects and returns a dictionary of cluster_ids:list[indices].
    Comment: based on https://greglandrum.github.io/rdkit-blog/posts/2020-11-18-sphere-exclusion-clustering.html
    """
    lg.info(f"Starting sphere exclusion clustering of {len(descriptors)} datapoints")
    if isinstance(descriptors, np.ndarray):
        descriptors = _ndarray_to_bitvects(descriptors)
    # lg.debug(f"picking leaders with threshold {threshold}")
    threshold = threshold
    picks = rdSimDivPickers.LeaderPicker().LazyBitVectorPick(
        descriptors,
        len(descriptors),
        threshold,  # <- minimum distance between cluster centroids
    )
    # for later visualisation on tmaps, umaps, tsnes, save leader indices
    if save_cluster_leaders_to is not None:
        np.savetxt(str(save_cluster_leaders_to), np.array(picks))
        lg.debug(f"saved clustering to {save_cluster_leaders_to}")

    clusters: dict[int, list] = defaultdict(list)
    for i, idx in enumerate(picks):
        clusters[i].append(idx)
    sims = np.zeros((len(picks), len(descriptors)))

    lg.info(
        f"Computing similarities for {len(picks)} cluster leaders against {len(descriptors)} datapoints"
    )
    for i in track(
        range(len(picks)), description="computing similarities for sphere exclusion"
    ):
        pick = picks[i]
        sims[i, :] = DataStructs.BulkTanimotoSimilarity(descriptors[pick], descriptors)
        sims[i, i] = 0
    lg.debug("Assigning datapoints to nearest cluster leaders")
    best = np.argmax(sims, axis=0)
    for i, idx in enumerate(best):
        if i not in picks:
            clusters[idx].append(i)
    # return clusters
    cluster_arr = np.empty(len(descriptors), dtype=int)
    for cluster_num, idxs in clusters.items():
        cluster_arr[idxs] = cluster_num
    return cluster_arr


def _bitbirch(descriptors: np.ndarray, **kwargs) -> np.ndarray:
    """from https://github.com/mqcomplab/bblean/blob/main/examples/dataset_splitting.ipynb"""
    # bitbirch = bblean.BitBirch(branching_factor=50, threshold=0.65)
    bitbirch = bblean.BitBirch(**kwargs)
    bitbirch.fit(descriptors)
    cluster_list = bitbirch.get_cluster_mol_ids()

    # Map each mol ID to its cluster ID
    n_molecules = len(descriptors)
    cluster_labels = [0] * n_molecules
    for cluster_id, indices in enumerate(cluster_list):
        for idx in indices:
            cluster_labels[idx] = cluster_id
    return np.array(cluster_labels, dtype=int)
