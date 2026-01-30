"""rdkit clustering"""

from typing import Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters import rdSimDivPickers


def _auto_n_clusters(X: Sequence) -> int:
    return len(X) // 10 + 1


def _smiles_to_murcko_scaffolds(smiles: Sequence[str]) -> list:
    return [MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smi)) for smi in smiles]


def _dissimilarity_cluster_assignment(fps: Sequence, centroids: Sequence) -> np.ndarray:
    clusters = np.empty(len(fps), dtype=int)
    for i, fp in enumerate(fps):
        similarities = [
            DataStructs.FingerprintSimilarity(fp, fps[j]) for j in centroids
        ]
        clusters[i] = np.argmax(similarities)
    return clusters


def random_clustering(
    X: Sequence, n_clusters: int | None = None, seed: int = 0
) -> np.ndarray:
    n_clusters = n_clusters if n_clusters is not None else _auto_n_clusters(X)
    return np.ndarray(np.random.RandomState(seed=seed).permutation(len(X))) % n_clusters


def murcko_scaffold_clustering(X: Sequence) -> np.ndarray:
    scaffolds = _smiles_to_murcko_scaffolds(X)
    scaffold_to_cluster = dict(zip(*enumerate(set(scaffolds))))
    return np.array([scaffold_to_cluster[scaf] for scaf in scaffolds])


def maxmin_clustering(
    X: Sequence, fps: Sequence, n_clusters: int | None = None, seed: int = 0
) -> np.ndarray:
    n_clusters = n_clusters if n_clusters is not None else _auto_n_clusters(X)
    centroids = rdSimDivPickers.MaxMinPicker().LazyBitVectorPick(
        fps, len(fps), n_clusters, seed=seed
    )
    return _dissimilarity_cluster_assignment(fps, centroids)


def leader_picker_clustering(
    X: Sequence, fps: Sequence, similarity_threshold: float = 0.7
) -> np.ndarray:
    centroids = rdSimDivPickers.LeaderPicker().LazyBitVectorPick(
        fps, len(fps), similarity_threshold
    )
    return _dissimilarity_cluster_assignment(fps, centroids)
