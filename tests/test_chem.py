import numpy as np
import pytest

from nanoom.chem import scaffold_clustering


class TestScaffoldClustering:
    def test_groups_by_scaffold(self):
        labels = scaffold_clustering(
            ["c1ccccc1C", "c1ccccc1CC(=O)N", "C1CCCCC1C", "CCO", "CCN"]
        )
        assert labels[0] == labels[1]  # shared benzene scaffold
        assert labels[0] != labels[2]  # cyclohexane differs from benzene
        assert labels[3] == labels[4]  # acyclic: shared empty scaffold
        assert labels[3] != labels[0]

    def test_deterministic_and_order_independent(self):
        smiles = ["c1ccccc1C", "C1CCCCC1C", "CCO", "c1ccncc1N"]
        labels = scaffold_clustering(smiles)
        assert labels.dtype == np.int64
        assert (labels >= 0).all()
        assert np.array_equal(labels, scaffold_clustering(smiles))
        assert np.array_equal(labels[::-1], scaffold_clustering(smiles[::-1]))

    def test_generic_merges_atom_types(self):
        smiles = ["c1ccccc1C", "C1CCCCC1C"]
        plain = scaffold_clustering(smiles)
        generic = scaffold_clustering(smiles, generic=True)
        assert plain[0] != plain[1]
        assert generic[0] == generic[1]

    def test_invalid_smiles_raises_with_position(self):
        with pytest.raises(ValueError, match="position 1"):
            scaffold_clustering(["CCO", "not_smiles"])
