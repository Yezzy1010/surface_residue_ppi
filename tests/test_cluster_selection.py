import tempfile
import unittest
from pathlib import Path

from surface_residue_ppi.cluster_selection import read_clusters


class ClusterSelectionTests(unittest.TestCase):
    def test_every_member_maps_to_its_representative(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clusters.tsv"
            path.write_text("1ABC_A\t1ABC_A\n1ABC_A\t2DEF_b\n3GHI_A\t3GHI_A\n")
            self.assertEqual(read_clusters(path), {
                "1ABC_A": "1ABC_A", "2DEF_b": "1ABC_A", "3GHI_A": "3GHI_A"})

    def test_conflicting_membership_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clusters.tsv"
            path.write_text("A\tA\nB\tB\nA\tX\nB\tX\n")
            with self.assertRaises(ValueError):
                read_clusters(path)
