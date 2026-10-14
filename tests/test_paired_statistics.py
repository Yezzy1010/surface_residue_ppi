import unittest

import numpy as np

from surface_residue_ppi.paired_statistics import component_ids, paired_cluster_bootstrap


class PairedStatisticsTests(unittest.TestCase):
    def test_transitive_shared_clusters_keep_both_chains_together(self):
        groups = component_ids([{'cluster_pair': x} for x in [('a','b'), ('c','d'), ('b','c'), ('e','f')]])
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[0], groups[2])
        self.assertNotEqual(groups[0], groups[3])

    def test_constant_paired_gain_and_undefined_chain(self):
        result = paired_cluster_bootstrap([.1,.2,np.nan,.3], [.2,.3,np.nan,.4], [0,0,1,1])
        for key in ('delta','low','high'):
            self.assertAlmostEqual(result[key], .1)
        self.assertEqual(result['undefined_chains'], 1)

    def test_chain_weighted_estimator_does_not_average_clusters(self):
        result = paired_cluster_bootstrap([0,0,0,0], [1,0,0,0], [0,1,1,1])
        self.assertEqual(result['delta'], .25)

    def test_mismatched_undefined_chains_are_rejected(self):
        with self.assertRaises(ValueError):
            paired_cluster_bootstrap([.1,np.nan], [.2,.3], [0,1])


if __name__ == '__main__':
    unittest.main()
