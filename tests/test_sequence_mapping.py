import tempfile
from pathlib import Path
import unittest

from scripts.audit_sequence_mapping import chain_code, one_letter, pdb_code


class SequenceMappingTest(unittest.TestCase):
    def test_pdb_and_chain_parsing(self):
        self.assertEqual(pdb_code("3mg7.pdb1.gz_1_L_3mg7.pdb1.gz_1_V"), "3MG7")
        self.assertEqual(chain_code("3mg7.pdb1.gz_1_L"), "L")

    def test_mapping_classes_remain_explicit(self):
        self.assertEqual(one_letter("ALA"), ("A", "canonical"))
        self.assertEqual(one_letter("MSE"), ("M", "extended_candidate"))
        self.assertEqual(one_letter("ASX"), ("X", "ambiguous_or_unknown"))
        self.assertEqual(one_letter("ZZZ"), ("X", "unmapped"))


if __name__ == "__main__":
    unittest.main()
