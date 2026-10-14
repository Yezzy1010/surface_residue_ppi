import unittest

from scripts.audit_residue_chemistry import classify


class ResidueChemistryTest(unittest.TestCase):
    def test_water_is_not_amino_acid_candidate(self):
        self.assertEqual(classify("W", "HOH"), ("water", None))

    def test_canonical_nonhetero_is_separate(self):
        self.assertEqual(classify(" ", "ALA"), ("canonical_nonhetero", "A"))

    def test_mse_hetero_is_only_a_candidate(self):
        category, letter = classify("H_MSE", "MSE")
        self.assertEqual(category, "hetero_extended_mapping_candidate")
        self.assertEqual(letter, "M")

    def test_unknown_nonhetero_remains_unmapped(self):
        self.assertEqual(classify(" ", "ZZZ"), ("nonhetero_unmapped", None))


if __name__ == "__main__":
    unittest.main()
