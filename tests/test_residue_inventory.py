import unittest

import pandas as pd

from scripts.audit_residue_inventory import (category, inventory_record, empty_summary,
                                             add_record_to_summary, merge_summary, validate_resume)


def atoms(side, rows):
    return pd.DataFrame([dict(ensemble="1abc_A_B", subunit=side, structure="1abc.pdb",
                              model=0, chain=side, residue=residue, hetero=hetero,
                              insertion_code=insertion, resname=resname, element=element,
                              name=name, altloc=" ", x=float(index), y=0.0, z=0.0)
                         for index, (residue, hetero, insertion, resname, element, name) in enumerate(rows)])


def labels():
    columns = [column + str(suffix) for suffix in (0, 1)
               for column in ("ensemble", "subunit", "structure", "model", "chain", "residue")] + ["label"]
    left = ("1abc_A_B", "A", "1abc.pdb", 0, "A", 1)
    right = ("1abc_A_B", "B", "1abc.pdb", 0, "B", 1)
    return pd.DataFrame([list(left) + list(right) + [1]], columns=columns)


class ResidueInventoryTest(unittest.TestCase):
    def test_category_does_not_treat_calcium_as_protein_ca(self):
        self.assertEqual(category(dict(hetero="H_CA", resname=" CA")), "other_hetero_or_ligand")
        self.assertEqual(category(dict(hetero=" ", resname="ALA")), "canonical_amino_acid_nonhetero")
        self.assertEqual(category(dict(hetero="W", resname="HOH")), "water")

    def test_full_identity_preserves_insertion_code_and_reports_collision(self):
        left = atoms("A", [(1, " ", " ", "ALA", "C", "CA"),
                           (1, " ", "A", "ASN", "C", "CA")])
        right = atoms("B", [(1, " ", " ", "GLY", "C", "CA")])
        item = dict(id="1abc_A_B", atoms_pairs=pd.concat([left, right], ignore_index=True),
                    atoms_neighbors=labels())
        result = inventory_record(item, "train", 0)
        side = result["subunits"][0]
        self.assertEqual(side["full_identity_residues"], 2)
        self.assertEqual(side["upstream_key_collision_groups"], 1)
        self.assertEqual(side["full_identity_rows_in_collision_groups"], 2)

    def test_water_and_nonprotein_ca_are_separated(self):
        left = atoms("A", [(1, "W", " ", "HOH", "O", "O"),
                           (2, "H_CA", " ", " CA", "CA", "CA"),
                           (3, " ", " ", "ALA", "C", "CA")])
        right = atoms("B", [(1, " ", " ", "GLY", "C", "CA")])
        item = dict(id="1abc_A_B", atoms_pairs=pd.concat([left, right], ignore_index=True),
                    atoms_neighbors=labels())
        categories = inventory_record(item, "train", 0)["subunits"][0]["category_counts"]
        self.assertEqual(categories["water"]["without_carbon_ca"], 1)
        self.assertEqual(categories["other_hetero_or_ligand"]["without_carbon_ca"], 1)
        self.assertEqual(categories["canonical_amino_acid_nonhetero"]["without_carbon_ca"], 0)

    def test_chunk_merge_matches_serial_aggregation(self):
        left = atoms("A", [(1, " ", " ", "ALA", "C", "CA"),
                           (1, " ", "A", "ASN", "C", "CA")])
        right = atoms("B", [(1, " ", " ", "GLY", "C", "CA")])
        serial, merged = empty_summary(), empty_summary()
        for index in range(4):
            item = dict(id="1abc_A_B_%s" % index, atoms_pairs=pd.concat([left, right]),
                        atoms_neighbors=labels())
            record = inventory_record(item, "train", index)
            add_record_to_summary(serial, record)
            chunk = empty_summary()
            add_record_to_summary(chunk, record)
            merge_summary(merged, chunk)
        self.assertEqual(serial, merged)

    def test_resume_rejects_inconsistent_or_ambiguous_prefix(self):
        summary = empty_summary()
        summary["records"] = 5149
        previous = dict(split_summaries={"train": summary}, failure_count=0,
                        processed_records=5149, test_records_read=0)
        self.assertEqual(validate_resume(previous), {"train":5149, "val":0})
        previous["processed_records"] = 5150
        with self.assertRaises(ValueError):
            validate_resume(previous)
        previous["failure_count"] = 1
        with self.assertRaises(ValueError):
            validate_resume(previous)


if __name__ == "__main__":
    unittest.main()
