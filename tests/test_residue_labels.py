import unittest

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from surface_residue_ppi.residue_labels import canonical_monomer, protein_contacts


def atoms(chain, rows):
    return pd.DataFrame([dict(ensemble="pair", subunit=chain, structure="1abc", model=1,
        chain=chain, hetero=hetero, residue=number, insertion_code=icode, resname=resname,
        element=element, name=name, altloc=" ", x=float(x), y=0., z=0.)
        for number, icode, resname, hetero, name, element, x in rows])


class ResidueLabelsTest(unittest.TestCase):
    def test_inserted_residue_does_not_inherit_merged_contact(self):
        left = canonical_monomer(atoms("A", [(144," ","ALA"," ","CA","C",0),
                                              (144,"A","GLY"," ","CA","C",20)]))
        right = canonical_monomer(atoms("B", [(1," ","ALA"," ","CA","C",1)]))
        pairs, labels = protein_contacts(left, right)
        self.assertEqual(pairs, {(0,0)})
        np.testing.assert_array_equal(labels[0], [True,False])
        self.assertEqual(left["residue_keys"][1][-1], "A")

    def test_missing_ca_masks_features_without_erasing_partner_contact(self):
        left = canonical_monomer(atoms("A", [(1," ","ALA"," ","CB","C",0)]))
        right = canonical_monomer(atoms("B", [(1," ","GLY"," ","CA","C",1)]))
        pairs, labels = protein_contacts(left,right)
        self.assertFalse(left["ca_valid"][0])
        self.assertTrue(labels[1][0])
        self.assertEqual(pairs,{(0,0)})

    def test_water_calcium_and_partner_fields_do_not_enter_monomer(self):
        frame = atoms("A", [(1," ","ALA"," ","CA","C",0),
                            (2," ","HOH","W","O","O",1),
                            (3," "," CA","H_CA","CA","CA",2)])
        left = canonical_monomer(frame)
        frame["ensemble"] = "different_partner"
        frame["subunit"] = "different_pair_role"
        frame["label"] = 1
        changed = canonical_monomer(frame)
        self.assertEqual(left["monomer_input_sha256"],changed["monomer_input_sha256"])
        self.assertEqual(len(left["residue_keys"]),1)
        self.assertEqual(left["excluded_atoms"],2)

    def test_sparse_contacts_match_dense_and_strict_boundary(self):
        left = canonical_monomer(atoms("A", [(1," ","ALA"," ","CA","C",0),
                                              (2," ","GLY"," ","CA","C",20)]))
        right = canonical_monomer(atoms("B", [(1," ","ALA"," ","CA","C",6),
                                               (2," ","GLY"," ","CA","C",21)]))
        expected = set(zip(*np.where(cdist(left["xyz"],right["xyz"]) < 6.0)))
        self.assertEqual(protein_contacts(left,right)[0],expected)
        self.assertEqual(expected,{(1,1)})

    def test_duplicate_atom_site_fails_instead_of_mixing_conformers(self):
        frame = atoms("A", [(1," ","ALA"," ","CA","C",0),
                            (1," ","ALA"," ","CA","C",1)])
        frame.loc[1,"altloc"] = "A"
        with self.assertRaisesRegex(ValueError,"Ambiguous"):
            canonical_monomer(frame)


if __name__ == "__main__":
    unittest.main()
