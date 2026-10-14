"""Synthetic contact/schema checks; these are not biological validation."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import lmdb
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from surface_residue_ppi.ppi_audit import (KEY_COLUMNS, ReleasedLMDB,
                                         audit_record, heavy_contacts, residue_keys)


def atoms(side, xs):
    return pd.DataFrame([dict(ensemble="1abc_A_B", subunit=side, structure="1abc.pdb",
        model=0, chain=side, residue=i + 1, hetero=" ", insertion_code=" ",
        resname="ALA", element="C", name="CA", altloc=" ", x=float(x), y=0., z=0.)
        for i, x in enumerate(xs)])


def contact_frame(pairs):
    return pd.DataFrame([list(a) + list(b) + [1] for a, b in pairs],
                        columns=[k + str(i) for i in (0, 1) for k in KEY_COLUMNS] + ["label"])


class ContactAuditTest(unittest.TestCase):
    def test_exact_cutoff_and_hydrogen_exclusion(self):
        left, right = atoms("A", [0., 20.]), atoms("B", [5.999, 6., 0., 100.])
        right.loc[2, "element"] = "H"
        result = heavy_contacts(left, right)
        self.assertEqual(result, {(residue_keys(left)[0], residue_keys(right)[0])})

    def test_sparse_contacts_match_independent_dense_reference(self):
        rng = np.random.RandomState(27)
        left, right = atoms("A", range(37)), atoms("B", range(43))
        left.loc[:, ["x", "y", "z"]] = rng.normal(size=(37, 3)) * 8
        right.loc[:, ["x", "y", "z"]] = rng.normal(size=(43, 3)) * 8
        i, j = np.where(cdist(left[["x", "y", "z"]], right[["x", "y", "z"]]) < 6.)
        k0, k1 = residue_keys(left), residue_keys(right)
        self.assertEqual(heavy_contacts(left, right), {(k0[a], k1[b]) for a, b in zip(i, j)})

    def test_missing_contact_blocks_operational_negatives(self):
        left, right = atoms("A", [0., 20.]), atoms("B", [1., 21.])
        pairs = sorted(heavy_contacts(left, right))
        item = dict(id="1abc_A_B", atoms_pairs=pd.concat([left, right]),
                    atoms_neighbors=contact_frame(pairs[:1]))
        report = audit_record(item)
        self.assertEqual(report["missing_released_positives"], 1)
        self.assertIsNone(report["operational_negative_pairs"])
        item["atoms_neighbors"] = contact_frame([(b, a) for a, b in pairs])
        report = audit_record(item)
        self.assertFalse(report["issues"])
        self.assertEqual(report["operational_negative_pairs"], 2)

    def test_insertion_code_collision_is_reported(self):
        left, right = atoms("A", [0., 1.]), atoms("B", [2.])
        left.loc[1, "residue"] = 1
        left.loc[1, "insertion_code"] = "A"
        item = dict(id="1abc_A_B", atoms_pairs=pd.concat([left, right]),
                    atoms_neighbors=contact_frame(heavy_contacts(left, right)))
        self.assertIn("upstream_residue_key_collision", audit_record(item)["issues"])

    def test_release_json_dataframe_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "db"
            original = atoms("A", [0., 1.])
            item = dict(id="1abc_A_B", atoms_pairs=original.to_dict(orient="split"),
                        types=dict(atoms_pairs=str(pd.DataFrame), id=str(str)))
            env = lmdb.open(str(path), map_size=1024 * 1024)
            with env.begin(write=True) as txn:
                txn.put(b"num_examples", b"1")
                txn.put(b"serialization_format", b"json")
                txn.put(b"id_to_idx", json.dumps({item["id"]: 0}).encode())
                txn.put(b"0", gzip.compress(json.dumps(item).encode()))
            env.close()
            db = ReleasedLMDB(path)
            try:
                pd.testing.assert_frame_equal(original, db[0]["atoms_pairs"])
                with self.assertRaises(IndexError):
                    db[1]
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
