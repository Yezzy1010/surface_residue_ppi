"""Inspect three already-audited train/val records; never open the test split.

This diagnostic preserves the original audit and does not create training labels.
Run with PYTHONPATH=src .venv-audit/bin/python scripts/inspect_atom3d_audit_cases.py.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import time

from surface_residue_ppi.ppi_audit import KEY_COLUMNS, ReleasedLMDB, heavy_contacts


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/atom3d_audit/20260924/local_20260924_workstation_01"
FULL_KEY = KEY_COLUMNS + ["hetero", "insertion_code", "resname"]
CANONICAL = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
CASES = [
    ("val", 12442, "3mg7.pdb1.gz_1_L_3mg7.pdb1.gz_1_V"),
    ("train", 1560, "2ad7.pdb1.gz_1_A_2ad7.pdb1.gz_1_B"),
    ("train", 75937, "4wba.pdb1.gz_1_D_4wba.pdb1.gz_1_E"),
]


def category(row):
    if row["hetero"].strip() == "W":
        return "water"
    if not row["hetero"].strip() and row["resname"].strip() in CANONICAL:
        return "canonical_amino_acid_nonhetero"
    return "other_hetero_or_noncanonical"


def inspect(item):
    atoms, labels = item["atoms_pairs"], item["atoms_neighbors"]
    sides, protein_sides = [], []
    for subunit, side in atoms.groupby("subunit", sort=False):
        residues = side[FULL_KEY].drop_duplicates().copy()
        residues["category"] = [category(row) for row in residues.to_dict("records")]
        alpha = side[side["name"].str.strip().eq("CA") & side["element"].str.strip().eq("C")]
        ca_keys = set(alpha[FULL_KEY].itertuples(index=False, name=None))
        residues["has_alpha_carbon"] = [key in ca_keys for key in residues[FULL_KEY].itertuples(index=False, name=None)]
        collisions = []
        duplicate = residues[residues.duplicated(KEY_COLUMNS, keep=False)]
        for key, group in duplicate.groupby(KEY_COLUMNS, sort=False, dropna=False):
            touching = set()
            for suffix in ("0", "1"):
                columns = [column + suffix for column in KEY_COLUMNS]
                for index, label_key in enumerate(labels[columns].itertuples(index=False, name=None)):
                    if label_key == key:
                        touching.add(index)
            scalar_key = [value.item() if hasattr(value, "item") else value for value in key]
            collisions.append(dict(upstream_key=dict(zip(KEY_COLUMNS, scalar_key)),
                                   variants=group[["hetero", "insertion_code", "resname", "has_alpha_carbon"]].to_dict("records"),
                                   released_positive_rows_touching_key=len(touching)))
        categories = {}
        for name, group in residues.groupby("category", sort=False):
            categories[name] = dict(residues=len(group), without_alpha_carbon=int((~group["has_alpha_carbon"]).sum()),
                                    resname_counts={str(k): int(v) for k, v in group["resname"].value_counts().items()})
        sides.append(dict(subunit=str(subunit), full_identity_residues=len(residues), categories=categories,
                          collisions=collisions))
        protein_sides.append(side[side["hetero"].str.strip().eq("") & side["resname"].str.strip().isin(CANONICAL)])
    return dict(id=item["id"], sides=sides, released_all_atom_positive_rows=len(labels),
                diagnostic_canonical_nonhetero_positive_upstream_keys=len(heavy_contacts(*protein_sides)),
                diagnostic_limit="Canonical nonhetero filter is descriptive only; contacts still use collapsed upstream keys. Not training labels or a final cleaning policy.")


def main():
    if ROOT != Path("/home/zy/projects/Surface-residue") or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("This follow-up is authorized only on the Linux workstation")
    output = RUN / "followup_cases.json"
    if output.exists():
        raise FileExistsError("Preserve the existing follow-up; use a separate run for changes")
    original = RUN / "report.json"
    report = dict(execution="local-workstation", host=socket.gethostname(), started_at=time.time(),
                  purpose="Explain previously observed insertion-code collisions and nonprotein no-CA counts",
                  original_report_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  test_records_read=0, training_enabled=False, records=[])
    for split, index, expected_id in CASES:
        db = ReleasedLMDB(ROOT / "data/atom3d/zenodo-4911102/lmdb/DIPS-split/data" / split)
        try:
            item = db[index]
            if item["id"] != expected_id:
                raise ValueError("Record ID mismatch")
            report["records"].append(dict(split=split, lmdb_index=index, **inspect(item)))
        finally:
            db.close()
    report["finished_at"] = time.time()
    report["wall_seconds"] = report["finished_at"] - report["started_at"]
    report["disposition"] = "Original audit remains needs_review. Preserve insertion codes in full residue identities; define protein eligibility and reconstruct labels before training. No source record was altered."
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    with output.open("x") as handle:
        handle.write(serialized)
    for record in report["records"]:
        print(record["id"], "released", record["released_all_atom_positive_rows"],
              "canonical-only diagnostic", record["diagnostic_canonical_nonhetero_positive_upstream_keys"])
        for side in record["sides"]:
            print(side["subunit"], {name: (row["residues"], row["without_alpha_carbon"]) for name, row in side["categories"].items()},
                  "collision_groups", len(side["collisions"]))
    print(output)


if __name__ == "__main__":
    main()
