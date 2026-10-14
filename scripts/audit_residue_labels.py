"""Validate provisional full-identity protein contacts on the existing audit set."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import time
import traceback

from surface_residue_ppi.ppi_audit import ReleasedLMDB, heavy_contacts
from surface_residue_ppi.residue_labels import AA, canonical_monomer, protein_contacts


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    if root != Path("/home/zy/projects/Surface-residue") or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("This diagnostic is authorized on the workstation only")
    manifest_path = root / "runs/atom3d_audit/20260924/local_20260924_workstation_01/sample_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if any(row["split"] not in ("train", "val") for row in manifest["records"]):
        raise ValueError("Test records must remain sealed")
    run = root / "runs/atom3d_residue_labels/20260924/local_20260924_residue_labels_01"
    run.mkdir(parents=True, exist_ok=False)
    sources = ["scripts/audit_residue_labels.py", "src/surface_residue_ppi/residue_labels.py",
               "tests/test_residue_labels.py", "src/surface_residue_ppi/ppi_audit.py"]
    hashes = {}
    for name in sources:
        destination = run / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination)
        hashes[name] = digest(destination)
    report = dict(status="running", host=socket.gethostname(), execution="local-workstation",
                  started_at=time.time(), source_sha256=hashes, training_enabled=False,
                  manifest_sha256=digest(manifest_path), test_records_read=0,
                  scope="Diagnostic canonical nonhetero amino acids only; not finalized training labels",
                  sequence_mapping="Observed residue order only; full polymer sequence mapping still required",
                  records=[], failures=[])
    dbs = {split: ReleasedLMDB(root / "data/atom3d/zenodo-4911102/lmdb/DIPS-split/data" / split)
           for split in ("train", "val")}
    try:
        for row in manifest["records"]:
            try:
                item = dbs[row["split"]][row["lmdb_index"]]
                if item["id"] != row["id"]:
                    raise ValueError("ID mismatch")
                atoms = item["atoms_pairs"]
                sides = [part for _, part in atoms.groupby("subunit", sort=False)]
                if len(sides) != 2:
                    raise ValueError("Expected two bound subunits")
                monomers = [canonical_monomer(side) for side in sides]
                pairs, labels = protein_contacts(*monomers)
                canonical_frames = [s[s["hetero"].str.strip().eq("") & s["resname"].str.strip().isin(AA)] for s in sides]
                # Independent old implementation collapses insertions; project the new
                # full-identity contact pairs to the old key only to verify equivalence.
                reference = heavy_contacts(*canonical_frames)
                old_keys = []
                for side, monomer in zip(sides, monomers):
                    first = side.iloc[0]
                    old_keys.append([(first["ensemble"], first["subunit"], key[0], key[1], key[2], key[4])
                                     for key in monomer["residue_keys"]])
                projected = {(old_keys[0][i],old_keys[1][j]) for i,j in pairs}
                if projected != reference:
                    raise ValueError("Full-identity contact projection differs from canonical reference")
                consistency = []
                for side, monomer in zip(sides, monomers):
                    changed = side.copy()
                    changed["ensemble"] = "unrelated_partner"
                    changed["subunit"] = "different_role"
                    changed["label"] = 999
                    other = canonical_monomer(changed)
                    consistency.append(other["monomer_input_sha256"] == monomer["monomer_input_sha256"])
                if not all(consistency):
                    raise ValueError("Partner/label metadata influenced monomer inputs")
                value = dict(**row, full_identity_positive_pairs=len(pairs),
                    projected_positive_pairs=len(projected), released_all_atom_positive_rows=len(item["atoms_neighbors"]),
                    projected_reference_equal=True, partner_metadata_independent=True,
                    side_summaries=[dict(residues=len(m["residue_keys"]), without_ca=int((~m["ca_valid"]).sum()),
                        interface_residues=int(label.sum()), excluded_atoms=m["excluded_atoms"],
                        selected_altloc_atoms=m["selected_altloc_atoms"],
                        monomer_input_sha256=m["monomer_input_sha256"])
                        for m,label in zip(monomers,labels)])
                report["records"].append(value)
                # Diagnostic labels live separately from monomer-only input summaries.
                detail = dict(scope=report["scope"], **row, contacts=sorted(pairs),
                    residues=[dict(residue_keys=m["residue_keys"], residue_names=m["residue_names"],
                        ca_valid=m["ca_valid"].tolist(), interface_labels=label.tolist(),
                        sequence_scope=m["sequence_scope"]) for m,label in zip(monomers,labels)])
                target = run / "diagnostic_labels" / (row["split"]+"_"+str(row["lmdb_index"])+".json")
                target.parent.mkdir(exist_ok=True)
                target.write_text(json.dumps(detail,indent=2)+"\n")
            except Exception:
                report["failures"].append(dict(**row,error=traceback.format_exc()))
    finally:
        for db in dbs.values():
            db.close()
    report["status"] = "diagnostic_passed_training_not_enabled" if not report["failures"] else "needs_review"
    report["finished_at"] = time.time()
    report["wall_seconds"] = report["finished_at"]-report["started_at"]
    (run / "report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("REPORT",run / "report.json",report["status"],len(report["records"]),flush=True)
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
