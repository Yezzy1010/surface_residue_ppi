"""Audit the selected Phase 2 task-one candidates on the workstation.

This reads train/val records only. It reconstructs released heavy-atom
contacts, separates residue identity from the partner coordinates, and writes
a structural common-mask proposal for each side. It never reads the test
LMDB records and does not produce model features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import pandas as pd

from surface_residue_ppi.ppi_audit import KEY_COLUMNS, ReleasedLMDB, audit_record


CANONICAL = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    """Convert pandas/numpy scalar values emitted by LMDB frames."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, set):
        return sorted(value, key=repr)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def norm(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def residue_key(row, suffix=""):
    return tuple(row[f"{column}{suffix}"] for column in KEY_COLUMNS)


def side_mask(side: pd.DataFrame, positive_keys: set[tuple]):
    grouped = side.groupby(KEY_COLUMNS, dropna=False, sort=False)
    rows = []
    invalid_identity = 0
    for key, group in grouped:
        names = {norm(value).upper() for value in group["resname"]}
        hetero = {norm(value).upper() for value in group["hetero"]}
        insertion = {norm(value) for value in group["insertion_code"]}
        altloc = {norm(value) for value in group["altloc"] if norm(value)}
        name = next(iter(names)) if len(names) == 1 else ""
        is_canonical = len(names) == 1 and name in CANONICAL and hetero <= {""}
        ca = group[group["name"].map(norm).str.upper() == "CA"]
        has_single_ca = len(ca) == 1
        finite = bool(group[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce").notna().all().all())
        identity_ok = len(names) == len(hetero) == len(insertion) == 1
        if not identity_ok:
            invalid_identity += 1
        valid = bool(is_canonical and has_single_ca and not altloc and finite and identity_ok)
        rows.append(dict(key=list(key), residue=int(key[-1]), insertion_code=next(iter(insertion), ""),
                         resname=name, positive=key in positive_keys, valid=valid,
                         atoms=len(group), ca_atoms=len(ca), altloc_count=len(altloc),
                         identity_ok=identity_ok, finite=finite))
    valid_keys = {tuple(row["key"]) for row in rows if row["valid"]}
    positive_valid = sum(row["valid"] and row["positive"] for row in rows)
    return dict(rows=rows, residue_count=len(rows), valid_count=len(valid_keys),
                positive_count=sum(row["positive"] for row in rows),
                positive_valid_count=positive_valid, invalid_identity=invalid_identity,
                valid_keys=[list(key) for key in sorted(valid_keys, key=repr)])


def audit_one(item):
    atoms = item["atoms_pairs"]
    labels = item["atoms_neighbors"]
    audit = audit_record(item)
    subunits = list(atoms["subunit"].drop_duplicates())
    side_key_sets = [set(tuple(values) for values in atoms[atoms["subunit"] == subunit][KEY_COLUMNS].itertuples(index=False, name=None))
                     for subunit in subunits]
    positives = {0: set(), 1: set()}
    for row in labels.itertuples(index=False):
        values = row._asdict()
        if int(values["label"]) != 1:
            raise ValueError("non-positive label in released contact table")
        key0 = tuple(values[f"{column}0"] for column in KEY_COLUMNS)
        key1 = tuple(values[f"{column}1"] for column in KEY_COLUMNS)
        if key0 in side_key_sets[0] and key1 in side_key_sets[1]:
            positives[0].add(key0)
            positives[1].add(key1)
        elif key1 in side_key_sets[0] and key0 in side_key_sets[1]:
            positives[0].add(key1)
            positives[1].add(key0)
        else:
            raise ValueError("released contact references an absent side residue")
    sides = []
    for side_index, subunit in enumerate(subunits):
        side = atoms[atoms["subunit"] == subunit].copy()
        sides.append(side_mask(side, positives[side_index]))
    blocking = {"released_contacts_differ_from_heavy_lt_6A", "upstream_residue_key_collision",
                "duplicate_atom_identity", "subunit_not_single_chain_model"}
    issues = set(audit["issues"])
    ready = not (issues & blocking) and all(side["valid_count"] > 0 for side in sides)
    return dict(id=item["id"], split=item["split"], lmdb_index=item["lmdb_index"],
                pdb=item["pdb"], cluster_pair=item["cluster_pair"], audit=audit,
                sides=sides, structural_common_mask_definition=(
                    "canonical non-hetero residue, exactly one CA, no altloc, finite coordinates, "
                    "unambiguous insertion/name/hetero identity"), ready_for_feature_extraction=ready,
                blocking_issues=sorted(issues & blocking))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if root != Path("/home/zy/projects/Surface-residue") or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Candidate audit is authorized only on the workstation")
    if not args.run_id.startswith("local_"):
        raise ValueError("Run ID must begin with local_")
    candidate_report = root / args.candidate_report
    report_input = json.loads(candidate_report.read_text())
    records = []
    for split in ("train", "val"):
        records.extend(report_input["selection"][split])
    # The strict val list is a view of val, not an additional decoding pass.
    unique = {(row["split"], row["lmdb_index"]): row for row in records}
    run = root / "runs/phase2_candidate_audit/20260925" / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    started = time.time()
    dbs = {split: ReleasedLMDB(root / "data/atom3d/zenodo-4911102/lmdb/DIPS-split/data" / split)
           for split in ("train", "val")}
    results = []
    failures = []
    try:
        for number, row in enumerate(unique.values(), 1):
            try:
                item = dbs[row["split"]][row["lmdb_index"]]
                if item["id"] != row["id"]:
                    raise ValueError("candidate ID/index mismatch")
                results.append(audit_one(row | {"atoms_pairs": item["atoms_pairs"],
                                                "atoms_neighbors": item["atoms_neighbors"]}))
            except Exception as exc:
                failures.append(dict(**row, error=repr(exc)))
            if number % 100 == 0:
                print("AUDITED", number, flush=True)
    finally:
        for db in dbs.values():
            db.close()
    (run / "records.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, default=json_default) for row in results) + "\n")
    ready = [dict(split=row["split"], id=row["id"], lmdb_index=row["lmdb_index"], pdb=row["pdb"],
                  cluster_pair=row["cluster_pair"]) for row in results if row["ready_for_feature_extraction"]]
    (run / "pilot_ready_manifest.json").write_text(json.dumps(ready, indent=2, ensure_ascii=False, default=json_default) + "\n")
    aggregate = {}
    for row in results:
        aggregate.setdefault(row["split"], dict(records=0, ready=0, blocking_issues={}, valid_residues=[], positive_valid=[]))
        value = aggregate[row["split"]]
        value["records"] += 1
        value["ready"] += int(row["ready_for_feature_extraction"])
        value["valid_residues"].extend(side["valid_count"] for side in row["sides"])
        value["positive_valid"].extend(side["positive_valid_count"] for side in row["sides"])
        for issue in row["blocking_issues"]:
            value["blocking_issues"][issue] = value["blocking_issues"].get(issue, 0) + 1
    report = dict(status="phase2_candidate_audit_complete" if not failures else "needs_review",
                  execution="local-workstation", run_id=args.run_id,
                  candidate_report=str(args.candidate_report), candidate_report_sha256=sha256(candidate_report),
                  started_at=started, finished_at=time.time(), wall_seconds=time.time() - started,
                  test_records_read=0, training_enabled=False, requested_records=len(unique),
                  audited_records=len(results), failures=failures, aggregate=aggregate,
                  structural_mask_policy=("canonical non-hetero residues with exactly one CA, no altloc, "
                                          "finite coordinates, and unambiguous insertion/name/hetero identity; "
                                          "labels are assigned only by residue keys from released contacts"),
                  max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (run / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=json_default) + "\n")
    print("REPORT", run / "report.json", report["status"], len(results), len(ready), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
