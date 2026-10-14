"""Combine structural and SEQRES audits into a frozen Phase 2 mask protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--audit-records", type=Path, required=True)
    parser.add_argument("--sequence-report", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--min-mapping-fraction", type=float, default=0.95)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if root != Path("/home/zy/projects/Surface-residue") or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Protocol finalization is authorized only on the workstation")
    if not args.run_id.startswith("local_"):
        raise ValueError("Run ID must begin with local_")
    candidate_report = json.loads((root / args.candidate_report).read_text())
    audit_records = {}
    with (root / args.audit_records).open() as stream:
        for line in stream:
            row = json.loads(line)
            audit_records[(row["split"], row["lmdb_index"])] = row
    sequence_report = json.loads((root / args.sequence_report).read_text())
    run = root / "runs/phase2_protocol/20260925" / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    records = []
    mismatch_examples = []
    for seq_record in sequence_report["records"]:
        key = (seq_record["split"], seq_record["lmdb_index"])
        audit = audit_records[key]
        side_outputs = []
        fractions = []
        record_mismatch = 0
        for side_index, mapping in enumerate(seq_record["side_mappings"]):
            exact = {}
            for mapped in mapping["exact_rows"]:
                if mapped["match"] != "exact":
                    continue
                lookup = (int(mapped["residue"]), str(mapped["insertion_code"]),
                          str(mapped["resname"]).strip().upper(), str(mapped.get("hetero", "")).strip())
                exact[lookup] = mapped
            valid_rows = [row for row in audit["sides"][side_index]["rows"] if row["valid"]]
            common = []
            mapped_count = 0
            for row in valid_rows:
                lookup = (int(row["residue"]), str(row["insertion_code"]),
                          str(row["resname"]).strip().upper(), "")
                mapped = exact.get(lookup)
                if mapped is None:
                    continue
                mapped_count += 1
                if mapped["sequence_letter"] == "X":
                    continue
                expected = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
                            "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
                            "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
                            "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}[row["resname"]]
                if mapped["sequence_letter"] != expected:
                    record_mismatch += 1
                    mismatch_examples.append(dict(id=seq_record["id"], residue=row["residue"],
                                                  resname=row["resname"], sequence_letter=mapped["sequence_letter"],
                                                  expected=expected))
                    continue
                common.append(dict(residue_key=row["key"], seq_id=int(mapped["seq_id"]),
                                   sequence_letter=mapped["sequence_letter"], positive=bool(row["positive"])))
            fraction = mapped_count / len(valid_rows) if valid_rows else 0.0
            fractions.append(fraction)
            side_outputs.append(dict(chain=mapping["chain"], structural_valid_count=len(valid_rows),
                                     sequence_mapped_count=mapped_count, mapping_fraction=fraction,
                                     common_valid_count=len(common), common_positive_count=sum(x["positive"] for x in common),
                                     common_valid=common))
        ready = bool(audit["ready_for_feature_extraction"] and
                     min(fractions, default=0.0) >= args.min_mapping_fraction and record_mismatch == 0)
        records.append(dict(split=seq_record["split"], id=seq_record["id"], lmdb_index=seq_record["lmdb_index"],
                            pdb=seq_record["pdb"], cluster_pair=audit["cluster_pair"],
                            sequence_scope=seq_record["sequence_scope"], sides=side_outputs,
                            min_mapping_fraction=min(fractions, default=0.0),
                            ready_for_feature_extraction=ready))
    # A mismatch in an earlier record should not invalidate all subsequent
    # records; the per-record flag below is corrected from its own side data.
    for record in records:
        record["ready_for_feature_extraction"] = bool(
            record["ready_for_feature_extraction"] and
            all(side["common_valid_count"] > 0 for side in record["sides"])
        )
    ready = [record for record in records if record["ready_for_feature_extraction"]]
    train = [record for record in ready if record["split"] == "train"]
    val = [record for record in ready if record["split"] == "val"]
    train_clusters = {cluster for record in train for cluster in record["cluster_pair"]}
    strict_val = [record for record in val if not (set(record["cluster_pair"]) & train_clusters)]
    for name, values in (("pilot_manifest.json", ready), ("train_manifest.json", train),
                         ("val_manifest.json", val), ("strict_val_manifest.json", strict_val),
                         ("records.json", records)):
        (run / name).write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n")
    report = dict(
        status="phase2_protocol_frozen_for_feature_extraction",
        candidate_report_sha256=sha256(root / args.candidate_report),
        audit_records_sha256=sha256(root / args.audit_records),
        sequence_report_sha256=sha256(root / args.sequence_report),
        min_mapping_fraction=args.min_mapping_fraction,
        records_total=len(records),
        records_ready=len(ready),
        ready_by_split={"train": len(train), "val": len(val), "strict_val": len(strict_val)},
        sequence_mapping_mismatch_count=len(mismatch_examples),
        sequence_mapping_mismatch_examples=mismatch_examples[:20],
        structural_mask_policy=("canonical non-hetero residues with exactly one CA, no altloc, finite coordinates, "
                                "unambiguous insertion/name/hetero identity"),
        sequence_mask_policy=("structural-valid residue with an exact auth chain/number/insertion mapping to SEQRES "
                              "and a canonical one-letter identity; minimum per-side mapped fraction is "
                              f"{args.min_mapping_fraction:.2f}"),
        labels_policy="positive labels are assigned from released contact residue keys after common masking; partner coordinates stay outside encoder inputs",
        test_records_read=0, training_enabled=False,
    )
    (run / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("REPORT", run / "report.json", report["status"], len(train), len(val), len(strict_val), flush=True)


if __name__ == "__main__":
    main()
