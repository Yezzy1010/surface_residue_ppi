"""Assemble a corrected Phase 2 feature cache from a base run and a patch run.

The base run remains untouched as an audit record.  Files are hard-linked into
the final directory where possible, while patch files replace only the
affected cache entries after unlinking the hard link first.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base_dir = args.base
    patch_dir = args.patch
    out_dir = args.out
    project_root = patch_dir.parents[2]
    if out_dir.exists():
        raise FileExistsError(out_dir)
    base = json.loads((base_dir / "report.json").read_text())
    patch = json.loads((patch_dir / "report.json").read_text())
    manifest = json.loads(args.manifest.read_text())
    expected = {(row["split"], int(row["lmdb_index"])): row for row in manifest}
    base_records = {(row["split"], int(row["lmdb_index"])): row for row in base["records"]}
    patch_records = {(row["split"], int(row["lmdb_index"])): row for row in patch["records"]}
    if set(patch_records) - set(expected):
        raise ValueError("Patch contains records outside the frozen manifest")
    if set(base_records) - set(expected):
        raise ValueError("Base contains records outside the frozen manifest")
    if patch.get("failures"):
        raise ValueError("Patch run still has failures")
    combined = dict(base_records)
    combined.update(patch_records)
    if set(combined) != set(expected):
        missing = sorted(set(expected) - set(combined))
        raise ValueError(f"Combined feature records are incomplete: {missing[:10]}")

    # Hard-link the unchanged cache so the final assembly does not duplicate
    # the multi-gigabyte feature arrays.  The base report is unlinked below.
    shutil.copytree(base_dir, out_dir, copy_function=os.link)

    # Replace only patch records.  Unlinking first is required if the final
    # destination was copied as a hard link by a future caller.
    for key, row in patch_records.items():
        for side in row["sides"]:
            source = project_root / Path(side["cache"])
            # The cache path is relative to the project root in the report.
            source_npz = source
            source_labels = source.with_name(source.stem + "_labels.npy")
            for source_path in (source_npz, source_labels):
                destination = out_dir / source_path.relative_to(patch_dir)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    destination.unlink()
                shutil.copy2(source_path, destination)

    # Rebuild a report with one canonical cache root and explicit provenance.
    report = copy.deepcopy(base)
    report.update(
        status="passed",
        job_id=f"{base['job_id']}+{patch['job_id']}",
        execution="slurm-gpu-composite",
        records_requested=len(expected),
        records=[],
        failures=[],
        source_jobs=[base["job_id"], patch["job_id"]],
        base_run_status=base["status"],
        patch_run_status=patch["status"],
        corrected_record_keys=[list(key) for key in sorted(patch_records)],
        correction="Case-sensitive SEQRES keys; protocol chain first, released-ID chain fallback",
        finished_at=patch.get("finished_at"),
        wall_seconds=float(base.get("wall_seconds", 0)) + float(patch.get("wall_seconds", 0)),
    )
    for key in sorted(combined, key=lambda value: (value[0], value[1])):
        row = copy.deepcopy(combined[key])
        manifest_row = expected[key]
        for side_index, side in enumerate(row["sides"]):
            side["seqres_key"] = f"{manifest_row['pdb'].lower()}_{manifest_row['sides'][side_index]['chain']}"
            side["cache"] = f"runs/phase2_features/{out_dir.name}/{row['split']}/{Path(side['cache']).name}"
        report["records"].append(row)
    report["records_requested"] = len(report["records"])
    (out_dir / "report.json").unlink()
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "records": len(report["records"]),
                      "corrected": len(patch_records), "output": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
