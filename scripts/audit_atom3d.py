"""Verify DIPS on a Slurm CPU node or the explicitly authorized workstation.

The test split is restricted to LMDB metadata. Coordinates and contact labels
are sampled from train/val only. A passing sample audit is not a full-dataset
leakage or label-completeness certificate and does not enable training.
"""
import argparse
import hashlib
import itertools
import json
import os
import platform
from pathlib import Path, PurePosixPath
import re
import resource
import shutil
import socket
import tarfile
import time
import traceback


EXPECTED_BYTES = 13568121746
EXPECTED_MD5 = "b6095db528e4a2e1139fddc64b6965de"


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def unpack(archive, parent, run, job):
    """Reuse only a verified complete extraction; preserve failed staging dirs."""
    target = parent / "lmdb"
    marker = target / "extraction.json"
    if target.exists():
        if marker.is_file() and json.loads(marker.read_text())["archive_md5"] == EXPECTED_MD5:
            return target
        raise RuntimeError("Existing extraction has no matching completion marker: %s" % target)
    staging = parent / ("lmdb.partial." + job)
    staging.mkdir(exist_ok=False)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isdir() or member.isfile()):
                raise ValueError("Unsafe archive member: %s" % member.name)
        total = sum(m.size for m in members if m.isfile())
        if shutil.disk_usage(parent).free < total + 10 * 1024 ** 3:
            raise RuntimeError("Insufficient free space for extraction and 10 GiB reserve")
        write_json(run / "archive_members.json", [dict(name=m.name, size=m.size) for m in members])
        for member in members:
            destination = staging / member.name
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, destination.open("xb") as out:
                    shutil.copyfileobj(source, out, length=8 * 1024 * 1024)
                if destination.stat().st_size != member.size:
                    raise IOError("Incomplete member: %s" % member.name)
    write_json(staging / "extraction.json", dict(archive_md5=EXPECTED_MD5,
               job_id=job, uncompressed_bytes=total, completed_at=time.time()))
    staging.rename(target)
    return target


def pdb_codes(identifier):
    # ATOM3D structure names may be 1abc.pdb, pdb1abc.ent.gz, etc.
    return set(m.lower() for m in re.findall(
        r"(?i)(?<![a-z0-9])(?:pdb)?([0-9][a-z0-9]{3})(?=[._/\-]|$)", identifier))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-split", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--execution", choices=("slurm", "local-workstation"), default="slurm")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.execution == "slurm":
        job = os.environ.get("SLURM_JOB_ID")
        if not job or os.environ.get("SLURM_JOB_PARTITION") != "hpc":
            raise RuntimeError("Use sbatch with hpc, or explicitly select the authorized workstation")
        if args.run_id:
            raise ValueError("Slurm runs use the actual job ID")
    else:
        if root != Path("/home/zy/projects/Surface-residue") or os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Local audit authorization applies only to the Linux workstation")
        job = args.run_id or ("local_" + time.strftime("%Y%m%d_%H%M%S") + "_" + str(os.getpid()))
        if not re.fullmatch(r"local_[A-Za-z0-9_-]+", job):
            raise ValueError("Local run IDs must start with local_ and contain safe characters")
    if args.samples_per_split < 1:
        raise ValueError("At least one training/validation sample is required")
    parent = root / "data/atom3d/zenodo-4911102"
    run = root / "runs/atom3d_audit" / str(args.seed) / job
    run.mkdir(parents=True, exist_ok=False)
    from importlib.metadata import version
    report = dict(status="running", stage="checksum", run_id=job,
                  job_id=os.environ.get("SLURM_JOB_ID"), execution=args.execution,
                  host=socket.gethostname(), python=platform.python_version(),
                  packages={name: version(name) for name in ("numpy", "pandas", "scipy", "lmdb")},
                  started_at=time.time(), records=[], failures=[], seed=args.seed,
                  samples_per_split=args.samples_per_split, test_records_read=0,
                  training_enabled=False, source_sha256={
                      str(p.relative_to(root)): digest(p) for p in [Path(__file__).resolve(),
                          root / "src/surface_residue_ppi/ppi_audit.py",
                          root / "slurm/audit_atom3d.sbatch",
                          root / "tests/test_ppi_audit.py"]})
    write_json(run / "report.json", report)
    databases = {}
    stage_start = time.perf_counter()
    report["stage_seconds"] = {}
    def stage(name):
        nonlocal stage_start
        report["stage_seconds"][report["stage"]] = time.perf_counter() - stage_start
        report["stage"] = name
        stage_start = time.perf_counter()
        write_json(run / "report.json", report)
        print("STAGE", name, flush=True)
    try:
        archive = parent / "PPI-DIPS-split.tar.gz"
        if archive.stat().st_size != EXPECTED_BYTES:
            raise ValueError("Main archive size mismatch")
        if digest(archive, "md5") != EXPECTED_MD5:
            raise ValueError("Main archive MD5 mismatch")
        report["archive"] = dict(path=str(archive), bytes=EXPECTED_BYTES, md5=EXPECTED_MD5)
        stage("extraction")
        extracted = unpack(archive, parent, run, job)
        from surface_residue_ppi.ppi_audit import ReleasedLMDB, audit_record
        stage("metadata")
        for split in ("train", "val", "test"):
            candidates = sorted({p.parent for p in extracted.rglob("data.mdb")
                                 if p.parent.name == split})
            if len(candidates) != 1:
                raise ValueError("Expected one LMDB for %s; got %s" % (split, candidates))
            databases[split] = ReleasedLMDB(candidates[0])
        indices, metadata = {}, {}
        for split, db in databases.items():
            file = parent / "extracted/indices" / (split + "_indices.txt")
            values = [int(line) for line in file.read_text().splitlines() if line.strip()]
            indices[split] = set(values)
            metadata[split] = dict(path=str(db.path), count=db.count, serialization=db.format,
                index_records=len(values), index_sha256=digest(file),
                index_duplicates=len(values) - len(indices[split]),
                original_index_min=min(values), original_index_max=max(values),
                count_matches_indices=db.count == len(values))
            if len(values) != len(indices[split]) or db.count != len(values):
                report["failures"].append(dict(split=split, error="release/index count mismatch or duplicates"))
        report["split_metadata"] = metadata
        report["split_overlaps"] = {}
        for a, b in itertools.combinations(databases, 2):
            ids = set(databases[a].id_to_idx) & set(databases[b].id_to_idx)
            overlap = indices[a] & indices[b]
            report["split_overlaps"][a + ":" + b] = dict(shared_ids=len(ids), shared_indices=len(overlap))
            if ids or overlap:
                report["failures"].append(dict(split=a + ":" + b, error="split metadata overlap"))
        lists = {}
        for file in sorted((root / "third_party/dMaSIF/lists").glob("*.txt")):
            values = [line.strip() for line in file.read_text().splitlines() if line.strip()]
            lists[file.name] = dict(pdbs={v.split("_")[0].lower() for v in values},
                                    lines=len(values), sha256=digest(file))
        if not lists:
            raise ValueError("Missing pinned dMaSIF lists; cannot report overlap audit")
        report["dmasif_pdb_overlap"] = {}
        for split, db in databases.items():
            parsed = [pdb_codes(key) for key in db.id_to_idx]
            available = set().union(*parsed)
            report["dmasif_pdb_overlap"][split] = dict(
                records_with_unparsed_pdb=sum(not s for s in parsed), pdb_count=len(available),
                lists={name: dict(sha256=value["sha256"], list_records=value["lines"],
                    shared_pdb_count=len(value["pdbs"] & available),
                    records_with_shared_pdb=sum(bool(s & value["pdbs"]) for s in parsed))
                       for name, value in lists.items()})
        report["overlap_limits"] = (
            "Repository list PDB overlap only; checkpoint membership and sequence homology remain unverified. "
            "Test split metadata only, no test coordinate/label/performance access.")
        stage("train_val_contact_audit")
        selected = []
        for split in ("train", "val"):
            db = databases[split]
            # Select by identity hash before looking at coordinates or labels.
            ordered = sorted(db.id_to_idx, key=lambda key: hashlib.sha256(
                (str(args.seed) + ":" + split + ":" + key).encode()).hexdigest())
            # One record per PDB for this audit, without inventing homology clusters.
            used = set()
            for identity in ordered:
                group = pdb_codes(identity) or {identity}
                if group & used:
                    continue
                used.update(group)
                selected.append(dict(split=split, id=identity, lmdb_index=db.id_to_idx[identity]))
                if sum(row["split"] == split for row in selected) >= args.samples_per_split:
                    break
        write_json(run / "sample_manifest.json", dict(selection="identity hash, at most one pair per PDB",
            seed=args.seed, purpose="data audit only; not Phase 2 training selection", records=selected))
        for row in selected:
            try:
                item = databases[row["split"]][row["lmdb_index"]]
                if item["id"] != row["id"]:
                    raise ValueError("ID-to-index mismatch")
                value = audit_record(item)
                report["records"].append(dict(split=row["split"], lmdb_index=row["lmdb_index"], **value))
                if value["issues"]:
                    report["failures"].append(dict(**row, error=value["issues"]))
                print("AUDIT", row["split"], row["id"], value["issues"], flush=True)
            except Exception:
                report["failures"].append(dict(**row, error=traceback.format_exc()))
            write_json(run / "report.json", report)
        report["status"] = "sample_audit_passed" if not report["failures"] else "needs_review"
        report["remaining_before_training"] = ["Full sequence mapping and structural cleaning",
            "Homology-aware Phase 2 selection and checkpoint overlap audit",
            "Full contact checks on every selected complex; physically separate labels and monomers",
            "Common valid masks, missing-residue/abnormal-atom tests and fixed pilot protocol",
            "Explain README count and split-ratio differences without altering released splits"]
    except Exception:
        report["status"] = "failed"
        report["failures"].append(dict(stage=report["stage"], error=traceback.format_exc()))
    finally:
        for db in databases.values():
            db.close()
        report["finished_at"] = time.time()
        report["stage_seconds"][report["stage"]] = time.perf_counter() - stage_start
        report["wall_seconds"] = report["finished_at"] - report["started_at"]
        report["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(run / "report.json", report)
        print("REPORT", run / "report.json", report["status"], flush=True)
    if report["status"] != "sample_audit_passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
