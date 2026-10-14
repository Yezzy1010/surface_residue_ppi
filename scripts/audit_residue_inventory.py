"""Inventory residue identities in ATOM3D DIPS train/val without rewriting data.

This is a read-only preprocessing audit. It deliberately scans only train and
validation records. It does not read test coordinates/labels, create targets,
choose a cleaning policy, or start model code.
"""
import argparse
from collections import Counter, deque
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import socket
import time
import traceback
import re
import shutil
import signal
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


CANONICAL_RESNAMES = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)
KEY_COLUMNS = ("ensemble", "subunit", "structure", "model", "chain", "residue")
FULL_COLUMNS = KEY_COLUMNS + ("hetero", "insertion_code", "resname")
SPLITS = ("train", "val")
MAX_EXAMPLES = 200


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("Not JSON serializable: %r" % (type(value),))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default) + "\n")
    temporary.replace(path)


def source_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def category(row):
    hetero = str(row["hetero"]).strip()
    resname = str(row["resname"]).strip().upper()
    if hetero == "W":
        return "water"
    if not hetero and resname in CANONICAL_RESNAMES:
        return "canonical_amino_acid_nonhetero"
    if not hetero:
        return "noncanonical_amino_acid_or_unknown_nonhetero"
    return "other_hetero_or_ligand"


def _record_error(split, index, item_id, error):
    return dict(split=split, lmdb_index=int(index), id=str(item_id), error=error)


def inventory_record(item, split, index):
    """Return descriptive counts for one decoded LMDB record.

    Full residue identities include hetero flag, insertion code and residue
    name. Upstream six-field keys are retained only to expose collisions.
    """
    import pandas as pd

    atoms = item["atoms_pairs"]
    labels = item["atoms_neighbors"]
    if not isinstance(atoms, pd.DataFrame) or not isinstance(labels, pd.DataFrame):
        raise ValueError("Expected atoms_pairs and atoms_neighbors DataFrames")
    required = set(FULL_COLUMNS) | {"element", "name", "x", "y", "z"}
    missing = sorted(required - set(atoms.columns))
    if missing:
        raise ValueError("Missing atom fields: %s" % missing)
    if "label" not in labels.columns:
        raise ValueError("Missing labels label column")

    record = dict(split=split, lmdb_index=int(index), id=str(item["id"]),
                  subunits=[], residues={}, collisions=[], label_rows=len(labels),
                  duplicate_atom_identity=0, alternate_atoms=0, deuterium_atoms=0)
    label_columns = set(labels.columns)
    if not all((column + suffix) in label_columns for suffix in ("0", "1") for column in KEY_COLUMNS):
        raise ValueError("Missing label residue key columns")
    label_counts = []
    for suffix in ("0", "1"):
        columns = [column + suffix for column in KEY_COLUMNS]
        label_counts.append(Counter(labels[columns].itertuples(index=False, name=None)))

    # One pass through atom rows avoids thousands of tiny pandas groups per record.
    atom_columns = list(FULL_COLUMNS) + ["element", "name"]
    has_altloc = "altloc" in atoms.columns
    if has_altloc:
        atom_columns.append("altloc")
    sides = {}
    for values in atoms[atom_columns].itertuples(index=False, name=None):
        full_key = values[:len(FULL_COLUMNS)]
        upstream_key = full_key[:len(KEY_COLUMNS)]
        subunit = full_key[1]
        side = sides.setdefault(subunit, dict(atoms=0, full={}, upstream={}, ca=set(),
                                             atom_ids=set(), duplicates=0, alternates=0, deuterium=0))
        side["atoms"] += 1
        if full_key not in side["full"]:
            side["full"][full_key] = None
            side["upstream"].setdefault(upstream_key, []).append(full_key)
        element, name = values[len(FULL_COLUMNS):len(FULL_COLUMNS) + 2]
        if str(name).strip() == "CA" and str(element).strip().upper() == "C":
            side["ca"].add(full_key)
        atom_id = upstream_key + (full_key[6], full_key[7], name)
        if has_altloc:
            altloc = values[-1]
            atom_id += (altloc,)
            side["alternates"] += int(not pd.isna(altloc) and str(altloc).strip() != "")
        side["duplicates"] += int(atom_id in side["atom_ids"])
        side["atom_ids"].add(atom_id)
        side["deuterium"] += int(str(element).strip().upper() == "D")
    for subunit, side in sides.items():
        collision_groups = collision_rows = touched_rows = 0
        collision_examples = []
        for key, variants in side["upstream"].items():
            if len(variants) <= 1:
                continue
            collision_groups += 1
            collision_rows += len(variants)
            touched = sum(counter.get(key, 0) for counter in label_counts)
            touched_rows += touched
            if len(collision_examples) < MAX_EXAMPLES:
                collision_examples.append(dict(
                    upstream_key=dict(zip(KEY_COLUMNS, key)),
                    variants=[dict(zip(FULL_COLUMNS[6:], variant[6:])) for variant in variants],
                    released_positive_rows_touching_key=int(touched)))
        counts = {}
        for key in side["full"]:
            name = category(dict(zip(FULL_COLUMNS, key)))
            entry = counts.setdefault(name, dict(residues=0, without_carbon_ca=0))
            entry["residues"] += 1
            entry["without_carbon_ca"] += int(key not in side["ca"])
        record["subunits"].append(dict(subunit=str(subunit), atoms=side["atoms"],
            full_identity_residues=len(side["full"]), category_counts=counts,
            upstream_key_residues=len(side["upstream"]),
            upstream_key_collision_groups=collision_groups,
            full_identity_rows_in_collision_groups=collision_rows,
            released_positive_rows_touching_collision_keys=touched_rows,
            collisions=collision_examples, duplicate_atom_identity=side["duplicates"],
            alternate_atoms=side["alternates"], deuterium_atoms=side["deuterium"]))
        record["collisions"].extend(collision_examples)
        record["duplicate_atom_identity"] += side["duplicates"]
        record["alternate_atoms"] += side["alternates"]
        record["deuterium_atoms"] += side["deuterium"]
    if len(record["subunits"]) != 2:
        raise ValueError("Expected two subunits, got %d" % len(record["subunits"]))
    return record


def add_record_to_summary(summary, record):
    summary["records"] += 1
    summary["label_rows"] += record["label_rows"]
    summary["atoms"] += sum(side["atoms"] for side in record["subunits"])
    summary["full_identity_residues"] += sum(side["full_identity_residues"] for side in record["subunits"])
    summary["upstream_key_residues"] += sum(side["upstream_key_residues"] for side in record["subunits"])
    summary["collision_groups"] += sum(side["upstream_key_collision_groups"] for side in record["subunits"])
    summary["full_identity_rows_in_collision_groups"] += sum(side["full_identity_rows_in_collision_groups"] for side in record["subunits"])
    summary["released_positive_rows_touching_collision_keys"] += sum(side["released_positive_rows_touching_collision_keys"] for side in record["subunits"])
    summary["duplicate_atom_identity"] += record["duplicate_atom_identity"]
    summary["alternate_atoms"] += record["alternate_atoms"]
    summary["deuterium_atoms"] += record["deuterium_atoms"]
    if record["collisions"]:
        summary["records_with_collisions"] += 1
    for side in record["subunits"]:
        for name, values in side["category_counts"].items():
            entry = summary["categories"].setdefault(name, dict(residues=0, without_carbon_ca=0))
            entry["residues"] += values["residues"]
            entry["without_carbon_ca"] += values["without_carbon_ca"]
    for side in record["subunits"]:
        summary["top_no_ca"].append(dict(split=record["split"], lmdb_index=record["lmdb_index"],
                                         id=record["id"], subunit=side["subunit"],
                                         no_ca=sum(v["without_carbon_ca"] for v in side["category_counts"].values()),
                                         categories=side["category_counts"]))
    summary["top_no_ca"] = sorted(summary["top_no_ca"], key=lambda row: (-row["no_ca"], row["id"], row["subunit"]))[:MAX_EXAMPLES]
    summary["collision_examples"].extend(record["collisions"][:MAX_EXAMPLES])
    summary["collision_examples"] = summary["collision_examples"][:MAX_EXAMPLES]


def empty_summary():
    return dict(records=0, label_rows=0, atoms=0, full_identity_residues=0,
                upstream_key_residues=0, collision_groups=0,
                full_identity_rows_in_collision_groups=0,
                released_positive_rows_touching_collision_keys=0,
                records_with_collisions=0, duplicate_atom_identity=0,
                alternate_atoms=0, deuterium_atoms=0, categories={},
                top_no_ca=[], collision_examples=[])


EXPECTED_COUNTS = {"train": 87303, "val": 31050}
_WORKER_DATABASES = {}
_STOP_REQUESTED = False


def worker_chunk(task):
    """A worker reads its own LMDB handle; no decoded structures cross processes."""
    from surface_residue_ppi.ppi_audit import ReleasedLMDB
    root, split, start, stop = task
    if split not in SPLITS:
        raise ValueError("Only train/val record access is allowed")
    if split not in _WORKER_DATABASES:
        _WORKER_DATABASES[split] = ReleasedLMDB(
            Path(root) / "data/atom3d/zenodo-4911102/lmdb/DIPS-split/data" / split)
    db = _WORKER_DATABASES[split]
    if db.count != EXPECTED_COUNTS[split]:
        raise ValueError("Unexpected split count")
    summary = empty_summary()
    failures, issues = [], []
    started = time.perf_counter()
    for index in range(start, stop):
        item_id = "unknown"
        try:
            item = db[index]
            item_id = item["id"]
            if db.id_to_idx[item_id] != index:
                raise ValueError("LMDB ID/index mismatch")
            record = inventory_record(item, split, index)
            add_record_to_summary(summary, record)
            categories = [side["category_counts"] for side in record["subunits"]]
            if (record["collisions"] or record["duplicate_atom_identity"] or
                    record["alternate_atoms"] or record["deuterium_atoms"] or
                    any(c.get("canonical_amino_acid_nonhetero", {}).get("without_carbon_ca", 0)
                        or c.get("noncanonical_amino_acid_or_unknown_nonhetero", {}).get("residues", 0)
                        for c in categories)):
                issues.append(record)
        except Exception:
            failures.append(_record_error(split, index, item_id, traceback.format_exc()))
    return dict(split=split, start=start, stop=stop, summary=summary, failures=failures,
                issues=issues, worker_seconds=time.perf_counter()-started,
                worker_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def merge_summary(target, addition):
    for key in empty_summary():
        if key == "categories":
            for category_name, values in addition[key].items():
                dest = target[key].setdefault(category_name, dict(residues=0, without_carbon_ca=0))
                for field, value in values.items():
                    dest[field] += value
        elif key == "top_no_ca":
            target[key] = sorted(target[key]+addition[key],
                                key=lambda row: (-row["no_ca"], row["id"], row["subunit"]))[:MAX_EXAMPLES]
        elif key == "collision_examples":
            target[key] = (target[key]+addition[key])[:MAX_EXAMPLES]
        else:
            target[key] += addition[key]


def validate_resume(prior):
    """Return committed per-split prefix lengths; reject ambiguous old failures."""
    processed = prior.get("split_processed")
    if processed is None:
        if prior.get("failure_count", 0):
            raise ValueError("Old report with failures has no reliable split cursors")
        processed = {split: prior.get("split_summaries", {}).get(split, {}).get("records", 0)
                     for split in SPLITS}
    if set(processed)-set(SPLITS):
        raise ValueError("Unexpected split in resume report")
    result = {split: int(processed.get(split, 0)) for split in SPLITS}
    if any(not 0 <= result[split] <= EXPECTED_COUNTS[split] for split in SPLITS):
        raise ValueError("Invalid resume prefix")
    if sum(result.values()) != prior["processed_records"]:
        raise ValueError("Inconsistent resume total")
    successful = sum(s["records"] for s in prior["split_summaries"].values())
    if successful + prior["failure_count"] != prior["processed_records"]:
        raise ValueError("Inconsistent success/failure counters")
    if prior.get("test_records_read") != 0:
        raise ValueError("Unexpected test access in prior inventory")
    return result


def request_stop(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def ordered_chunks(pool, tasks, max_pending):
    """Bound pending work so stopping needs to drain at most one small window."""
    iterator = iter(tasks)
    pending = deque()
    try:
        for _ in range(max_pending):
            task = next(iterator, None)
            if task is None:
                break
            pending.append(pool.submit(worker_chunk, task))
        while pending:
            yield pending.popleft().result()
            if _STOP_REQUESTED:
                break
            task = next(iterator, None)
            if task is not None:
                pending.append(pool.submit(worker_chunk, task))
    finally:
        for future in pending:
            future.cancel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--execution", choices=("slurm-hpc", "local-workstation"), default="slurm-hpc")
    parser.add_argument("--run-id")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.execution == "slurm-hpc":
        job = os.environ.get("SLURM_JOB_ID")
        if not job or os.environ.get("SLURM_JOB_PARTITION") != "hpc" or args.run_id:
            raise RuntimeError("Run the inventory through sbatch on hpc")
        if args.workers > int(os.environ.get("SLURM_CPUS_PER_TASK", "1")):
            raise ValueError("Worker count exceeds Slurm CPU allocation")
    else:
        if root != Path("/home/zy/projects/Surface-residue") or os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Local inventory authorization applies only to the Linux workstation")
        job = args.run_id or ("local_" + time.strftime("%Y%m%d_%H%M%S") + "_" + str(os.getpid()))
        if not re.fullmatch(r"local_[A-Za-z0-9_-]+", job):
            raise ValueError("Local run IDs must start with local_ and contain safe characters")
    if not 1 <= args.workers <= 16 or args.chunk_size < 1:
        raise ValueError("Use 1-16 workers and a positive chunk size")
    from importlib.metadata import version
    report = dict(status="running", run_id=job, job_id=os.environ.get("SLURM_JOB_ID"),
                  execution=args.execution, host=socket.gethostname(), python=platform.python_version(),
                  packages={name:version(name) for name in ("numpy","pandas","scipy","lmdb")},
                  started_at=time.time(), test_records_read=0, training_enabled=False,
                  split_summaries={s:empty_summary() for s in SPLITS},
                  split_processed={s:0 for s in SPLITS}, expected_counts=EXPECTED_COUNTS,
                  failures=[], failure_count=0, processed_records=0, workers=args.workers,
                  chunk_size=args.chunk_size, worker_max_rss_kib=0,
                  notes=["Residue/atom counts are per complex occurrence, not unique proteins.",
                         "Collision label-row counts count endpoint/key incidences; both ends may be counted.",
                         "This inventory does not reconstruct contacts, validate full sequences or enable training."])
    if args.resume_from:
        prior = json.loads(args.resume_from.read_text())
        report["split_processed"] = validate_resume(prior)
        for split in SPLITS:
            if split in prior["split_summaries"]:
                report["split_summaries"][split] = prior["split_summaries"][split]
        report["failure_count"] = prior["failure_count"]
        report["failures"] = prior.get("failures", [])[:MAX_EXAMPLES]
        report["processed_records"] = prior["processed_records"]
        report["resume"] = dict(report=str(args.resume_from), sha256=source_digest(args.resume_from),
                              inherited_records=prior["processed_records"],
                              inherited_wall_seconds=prior.get("wall_seconds"),
                              prior_status=prior["status"],
                              reason="Continue saved prefix after interrupting slow single-process implementation")
    report["detailed_issue_log_start_indices"] = dict(report["split_processed"])
    report["notes"].append("Detailed issue/failure JSONL covers newly scanned ranges only; inherited prefix retains its original bounded examples.")
    run = root / "runs/atom3d_residue_inventory/20260924" / job
    run.mkdir(parents=True, exist_ok=False)
    source_names = ("scripts/audit_residue_inventory.py", "tests/test_residue_inventory.py",
                    "src/surface_residue_ppi/ppi_audit.py", "slurm/audit_residue_inventory.sbatch")
    report["source_sha256"] = {}
    for name in source_names:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
        report["source_sha256"][name] = source_digest(target)
    write_json(run / "report.json", report)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    tasks = [(str(root), split, start, min(start + args.chunk_size, EXPECTED_COUNTS[split]))
             for split in SPLITS
             for start in range(report["split_processed"][split], EXPECTED_COUNTS[split], args.chunk_size)]
    stage_start = time.perf_counter()
    initial_count = report["processed_records"]
    print("START", job, "inherited_records", initial_count, "remaining", sum(EXPECTED_COUNTS.values())-initial_count,
          "workers", args.workers, flush=True)
    try:
        with (run / "issues.jsonl").open("x") as issues_file, (run / "failures.jsonl").open("x") as failures_file:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                for result in ordered_chunks(pool, tasks, args.workers * 2):
                    split = result["split"]
                    if report["split_processed"][split] != result["start"]:
                        raise ValueError("Non-contiguous completion; refusing to advance cursor")
                    merge_summary(report["split_summaries"][split], result["summary"])
                    report["failure_count"] += len(result["failures"])
                    report["failures"] = (report["failures"]+result["failures"])[:MAX_EXAMPLES]
                    for row in result["issues"]:
                        issues_file.write(json.dumps(row, default=json_default)+"\n")
                    for row in result["failures"]:
                        failures_file.write(json.dumps(row, default=json_default)+"\n")
                    issues_file.flush()
                    failures_file.flush()
                    report["split_processed"][split] = result["stop"]
                    report["processed_records"] = sum(report["split_processed"].values())
                    report["worker_max_rss_kib"] = max(report["worker_max_rss_kib"], result["worker_max_rss_kib"])
                    elapsed = time.perf_counter()-stage_start
                    speed = (report["processed_records"]-initial_count)/max(elapsed, 1e-9)
                    progress = dict(processed_records=report["processed_records"], total_records=sum(EXPECTED_COUNTS.values()),
                        split=split, split_cursor=result["stop"], elapsed_seconds=elapsed,
                        records_per_second=speed, estimated_remaining_seconds=(sum(EXPECTED_COUNTS.values())-report["processed_records"])/speed)
                    write_json(run / "progress.json", progress)
                    write_json(run / "report.json", report)
                    if (report["processed_records"]-initial_count) % (args.chunk_size*8) == 0:
                        print("PROGRESS", json.dumps(progress), flush=True)
                    if _STOP_REQUESTED:
                        report["status"] = "interrupted_resumable"
                        break
        if report["status"] != "interrupted_resumable":
            if report["split_processed"] != EXPECTED_COUNTS:
                raise ValueError("Incomplete coverage")
            report["status"] = "needs_review" if report["failure_count"] else "inventory_complete_needs_policy"
    except Exception:
        report["fatal_error"] = traceback.format_exc()
        report["status"] = "failed"
    finally:
        report["finished_at"] = time.time()
        report["wall_seconds"] = report["finished_at"]-report["started_at"]
        report["parent_max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(run / "report.json", report)
    print("REPORT", run / "report.json", report["status"], flush=True)
    if report["status"] in ("failed", "interrupted_resumable"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
