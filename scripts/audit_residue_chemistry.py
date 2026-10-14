"""Inventory modified/noncanonical residue chemistry in DIPS train/val.

This is descriptive evidence for choosing a cleaning policy. It never rewrites
the released LMDB, never reads test coordinates/labels, and never promotes a
PDB residue-name dictionary entry to a final biological mapping automatically.
"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import shutil
import socket
import time
import traceback

from Bio.Data.PDBData import protein_letters_3to1_extended


CANONICAL = frozenset("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
KEY_COLUMNS = ("ensemble", "subunit", "structure", "model", "chain", "residue")
MAX_EXAMPLES = 8
EXPECTED_COUNTS = {"train": 87303, "val": 31050}
SPLITS = ("train", "val")
_DBS = {}


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized_mapping():
    return {str(name).strip().upper(): letter for name, letter in protein_letters_3to1_extended.items()}


EXTENDED = normalized_mapping()


def classify(hetero, resname):
    hetero = str(hetero).strip().upper()
    resname = str(resname).strip().upper()
    mapped = EXTENDED.get(resname)
    if hetero == "W" or resname in {"HOH", "WAT", "DOD"}:
        return "water", mapped
    if not hetero and resname in CANONICAL:
        return "canonical_nonhetero", mapped
    if not hetero and mapped:
        return "nonhetero_extended_mapping_candidate", mapped
    if not hetero:
        return "nonhetero_unmapped", None
    if mapped and resname not in {"HOH", "WAT", "DOD"}:
        return "hetero_extended_mapping_candidate", mapped
    return "other_hetero_or_ligand", None


def blank_result(split, start, stop):
    return dict(split=split, start=start, stop=stop, records=0, failed=[],
                classes=Counter(), names=Counter(), examples=defaultdict(list),
                residue_instances=0, atom_rows=0, records_by_class=Counter(),
                ca_missing_by_class=Counter(), element_signatures=Counter(),
                variant_groups=0, variant_examples=[])


def worker(task):
    from surface_residue_ppi.ppi_audit import ReleasedLMDB
    root, split, start, stop = task
    if split not in SPLITS:
        raise ValueError("Test split is forbidden")
    if split not in _DBS:
        _DBS[split] = ReleasedLMDB(Path(root) / "data/atom3d/zenodo-4911102/lmdb/DIPS-split/data" / split)
    db = _DBS[split]
    if db.count != EXPECTED_COUNTS[split]:
        raise ValueError("Unexpected %s count: %s" % (split, db.count))
    result = blank_result(split, start, stop)
    for index in range(start, stop):
        item_id = "unknown"
        try:
            item = db[index]
            item_id = item["id"]
            if db.id_to_idx[item_id] != index:
                raise ValueError("ID/index mismatch")
            atoms = item["atoms_pairs"]
            result["records"] += 1
            result["records_by_class"]["records_total"] += 1
            grouped = {}
            variants = defaultdict(set)
            for row in atoms.itertuples(index=False):
                values = row._asdict()
                hetero = str(values["hetero"]).strip()
                resname = str(values["resname"]).strip().upper()
                full_key = tuple(values[column] for column in KEY_COLUMNS) + (hetero, str(values["insertion_code"]).strip())
                variant = (hetero, resname)
                entry = grouped.setdefault(full_key, dict(hetero=hetero, resname=resname,
                    category=None, mapped=None, atoms=0, has_ca=False, elements=set()))
                entry["atoms"] += 1
                entry["has_ca"] |= str(values["name"]).strip().upper() == "CA" and str(values["element"]).strip().upper() == "C"
                entry["elements"].add(str(values["element"]).strip().upper())
                variants[full_key].add(variant)
            record_classes = set()
            for key, entry in grouped.items():
                category, mapped = classify(entry["hetero"], entry["resname"])
                entry["category"], entry["mapped"] = category, mapped
                name_key = (entry["hetero"], entry["resname"], category, mapped or "")
                result["residue_instances"] += 1
                result["atom_rows"] += entry["atoms"]
                result["classes"][category] += 1
                result["names"][name_key] += 1
                result["ca_missing_by_class"][category] += int(not entry["has_ca"])
                result["element_signatures"][(category, ",".join(sorted(entry["elements"])), entry["resname"])] += 1
                record_classes.add(category)
                if len(result["examples"][name_key]) < MAX_EXAMPLES:
                    result["examples"][name_key].append(dict(split=split, lmdb_index=index,
                        id=item_id, file_path=item.get("file_path"),
                        residue_key=list(key), atoms=entry["atoms"], has_ca=entry["has_ca"]))
            for category in record_classes:
                result["records_by_class"][category] += 1
            for key, variant_set in variants.items():
                if len(variant_set) > 1:
                    result["variant_groups"] += 1
                    if len(result["variant_examples"]) < MAX_EXAMPLES:
                        result["variant_examples"].append(dict(split=split, lmdb_index=index, id=item_id,
                            residue_key=list(key), variants=sorted([list(x) for x in variant_set])))
        except Exception:
            result["failed"].append(dict(split=split, lmdb_index=index, id=item_id, error=traceback.format_exc()))
    result["classes"] = dict(result["classes"])
    result["names"] = {"\x1f".join(key): value for key, value in result["names"].items()}
    result["examples"] = {"\x1f".join(key): value for key, value in result["examples"].items()}
    result["records_by_class"] = dict(result["records_by_class"])
    result["ca_missing_by_class"] = dict(result["ca_missing_by_class"])
    result["element_signatures"] = {"\x1f".join(key): value for key, value in result["element_signatures"].items()}
    return result


def merge(target, result):
    target["records"] += result["records"]
    target["residue_instances"] += result["residue_instances"]
    target["atom_rows"] += result["atom_rows"]
    target["variant_groups"] += result["variant_groups"]
    target["failed"].extend(result["failed"])
    for field in ("classes", "names", "records_by_class", "ca_missing_by_class", "element_signatures"):
        for key, value in result[field].items():
            target[field][key] += value
    for key, values in result["examples"].items():
        target["examples"].setdefault(key, [])
        if len(target["examples"][key]) < MAX_EXAMPLES:
            target["examples"][key].extend(values[:MAX_EXAMPLES-len(target["examples"][key])])
    target["variant_examples"].extend(result["variant_examples"][:max(0, MAX_EXAMPLES-len(target["variant_examples"]))])


def serializable_summary(summary):
    return dict(summary, classes=dict(sorted(summary["classes"].items())),
        names=dict(sorted(summary["names"].items(), key=lambda item: (-item[1], item[0]))),
        records_by_class=dict(sorted(summary["records_by_class"].items())),
        ca_missing_by_class=dict(sorted(summary["ca_missing_by_class"].items())),
        element_signatures=dict(sorted(summary["element_signatures"].items(), key=lambda item: (-item[1], item[0]))),
        examples=dict(sorted(summary["examples"].items())))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if root != Path("/home/zy/projects/Surface-residue") or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("This local chemistry audit is authorized only on the Linux workstation")
    if not args.run_id.startswith("local_") or not 1 <= args.workers <= 16:
        raise ValueError("Use a local_ run ID and 1-16 workers")
    run = root / "runs/atom3d_residue_chemistry/20260925" / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    source_names = ("scripts/audit_residue_chemistry.py", "src/surface_residue_ppi/ppi_audit.py")
    hashes = {}
    for name in source_names:
        destination = run / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination)
        hashes[name] = digest(destination)
    report = dict(status="running", execution="local-workstation", run_id=args.run_id,
        job_id=None, host=socket.gethostname(), python=platform.python_version(),
        packages={}, started_at=time.time(), test_records_read=0, training_enabled=False,
        source_sha256=hashes, candidate_mapping_source="BioPython protein_letters_3to1_extended",
        candidate_mapping_scope="Descriptive candidates only; no automatic final residue policy",
        split_summaries={split:blank_result(split, 0, EXPECTED_COUNTS[split]) for split in SPLITS},
        processed_records=0, failures=[], workers=args.workers, chunk_size=args.chunk_size)
    from importlib.metadata import version
    for package in ("numpy", "pandas", "scipy", "lmdb"):
        report["packages"][package] = version(package)
    write_json(run / "report.json", report)
    tasks = [(str(root), split, start, min(start+args.chunk_size, EXPECTED_COUNTS[split]))
             for split in SPLITS for start in range(0, EXPECTED_COUNTS[split], args.chunk_size)]
    started = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for result in pool.map(worker, tasks):
                split = result["split"]
                merge(report["split_summaries"][split], result)
                report["processed_records"] += result["records"] + len(result["failed"])
                report["failures"].extend(result["failed"][:MAX_EXAMPLES-len(report["failures"])])
                write_json(run / "progress.json", dict(processed_records=report["processed_records"],
                    total_records=sum(EXPECTED_COUNTS.values()), split=split,
                    elapsed_seconds=time.perf_counter()-started,
                    records_per_second=report["processed_records"]/max(time.perf_counter()-started,1e-9)))
                write_json(run / "report.json", report)
    except Exception:
        report["status"] = "failed"
        report["fatal_error"] = traceback.format_exc()
    else:
        report["status"] = "chemistry_inventory_complete_needs_policy" if not report["failures"] else "needs_review"
    report["finished_at"] = time.time()
    report["wall_seconds"] = report["finished_at"] - report["started_at"]
    report["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for split in SPLITS:
        report["split_summaries"][split] = serializable_summary(report["split_summaries"][split])
    write_json(run / "report.json", report)
    print("REPORT", run / "report.json", report["status"], flush=True)
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
