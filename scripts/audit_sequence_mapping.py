"""Map observed LMDB residues to RCSB mmCIF polymer sequences for audit samples.

This is a small, read-only mapping diagnostic. It downloads only the PDB
entries used by the existing 20-record train/val audit manifest. It does not
read the ATOM3D test split, alter LMDB records, or create training features.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import time
import traceback
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from Bio.Data.PDBData import protein_letters_3to1, protein_letters_3to1_extended
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from surface_residue_ppi.ppi_audit import ReleasedLMDB


CANONICAL = {key.upper(): value for key, value in protein_letters_3to1.items()}
EXTENDED = {key.strip().upper(): value for key, value in protein_letters_3to1_extended.items()}
BASE_URL = "https://files.rcsb.org/download/{}.cif"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pdb_code(identifier):
    matches = re.findall(r"(?i)(?<![a-z0-9])(?:pdb)?([0-9][a-z0-9]{3})(?=[._/\\-]|$)", identifier)
    if not matches:
        raise ValueError("Cannot find PDB code: %s" % identifier)
    return matches[0].upper()


def chain_code(subunit):
    value = str(subunit).rsplit("_", 1)[-1]
    if not value:
        raise ValueError("Cannot find chain in subunit: %s" % subunit)
    return value


def one_letter(name):
    name = str(name).strip().upper()
    if name in CANONICAL:
        return CANONICAL[name], "canonical"
    if name == "SEC":
        return "U", "selenocysteine_explicit"
    if name == "PYL":
        return "O", "pyrrolysine_explicit"
    if name in {"ASX", "GLX", "UNK"}:
        return "X", "ambiguous_or_unknown"
    if name in EXTENDED:
        return EXTENDED[name], "extended_candidate"
    return "X", "unmapped"


def download_one(code, directory, attempts=3):
    target = directory / (code.lower() + ".cif")
    if target.is_file() and target.stat().st_size > 0:
        return dict(code=code, path=str(target), bytes=target.stat().st_size,
                    sha256=sha256(target), downloaded=False)
    last = None
    for attempt in range(attempts):
        try:
            request = Request(BASE_URL.format(code), headers={"User-Agent": "SurfaceResidue-PPI-audit/2026"})
            with urlopen(request, timeout=30) as response:
                payload = response.read()
            temporary = target.with_suffix(".cif.partial")
            temporary.write_bytes(payload)
            temporary.replace(target)
            return dict(code=code, path=str(target), bytes=len(payload), sha256=sha256(target), downloaded=True)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last = repr(exc)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("Download failed for %s: %s" % (code, last))


def mmcif_rows(path):
    data = MMCIF2Dict(str(path))
    prefix = "_pdbx_poly_seq_scheme."
    required = ["asym_id", "auth_seq_num", "pdb_ins_code", "pdb_strand_id", "auth_mon_id", "pdb_mon_id", "seq_id", "hetero"]
    if any(prefix + key not in data for key in required):
        raise ValueError("Missing _pdbx_poly_seq_scheme fields in %s" % path)
    length = len(data[prefix + "seq_id"])
    rows = []
    for index in range(length):
        row = {key: str(data[prefix + key][index]) for key in required}
        row["insertion_code"] = "" if row["pdb_ins_code"] in (".", "?") else row["pdb_ins_code"].strip()
        row["chain"] = row["pdb_strand_id"] if row["pdb_strand_id"] not in (".", "?") else row["asym_id"]
        try:
            row["auth_seq_num_int"] = int(row["auth_seq_num"])
        except ValueError:
            row["auth_seq_num_int"] = None
        row["mapped_letter"], row["mapping_class"] = one_letter(row["auth_mon_id"])
        rows.append(row)
    return rows


def map_side(atoms, rows, chain):
    # Prefer an exact auth/asym chain spelling.  A case-insensitive fallback is
    # useful for archives that normalize chain identifiers, but using it when
    # an exact chain exists can merge distinct chains such as ``B`` and ``b``.
    exact_chain_rows = [row for row in rows if row["chain"] == chain]
    chain_rows = exact_chain_rows or [
        row for row in rows if row["chain"].upper() == chain.upper()
    ]
    by_identity = {}
    for row in chain_rows:
        key = (row["auth_seq_num_int"], row["insertion_code"], row["auth_mon_id"].strip().upper())
        by_identity.setdefault(key, []).append(row)
    observed = atoms.copy()
    observed["hetero_norm"] = observed["hetero"].astype(str).str.strip()
    observed["resname_norm"] = observed["resname"].astype(str).str.strip().str.upper()
    residues = observed[["residue", "insertion_code", "resname_norm", "hetero_norm"]].drop_duplicates()
    matches = []
    for item in residues.itertuples(index=False):
        number, insertion, resname, hetero = item
        insertion = str(insertion).strip()
        key = (int(number), insertion, resname)
        candidates = by_identity.get(key, [])
        # A hetero record may have auth_mon_id but is not part of the polymer scheme.
        mapping = one_letter(resname)
        if len(candidates) == 1:
            row = candidates[0]
            matches.append(dict(residue=int(number), insertion_code=insertion, resname=resname,
                hetero=hetero, match="exact", seq_id=int(row["seq_id"]), sequence_letter=row["mapped_letter"],
                mapping_class=row["mapping_class"]))
        elif len(candidates) > 1:
            matches.append(dict(residue=int(number), insertion_code=insertion, resname=resname,
                hetero=hetero, match="ambiguous", candidates=len(candidates)))
        else:
            matches.append(dict(residue=int(number), insertion_code=insertion, resname=resname,
                hetero=hetero, match="unmatched", sequence_letter=mapping[0], mapping_class=mapping[1]))
    return dict(chain=chain, polymer_rows=len(chain_rows), observed_residues=len(matches),
                exact=sum(row["match"] == "exact" for row in matches),
                ambiguous=sum(row["match"] == "ambiguous" for row in matches),
                unmatched=sum(row["match"] == "unmatched" for row in matches),
                observed_nonhetero=sum(not row["hetero"] for row in matches),
                exact_rows=matches)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
        default=Path("runs/atom3d_audit/20260924/local_20260924_workstation_01/sample_manifest.json"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if root != Path("/home/zy/projects/Surface-residue") or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Sequence mapping diagnostic is authorized only on the workstation")
    if not args.run_id.startswith("local_"):
        raise ValueError("Run ID must begin with local_")
    manifest_path = root / args.manifest
    manifest = json.loads(manifest_path.read_text())
    if any(row["split"] == "test" for row in manifest["records"]):
        raise ValueError("Test records are sealed")
    run = root / "runs/atom3d_sequence_mapping/20260925" / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    mmcif_dir = root / "data/rcsb/mmcif"
    mmcif_dir.mkdir(parents=True, exist_ok=True)
    entries = {}
    for row in manifest["records"]:
        code = pdb_code(row["id"])
        entries.setdefault(code, []).append(row)
    report = dict(status="running", execution="local-workstation", run_id=args.run_id,
        host=socket.gethostname(), started_at=time.time(), manifest_sha256=sha256(manifest_path),
        training_enabled=False, test_records_read=0, source="RCSB files.rcsb.org mmCIF",
        records_requested=len(manifest["records"]), unique_pdb_codes=sorted(entries),
        downloads=[], records=[], failures=[])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, code, mmcif_dir) for code in sorted(entries)]
        for future in as_completed(futures):
            try:
                report["downloads"].append(future.result())
            except Exception:
                report["failures"].append(dict(stage="download", error=traceback.format_exc()))
    dbs = {split: ReleasedLMDB(root / "data/atom3d/zenodo-4911102/lmdb/DIPS-split/data" / split)
           for split in ("train", "val")}
    try:
        for sample in manifest["records"]:
            try:
                code = pdb_code(sample["id"])
                path = mmcif_dir / (code.lower() + ".cif")
                item = dbs[sample["split"]][sample["lmdb_index"]]
                if item["id"] != sample["id"]:
                    raise ValueError("ID mismatch")
                rows = mmcif_rows(path)
                atoms = item["atoms_pairs"]
                sides = []
                for subunit, side in atoms.groupby("subunit", sort=False):
                    sides.append(map_side(side, rows, chain_code(subunit)))
                record = dict(**sample, pdb_code=code, mmcif_sha256=sha256(path),
                    side_mappings=sides, polymer_rows_total=len(rows),
                    sequence_scope="RCSB _pdbx_poly_seq_scheme auth chain/number/insertion mapping")
                report["records"].append(record)
            except Exception:
                report["failures"].append(dict(**sample, error=traceback.format_exc()))
    finally:
        for db in dbs.values():
            db.close()
    report["downloads"] = sorted(report["downloads"], key=lambda row: row["code"])
    report["status"] = "sequence_mapping_diagnostic_passed" if not report["failures"] else "needs_review"
    report["finished_at"] = time.time()
    report["wall_seconds"] = report["finished_at"] - report["started_at"]
    (run / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (run / "source_manifest.json").write_text(json.dumps(dict(manifest=manifest_path.as_posix(),
        manifest_sha256=report["manifest_sha256"], pdb_codes=report["unique_pdb_codes"]), indent=2) + "\n")
    print("REPORT", run / "report.json", report["status"], len(report["records"]), flush=True)
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
