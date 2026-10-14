"""Extract frozen ESM2, GearNet and dMaSIF features for Phase 2 candidates.

This is intentionally a Slurm GPU-only script. It reads train/val LMDB
records selected by the frozen protocol, keeps partner coordinates out of
each monomer encoder, and writes labels separately from encoder inputs.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from surface_residue_ppi.adapters import neighborhoods
from surface_residue_ppi.encoders import FrozenGearNet, load_surface, surface_features
from surface_residue_ppi.ppi_audit import ReleasedLMDB
from surface_residue_ppi.structures import read_monomer


KEY_COLUMNS = ["ensemble", "subunit", "structure", "model", "chain", "residue"]
ELEMENTS = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_seqres_key(token: str) -> str:
    """Normalize only the PDB identifier; chain case is biologically significant."""
    pdb, chain = token.split("_", 1)
    return f"{pdb.lower()}_{chain}"


def iter_seqres(path: Path, requested: set[str]):
    header = None
    sequence = []
    with gzip.open(path, "rt", encoding="ascii", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    key = normalize_seqres_key(header.split()[0])
                    if key in requested:
                        yield key, "".join(sequence).upper()
                header, sequence = line[1:], []
            else:
                sequence.append(line)
    if header is not None:
        key = normalize_seqres_key(header.split()[0])
        if key in requested:
            yield key, "".join(sequence).upper()


def raw_id_chains(record_id: str) -> tuple[str, str]:
    """Return the two chain tokens encoded in a released DIPS pair ID.

    Released IDs have the form ``<pdbfile>_<model>_<chain>`` repeated for the
    two monomers.  Keep this parser deliberately strict so a future ID format
    cannot silently associate a sequence with the wrong side.
    """
    parts = record_id.split("_")
    if len(parts) != 6:
        raise ValueError(f"Unexpected released pair ID format: {record_id}")
    return parts[2], parts[5]


def seqres_candidates(record: dict, protocol_side: dict, side_index: int) -> list[str]:
    """Return normalized SEQRES keys in protocol-first order.

    The frozen protocol uses the RCSB auth chain.  Some SEQRES mirrors use a
    different case or retain the chain token from the original released ID,
    so the latter is a safe fallback when the protocol key is absent.
    """
    if 'seqres_key' in protocol_side:
        return [protocol_side['seqres_key']]
    protocol_key = f"{record['pdb'].lower()}_{protocol_side['chain']}"
    raw_chain = raw_id_chains(record["id"])[side_index]
    raw_key = f"{record['pdb'].lower()}_{raw_chain}"
    return list(dict.fromkeys((protocol_key, raw_key)))


def key_tuple(row):
    return tuple(row[column] for column in KEY_COLUMNS)


def write_pdb(path: Path, atoms: pd.DataFrame, common_to_seq: dict[tuple, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = 1
    lines = []
    for _, row in atoms.iterrows():
        key = key_tuple(row)
        if key not in common_to_seq:
            continue
        element = norm(row["element"]).upper()
        if element in {"H", "D"} or element not in ELEMENTS:
            continue
        name = norm(row["name"])[:4]
        resname = norm(row["resname"]).upper()[:3]
        residue = int(row["residue"])
        insertion = norm(row["insertion_code"])[:1]
        x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
        occupancy = float(row.get("occupancy", 1.0)) if pd.notna(row.get("occupancy", 1.0)) else 1.0
        bfactor = float(row.get("bfactor", 0.0)) if pd.notna(row.get("bfactor", 0.0)) else 0.0
        lines.append(f"ATOM  {serial:5d} {name:^4s} {resname:>3s} A{residue:4d}{insertion:1s}   "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{bfactor:6.2f}          {element:>2s}\n")
        serial += 1
    if not lines:
        raise ValueError("No supported atoms for selected common residues")
    path.write_text("".join(lines) + "TER\nEND\n")


def prepare_side(item, protocol_side):
    atoms = item["atoms_pairs"]
    # Match by the protocol's chain and the two subunits' order. The protocol
    # stores the common residue keys, so subunit identity is unambiguous.
    common = protocol_side["common_valid"]
    common_to_seq = {tuple(entry["residue_key"]): int(entry["seq_id"]) for entry in common}
    if not common_to_seq:
        raise ValueError("Empty common mask")
    subunits = atoms["subunit"].drop_duplicates().tolist()
    side_candidates = []
    for subunit_name in subunits:
        side = atoms[atoms["subunit"] == subunit_name]
        keys = set(key_tuple(row) for _, row in side.iterrows())
        if set(common_to_seq) <= keys:
            side_candidates.append((subunit_name, side))
    if len(side_candidates) != 1:
        raise ValueError("Protocol common keys do not identify exactly one monomer")
    _, side = side_candidates[0]
    selected = side[side.apply(lambda row: key_tuple(row) in common_to_seq, axis=1)].copy()
    coords = selected[["x", "y", "z"]].to_numpy(dtype=np.float32)
    elements = selected["element"].map(lambda value: norm(value).upper())
    keep = elements.isin(ELEMENTS) & ~elements.isin({"H", "D"})
    selected = selected.loc[keep].copy()
    elements = elements.loc[keep]
    coords = selected[["x", "y", "z"]].to_numpy(dtype=np.float32)
    if not len(coords) or not np.isfinite(coords).all():
        raise ValueError("No finite supported atom coordinates")
    center = coords.mean(axis=0)
    xyz = coords - center
    atom_to_seq = np.asarray([common_to_seq[key_tuple(row)] - 1 for _, row in selected.iterrows()], dtype=np.int64)
    atom_types = np.eye(len(ELEMENTS), dtype=np.float32)[[ELEMENTS[value] for value in elements]]
    residue_insertion_to_seq = {
        (int(row["residue"]), norm(row["insertion_code"])): common_to_seq[key_tuple(row)]
        for _, row in selected.drop_duplicates(KEY_COLUMNS).iterrows()
    }
    return dict(selected=selected, xyz=xyz, atom_to_seq=atom_to_seq, atom_types=atom_types,
                center=center, common_to_seq=common_to_seq,
                residue_insertion_to_seq=residue_insertion_to_seq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--lmdb-indices", nargs="+", type=int, default=None,
                        help="Extract only these frozen train/val LMDB indices for a continuation run")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--esm-layer", type=int, default=12)
    parser.add_argument("--output-root", default="runs/phase2_features")
    parser.add_argument("--exclude-report", type=Path, default=None,
                        help="Reuse completed records elsewhere; extract only new records")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Submit feature extraction with sbatch on a GPU node")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol_path = root / args.protocol
    protocol = json.loads(protocol_path.read_text())
    if args.output_root == 'runs/phase3_features' and (
            protocol.get('status') != 'protocol_ready' or protocol.get('failures')
            or protocol.get('test_records_read') != 0 or protocol.get('test_labels_read') != 0):
        raise ValueError('Phase 3 requires a finalized train/val-only protocol')
    records = json.loads((protocol_path.parent / "pilot_manifest.json").read_text())
    if args.split:
        records = [row for row in records if row["split"] == args.split]
    if args.lmdb_indices is not None:
        requested_indices = set(args.lmdb_indices)
        records = [row for row in records if row["lmdb_index"] in requested_indices]
        found_indices = {row["lmdb_index"] for row in records}
        if found_indices != requested_indices:
            raise ValueError(f"Requested LMDB indices not present in frozen manifest: "
                             f"{sorted(requested_indices - found_indices)}")
    records = sorted(records, key=lambda row: (row["split"], row["lmdb_index"]))
    if any(row['split'] not in ('train', 'val') for row in records):
        raise ValueError('Test records are sealed')
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Invalid shard index/count')
    if args.exclude_report is not None:
        old_report = json.loads((root / args.exclude_report).read_text())
        if old_report.get('status') != 'passed' or old_report.get('test_records_read') != 0:
            raise ValueError('Invalid source feature report')
        old_keys = {(row['split'], row['lmdb_index']) for row in old_report['records']}
        records = [row for row in records if (row['split'], row['lmdb_index']) not in old_keys]
    records = records[args.shard_index::args.shard_count]
    if args.limit_per_split is not None:
        records = [row for split in ("train", "val")
                   for row in [candidate for candidate in records if candidate["split"] == split][:args.limit_per_split]]
    elif args.limit is not None:
        records = records[:args.limit]
    job = os.environ["SLURM_JOB_ID"]
    run_id = job + ('_' + os.environ['SLURM_ARRAY_TASK_ID'] if 'SLURM_ARRAY_TASK_ID' in os.environ else '')
    out = root / args.output_root / run_id
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", job_id=job, host=platform.node(), started_at=time.time(),
                  protocol=str(args.protocol), protocol_sha256=sha256(protocol_path),
                  records_requested=len(records), records=[], failures=[], test_records_read=0,
                  labels_outside_encoder_inputs=True, execution="slurm-gpu")
    report.update(shard_index=args.shard_index, shard_count=args.shard_count,
                  config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  source_sha256=sha256(Path(__file__)), array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'))
    report.update(test_labels_read=0,
                  manifest_sha256=sha256(protocol_path.parent / 'pilot_manifest.json'),
                  encoder_config_sha256=sha256(root / 'configs/smoke.json'),
                  module_sha256={name: sha256(root / name) for name in (
                      'src/surface_residue_ppi/adapters.py', 'src/surface_residue_ppi/encoders.py',
                      'src/surface_residue_ppi/structures.py')})
    dbs = {split: ReleasedLMDB(root / "data/atom3d/zenodo-4911102/lmdb/DIPS-split/data" / split)
           for split in ("train", "val")}
    try:
        import esm
        config = json.loads((root / "configs/smoke.json").read_text())
        candidate_keys = {
            (record_index, side_index): seqres_candidates(record, side, side_index)
            for record_index, record in enumerate(records)
            for side_index, side in enumerate(record["sides"])
        }
        requested = {key for candidates in candidate_keys.values() for key in candidates}
        seqres = {}
        for key, value in iter_seqres(root / "data/rcsb/derived_data/pdb_seqres.txt.gz", requested):
            seqres[key] = value
        missing = [
            dict(record_index=record_index, side=side_index, candidates=candidates)
            for (record_index, side_index), candidates in candidate_keys.items()
            if not any(key in seqres for key in candidates)
        ]
        if missing:
            raise ValueError(f"Missing SEQRES chains after protocol/raw fallback: {missing[:10]}")
        surface, surface_checkpoint = load_surface(root, config["dmasif"])
        gearnet = FrozenGearNet(config["gearnet_checkpoint"]).cuda()
        sequence_model, alphabet = getattr(esm.pretrained, config["esm_variant"])()
        sequence_model = sequence_model.cuda().eval().requires_grad_(False)
        converter = alphabet.get_batch_converter()
        esm_path = Path(torch.hub.get_dir()) / "checkpoints" / (config["esm_variant"] + ".pt")
        dmasif_branch = config["dmasif_branch"]
        report.update(dmasif_sha256=sha256(surface_checkpoint), gearnet_sha256=sha256(Path(config["gearnet_checkpoint"])),
                      esm_sha256=sha256(esm_path), esm_variant=config["esm_variant"],
                      esm_dim=None, gearnet_dim=FrozenGearNet.output_dim, dmasif_branch=dmasif_branch,
                      dmasif_dim=None)
        with tempfile.TemporaryDirectory(prefix="srppi_phase2_", dir=str(root / "data/cache")) as temp_dir:
            temp_dir = Path(temp_dir)
            for number, record in enumerate(records, 1):
                started = time.time()
                try:
                    item = dbs[record["split"]][record["lmdb_index"]]
                    if item["id"] != record["id"]:
                        raise ValueError("protocol ID/index mismatch")
                    side_results = []
                    for side_index, protocol_side in enumerate(record["sides"]):
                        prep = prepare_side(item, protocol_side)
                        seqres_key = next(key for key in candidate_keys[(number - 1, side_index)] if key in seqres)
                        seq = seqres[seqres_key]
                        seq_len = len(seq)
                        if protocol_side.get('sequence_sha256') and hashlib.sha256(seq.encode()).hexdigest() != protocol_side['sequence_sha256']:
                            raise ValueError('Frozen SEQRES sequence hash mismatch')
                        for entry in protocol_side['common_valid']:
                            if not 1 <= int(entry['seq_id']) <= seq_len or seq[int(entry['seq_id']) - 1] != entry['sequence_letter']:
                                raise ValueError('SEQRES identity differs from frozen protocol')
                        if max(prep["common_to_seq"].values()) > seq_len:
                            raise ValueError("SEQRES length is shorter than mapped seq_id")
                        temp_pdb = temp_dir / f"{number}_{side_index}.pdb"
                        write_pdb(temp_pdb, prep["selected"], prep["common_to_seq"])
                        parsed = read_monomer(temp_pdb)
                        atoms = torch.from_numpy(prep["xyz"]).cuda()
                        atom_ids = torch.from_numpy(prep["atom_to_seq"]).cuda()
                        types = torch.from_numpy(prep["atom_types"]).cuda()
                        surface_seed = args.seed + number * 2 + side_index
                        if args.output_root == 'runs/phase3_features':
                            surface_seed = int(hashlib.sha256((str(args.seed) + ':' + record['split'] + ':' + record['id'] + ':' + str(side_index)).encode()).hexdigest()[:8], 16) % (2**31)
                        point_data = surface_features(surface, atoms, types, surface_seed)
                        with torch.no_grad():
                            raw_gear, gear_keys, _ = gearnet(temp_pdb)
                            g = torch.zeros(seq_len, raw_gear.shape[1], device="cuda")
                            gear_seen = set()
                            for gear_row, gear_key in zip(raw_gear, gear_keys):
                                lookup = (int(gear_key[1]), str(gear_key[2]).strip())
                                matches = [seq_id for (residue, insertion), seq_id in prep["residue_insertion_to_seq"].items()
                                           if (residue, insertion) == lookup]
                                if len(matches) == 1:
                                    g[matches[0] - 1] = gear_row
                                    gear_seen.add(matches[0])
                            _, _, tokens = converter([(record["id"] + f"_{side_index}", seq)])
                            e = sequence_model(tokens.cuda(), repr_layers=[args.esm_layer],
                                               return_contacts=False)["representations"][args.esm_layer][0, 1:seq_len + 1]
                        if gear_seen != set(prep['common_to_seq'].values()):
                            raise ValueError('GearNet did not cover every common-valid residue')
                        a, b, owner = neighborhoods(atoms, atom_ids, point_data["xyz"], seq_len, **config["mapping"])
                        valid = torch.zeros(seq_len, device="cuda", dtype=torch.bool)
                        valid[torch.tensor([value - 1 for value in prep["common_to_seq"].values()], device="cuda")] = True
                        output_dir = out / record["split"]
                        output_dir.mkdir(parents=True, exist_ok=True)
                        prefix = output_dir / f"{record['lmdb_index']}_{side_index}"
                        np.savez_compressed(prefix.with_suffix(".npz"), esm=e.cpu().numpy(), gearnet=g.cpu().numpy(),
                                            surface=point_data[dmasif_branch].cpu().numpy(),
                                            surface_xyz=point_data["xyz"].cpu().numpy(), common_valid=valid.cpu().numpy(),
                                            point_owner=owner.cpu().numpy(), atom_xyz=prep["xyz"], atom_to_seq=prep["atom_to_seq"],
                                            A_residue=a.residue.cpu().numpy(), A_point=a.point.cpu().numpy(), A_distance=a.distance.cpu().numpy(),
                                            BC_residue=b.residue.cpu().numpy(), BC_point=b.point.cpu().numpy(), BC_distance=b.distance.cpu().numpy())
                        labels = np.zeros(seq_len, dtype=np.int8)
                        for entry in protocol_side["common_valid"]:
                            if entry["positive"]:
                                labels[int(entry["seq_id"]) - 1] = 1
                        np.save(prefix.with_name(prefix.name + "_labels.npy"), labels)
                        side_results.append(dict(side=side_index, seqres_key=seqres_key,
                                                sequence_length=seq_len,
                                                sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(), surface_seed=surface_seed,
                                                 common_valid=int(valid.sum()), gearnet_seen=len(gear_seen),
                                                 surface_points=len(point_data[dmasif_branch]),
                                                 cache=str(prefix.with_suffix(".npz").relative_to(root))))
                        del point_data, e, g, raw_gear
                        torch.cuda.empty_cache()
                    result = dict(split=record["split"], id=record["id"], lmdb_index=record["lmdb_index"],
                                  sides=side_results, seconds=time.time() - started,
                                  peak_gpu_bytes=torch.cuda.max_memory_allocated())
                    report["records"].append(result)
                    temporary_report = out / 'report.partial.json'
                    temporary_report.write_text(json.dumps(report, indent=2) + '\n')
                    temporary_report.replace(out / 'report.json')
                    print("RECORD", json.dumps(result), flush=True)
                except Exception:
                    failure = dict(split=record["split"], id=record["id"], lmdb_index=record["lmdb_index"],
                                   error=traceback.format_exc())
                    report["failures"].append(failure)
                    print(failure, flush=True)
        report["status"] = "passed" if not report["failures"] else "partial"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    finally:
        for db in dbs.values():
            db.close()
        report["finished_at"] = time.time()
        report["wall_seconds"] = report["finished_at"] - report["started_at"]
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("REPORT", out / "report.json", report["status"], flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
