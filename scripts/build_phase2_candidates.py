"""Build a homology-aware Phase 2 task-one candidate manifest locally.

The script reads only train/val LMDB id maps and the public RCSB SEQRES
catalog. It does not decode test records or read train/val coordinates,
labels, or interface contacts. Sequence clusters are produced with a pinned
local MMseqs2 binary, then a deterministic one-complex-per-PDB and
one-complex-signature-per-cluster-pair selection is made.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from surface_residue_ppi.ppi_audit import ReleasedLMDB


ID_RE = re.compile(
    r"(?P<a>(?:pdb)?[0-9A-Za-z]{4}\.pdb\d+\.gz_\d+_[^_]+)_"
    r"(?P<b>(?:pdb)?[0-9A-Za-z]{4}\.pdb\d+\.gz_\d+_[^_]+)$",
    re.I,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_pair(identifier: str) -> tuple[tuple[str, str], tuple[str, str]]:
    match = ID_RE.match(str(identifier))
    if not match:
        raise ValueError(f"Cannot parse DIPS pair identifier: {identifier}")
    chains = []
    for subunit in (match.group("a"), match.group("b")):
        code = subunit.split(".", 1)[0].removeprefix("pdb").upper()
        chain = subunit.rsplit("_", 1)[-1]
        chains.append((code, chain))
    return chains[0], chains[1]


def iter_seqres(path: Path):
    header = None
    sequence = []
    with gzip.open(path, "rt", encoding="ascii", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(sequence)
                header = line[1:]
                sequence = []
            else:
                sequence.append(line)
    if header is not None:
        yield header, "".join(sequence)


def extract_requested_sequences(seqres: Path, requested: set[tuple[str, str]], output: Path):
    exact = {}
    fallback = {}
    for header, sequence in iter_seqres(seqres):
        token = header.split()[0]
        if "_" not in token:
            continue
        code, chain = token.split("_", 1)
        key = (code.upper(), chain)
        if key in requested:
            exact[key] = (token, header, sequence.upper())
        fallback.setdefault((code.upper(), chain.upper()), []).append((token, header, sequence.upper()))
    selected = {}
    ambiguous_fallback = []
    for key in sorted(requested):
        if key in exact:
            selected[key] = exact[key]
            continue
        choices = fallback.get((key[0], key[1].upper()), [])
        if len(choices) == 1:
            selected[key] = choices[0]
        elif choices:
            ambiguous_fallback.append(dict(code=key[0], chain=key[1], choices=len(choices)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for key in sorted(selected):
            token, header, sequence = selected[key]
            # Keep only polymer protein chains for task-one candidates. The
            # source header is retained in the manifest for auditability.
            if "mol:protein" not in header.lower():
                continue
            if len(sequence) < 1:
                continue
            record_id = f"{key[0]}_{key[1]}"
            stream.write(f">{record_id}\n{sequence}\n")
    return selected, ambiguous_fallback


def run_mmseqs(mmseqs: Path, fasta: Path, work: Path, min_seq_id: float, coverage: float):
    work.mkdir(parents=True, exist_ok=True)
    db = work / "chains_db"
    clu = work / "chains_cluster"
    tsv = work / "chains_cluster.tsv"
    tmp = work / "tmp"
    env = os.environ.copy()
    lib = mmseqs.parent.parent / "lib/mmseqs2"
    gzlib = root_from_mmseqs(mmseqs) / "tools/gzstream/usr/lib/x86_64-linux-gnu"
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(gzlib), str(lib)] + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
    )
    commands = [
        [str(mmseqs), "createdb", str(fasta), str(db)],
        [str(mmseqs), "cluster", str(db), str(clu), str(tmp), "--min-seq-id", str(min_seq_id),
         "-c", str(coverage), "--cov-mode", "0", "--cluster-mode", "0", "--threads", "16"],
        [str(mmseqs), "createtsv", str(db), str(db), str(clu), str(tsv)],
    ]
    logs = []
    for command in commands:
        result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
        logs.append(dict(command=command, stdout=result.stdout[-2000:], stderr=result.stderr[-2000:]))
    representatives = {}
    for line in tsv.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) >= 2:
            representatives[fields[0]] = fields[1]
    return representatives, logs


def root_from_mmseqs(mmseqs: Path) -> Path:
    """Resolve the workstation project root from the bundled binary path."""
    return Path(__file__).resolve().parents[1]


def ordered_selection(records, reps, limit, blocked_clusters=None):
    blocked_clusters = set(blocked_clusters or ())
    selected = []
    used_pdb = set()
    used_signature = set()
    skipped = {"missing_sequence": 0, "nonprotein_or_short": 0, "blocked_cluster": 0,
               "repeated_pdb": 0, "repeated_signature": 0}
    for record in sorted(records, key=lambda row: hashlib.sha256(row["sort_key"].encode()).hexdigest()):
        chain_ids = [f"{code}_{chain}" for code, chain in record["chains"]]
        if any(chain_id not in reps for chain_id in chain_ids):
            skipped["missing_sequence"] += 1
            continue
        cluster_pair = tuple(sorted(reps[chain_id] for chain_id in chain_ids))
        if blocked_clusters and set(cluster_pair) & blocked_clusters:
            skipped["blocked_cluster"] += 1
            continue
        if record["pdb"] in used_pdb:
            skipped["repeated_pdb"] += 1
            continue
        if cluster_pair in used_signature:
            skipped["repeated_signature"] += 1
            continue
        used_pdb.add(record["pdb"])
        used_signature.add(cluster_pair)
        selected.append(dict(record, chain_ids=chain_ids, cluster_pair=list(cluster_pair)))
        if len(selected) >= limit:
            break
    return selected, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seqres", type=Path, default=Path("data/rcsb/derived_data/pdb_seqres.txt.gz"))
    parser.add_argument("--mmseqs", type=Path, default=Path("tools/mmseqs2/usr/bin/mmseqs-avx2"))
    parser.add_argument("--min-seq-id", type=float, default=0.9)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--val-limit", type=int, default=100)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if root != Path("/home/zy/projects/Surface-residue") or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Candidate selection is authorized only on the workstation")
    if not args.run_id.startswith("local_"):
        raise ValueError("Run ID must begin with local_")
    if not args.seqres.is_file():
        raise FileNotFoundError(args.seqres)
    if not args.mmseqs.is_file():
        raise FileNotFoundError(args.mmseqs)
    run = root / "runs/phase2_candidates/20260925" / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    started = time.time()
    dbs = {}
    records = {"train": [], "val": []}
    requested = set()
    try:
        for split in ("train", "val"):
            db = ReleasedLMDB(root / "data/atom3d/zenodo-4911102/lmdb/DIPS-split/data" / split)
            dbs[split] = db
            for identifier, index in db.id_to_idx.items():
                chains = parse_pair(identifier)
                requested.update(chains)
                pdb = chains[0][0]
                records[split].append(dict(split=split, id=identifier, lmdb_index=int(index),
                                           pdb=pdb, chains=[list(chains[0]), list(chains[1])],
                                           sort_key=f"20260925-phase2:{split}:{identifier}"))
        fasta = run / "requested_protein_chains.fasta"
        selected_sequences, ambiguous_fallback = extract_requested_sequences(args.seqres, requested, fasta)
        reps, mmseqs_logs = run_mmseqs(args.mmseqs, fasta, run / "mmseqs", args.min_seq_id, args.coverage)
        train, train_skipped = ordered_selection(records["train"], reps, args.train_limit)
        train_clusters = {cluster for row in train for cluster in row["cluster_pair"]}
        val, val_skipped = ordered_selection(records["val"], reps, args.val_limit)
        strict_val, strict_val_skipped = ordered_selection(records["val"], reps, args.val_limit, train_clusters)
        mmseqs_env = os.environ.copy()
        mmseqs_env["LD_LIBRARY_PATH"] = ":".join([
            str(root / "tools/gzstream/usr/lib/x86_64-linux-gnu"),
            str(args.mmseqs.parent.parent / "lib/mmseqs2"),
        ] + ([mmseqs_env["LD_LIBRARY_PATH"]] if mmseqs_env.get("LD_LIBRARY_PATH") else []))
        report = dict(
            status="phase2_candidate_manifest_complete",
            execution="local-workstation",
            run_id=args.run_id,
            started_at=started,
            finished_at=time.time(),
            source=dict(seqres_path=str(args.seqres), seqres_sha256=sha256(args.seqres),
                        mmseqs_path=str(args.mmseqs), mmseqs_version=subprocess.run(
                            [str(args.mmseqs), "version"], env=mmseqs_env,
                            text=True, capture_output=True, check=True).stdout.strip(),
                        min_seq_id=args.min_seq_id, coverage=args.coverage),
            test_records_read=0, training_enabled=False,
            records_by_split={split: len(values) for split, values in records.items()},
            unique_pdbs_by_split={split: len({row["pdb"] for row in values}) for split, values in records.items()},
            requested_chains=len(requested), seqres_selected=len(selected_sequences),
            protein_fasta_records=sum(1 for line in fasta.read_text().splitlines() if line.startswith(">")),
            missing_requested_chains=len(requested - set(selected_sequences)),
            ambiguous_case_fallback=ambiguous_fallback,
            cluster_members=len(reps),
            selection=dict(train=train, val=val, strict_val=strict_val),
            skipped=dict(train=train_skipped, val=val_skipped, strict_val=strict_val_skipped),
            mmseqs_logs=mmseqs_logs,
        )
        report["finished_at"] = time.time()
        report["wall_seconds"] = report["finished_at"] - started
        (run / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        for name, values in (("train_candidates.json", train), ("val_candidates.json", val),
                             ("strict_val_candidates.json", strict_val)):
            (run / name).write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n")
        print("REPORT", run / "report.json", report["status"], len(train), len(val), len(strict_val), flush=True)
    finally:
        for db in dbs.values():
            db.close()


if __name__ == "__main__":
    main()
