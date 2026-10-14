"""Expand train/val using existing sequences/clusters; never decode test records."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import time

from surface_residue_ppi.cluster_selection import read_clusters
from surface_residue_ppi.ppi_audit import ReleasedLMDB


BASE = Path("runs/phase2_candidates/20260925/local_20250925_phase2_candidates_03")
OLD = Path("runs/phase2_protocol/20260925/local_20250925_phase2_protocol_01")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose(pool, anchors, limit, blocked_clusters, blocked_pdbs):
    chosen = list(anchors)
    pdbs = {x['pdb'] for x in chosen}
    signatures = {tuple(x['cluster_pair']) for x in chosen}
    keys = {(x['split'], x['lmdb_index']) for x in chosen}
    skipped = Counter()
    for row in sorted(pool, key=lambda x: hashlib.sha256(
            ('phase3-20260927:' + x['split'] + ':' + x['id']).encode()).hexdigest()):
        if len(chosen) >= limit:
            break
        if (row['split'], row['lmdb_index']) in keys:
            continue
        if set(row['cluster_pair']) & blocked_clusters:
            skipped['blocked_cluster'] += 1
        elif row['pdb'] in blocked_pdbs or row['pdb'] in pdbs:
            skipped['repeated_or_blocked_pdb'] += 1
        elif tuple(row['cluster_pair']) in signatures:
            skipped['repeated_cluster_pair'] += 1
        else:
            chosen.append(row)
            pdbs.add(row['pdb'])
            signatures.add(tuple(row['cluster_pair']))
    return chosen, dict(skipped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execution', choices=['slurm', 'local-workstation'], required=True)
    parser.add_argument('--train-limit', type=int, default=5000)
    parser.add_argument('--val-limit', type=int, default=500)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.execution == 'slurm':
        if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'hpc':
            raise RuntimeError('Use a CPU Slurm job on hpc')
    elif root != Path('/home/zy/projects/Surface-residue'):
        raise RuntimeError('Workstation audit exception is limited to the authorized root')
    started = time.time()
    out = root / args.out
    out.mkdir(parents=True, exist_ok=False)
    cluster_path = root / BASE / 'mmseqs/chains_cluster.tsv'
    reps = read_clusters(cluster_path)
    sequences = {}
    header = None
    for line in (root / BASE / 'requested_protein_chains.fasta').read_text().splitlines():
        if line.startswith('>'):
            header = line[1:].split()[0]
            sequences[header] = ''
        else:
            sequences[header] += line.strip()
    assert set(sequences) == set(reps), 'FASTA/cluster coverage mismatch'
    pools, by_key, excluded = {}, {}, []
    for split in ('train', 'val'):
        db = ReleasedLMDB(root / 'data/atom3d/zenodo-4911102/lmdb/DIPS-split/data' / split)
        pools[split] = []
        try:
            for identifier, index in db.id_to_idx.items():
                parts = identifier.split('_')
                if len(parts) != 6:
                    raise ValueError('Unexpected DIPS identifier: ' + identifier)
                pdbs = [re.sub('^pdb', '', parts[i].split('.')[0], flags=re.I).upper()
                        for i in (0, 3)]
                if pdbs[0] != pdbs[1]:
                    raise ValueError('Pair spans two PDBs: ' + identifier)
                chains = [pdbs[0] + '_' + parts[2], pdbs[1] + '_' + parts[5]]
                lengths = [len(sequences.get(c, '')) for c in chains]
                # Resource limits are sequence-only and applied before labels are read.
                if not all(c in reps for c in chains) or min(lengths) < 1 or max(lengths) > 2000:
                    excluded.append(dict(split=split, id=identifier, lmdb_index=index,
                                         reason='missing_sequence_or_length_outside_1_2000', lengths=lengths))
                    continue
                row = dict(split=split, id=identifier, lmdb_index=int(index), pdb=pdbs[0],
                           chain_ids=chains, sequence_lengths=lengths,
                           cluster_pair=sorted(reps[c] for c in chains))
                pools[split].append(row)
                by_key[(split, int(index))] = row
        finally:
            db.close()
    anchors = {}
    for split in ('train', 'val'):
        old = json.loads((root / OLD / (split + '_manifest.json')).read_text())
        anchors[split] = [by_key[(split, x['lmdb_index'])] for x in old]
        assert all(a['id'] == b['id'] for a, b in zip(anchors[split], old))
    clusters = lambda rows: {c for r in rows for c in r['cluster_pair']}
    actual_old_overlap = clusters(anchors['train']) & clusters(anchors['val'])
    assert not actual_old_overlap, 'Existing split needs isolation repair'
    previous_val = json.loads((root / BASE / 'val_candidates.json').read_text())
    protected_val = [by_key[('val', r['lmdb_index'])] for r in previous_val
                     if ('val', r['lmdb_index']) in by_key]
    val, val_skips = choose(pools['val'], anchors['val'], args.val_limit,
                           clusters(anchors['train']), {r['pdb'] for r in anchors['train']})
    train, train_skips = choose(pools['train'], anchors['train'], args.train_limit,
                               clusters(val + protected_val), {r['pdb'] for r in val + protected_val})
    overlap = clusters(train) & clusters(val)
    assert not overlap
    assert not ({r['pdb'] for r in train} & {r['pdb'] for r in val})
    report = dict(status='candidates_ready_for_audit', execution=args.execution,
                  host=platform.node(), python=platform.python_version(),
                  started_at=started, wall_seconds=time.time() - started,
                  targets=dict(train=args.train_limit, val=args.val_limit),
                  counts=dict(train=len(train), val=len(val)),
                  anchors={k: len(v) for k, v in anchors.items()},
                  cluster_overlap_count=len(overlap), old_actual_cluster_overlap_count=0,
                  cluster_tsv_sha256=digest(cluster_path),
                  cluster_policy=dict(identity=0.9, coverage=0.8, cov_mode=0,
                                      mapping='member (column 2) -> representative (column 1)'),
                  old_selection_defect='representative keys excluded non-representative members; existing split still disjoint',
                  missing_or_length_exclusions=len(excluded), skipped=dict(train=train_skips, val=val_skips),
                  test_records_read=0, test_labels_read=0, train_val_coordinates_read=0,
                  selection=dict(train=train, val=val, strict_val=val))
    for name, value in [('report.json', report), ('train_candidates.json', train),
                        ('val_candidates.json', val), ('excluded.json', excluded)]:
        (out / name).write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'selection'}), flush=True)


if __name__ == '__main__':
    main()
