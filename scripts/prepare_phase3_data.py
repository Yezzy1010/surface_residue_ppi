"""Resumable CPU preparation of Phase 3 data; reuses the Phase 2 protocol.

Only selected train/val records are decoded. Per-record artifacts survive
interruptions. Model features and all model execution stay on Slurm GPUs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np

from audit_phase2_candidates import audit_one, json_default
from audit_sequence_mapping import download_one, mmcif_rows, map_side, chain_code, CANONICAL
from surface_residue_ppi.ppi_audit import ReleasedLMDB


ROOT = None
SEQUENCES = None
OFFLINE = False
OLD = Path('runs/phase2_protocol/20260925/local_20250925_phase2_protocol_01')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, default=json_default) + '\n')
    temp.replace(path)


def init_worker(root, sequences, offline):
    global ROOT, SEQUENCES, OFFLINE
    ROOT, SEQUENCES, OFFLINE = Path(root), sequences, offline


def add_sequence_identity(record):
    for i, side in enumerate(record['sides']):
        candidates = [record['pdb'].lower() + '_' + side['chain'],
                      record['pdb'].lower() + '_' + record['id'].split('_')[2 + 3 * i]]
        key = next((k for k in candidates if k in SEQUENCES), None)
        if key is None:
            raise ValueError('Missing exact protocol/raw SEQRES chain ' + str(candidates))
        sequence = SEQUENCES[key]
        positions = [x['seq_id'] for x in side['common_valid']]
        if len(set(positions)) != len(positions):
            raise ValueError('Duplicate sequence position')
        for residue in side['common_valid']:
            pos = residue['seq_id'] - 1
            if not 0 <= pos < len(sequence) or sequence[pos] != residue['sequence_letter']:
                raise ValueError('SEQRES residue identity mismatch ' + key)
        side.update(seqres_key=key, sequence_length=len(sequence),
                    sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest())
    return record


def prepare(row):
    started = time.time()
    result = dict(split=row['split'], lmdb_index=row['lmdb_index'], id=row['id'],
                  pdb=row['pdb'], test_records_read=0)
    try:
        if row['split'] not in ('train', 'val'):
            raise ValueError('Test split sealed')
        database = ReleasedLMDB(ROOT / 'data/atom3d/zenodo-4911102/lmdb/DIPS-split/data' / row['split'])
        try:
            item = database[row['lmdb_index']]
        finally:
            database.close()
        if item['id'] != row['id']:
            raise ValueError('ID/index mismatch')
        atoms = item['atoms_pairs']
        if not np.isfinite(atoms[['x', 'y', 'z']].to_numpy(dtype=float)).all():
            return dict(result, status='excluded', reasons=['nonfinite_coordinates'])
        audited = audit_one(dict(row, atoms_pairs=atoms, atoms_neighbors=item['atoms_neighbors']))
        result['audit'] = audited
        if not audited['ready_for_feature_extraction']:
            return dict(result, status='excluded', reasons=audited['blocking_issues'] or ['empty_structural_mask'])
        cif = ROOT / 'data/rcsb/mmcif' / (row['pdb'].lower() + '.cif')
        if OFFLINE and not cif.is_file():
            raise FileNotFoundError('mmCIF missing in offline mode: ' + str(cif))
        result['mmcif'] = download_one(row['pdb'], cif.parent)
        scheme = mmcif_rows(cif)
        sides = []
        exclusions = []
        for i, (subunit, frame) in enumerate(atoms.groupby('subunit', sort=False)):
            chain = chain_code(subunit)
            exact_chains = {x['chain'] for x in scheme if x['chain'] == chain}
            fallback = {x['chain'] for x in scheme if x['chain'].upper() == chain.upper()}
            if not exact_chains and len(fallback) != 1:
                exclusions.append('ambiguous_or_absent_auth_chain')
                continue
            mapping = map_side(frame, scheme, chain)
            exact = {(x['residue'], x['insertion_code'], x['resname'], x['hetero']): x
                     for x in mapping['exact_rows'] if x['match'] == 'exact'}
            valid = [x for x in audited['sides'][i]['rows'] if x['valid']]
            common = []
            mapped = 0
            for residue in valid:
                match = exact.get((residue['residue'], residue['insertion_code'], residue['resname'], ''))
                if match is None:
                    continue
                mapped += 1
                if match['sequence_letter'] == 'X':
                    continue
                if CANONICAL[residue['resname']] != match['sequence_letter']:
                    exclusions.append('sequence_identity_mismatch')
                    continue
                common.append(dict(residue_key=residue['key'], seq_id=int(match['seq_id']),
                                   sequence_letter=match['sequence_letter'], positive=bool(residue['positive'])))
            fraction = mapped / len(valid) if valid else 0.0
            if fraction < 0.95 or not common:
                exclusions.append('mapping_below_0.95_or_empty_mask')
            sides.append(dict(chain=chain, structural_valid_count=len(valid), sequence_mapped_count=mapped,
                              mapping_fraction=fraction, common_valid_count=len(common),
                              common_positive_count=sum(x['positive'] for x in common), common_valid=common))
        if exclusions or len(sides) != 2:
            return dict(result, status='excluded', reasons=sorted(set(exclusions)))
        protocol = dict(row, sides=sides, ready_for_feature_extraction=True,
                        sequence_scope='RCSB auth chain/number/insertion exact mapping',
                        min_mapping_fraction=min(x['mapping_fraction'] for x in sides))
        try:
            result['protocol'] = add_sequence_identity(protocol)
        except ValueError as error:
            return dict(result, status='excluded', reasons=['sequence_identity_gate'], detail=str(error))
        result['status'] = 'ready'
    except Exception:
        result.update(status='failed', error=traceback.format_exc())
    result['seconds'] = time.time() - started
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execution', choices=['slurm', 'local-workstation'], required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--retry-failures', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.execution == 'slurm':
        if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'hpc':
            raise RuntimeError('CPU preparation requires hpc Slurm')
    elif root != Path('/home/zy/projects/Surface-residue'):
        raise RuntimeError('Unauthorized workstation root')
    out = root / args.out
    (out / 'results').mkdir(parents=True, exist_ok=True)
    lock = (out / 'preparation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = root / args.candidates
    report_input = json.loads(source.read_text())
    rows = report_input['selection']['train'] + report_input['selection']['val']
    assert all(x['split'] in ('train', 'val') for x in rows)
    provenance = dict(candidates_sha256=digest(source), old_protocol_sha256=digest(root / OLD / 'pilot_manifest.json'),
                      seqres_sha256=digest(root / 'data/rcsb/derived_data/pdb_seqres.txt.gz'),
                      script_sha256=digest(Path(__file__)))
    config_path = out / 'inputs.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == provenance, 'Resume input/code changed'
    else:
        save(config_path, provenance)
    pdbs = {x['pdb'].lower() for x in rows}
    sequences = {}
    key = None
    with gzip.open(root / 'data/rcsb/derived_data/pdb_seqres.txt.gz', 'rt') as stream:
        for line in stream:
            if line.startswith('>'):
                token = line[1:].split()[0]
                pdb, chain = token.split('_', 1)
                key = pdb.lower() + '_' + chain if pdb.lower() in pdbs else None
                if key:
                    if key in sequences:
                        raise ValueError('Duplicate SEQRES key ' + key)
                    sequences[key] = ''
            elif key:
                sequences[key] += line.strip().upper()
    init_worker(root, sequences, args.offline)
    old = {(x['split'], x['lmdb_index']): x for x in json.loads((root / OLD / 'pilot_manifest.json').read_text())}
    results, pending = [], []
    for row in rows:
        key = (row['split'], row['lmdb_index'])
        path = out / 'results' / ('%s_%d.json' % key)
        if path.exists():
            value = json.loads(path.read_text())
            assert value['id'] == row['id']
            if value['status'] != 'failed' or not args.retry_failures:
                results.append(value)
                continue
            path.rename(path.with_suffix('.failed_%d.json' % time.time_ns()))
        if key in old:
            protocol = copy.deepcopy(old[key])
            assert protocol['id'] == row['id']
            protocol['cluster_pair'] = row['cluster_pair']
            value = dict(split=key[0], lmdb_index=key[1], id=row['id'], pdb=row['pdb'],
                         status='ready', protocol=add_sequence_identity(protocol),
                         reused_phase2=True, source_protocol_sha256=provenance['old_protocol_sha256'])
            save(path, value)
            results.append(value)
        else:
            pending.append(row)
    started = time.time()
    report = dict(status='running', host=platform.node(), python=platform.python_version(),
                  numpy=np.__version__, execution=args.execution, job_id=os.environ.get('SLURM_JOB_ID'),
                  workers=args.workers, records_requested=len(rows), reused_or_resumed=len(results),
                  new_records_pending=len(pending), test_records_read=0, test_labels_read=0,
                  started_at=started, provenance=provenance)
    save(out / 'report.json', report)
    print('PREPARATION', json.dumps(report), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(root, sequences, args.offline)) as executor:
        futures = {executor.submit(prepare, row): row for row in pending}
        for future in as_completed(futures):
            row = futures[future]
            value = future.result()
            save(out / 'results' / ('%s_%d.json' % (row['split'], row['lmdb_index'])), value)
            results.append(value)
            if len(results) % 25 == 0:
                progress = dict(completed=len(results), requested=len(rows),
                                counts=dict(Counter(x['status'] for x in results)), seconds=time.time()-started)
                save(out / 'progress.json', progress)
                print('PROGRESS', json.dumps(progress), flush=True)
    ready = [x['protocol'] for x in results if x['status'] == 'ready']
    ready.sort(key=lambda x: (x['split'], x['lmdb_index']))
    failures = [{k:v for k,v in x.items() if k != 'audit'} for x in results if x['status'] == 'failed']
    excluded = [{k:v for k,v in x.items() if k != 'audit'} for x in results if x['status'] == 'excluded']
    split_rows = {s: [x for x in ready if x['split'] == s] for s in ('train', 'val')}
    clusters = {s: {c for x in v for c in x['cluster_pair']} for s,v in split_rows.items()}
    assert not (clusters['train'] & clusters['val'])
    report.update(status='needs_retry' if failures else 'protocol_ready',
                  ready_by_split={s:len(v) for s,v in split_rows.items()},
                  records_ready=len(ready), failures=failures, exclusions=len(excluded),
                  exclusion_reasons=dict(Counter(reason for x in excluded for reason in x['reasons'])),
                  cluster_overlap_count=0, finished_at=time.time(), wall_seconds=time.time()-started)
    save(out / 'excluded.json', excluded)
    if not failures:
        for name, value in [('pilot_manifest.json', ready), ('train_manifest.json', split_rows['train']),
                            ('val_manifest.json', split_rows['val']), ('strict_val_manifest.json', split_rows['val'])]:
            save(out / name, value)
    save(out / 'report.json', report)
    print('REPORT', json.dumps(report), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
