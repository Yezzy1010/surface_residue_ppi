"""Assemble Phase 3 feature references; training validates the actual arrays.

No models or arrays are loaded here. Run this metadata audit on the authorized
workstation with copied reports. Feature NPZs remain on the cluster.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


BACKBONE_KEYS = ('dmasif_sha256', 'gearnet_sha256', 'esm_sha256', 'esm_variant',
                 'gearnet_dim', 'dmasif_branch')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def keyed(rows):
    result = {}
    for row in rows:
        key = (row['split'], int(row['lmdb_index']))
        if key[0] not in ('train', 'val') or key in result:
            raise ValueError('Forbidden split or duplicate record: ' + str(key))
        result[key] = row
    return result


def validate_side(side, expected, index, reused):
    if (side['side'] != index or side['seqres_key'] != expected['seqres_key']
            or side['sequence_length'] != expected['sequence_length']
            or side['common_valid'] != len(expected['common_valid'])
            or side['gearnet_seen'] != len(expected['common_valid'])
            or side['surface_points'] <= 0):
        raise ValueError('Feature side does not match frozen sequence/common mask')
    if not reused and side.get('sequence_sha256') != expected['sequence_sha256']:
        raise ValueError('New cache sequence digest mismatch')
    path = Path(side['cache'])
    if path.is_absolute() or '..' in path.parts or path.suffix != '.npz':
        raise ValueError('Invalid cache path')


def assemble(root, protocol_path, reuse_path, shard_paths):
    protocol = read(protocol_path)
    if (protocol.get('status') != 'protocol_ready' or protocol.get('failures')
            or protocol.get('test_records_read') != 0 or protocol.get('test_labels_read') != 0):
        raise ValueError('Protocol is not finalized')
    manifest = keyed(read(protocol_path.parent / 'pilot_manifest.json'))
    if len(manifest) != protocol['records_ready']:
        raise ValueError('Protocol count mismatch')
    clusters = {split: {c for k, row in manifest.items() if k[0] == split
                        for c in row['cluster_pair']} for split in ('train', 'val')}
    if clusters['train'] & clusters['val']:
        raise ValueError('Train/validation cluster overlap')
    old = read(reuse_path)
    if old['status'] != 'passed' or old['failures'] or old['test_records_read'] != 0:
        raise ValueError('Invalid reuse report')
    old_protocol = root / old['protocol']
    if digest(old_protocol) != old['protocol_sha256']:
        raise ValueError('Old feature protocol digest mismatch')
    old_manifest = keyed(read(old_protocol.parent / 'pilot_manifest.json'))
    result = keyed(old['records'])
    if set(result) != set(old_manifest) or not set(result) <= set(manifest):
        raise ValueError('Original cache set must be preserved completely')
    for key, row in result.items():
        expected = manifest[key]
        original = old_manifest[key]
        if row['id'] != expected['id'] or original['id'] != expected['id']:
            raise ValueError('Reused record identity mismatch')
        if len(row['sides']) != 2 or len(expected['sides']) != 2:
            raise ValueError('Expected two monomers')
        for i, side in enumerate(row['sides']):
            if original['sides'][i]['common_valid'] != expected['sides'][i]['common_valid']:
                raise ValueError('Reused residue identities/labels have changed')
            validate_side(side, expected['sides'][i], i, True)
    reused_keys = set(result)
    sources = [dict(path=str(reuse_path.relative_to(root)), sha256=digest(reuse_path),
                    job_id=old['job_id'], role='unchanged_phase2_cache')]
    protocol_hash = digest(protocol_path)
    source_hash = None
    seen_shards = set()
    for path in shard_paths:
        report = read(path)
        if (report['status'] != 'passed' or report['failures']
                or report['test_records_read'] != 0
                or report['protocol_sha256'] != protocol_hash
                or not report.get('labels_outside_encoder_inputs')):
            raise ValueError('Incomplete/incompatible shard: ' + str(path))
        if any(report.get(k) != old.get(k) for k in BACKBONE_KEYS):
            raise ValueError('Backbone checkpoint/configuration changed')
        if source_hash is not None and report['source_sha256'] != source_hash:
            raise ValueError('Shard extractor source differs')
        source_hash = report['source_sha256']
        shard = (report['shard_count'], report['shard_index'])
        if shard in seen_shards:
            raise ValueError('Duplicate shard')
        seen_shards.add(shard)
        new_rows = keyed(report['records'])
        if len(new_rows) != report['records_requested']:
            raise ValueError('Shard record count mismatch')
        for key, row in new_rows.items():
            if key not in manifest or key in result or row['id'] != manifest[key]['id']:
                raise ValueError('Unexpected/duplicate feature record: ' + str(key))
            if len(row['sides']) != 2:
                raise ValueError('Expected two monomers')
            for i, side in enumerate(row['sides']):
                validate_side(side, manifest[key]['sides'][i], i, False)
            result[key] = row
        sources.append(dict(path=str(path.relative_to(root)), sha256=digest(path),
                            job_id=report['job_id'], role='new_phase3_cache'))
    if set(result) != set(manifest):
        raise ValueError('Missing feature records: ' + str(len(set(manifest) - set(result))))
    return dict(status='passed', validation_scope='complete manifest and cache metadata; no NPZ arrays loaded',
                cache_arrays_verified=False, array_validation_gate='train_phase3_task1.load_examples before training',
                protocol=str(protocol_path.relative_to(root)), protocol_sha256=protocol_hash,
                manifest_sha256=digest(protocol_path.parent / 'pilot_manifest.json'),
                records_requested=len(result), records=[result[k] for k in sorted(result)], failures=[],
                reused_records=len(reused_keys), new_records=len(result)-len(reused_keys),
                record_counts=dict(Counter(k[0] for k in result)), cluster_overlap_count=0,
                sources=sources, test_records_read=0, test_labels_read=0,
                labels_outside_encoder_inputs=True, execution='metadata-assembly',
                **{k: old[k] for k in BACKBONE_KEYS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--reuse', type=Path, required=True)
    parser.add_argument('--shards', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if root != Path('/home/zy/projects/Surface-residue'):
        raise RuntimeError('Run metadata assembly on the authorized workstation')
    result = assemble(root, root / args.protocol, root / args.reuse, [root / p for p in args.shards])
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
