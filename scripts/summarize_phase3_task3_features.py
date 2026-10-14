"""Audit completed extraction reports on the authorized CPU workstation.

This checks metadata and full manifest coverage. The Slurm training loader
independently hashes and validates the actual NPZ files on the cluster.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--array', default='276324')
    parser.add_argument('--protocol', type=Path,
                        default=Path('runs/phase3_task3_protocol/20260928/structure_v2'))
    parser.add_argument('--features', type=Path,
                        default=Path('runs/phase3_task3_features/structure_v1'))
    args = parser.parse_args()
    protocol = json.loads((args.protocol / 'report.json').read_text())
    manifest = args.protocol / 'proteins.json'
    if protocol['status'] != 'passed' or digest(manifest) != protocol['artifact_sha256']['proteins.json']:
        raise ValueError('Frozen protocol or manifest invalid')
    expected = {r['id']: r for r in json.loads(manifest.read_text())}
    if len(expected) != protocol['used_proteins']:
        raise ValueError('Manifest count mismatch')
    records, sources, backbones, configs, scripts = {}, [], set(), set(), set()
    for path in sorted(args.features.glob('job_*/report.json')):
        report = json.loads(path.read_text())
        if str(report.get('array_job_id')) != args.array:
            continue
        if report['status'] != 'passed' or report['failures']:
            raise ValueError('Extraction incomplete or failed: ' + str(path))
        if report['test_records_read'] or report['test_labels_read']:
            raise ValueError('Unexpected test access')
        if report['manifest_sha256'] != digest(manifest):
            raise ValueError('Extraction used a different manifest')
        if len(report['records']) != report['records_requested']:
            raise ValueError('Shard coverage mismatch')
        configs.add(report['encoder_config_sha256'])
        scripts.add(report['source_sha256'])
        backbones.add(json.dumps(report['checkpoint_sha256'], sort_keys=True))
        for row in report['records']:
            key = row['id']
            if key not in expected or key in records:
                raise ValueError('Unexpected or duplicate protein: ' + key)
            for field in ('pdb_sha256', 'sequence_sha256'):
                if row[field] != expected[key][field]:
                    raise ValueError('Protein provenance mismatch: ' + key)
            if row['length'] != len(expected[key]['sequence']):
                raise ValueError('Protein length mismatch: ' + key)
            if row['checkpoint_sha256'] != report['checkpoint_sha256'] or row['encoder_config_sha256'] != report['encoder_config_sha256']:
                raise ValueError('Cache encoder provenance mismatch: ' + key)
            records[key] = row
        sources.append(dict(path=str(path), sha256=digest(path), job_id=report['job_id'],
                            records=len(report['records']), seconds=report['wall_seconds'],
                            gpu=report['gpu'], peak_gpu_bytes=report['peak_gpu_bytes']))
    if len(sources) != 4 or set(records) != set(expected):
        raise ValueError('Need all four shards covering the complete frozen manifest')
    if len(backbones) != 1 or len(configs) != 1 or len(scripts) != 1:
        raise ValueError('Extraction provenance differs across shards')
    summary = dict(status='passed', array_job_id=args.array, records=len(records), failures=0,
                   protocol_sha256=digest(args.protocol / 'report.json'),
                   manifest_sha256=digest(manifest), sources=sources,
                   backbone_sha256=json.loads(next(iter(backbones))),
                   encoder_config_sha256=next(iter(configs)), source_sha256=next(iter(scripts)),
                   audit_script_sha256=digest(__file__), test_records_read=0, test_labels_read=0,
                   scope='Full metadata coverage audit; actual NPZ content is checked by the Slurm training loader',
                   feature_hashes={k: records[k]['cache_sha256'] for k in sorted(records)})
    output = args.features / 'report.json'
    with output.open('x') as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(status=summary['status'], records=len(records), report=str(output))))


if __name__ == '__main__':
    main()
