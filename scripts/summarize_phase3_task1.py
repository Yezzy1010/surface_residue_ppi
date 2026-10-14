"""Recompute saved validation predictions and paired multi-seed statistics.

Run on the authorized CPU workstation after downloading reports and predictions.
Does not load a model or inspect any test sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from phase2_training_metrics import binary_metrics, macro_metrics
from surface_residue_ppi.paired_statistics import component_ids, paired_cluster_bootstrap


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_path(root, path):
    path = Path(path)
    if path.is_absolute():
        path = path.relative_to('/datapool/home/zhangye/projects/surface-residue-ppi')
    if '..' in path.parts:
        raise ValueError('Invalid artifact path')
    return root / path


def predictions(root, result, expected):
    info = result['predictions']
    path = local_path(root, info['path'])
    if digest(path) != info['sha256'] or digest(path.with_suffix('.json')) != info['chain_metadata_sha256']:
        raise ValueError('Prediction artifact digest mismatch')
    metadata = read(path.with_suffix('.json'))
    metrics, identities = [], []
    with np.load(path, allow_pickle=False) as a:
        labels, scores, positions, offsets = (a[k] for k in ('labels','logits','seq_ids','offsets'))
        if (len(offsets) != len(metadata)+1 or offsets[0] != 0 or offsets[-1] != len(labels)
                or labels.shape != scores.shape or labels.shape != positions.shape
                or not np.isfinite(scores).all() or np.any(np.diff(offsets) <= 0)):
            raise ValueError('Invalid prediction layout')
        for row, begin, end in zip(metadata, offsets[:-1], offsets[1:]):
            key = (row['split'], int(row['lmdb_index']), int(row['side']))
            if key not in expected:
                raise ValueError('Unexpected validation chain')
            residues = sorted(expected[key]['common_valid'], key=lambda x:x['seq_id'])
            if (not np.array_equal(positions[begin:end], [r['seq_id'] for r in residues])
                    or not np.array_equal(labels[begin:end], [int(r['positive']) for r in residues])):
                raise ValueError('Prediction residue identities/labels do not match frozen validation')
            identities.append(key)
            metrics.append(binary_metrics(labels[begin:end], scores[begin:end]))
        if len(set(identities)) != len(identities) or set(identities) != set(expected):
            raise ValueError('Missing or duplicate validation predictions')
        macro = macro_metrics(metrics, binary_metrics(labels, scores))
    if any(abs(macro[k] - result['recomputed_validation'][k]) > 1e-12 for k in macro):
        raise ValueError('Saved prediction metrics differ from training report')
    by_key = dict(zip(identities, metrics))
    return {m:np.array([by_key[k][m] for k in sorted(expected)]) for m in ('average_precision','auroc')}, macro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/experiments/phase3_20260927.json'))
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path('.').resolve()
    if root != Path('/home/zy/projects/Surface-residue'):
        raise RuntimeError('Use the authorized CPU workstation')
    cfg = read(args.config)
    protocol = root / cfg['protocol_dir']
    val = sorted(read(protocol / 'val_manifest.json'), key=lambda r:r['lmdb_index'])
    expected = {(r['split'], int(r['lmdb_index']), side):s
                for r in val for side,s in enumerate(r['sides'])}
    if any(k[0] != 'val' for k in expected):
        raise ValueError('Only validation is allowed')
    components = component_ids(val)
    component_map = {(r['split'], int(r['lmdb_index']), side):int(c)
                     for r,c in zip(val,components) for side in (0,1)}
    chain_components = np.array([component_map[k] for k in sorted(expected)])
    arrays, summaries, sources, feature_hashes = {}, [], [], set()
    for path in args.reports:
        report = read(path)
        seed = report['seed']
        if (report['status'] != 'passed' or report['failures'] or seed in arrays
                or report['test_records_read'] != 0 or report['test_labels_read'] != 0
                or report['config_sha256'] != digest(args.config)
                or report['manifest_sha256'] != digest(protocol / 'pilot_manifest.json')):
            raise ValueError('Invalid or incompatible training report')
        feature_hashes.add(report['feature_report_sha256'])
        arrays[seed] = {}
        for result in report['results']:
            variant = result['variant']
            if variant in arrays[seed]:
                raise ValueError('Duplicate variant')
            metrics, macro = predictions(root, result, expected)
            arrays[seed][variant] = metrics
            summaries.append(dict(seed=seed, variant=variant, metrics=macro,
                                  best_epoch=result['best_epoch'], seconds=result['seconds']))
        if set(arrays[seed]) != set(cfg['main_variants']):
            raise ValueError('Incomplete variant set')
        sources.append(dict(path=str(path), sha256=digest(path)))
    if set(arrays) != set(cfg['seeds']) or len(feature_hashes) != 1:
        raise ValueError('Expected all predefined seeds and identical feature reports')
    aggregate, comparisons = {}, {}
    stat = cfg['statistics']
    for metric in ('average_precision','auroc'):
        aggregate[metric], comparisons[metric] = {}, {}
        for variant in cfg['main_variants']:
            values = [float(np.nanmean(arrays[s][variant][metric])) for s in cfg['seeds']]
            aggregate[metric][variant] = dict(per_seed=values, mean=float(np.mean(values)),
                                            sample_std=float(np.std(values, ddof=1)))
        for base, method in (('ES','ES-C'), ('ES','ES-B'), ('ES-B','ES-C')):
            left = np.stack([arrays[s][base][metric] for s in cfg['seeds']])
            right = np.stack([arrays[s][method][metric] for s in cfg['seeds']])
            if not np.all(np.isfinite(left) == np.isfinite(left[0])) or not np.array_equal(np.isfinite(left),np.isfinite(right)):
                raise ValueError('Defined metric sets differ across seeds/methods')
            per_seed = [float(np.nanmean(r-l)) for l,r in zip(left,right)]
            interval = paired_cluster_bootstrap(left.mean(axis=0), right.mean(axis=0), chain_components,
                replicates=stat['bootstrap_replicates'], seed=stat['bootstrap_seed'], confidence=stat['confidence_level'])
            comparisons[metric][method+' minus '+base] = dict(per_seed=per_seed,
                mean=float(np.mean(per_seed)), sample_std=float(np.std(per_seed, ddof=1)),
                validation_component_interval=interval)
    report = dict(status='passed', scope='developmental validation; not formal test',
                  config_sha256=digest(args.config), sources=sources, seeds=cfg['seeds'],
                  validation_complexes=len(val), validation_chains=len(expected),
                  components=len(set(components.tolist())),
                  component_complex_sizes=sorted(np.bincount(components).tolist(),reverse=True),
                  results=summaries, aggregate=aggregate, comparisons=comparisons,
                  interval_limits='Conditional on the three trained seeds and selected checkpoints; '
                    'component bootstrap does not correct validation model selection or estimate seed-population uncertainty',
                  test_records_read=0, test_labels_read=0)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(aggregate=aggregate,comparisons=comparisons),indent=2))


if __name__ == '__main__':
    main()
