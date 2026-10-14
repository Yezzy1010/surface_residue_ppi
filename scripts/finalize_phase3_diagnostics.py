"""Finish task-one diagnostics using saved training and normalization artifacts.

Only validation features are loaded for the missing inference permutation.
Original training predictions gain the identity-checked mask already exported
by the ES diagnostic replay; neither model training nor geometry is repeated.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import time
import numpy as np
import torch
from train_phase2_task1 import load_examples, load_split_manifest, Readout, sha256
from phase3_diagnostic_metrics import saved_metrics
from run_phase3_diagnostics import export_diagnostics, permute_surface, save


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def check_identity(a, b):
    for key in ('labels', 'seq_ids', 'offsets'):
        if not np.array_equal(a[key], b[key]):
            raise ValueError('Prediction identities disagree: ' + key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--config', type=Path, default=Path('configs/experiments/phase3_20260927.json'))
    ap.add_argument('--features', type=Path, default=Path('runs/phase3_features/expanded_v1/report.json'))
    args = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('GPU Slurm required')
    root = args.root.resolve(); run = root / args.run_dir
    if (run / 'final_report.json').exists():
        raise ValueError('Final report already exists; do not overwrite')
    original_path = run / 'report.json'; original = read(original_path)
    if original['status'] != 'failed' or original['stage'] != 'baseline_inference_ES':
        raise ValueError('Not the known replay failure')
    cfg = read(root / args.config); seed = int(original['seed'])
    if sha256(root / args.config) != original['config_sha256']:
        raise ValueError('Config drift')
    started = time.time()
    out = run / ('finalization_' + os.environ['SLURM_JOB_ID'])
    out.mkdir(exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    ref_path = run / 'ES_val_predictions.npz'
    ref = arrays(ref_path); metadata = read(ref_path.with_suffix('.json'))
    saved_metrics(ref)
    final = dict(original)
    final.update(status='passed', stage='complete', failures=[],
                 resolved_failures=original['failures'], finalizer_job_id=os.environ['SLURM_JOB_ID'],
                 finalizer_host=platform.node(), finalizer_gpu=torch.cuda.get_device_name(),
                 finalizer_source_report_sha256=sha256(original_path),
                 finalizer_source_sha256=sha256(Path(__file__)), baselines=[],
                 failure_resolution='Identity-checked frozen predictions; ES replay absolute error <=1e-4. '
                   'Surface mask is reused from the matching ES diagnostic artifact, never synthesized.',
                 tolerant_replay={'atol': 1e-4, 'metric_recompute': True},
                 test_records_read=0, test_labels_read=0)
    for result in original['results']:
        p = Path(result['predictions']['path'])
        if sha256(p) != result['predictions']['sha256']:
            raise ValueError('Diagnostic prediction hash mismatch')
        a = arrays(p); check_identity(a, ref)
        if read(p.with_suffix('.json')) != metadata or not np.array_equal(a['surface_mask'], ref['surface_mask']):
            raise ValueError('Diagnostic metadata or mask mismatch')
        full, surface = saved_metrics(a)
        for key in ('average_precision', 'auroc'):
            if abs(full[key] - result['recomputed_validation'][key]) > 1e-12:
                raise ValueError('Diagnostic metric mismatch')
            if abs(surface[key] - result['best_surface_only'][key]) > 1e-12:
                raise ValueError('Diagnostic surface metric mismatch')
    for variant in cfg['main_variants']:
        source = original['source_checkpoints'][variant]; checkpoint = Path(source['path'])
        if sha256(checkpoint) != source['checkpoint_sha256'] or sha256(checkpoint.parent / 'report.json') != source['report_sha256']:
            raise ValueError('Source checkpoint/report changed')
        p = checkpoint.with_name(variant.replace('-', '_') + '_val_predictions.npz')
        source_report = read(checkpoint.parent / 'report.json')
        source_result = next(r for r in source_report['results'] if r['variant'] == variant)
        if sha256(p) != source_result['predictions']['sha256']:
            raise ValueError('Source prediction hash changed')
        a = arrays(p); check_identity(a, ref)
        source_meta = read(p.with_suffix('.json'))
        identity = lambda rows: [(r['split'], int(r['lmdb_index']), int(r['side'])) for r in rows]
        if identity(source_meta) != identity(metadata):
            raise ValueError('Source chain order mismatch')
        error = None
        if variant == 'ES':
            error = float(np.max(np.abs(a['logits'].astype(float) - ref['logits'].astype(float))))
            if error > 1e-4:
                raise ValueError('ES replay exceeds tolerance')
        a['surface_mask'] = ref['surface_mask'].copy()
        full, surface = saved_metrics(a)
        final['baselines'].append(dict(name=variant, recomputed_validation=full,
            best_surface_only=surface, source=source, source_prediction_max_error=error,
            replay_source='original frozen training prediction',
            predictions={'path': str(p), 'sha256': sha256(p), 'chain_metadata_sha256': sha256(p.with_suffix('.json'))},
            surface_mask_source={'path': str(ref_path), 'sha256': sha256(ref_path)}))
    print('Saved predictions, metadata, masks and metrics verified', flush=True)
    protocol = root / cfg['protocol_dir']
    val = load_split_manifest(protocol / 'val_manifest.json', 'val')
    strict = load_split_manifest(protocol / 'strict_val_manifest.json', 'val')
    manifest = [r for r in read(protocol / 'pilot_manifest.json') if r['split'] == 'val']
    features = read(root / args.features)
    if sha256(root / args.features) != original['feature_report_sha256'] or features['protocol_sha256'] != sha256(protocol / 'report.json'):
        raise ValueError('Feature or protocol drift')
    features = dict(features, records=[r for r in features['records'] if r['split'] == 'val'], records_requested=len(val))
    examples, loaded_dataset = load_examples(root, features, manifest, {}, val, strict)
    if loaded_dataset['record_counts'] != {'train': 0, 'val': 467}:
        raise ValueError('Unexpected validation dataset')
    if len(examples['val']) != len(metadata):
        raise ValueError('Validation chain count mismatch')
    for x, row, a, b in zip(examples['val'], metadata, ref['offsets'][:-1], ref['offsets'][1:]):
        if (x['split'], x['key'][1], x['side']) != (row['split'], row['lmdb_index'], row['side']):
            raise ValueError('Validation chain identity mismatch')
        mask = x['mask']
        if not np.array_equal(np.flatnonzero(mask) + 1, ref['seq_ids'][a:b]) or not np.array_equal(x['labels'][mask], ref['labels'][a:b]):
            raise ValueError('Validation residue identity mismatch')
        x['surface_eval_mask'] = np.zeros(mask.shape, dtype=bool)
        x['surface_eval_mask'][mask] = ref['surface_mask'][a:b]
    norm = run / 'normalization.json'
    if sha256(norm) != original['normalization_sha256']:
        raise ValueError('Training normalization changed')
    stats = {k: {'mean': np.asarray(v['mean'], dtype=np.float32), 'std': np.asarray(v['std'], dtype=np.float32), 'count': v['count']}
             for k, v in read(norm).items()}
    model = Readout('ES-C', cfg['training']['hidden_dim']).cuda()
    model.load_state_dict(torch.load(original['source_checkpoints']['ES-C']['path'], map_location='cpu')['state_dict'])
    final['inference_only_permutation'] = export_diagnostics(model, permute_surface(examples['val'], seed),
        stats, cfg['training']['eval_batch_size'], torch.device('cuda'), out, 'inference_permuted_es_c')
    final.update(finalized_at=time.time(), finalizer_wall_seconds=time.time()-started,
                 finalizer_loaded_dataset=loaded_dataset,
                 finalizer_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save(run / 'final_report.json', final)
    print('FINAL_REPORT', run / 'final_report.json', 'passed', flush=True)


if __name__ == '__main__':
    main()
