"""Run the pre-specified Phase 3 diagnostic controls on frozen train/val caches.

Controls are trained with the same budget and three seeds as the main comparison:
capacity-matched ES, ES plus monomer surface exposure/depth proxies, and an
ES-C run with within-chain point-feature permutation during both train and
validation. Existing ES-C checkpoints are also evaluated after inference-only
permutation. Test records and labels are never opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import time
import traceback

import numpy as np
import torch
from torch import nn

from train_phase2_task1 import (Readout, load_examples, load_split_manifest,
                                fit_statistics, make_batch, sha256)
from phase2_training_metrics import binary_metrics, macro_metrics


DIAGNOSTIC_VARIANTS = ('capacity_es', 'exposure_depth_es', 'permuted_es_c')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seeded_model(name, seed, hidden=256):
    set_seed(seed)
    return model_for(name, hidden)


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def stable_seed(seed, example):
    token = f'{seed}:{example["split"]}:{example["key"][1]}:{example["side"]}'
    return int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)


def attach_geometry(examples, progress=None):
    """Add monomer-only exposure/depth proxies and a surface-valid mask.

    Exposure is log1p of dMaSIF surface points assigned to a residue by the
    nearest heavy atom. Depth is the minimum distance from that residue's atom
    centroid to its assigned surface points; residues without an assigned
    point receive the chain's largest defined depth plus one Angstrom.
    """
    for index, example in enumerate(examples):
        with np.load(example['cache'], allow_pickle=False) as archive:
            owner = archive['point_owner'].astype(np.int64, copy=False)
            surface_xyz = archive['surface_xyz'].astype(np.float32, copy=False)
            atom_xyz = archive['atom_xyz'].astype(np.float32, copy=False)
            atom_to_seq = archive['atom_to_seq'].astype(np.int64, copy=False)
        length = len(example['labels'])
        if (surface_xyz.shape != (len(owner), 3) or atom_xyz.shape != (len(atom_to_seq), 3)
                or owner.ndim != 1 or len(owner) != len(surface_xyz)
                or atom_to_seq.ndim != 1 or len(atom_to_seq) != len(atom_xyz)
                or np.any(owner < 0) or np.any(owner >= length)
                or np.any(atom_to_seq < 0) or np.any(atom_to_seq >= length)
                or not np.isfinite(surface_xyz).all() or not np.isfinite(atom_xyz).all()):
            raise ValueError(f'Invalid geometry arrays at {example["cache"]}')
        counts = np.bincount(owner, minlength=length).astype(np.float32)
        if np.any(np.bincount(atom_to_seq, minlength=length)[owner] == 0):
            raise ValueError('Surface owner has no residue atoms')
        depth = np.full(length, np.nan, dtype=np.float32)
        for residue in np.unique(owner):
            atom_mask = atom_to_seq == residue
            point_mask = owner == residue
            if not atom_mask.any() or not point_mask.any():
                continue
            center = atom_xyz[atom_mask].mean(axis=0)
            depth[residue] = np.linalg.norm(surface_xyz[point_mask] - center, axis=1).min()
        defined = np.isfinite(depth)
        if not defined.any():
            raise ValueError(f'No surface depth values at {example["cache"]}')
        fill = float(depth[defined].max() + 1.0)
        depth[~defined] = fill
        example['exposure_depth'] = np.column_stack((np.log1p(counts), depth)).astype(np.float32)
        example['surface_eval_mask'] = example['mask'] & (counts > 0)
        if progress and ((index + 1) % 1000 == 0 or index + 1 == len(examples)):
            progress(geometry_chains_done=index + 1, geometry_chains_total=len(examples))


def permute_surface(examples, seed):
    result = []
    for example in examples:
        copy = dict(example)
        rng = np.random.default_rng(stable_seed(seed, example))
        order = rng.permutation(len(example['surface']))
        copy['surface'] = example['surface'][order].copy()
        result.append(copy)
    return result


def fit_exposure_statistics(examples):
    values = np.concatenate([x['exposure_depth'][x['mask']] for x in examples], axis=0).astype(np.float64)
    mean = values.mean(axis=0)
    std = np.sqrt(np.maximum((values * values).mean(axis=0) - mean * mean, 1e-8))
    return dict(mean=mean.astype(np.float32), std=std.astype(np.float32), count=int(len(values)))


def make_diag_batch(examples, stats, device, include_exposure=False):
    batch = make_batch(examples, stats, device)
    if include_exposure:
        values = np.concatenate([x['exposure_depth'] for x in examples], axis=0)
        values = (values - stats['exposure_depth']['mean']) / stats['exposure_depth']['std']
        batch['exposure_depth'] = torch.from_numpy(values.astype(np.float32, copy=False)).to(device)
    return batch


class CapacityMatchedES(nn.Module):
    """ES with a parameter-matched residue-only residual branch.

    The 256->528->256 branch has 271,120 parameters; 16 shared padding
    group biases bring added capacity to exactly the ES-C surface adapter's
    271,136 parameters. Each bias affects a distinct group of 16 channels.
    """
    def __init__(self, hidden=256):
        super().__init__()
        if hidden != 256:
            raise ValueError('Capacity control is defined for hidden_dim=256')
        self.base = Readout('ES', hidden)
        self.residual = nn.Sequential(nn.Linear(hidden, 528), nn.GELU(), nn.Linear(528, hidden))
        self.capacity_padding = nn.Parameter(torch.zeros(16))

    def forward(self, batch):
        h = self.base.project(torch.cat((batch['esm'], batch['gearnet']), dim=-1))
        h = h + self.residual(h) + self.capacity_padding.repeat_interleave(16)
        return self.base.head(h).squeeze(-1)


class ExposureDepthES(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.base = Readout('ES', hidden)
        self.exposure_projection = nn.Linear(2, hidden, bias=False)

    def forward(self, batch):
        h = self.base.project(torch.cat((batch['esm'], batch['gearnet']), dim=-1))
        h = h + self.exposure_projection(batch['exposure_depth'])
        return self.base.head(h).squeeze(-1)


def model_for(variant, hidden=256):
    if variant == 'capacity_es':
        return CapacityMatchedES(hidden)
    if variant == 'exposure_depth_es':
        return ExposureDepthES(hidden)
    if variant in ('permuted_es_c', 'es_c'):
        return Readout('ES-C', hidden)
    if variant in ('ES', 'ES-B', 'ES-C'):
        return Readout(variant, hidden)
    raise ValueError(variant)


@torch.no_grad()
def evaluate_diag(model, examples, stats, batch_size, device, include_exposure=False, surface_only=False):
    model.eval()
    instance_metrics, labels, scores = [], [], []
    excluded = 0
    for start in range(0, len(examples), batch_size):
        selected = examples[start:start + batch_size]
        batch = make_diag_batch(selected, stats, device, include_exposure)
        logits = model(batch).cpu().numpy()
        offset = 0
        for example in selected:
            length = len(example['labels'])
            effective = example['mask'].copy()
            if surface_only:
                effective &= example['surface_eval_mask']
            y = example['labels'][effective].astype(np.int8)
            s = logits[offset:offset + length][effective]
            offset += length
            if not len(y):
                excluded += 1
                continue
            row = binary_metrics(y, s)
            instance_metrics.append(row)
            labels.append(y)
            scores.append(s)
    if not instance_metrics:
        raise ValueError('No nonempty evaluation chains')
    result = macro_metrics(instance_metrics, binary_metrics(np.concatenate(labels), np.concatenate(scores)))
    result['surface_only'] = surface_only
    result['excluded_empty_chains'] = excluded
    return result


@torch.no_grad()
def export_diagnostics(model, examples, stats, batch_size, device, output, name, include_exposure=False):
    model.eval()
    labels, scores, positions, surface_masks, metadata = [], [], [], [], []
    offsets = [0]
    for start in range(0, len(examples), batch_size):
        selected = examples[start:start + batch_size]
        logits = model(make_diag_batch(selected, stats, device, include_exposure)).cpu().numpy()
        offset = 0
        for example in selected:
            mask = example['mask']
            length = len(mask)
            labels.append(example['labels'][mask].astype(np.int8))
            scores.append(logits[offset:offset + length][mask])
            positions.append(np.flatnonzero(mask) + 1)
            surface_masks.append(example['surface_eval_mask'][mask])
            offsets.append(offsets[-1] + int(mask.sum()))
            metadata.append(dict(split=example['split'], lmdb_index=example['key'][1], side=example['side']))
            offset += length
    path = output / (name + '_val_predictions.npz')
    np.savez_compressed(path, labels=np.concatenate(labels), logits=np.concatenate(scores),
                        seq_ids=np.concatenate(positions), offsets=np.array(offsets, dtype=np.int64),
                        surface_mask=np.concatenate(surface_masks))
    save(path.with_suffix('.json'), metadata)
    from phase3_diagnostic_metrics import saved_metrics
    # Re-read exported scores; both subsets use exactly the same forward pass.
    with np.load(path, allow_pickle=False) as archive:
        full, surface = saved_metrics(archive)
    return dict(name=name, recomputed_validation=full, best_surface_only=surface,
                predictions=dict(path=str(path), sha256=sha256(path),
                                 chain_metadata_sha256=sha256(path.with_suffix('.json'))))


def train_one(train_examples, val_examples, stats, cfg, seed, device, include_exposure,
              output, name, pos_weight):
    model = seeded_model(name, seed, cfg['hidden_dim']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device), reduction='none')
    best_ap, best_state, best_metrics, best_epoch, stale = -float('inf'), None, None, 0, 0
    history = []
    started = time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        order = list(range(len(train_examples)))
        random.shuffle(order)
        total, count = 0.0, 0
        for start in range(0, len(order), cfg['batch_size']):
            selected = [train_examples[i] for i in order[start:start + cfg['batch_size']]]
            batch = make_diag_batch(selected, stats, device, include_exposure)
            loss = criterion(model(batch), batch['labels'])[batch['mask']].mean()
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite loss for {name}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            valid = int(batch['mask'].sum())
            total += float(loss.detach()) * valid
            count += valid
        metrics = evaluate_diag(model, val_examples, stats, cfg['eval_batch_size'], device, include_exposure)
        history.append(dict(epoch=epoch, train_weighted_bce=total / count, validation=metrics))
        print('EPOCH', name, seed, json.dumps(history[-1]), flush=True)
        save(output / (name + '_history.json'), history)
        score = metrics['average_precision']
        if score > best_ap + cfg['min_delta']:
            best_ap, best_metrics, best_epoch, stale = score, metrics, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= cfg['patience']:
            break
    if best_state is None:
        raise ValueError(f'No checkpoint for {name}')
    model.load_state_dict(best_state)
    checkpoint = output / f'{name}.pt'
    torch.save(dict(name=name, seed=seed, epoch=best_epoch, state_dict=best_state), checkpoint)
    exported = export_diagnostics(model, val_examples, stats, cfg['eval_batch_size'], device,
                                  output, name, include_exposure)
    return dict(**exported, seed=seed, best_epoch=best_epoch, epochs_run=len(history),
                trainable_parameters=sum(p.numel() for p in model.parameters()),
                best_validation=best_metrics,
                checkpoint_sha256=sha256(checkpoint), history=history,
                seconds=time.time() - started)


def checkpoint_for(root, seed, variant='ES-C'):
    filename = variant.replace('-', '_') + '.pt'
    candidates = sorted((root / 'runs/phase3_task1_training').glob(f'*_{seed}/{filename}'))
    if len(candidates) != 1:
        raise ValueError(f'Expected one {variant} checkpoint for seed {seed}, found {candidates}')
    report = candidates[0].parent / 'report.json'
    data = json.loads(report.read_text())
    if (data.get('status') != 'passed' or data.get('test_records_read') != 0
            or data.get('test_labels_read') != 0):
        raise ValueError('Invalid source checkpoint report')
    result = next(x for x in data['results'] if x['variant'] == variant)
    if sha256(candidates[0]) != result['checkpoint_sha256']:
        raise ValueError('Source checkpoint hash mismatch')
    return candidates[0], sha256(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--features', type=Path, default=Path('runs/phase3_features/expanded_v1/report.json'))
    parser.add_argument('--config', type=Path, default=Path('configs/experiments/phase3_20260927.json'))
    parser.add_argument('--seed-index', type=int, required=True)
    parser.add_argument('--smoke', action='store_true', help='Small GPU gradient/training/export regression check')
    args = parser.parse_args()
    root = args.root.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('GPU Slurm required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = json.loads((root / args.config).read_text())
    seed = cfg['seeds'][args.seed_index]
    protocol = root / cfg['protocol_dir']
    manifest = json.loads((protocol / 'pilot_manifest.json').read_text())
    splits = {s: load_split_manifest(protocol / (s + '_manifest.json'), s) for s in ('train', 'val')}
    strict = load_split_manifest(protocol / 'strict_val_manifest.json', 'val')
    feature_report = json.loads((root / args.features).read_text())
    if feature_report.get('protocol_sha256') != sha256(protocol / 'report.json'):
        raise ValueError('Feature report belongs to a different protocol')
    if args.smoke:
        # The full 5,328-record geometry smoke already passed in job 276273.
        # Only test the changed initialization/training/export paths here.
        manifest = [r for split in ('train', 'val') for r in
                    sorted([r for r in manifest if r['split'] == split], key=lambda r:r['lmdb_index'])[:8]]
        keys = {(r['split'], int(r['lmdb_index'])) for r in manifest}
        splits = {s:{k:v for k,v in values.items() if k in keys} for s,values in splits.items()}
        strict = {k:v for k,v in strict.items() if k in keys}
        feature_report['records'] = [r for r in feature_report['records'] if (r['split'], int(r['lmdb_index'])) in keys]
        feature_report['records_requested'] = len(manifest)
    job = os.environ['SLURM_JOB_ID']
    output = root / 'runs/phase3_diagnostics' / f'{job}_{seed}'
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', run_kind='smoke' if args.smoke else 'diagnostics',
                  job_id=job, array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                  array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'), seed=seed,
                  host=platform.node(), gpu=torch.cuda.get_device_name(),
                  partition=os.environ.get('SLURM_JOB_PARTITION'), started_at=time.time(),
                  config_sha256=sha256(root / args.config), feature_report_sha256=sha256(root / args.features),
                  manifest_sha256=sha256(protocol / 'pilot_manifest.json'), test_records_read=0,
                  test_labels_read=0, frozen_backbones=True, tf32=False, results=[], baselines=[], failures=[],
                  initialization='seed set before model construction for each variant',
                  permutation='one fixed within-chain point-feature row permutation per chain and seed; '
                              'same permutation in every epoch; coordinates, neighbors, labels and masks unchanged',
                  diagnostic_spec=dict(capacity='256->528->256 residue residual MLP plus 16 distinct group biases',
                      exposure='log1p(assigned point count)', depth='min assigned-point distance to residue atom centroid; '
                          'no-point depth=max defined depth+1 Angstrom', surface_subset='common mask AND assigned point count>0',
                      geometry_limit='Sampling-dependent geometry proxies, not physical SASA or exact residue depth'))
    source_files = ['scripts/run_phase3_diagnostics.py', 'scripts/phase3_diagnostic_metrics.py',
                    'scripts/train_phase2_task1.py', 'scripts/phase2_training_metrics.py',
                    'src/surface_residue_ppi/adapters.py']
    report['source_sha256'] = {f:sha256(root / f) for f in source_files}
    for filename in source_files:
        dest = output / 'source' / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / filename, dest)
    shutil.copy2(root / args.config, output / 'config.json')

    def progress(**updates):
        report.update(updates, updated_at=time.time())
        save(output / 'report.json', report)
        print('PROGRESS', json.dumps(updates), flush=True)

    try:
        torch.set_num_threads(4)
        sources = {}
        for variant in cfg['main_variants']:
            path, report_hash = checkpoint_for(root, seed, variant)
            main_report = json.loads((path.parent / 'report.json').read_text())
            for key in ('config_sha256', 'feature_report_sha256', 'manifest_sha256'):
                if main_report[key] != report[key]:
                    raise ValueError('Source checkpoint protocol mismatch: ' + key)
            sources[variant] = dict(path=str(path), report_sha256=report_hash, checkpoint_sha256=sha256(path))
        report['source_checkpoints'] = sources
        progress(stage='loading_frozen_train_val')
        examples, dataset = load_examples(root, feature_report, manifest, splits['train'], splits['val'], strict)
        if not args.smoke and dataset['record_counts'] != dict(train=4861, val=467):
            raise ValueError('Diagnostic dataset differs from frozen Phase 3 dataset')
        report['dataset'] = dataset
        for split in ('train', 'val'):
            progress(stage='geometry_' + split)
            attach_geometry(examples[split], progress)
        progress(stage='normalization_train_only')
        base_stats = fit_statistics(examples['train'])
        base_stats['exposure_depth'] = fit_exposure_statistics(examples['train'])
        # Use/verify the exact normalization used by the original checkpoint.
        for variant in cfg['main_variants']:
            source_norm = json.loads((Path(sources[variant]['path']).parent / 'normalization.json').read_text())
            for name in ('esm', 'gearnet', 'surface'):
                expected = {k:np.array(v, dtype=np.float32) if k != 'count' else v for k,v in source_norm[name].items()}
                if args.smoke:
                    base_stats[name] = expected
                else:
                    for key in ('mean', 'std', 'count'):
                        if not np.array_equal(base_stats[name][key], expected[key]):
                            raise ValueError('Train normalization differs from source checkpoint')
        save(output / 'normalization.json', {k:{'mean':v['mean'].tolist(), 'std':v['std'].tolist(), 'count':v['count']}
                                             for k,v in base_stats.items()})
        report['normalization_sha256'] = sha256(output / 'normalization.json')
        report['surface_subset_residues'] = {s:sum(int(x['surface_eval_mask'].sum()) for x in examples[s]) for s in examples}
        positive = dataset['positive_residues']['train']
        pos_weight = (dataset['valid_residues']['train'] - positive) / positive
        report['positive_class_weight'] = pos_weight
        training_cfg = dict(cfg['training'])
        if args.smoke:
            training_cfg.update(epochs=1, batch_size=4, eval_batch_size=4)
        report['training_budget'] = training_cfg
        counts = {n:sum(p.numel() for p in model_for(n).parameters()) for n in (*DIAGNOSTIC_VARIANTS, 'ES', 'ES-C')}
        report['parameter_counts'] = counts
        if counts['capacity_es'] != counts['ES-C'] or counts['capacity_es'] - counts['ES'] != 271136:
            raise ValueError('Capacity control parameter mismatch')
        progress(stage='permuting_features')
        perm_train = permute_surface(examples['train'], seed)
        perm_val = permute_surface(examples['val'], seed)
        if args.smoke:
            for original, shuffled, repeat in zip(examples['val'], perm_val, permute_surface(examples['val'], seed)):
                assert np.array_equal(shuffled['surface'], repeat['surface'])
                assert shuffled['mask'] is original['mask'] and shuffled['labels'] is original['labels']
                assert shuffled['BC_point'] is original['BC_point']
                # Compare the multiset of entire feature vectors, not just scalar marginals.
                assert sorted(map(bytes, original['surface'])) == sorted(map(bytes, shuffled['surface']))
            for name in DIAGNOSTIC_VARIANTS:
                first = seeded_model(name, seed)
                torch.rand(11)
                second = seeded_model(name, seed)
                assert all(torch.equal(first.state_dict()[k], v) for k,v in second.state_dict().items())
                first = first.to('cuda')
                batch = make_diag_batch(examples['train'][:4], base_stats, torch.device('cuda'), name == 'exposure_depth_es')
                loss = nn.functional.binary_cross_entropy_with_logits(first(batch)[batch['mask']], batch['labels'][batch['mask']])
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in first.parameters())
            report['smoke_checks'] = ['seeded initialization', 'finite gradients', 'permutation invariants', 'one-epoch training and export']
        for name in DIAGNOSTIC_VARIANTS:
            progress(stage='training_' + name)
            result = train_one(perm_train if name == 'permuted_es_c' else examples['train'],
                perm_val if name == 'permuted_es_c' else examples['val'], base_stats,
                training_cfg, seed, torch.device('cuda'), name == 'exposure_depth_es', output, name, pos_weight)
            report['results'].append(result)
            progress(completed_variants=[r['name'] for r in report['results']])
        for variant in cfg['main_variants']:
            progress(stage='baseline_inference_' + variant)
            model = model_for(variant).to('cuda')
            model.load_state_dict(torch.load(sources[variant]['path'], map_location='cpu')['state_dict'])
            result = export_diagnostics(model, examples['val'], base_stats, training_cfg['eval_batch_size'],
                torch.device('cuda'), output, variant)
            result['source'] = sources[variant]
            # Check the restored model against the original saved validation scores.
            original_path = Path(sources[variant]['path']).with_name(variant.replace('-', '_') + '_val_predictions.npz')
            original_meta = json.loads(original_path.with_suffix('.json').read_text())
            source_report = json.loads((original_path.parent / 'report.json').read_text())
            source_info = next(r for r in source_report['results'] if r['variant'] == variant)['predictions']
            if sha256(original_path) != source_info['sha256'] or sha256(original_path.with_suffix('.json')) != source_info['chain_metadata_sha256']:
                raise ValueError('Original prediction hash mismatch')
            lookup = {(r['split'], int(r['lmdb_index']), int(r['side'])):i for i,r in enumerate(original_meta)}
            with np.load(original_path, allow_pickle=False) as old, np.load(result['predictions']['path'], allow_pickle=False) as new:
                max_error = 0.0
                for i, example in enumerate(examples['val']):
                    j = lookup[(example['split'], example['key'][1], example['side'])]
                    a,b = old['offsets'][j:j+2]
                    c,d = new['offsets'][i:i+2]
                    np.testing.assert_array_equal(old['seq_ids'][a:b], new['seq_ids'][c:d])
                    np.testing.assert_array_equal(old['labels'][a:b], new['labels'][c:d])
                    np.testing.assert_allclose(old['logits'][a:b], new['logits'][c:d], rtol=1e-5, atol=1e-6)
                    max_error = max(max_error, float(np.max(np.abs(old['logits'][a:b] - new['logits'][c:d]))))
            result['source_prediction_max_error'] = max_error
            report['baselines'].append(result)
            progress(completed_baselines=[r['name'] for r in report['baselines']])
            if variant == 'ES-C':
                report['inference_only_permutation'] = export_diagnostics(model, perm_val, base_stats,
                    training_cfg['eval_batch_size'], torch.device('cuda'), output, 'inference_permuted_es_c')
        progress(status='passed', stage='complete')
    except Exception:
        report['status'] = 'failed'
        report['failures'].append(traceback.format_exc())
        print(report['failures'][-1], flush=True)
    finally:
        report.update(finished_at=time.time(), wall_seconds=time.time() - report['started_at'],
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated())
        save(output / 'report.json', report)
        print('REPORT', output / 'report.json', report['status'], flush=True)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
