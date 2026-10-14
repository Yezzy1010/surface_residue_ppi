"""Three-seed Phase 3 readouts with saved, recomputable validation predictions."""
from __future__ import annotations

import argparse
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
                               fit_statistics, make_batch, evaluate, sha256)
from phase2_training_metrics import binary_metrics, macro_metrics


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@torch.no_grad()
def export_predictions(model, examples, stats, batch_size, device, out, variant):
    model.eval()
    labels, scores, positions, metadata, metrics = [], [], [], [], []
    offsets = [0]
    for start in range(0, len(examples), batch_size):
        selected = examples[start:start + batch_size]
        batch = make_batch(selected, stats, device)
        prediction = model(batch).cpu().numpy()
        begin = 0
        for example in selected:
            length = len(example['labels'])
            mask = example['mask']
            y = example['labels'][mask].astype(np.int8)
            score = prediction[begin:begin + length][mask]
            begin += length
            metric = binary_metrics(y, score)
            metrics.append(metric)
            labels.append(y)
            scores.append(score)
            positions.append(np.flatnonzero(mask) + 1)
            offsets.append(offsets[-1] + len(y))
            metadata.append(dict(split=example['split'], lmdb_index=example['key'][1], side=example['side'],
                                 cache=str(example['cache']), positives=metric['positives'], negatives=metric['negatives'],
                                 average_precision=metric['average_precision'] if np.isfinite(metric['average_precision']) else None,
                                 auroc=metric['auroc'] if np.isfinite(metric['auroc']) else None))
    flat_labels, flat_scores = np.concatenate(labels), np.concatenate(scores)
    prediction_path = out / (variant.replace('-', '_') + '_val_predictions.npz')
    np.savez_compressed(prediction_path, labels=flat_labels, logits=flat_scores,
                        seq_ids=np.concatenate(positions), offsets=np.array(offsets, dtype=np.int64))
    save(prediction_path.with_suffix('.json'), metadata)
    # Read back saved predictions and recalculate; detect serialization/order errors.
    with np.load(prediction_path, allow_pickle=False) as archive:
        reread = [binary_metrics(archive['labels'][a:b], archive['logits'][a:b])
                  for a,b in zip(archive['offsets'][:-1], archive['offsets'][1:])]
        result = macro_metrics(reread, binary_metrics(archive['labels'], archive['logits']))
    return result, dict(path=str(prediction_path), sha256=sha256(prediction_path),
                        chain_metadata_sha256=sha256(prediction_path.with_suffix('.json')))


def train(variant, train_examples, val_examples, stats, cfg, seed, device, out, pos_weight):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Readout(variant, cfg['hidden_dim']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device), reduction='none')
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    best_ap, best_state, best_metrics, best_epoch, stale = -float('inf'), None, None, 0, 0
    history = []
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        order = list(range(len(train_examples)))
        random.shuffle(order)
        total, count = 0.0, 0
        for start in range(0, len(order), cfg['batch_size']):
            batch = make_batch([train_examples[i] for i in order[start:start+cfg['batch_size']]], stats, device)
            loss = criterion(model(batch), batch['labels'])[batch['mask']].mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            valid = int(batch['mask'].sum())
            total += float(loss.detach()) * valid
            count += valid
        metric = evaluate(model, val_examples, stats, cfg['eval_batch_size'], device)
        history.append(dict(epoch=epoch, train_weighted_bce=total/count, validation=metric))
        print('EPOCH', variant, seed, json.dumps(history[-1]), flush=True)
        if metric['average_precision'] > best_ap + cfg['min_delta']:
            best_ap, best_metrics, best_epoch, stale = metric['average_precision'], metric, epoch, 0
            best_state = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        else:
            stale += 1
        save(out / (variant.replace('-', '_') + '_history.json'), history)
        if stale >= cfg['patience']:
            break
    if best_state is None:
        raise ValueError('No valid checkpoint')
    checkpoint = out / (variant.replace('-', '_') + '.pt')
    torch.save(dict(variant=variant, seed=seed, epoch=best_epoch, state_dict=best_state), checkpoint)
    model.load_state_dict(best_state)
    recomputed, prediction_info = export_predictions(model, val_examples, stats, cfg['eval_batch_size'], device, out, variant)
    if abs(recomputed['average_precision'] - best_metrics['average_precision']) > 1e-6:
        raise ValueError('Saved prediction AP does not match selected checkpoint')
    return dict(variant=variant, seed=seed, best_epoch=best_epoch, best_validation=best_metrics,
                predictions=prediction_info, recomputed_validation=recomputed,
                checkpoint_sha256=sha256(checkpoint), trainable_parameters=sum(p.numel() for p in model.parameters()),
                epochs_run=len(history), history=history, seconds=time.time()-started,
                peak_gpu_bytes=torch.cuda.max_memory_allocated())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--config', type=Path, default=Path('configs/experiments/phase3_20260927.json'))
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--seed-index', type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Use GPU Slurm')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = json.loads((root / args.config).read_text())
    seed = cfg['seeds'][args.seed_index]
    protocol = root / cfg['protocol_dir']
    manifest = json.loads((protocol / 'pilot_manifest.json').read_text())
    splits = {s:load_split_manifest(protocol / (s+'_manifest.json'), s) for s in ('train', 'val')}
    strict = load_split_manifest(protocol / 'strict_val_manifest.json', 'val')
    feature_report = json.loads((root / args.features).read_text())
    if feature_report.get('protocol_sha256') != sha256(protocol / 'report.json'):
        raise ValueError('Feature report belongs to a different protocol')
    job = os.environ['SLURM_JOB_ID']
    out = root / 'runs/phase3_task1_training' / (job + '_' + str(seed))
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(root / args.config, out / 'config.json')
    report = dict(status='running', job_id=job, seed=seed, host=platform.node(), started_at=time.time(),
                  config=cfg, config_sha256=sha256(root / args.config), protocol=str(protocol),
                  manifest_sha256=sha256(protocol / 'pilot_manifest.json'),
                  feature_report_sha256=sha256(root / args.features), feature_report=str(args.features),
                  source_sha256={name:sha256(root/name) for name in ['scripts/train_phase3_task1.py',
                    'scripts/train_phase2_task1.py', 'scripts/phase2_training_metrics.py', 'src/surface_residue_ppi/adapters.py']},
                  test_records_read=0, test_labels_read=0, backbone_features_frozen=True,
                  normalization_fit_split='train', failures=[], results=[])
    report['arithmetic'] = 'float32; CUDA matmul/cudnn TF32 disabled'
    save(out / 'report.json', report)
    try:
        torch.set_num_threads(4)
        examples, dataset = load_examples(root, feature_report, manifest, splits['train'], splits['val'], strict)
        if dataset['record_counts']['train'] <= 945:
            raise ValueError('Phase 3 training set has not expanded')
        report['dataset'] = dataset
        print('DATASET', json.dumps(dataset), flush=True)
        stats = fit_statistics(examples['train'])
        normalization = {k:{'count':v['count'], 'mean':v['mean'].tolist(), 'std':v['std'].tolist()} for k,v in stats.items()}
        save(out / 'normalization.json', normalization)
        report['normalization_sha256'] = sha256(out / 'normalization.json')
        positive = dataset['positive_residues']['train']
        pos_weight = (dataset['valid_residues']['train'] - positive) / positive
        report['positive_class_weight'] = pos_weight
        for variant in cfg['main_variants']:
            result = train(variant, examples['train'], examples['val'], stats, cfg['training'], seed,
                           torch.device('cuda'), out, pos_weight)
            report['results'].append(result)
            save(out / 'report.json', report)
        report['status'] = 'passed'
    except Exception:
        report['status'] = 'failed'
        report['failures'].append(traceback.format_exc())
        print(report['failures'][-1], flush=True)
    finally:
        report.update(finished_at=time.time(), wall_seconds=time.time()-report['started_at'],
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(out / 'report.json', report)
        print('REPORT', out / 'report.json', report['status'], flush=True)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
