"""Phase 3 task two: cross-chain residue-contact prediction on frozen features.

Only the selected train/validation LMDB records are read. Partner coordinates
are used to reconstruct supervision pairs, never passed to the encoders. Each
variant uses the same bilinear pair head and sampled negatives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import time
import traceback
import platform

import numpy as np
import torch
from torch import nn

from surface_residue_ppi.ppi_audit import ReleasedLMDB
from train_phase2_task1 import (Readout, fit_statistics, load_examples,
                                load_split_manifest, make_batch, sha256)
from phase2_training_metrics import binary_metrics

VARIANTS = ('ES', 'ES-B', 'ES-C')
PROTOCOL_VERSION = 'task2-common-mask-v2'


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def stable_seed(seed, key, epoch=0):
    return int(hashlib.sha256(f'{seed}:{key}:{epoch}'.encode()).hexdigest()[:8], 16)


class PairHead(nn.Module):
    def __init__(self, hidden=256, pair_hidden=128):
        super().__init__()
        self.left = nn.Linear(hidden, pair_hidden, bias=False)
        self.right = nn.Linear(hidden, pair_hidden, bias=False)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, left, right):
        return (self.left(left) * self.right(right)).sum(-1) / np.sqrt(self.left.out_features) + self.bias


def encode(model, variant, batch):
    h = model.project(torch.cat((batch['esm'], batch['gearnet']), dim=-1))
    if model.adapter is not None:
        edges = batch['A'] if variant == 'ES-A' else batch['B']
        h = model.adapter(h, batch['surface'], edges)
    return h


def map_side(side):
    return {tuple(row['residue_key']): int(row['seq_id']) - 1
            for row in side['common_valid']}


def load_pair_records(root, manifest, examples):
    by_key = {(x['split'], x['key'][1], x['side']): x for s in ('train', 'val') for x in examples[s]}
    records = {}
    dbs = {}
    try:
        for split in ('train', 'val'):
            dbs[split] = ReleasedLMDB(root / 'data/atom3d/zenodo-4911102/lmdb/DIPS-split/data' / split)
        for row in manifest:
            split, index = row['split'], int(row['lmdb_index'])
            left_map, right_map = map_side(row['sides'][0]), map_side(row['sides'][1])
            if not left_map or not right_map:
                raise ValueError(f'Empty common residue map at {split}/{index}')
            item = dbs[split][index]
            labels = item['atoms_neighbors']
            pairs = set()
            for rec in labels.itertuples(index=False):
                a = tuple(getattr(rec, f'{c}0') for c in ('ensemble','subunit','structure','model','chain','residue'))
                b = tuple(getattr(rec, f'{c}1') for c in ('ensemble','subunit','structure','model','chain','residue'))
                if a in left_map and b in right_map:
                    pairs.add((left_map[a], right_map[b]))
                elif b in left_map and a in right_map:
                    pairs.add((left_map[b], right_map[a]))
                # Contacts involving a residue outside the frozen common mask
                # are intentionally excluded from this task-two label set.
            key = (split, index)
            records[key] = dict(key=key, left=by_key[(split,index,0)], right=by_key[(split,index,1)],
                                positives=np.asarray(sorted(pairs), dtype=np.int64),
                                left_valid=np.flatnonzero(by_key[(split,index,0)]['mask']),
                                right_valid=np.flatnonzero(by_key[(split,index,1)]['mask']),
                                left_length=len(by_key[(split,index,0)]['labels']),
                                right_length=len(by_key[(split,index,1)]['labels']))
            if not pairs:
                raise ValueError(f'No positive pair after common-mask mapping at {split}/{index}')
    finally:
        for db in dbs.values(): db.close()
    return [records[(r['split'], int(r['lmdb_index']))] for r in manifest]


def sample_pairs(record, seed, epoch, ratio=3):
    positives = record['positives']
    left_valid, right_valid = record['left_valid'], record['right_valid']
    width = len(right_valid)
    total = len(left_valid) * width
    if not total or len(positives) == 0:
        raise ValueError('Empty valid candidate set or positive set')
    i = np.searchsorted(left_valid, positives[:, 0])
    j = np.searchsorted(right_valid, positives[:, 1])
    if (np.any(i >= len(left_valid)) or np.any(j >= width)
            or not np.array_equal(left_valid[i], positives[:, 0])
            or not np.array_equal(right_valid[j], positives[:, 1])):
        raise ValueError('A positive contact is outside the common mask')
    positive_set = set((i * width + j).tolist())
    if len(positive_set) != len(positives):
        raise ValueError('Duplicate positive pair')
    need = min(total - len(positive_set), max(1, len(positives) * ratio))
    rng = np.random.default_rng(stable_seed(seed, record['key'], epoch))
    negatives = []
    # Rejection sampling is efficient because the contact rate is low.
    while len(negatives) < need:
        count = max(need * 2, 128)
        flat = rng.integers(total, size=count)
        for value in flat:
            value = int(value)
            if value not in positive_set:
                positive_set.add(value)
                negatives.append((left_valid[value // width], right_valid[value % width]))
                if len(negatives) == need: break
    neg = np.asarray(negatives, dtype=np.int64).reshape(-1, 2)
    pairs = np.concatenate((positives, neg), axis=0)
    labels = np.concatenate((np.ones(len(positives), dtype=np.float32), np.zeros(len(negatives), dtype=np.float32)))
    return pairs, labels


def metrics_for_scores(labels, scores, ratio=10):
    rows = []
    for y,s in zip(labels,scores):
        rows.append(binary_metrics(y,s))
    valid_ap = [x['average_precision'] for x in rows if np.isfinite(x['average_precision'])]
    valid_auc = [x['auroc'] for x in rows if np.isfinite(x['auroc'])]
    return dict(average_precision=float(np.mean(valid_ap)), auroc=float(np.mean(valid_auc)),
                complexes=len(rows), ap_complexes=len(valid_ap), auroc_complexes=len(valid_auc),
                positives=int(sum(x['positives'] for x in rows)), negatives=int(sum(x['negatives'] for x in rows)),
                ap_undefined_complexes=len(rows)-len(valid_ap),
                auroc_undefined_complexes=len(rows)-len(valid_auc),
                sampled_negative_ratio=f'up to {ratio} negatives per positive',
                scope='developmental sampled-negative validation; not an official ATOM3D benchmark')


@torch.no_grad()
def evaluate(model, pair_head, records, stats, device, seed, ratio=10, export_path=None):
    model.eval(); pair_head.eval(); labels, scores = [], []
    saved_pairs, offsets, keys = [], [0], []
    for record in records:
        batch = make_batch([record['left'], record['right']], stats, device)
        h = encode(model, model.variant, batch)
        n0 = len(record['left']['labels']); n1 = len(record['right']['labels'])
        h0, h1 = h[:n0], h[n0:n0+n1]
        pairs, y = sample_pairs(record, seed, 100000 + ratio, ratio)
        logits = pair_head(h0[torch.from_numpy(pairs[:,0]).to(device)], h1[torch.from_numpy(pairs[:,1]).to(device)]).cpu().numpy()
        labels.append(y); scores.append(logits)
        if export_path is not None:
            saved_pairs.append(pairs + 1); offsets.append(offsets[-1] + len(y)); keys.append(record['key'])
    result = metrics_for_scores(labels, scores, ratio)
    if export_path is not None:
        from sklearn.metrics import average_precision_score, roc_auc_score
        for y, s in zip(labels, scores):
            m = binary_metrics(y, s)
            if abs(m['average_precision'] - average_precision_score(y, s)) > 1e-10:
                raise ValueError('Independent AP check failed')
            if len(np.unique(y)) == 2 and abs(m['auroc'] - roc_auc_score(y, s)) > 1e-10:
                raise ValueError('Independent AUROC check failed')
        np.savez_compressed(export_path, labels=np.concatenate(labels), logits=np.concatenate(scores),
                            pairs=np.concatenate(saved_pairs), offsets=np.asarray(offsets),
                            lmdb_indices=np.asarray([k[1] for k in keys], dtype=np.int64))
        save(export_path.with_suffix('.json'), dict(split='val', keys=keys, seed=seed,
             negative_ratio=ratio, protocol_version=PROTOCOL_VERSION, sha256=sha256(export_path)))
        result['prediction_path'] = str(export_path)
        result['prediction_sha256'] = sha256(export_path)
    return result


def train_variant(variant, records_train, records_val, stats, cfg, seed, device, output, pos_weight):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = Readout(variant, cfg['hidden_dim']).to(device)
    pair_head = PairHead(cfg['hidden_dim']).to(device)
    params = list(model.parameters()) + list(pair_head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    best, best_state, best_epoch, stale, history = -1, None, 0, 0, []
    for epoch in range(1, cfg['epochs'] + 1):
        model.train(); pair_head.train(); order = list(range(len(records_train))); random.shuffle(order); losses=[]
        for start in range(0, len(order), cfg['batch_size']):
            selected = [records_train[i] for i in order[start:start+cfg['batch_size']]]
            examples = sum(([r['left'], r['right']] for r in selected), [])
            batch = make_batch(examples, stats, device); h = encode(model, variant, batch)
            logits_all=[]; labels_all=[]; offset=0
            for record in selected:
                n0=len(record['left']['labels']); n1=len(record['right']['labels']); h0=h[offset:offset+n0]; h1=h[offset+n0:offset+n0+n1]; offset += n0+n1
                pairs,y=sample_pairs(record,seed,epoch,3)
                ix=torch.from_numpy(pairs[:,0]).to(device); jx=torch.from_numpy(pairs[:,1]).to(device)
                logits_all.append(pair_head(h0[ix],h1[jx])); labels_all.append(torch.from_numpy(y).to(device))
            loss=criterion(torch.cat(logits_all),torch.cat(labels_all)); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); losses.append(float(loss.detach()))
        val=evaluate(model,pair_head,records_val,stats,device,seed)
        row=dict(epoch=epoch,train_bce=float(np.mean(losses)),validation=val); history.append(row); save(output/(variant.replace('-','_')+'_history.json'),history); print('EPOCH',variant,seed,json.dumps(row),flush=True)
        if val['average_precision'] > best + cfg['min_delta']:
            best=val['average_precision']; best_epoch=epoch; stale=0; best_state={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'pair_head':{k:v.detach().cpu().clone() for k,v in pair_head.state_dict().items()}}
        else: stale += 1
        torch.save(dict(epoch=epoch, best=best, best_epoch=best_epoch, stale=stale,
                        model=model.state_dict(), pair_head=pair_head.state_dict(),
                        optimizer=opt.state_dict(), best_state=best_state, history=history,
                        seed=seed, variant=variant, protocol_version=PROTOCOL_VERSION,
                        python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()),
                   output/(variant.replace('-','_')+'_resume.pt'))
        if stale >= cfg['patience']: break
    if best_state is None: raise ValueError('No task2 checkpoint')
    model.load_state_dict(best_state['model']); pair_head.load_state_dict(best_state['pair_head'])
    checkpoint=output/(variant.replace('-','_')+'.pt'); torch.save(dict(variant=variant,seed=seed,epoch=best_epoch,model=best_state['model'],pair_head=best_state['pair_head']),checkpoint)
    val=evaluate(model,pair_head,records_val,stats,device,seed,
                 export_path=output/(variant.replace('-','_')+'_val_predictions.npz'))
    return dict(variant=variant,seed=seed,best_epoch=best_epoch,epochs_run=len(history),validation=val,trainable_parameters=sum(p.numel() for p in params),checkpoint_sha256=sha256(checkpoint),history=history)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('.')); ap.add_argument('--seed-index',type=int,required=True); ap.add_argument('--config',type=Path,default=Path('configs/experiments/phase3_20260927.json')); ap.add_argument('--limit',type=int,default=0); args=ap.parse_args(); root=args.root.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available(): raise RuntimeError('GPU Slurm required')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4)
    cfg=json.loads((root/args.config).read_text()); seed=cfg['seeds'][args.seed_index]; protocol=root/cfg['protocol_dir']; manifest=json.loads((protocol/'pilot_manifest.json').read_text()); splits={s:load_split_manifest(protocol/(s+'_manifest.json'),s) for s in ('train','val')}; strict=load_split_manifest(protocol/'strict_val_manifest.json','val'); features=json.loads((root/'runs/phase3_features/expanded_v1/report.json').read_text())
    if args.limit:
        chosen_train=sorted([r for r in manifest if r['split']=='train'],key=lambda r:r['lmdb_index'])[:args.limit]
        chosen_val=sorted([r for r in manifest if r['split']=='val'],key=lambda r:r['lmdb_index'])[:max(2,args.limit//5)]
        manifest=chosen_train+chosen_val
        allowed={(r['split'],int(r['lmdb_index'])) for r in manifest}
        splits={s:{k:v for k,v in values.items() if k in allowed} for s,values in splits.items()}
        strict={k:v for k,v in strict.items() if k in allowed}
        features=dict(features,records=[r for r in features['records'] if (r['split'],int(r['lmdb_index'])) in allowed])
        features['records_requested']=len(manifest)
    out=root/'runs/phase3_task2_training'/f'{os.environ["SLURM_JOB_ID"]}_{seed}'; out.mkdir(parents=True,exist_ok=False); report=dict(status='running',job_id=os.environ['SLURM_JOB_ID'],seed=seed,test_records_read=0,test_labels_read=0,started_at=time.time(),failures=[],protocol_sha256=sha256(protocol/'report.json'),feature_report_sha256=sha256(root/'runs/phase3_features/expanded_v1/report.json'))
    report.update(protocol_version=PROTOCOL_VERSION, host=platform.node(), gpu=torch.cuda.get_device_name(),
                  array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'), array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
                  source_sha256=sha256(Path(__file__)), config_sha256=sha256(root/args.config),
                  tf32=False, smoke=bool(args.limit), negative_universe='Cartesian product of the two common valid masks',
                  evaluation_scope='custom sampled-negative validation, not official benchmark')
    save(out/'report.json',report)
    try:
        examples,dataset=load_examples(root,features,manifest,splits['train'],splits['val'],strict); stats=fit_statistics(examples['train']); train_manifest=[r for r in manifest if r['split']=='train']; val_manifest=[r for r in manifest if r['split']=='val'];
        train_rows=load_pair_records(root,train_manifest,examples); val_rows=load_pair_records(root,val_manifest,examples); report['dataset']=dataset; report['pair_counts']=dict(train=sum(len(x['positives']) for x in train_rows),val=sum(len(x['positives']) for x in val_rows),train_complexes=len(train_rows),val_complexes=len(val_rows)); save(out/'report.json',report)
        save(out/'normalization.json', {k:{q:v.tolist() if isinstance(v,np.ndarray) else v for q,v in d.items()} for k,d in stats.items()})
        pos_weight=3.0; report['negative_sampling']='up to three valid noncontact negatives per positive for training; up to ten for validation; fixed within seed across methods'; report['results']=[]
        if args.limit: cfg['training']=dict(cfg['training'],epochs=1,patience=1,batch_size=2,eval_batch_size=2)
        for variant in VARIANTS:
            report['results'].append(train_variant(variant,train_rows,val_rows,stats,cfg['training'],seed,torch.device('cuda'),out,pos_weight)); save(out/'report.json',report)
        report['status']='passed'
    except Exception:
        report['status']='failed'; report['failures'].append(traceback.format_exc()); print(report['failures'][-1],flush=True)
    finally:
        report.update(finished_at=time.time(),wall_seconds=time.time()-report['started_at'],peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_gpu_bytes=torch.cuda.max_memory_allocated()); save(out/'report.json',report); print('REPORT',out/'report.json',report['status'],flush=True)
    if report['status']!='passed': raise SystemExit(1)


if __name__=='__main__': main()
