"""Verify task-two common-mask predictions and compare three paired seeds."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from phase2_training_metrics import binary_metrics
from summarize_phase3_task1 import digest, local_path
from surface_residue_ppi.paired_statistics import component_ids, paired_cluster_bootstrap


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('runs/phase3_task2_summary/20260928/common_mask_v2/report.json'))
    args=ap.parse_args();root=Path('.').resolve()
    cfg=json.loads(Path('configs/experiments/phase3_20260927.json').read_text())
    rows=sorted(json.loads((Path(cfg['protocol_dir'])/'val_manifest.json').read_text()),key=lambda r:r['lmdb_index'])
    expected=[r['lmdb_index'] for r in rows];components=component_ids(rows)
    reports=[]
    for p in Path('runs/phase3_task2_training').glob('*/report.json'):
        d=json.loads(p.read_text())
        if d.get('protocol_version')=='task2-common-mask-v2' and d['status']=='passed' and not d.get('smoke'):
            reports.append((p,d))
    if len(reports)!=3 or {d['seed'] for p,d in reports}!=set(cfg['seeds']):
        raise ValueError('Need exactly three completed, corrected full runs')
    if len({d['feature_report_sha256'] for p,d in reports})!=1 or len({d['protocol_sha256'] for p,d in reports})!=1:
        raise ValueError('Seeds used different frozen inputs')
    arrays={};sources=[];pairs_total={}
    for p,d in reports:
        seed=d['seed'];arrays[seed]={};ref=None
        if d['test_records_read'] or d['test_labels_read'] or d['failures']:
            raise ValueError('Unexpected test access or failure')
        if d['config_sha256']!=digest(Path('configs/experiments/phase3_20260927.json')):
            raise ValueError('Config mismatch')
        for result in d['results']:
            v=result['validation'];path=local_path(root,v['prediction_path'])
            if digest(path)!=v['prediction_sha256']:raise ValueError('Prediction hash mismatch')
            with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            if a['lmdb_indices'].tolist()!=expected:raise ValueError('Validation complexes differ')
            if len(a['offsets'])!=len(rows)+1 or a['offsets'][0]!=0 or a['offsets'][-1]!=len(a['labels']) or (np.diff(a['offsets'])<=0).any():
                raise ValueError('Invalid offsets')
            if a['logits'].shape!=a['labels'].shape or not np.isfinite(a['logits']).all():
                raise ValueError('Invalid predictions')
            if ref is not None:
                for k in ('pairs','labels','offsets','lmdb_indices'):
                    if not np.array_equal(a[k],ref[k]):raise ValueError('Unpaired method comparison')
            ref=a;metrics=[]
            for row,lo,hi in zip(rows,a['offsets'][:-1],a['offsets'][1:]):
                pair=a['pairs'][lo:hi];y=a['labels'][lo:hi];s=a['logits'][lo:hi]
                if not np.isin(pair[:,0],[r['seq_id'] for r in row['sides'][0]['common_valid']]).all() or not np.isin(pair[:,1],[r['seq_id'] for r in row['sides'][1]['common_valid']]).all():
                    raise ValueError('Pair outside common mask')
                if len(set(map(tuple,pair)))!=len(pair):raise ValueError('Duplicate sampled pair')
                m=binary_metrics(y,s)
                if abs(m['average_precision']-average_precision_score(y,s))>1e-10 or abs(m['auroc']-roc_auc_score(y,s))>1e-10:
                    raise ValueError('Independent metric mismatch')
                metrics.append(m)
            name=result['variant'];arrays[seed][name]={}
            for metric in ('average_precision','auroc'):
                values=np.array([m[metric] for m in metrics])
                if abs(float(np.nanmean(values))-v[metric])>1e-12:raise ValueError('Saved mean differs')
                arrays[seed][name][metric]=values
        if set(arrays[seed])!=set(cfg['main_variants']):raise ValueError('Incomplete models')
        pairs_total[str(seed)]={'positive':int(ref['labels'].sum()),'negative':int(len(ref['labels'])-ref['labels'].sum())}
        sources.append({'path':str(p),'sha256':digest(p)})
    aggregate={};comparisons={}
    for m in ('average_precision','auroc'):
        aggregate[m]={};comparisons[m]={}
        for v in cfg['main_variants']:
            means=[float(np.nanmean(arrays[s][v][m])) for s in cfg['seeds']]
            aggregate[m][v]=dict(per_seed=means,mean=float(np.mean(means)),sample_sd=float(np.std(means,ddof=1)))
        for base,variant in [('ES','ES-C'),('ES','ES-B'),('ES-B','ES-C')]:
            a=np.stack([arrays[s][base][m] for s in cfg['seeds']]);b=np.stack([arrays[s][variant][m] for s in cfg['seeds']])
            delta=[float(np.nanmean(y-x)) for x,y in zip(a,b)]
            ci=paired_cluster_bootstrap(a.mean(0),b.mean(0),components,replicates=2000,seed=20260927)
            comparisons[m][variant+' minus '+base]=dict(per_seed=delta,mean=float(np.mean(delta)),sample_sd=float(np.std(delta,ddof=1)),interval=ci)
    out=dict(status='passed',protocol_version='task2-common-mask-v2',sources=sources,seeds=cfg['seeds'],
             validation_complexes=len(rows),aggregate=aggregate,comparisons=comparisons,pair_counts=pairs_total,
             test_records_read=0,test_labels_read=0,
             scope='developmental sampled-negative contact validation, not official benchmark',
             limits=['Common masks on both sides; up to 3 train and 10 validation negatives per contact.',
                     'Validation negative sets are fixed across variants within each seed; vary across seeds.',
                     'Bootstrap resamples homology-linked complex components; conditional on selected checkpoints.',
                     'DIPS bound structures and shared supervision with task one; not independent unbound evidence.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
