"""Verify PPI predictions and bootstrap paired cluster-connected components."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from summarize_phase3_task1 import digest, local_path
from surface_residue_ppi.paired_statistics import component_ids


def weighted_metrics(y,score,weight):
    order=np.argsort(-score,kind='mergesort');s=score[order];y=y[order];w=weight[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    tp=np.cumsum(y*w)[ends];fp=np.cumsum((1-y)*w)[ends]
    if tp[-1]<=0 or fp[-1]<=0:return None
    keep=tp+fp>0;tp=tp[keep];fp=fp[keep]
    dt=np.diff(np.r_[0.,tp]);df=np.diff(np.r_[0.,fp])
    ap=float(np.sum(dt*tp/(tp+fp))/tp[-1])
    auc=float(np.sum(df*(tp+np.r_[0.,tp[:-1]])*.5)/(tp[-1]*fp[-1]))
    return np.array([ap,auc])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,default=Path('runs/phase3_task3_protocol/20260928/structure_v2'))
    ap.add_argument('--out',type=Path,default=Path('runs/phase3_task3_summary/20260928/structure_v2/report.json'))
    args=ap.parse_args();root=Path('.').resolve();protocol=json.loads((args.protocol/'report.json').read_text())
    rows=json.loads((args.protocol/'val.json').read_text());y=np.array([r['label'] for r in rows])
    components=component_ids(rows);ncomp=int(components.max())+1;seeds=protocol['seeds'];variants=protocol['variants']
    reports=[]
    for p in Path('runs/phase3_task3_structure_training').glob('*/report.json'):
        r=json.loads(p.read_text())
        if r['status']=='passed' and not r['smoke'] and r['protocol_sha256']==digest(args.protocol/'report.json'):
            reports.append((p,r))
    if len(reports)!=len(seeds) or {r['seed'] for p,r in reports}!=set(seeds):raise ValueError('Need all three seeds')
    if len({r['feature_hashes_sha256'] for p,r in reports})!=1:
        raise ValueError('Feature caches differ across seeds')
    scores={};source=[]
    for p,r in reports:
        if r['test_records_read'] or r['test_labels_read'] or r['failures']:raise ValueError('Unexpected test access/failure')
        scores[r['seed']]={}
        for result in r['results']:
            info=result['predictions'];path=local_path(root,info['path'])
            if digest(path)!=info['sha256']:raise ValueError('Prediction hash mismatch')
            with np.load(path,allow_pickle=False) as z:
                if not np.array_equal(z['labels'],y) or z['a'].tolist()!=[x['a'] for x in rows] or z['b'].tolist()!=[x['b'] for x in rows] or z['source_rows'].tolist()!=[x['source_row'] for x in rows]:
                    raise ValueError('Frozen pair identities differ')
                score=z['logits'].copy()
            if not np.isfinite(score).all():raise ValueError('Invalid scores')
            m=weighted_metrics(y,score,np.ones(len(y)))
            if not np.allclose(m,[average_precision_score(y,score),roc_auc_score(y,score)],atol=1e-12,rtol=0):
                raise ValueError('Independent metrics disagree')
            if not np.allclose(m,[result['validation']['average_precision'],result['validation']['auroc']],atol=1e-12,rtol=0):
                raise ValueError('Reported metrics disagree')
            scores[r['seed']][result['variant']]=score
        if set(scores[r['seed']])!=set(variants):raise ValueError('Missing variant')
        source.append(dict(path=str(p),sha256=digest(p)))
    aggregate={};comparisons={}
    for mi,m in enumerate(('average_precision','auroc')):
        aggregate[m]={}
        for v in variants:
            values=[float(weighted_metrics(y,scores[s][v],np.ones(len(y)))[mi]) for s in seeds]
            aggregate[m][v]=dict(per_seed=values,mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)))
    rng=np.random.default_rng(20260928);boot={v:[] for v in variants};undefined=0
    for i in range(2000):
        counts=np.bincount(rng.integers(ncomp,size=ncomp),minlength=ncomp);weight=counts[components]
        if not (y*weight).sum() or not ((1-y)*weight).sum():undefined+=1;continue
        for v in variants:boot[v].append(np.mean([weighted_metrics(y,scores[s][v],weight) for s in seeds],axis=0))
    for base,method in [('ES','ES-C'),('ES','ES-B'),('ES-B','ES-C'),('E','ES')]:
        delta=np.array(boot[method])-np.array(boot[base]);comparisons[method+' minus '+base]={}
        for mi,m in enumerate(('average_precision','auroc')):
            a=aggregate[m][base];b=aggregate[m][method];values=np.array(b['per_seed'])-np.array(a['per_seed'])
            lo,hi=np.quantile(delta[:,mi],[.025,.975])
            comparisons[method+' minus '+base][m]=dict(per_seed=values.tolist(),mean=float(values.mean()),sample_sd=float(values.std(ddof=1)),
                interval=dict(low=float(lo),high=float(hi),replicates=2000,undefined_draws=undefined,components=ncomp,
                              method='paired connected-component percentile bootstrap, averaged over three fixed fitted seeds'))
    out=dict(status='passed',scope='structure-based PPI developmental cluster-disjoint validation',
             source=source,protocol_sha256=digest(args.protocol/'report.json'),seeds=seeds,variants=variants,
             dataset=protocol['splits'],aggregate=aggregate,comparisons=comparisons,
             component_pair_sizes=sorted(np.bincount(components).tolist(),reverse=True),
             test_records_read=0,test_labels_read=0,limits=protocol['limits']+[
                 'Intervals condition on selected validation checkpoints; no model-selection or seed-population correction.',
                 'No task-two/task-three mechanism diagnostics were run.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
