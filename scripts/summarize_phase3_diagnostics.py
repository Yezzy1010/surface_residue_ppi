"""Independent saved-prediction checks and paired task-one diagnostic summary."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from phase3_diagnostic_metrics import metric_arrays
from summarize_phase3_task1 import digest, local_path
from surface_residue_ppi.paired_statistics import component_ids, paired_cluster_bootstrap


def read(p):
    return json.loads(Path(p).read_text())


def load(root, info):
    p = local_path(root, info['path'])
    if digest(p) != info['sha256'] or digest(p.with_suffix('.json')) != info['chain_metadata_sha256']:
        raise ValueError('Prediction hash mismatch: ' + str(p))
    with np.load(p, allow_pickle=False) as z:
        a = {k:z[k].copy() for k in z.files}
    meta = read(p.with_suffix('.json'))
    keys = [(r['split'], int(r['lmdb_index']), int(r['side'])) for r in meta]
    return a, keys


def main():
    root = Path('.').resolve()
    cfg = read(root / 'configs/experiments/phase3_20260927.json')
    protocol = root / cfg['protocol_dir']
    val = sorted(read(protocol / 'val_manifest.json'), key=lambda r:r['lmdb_index'])
    expected = [(r['split'], r['lmdb_index'], side) for r in val for side in (0,1)]
    component = np.repeat(component_ids(val), 2)
    paths = sorted((root / 'runs/phase3_diagnostics').glob('*/final_report.json'))
    if len(paths) != 3:
        raise ValueError('Expected three finalized diagnostic reports')
    scores = {}; sources=[]; selected=[]
    for path in paths:
        r=read(path); seed=r['seed']
        if r['status'] != 'passed' or r['failures'] or r['test_records_read'] or r['test_labels_read'] or seed in scores:
            raise ValueError('Invalid diagnostic report')
        ref=None; scores[seed]={}
        for item in r['results'] + r['baselines'] + [r['inference_only_permutation']]:
            a, keys = load(root, item['predictions'])
            if keys != expected:
                raise ValueError('Chain order mismatch')
            if 'surface_mask' not in a:
                mask_info=item['surface_mask_source']; p=local_path(root,mask_info['path'])
                if digest(p)!=mask_info['sha256']:
                    raise ValueError('Mask source hash mismatch')
                with np.load(p,allow_pickle=False) as z:
                    for k in ('labels','seq_ids','offsets'):
                        if not np.array_equal(a[k], z[k]):raise ValueError('Mask source identities differ')
                    a['surface_mask']=z['surface_mask'].copy()
            if ref is None:
                ref=a
                for i,key in enumerate(expected):
                    side=val[i//2]['sides'][i%2]
                    residues=sorted(side['common_valid'],key=lambda x:x['seq_id'])
                    lo,hi=a['offsets'][i:i+2]
                    if not np.array_equal(a['seq_ids'][lo:hi],[x['seq_id'] for x in residues]) or not np.array_equal(a['labels'][lo:hi],[int(x['positive']) for x in residues]):
                        raise ValueError('Frozen protocol residue mismatch')
            for k in ('labels','seq_ids','offsets','surface_mask'):
                if not np.array_equal(a[k],ref[k]): raise ValueError('Comparison identity/mask mismatch')
            scores[seed][item['name']]={}
            for subset,surface in [('all',False),('surface',True)]:
                per,metric=metric_arrays(a,surface)
                target=item['best_surface_only' if surface else 'recomputed_validation']
                for m in ('average_precision','auroc'):
                    if abs(metric[m]-target[m])>1e-12:raise ValueError('Metric mismatch')
                scores[seed][item['name']][subset]=per
                selected.append(dict(seed=seed,name=item['name'],subset=subset,metrics=metric))
        sources.append(dict(path=str(path.relative_to(root)),sha256=digest(path)))
    if set(scores)!=set(cfg['seeds']): raise ValueError('Seed mismatch')
    names=list(scores[cfg['seeds'][0]])
    aggregate={};comparisons={}
    pairs=[('ES','capacity_es'),('ES','exposure_depth_es'),('ES','permuted_es_c'),
           ('ES','ES-C'),('capacity_es','ES-C'),('exposure_depth_es','ES-C'),
           ('permuted_es_c','ES-C'),('ES-C','inference_permuted_es_c')]
    for subset in ('all','surface'):
        aggregate[subset]={};comparisons[subset]={}
        for metric in ('average_precision','auroc'):
            aggregate[subset][metric]={};comparisons[subset][metric]={}
            for name in names:
                v=[float(np.nanmean(scores[s][name][subset][metric])) for s in cfg['seeds']]
                aggregate[subset][metric][name]=dict(per_seed=v,mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1)))
            for base,method in pairs:
                a=np.stack([scores[s][base][subset][metric] for s in cfg['seeds']])
                b=np.stack([scores[s][method][subset][metric] for s in cfg['seeds']])
                if not np.array_equal(np.isfinite(a),np.isfinite(b)):
                    raise ValueError('Defined metric sets differ')
                delta=[float(np.nanmean(y-x)) for x,y in zip(a,b)]
                ci=paired_cluster_bootstrap(a.mean(0),b.mean(0),component,replicates=2000,seed=20260927)
                comparisons[subset][metric][method+' minus '+base]=dict(per_seed=delta,mean=float(np.mean(delta)),sample_sd=float(np.std(delta,ddof=1)),interval=ci)
    out=root/'runs/phase3_diagnostics_summary/20260928/main_v1/report.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    result=dict(status='passed',scope='task-one developmental diagnostic validation',sources=sources,
                seeds=cfg['seeds'],aggregate=aggregate,comparisons=comparisons,metrics=selected,
                test_records_read=0,test_labels_read=0,
                limits=['Intervals are conditional on selected checkpoints and do not correct validation model selection.',
                        'Inference-only permutation is a distribution shift sensitivity, not standalone causal proof.',
                        'Exposure and depth are monomer surface proxies, not exact solvent accessibility.'])
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
