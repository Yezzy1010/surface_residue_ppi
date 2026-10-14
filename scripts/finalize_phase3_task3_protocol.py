"""Freeze the structural PPI comparison using only official TRAIN pairs."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def distribution(x):
    x=np.asarray(x,dtype=float)
    return dict(count=len(x),min=float(x.min()),median=float(np.median(x)),mean=float(x.mean()),max=float(x.max()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,default=Path('runs/phase3_task3_structure_audit/20260928/exact_v1'))
    ap.add_argument('--out',type=Path,default=Path('runs/phase3_task3_protocol/20260928/structure_v2'))
    args=ap.parse_args();audit=json.loads((args.audit/'report.json').read_text())
    if audit['status']!='passed' or audit['counts']['failed']:raise ValueError('Structure audit incomplete')
    mapping=json.loads((args.audit/'mapping.json').read_text())
    records={r['id']:json.loads((args.audit/'records'/(r['id']+'.json')).read_text()) for r in mapping}
    ready={k:r for k,r in records.items() if r['status']=='ready'}
    for r in ready.values():
        if (not np.isfinite(r['mean_plddt']) or not 70<=r['mean_plddt']<=100
                or set(r['sequence'])-set('ACDEFGHIKLMNPQRSTVWY')):
            raise ValueError('Invalid quality or noncanonical sequence in ready structure')
    cluster={}
    for line in (args.audit/'seq40_cluster.tsv').read_text().splitlines():
        representative,member=line.split('\t')
        if member in cluster:raise ValueError('Duplicate cluster member')
        cluster[member]=representative
    if set(cluster)!=set(records):raise ValueError('Cluster coverage mismatch')
    pool={k:'val' if int(hashlib.sha256(('20260928:'+c).encode()).hexdigest()[:8],16)/(2**32)<0.30 else 'train' for k,c in cluster.items()}
    source=Path('runs/phase3_task3_protocol/20260928/subset_v1/train.tsv')
    if digest(source)!=audit['train_sha256']:raise ValueError('Train source drift')
    selected={'train':[],'val':[]};excluded=[];seen={};eligible=[]
    for i,line in enumerate(source.read_text().splitlines()):
        a,b,y=line.split('\t');y=int(y);key=tuple(sorted((a,b)))
        if key in seen:
            if seen[key]!=y:raise ValueError('Conflicting unordered pair labels')
            excluded.append(dict(source_row=i,reason='duplicate_unordered_pair'));continue
        seen[key]=y
        row=dict(a=a,b=b,label=y,source_row=i,cluster_pair=[cluster[a],cluster[b]])
        if a not in ready or b not in ready:
            excluded.append(dict(row,reason='missing_or_quality_excluded_structure'));continue
        eligible.append(row)
        if pool[a]!=pool[b]:
            excluded.append(dict(row,reason='cross_pool_pair_excluded_for_cluster_disjointness'));continue
        selected[pool[a]].append(row)
    ids={s:{k for r in rs for k in (r['a'],r['b'])} for s,rs in selected.items()}
    cids={s:{cluster[k] for k in ix} for s,ix in ids.items()}
    if ids['train']&ids['val'] or cids['train']&cids['val']:raise ValueError('Split leakage')
    for s,rs in selected.items():
        if {r['label'] for r in rs}!={0,1} or len(rs)<30:raise ValueError('Insufficient split: '+s)
    used=ids['train']|ids['val'];proteins=[ready[k] for k in sorted(used)]
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    def write(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
    for s,rs in selected.items():write(s+'.json',rs)
    write('proteins.json',proteins);write('exclusions.json',excluded)
    report=dict(status='passed',version='task3-structure-v2',host=platform.node(),created_at=time.time(),
                source='selected official D-SCRIPT human TRAIN only',source_sha256=digest(source),
                audit_report_sha256=digest(args.audit/'report.json'),clusters_sha256=digest(args.audit/'seq40_cluster.tsv'),
                source_script_sha256=digest(Path(__file__)),test_records_read=0,test_labels_read=0,
                upstream_test_access_deviation='Sequence pilot 276306 and subset_v1 preparation accessed official test; preserved separately. This protocol reads no test pairs, labels or scores.',
                original_train_pairs=len(seen),original_train_positive=sum(seen.values()),
                structure_eligible_pairs=len(eligible),structure_eligible_positive=sum(r['label'] for r in eligible),
                original_proteins=len(records),structure_ready_proteins=len(ready),used_proteins=len(used),
                splits={s:dict(pairs=len(rs),positives=sum(r['label'] for r in rs),negatives=sum(1-r['label'] for r in rs),
                               proteins=len(ids[s]),clusters=len(cids[s])) for s,rs in selected.items()},
                protein_overlap=0,cluster_overlap=0,
                partition=dict(seed=20260928,rule='hash cluster representative into 70% train / 30% validation protein pools; only within-pool pairs',
                               mmseqs_identity=0.4,mmseqs_coverage=0.8,mmseqs_cov_mode=0,mmseqs_cluster_mode=0),
                length_distribution={'original':distribution([len(r['sequence']) for r in records.values()]),
                                     'structure_ready':distribution([r['length'] for r in ready.values()]),
                                     'used':distribution([r['length'] for r in proteins])},
                mean_plddt_distribution=distribution([r['mean_plddt'] for r in proteins]),
                quality_rules=audit['rules'],variants=['E','ES','ES-B','ES-C'],seeds=[20260927,20260928,20260929],
                training=dict(epochs=30,patience=5,min_delta=0.0001,batch_size=24,learning_rate=0.001,
                              weight_decay=0.0001,hidden_dim=256,selection_metric='validation_pair_average_precision',
                              normalization='unique training proteins only',head='symmetric sum/absdiff/product MLP, hidden 128'),
                limits=['Developmental validation only; official test is not an independent final evaluation after prior access.',
                        'Cluster disjoint at specified MMseqs thresholds does not eliminate all remote homology.',
                        'Operational random PPI negatives; not all experimentally verified noninteractions.',
                        'Full exact sequence and AlphaFold quality filtering introduces selection bias.',
                        'Predicted monomer geometry and confidence differ from bound DIPS structures.'])
    report['artifact_sha256']={name:digest(out/name) for name in ('train.json','val.json','proteins.json','exclusions.json')}
    write('report.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
