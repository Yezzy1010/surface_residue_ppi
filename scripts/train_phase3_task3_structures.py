"""Three-seed PPI main comparison on frozen, cluster-disjoint monomer features."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import random
import resource
import time
import traceback
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score
from train_phase2_task1 import Readout, make_batch, fit_statistics, sha256
from run_phase3_diagnostics import save
from phase2_training_metrics import binary_metrics


class ProteinPair(nn.Module):
    def __init__(self,variant,hidden=256):
        super().__init__()
        self.encoder=Readout(variant,hidden)
        self.encoder.head=nn.Identity()
        self.relation=nn.Sequential(nn.Linear(3*hidden,128),nn.GELU(),nn.Dropout(0.1),nn.Linear(128,1))

    def forward(self,batch,lengths,ix,jx):
        enc=self.encoder
        h=enc.project(batch['esm'] if enc.variant=='E' else torch.cat((batch['esm'],batch['gearnet']),-1))
        if enc.adapter is not None:h=enc.adapter(h,batch['surface'],batch['B'])
        pooled=[];offset=0
        for n in lengths:
            mask=batch['mask'][offset:offset+n]
            if not mask.any():raise ValueError('Empty common mask')
            pooled.append(h[offset:offset+n][mask].mean(0));offset+=n
        pooled=torch.stack(pooled);u=pooled[ix];v=pooled[jx]
        return self.relation(torch.cat((u+v,torch.abs(u-v),u*v),-1)).squeeze(-1)


def batch_rows(rows,examples,stats,device):
    keys=list(dict.fromkeys(k for r in rows for k in (r['a'],r['b'])))
    bykey={k:i for i,k in enumerate(keys)}
    ex=[examples[k] for k in keys]
    b=make_batch(ex,stats,device)
    ix=torch.tensor([bykey[r['a']] for r in rows],device=device)
    jx=torch.tensor([bykey[r['b']] for r in rows],device=device)
    y=torch.tensor([r['label'] for r in rows],device=device,dtype=torch.float32)
    return b,[len(x['mask']) for x in ex],ix,jx,y


@torch.no_grad()
def evaluate(model,rows,examples,stats,device,batch_size,check_symmetry=False):
    model.eval();ys=[];scores=[]
    for start in range(0,len(rows),batch_size):
        b,n,i,j,y=batch_rows(rows[start:start+batch_size],examples,stats,device)
        logits=model(b,n,i,j)
        if check_symmetry and start==0:
            swapped=model(b,n,j,i)
            if not torch.allclose(logits,swapped,atol=1e-5,rtol=1e-5):raise ValueError('Symmetry check failed')
        ys.extend(y.cpu().numpy().tolist());scores.extend(logits.cpu().numpy().tolist())
    y=np.asarray(ys,dtype=np.int8);s=np.asarray(scores,dtype=np.float32);metrics=binary_metrics(y,s)
    if abs(metrics['average_precision']-average_precision_score(y,s))>1e-10 or abs(metrics['auroc']-roc_auc_score(y,s))>1e-10:
        raise ValueError('Independent metrics disagree')
    return metrics,y,s


def train(variant,seed,train_rows,val_rows,examples,stats,cfg,device,out):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=ProteinPair(variant,cfg['hidden_dim']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    npos=sum(r['label'] for r in train_rows);weight=(len(train_rows)-npos)/npos
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device=device))
    best=-1.;best_state=None;best_epoch=0;history=[];stale=0;started=time.time()
    for epoch in range(1,cfg['epochs']+1):
        model.train();order=list(range(len(train_rows)));random.shuffle(order);losses=[]
        for start in range(0,len(order),cfg['batch_size']):
            selected=[train_rows[i] for i in order[start:start+cfg['batch_size']]]
            b,n,i,j,y=batch_rows(selected,examples,stats,device)
            loss=loss_fn(model(b,n,i,j),y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            losses.append(float(loss.detach()))
        metrics,_,_=evaluate(model,val_rows,examples,stats,device,cfg['batch_size'])
        row=dict(epoch=epoch,train_bce=float(np.mean(losses)),validation=metrics);history.append(row)
        save(out/(variant.replace('-','_')+'_history.json'),history)
        print('EPOCH',variant,seed,json.dumps(row),flush=True)
        if metrics['average_precision']>best+cfg['min_delta']:
            best=metrics['average_precision'];best_epoch=epoch;stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        checkpoint=dict(seed=seed,variant=variant,epoch=epoch,best_epoch=best_epoch,best=best,stale=stale,
                        model=model.state_dict(),best_state=best_state,optimizer=opt.state_dict(),history=history,
                        python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        temp=out/(variant.replace('-','_')+'_resume.partial.pt')
        torch.save(checkpoint,temp);temp.replace(out/(variant.replace('-','_')+'_resume.pt'))
        if stale>=cfg['patience']:break
    model.load_state_dict(best_state)
    checkpoint=out/(variant.replace('-','_')+'.pt')
    torch.save(dict(seed=seed,variant=variant,epoch=best_epoch,state_dict=best_state),checkpoint)
    metric,y,s=evaluate(model,val_rows,examples,stats,device,cfg['batch_size'],True)
    p=out/(variant.replace('-','_')+'_val_predictions.npz')
    np.savez_compressed(p,labels=y,logits=s,a=np.array([r['a'] for r in val_rows]),b=np.array([r['b'] for r in val_rows]),
                        source_rows=np.array([r['source_row'] for r in val_rows]))
    return dict(variant=variant,seed=seed,validation=metric,best_epoch=best_epoch,epochs_run=len(history),
                seconds=time.time()-started,checkpoint_sha256=sha256(checkpoint),
                trainable_parameters=sum(p.numel() for p in model.parameters()),predictions=dict(path=str(p),sha256=sha256(p)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed-index',type=int,required=True)
    ap.add_argument('--protocol',type=Path,default=Path('runs/phase3_task3_protocol/20260928/structure_v2'))
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args();root=Path('.').resolve()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm GPU required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=json.loads((args.protocol/'report.json').read_text())
    if protocol['status']!='passed' or protocol['test_records_read'] or protocol['test_labels_read']:raise ValueError('Invalid protocol')
    for name,digest in protocol['artifact_sha256'].items():
        if sha256(args.protocol/name)!=digest:raise ValueError('Protocol artifact changed')
    seed=protocol['seeds'][args.seed_index];cfg=dict(protocol['training'])
    train_rows=json.loads((args.protocol/'train.json').read_text());val_rows=json.loads((args.protocol/'val.json').read_text())
    if args.smoke:
        train_rows=sum(([r for r in train_rows if r['label']==y][:4] for y in (0,1)),[])
        val_rows=sum(([r for r in val_rows if r['label']==y][:2] for y in (0,1)),[])
        cfg.update(epochs=1,patience=1,batch_size=2)
    out=root/'runs/phase3_task3_structure_training'/(os.environ['SLURM_JOB_ID']+'_'+str(seed));out.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',seed=seed,smoke=args.smoke,job_id=os.environ['SLURM_JOB_ID'],
                array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),host=platform.node(),gpu=torch.cuda.get_device_name(),
                protocol_sha256=sha256(args.protocol/'report.json'),source_sha256=sha256(Path(__file__)),
                protocol=str(args.protocol),test_records_read=0,test_labels_read=0,tf32=False,
                started_at=time.time(),results=[],failures=[],training=cfg)
    save(out/'report.json',report)
    try:
        ids={k for r in train_rows+val_rows for k in (r['a'],r['b'])}
        train_ids={k for r in train_rows for k in (r['a'],r['b'])};val_ids={k for r in val_rows for k in (r['a'],r['b'])}
        if train_ids&val_ids:raise ValueError('Protein overlap')
        frozen={r['id']:r for r in json.loads((args.protocol/'proteins.json').read_text())}
        examples={};digests={};checkpoint_hashes=[]
        for i,k in enumerate(sorted(ids),1):
            base=root/'runs/phase3_task3_features/structure_v1/cache';p=base/(k+'.npz');m=json.loads(p.with_suffix('.json').read_text())
            if sha256(p)!=m['cache_sha256'] or m['pdb_sha256']!=frozen[k]['pdb_sha256'] or m['sequence_sha256']!=frozen[k]['sequence_sha256']:
                raise ValueError('Feature provenance mismatch')
            checkpoint_hashes.append(m['checkpoint_sha256']);digests[k]=m['cache_sha256']
            with np.load(p,allow_pickle=False) as z:a={name:z[name] for name in z.files}
            n=len(frozen[k]['sequence'])
            if a['esm'].shape!=(n,480) or a['gearnet'].shape!=(n,3072) or not a['common_valid'].all():
                raise ValueError('Feature dimensions/common mask mismatch')
            if not all(np.isfinite(v).all() for v in a.values()):raise ValueError('Nonfinite cache')
            examples[k]=dict(esm=a['esm'],gearnet=a['gearnet'],surface=a['surface'],mask=a['common_valid'],
                            labels=np.zeros(n,dtype=np.float32),**{key:a[key] for key in ('A_residue','A_point','A_distance','BC_residue','BC_point','BC_distance')})
            if i%200==0:print('LOADED',i,len(ids),flush=True)
        if any(h!=checkpoint_hashes[0] for h in checkpoint_hashes):raise ValueError('Backbone weights differ')
        save(out/'feature_hashes.json',digests)
        stats=fit_statistics([examples[k] for k in sorted(train_ids)])
        save(out/'normalization.json',{k:{q:v.tolist() if isinstance(v,np.ndarray) else v for q,v in d.items()} for k,d in stats.items()})
        report.update(train_pairs=len(train_rows),val_pairs=len(val_rows),train_proteins=len(train_ids),val_proteins=len(val_ids),
                      feature_hashes_sha256=sha256(out/'feature_hashes.json'),backbone_sha256=checkpoint_hashes[0])
        for variant in protocol['variants']:
            result=train(variant,seed,train_rows,val_rows,examples,stats,cfg,torch.device('cuda'),out)
            report['results'].append(result);save(out/'report.json',report)
        report['status']='passed'
    except Exception:
        report['status']='failed';report['failures'].append(traceback.format_exc());print(report['failures'][-1],flush=True)
    finally:
        report.update(wall_seconds=time.time()-report['started_at'],peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated());save(out/'report.json',report)
    print('REPORT',out/'report.json',report['status'],flush=True)
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
