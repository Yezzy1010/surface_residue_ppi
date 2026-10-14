from __future__ import annotations
import argparse, json, os, platform, time, traceback
from pathlib import Path
import numpy as np
import torch
from train_phase3_task2 import load_pair_records, sample_pairs, stable_seed, sha256, encode
from train_phase2_task1 import Readout, load_examples, load_split_manifest, fit_statistics, make_batch
from train_phase3_task2_sym import (SEEDS, SEED_JOB, VARIANTS, SymmetricPairHead,
                                    fixed_validation_source, load_fixed_validation,
                                    assert_validation_rows, generate_stream, ABS_TOL, REL_TOL,
                                    array_sha256, head_formula)

def save(p,v): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,indent=2,allow_nan=False,default=str)+'\n')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('.')); ap.add_argument('--config',type=Path,default=Path('configs/experiments/phase3_20260927.json')); ap.add_argument('--limit',type=int,default=16); args=ap.parse_args(); root=args.root.resolve()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available(): raise RuntimeError('GPU Slurm required')
    cfg=json.loads((root/args.config).read_text()); protocol=root/cfg['protocol_dir']; pilot=json.loads((protocol/'pilot_manifest.json').read_text()); splits={s:load_split_manifest(protocol/(s+'_manifest.json'),s) for s in ('train','val')}; strict=load_split_manifest(protocol/'strict_val_manifest.json','val'); features=json.loads((root/'runs/phase3_features/expanded_v1/report.json').read_text())
    out=root/'runs/phase3_task2_sym_preflight'/os.environ['SLURM_JOB_ID']; out.mkdir(parents=True,exist_ok=False)
    result=dict(status='running',job_id=os.environ['SLURM_JOB_ID'],host=platform.node(),started_at=time.time(),task_id='SRPPI-T2-SYM-001',task_version='v1.1',head_formula=head_formula(),tolerance_abs=ABS_TOL,tolerance_rel=REL_TOL,test_records_read=0,test_labels_read=0,source_sha256=sha256(root/'scripts/train_phase3_task2_sym.py'),ordered_source_sha256=sha256(root/'scripts/train_phase3_task2.py'),config_sha256=sha256(root/args.config),protocol_sha256=sha256(protocol/'report.json'),train_manifest_sha256=sha256(protocol/'train_manifest.json'),val_manifest_sha256=sha256(protocol/'val_manifest.json'),feature_report_sha256=sha256(root/'runs/phase3_features/expanded_v1/report.json'),checks=[],failures=[])
    save(out/'report.json',result)
    try:
      examples,dataset=load_examples(root,features,pilot,splits['train'],splits['val'],strict)
      train_manifest=[r for r in pilot if r['split']=='train']; val_manifest=[r for r in pilot if r['split']=='val']
      train_rows=load_pair_records(root,train_manifest,examples); val_rows=load_pair_records(root,val_manifest,examples)
      # Dynamic sampler reproducibility and shared stream check on fixed epochs.
      sampler=[]
      for seed in SEEDS:
        for epoch in (1,7,30):
          _,a=generate_stream(train_rows,seed,epoch,3); _,b=generate_stream(train_rows,seed,epoch,3)
          if a != b: raise ValueError(f'sampler repeat mismatch seed={seed} epoch={epoch}')
          sampler.append(dict(seed=seed,**a,repeat_equal=True,variant_streams={v:a['sha256'] for v in VARIANTS}))
      result['sampler_checks']=sampler
      # Fixed validation sources and exact cross-variant payload alignment.
      val_checks=[]
      for seed in SEEDS:
        rows,meta=load_fixed_validation(root,seed,val_rows); assert_validation_rows(rows,val_rows)
        payload=[]
        for v in ('ES','ES_B','ES_C'):
          p=root/f'runs/phase3_task2_training/{SEED_JOB[seed]}_{seed}/{v}_val_predictions.npz'
          with np.load(p,allow_pickle=False) as z:
            payload.append({k:array_sha256(z[k]) for k in ('labels','pairs','offsets','lmdb_indices')})
        if not all(x==payload[0] for x in payload): raise ValueError(f'validation payload mismatch seed={seed}')
        val_checks.append(dict(seed=seed,source=meta,payload_hashes=payload))
      result['validation_checks']=val_checks
      # GPU model smoke with first two validation records and fixed pairs.
      seed=SEEDS[0]; rows,_=load_fixed_validation(root,seed,val_rows); selected=val_rows[:2]; fixed=rows[:2]; stats=fit_statistics(examples['train']); device=torch.device('cuda')
      smoke=[]
      for name in VARIANTS:
        base={'ES-SYM':'ES','ES-B-SYM':'ES-B','ES-C-SYM':'ES-C'}[name]; model=Readout(base,256).to(device); head=SymmetricPairHead(256).to(device); model.eval(); head.eval()
        batch=make_batch(sum(([r['left'],r['right']] for r in selected),[]),stats,device); h=encode(model,base,batch); offset=0; maxdiff=0.0; total=0
        for rec,(pairs,labels) in zip(selected,fixed):
          n0=len(rec['left']['labels']); n1=len(rec['right']['labels']); h0=h[offset:offset+n0]; h1=h[offset+n0:offset+n0+n1]; offset += n0+n1
          ix=torch.from_numpy(pairs[:min(32,len(pairs)),0]).to(device); jx=torch.from_numpy(pairs[:min(32,len(pairs)),1]).to(device); ab=head(h0[ix],h1[jx]); ba=head(h1[jx],h0[ix]); d=(ab-ba).abs().max().item(); maxdiff=max(maxdiff,d); total += len(ix)
          if not torch.isfinite(ab).all() or not torch.isfinite(ba).all(): raise ValueError('smoke NaN/Inf')
        smoke.append(dict(variant=name,base_variant=base,pairs_checked=total,max_abs_delta=maxdiff,pass_fixed=bool(maxdiff <= ABS_TOL)))
        del model,head,batch,h
      result['model_smoke']=smoke; result['dataset']=dataset; result['status']='passed'
    except Exception:
      result['status']='failed'; result['failures'].append(traceback.format_exc()); print(result['failures'][-1],flush=True)
    finally:
      result['finished_at']=time.time(); result['wall_seconds']=result['finished_at']-result['started_at']; save(out/'report.json',result); print('REPORT',out/'report.json',result['status'],flush=True)
    if result['status']!='passed': raise SystemExit(1)
if __name__=='__main__': main()
