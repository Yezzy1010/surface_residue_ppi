"""Extract frozen ESM2 mean sequence embeddings for the audited task-three subset."""
from __future__ import annotations
import argparse, hashlib, json, os, resource, time
from pathlib import Path
import numpy as np
import torch

def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('.')); ap.add_argument('--protocol',type=Path,required=True); args=ap.parse_args(); root=args.root.resolve()
 if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available(): raise RuntimeError('GPU Slurm required')
 p=root/args.protocol; rows=json.loads((p/'report.json').read_text()); seq={}
 for block in (p/'sequences.fasta').read_text().split('>')[1:]:
  lines=block.splitlines(); seq[lines[0].split()[0]]=''.join(lines[1:]).upper()
 import esm
 model,alphabet=esm.pretrained.esm2_t12_35M_UR50D(); model=model.cuda().eval(); converter=alphabet.get_batch_converter(); model.requires_grad_(False)
 keys=sorted(seq); emb=[]; started=time.time();
 with torch.no_grad():
  for start in range(0,len(keys),4):
   selected=keys[start:start+4]; _,_,tokens=converter([(k,seq[k]) for k in selected]); tokens=tokens.cuda(); out=model(tokens,repr_layers=[12],return_contacts=False)['representations'][12]
   for i,k in enumerate(selected): emb.append(out[i,1:len(seq[k])+1].mean(0).float().cpu().numpy())
 outdir=p/'esm_embeddings'; outdir.mkdir(exist_ok=False); np.savez_compressed(outdir/'embeddings.npz',keys=np.asarray(keys),embeddings=np.asarray(emb,dtype=np.float32)); report=dict(status='passed',job_id=os.environ['SLURM_JOB_ID'],protocol=str(args.protocol),protocol_report_sha256=sha(p/'report.json'),esm_variant='esm2_t12_35M_UR50D',esm_layer=12,embedding_dim=480,proteins=len(keys),test_records_read=0,test_labels_read=0,sha256=sha(outdir/'embeddings.npz'),wall_seconds=time.time()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss); (outdir/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
