"""Frozen ESM2, GearNet and dMaSIF features for audited independent monomers."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from surface_residue_ppi.encoders import FrozenGearNet, load_surface, surface_features
from surface_residue_ppi.structures import read_monomer
from surface_residue_ppi.adapters import neighborhoods
from train_phase2_task1 import sha256
from run_phase3_diagnostics import save


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--shard-index',type=int,default=0);ap.add_argument('--shard-count',type=int,default=1)
    args=ap.parse_args();root=Path('.').resolve()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm GPU required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not 0<=args.shard_index<args.shard_count:raise ValueError('Invalid shard')
    config=json.loads((root/'configs/smoke.json').read_text())
    rows=json.loads(args.manifest.read_text())
    rows=sorted(rows,key=lambda r:r['id'])[args.shard_index::args.shard_count]
    base=root/'runs/phase3_task3_features/structure_v1'
    out=base/('job_'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True,exist_ok=False)
    cache=base/'cache';cache.mkdir(exist_ok=True)
    # KeOps 1.x ignores PYKEOPS_BUILD_FOLDER and may reuse a module compiled
    # for a different GPU at the same CUDA_VISIBLE_DEVICES index.
    import pykeops
    import sys
    from pykeops.common.set_path import set_bin_folder
    old_keops_folder=pykeops.bin_folder
    arch='sm_%d%d'%torch.cuda.get_device_capability()
    set_bin_folder(str(root/'data/cache'/('keops_task3_'+arch)))
    sys.path[:]=[p for p in sys.path if p!=old_keops_folder]
    sys.path.insert(0,pykeops.bin_folder)
    report=dict(status='running',job_id=os.environ['SLURM_JOB_ID'],array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                host=platform.node(),gpu=torch.cuda.get_device_name(),manifest=str(args.manifest),
                manifest_sha256=sha256(args.manifest),source_sha256=sha256(Path(__file__)),
                encoder_config_sha256=sha256(root/'configs/smoke.json'),records_requested=len(rows),
                started_at=time.time(),records=[],failures=[],test_records_read=0,test_labels_read=0,
                keops_build_folder=pykeops.bin_folder,keops_version=pykeops.__version__)
    save(out/'report.json',report)
    import esm
    surface,surface_checkpoint=load_surface(root,config['dmasif'])
    gearnet=FrozenGearNet(config['gearnet_checkpoint']).cuda()
    seqmodel,alphabet=getattr(esm.pretrained,config['esm_variant'])()
    seqmodel=seqmodel.cuda().eval().requires_grad_(False);convert=alphabet.get_batch_converter()
    esm_checkpoint=Path(torch.hub.get_dir())/'checkpoints'/(config['esm_variant']+'.pt')
    hashes=dict(esm=sha256(esm_checkpoint),gearnet=sha256(Path(config['gearnet_checkpoint'])),dmasif=sha256(surface_checkpoint))
    report['checkpoint_sha256']=hashes
    for i,row in enumerate(rows,1):
        try:
            pdb=root/row['pdb']
            if sha256(pdb)!=row['pdb_sha256']:raise ValueError('PDB hash mismatch')
            npz=cache/(row['id']+'.npz');meta=npz.with_suffix('.json')
            if meta.exists():
                cached=json.loads(meta.read_text())
                if cached['pdb_sha256']!=row['pdb_sha256'] or cached['checkpoint_sha256']!=hashes or cached['cache_sha256']!=sha256(npz) or cached['encoder_config_sha256']!=report['encoder_config_sha256']:
                    raise ValueError('Cache provenance changed')
                report['records'].append(dict(cached,reused=True))
            else:
                started=time.time();parsed=read_monomer(pdb)
                if parsed['sequence']!=row['sequence']:raise ValueError('Sequence differs from audited D-SCRIPT input')
                n=len(parsed['sequence'])
                atoms=torch.from_numpy(parsed['atom_xyz']).cuda()
                atom_ids=torch.from_numpy(parsed['atom_to_residue']).cuda()
                seed=int(hashlib.sha256(('20260928:'+row['id']).encode()).hexdigest()[:8],16)%(2**31)
                point=surface_features(surface,atoms,torch.from_numpy(parsed['atom_types']).cuda(),seed)
                with torch.no_grad():
                    raw,keys,positions=gearnet(pdb)
                    lookup={key:i for i,key in enumerate(parsed['residue_keys'])}
                    if len(keys)!=n or set(keys)!=set(lookup):raise ValueError('GearNet common-mask mismatch')
                    mapping=torch.tensor([lookup[k] for k in keys],device='cuda')
                    if not torch.allclose(positions-torch.from_numpy(parsed['center']).cuda(),torch.from_numpy(parsed['ca_xyz']).cuda()[mapping],atol=1e-3):
                        raise ValueError('GearNet residue coordinate mismatch')
                    g=torch.zeros(n,raw.shape[1],device='cuda');g[mapping]=raw
                    _,_,tokens=convert([(row['id'],parsed['sequence'])])
                    e=seqmodel(tokens.cuda(),repr_layers=[config['esm_layer']],return_contacts=False)['representations'][config['esm_layer']][0,1:n+1]
                a,b,owner=neighborhoods(atoms,atom_ids,point['xyz'],n,**config['mapping'])
                data=dict(esm=e.cpu().numpy(),gearnet=g.cpu().numpy(),surface=point[config['dmasif_branch']].cpu().numpy(),
                          surface_xyz=point['xyz'].cpu().numpy(),common_valid=np.ones(n,dtype=bool),
                          point_owner=owner.cpu().numpy(),atom_xyz=parsed['atom_xyz'],atom_to_seq=parsed['atom_to_residue'])
                for name,edges in [('A',a),('BC',b)]:
                    data.update({name+'_residue':edges.residue.cpu().numpy(),name+'_point':edges.point.cpu().numpy(),
                                 name+'_distance':edges.distance.cpu().numpy()})
                if data['esm'].shape!=(n,480) or data['gearnet'].shape!=(n,3072) or data['surface'].shape[1]!=16:
                    raise ValueError('Unexpected feature dimension')
                if any(not np.isfinite(x).all() for x in data.values()):raise ValueError('Nonfinite cache')
                temp=npz.with_suffix('.partial.npz');np.savez_compressed(temp,**data);temp.replace(npz)
                result=dict(id=row['id'],cache=str(npz.relative_to(root)),cache_sha256=sha256(npz),
                            pdb_sha256=row['pdb_sha256'],sequence_sha256=row['sequence_sha256'],
                            encoder_config_sha256=report['encoder_config_sha256'],checkpoint_sha256=hashes,
                            length=n,surface_points=len(data['surface']),surface_seed=seed,seconds=time.time()-started)
                save(meta,result);report['records'].append(result)
                del point,raw,g,e,data;torch.cuda.empty_cache()
            print('PROTEIN',i,len(rows),row['id'],'passed',flush=True)
        except Exception as error:
            import traceback
            failure=dict(id=row['id'],error=traceback.format_exc());report['failures'].append(failure)
            print('FAILURE',json.dumps(failure),flush=True)
        save(out/'report.json',report)
    report.update(status='failed' if report['failures'] else 'passed',wall_seconds=time.time()-report['started_at'],
                  peak_gpu_bytes=torch.cuda.max_memory_allocated())
    save(out/'report.json',report)
    if report['failures']:raise SystemExit(1)


if __name__=='__main__':main()
