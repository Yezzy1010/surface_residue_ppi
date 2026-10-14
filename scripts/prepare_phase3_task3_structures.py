"""Resume exact-sequence AlphaFold structure audit using D-SCRIPT TRAIN only."""
from __future__ import annotations
import argparse
import hashlib
import gzip
import json
import platform
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, d):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)


def fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=30) as r:
                data=r.read()
                return gzip.decompress(data) if data.startswith(b'\x1f\x8b') else data
        except urllib.error.HTTPError as e:
            if e.code in (404,422):return None
            if attempt==2:raise
        except Exception:
            if attempt==2:raise
        time.sleep(attempt+1)


def work(row, out):
    p=out/'records'/(row['id']+'.json')
    if p.exists():
        old=json.loads(p.read_text())
        if old['status']!='failed':
            if old['sequence_sha256'] != row['sequence_sha256']:
                raise ValueError('Resume input changed')
            if old['status']=='ready' and sha(old['pdb'])!=old['pdb_sha256']:
                raise ValueError('Saved PDB changed')
            return old
    r=dict(row,status='excluded',attempts=[])
    try:
        candidates=[a for a in row['accessions'] if '-' not in a]
        if not candidates:r['reason']='no_canonical_uniprot_exact_sequence'
        for accession in candidates:
            api='https://alphafold.ebi.ac.uk/api/prediction/'+accession
            raw=fetch(api)
            if raw is None:
                r['attempts'].append(dict(accession=accession,result='no_afdb_entry'));continue
            metadata_path=out/'metadata'/(accession+'.json')
            metadata_path.parent.mkdir(exist_ok=True)
            metadata_path.write_bytes(raw)
            entries=json.loads(raw)
            exact=[m for m in entries if m.get('sequence')==row['sequence'] and
                   m.get('sequenceStart')==1 and m.get('sequenceEnd')==len(row['sequence'])
                   and m.get('pdbUrl')]
            if not exact:
                r['attempts'].append(dict(accession=accession,result='no_full_exact_sequence_model'));continue
            m=sorted(exact,key=lambda v: str(v.get('modelEntityId')))[0]
            pdb=out/'pdb'/(row['id']+'.pdb');pdb.parent.mkdir(exist_ok=True)
            content=fetch(m['pdbUrl'])
            if content is None:raise ValueError('Listed PDB unavailable')
            tmp=pdb.with_suffix('.partial.pdb');tmp.write_bytes(content)
            models=list(PDBParser(QUIET=True).get_structure(row['id'],str(tmp)))
            if len(models)!=1 or len(list(models[0]))!=1:raise ValueError('Not a single monomer')
            chain=next(iter(models[0]));res=list(chain)
            seq=''.join(seq1(x.resname) for x in res)
            if seq!=row['sequence']:raise ValueError('PDB sequence differs from D-SCRIPT')
            if any(x.id[0]!=' ' or x.id[2]!=' ' or 'CA' not in x for x in res):
                raise ValueError('Nonstandard residue, insertion or missing CA')
            if [x.id[1] for x in res]!=list(range(1,len(res)+1)):
                raise ValueError('Noncontiguous residue numbering')
            ca=np.array([x['CA'].coord for x in res])
            if not np.isfinite(ca).all():raise ValueError('Nonfinite coordinates')
            confidence=np.array([x['CA'].bfactor for x in res])
            tmp.replace(pdb)
            r.update(accession=accession,api=api,metadata=str(metadata_path),
                     metadata_sha256=sha(metadata_path),pdb=str(pdb),pdb_sha256=sha(pdb),
                     pdb_url=m['pdbUrl'],afdb_version=m.get('latestVersion'),
                     model_created=m.get('modelCreatedDate'),chain=chain.id,length=len(res),
                     sequence_coverage=1.0,mean_plddt=float(confidence.mean()),
                     fraction_plddt_below_50=float((confidence<50).mean()),
                     structure_source='AlphaFold DB independent monomer')
            if confidence.mean()<70 or (confidence<50).mean()>0.3:
                r['reason']='mean_plddt_below_70_or_low_confidence_fraction_above_0.30'
            else:r['status']='ready'
            break
        if r['status']=='excluded' and 'reason' not in r:
            r['reason']='no_full_exact_sequence_alphafold_model'
    except Exception as e:
        r.update(status='failed',error=type(e).__name__+': '+str(e))
    save(p,r)
    return r


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path('runs/phase3_task3_structure_audit/20260928/exact_v1'))
    ap.add_argument('--workers',type=int,default=6)
    args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    started=time.time();protocol=Path('runs/phase3_task3_protocol/20260928/subset_v1')
    train=protocol/'train.tsv'
    ids={p for line in train.read_text().splitlines() for p in line.split('\t')[:2]}
    seq={str(r.id):str(r.seq) for r in SeqIO.parse(str(protocol/'sequences.fasta'),'fasta') if str(r.id) in ids}
    if ids!=set(seq):raise ValueError('Missing training sequence')
    reference=Path('data/task3/uniprot/human_reference_isoforms.fasta')
    matches={s:[] for s in seq.values()}
    for r in SeqIO.parse(str(reference),'fasta'):
        if str(r.seq) in matches:matches[str(r.seq)].append(str(r.id).split('|')[1])
    rows=[dict(id=k,sequence=seq[k],sequence_sha256=hashlib.sha256(seq[k].encode()).hexdigest(),
               accessions=sorted(set(matches[seq[k]]),key=lambda a:('-' in a,len(a),a))) for k in sorted(ids)]
    audit=dict(status='running',execution='authorized Linux workstation CPU/network',
               host=platform.node(),started_at=started,train_sha256=sha(train),
               uniprot_sha256=sha(reference),source_sha256=sha(Path(__file__)),
               unique_train_proteins=len(rows),test_records_read=0,test_labels_read=0,
               exact_uniprot_proteins=sum(bool(r['accessions']) for r in rows),
               rules={'mapping':'100% full-sequence exact identity; canonical UniProt only',
                      'structure':'single complete monomer, no partner input',
                      'mean_plddt_min':70,'fraction_plddt_below_50_max':0.30,
                      'sequence_length':'50-2000, inherited frozen train subset'},
               counts={'ready':0,'excluded':0,'failed':0})
    mapping=out/'mapping.json'
    if mapping.exists() and json.loads(mapping.read_text())!=rows:raise ValueError('Mapping drift')
    if not mapping.exists():save(mapping,rows)
    save(out/'report.json',audit)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(work,row,out) for row in rows]
        for i,future in enumerate(as_completed(futures),1):
            result=future.result();audit['counts'][result['status']]+=1
            if i%100==0 or i==len(rows):
                audit.update(processed=i,wall_seconds=time.time()-started)
                save(out/'report.json',audit);print('PROGRESS',i,audit['counts'],flush=True)
    audit.update(status='needs_retry' if audit['counts']['failed'] else 'passed',
                 finished_at=time.time(),wall_seconds=time.time()-started)
    save(out/'report.json',audit);print('REPORT',json.dumps(audit),flush=True)


if __name__=='__main__':main()
