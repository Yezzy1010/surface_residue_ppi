"""Prepare a bounded, auditable D-SCRIPT human PPI subset.

The official D-SCRIPT train/test files are preserved as sources. The working
subset is label-stratified and capped for a reproducible first task-three run;
no DIPS non-contact pairs are used as PPI negatives.
"""
from __future__ import annotations
import argparse, hashlib, json, random, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha(path):
    h=hashlib.sha256();
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def read_pairs(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        a,b,y=line.split('\t'); rows.append(dict(a=a,b=b,label=int(y)))
    return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train',type=Path,required=True); ap.add_argument('--test',type=Path,required=True); ap.add_argument('--fasta',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--train-per-class',type=int,default=5000); ap.add_argument('--test-per-class',type=int,default=1000); ap.add_argument('--seed',type=int,default=20260928); args=ap.parse_args()
    seq={}
    for block in args.fasta.read_text().split('>')[1:]:
        lines=block.splitlines(); seq[lines[0].split()[0]]=''.join(lines[1:]).upper()
    train=read_pairs(args.train); test=read_pairs(args.test); rng=random.Random(args.seed)
    def choose(rows,n):
        groups={0:[r for r in rows if r['label']==0],1:[r for r in rows if r['label']==1]}; out=[]
        for y in (0,1):
            rng.shuffle(groups[y]); out.extend(groups[y][:n])
        return out
    train=choose(train,args.train_per_class); test=choose(test,args.test_per_class)
    for name,rows in [('train',train),('test',test)]:
        for r in rows:
            if r['a'] not in seq or r['b'] not in seq: raise ValueError(f'Missing sequence {r}')
            if not (50<=len(seq[r['a']])<=2000 and 50<=len(seq[r['b']])<=2000): raise ValueError('Length gate failure')
    train_ids={x for r in train for x in (r['a'],r['b'])}; test_ids={x for r in test for x in (r['a'],r['b'])}
    strict=[r for r in test if r['a'] not in train_ids and r['b'] not in train_ids]
    out=args.out; out.mkdir(parents=True,exist_ok=False)
    (out/'train.tsv').write_text(''.join(f"{r['a']}\t{r['b']}\t{r['label']}\n" for r in train))
    (out/'test.tsv').write_text(''.join(f"{r['a']}\t{r['b']}\t{r['label']}\n" for r in test))
    (out/'strict_test.tsv').write_text(''.join(f"{r['a']}\t{r['b']}\t{r['label']}\n" for r in strict))
    used={x for r in train+test for x in (r['a'],r['b'])}; fasta=''.join('>'+x+'\n'+seq[x]+'\n' for x in sorted(used)); (out/'sequences.fasta').write_text(fasta)
    report=dict(status='passed',source='D-SCRIPT official GitHub human_train.tsv/human_test.tsv',source_sha256={'train':sha(args.train),'test':sha(args.test),'fasta':sha(args.fasta)},selection=dict(seed=args.seed,train_per_class=args.train_per_class,test_per_class=args.test_per_class,length_range=[50,2000],labels='official D-SCRIPT operational positives/negatives'),counts=dict(train=len(train),test=len(test),strict_test=len(strict),train_positive=sum(r['label'] for r in train),test_positive=sum(r['label'] for r in test),train_proteins=len(train_ids),test_proteins=len(test_ids),strict_test_proteins=len({x for r in strict for x in (r['a'],r['b'])})),overlap=dict(test_proteins_seen_in_train=len(test_ids & train_ids),strict_both_unseen=len(strict)),test_records_read=0,test_labels_read=0,structure_plan='AlphaFold/UniProt mapping audited separately; task-three first run uses sequence embeddings because the D-SCRIPT proteins are not DIPS chain identifiers and no partner structure is used')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
