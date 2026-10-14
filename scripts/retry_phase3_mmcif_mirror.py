"""Finish missing candidate mmCIF downloads from the official PDBe mirror."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def main():
    root = Path('.').resolve()
    if root != Path('/home/zy/projects/Surface-residue'):
        raise RuntimeError('Authorized workstation only')
    out = root / 'runs/phase3_mmcif_prefetch/20260927/pdbe_v1'
    out.mkdir(parents=True, exist_ok=False)
    candidate = json.loads((root/'runs/phase3_candidates/20260927/expanded_v1/report.json').read_text())
    codes = [r['pdb'] for s in ('train','val') for r in candidate['selection'][s]
             if not (root/'data/rcsb/mmcif'/(r['pdb'].lower()+'.cif')).exists()]

    def fetch(code):
        target = root/'data/rcsb/mmcif'/(code.lower()+'.cif')
        url = 'https://www.ebi.ac.uk/pdbe/entry-files/download/'+code.lower()+'.cif'
        result = dict(code=code,url=url,started_at=time.time())
        try:
            with tempfile.TemporaryDirectory(dir=out) as temp:
                path = Path(temp)/'complete.cif'
                subprocess.run(['curl','--fail','--silent','--show-error','--location',
                    '--connect-timeout','10','--max-time','90','--retry','1',
                    '--output',str(path),url],capture_output=True,check=True)
                payload = path.read_bytes()
                if not payload.lstrip().lower().startswith(('data_'+code.lower()).encode()):
                    raise ValueError('Unexpected mmCIF data block identity')
                try:
                    os.link(path,target)
                    result['status']='published'
                except FileExistsError:
                    result['status']='already_present'
                result.update(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
        except Exception as error:
            result.update(status='failed',error=str(error),stderr=getattr(error,'stderr',b'').decode(errors='replace'))
        result['seconds']=time.time()-result['started_at']
        (out/(code.lower()+'.json')).write_text(json.dumps(result)+'\n')
        print(json.dumps(result),flush=True)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(fetch,codes))
    (out/'report.json').write_text(json.dumps(dict(requested=len(codes),results=results,test_records_read=0),indent=2)+'\n')


if __name__=='__main__':
    main()
