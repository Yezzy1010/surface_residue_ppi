"""Fetch official compressed mmCIFs for the frozen candidates on the workstation.

Safe to run alongside preparation: publish complete validated files atomically,
never replace an existing mmCIF, and retain URL/digest evidence for each download.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def fetch(code, root, evidence):
    target = root / 'data/rcsb/mmcif' / (code.lower() + '.cif')
    if target.exists():
        return dict(code=code, status='already_present')
    url = 'https://files.rcsb.org/download/' + code.upper() + '.cif.gz'
    started = time.time()
    result = dict(code=code, url=url)
    try:
        with tempfile.TemporaryDirectory(prefix='prefetch_', dir=evidence) as temporary:
            packed = Path(temporary) / 'source.cif.gz'
            subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location',
                            '--connect-timeout', '10', '--max-time', '40', '--retry', '1',
                            '--output', str(packed), url], check=True, capture_output=True)
            payload = gzip.decompress(packed.read_bytes())
            if not payload.lstrip().lower().startswith(('data_' + code.lower()).encode()):
                raise ValueError('Downloaded mmCIF has unexpected data block identity')
            complete = Path(temporary) / 'complete.cif'
            complete.write_bytes(payload)
            try:
                os.link(complete, target)
                status = 'published'
            except FileExistsError:
                status = 'concurrent_download_already_published'
            result.update(status=status, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                          compressed_sha256=hashlib.sha256(packed.read_bytes()).hexdigest())
    except Exception as error:
        result.update(status='failed', error=str(error),
                      stderr=getattr(error, 'stderr', b'').decode(errors='replace')[-1000:])
    result['seconds'] = time.time()-started
    (evidence / (code.lower() + '.json')).write_text(json.dumps(result) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    root = Path('.').resolve()
    if root != Path('/home/zy/projects/Surface-residue'):
        raise RuntimeError('Use the authorized workstation')
    args.out.mkdir(parents=True, exist_ok=False)
    candidate = json.loads(args.candidates.read_text())
    rows = candidate['selection']['train'] + candidate['selection']['val']
    if any(r['split'] not in ('train', 'val') for r in rows):
        raise ValueError('Test is sealed')
    # Audit processes candidates from the front; prefetch from the end to avoid
    # duplicating its in-flight requests whenever possible.
    codes = list(dict.fromkeys(r['pdb'] for r in reversed(rows)))
    codes = [c for c in codes if not (root / 'data/rcsb/mmcif' / (c.lower()+'.cif')).exists()]
    started = time.time()
    counts = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, c, root, args.out) for c in codes]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            counts[result['status']] = counts.get(result['status'], 0) + 1
            if i % 50 == 0:
                print(json.dumps(dict(completed=i, requested=len(codes), counts=counts,
                                      seconds=time.time()-started)), flush=True)
    report = dict(requested=len(codes), counts=counts, wall_seconds=time.time()-started,
                  candidate_sha256=hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
                  source='official RCSB compressed mmCIF; TLS validation enabled', test_records_read=0)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('REPORT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
