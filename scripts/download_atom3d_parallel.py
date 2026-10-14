#!/usr/bin/env python3
"""Resumable ranged downloader for the ATOM3D DIPS archive.

The final archive is produced only after every range has the expected length
and the concatenated file matches Zenodo's MD5 and byte count. Partial chunks
are retained so an interrupted run can resume without redownloading them.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import requests


URL = "https://zenodo.org/api/records/4911102/files/PPI-DIPS-split.tar.gz/content"
TOTAL = 13_568_121_746
MD5 = "b6095db528e4a2e1139fddc64b6965de"


def archive_md5(path):
    value = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fetch_chunk(args, index, total_chunks, parts, lock):
    start = index * args.chunk_size
    end = min(TOTAL, start + args.chunk_size)
    expected = end - start
    target = parts / ("chunk_%06d" % index)
    if target.is_file() and target.stat().st_size == expected:
        with lock:
            print("SKIP", index, expected, flush=True)
        return expected
    partial = target.with_suffix(".partial")
    if partial.exists():
        backup = partial.with_name(partial.name + ".stale.%d" % int(time.time()))
        os.replace(partial, backup)
    headers = {"Range": "bytes=%d-%d" % (start, end - 1)}
    last_error = None
    for attempt in range(1, args.retries + 1):
        try:
            with requests.get(URL, headers=headers, stream=True,
                              timeout=(60, args.timeout)) as response:
                response.raise_for_status()
                expected_range = "bytes %d-%d/%d" % (start, end - 1, TOTAL)
                if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                    raise RuntimeError("unexpected response %s %s" %
                                       (response.status_code, response.headers.get("Content-Range")))
                with partial.open("wb") as output:
                    for block in response.iter_content(8 * 1024 * 1024):
                        if block:
                            output.write(block)
                if partial.stat().st_size != expected:
                    raise RuntimeError("size %d expected %d" % (partial.stat().st_size, expected))
            os.replace(partial, target)
            with lock:
                print("DONE", index, expected, "of", total_chunks, flush=True)
            return expected
        except Exception as error:
            last_error = error
            if partial.exists():
                backup = partial.with_name(partial.name + ".failed.%d.%d" %
                                            (attempt, int(time.time())))
                os.replace(partial, backup)
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError("chunk %d failed after %d attempts: %s" %
                       (index, args.retries, last_error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/atom3d/zenodo-4911102"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 32:
        raise ValueError("workers must be between 1 and 32")
    if args.chunk_mib < 8:
        raise ValueError("chunk size must be at least 8 MiB")
    args.root.mkdir(parents=True, exist_ok=True)
    parts = args.root / ("PPI-DIPS-split.parts_%dMiB" % args.chunk_mib)
    parts.mkdir(exist_ok=True)
    final = args.root / "PPI-DIPS-split.tar.gz"
    if final.is_file() and final.stat().st_size == TOTAL and archive_md5(final) == MD5:
        print("already verified", final, flush=True)
        return
    chunk_size = args.chunk_mib * 1024 * 1024
    args.chunk_size = chunk_size
    total_chunks = (TOTAL + chunk_size - 1) // chunk_size
    manifest = args.root / "parallel_download_manifest.json"
    manifest.write_text(json.dumps(dict(url=URL, total_bytes=TOTAL, md5=MD5,
        chunk_size=chunk_size, workers=args.workers, total_chunks=total_chunks,
        started=time.time()), indent=2) + "\n")
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(fetch_chunk, args, index, total_chunks, parts, lock)
                for index in range(total_chunks)]
        completed = 0
        for future in concurrent.futures.as_completed(jobs):
            completed += future.result()
            print("PROGRESS_BYTES", completed, "of", TOTAL, flush=True)
    incoming = final.with_name(final.name + ".assembled.partial")
    if incoming.exists():
        incoming = final.with_name(final.name + ".assembled.%d.partial" % int(time.time()))
    with incoming.open("wb") as output:
        for index in range(total_chunks):
            target = parts / ("chunk_%06d" % index)
            expected = min(TOTAL, (index + 1) * chunk_size) - index * chunk_size
            if target.stat().st_size != expected:
                raise RuntimeError("chunk size changed before assembly: %s" % target)
            with target.open("rb") as source:
                for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
                    output.write(block)
    if incoming.stat().st_size != TOTAL or archive_md5(incoming) != MD5:
        raise RuntimeError("assembled archive failed size or MD5 verification")
    if final.exists():
        invalid = final.with_name(final.name + ".preexisting.%d" % int(time.time()))
        os.replace(final, invalid)
    os.replace(incoming, final)
    manifest.write_text(json.dumps(dict(url=URL, total_bytes=TOTAL, md5=MD5,
        chunk_size=chunk_size, workers=args.workers, total_chunks=total_chunks,
        finished=time.time(), status="verified", final=str(final)), indent=2) + "\n")
    print("VERIFIED", final, TOTAL, MD5, flush=True)


if __name__ == "__main__":
    main()
