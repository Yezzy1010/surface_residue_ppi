"""Download and byte-verify official weights; never import or execute a model."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def file_hashes(path):
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest(),
            "size_bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--direct", action="store_true", help="Bypass proxy settings for this download only")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "configs/dependencies/esm_gearnet.json").read_text())
    checkpoint = manifest["checkpoint"]
    directory = root / checkpoint["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / checkpoint["filename"]
    partial = destination.with_suffix(destination.suffix + ".partial")
    records = directory / "download_records"
    records.mkdir(exist_ok=True)
    started = time.time()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    record_path = records / (stamp + "_" + str(os.getpid()) + ".json")
    record = dict(status="started", started_at_utc=stamp, host=platform.node(),
                  python=platform.python_version(), execution="file-download-and-checksum-only",
                  url=checkpoint["url"], expected_md5=checkpoint["md5"],
                  destination=str(destination), source_commit=manifest["source"]["commit"],
                  model_loaded=False, tensor_output_verified=False)
    exit_code = 1
    try:
        with (directory / "download.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if destination.is_file():
                candidate = destination
                record["reused_existing_file"] = True
            elif args.verify_only:
                raise FileNotFoundError("Checkpoint is not downloaded: " + str(destination))
            else:
                candidate = partial
                command = ["curl", "--fail", "--location", "--silent", "--show-error",
                           "--continue-at", "-", "--retry", "2", "--retry-delay", "2",
                           "--retry-all-errors", "--connect-timeout", "20"]
                # Old system curl versions lack --retry-all-errors.
                version = subprocess.check_output(["curl", "--version"], text=True).splitlines()[0]
                record["curl"] = version
                curl_version = tuple(int(v) for v in version.split()[1].split(".")[:2])
                if curl_version < (7, 71):
                    command.remove("--retry-all-errors")
                if args.direct:
                    command += ["--noproxy", "*"]
                command += ["--output", str(partial), checkpoint["url"]]
                print("Downloading " + checkpoint["filename"] + " (partial files are retained)", flush=True)
                result = subprocess.run(command, check=False)
                record["curl_exit_code"] = result.returncode
                if result.returncode:
                    raise RuntimeError("Official download failed; curl exit code " + str(result.returncode))
            digests = file_hashes(candidate)
            record.update(digests)
            if digests["md5"] != checkpoint["md5"]:
                raise ValueError("Published MD5 mismatch; original file has been retained")
            if candidate == partial:
                if destination.exists():
                    raise FileExistsError("Refusing to replace an existing checkpoint")
                partial.rename(destination)
            record["status"] = "checksum_verified"
            exit_code = 0
    except (Exception, KeyboardInterrupt) as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        record.update(wall_seconds=time.time() - started,
                      finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      partial_size_bytes=partial.stat().st_size if partial.exists() else 0)
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"status": record["status"], "record": str(record_path)}, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
