# surface-residue-ppi execution constraints

- User-authorized exception (2026-09-24): the ATOM3D CPU data audit may run in `/home/zy/projects/Surface-residue` on the Linux workstation, with reports and reusable data synced to the cluster. Use an explicit local execution mode and record actual host, timings, and dependency versions. Model feature extraction and training still use Slurm.

- Run all model training, inference, feature extraction, smoke tests, and heavy preprocessing through Slurm.
- Use `hpc` only for CPU work and a GPU partition only for GPU work. Do not infer resource flags from old projects; record the actual `scontrol`/`sinfo` result in the run record.
- Do not modify `/datapool/home/zhangye/projects/GearNet`, `/datapool/home/zhangye/projects/Surface-GearNet`, or the shared `gearnet` environment.
- Shared code, checkpoints, data, and environments are read-only unless this project explicitly owns the path.
- Use exact project job IDs for cancellation. Never use `scancel -u`.
- Do not delete old logs, checkpoints, caches, or failed runs automatically.
- Do not print or store credentials, tokens, private keys, or passwords.
- Keep labels and partner coordinates outside encoder inputs. Preserve common masks and hashes for all comparisons.
- Do not claim dMaSIF or ESM2 availability until the actual environment, checkpoint, load result, and output tensor have been verified.
