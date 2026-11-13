"""Run the resumable experiment suite on Modal (many small CPU containers in parallel).

    modal run scripts/modal_suite.py --phase phase5b            # 8-seed headline (distillation) jobs
    modal run scripts/modal_suite.py --phase phase4c            # remaining baselines at window 50
    modal run scripts/modal_suite.py --phase phase5b --limit 4  # smoke test

Same job definitions and same code path as scripts/run_suite.py (it imports run_suite.run). Results come back as the
same one-JSON-per-job files in results/suite/, so analysis.py / aggregate.py need no change. Jobs that already have a
local JSON are skipped. Everything runs on CPU in the container (jobs assigned to 'mps' locally are switched to 'cpu').
"""
import json
import shutil
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/pbl"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas", "scipy", "scikit-learn", extra_options="--extra-index-url https://download.pytorch.org/whl/cpu")
    .pip_install("torch", extra_options="--extra-index-url https://download.pytorch.org/whl/cpu")
    .add_local_dir(str(ROOT / "src"), f"{REMOTE}/src")
    .add_local_dir(str(ROOT / "data/raw"), f"{REMOTE}/data/raw")
    .add_local_dir(str(ROOT / "results/suite/ckpt"), f"{REMOTE}/ckpt_ro")
    .add_local_file(str(ROOT / "scripts/run_suite.py"), f"{REMOTE}/scripts/run_suite.py")
)
app = modal.App("spikedistill-rul", image=image)


@app.function(cpu=2, memory=4096, timeout=3 * 3600, retries=1, max_containers=100)
def run_job(job: dict):
    import os
    os.chdir(REMOTE)
    sys.path.insert(0, f"{REMOTE}/scripts")
    import run_suite
    run_suite.OUT = Path("/tmp/out"); run_suite.OUT.mkdir(parents=True, exist_ok=True)
    run_suite.CK = Path("/tmp/ck")
    if not run_suite.CK.exists():
        shutil.copytree(f"{REMOTE}/ckpt_ro", run_suite.CK)          # writable copy; teachers are read from here
    job = dict(job); job["device"] = "cpu"
    jid, msg = run_suite.run(job)
    f = run_suite.OUT / f"{jid.replace('>', '').replace('/', '_')}.json"
    new_ck = {p.name: p.read_bytes() for p in run_suite.CK.glob("*.pt") if not (Path(REMOTE) / "ckpt_ro" / p.name).exists()}   # teachers trained here
    return jid, msg, (f.read_text() if f.exists() else None), new_ck


@app.local_entrypoint()
def main(phase: str, only: str = "", exclude: str = "", limit: int = 0):
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_suite
    jobs = {"phase4c": run_suite.phase4c, "phase5a": run_suite.phase5a, "phase5b": run_suite.phase5b, "phase4": run_suite.phase4,
            "sweep": run_suite.sweep, "v2base": run_suite.v2base, "v2distill": run_suite.v2distill, "pooled": run_suite.pooled, "v2abl": run_suite.v2abl}[phase]()
    out = ROOT / "results/suite"
    name = lambda j: j["id"].replace(">", "").replace("/", "_") + ".json"
    pending = [j for j in jobs if only in j["id"] and not (exclude and exclude in j["id"]) and not (out / name(j)).exists()]
    if limit:
        pending = pending[:limit]
    print(f"{len(jobs)} jobs in {phase}, {len(pending)} pending -> Modal", flush=True)
    done = 0
    for res in run_job.map(pending, order_outputs=False, return_exceptions=True):
        done += 1
        if isinstance(res, Exception):
            print(f"[{done}/{len(pending)}] ERROR {res!r}"[:300], flush=True); continue
        jid, msg, text, new_ck = res
        for name, blob in new_ck.items():
            (out / "ckpt").mkdir(exist_ok=True); (out / "ckpt" / name).write_bytes(blob)
        if text:
            (out / (jid.replace(">", "").replace("/", "_") + ".json")).write_text(text)
        print(f"[{done}/{len(pending)}] {jid}: {msg}"[:240], flush=True)
