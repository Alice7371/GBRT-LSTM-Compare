"""Overnight queue driver: executes all remaining stages in priority order.
Each stage is resumable on its own, so the driver can be killed and restarted.
Sharded stages use 2 shards (3-way sharding oversubscribed the 8 GB card).

Priority order (review guidance: h<=30 completeness first):
  1. pooled short  (h1/3/7 variants)          ~3.5h
  2. pooled xgb    (xgb-pooled family)        ~1.5h
  3. extend 30     (stride 2, 3 shards)       ~2h
  4. pooled mid    (h10/15)                   ~1.5h
  5. pooled long   (h30)                      ~1.5h
  6. hpt           (Optuna 50 trials)         ~1h
  7. pooled volume (5y/10y)                   ~0.8h
  8. extend 90/180 (stride 4, 3 shards)       ~2.5h
  9. pooled xseasonal (h90/180, last)         ~2h
"""
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PY = sys.executable
OUT = Path("study_results")
LOG = OUT / "queue.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def rows(csv):
    p = OUT / csv
    return len(pd.read_csv(p)) if p.exists() else 0


def wait_for(shard_csvs, target):
    while any(rows(c) < target for c in shard_csvs):
        time.sleep(120)
    log(f"wait_for {shard_csvs} reached {target}")


def run(stage, cmd):
    log(f"stage {stage}: start")
    t0 = time.time()
    r = subprocess.run(cmd)
    log(f"stage {stage}: exit {r.returncode} after {(time.time()-t0)/60:.1f} min")
    return r.returncode


def run_parallel(stage, cmds):
    log(f"stage {stage}: start ({len(cmds)} parallel)")
    t0 = time.time()
    procs = [subprocess.Popen(c) for c in cmds]
    codes = [p.wait() for p in procs]
    log(f"stage {stage}: exits {codes} after {(time.time()-t0)/60:.1f} min")
    return codes


def main():
    # 0) extend(10,15): 2 shards (3-way sharding oversubscribed the 8GB card)
    run_parallel("extend-10-15", [
        [PY, "run_experiments.py", "extend", "--horizons", "10", "15",
         "--shard", str(i), "--nshards", "2"] for i in range(2)])

    # 1-2) pooled short + xgb (single GPU process each, sequential)
    run("pooled-short", [PY, "pooled.py", "--stage", "short"])
    run("pooled-xgb", [PY, "pooled.py", "--stage", "xgb"])

    # 3) extend h30, stride 2, 2 shards
    run_parallel("extend-30", [
        [PY, "run_experiments.py", "extend", "--horizons", "30", "--stride", "2",
         "--shard", str(i), "--nshards", "2"] for i in range(2)])

    # 4-5) pooled mid + long
    run("pooled-mid", [PY, "pooled.py", "--stage", "mid"])
    run("pooled-long", [PY, "pooled.py", "--stage", "long"])

    # 6) HPT
    run("hpt", [PY, "hpt.py"])

    # 7) volume axis
    run("pooled-volume", [PY, "pooled.py", "--stage", "volume"])

    # 8) extend h90/180, stride 4, 2 shards
    run_parallel("extend-90-180", [
        [PY, "run_experiments.py", "extend", "--horizons", "90", "180",
         "--stride", "4", "--shard", str(i), "--nshards", "2"] for i in range(2)])

    # 9) pooled xseasonal (first to cut if time runs out)
    run("pooled-xseasonal", [PY, "pooled.py", "--stage", "xseasonal"])

    # 10) augmentation ablation (CPU-light, GPU trainings ~30 small runs)
    run("aug", [PY, "aug.py"])
    log("ALL STAGES COMPLETE")


if __name__ == "__main__":
    main()
