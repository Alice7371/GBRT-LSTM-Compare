"""Augmentation ablation (review subplot C):

1. ±aug on 12 representative basins (h=1, jitter 0.05 + mask 10%)
2. jitter sensitivity sigma in {0.01, 0.05, 0.1, 0.2} on 4 basins
3. noise robustness: ±aug models evaluated with eval_noise=0.1 test inputs —
   does augmentation inject invariance ("external memory" operationalization)?
Output: study_results/aug.csv (columns: station, track, horizon, seed, years,
nse, kge, mape, rmse, seconds + tag column for the ablation arm).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

import data as D
import models as M
from hpt import pick_representatives

OUT = Path("study_results")
AUG = (0.05, 0.10)   # jitter sigma (normalized space), mask fraction
NOISE = 0.10         # eval-time probe noise


def run_arm(station, arm, **kw):
    df = D.load_station(station)
    t0 = time.time()
    res = M.run_lstm(df, None, "zscore", horizon=1, seed=0, **kw)
    res["seconds"] = round(time.time() - t0, 1)
    return dict(station=station, track=f"lstm-aug-{arm}", horizon=1, seed=0,
                years=0, tag=arm, **res)


def done_set(csv):
    if not csv.exists():
        return set()
    return set(pd.read_csv(csv, dtype={"station": str}).tag + "|" +
               pd.read_csv(csv, dtype={"station": str}).station)


def main():
    csv = OUT / "aug.csv"
    done = done_set(csv)

    def emit(station, arm, **kw):
        if f"{arm}|{station}" in done:
            return
        row = run_arm(station, arm, **kw)
        pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(),
                                   index=False)
        print(f"{station} {arm}: NSE={row['nse']:.3f} ({row['seconds']}s)",
              flush=True)

    # 1) ±aug on representatives
    REPS = pick_representatives()
    for s in REPS:
        emit(s, "clean")
        emit(s, "aug")
        emit(s, "aug_noise", aug=AUG, eval_noise=NOISE)
        emit(s, "clean_noise", eval_noise=NOISE)

    # 2) jitter sensitivity on 4 basins
    for s in REPS[:4]:
        for sig in (0.01, 0.05, 0.1, 0.2):
            emit(s, f"jitter{sig}", aug=(sig, 0.0))


if __name__ == "__main__":
    main()
