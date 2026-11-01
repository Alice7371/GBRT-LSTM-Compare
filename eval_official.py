"""Evaluate official CAMELS model outputs (NWS HL-RMS, Daymet forcing) under
our exact protocol: window 1990-2009, chronological 80/20 split, test segment.

Per gauge there are ~11 calibrated variants (<gauge>_<nn>_model_output.txt).
Selection rule: best TRAIN-segment NSE variant per gauge (selection on train
only; test untouched). Output: study_results/official_models.csv.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

import data as D

OUT = Path("study_results")
ROOT = Path(r"E:\ProjectFile\camels_modelOutput\model_output_daymet\model_output\flow_timeseries\daymet")


def eval_file(f: Path) -> dict:
    df = pd.read_csv(f, sep=r"\s+", skiprows=1, header=None,
                     names=["YR", "MNTH", "DY", "HR", "SWE", "PRCP", "RAIM",
                            "TAIR", "PET", "ET", "MOD_RUN", "OBS_RUN"],
                     usecols=["YR", "MNTH", "DY", "MOD_RUN", "OBS_RUN"])
    df["Date"] = pd.to_datetime(dict(year=df.YR, month=df.MNTH, day=df.DY))
    df = df[(df.Date >= "1990-01-01") & (df.Date <= "2009-12-31")]
    df = df.sort_values("Date").reset_index(drop=True)
    if len(df) < 3000:  # incomplete record for our window
        return None
    split = D.split_index(len(df))
    y_obs = df.OBS_RUN.values.astype(float)
    y_mod = df.MOD_RUN.values.astype(float)
    tr_nse = D.nse(y_obs[:split], y_mod[:split])
    te = D.evaluate(y_obs[split:], y_mod[split:])
    return {"train_nse": tr_nse, **te}


def main():
    csv = OUT / "official_models.csv"
    files = sorted(ROOT.glob("*/*_model_output.txt"))
    print(f"{len(files)} model output files")
    # group by gauge (8-digit prefix)
    by_gauge = {}
    for f in files:
        g = f.name.split("_")[0]
        by_gauge.setdefault(g, []).append(f)
    print(f"{len(by_gauge)} gauges")

    rows, t0 = [], time.time()
    for i, (g, fs) in enumerate(sorted(by_gauge.items()), 1):
        best = None
        for f in fs:
            try:
                r = eval_file(f)
            except Exception:
                continue
            if r and (best is None or r["train_nse"] > best["train_nse"]):
                best = r
        if best is None:
            continue
        rows.append(dict(station=g, track="official-HLRMS", horizon=1,
                         seed=0, years=0, **{k: v for k, v in best.items()
                                             if k != "train_nse"}))
        if i % 100 == 0:
            print(f"[{i}/{len(by_gauge)}] rows={len(rows)} ({time.time()-t0:.0f}s)",
                  flush=True)
    pd.DataFrame(rows).to_csv(csv, index=False)
    d = pd.DataFrame(rows)
    print(f"official-HLRMS: {len(d)} gauges | median test NSE {d.nse.median():.3f} "
          f"| {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
