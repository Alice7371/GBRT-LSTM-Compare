"""Evaluate official CAMELS calibrated model outputs (Daymet forcing variant)
under our exact protocol: test segment = last 20% of 1990-2009, chronological.

The official per-basin model output files contain daily QObs and simulations
from SAC-SMA, VIC, HBV, FUSE, mHM, NWM(?) — column names detected dynamically.
Output: study_results/official_models.csv (schema-compatible).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

import data as D

OUT = Path("study_results")
MO_DIR = Path("data_official/modelOutput_daymet")


def eval_station(csv_path: Path, station: str) -> list[dict]:
    df = pd.read_csv(csv_path, sep=r"\s+", skiprows=1, header=None, dtype=float)
    cols = pd.read_csv(csv_path, sep=r"\s+", nrows=0).columns.tolist()
    df.columns = cols
    # locate date + observed + simulated columns
    qcol = [c for c in cols if c.upper().startswith("QObs")][0]
    date_cols = ["YYYY", "MM", "DD"]
    df["Date"] = pd.to_datetime(dict(
        year=df[date_cols[0]].astype(int), month=df[date_cols[1]].astype(int),
        day=df[date_cols[2]].astype(int)))
    df = df.sort_values("Date").reset_index(drop=True)

    split = D.split_index(len(df))
    rows = []
    sim_cols = [c for c in cols if c not in date_cols and c != qcol
                and df[c].notna().sum() > 0.5 * len(df)]
    for c in sim_cols:
        model = c.split("(")[0].strip()
        y_true = df[qcol].values[split:]
        y_hat = df[c].values[split:]
        ok = ~np.isnan(y_true) & ~np.isnan(y_hat)
        if ok.sum() < 0.8 * len(y_true):
            continue
        m = D.evaluate(y_true[ok], y_hat[ok])
        rows.append(dict(station=station, track=f"official-{model}", horizon=1,
                         seed=0, years=0, **m))
    return rows


def main():
    csv = OUT / "official_models.csv"
    files = sorted(MO_DIR.glob("*_modelOutput.txt"))
    print(f"found {len(files)} model output files")
    done = set()
    if csv.exists():
        d0 = pd.read_csv(csv, dtype={"station": str})
        done = set(d0.station)
        print(f"{len(done)} stations already evaluated")
    t0 = time.time()
    n_rows = 0
    for i, f in enumerate(files, 1):
        station = f.name.split("_")[0]
        if station in done:
            continue
        try:
            rows = eval_station(f, station)
            if rows:
                pd.DataFrame(rows).to_csv(csv, mode="a",
                                          header=not csv.exists(), index=False)
                n_rows += len(rows)
        except Exception as e:
            print(f"FAIL {station}: {e}", flush=True)
        if i % 50 == 0:
            print(f"[{i}/{len(files)}] rows={n_rows} ({time.time()-t0:.0f}s)",
                  flush=True)
    print(f"official models: {n_rows} new rows, {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
