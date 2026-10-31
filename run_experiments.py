"""Experiment scheduler for the scaled-up study (multi-horizon, feature
engineering, data-volume axes). Resumable: completed (station, track, horizon,
seed, years) rows in each shard CSV are skipped on re-run.

Phase main:   all basins x tracks x horizons (h=1/3/7; min-max LSTM only h=1)
Phase volume: basin subset x {xgb-raw, xgb-fe, lstm-raw-zscore} x h=1 x years {5,10}
"""
import argparse
import time
from pathlib import Path

import pandas as pd

import data as D
import models as M

OUT = Path("study_results")
WINDOW = 15  # window sensitivity was flat over 15-50 in pilot runs

MAIN_TRACKS = [
    ("persistence", {}),
    ("xgb-raw", {}),
    ("xgb-fe", {}),
    ("lstm-raw-zscore", {}),
    ("lstm-raw-minmax", {"horizons": [1]}),   # normalization contrast at h=1
    ("lstm-fe-zscore", {}),
]
HORIZONS = [1, 3, 7]


def done_keys(csv: Path):
    if not csv.exists():
        return set()
    df = pd.read_csv(csv)
    return set(zip(df.station, df.track, df.horizon, df.seed, df.years))


def append_row(csv: Path, row: dict):
    pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(), index=False)


def dispatch(track: str, df, horizon: int, seed: int, years: int, epochs: int):
    t0 = time.time()
    if track == "persistence":
        res = M.run_persistence(df, horizon)
    elif track == "climatology":
        res = M.run_climatology(df, horizon)
    elif track == "xgb-raw":
        res = M.run_xgb(df, None, horizon, fe=False, years=years)
    elif track == "xgb-fe":
        res = M.run_xgb(df, None, horizon, fe=True, years=years)
    elif track == "xgb-fe-long":
        res = M.run_xgb(df, None, horizon, fe=True, long=True, years=years)
    elif track.startswith("lstm-"):
        parts = track.split("-")  # lstm raw|fe[-long] zscore|minmax
        fe = "fe" in parts
        long_ = "long" in parts
        norm = parts[-1]
        res = M.run_lstm(df, None, norm, horizon=horizon, fe=fe, long=long_,
                         seed=seed, years=years, epochs=epochs)
    else:
        raise ValueError(track)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def sweep(stations, jobs, csv: Path, epochs: int = None):
    done = done_keys(csv)
    total = len(jobs)
    k = 0
    for i, station in enumerate(stations, 1):
        try:
            df = D.load_station(station)
        except Exception as e:
            print(f"[{i}/{len(stations)}] {station} LOAD FAIL: {e}", flush=True)
            continue
        for track, horizon, seed, years in jobs:
            k += 1
            if (station, track, horizon, seed, years) in done:
                continue
            try:
                res = dispatch(track, df, horizon, seed, years, epochs)
            except Exception as e:
                print(f"  FAIL {station} {track} h{horizon} y{years}: {e}", flush=True)
                continue
            append_row(csv, dict(station=station, track=track, horizon=horizon,
                                 seed=seed, years=years, **res))
            print(f"[{i}/{len(stations)} {k}/{total}] {station} {track} h{horizon} "
                  f"y{years} s{seed}: NSE={res['nse']:.3f} KGE={res['kge']:.3f} "
                  f"({res['seconds']}s)", flush=True)


def station_list(stride: int, shard: int, nshards: int):
    ids = sorted(p.stem for p in Path(D.DATA_DIR).glob("*.csv"))
    ids = [s for s in ids if s not in D.EXCLUDE]
    ids = ids[::stride][shard::nshards]
    return ids


def extend_tracks(horizon: int, slim: bool = False):
    """Track set for the extension phases; window adapts inside the models.
    slim (seasonal tiers): drop the mid-range fe variant — keep the fe-long
    (snow-memory) variant so the external-memory contrast stays testable."""
    tracks = ["climatology", "persistence", "xgb-raw", "xgb-fe"]
    tracks += ["lstm-raw-zscore"]
    if horizon == 10:
        tracks.append("lstm-raw-minmax")   # one mid-range normalization point
    if slim:
        tracks += ["xgb-fe-long", "lstm-fe-long-zscore"]
        return tracks
    tracks += ["lstm-fe-zscore"]
    if horizon >= 30:
        tracks += ["xgb-fe-long", "lstm-fe-long-zscore"]
    return tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["main", "extend", "smoke"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--horizons", type=int, nargs="+", default=[])
    ap.add_argument("--lstm-epochs", type=int, default=None)
    ap.add_argument("--slim", action="store_true", help="seasonal slim track set")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    suffix = f".shard{args.shard}" if args.nshards > 1 else ""

    stations = station_list(args.stride, args.shard, args.nshards)
    if args.limit:
        stations = stations[:args.limit]

    if args.phase == "smoke":
        jobs = [(t, h, 0, 0) for h in (10, 30) for t in extend_tracks(h)]
        sweep(station_list(670, 0, 1)[:1], jobs, OUT / "smoke.csv",
              epochs=args.lstm_epochs or 3)
    elif args.phase == "main":
        jobs = [(t, h, 0, 0) for t, o in MAIN_TRACKS for h in o.get("horizons", HORIZONS)]
        sweep(stations, jobs, OUT / f"main{suffix}.csv", epochs=args.lstm_epochs)
    else:  # extend: --horizons 10 15 | 30 | 90 180 (stride set by caller)
        horizons = args.horizons or [10, 15]
        jobs = [(t, h, 0, 0) for h in horizons for t in extend_tracks(h, args.slim)]
        tag = "_".join(map(str, horizons))
        sweep(stations, jobs, OUT / f"extend_{tag}{suffix}.csv",
              epochs=args.lstm_epochs)


if __name__ == "__main__":
    main()
