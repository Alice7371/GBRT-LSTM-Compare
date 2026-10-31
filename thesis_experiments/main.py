"""Entry point replicating the thesis experiments (GBRT vs LSTM, 27 basins).

Examples:
  python main.py --model gbrt --station 01013500 --window 30
  python main.py --model lstm --norm zscore --station 01013500 --window 30
  python main.py --model lstm --norm minmax --all          # every basin x window
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from common import WINDOWS, load_station
from gbrt_model import run_gbrt
from lstm_model import run_lstm


def station_ids(data_dir: str):
    return sorted(p.stem for p in Path(data_dir).glob("*.csv"))


def main():
    ap = argparse.ArgumentParser(description="Thesis replication: GBRT vs LSTM runoff prediction")
    ap.add_argument("--model", choices=["gbrt", "lstm"], required=True)
    ap.add_argument("--norm", choices=["minmax", "zscore"], default="zscore",
                    help="normalization for the LSTM track (ignored by GBRT)")
    ap.add_argument("--station", help="basin ID, e.g. 01013500")
    ap.add_argument("--window", type=int, default=30, help="history window in days")
    ap.add_argument("--all", action="store_true", help="run every basin x every window")
    ap.add_argument("--data", default="dataset", help="dataset directory")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=None, help="override LSTM epochs (debug)")
    args = ap.parse_args()

    if not args.all and not args.station:
        ap.error("--station is required unless --all is given")

    stations = station_ids(args.data) if args.all else [args.station]
    windows = WINDOWS if args.all else [args.window]

    for station in stations:
        df = load_station(station, args.data)
        for window in windows:
            tag = args.model if args.model == "gbrt" else f"lstm_{args.norm}"
            out_dir = f"{args.out}/{tag}/{station}_ws{window}"
            t0 = time.time()
            if args.model == "gbrt":
                res = run_gbrt(df, window, out_dir)
            else:
                res = run_lstm(df, window, args.norm, out_dir, num_epochs=args.epochs)
            res["station"], res["window"], res["seconds"] = station, window, round(time.time() - t0, 1)
            with open(f"{out_dir}/metrics.json", "w") as f:
                json.dump(res, f, indent=2)
            print(f"{station} ws{window} [{tag}] test: {res['test']}")


if __name__ == "__main__":
    main()
