"""Default vs HPT-tuned config on the 12 representative basins (test NSE)."""
import json
import pandas as pd
import torch

import data as D
import models as M
from hpt import pick_representatives

best = json.load(open("study_results/hpt_best.json"))
rows = []
for s in pick_representatives():
    df = D.load_station(s)
    # default (as used across the study)
    r_def = M.run_lstm(df, None, "zscore", horizon=1, seed=0)
    # tuned: monkey-patch module constants
    M.HIDDEN, M.DROPOUT, M.BATCH = best["hidden"], best["dropout"], best["batch"]
    r_tun = M.run_lstm(df, None, best["norm"], horizon=1, seed=0)
    M.HIDDEN, M.DROPOUT, M.BATCH = 64, 0.1, 1024  # restore
    rows.append(dict(station=s, default=r_def["nse"], tuned=r_tun["nse"]))
    print(f"{s}: default={r_def['nse']:.3f} tuned={r_tun['nse']:.3f}", flush=True)
t = pd.DataFrame(rows)
t.to_csv("study_results/hpt_validate.csv", index=False)
print(f"\nmedian default={t.default.median():.4f} tuned={t.tuned.median():.4f} "
      f"delta={float((t.tuned-t.default).median()):+.4f}")
