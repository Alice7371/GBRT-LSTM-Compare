"""Hyperparameter search (Optuna TPE) on 12 stratified representative basins.

Protocol: within each basin's TRAIN segment, the last 20% is the validation
split — the test segment is never touched. Objective = mean validation NSE
over the 12 basins at h=1 (window 15). Best config -> study_results/hpt_best.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import torch

import data as D
import models as M

OUT = Path("study_results")
N_TRIALS = 50

# 12 stratified representatives selected from basin descriptors: snow/rain
# (frozen_frac>0.15) x high/low CV, 3 basins per cell, deterministic seed.
def pick_representatives():
    desc = pd.read_csv(OUT / "basin_desc.csv", dtype={"station": str})
    desc = desc.sort_values("station")
    desc["snow"] = desc.frozen_frac > 0.15
    desc["hi_cv"] = desc.cv > desc.cv.median()
    rng = np.random.default_rng(0)
    reps = []
    for snow in (True, False):
        for hi_cv in (True, False):
            cell = desc[(desc.snow == snow) & (desc.hi_cv == hi_cv)]
            reps += rng.choice(cell.station.values, 3, replace=False).tolist()
    return sorted(reps)


def val_split_lstm(df, window, horizon, norm, seed):
    """Train-segment only; last 20% of train windows = validation."""
    torch.manual_seed(seed); np.random.seed(seed)
    values = df[D.FEATURES].values.astype(np.float32)
    q = df["Discharge"].values.astype(np.float32).reshape(-1, 1)
    split = D.split_index(len(df))
    fwd, _ = D.fit_normalizer(values[:split], norm)
    _, inv_q = D.fit_normalizer(q[:split], norm)
    vn = fwd(values)
    n = split - window - horizon + 1
    starts = np.arange(n)
    cut = int(n * 0.8)
    X_tr = np.stack([vn[i:i + window] for i in starts[:cut]])
    y_tr = vn[starts[:cut] + window - 1 + horizon, -1]
    X_va = np.stack([vn[i:i + window] for i in starts[cut:]])
    y_va = vn[starts[cut:] + window - 1 + horizon, -1]
    return (torch.tensor(X_tr), torch.tensor(y_tr)), (torch.tensor(X_va), torch.tensor(y_va)), inv_q


def objective(trial, frames, device):
    hidden = trial.suggest_categorical("hidden", [32, 64, 128])
    layers = trial.suggest_categorical("layers", [1, 2])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])
    lr = trial.suggest_categorical("lr", [3e-4, 1e-3])
    batch = trial.suggest_categorical("batch", [512, 1024])
    norm = trial.suggest_categorical("norm", ["zscore", "minmax"])

    scores = []
    for df in frames:
        (Xtr, ytr), (Xva, yva), inv_q = val_split_lstm(df, 15, 1, norm, seed=0)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xtr, ytr), batch_size=batch, shuffle=False)
        model = M.LSTMRegressor(len(D.FEATURES)).to(device)
        model.lstm = torch.nn.LSTM(len(D.FEATURES), hidden, layers,
                                   batch_first=True, dropout=dropout).to(device)
        model.fc = torch.nn.Linear(hidden, 1).to(device)
        crit = torch.nn.SmoothL1Loss()
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.StepLR(opt, 20, 0.5)
        for _ in range(40):
            model.train()
            for bx, by in loader:
                opt.zero_grad(); crit(model(bx.to(device)), by.to(device)).backward(); opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xva.to(device)).cpu().numpy()
        yt = inv_q(yva.numpy().reshape(-1, 1)).ravel()
        yp = inv_q(pred.reshape(-1, 1)).ravel()
        scores.append(D.nse(yt, yp))
    return float(np.mean(scores))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reps = pick_representatives()
    print("representatives:", reps)
    frames = [D.load_station(s) for s in reps]
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda t: objective(t, frames, device), n_trials=args.trials)
    best = study.best_params
    OUT.joinpath("hpt_best.json").write_text(json.dumps(best, indent=2))
    print("best:", best, "val NSE:", study.best_value)
    OUT.joinpath("hpt_trials.csv").write_text(
        study.trials_dataframe().to_csv(index=False))


if __name__ == "__main__":
    main()
