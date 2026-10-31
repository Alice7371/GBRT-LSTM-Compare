"""Pooled (regional) model variants: one shared model trained on all basins.

Variants (BACKGROUND.md sec.4):
  lstm-pooled-plain       no station identifier (Kratzert-2018 style)
  lstm-pooled-emb         + nn.Embedding(n_stations, 16) broadcast to timesteps
  lstm-pooled-emb-fe      + engineered state/seasonality channels
  lstm-pooled-emb-minmax  normalization ablation
  xgb-pooled-raw/fe       + station integer code as a split feature
Protocol: pooled training uses ONLY each basin's train segment; every basin's
test segment is held out. Per-station normalization (train-segment stats).
Results append per basin to study_results/pooled_{tag}.csv — one file per
(track, horizon, seed, years) training run; a complete file (n_stations rows)
marks the run done and is skipped on re-execution.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

import data as D
from featurize import build_features, LSTM_EXTRA, LSTM_EXTRA_LONG, \
    XGB_FE_FEATURES, XGB_FE_LONG_EXTRA
from models import XGB_PARAMS, HIDDEN, LAYERS, DROPOUT, EPOCHS, LR, WD

OUT = Path("study_results")


# ---------------- data building ----------------

def station_matrix(df, horizon: int, fe: bool, long: bool, norm: str, years=None):
    """Normalized per-station feature matrix + shared plumbing."""
    q = df["Discharge"].values.astype(np.float32).reshape(-1, 1)
    split = D.split_index(len(df))
    lo = max(0, split - years * 365) if years else 0

    extra = (LSTM_EXTRA_LONG if long else LSTM_EXTRA) if fe else []
    cols = D.FEATURES + extra
    if fe:
        values = np.nan_to_num(build_features(df)[cols].values.astype(np.float32), nan=0.0)
        warmup = 91
    else:
        values = df[D.FEATURES].values.astype(np.float32)
        warmup = 0

    fwd, _ = D.fit_normalizer(values[:split], norm)
    _, inv_q = D.fit_normalizer(q[:split], norm)
    return fwd(values), inv_q, cols, split, max(lo, warmup)


def build_lstm_pool(stations, horizon, fe=False, long=False, norm="zscore",
                    years=None, max_per_station=None, seed=0):
    """Concatenate per-station training sequences; keep per-station test sets."""
    window = D.window_for(horizon)
    rng = np.random.default_rng(seed)
    Xs, ys, sids, tests = [], [], [], {}
    for sid, s in enumerate(stations):
        df = D.load_station(s)
        vals, inv_q, cols, split, lo = station_matrix(df, horizon, fe, long, norm, years)
        n = len(vals) - window - horizon + 1
        starts = np.arange(n)
        tr = starts[(starts >= lo) & (starts + window - 1 + horizon < split)]
        te = starts[starts >= split - window]
        if max_per_station and len(tr) > max_per_station:
            tr = rng.choice(tr, max_per_station, replace=False)
        Xs.append(np.stack([vals[i:i + window] for i in tr]))
        ys.append(vals[tr + window - 1 + horizon, cols.index("Discharge")])
        sids.append(np.full(len(tr), sid, dtype=np.int64))
        tests[s] = (np.stack([vals[i:i + window] for i in te]),
                    vals[te + window - 1 + horizon, cols.index("Discharge")],
                    inv_q, sid)
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(sids),
            tests, len(cols))


def build_xgb_pool(stations, horizon, fe=False, long=False, years=None,
                   max_per_station=None, with_id=True, seed=0):
    window = D.window_for(horizon)
    rng = np.random.default_rng(seed)
    Xs, ys, sids, tests = [], [], [], {}
    for sid, s in enumerate(stations):
        df = D.load_station(s)
        q = df["Discharge"].values.astype(np.float32)
        split = D.split_index(len(df))
        lo = max(0, split - years * 365) if years else 0
        if fe:
            feats = build_features(df)
            cols = XGB_FE_FEATURES + (XGB_FE_LONG_EXTRA if long else [])
            mat = np.nan_to_num(feats[cols].values.astype(np.float32), nan=0.0)
            t_tr = np.arange(max(91, lo), split - horizon)
            t_te = np.arange(split, len(q) - horizon)
            if with_id:
                mat = np.hstack([mat, np.full((len(mat), 1), sid, dtype=np.float32)])
        else:
            values = df[D.FEATURES].values.astype(np.float32)
            i_tr = np.arange(max(lo, 0), split - window - horizon + 1)
            i_te = np.arange(split - window + 1, len(values) - window - horizon + 1)
            flat = D.make_flat_windows(values, window)
            mat = np.hstack([flat, np.full((len(flat), 1), sid, np.float32)]) if with_id else flat
        if fe:
            if max_per_station and len(t_tr) > max_per_station:
                t_tr = rng.choice(t_tr, max_per_station, replace=False)
            Xs.append(mat[t_tr]); ys.append(q[t_tr + horizon]); sids.append(np.full(len(t_tr), sid))
            tests[s] = (mat[t_te], q[t_te + horizon])
        else:
            if max_per_station and len(i_tr) > max_per_station:
                i_tr = rng.choice(i_tr, max_per_station, replace=False)
            Xs.append(mat[i_tr]); ys.append(q[i_tr + window - 1 + horizon]); sids.append(np.full(len(i_tr), sid))
            tests[s] = (mat[i_te], q[i_te + window - 1 + horizon])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(sids), tests


# ---------------- models ----------------

class PooledLSTM(nn.Module):
    """cell='lstm'|'gru'; kind='rnn'|'attn' switches the sequence encoder."""
    def __init__(self, input_size, n_stations=0, emb_dim=16, cell="lstm", kind="rnn"):
        super().__init__()
        self.emb = nn.Embedding(n_stations, emb_dim) if n_stations else None
        size = input_size + (emb_dim if n_stations else 0)
        self.kind = kind
        if kind == "rnn":
            rnn = nn.LSTM if cell == "lstm" else nn.GRU
            self.enc = rnn(size, HIDDEN, LAYERS, batch_first=True, dropout=DROPOUT)
            self.fc = nn.Linear(HIDDEN, 1)
        else:  # single-head self-attention over the window
            d = 64
            self.embed = nn.Linear(size, d)
            self.attn = nn.MultiheadAttention(d, 1, batch_first=True)
            self.norm = nn.LayerNorm(d)
            self.fc = nn.Linear(d, 1)

    def forward(self, x, sid=None):
        if self.emb is not None:
            e = self.emb(sid).unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, e], dim=-1)
        if self.kind == "rnn":
            out, _ = self.enc(x)
            return self.fc(out[:, -1, :]).squeeze(-1)
        h = self.embed(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        return self.fc(self.norm(h + a)[:, -1, :]).squeeze(-1)


# ---------------- runners ----------------

def append_rows(csv: Path, rows):
    pd.DataFrame(rows).to_csv(csv, mode="a", header=not csv.exists(), index=False)


def run_lstm_pool(stations, horizon, variant="emb", fe=False, long=False,
                  norm="zscore", years=None, seed=0, epochs=EPOCHS,
                  max_per_station=None, device=None, cell="lstm", kind="rnn"):
    tag = f"{cell if kind == 'rnn' else 'attn'}-pooled-{variant}_h{horizon}_s{seed}_y{years or 0}"
    csv = OUT / f"pooled_{tag}.csv"
    if csv.exists() and len(pd.read_csv(csv)) >= len(stations):
        print(f"[skip] {tag} complete"); return
    t0 = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)

    X, y, sid, tests, n_cols = build_lstm_pool(
        stations, horizon, fe=fe, long=long, norm=norm, years=years,
        max_per_station=max_per_station, seed=seed)
    n_emb = len(stations) if variant in ("emb", "emb-fe", "emb-minmax") else 0
    Xt, yt, st = (torch.tensor(X), torch.tensor(y), torch.tensor(sid))
    loader = DataLoader(TensorDataset(Xt, yt, st), batch_size=1024, shuffle=True)

    model = PooledLSTM(n_cols, n_stations=n_emb, cell=cell, kind=kind).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        losses = []
        for bx, by, bs in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx.to(device), bs.to(device)), by.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  {tag} epoch {epoch+1}/{epochs} loss={np.mean(losses):.5f}", flush=True)

    rows = []
    model.eval()
    with torch.no_grad():
        for s, (X_te, y_te, inv_q, sid) in tests.items():
            sid_te = torch.full((len(X_te),), sid, dtype=torch.long, device=device)
            pred = model(torch.tensor(X_te).to(device), sid_te).cpu().numpy()
            yt_ = inv_q(y_te.reshape(-1, 1)).ravel()
            yp_ = inv_q(pred.reshape(-1, 1)).ravel()
            r = D.evaluate(yt_, yp_)
            tname = f"{'lstm' if (cell == 'lstm' and kind == 'rnn') else (cell if kind == 'rnn' else 'attn')}-pooled-{variant}"
            rows.append(dict(station=s, track=tname, horizon=horizon,
                             seed=seed, years=years or 0, **r))
    append_rows(csv, rows)
    print(f"[done] {tag}: median NSE {np.median([r['nse'] for r in rows]):.3f} "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)


def run_xgb_pool(stations, horizon, fe=False, long=False, years=None,
                 max_per_station=None, with_id=True):
    name = "xgb-pooled-fe" if fe else "xgb-pooled-raw"
    if fe and not with_id:
        name = "xgb-pooled-fe-noid"
    tag = f"{name}_h{horizon}_y{years or 0}"
    csv = OUT / f"pooled_{tag}.csv"
    if csv.exists() and len(pd.read_csv(csv)) >= len(stations):
        print(f"[skip] {tag} complete"); return
    t0 = time.time()
    X, y, sid, tests = build_xgb_pool(stations, horizon, fe=fe, long=long,
                                      years=years, max_per_station=max_per_station,
                                      with_id=with_id)
    cut = int(len(X) * 0.8)
    model = xgb.XGBRegressor(**{**XGB_PARAMS, "n_jobs": None, "device": "cuda"})
    model.fit(X[:cut], y[:cut], eval_set=[(X[cut:], y[cut:])], verbose=False)

    rows = []
    for s, (X_te, y_te) in tests.items():
        r = D.evaluate(y_te, model.predict(X_te))
        rows.append(dict(station=s, track=name, horizon=horizon, seed=0,
                         years=years or 0, **r))
    append_rows(csv, rows)
    print(f"[done] {tag}: median NSE {np.median([r['nse'] for r in rows]):.3f} "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)


def station_pool(stride=1, limit=None):
    ids = sorted(p.stem for p in Path(D.DATA_DIR).glob("*.csv"))
    ids = [s for s in ids if s not in D.EXCLUDE][::stride]
    return ids[:limit] if limit else ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["short", "mid", "long", "xseasonal", "xgb", "volume", "extra"])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    S = station_pool(args.stride)
    ep = args.epochs or EPOCHS

    if args.stage == "short":       # windows 15, full training data
        for h in (1, 3, 7):
            run_lstm_pool(S, h, variant="emb", epochs=ep)
        run_lstm_pool(S, 1, variant="emb", seed=1, epochs=ep)
        for h in (1, 7):
            run_lstm_pool(S, h, variant="plain", epochs=ep)
            run_lstm_pool(S, h, variant="emb-fe", fe=True, epochs=ep)
        run_lstm_pool(S, 1, variant="emb-minmax", norm="minmax", epochs=ep)
    elif args.stage == "mid":       # windows 60; subsample to bound memory
        for h in (10, 15):
            run_lstm_pool(S, h, variant="emb", max_per_station=4500, epochs=ep)
        run_lstm_pool(S, 15, variant="plain", max_per_station=4500, epochs=ep)
        run_lstm_pool(S, 15, variant="emb-fe", fe=True, max_per_station=4500, epochs=ep)
    elif args.stage == "long":      # window 90
        for v, fe in (("emb", False), ("emb-fe", True)):
            run_lstm_pool(S, 30, variant=v, fe=fe, long=fe, max_per_station=3000, epochs=ep)
        run_xgb_pool(S, 30, fe=True, long=True, max_per_station=3000)
    elif args.stage == "xseasonal":  # window 270; heavy subsampling
        for h in (90, 180):
            run_lstm_pool(S, h, variant="emb", long=True, max_per_station=1000, epochs=ep)
            run_xgb_pool(S, h, fe=True, long=True, max_per_station=2000)
    elif args.stage == "xgb":
        for h in (1, 7, 15, 30):
            run_xgb_pool(S, h, fe=True, long=(h >= 30),
                         max_per_station=4500 if h >= 10 else None)
        for h in (1, 7):
            run_xgb_pool(S, h, fe=False)
        run_xgb_pool(S, 1, fe=True, with_id=False)
    elif args.stage == "extra":   # additional families, pooled with emb, h1/7
        for h in (1, 7):
            run_lstm_pool(S, h, variant="emb", cell="gru", epochs=ep,
                          max_per_station=4500 if h > 7 else None)
            run_lstm_pool(S, h, variant="emb", kind="attn", epochs=ep,
                          max_per_station=4500 if h > 7 else None)
    elif args.stage == "volume":
        for y in (5, 10):
            run_lstm_pool(S, 1, variant="emb", years=y, epochs=ep)
