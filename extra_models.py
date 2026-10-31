"""Additional model families for the state-capacity axis:

- MLP: flattened window -> deep FC. Stateless + unbounded capacity (the third
  pole: trees = stateless/shallow-interaction, MLP = stateless/universal,
  LSTM = stateful).
- Attention: per-timestep linear embed -> single-head self-attention over the
  window -> last-token readout. Random-access memory vs LSTM's sequential state.
- GRU: run via models.run_lstm(cell='gru') — no separate code here.

Per-station CLI:
  python extra_models.py --horizons 1 7 30 --stride 2
Output: study_results/extra.csv (schema-compatible with extend csvs).
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

import data as D
from featurize import build_features, LSTM_EXTRA, LSTM_EXTRA_LONG
from models import DROPOUT, EPOCHS, LR, WD

OUT = Path("study_results")


class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden=(256, 128)):
        super().__init__()
        layers, prev = [], input_size
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(DROPOUT)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1)).squeeze(-1)


class WindowAttentionRegressor(nn.Module):
    def __init__(self, input_size, d=64, heads=1):
        super().__init__()
        self.embed = nn.Linear(input_size, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 1)

    def forward(self, x):
        h = self.embed(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm(h + a)
        return self.fc(h[:, -1, :]).squeeze(-1)


def _prepare(df, horizon, fe, long, norm):
    window = D.window_for(horizon)
    q = df["Discharge"].values.astype(np.float32).reshape(-1, 1)
    split = D.split_index(len(df))
    extra = (LSTM_EXTRA_LONG if long else LSTM_EXTRA) if fe else []
    cols = D.FEATURES + extra
    values = (np.nan_to_num(build_features(df)[cols].values.astype(np.float32), nan=0.0)
              if fe else df[D.FEATURES].values.astype(np.float32))
    fwd, _ = D.fit_normalizer(values[:split], norm)
    _, inv_q = D.fit_normalizer(q[:split], norm)
    vn = fwd(values)
    n = len(vn) - window - horizon + 1
    starts = np.arange(n)
    tr = starts[starts + window - 1 + horizon < split]
    te = starts[starts >= split - window]
    X_all = np.stack([vn[i:i + window] for i in range(n)])
    y_all = vn[window - 1 + horizon: window - 1 + horizon + n, cols.index("Discharge")]
    return X_all, y_all, tr, te, inv_q, window


def run_extra(df, model_kind: str, horizon: int, norm: str = "zscore",
              fe: bool = False, long: bool = False, seed: int = 0,
              device: str = None, epochs: int = None) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    epochs = epochs or EPOCHS
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X_all, y_all, tr, te, inv_q, window = _prepare(df, horizon, fe, long, norm)
    model_cls = MLPRegressor if model_kind == "mlp" else WindowAttentionRegressor
    model = model_cls(X_all.shape[-1] * window if model_kind == "mlp"
                      else X_all.shape[-1]).to(device)
    loader = DataLoader(TensorDataset(torch.tensor(X_all[tr]), torch.tensor(y_all[tr])),
                        batch_size=1024, shuffle=False)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.5)
    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            criterion(model(bx.to(device)), by.to(device)).backward()
            optimizer.step()
        scheduler.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_all[te]).to(device)).cpu().numpy()
    y_true = inv_q(y_all[te].reshape(-1, 1)).ravel()
    y_hat = inv_q(pred.reshape(-1, 1)).ravel()
    return D.evaluate(y_true, y_hat)


def station_list(stride):
    ids = sorted(p.stem for p in Path(D.DATA_DIR).glob("*.csv"))
    return [s for s in ids if s not in D.EXCLUDE][::stride]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 7, 30])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=["mlp", "attn"])
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    csv = OUT / "extra.csv"

    def done():
        if not csv.exists():
            return set()
        d = pd.read_csv(csv, dtype={"station": str})
        return set(zip(d.station, d.track, d.horizon))

    stations = station_list(args.stride)
    if args.limit:
        stations = stations[:args.limit]
    known = done()
    for i, s in enumerate(stations, 1):
        df = D.load_station(s)
        for h in args.horizons:
            for kind in args.models:
                track = f"{kind}-zscore"
                if (s, track, h) in known:
                    continue
                t0 = time.time()
                try:
                    if kind == "gru":
                        import models as M
                        res = M.run_lstm(df, None, "zscore", horizon=h, cell="gru")
                    else:
                        res = run_extra(df, kind, h)
                except Exception as e:
                    print(f"FAIL {s} {track} h{h}: {e}", flush=True)
                    continue
                pd.DataFrame([dict(station=s, track=track, horizon=h, seed=0,
                                   years=0, **res,
                                   seconds=round(time.time() - t0, 1))]
                             ).to_csv(csv, mode="a", header=not csv.exists(),
                                      index=False)
                print(f"[{i}/{len(stations)}] {s} {track} h{h}: "
                      f"NSE={res['nse']:.3f} ({round(time.time()-t0,1)}s)", flush=True)


if __name__ == "__main__":
    main()
