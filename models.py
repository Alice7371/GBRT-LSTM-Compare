"""Model tracks for the scaled-up study.

All forecasts are DIRECT multi-horizon: inputs use only data up to day t,
targets are discharge at t+h (h = 1/3/7). No recursion, no future forcing.

- persistence: q(t) as the forecast for q(t+h)
- xgb-raw: XGBoost on the flattened raw window (thesis-style)
- xgb-fe:   XGBoost on the engineered same-day feature vector (featurize.py)
- lstm-raw / lstm-fe: LSTM with Min-Max or Z-Score normalization; the -fe
  variant appends engineered state/seasonality channels to the input window
"""
import numpy as np
import pandas as pd
import torch
torch.backends.cudnn.benchmark = False  # deterministic workspaces: 3-way sharding
# once oversubscribed the 8 GB card and stalled; keep the memory footprint flat
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

import data as D
from featurize import build_features, LSTM_EXTRA, LSTM_EXTRA_LONG, \
    XGB_FE_FEATURES, XGB_FE_LONG_EXTRA

WARMUP = 91  # longest rolling window in featurize.py (90) + 1


# ---------------- persistence & climatology ----------------

def run_persistence(df, horizon: int) -> dict:
    q = df["Discharge"].values.astype(np.float32)
    split = D.split_index(len(q))
    t_test = np.arange(split, len(q) - horizon)
    return D.evaluate(q[t_test + horizon], q[t_test])


def run_climatology(df, horizon: int) -> dict:
    """Day-of-year climatology: training-period doy mean, 11-day centered smoothing."""
    q = df["Discharge"].values.astype(np.float32)
    doy = df["Date"].dt.dayofyear.to_numpy()
    split = D.split_index(len(q))
    # training-period doy means, smoothed over neighbouring days-of-year
    clim = np.zeros(367)
    for d in range(1, 367):
        vals = []
        for off in range(-5, 6):
            vals.append(q[:split][doy[:split] == ((d - 1 + off) % 366) + 1])
        clim[d] = np.concatenate(vals).mean() if any(len(v) for v in vals) else np.nan
    t_test = np.arange(split, len(q) - horizon)
    pred = clim[doy[t_test + horizon]]
    pred = np.nan_to_num(pred, nan=q[:split].mean())  # rare doys (Feb 29)
    return D.evaluate(q[t_test + horizon], pred)


# ---------------- XGBoost ----------------

XGB_PARAMS = dict(
    n_estimators=2000,
    early_stopping_rounds=50,
    learning_rate=0.05,
    max_leaves=35,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1,   # CPU: keeps the GPU free for the LSTM shards
    verbosity=0,
)


def run_xgb(df, window: int, horizon: int, fe: bool = False, long: bool = False,
            years: int = None) -> dict:
    window = window or D.window_for(horizon)
    q = df["Discharge"].values.astype(np.float32)
    split = D.split_index(len(q))
    lo = max(0, split - years * 365) if years else 0  # data-volume axis

    if fe:
        feats = build_features(df)
        cols = XGB_FE_FEATURES + (XGB_FE_LONG_EXTRA if long else [])
        X_all = feats[cols].values.astype(np.float32)
        X_all = np.nan_to_num(X_all, nan=0.0)
        t_train = np.arange(max(WARMUP, lo), split - horizon)
        t_test = np.arange(split, len(q) - horizon)
        cut = int(len(t_train) * 0.8)
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_all[t_train[:cut]], q[t_train[:cut] + horizon],
                  eval_set=[(X_all[t_train[cut:]], q[t_train[cut:] + horizon])],
                  verbose=False)
        pred = model.predict(X_all[t_test])
    else:
        values = df[D.FEATURES].values.astype(np.float32)
        # window rows [i, i+w) -> target row i+w-1+h; train targets stay < split
        i_train = np.arange(max(0, lo - window + 1), split - window - horizon + 1)
        i_test = np.arange(split - window + 1, len(values) - window - horizon + 1)
        flat = D.make_flat_windows(values, window)
        Xtr, Xte = flat[i_train], flat[i_test]
        ytr = q[i_train + window - 1 + horizon]
        cut = int(len(Xtr) * 0.8)
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(Xtr[:cut], ytr[:cut], eval_set=[(Xtr[cut:], ytr[cut:])], verbose=False)
        pred = model.predict(Xte)

    y_true = q[np.arange(split, len(q) - horizon) + horizon]
    return D.evaluate(y_true, pred)


# ---------------- LSTM ----------------

HIDDEN, LAYERS, DROPOUT = 64, 2, 0.1
EPOCHS, LR, WD, BATCH = 40, 1e-3, 1e-4, 1024
LR_STEP, LR_GAMMA = 20, 0.5  # halve the learning rate twice over 40 epochs


class LSTMRegressor(nn.Module):
    """cell='lstm'|'gru' — identical interface, gates ablation."""
    def __init__(self, input_size, cell: str = "lstm"):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.lstm = rnn(input_size, HIDDEN, LAYERS, batch_first=True, dropout=DROPOUT)
        self.fc = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def run_lstm(df, window: int, norm: str, horizon: int = 1, fe: bool = False,
             long: bool = False, seed: int = 0, years: int = None,
             device: str = None, epochs: int = None,
             aug: tuple = None, eval_noise: float = 0.0, cell: str = "lstm") -> dict:
    """aug = (jitter_sigma, mask_frac) applied to meteorological channels of
    training batches only (train-time augmentation); eval_noise injects noise
    into the TEST inputs for robustness evaluation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    epochs = epochs or EPOCHS
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    window = window or D.window_for(horizon)

    q = df["Discharge"].values.astype(np.float32).reshape(-1, 1)
    split = D.split_index(len(df))
    lo = max(0, split - years * 365) if years else 0

    extra = (LSTM_EXTRA_LONG if long else LSTM_EXTRA) if fe else []
    cols = D.FEATURES + extra
    if fe:
        feats = build_features(df)
        values = feats[cols].values.astype(np.float32)
        values = np.nan_to_num(values, nan=0.0)
    else:
        values = df[D.FEATURES].values.astype(np.float32)
    q_col = cols.index("Discharge")  # not last once extra channels are appended

    fwd, _ = D.fit_normalizer(values[:split], norm)
    _, inv_q = D.fit_normalizer(q[:split], norm)
    values_n = fwd(values)

    # sequence start i -> rows [i, i+window); target row i+window-1+horizon.
    # train: target < split and window start >= lo; test: window start >= split-window
    n = len(values) - window - horizon + 1
    starts = np.arange(n)
    tr = starts[(starts >= lo) & (starts + window - 1 + horizon < split)]
    te = starts[starts >= split - window]

    X_all = np.stack([values_n[i:i + window] for i in range(n)])
    y_all = values_n[window - 1 + horizon: window - 1 + horizon + n, q_col]
    X_train, y_train = X_all[tr], y_all[tr]
    X_test, y_test = X_all[te], y_all[te]

    loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                        batch_size=BATCH, shuffle=False)
    model = LSTMRegressor(len(cols), cell=cell).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

    def _augment(x):
        """Train-time augmentation on meteorological channels (all but q_col)."""
        if aug is None:
            return x
        jitter, mask_frac = aug
        met = [c for c in range(x.shape[-1]) if c != q_col_in]
        if jitter > 0:
            noise = torch.randn_like(x[..., met]) * jitter
            x = x.clone(); x[..., met] = x[..., met] + noise
        if mask_frac > 0:
            m = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) < mask_frac)
            x = x.clone(); x[..., met] = x[..., met] * (~m).float()
        return x

    q_col_in = cols.index("Discharge")
    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(_augment(bx.to(device))), by.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    X_test_t = torch.tensor(X_test).to(device)
    if eval_noise > 0:  # robustness probe: noise on test meteorological channels
        met = [c for c in range(X_test_t.shape[-1]) if c != q_col_in]
        X_test_t = X_test_t.clone()
        X_test_t[..., met] += torch.randn_like(X_test_t[..., met]) * eval_noise
    with torch.no_grad():
        pred = model(X_test_t).cpu().numpy()
    y_true = inv_q(y_test.reshape(-1, 1)).ravel()
    y_hat = inv_q(pred.reshape(-1, 1)).ravel()
    return D.evaluate(y_true, y_hat)
